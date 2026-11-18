//! Persistent matmul worker pool — avoids per-matvec `thread::scope` spawn/join churn
//! on Q4_K×Q8_K, Q6_K×Q8_K, Q5_0×Q8_0, and Q8_0×Q8_0 hot paths.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};
use std::thread::{self, JoinHandle};

use super::ggml_ffi::{
    adapt_q80_to_ggml, cpu_repack_q4k_active, ggml_ffi_active, ggml_ffi_q8_active,
    ggml_ffi_qk_active, vec_dot_q4_k_q8_k_ggml, vec_dot_q5_0_q8_0_ggml_nrows,
    vec_dot_q6_k_q8_k_ggml, vec_dot_q8_0_q8_0_ggml_nrows, BlockQ80Ggml,
};
use super::q4k_repack::{gemv_q4_kx8_q8_k_nrows, q4_k_x8_cached};
use super::q50::{vec_dot_q5_0_q8_0_nrows, vec_dot_q8_0_q8_0_nrows, BlockQ80};
use super::q50_repack::{q5_0_x4_cached, vec_dot_q5_0x4_q8_0_nrows};
use super::q6k::vec_dot_q6_k_q8_k;
use super::q8k::{vec_dot_q4_k_q8_k, BlockQ8K};
use crate::error::{CoreError, Result};

enum JobKind {
    Q4K {
        y: *const BlockQ8K,
        y_len: usize,
    },
    /// Vendor CPU_REPACK Q4_K×8 gemv; `weight` points at packed buffer.
    Q4Kx8 {
        y: *const BlockQ8K,
        y_len: usize,
    },
    Q6K {
        y: *const BlockQ8K,
        y_len: usize,
    },
    Q50 {
        y: *const BlockQ80,
        y_len: usize,
    },
    /// Q5_0 weights interleaved ×4 (ggml-style); `weight` points at packed buffer.
    Q50x4 {
        y: *const BlockQ80,
        y_len: usize,
    },
    /// Vendor ggml Q5_0 row-major with f16 act blocks (ARTIFICER_GGML_FFI).
    Q50Ggml {
        y: *const BlockQ80Ggml,
        y_len: usize,
    },
    Q80 {
        y: *const BlockQ80,
        y_len: usize,
    },
    /// Vendor ggml Q8_0 row-major with f16 act blocks (ARTIFICER_GGML_FFI=q8).
    Q80Ggml {
        y: *const BlockQ80Ggml,
        y_len: usize,
    },
    /// Test-only job: sleeps, fills its rows with 1.0, then bumps `done`.
    #[cfg(test)]
    TestSleep {
        delay_ms: u64,
        done: *const AtomicUsize,
    },
    /// Test-only job that panics.
    #[cfg(test)]
    TestPanic,
}

struct Job {
    weight: *const u8,
    weight_len: usize,
    k: usize,
    kind: JobKind,
    out: *mut f32,
    row_base: usize,
    n_rows: usize,
    remaining: *const AtomicUsize,
    error: *const Mutex<Option<CoreError>>,
}

// SAFETY: a `Job` only carries raw pointers. The submitter (`parallel_rows` /
// `parallel_rows_aligned`) keeps every pointee alive until `remaining` reaches 0, including when
// its own local batch returns an error or panics (see `WaitForJobs`).
unsafe impl Send for Job {}

struct PoolState {
    jobs: Mutex<Vec<Job>>,
    cv: Condvar,
    shutdown: AtomicBool,
}

struct Pool {
    state: &'static PoolState,
    _handles: Vec<JoinHandle<()>>,
    n_workers: usize,
}

fn global_pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        // ARTIFICER_POOL overrides worker count (profile-driven BW/compute trade).
        let n = std::env::var("ARTIFICER_POOL")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&v| (1..=64).contains(&v))
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4)
                    .min(16)
                    .max(1)
            });
        let state: &'static PoolState = Box::leak(Box::new(PoolState {
            jobs: Mutex::new(Vec::new()),
            cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
        }));
        let mut handles = Vec::with_capacity(n);
        for id in 0..n {
            handles.push(
                thread::Builder::new()
                    .name(format!("artificer-matmul-{id}"))
                    .spawn(move || worker_loop(state))
                    .expect("spawn matmul worker"),
            );
        }
        Pool {
            state,
            _handles: handles,
            n_workers: n,
        }
    })
}

fn worker_loop(state: &'static PoolState) {
    loop {
        let job = {
            let mut guard = state.jobs.lock().unwrap();
            loop {
                if state.shutdown.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(j) = guard.pop() {
                    break j;
                }
                guard = state.cv.wait(guard).unwrap();
            }
        };
        let (remaining, error) = (job.remaining, job.error);
        // A panicking job must not kill the worker or leave `remaining` above 0 (the submitter
        // would spin forever): record an error and do the decrement `run_job` never reached.
        // AssertUnwindSafe is fine: after a panic nothing of the job is used again except
        // `error` and `remaining`, and the submitter gets an error instead of trusting `out`.
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| run_job(job))) {
            // SAFETY: the submitter keeps `error` and `remaining` alive until `remaining` reaches
            // 0, which happens only at the decrement below. That decrement is the last access to
            // the job (`Job` has no Drop), so it comes before dropping the payload, whose own
            // Drop could panic.
            unsafe {
                let mut g = (*error).lock().unwrap_or_else(PoisonError::into_inner);
                g.get_or_insert(CoreError::Backend("matmul worker panicked".into()));
                drop(g);
                (*remaining).fetch_sub(1, Ordering::Release);
            }
            drop(payload);
        }
    }
}

fn run_job(job: Job) {
    // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone until
    // `remaining` reaches 0; `weight` is the full weight slice.
    let weight = unsafe { std::slice::from_raw_parts(job.weight, job.weight_len) };
    // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone until
    // `remaining` reaches 0; `out` is this job's disjoint sub-slice of the output.
    let out = unsafe { std::slice::from_raw_parts_mut(job.out, job.n_rows) };
    match job.kind {
        JobKind::Q4K { y, y_len } => {
            // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone
            // until `remaining` reaches 0; `y` is the shared, read-only activation slice.
            let y = unsafe { std::slice::from_raw_parts(y, y_len) };
            let use_ggml = ggml_ffi_qk_active();
            for (local_j, out_j) in out.iter_mut().enumerate() {
                let r = if use_ggml {
                    vec_dot_q4_k_q8_k_ggml(weight, (job.row_base + local_j) * job.k, job.k, y)
                } else {
                    vec_dot_q4_k_q8_k(weight, (job.row_base + local_j) * job.k, job.k, y)
                };
                match r {
                    Ok(v) => *out_j = v,
                    Err(e) => {
                        // SAFETY: `error` points at the submitter's mutex, which outlives the job.
                        if let Ok(mut g) = unsafe { (*job.error).lock() } {
                            if g.is_none() {
                                *g = Some(e);
                            }
                        }
                        break;
                    }
                }
            }
        }
        JobKind::Q4Kx8 { y, y_len } => {
            // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone
            // until `remaining` reaches 0; `y` is the shared, read-only activation slice.
            let y = unsafe { std::slice::from_raw_parts(y, y_len) };
            if let Err(e) = gemv_q4_kx8_q8_k_nrows(weight, job.row_base, job.k, y, out) {
                // SAFETY: `error` points at the submitter's mutex, which outlives the job.
                if let Ok(mut g) = unsafe { (*job.error).lock() } {
                    if g.is_none() {
                        *g = Some(e);
                    }
                }
            }
        }
        JobKind::Q6K { y, y_len } => {
            // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone
            // until `remaining` reaches 0; `y` is the shared, read-only activation slice.
            let y = unsafe { std::slice::from_raw_parts(y, y_len) };
            let use_ggml = ggml_ffi_qk_active();
            for (local_j, out_j) in out.iter_mut().enumerate() {
                let r = if use_ggml {
                    vec_dot_q6_k_q8_k_ggml(weight, (job.row_base + local_j) * job.k, job.k, y)
                } else {
                    vec_dot_q6_k_q8_k(weight, (job.row_base + local_j) * job.k, job.k, y)
                };
                match r {
                    Ok(v) => *out_j = v,
                    Err(e) => {
                        // SAFETY: `error` points at the submitter's mutex, which outlives the job.
                        if let Ok(mut g) = unsafe { (*job.error).lock() } {
                            if g.is_none() {
                                *g = Some(e);
                            }
                        }
                        break;
                    }
                }
            }
        }
        JobKind::Q50 { y, y_len } => {
            // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone
            // until `remaining` reaches 0; `y` is the shared, read-only activation slice.
            let y = unsafe { std::slice::from_raw_parts(y, y_len) };
            // Multi-row GEMV: share y across consecutive weight rows.
            if let Err(e) = vec_dot_q5_0_q8_0_nrows(weight, job.row_base * job.k, job.k, y, out) {
                // SAFETY: `error` points at the submitter's mutex, which outlives the job.
                if let Ok(mut g) = unsafe { (*job.error).lock() } {
                    if g.is_none() {
                        *g = Some(e);
                    }
                }
            }
        }
        JobKind::Q50x4 { y, y_len } => {
            // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone
            // until `remaining` reaches 0; `y` is the shared, read-only activation slice.
            let y = unsafe { std::slice::from_raw_parts(y, y_len) };
            if let Err(e) = vec_dot_q5_0x4_q8_0_nrows(weight, job.row_base, job.k, y, out) {
                // SAFETY: `error` points at the submitter's mutex, which outlives the job.
                if let Ok(mut g) = unsafe { (*job.error).lock() } {
                    if g.is_none() {
                        *g = Some(e);
                    }
                }
            }
        }
        JobKind::Q50Ggml { y, y_len } => {
            // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone
            // until `remaining` reaches 0; `y` is the shared, read-only activation slice.
            let y = unsafe { std::slice::from_raw_parts(y, y_len) };
            if let Err(e) =
                vec_dot_q5_0_q8_0_ggml_nrows(weight, job.row_base * job.k, job.k, y, out)
            {
                // SAFETY: `error` points at the submitter's mutex, which outlives the job.
                if let Ok(mut g) = unsafe { (*job.error).lock() } {
                    if g.is_none() {
                        *g = Some(e);
                    }
                }
            }
        }
        JobKind::Q80 { y, y_len } => {
            // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone
            // until `remaining` reaches 0; `y` is the shared, read-only activation slice.
            let y = unsafe { std::slice::from_raw_parts(y, y_len) };
            if let Err(e) = vec_dot_q8_0_q8_0_nrows(weight, job.row_base * job.k, job.k, y, out) {
                // SAFETY: `error` points at the submitter's mutex, which outlives the job.
                if let Ok(mut g) = unsafe { (*job.error).lock() } {
                    if g.is_none() {
                        *g = Some(e);
                    }
                }
            }
        }
        JobKind::Q80Ggml { y, y_len } => {
            // SAFETY: the submitter keeps the job's buffers alive and leaves this row range alone
            // until `remaining` reaches 0; `y` is the shared, read-only activation slice.
            let y = unsafe { std::slice::from_raw_parts(y, y_len) };
            if let Err(e) =
                vec_dot_q8_0_q8_0_ggml_nrows(weight, job.row_base * job.k, job.k, y, out)
            {
                // SAFETY: `error` points at the submitter's mutex, which outlives the job.
                if let Ok(mut g) = unsafe { (*job.error).lock() } {
                    if g.is_none() {
                        *g = Some(e);
                    }
                }
            }
        }
        #[cfg(test)]
        JobKind::TestSleep { delay_ms, done } => {
            thread::sleep(std::time::Duration::from_millis(delay_ms));
            out.fill(1.0);
            // SAFETY: `done` points at the test's counter, which outlives every job.
            unsafe { (*done).fetch_add(1, Ordering::SeqCst) };
        }
        #[cfg(test)]
        JobKind::TestPanic => panic!("injected worker panic"),
    }
    // SAFETY: `remaining` points at the submitter's counter, which it polls until 0 before
    // returning (also when its own batch failed or panicked); this is the job's last access.
    unsafe {
        (*job.remaining).fetch_sub(1, Ordering::Release);
    }
}

/// On drop, spins until `remaining` is 0. `parallel_rows*` (and the SSM scan pool) create it
/// right after queueing jobs, so no exit (normal return, error, or unwinding out of the local
/// batch) leaves while queued jobs still point at the submitter's buffers and counters.
pub(crate) struct WaitForJobs<'a>(pub(crate) &'a AtomicUsize);

impl Drop for WaitForJobs<'_> {
    fn drop(&mut self) {
        while self.0.load(Ordering::Acquire) > 0 {
            thread::yield_now();
        }
    }
}

fn parallel_rows(
    k: usize,
    out: &mut [f32],
    n_threads: usize,
    kind_local: JobKind,
    make_farm: impl Fn(
        usize,
        usize,
        *mut f32,
        usize,
        *const AtomicUsize,
        *const Mutex<Option<CoreError>>,
    ) -> Job,
    run_local_batch: impl Fn(usize, &mut [f32]) -> Result<()>,
) -> Result<()> {
    let n = out.len();
    if n_threads <= 1 || n < 64 {
        run_local_batch(0, out)?;
        let _ = kind_local;
        return Ok(());
    }
    let pool = global_pool();
    let n_threads = n_threads.min(n).min(pool.n_workers.max(1));
    // Tighter tiles for fat-K (ssm_out / Q6 ffn_down) and large-N Q5/Q8
    // (ssm_in n=17504, ffn_up n=12544, LM head): more jobs than workers so
    // weight-row working sets stay L2-friendly and steal is smoother.
    let mut chunk = n.div_ceil(n_threads);
    let fat_k = k >= 6144 && chunk > 96;
    let fat_n = n >= 8192 && chunk > 128;
    if fat_k || fat_n {
        // 4× jobs for fat-N Q5 (ssm_in/ffn_up) keeps weight tiles L2-resident;
        // fat-K stays 2× (Q4/Q6) — finer than that was noisy in benchmarks.
        let mul = if fat_n { 4 } else { 2 };
        let n_jobs = n_threads.saturating_mul(mul).max(1);
        chunk = n.div_ceil(n_jobs);
        chunk = chunk.max(if fat_k { 48 } else { 32 });
        // Keep Q5×4 interleave tiles aligned (harmless for other kinds).
        if fat_n {
            chunk = (chunk + 3) & !3usize;
        }
    }

    let remaining = AtomicUsize::new(0);
    let error: Mutex<Option<CoreError>> = Mutex::new(None);

    let mut jobs = Vec::new();
    let mut start = chunk.min(n);
    while start < n {
        let end = (start + chunk).min(n);
        let part = &mut out[start..end];
        jobs.push(make_farm(
            start,
            end - start,
            part.as_mut_ptr(),
            k,
            &remaining,
            &error,
        ));
        start = end;
    }
    remaining.store(jobs.len(), Ordering::Release);
    if !jobs.is_empty() {
        {
            let mut guard = pool.state.jobs.lock().unwrap();
            guard.extend(jobs);
        }
        pool.state.cv.notify_all();
    }
    // Declared after `remaining` and `error`, so it drops (and waits) before they do, also when
    // `run_local_batch` panics.
    let wait = WaitForJobs(&remaining);

    let local_n = chunk.min(n);
    let local = run_local_batch(0, &mut out[..local_n]);
    drop(wait);
    local?;
    if let Some(e) = error.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(e);
    }
    Ok(())
}

/// Like `parallel_rows`, but forces chunk starts/sizes to multiples of `align`
/// (needed for Q4_K×8 / Q5_0×4 interleaved tiles).
fn parallel_rows_aligned(
    k: usize,
    out: &mut [f32],
    n_threads: usize,
    align: usize,
    kind_local: JobKind,
    make_farm: impl Fn(
        usize,
        usize,
        *mut f32,
        usize,
        *const AtomicUsize,
        *const Mutex<Option<CoreError>>,
    ) -> Job,
    run_local_batch: impl Fn(usize, &mut [f32]) -> Result<()>,
) -> Result<()> {
    let n = out.len();
    if align == 0 || !n.is_multiple_of(align) {
        return Err(CoreError::Shape("parallel_rows_aligned: n%align!=0".into()));
    }
    if n_threads <= 1 || n < 64 {
        run_local_batch(0, out)?;
        let _ = kind_local;
        return Ok(());
    }
    let pool = global_pool();
    let n_threads = n_threads.min(n).min(pool.n_workers.max(1));
    let mut chunk = n.div_ceil(n_threads);
    let fat_k = k >= 6144 && chunk > 96;
    let fat_n = n >= 8192 && chunk > 128;
    if fat_k || fat_n {
        let mul = if fat_n { 4 } else { 2 };
        let n_jobs = n_threads.saturating_mul(mul).max(1);
        chunk = n.div_ceil(n_jobs);
        chunk = chunk.max(if fat_k { 48 } else { 32 });
    }
    // Round chunk up to `align` so every farm tile is interleaved-legal.
    chunk = chunk.div_ceil(align) * align;
    if chunk == 0 {
        chunk = align;
    }

    let remaining = AtomicUsize::new(0);
    let error: Mutex<Option<CoreError>> = Mutex::new(None);

    let mut jobs = Vec::new();
    let mut start = chunk.min(n);
    // Ensure start is align-multiple (chunk is).
    while start < n {
        let end = (start + chunk).min(n);
        // Truncate last job to align if needed (n already %align==0 so end is fine).
        let part = &mut out[start..end];
        jobs.push(make_farm(
            start,
            end - start,
            part.as_mut_ptr(),
            k,
            &remaining,
            &error,
        ));
        start = end;
    }
    remaining.store(jobs.len(), Ordering::Release);
    if !jobs.is_empty() {
        {
            let mut guard = pool.state.jobs.lock().unwrap();
            guard.extend(jobs);
        }
        pool.state.cv.notify_all();
    }
    // Declared after `remaining` and `error`, so it drops (and waits) before they do, also when
    // `run_local_batch` panics.
    let wait = WaitForJobs(&remaining);

    let local_n = chunk.min(n);
    let local = run_local_batch(0, &mut out[..local_n]);
    drop(wait);
    local?;
    if let Some(e) = error.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(e);
    }
    Ok(())
}

/// Parallelize Q4_K×Q8_K integer matvec over output rows via the persistent pool.
///
/// `allow_repack = false` (MoE expert slices) never consults or fills the Q4_K×8 cache:
/// each expert is a sub-slice of one 3-D tensor, so a per-pointer repack would copy every
/// expert of every layer into the heap.
pub(crate) fn parallel_matmul_q4k_q8k(
    weight: &[u8],
    k: usize,
    y: &[BlockQ8K],
    out: &mut [f32],
    n_threads: usize,
    allow_repack: bool,
) -> Result<()> {
    // Env-gated vendor CPU_REPACK gemv (Q4_K×8). Default off.
    // Composes with Q5_0×4 (this path is Q4-only). Skips the Q4_K/Q6_K vec_dot FFI.
    if allow_repack && cpu_repack_q4k_active() {
        let n = out.len();
        let n8 = n & !7usize;
        if let Some(packed) = q4_k_x8_cached(weight, k, n) {
            let y_ptr = y.as_ptr();
            let y_len = y.len();
            let w_ptr = packed.as_ptr();
            let w_len = packed.len();
            let _keep = packed;
            parallel_rows_aligned(
                k,
                &mut out[..n8],
                n_threads,
                8,
                JobKind::Q4Kx8 { y: y_ptr, y_len },
                |row_base, n_rows, out_ptr, k, remaining, error| Job {
                    weight: w_ptr,
                    weight_len: w_len,
                    k,
                    kind: JobKind::Q4Kx8 { y: y_ptr, y_len },
                    out: out_ptr,
                    row_base,
                    n_rows,
                    remaining,
                    error,
                },
                |row_base, out_chunk| {
                    gemv_q4_kx8_q8_k_nrows(_keep.as_slice(), row_base, k, y, out_chunk)
                },
            )?;
            if n8 < n {
                for (local_j, out_j) in out[n8..].iter_mut().enumerate() {
                    *out_j = vec_dot_q4_k_q8_k(weight, (n8 + local_j) * k, k, y)?;
                }
            }
            return Ok(());
        }
    }

    let y_ptr = y.as_ptr();
    let y_len = y.len();
    let w_ptr = weight.as_ptr();
    let w_len = weight.len();
    parallel_rows(
        k,
        out,
        n_threads,
        JobKind::Q4K { y: y_ptr, y_len },
        |row_base, n_rows, out_ptr, k, remaining, error| Job {
            weight: w_ptr,
            weight_len: w_len,
            k,
            kind: JobKind::Q4K { y: y_ptr, y_len },
            out: out_ptr,
            row_base,
            n_rows,
            remaining,
            error,
        },
        |row_base, out_chunk| {
            let use_ggml = ggml_ffi_qk_active();
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                *out_j = if use_ggml {
                    vec_dot_q4_k_q8_k_ggml(weight, (row_base + local_j) * k, k, y)?
                } else {
                    vec_dot_q4_k_q8_k(weight, (row_base + local_j) * k, k, y)?
                };
            }
            Ok(())
        },
    )
}

/// Parallelize Q5_0×Q8_0 integer matvec over output rows via the persistent pool.
/// Prefer ggml-style Q5_0×4 interleave when the mat is fat-N (cached after first call).
/// `allow_repack = false` (MoE expert slices) skips the Q5_0×4 cache entirely and runs the
/// row-major integer path on the slice in place.
pub(crate) fn parallel_matmul_q50_q80(
    weight: &[u8],
    k: usize,
    y: &[BlockQ80],
    out: &mut [f32],
    n_threads: usize,
    allow_repack: bool,
) -> Result<()> {
    // Vendor ggml row-major Q5 path (f16 act adapter). Bypasses the Q5_0×4 kernels so
    // ggml and Artificer kernels can be compared under the same thread pool.
    if ggml_ffi_active() {
        let mut y_g = vec![
            BlockQ80Ggml {
                d: 0,
                qs: [0; super::blocks::QK8_0],
            };
            y.len()
        ];
        adapt_q80_to_ggml(y, &mut y_g);
        let y_ptr = y_g.as_ptr();
        let y_len = y_g.len();
        let w_ptr = weight.as_ptr();
        let w_len = weight.len();
        return parallel_rows(
            k,
            out,
            n_threads,
            JobKind::Q50Ggml { y: y_ptr, y_len },
            |row_base, n_rows, out_ptr, k, remaining, error| Job {
                weight: w_ptr,
                weight_len: w_len,
                k,
                kind: JobKind::Q50Ggml { y: y_ptr, y_len },
                out: out_ptr,
                row_base,
                n_rows,
                remaining,
                error,
            },
            |row_base, out_chunk| {
                vec_dot_q5_0_q8_0_ggml_nrows(weight, row_base * k, k, &y_g, out_chunk)
            },
        );
    }

    let n = out.len();
    let n4 = n & !3usize;
    let packed = if allow_repack {
        q5_0_x4_cached(weight, k, n)
    } else {
        None
    };
    if let Some(packed) = packed {
        let y_ptr = y.as_ptr();
        let y_len = y.len();
        let w_ptr = packed.as_ptr();
        let w_len = packed.len();
        // Keep Arc alive across pool join via stack binding.
        let _keep = packed;
        parallel_rows(
            k,
            &mut out[..n4],
            n_threads,
            JobKind::Q50x4 { y: y_ptr, y_len },
            |row_base, n_rows, out_ptr, k, remaining, error| Job {
                weight: w_ptr,
                weight_len: w_len,
                k,
                kind: JobKind::Q50x4 { y: y_ptr, y_len },
                out: out_ptr,
                row_base,
                n_rows,
                remaining,
                error,
            },
            |row_base, out_chunk| {
                // Align local batch to 4-row tiles when possible.
                vec_dot_q5_0x4_q8_0_nrows(_keep.as_slice(), row_base, k, y, out_chunk)
            },
        )?;
        if n4 < n {
            vec_dot_q5_0_q8_0_nrows(weight, n4 * k, k, y, &mut out[n4..])?;
        }
        return Ok(());
    }

    let y_ptr = y.as_ptr();
    let y_len = y.len();
    let w_ptr = weight.as_ptr();
    let w_len = weight.len();
    parallel_rows(
        k,
        out,
        n_threads,
        JobKind::Q50 { y: y_ptr, y_len },
        |row_base, n_rows, out_ptr, k, remaining, error| Job {
            weight: w_ptr,
            weight_len: w_len,
            k,
            kind: JobKind::Q50 { y: y_ptr, y_len },
            out: out_ptr,
            row_base,
            n_rows,
            remaining,
            error,
        },
        |row_base, out_chunk| vec_dot_q5_0_q8_0_nrows(weight, row_base * k, k, y, out_chunk),
    )
}

/// Parallelize Q6_K×Q8_K integer matvec over output rows via the persistent pool.
pub(crate) fn parallel_matmul_q6k_q8k(
    weight: &[u8],
    k: usize,
    y: &[BlockQ8K],
    out: &mut [f32],
    n_threads: usize,
) -> Result<()> {
    let y_ptr = y.as_ptr();
    let y_len = y.len();
    let w_ptr = weight.as_ptr();
    let w_len = weight.len();
    parallel_rows(
        k,
        out,
        n_threads,
        JobKind::Q6K { y: y_ptr, y_len },
        |row_base, n_rows, out_ptr, k, remaining, error| Job {
            weight: w_ptr,
            weight_len: w_len,
            k,
            kind: JobKind::Q6K { y: y_ptr, y_len },
            out: out_ptr,
            row_base,
            n_rows,
            remaining,
            error,
        },
        |row_base, out_chunk| {
            let use_ggml = ggml_ffi_qk_active();
            for (local_j, out_j) in out_chunk.iter_mut().enumerate() {
                *out_j = if use_ggml {
                    vec_dot_q6_k_q8_k_ggml(weight, (row_base + local_j) * k, k, y)?
                } else {
                    vec_dot_q6_k_q8_k(weight, (row_base + local_j) * k, k, y)?
                };
            }
            Ok(())
        },
    )
}

/// Parallelize Q8_0×Q8_0 integer matvec over output rows via the persistent pool.
pub(crate) fn parallel_matmul_q80_q80(
    weight: &[u8],
    k: usize,
    y: &[BlockQ80],
    out: &mut [f32],
    n_threads: usize,
) -> Result<()> {
    // Vendor ggml Q8_0 LM-head path (f16 act adapter). Keeps Q5_0×4.
    if ggml_ffi_q8_active() {
        let mut y_g = vec![
            BlockQ80Ggml {
                d: 0,
                qs: [0; super::blocks::QK8_0],
            };
            y.len()
        ];
        adapt_q80_to_ggml(y, &mut y_g);
        let y_ptr = y_g.as_ptr();
        let y_len = y_g.len();
        let w_ptr = weight.as_ptr();
        let w_len = weight.len();
        return parallel_rows(
            k,
            out,
            n_threads,
            JobKind::Q80Ggml { y: y_ptr, y_len },
            |row_base, n_rows, out_ptr, k, remaining, error| Job {
                weight: w_ptr,
                weight_len: w_len,
                k,
                kind: JobKind::Q80Ggml { y: y_ptr, y_len },
                out: out_ptr,
                row_base,
                n_rows,
                remaining,
                error,
            },
            |row_base, out_chunk| {
                vec_dot_q8_0_q8_0_ggml_nrows(weight, row_base * k, k, &y_g, out_chunk)
            },
        );
    }

    let y_ptr = y.as_ptr();
    let y_len = y.len();
    let w_ptr = weight.as_ptr();
    let w_len = weight.len();
    parallel_rows(
        k,
        out,
        n_threads,
        JobKind::Q80 { y: y_ptr, y_len },
        |row_base, n_rows, out_ptr, k, remaining, error| Job {
            weight: w_ptr,
            weight_len: w_len,
            k,
            kind: JobKind::Q80 { y: y_ptr, y_len },
            out: out_ptr,
            row_base,
            n_rows,
            remaining,
            error,
        },
        |row_base, out_chunk| vec_dot_q8_0_q8_0_nrows(weight, row_base * k, k, y, out_chunk),
    )
}

/// Number of persistent matmul workers.
pub fn pool_workers() -> usize {
    global_pool().n_workers
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: when the caller's local batch fails after farm jobs were queued,
    /// `parallel_rows*` must still wait for every queued job before returning the error. The
    /// jobs point at `out`, `remaining` and `error` on the caller's side, so returning early let
    /// workers write into freed memory.
    fn check_local_error_waits_for_workers(aligned: bool) {
        const N: usize = 256;
        const DELAY_MS: u64 = 200;
        // `run_job` builds a slice from `weight`, so it must be non-null even though unused.
        static WEIGHT: [u8; 1] = [0];
        let n_threads = pool_workers().min(4);
        assert!(n_threads >= 2, "test needs at least 2 pool workers");
        let done = AtomicUsize::new(0);
        let queued = AtomicUsize::new(0);
        let local_ran = AtomicBool::new(false);
        let done_ptr: *const AtomicUsize = &done;
        let mut out = vec![0.0f32; N];
        let make_farm = |row_base, n_rows, out_ptr, k, remaining, error| {
            queued.fetch_add(1, Ordering::SeqCst);
            Job {
                weight: WEIGHT.as_ptr(),
                weight_len: WEIGHT.len(),
                k,
                kind: JobKind::TestSleep {
                    delay_ms: DELAY_MS,
                    done: done_ptr,
                },
                out: out_ptr,
                row_base,
                n_rows,
                remaining,
                error,
            }
        };
        // Fails immediately, i.e. after the farm jobs were queued and while they still sleep.
        let local = |_row_base: usize, _out: &mut [f32]| -> Result<()> {
            local_ran.store(true, Ordering::SeqCst);
            Err(CoreError::Shape("injected local batch error".into()))
        };
        let kind = JobKind::TestSleep {
            delay_ms: 0,
            done: done_ptr,
        };
        let r = if aligned {
            parallel_rows_aligned(1, &mut out, n_threads, 8, kind, make_farm, local)
        } else {
            parallel_rows(1, &mut out, n_threads, kind, make_farm, local)
        };

        // The injected error came back, through the queued-jobs path.
        assert!(local_ran.load(Ordering::SeqCst));
        assert!(matches!(r, Err(CoreError::Shape(ref m)) if m.contains("injected")));
        let queued = queued.load(Ordering::SeqCst);
        assert!(queued >= 1, "no farm jobs were queued");
        // Every queued job finished (so `remaining` was 0) before the call returned.
        assert_eq!(done.load(Ordering::SeqCst), queued);
        let farm_rows = out.iter().filter(|&&v| v == 1.0).count();
        assert!(farm_rows > 0 && farm_rows < N);
        // No late writes: nothing changes after the call returned.
        let snapshot = out.clone();
        thread::sleep(std::time::Duration::from_millis(2 * DELAY_MS));
        assert_eq!(out, snapshot);
        assert_eq!(done.load(Ordering::SeqCst), queued);
    }

    /// Like `check_local_error_waits_for_workers`, but the local batch panics after the jobs
    /// were queued: the unwind must not leave `parallel_rows*` before every queued job finished.
    fn check_local_panic_waits_for_workers(aligned: bool) {
        const N: usize = 256;
        const DELAY_MS: u64 = 200;
        static WEIGHT: [u8; 1] = [0];
        let n_threads = pool_workers().min(4);
        assert!(n_threads >= 2, "test needs at least 2 pool workers");
        let done = AtomicUsize::new(0);
        let queued = AtomicUsize::new(0);
        let done_ptr: *const AtomicUsize = &done;
        let mut out = vec![0.0f32; N];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let make_farm = |row_base, n_rows, out_ptr, k, remaining, error| {
                queued.fetch_add(1, Ordering::SeqCst);
                Job {
                    weight: WEIGHT.as_ptr(),
                    weight_len: WEIGHT.len(),
                    k,
                    kind: JobKind::TestSleep {
                        delay_ms: DELAY_MS,
                        done: done_ptr,
                    },
                    out: out_ptr,
                    row_base,
                    n_rows,
                    remaining,
                    error,
                }
            };
            let local = |_row_base: usize, _out: &mut [f32]| -> Result<()> {
                panic!("injected local batch panic");
            };
            let kind = JobKind::TestSleep {
                delay_ms: 0,
                done: done_ptr,
            };
            if aligned {
                parallel_rows_aligned(1, &mut out, n_threads, 8, kind, make_farm, local)
            } else {
                parallel_rows(1, &mut out, n_threads, kind, make_farm, local)
            }
        }));
        // Read before anything else: the unwind must already have waited for every job.
        let done_at_return = done.load(Ordering::SeqCst);
        assert!(r.is_err(), "the injected panic must propagate");
        let queued = queued.load(Ordering::SeqCst);
        assert!(queued >= 1, "no farm jobs were queued");
        assert_eq!(done_at_return, queued);
        let snapshot = out.clone();
        thread::sleep(std::time::Duration::from_millis(2 * DELAY_MS));
        assert_eq!(out, snapshot);
        assert_eq!(done.load(Ordering::SeqCst), queued);
    }

    #[test]
    fn parallel_rows_local_panic_waits_for_queued_jobs() {
        check_local_panic_waits_for_workers(false);
    }

    #[test]
    fn parallel_rows_aligned_local_panic_waits_for_queued_jobs() {
        check_local_panic_waits_for_workers(true);
    }

    /// Live pool worker threads (Linux: thread names in /proc, truncated to 15 bytes). A worker
    /// sets its name only once it first runs, so a worker the scheduler has not started yet is not
    /// counted: poll until the count is `want` or 10 s pass, and return the last count.
    #[cfg(target_os = "linux")]
    fn live_pool_threads(want: usize) -> usize {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let live = std::fs::read_dir("/proc/self/task")
                .unwrap()
                .filter_map(|t| std::fs::read_to_string(t.ok()?.path().join("comm")).ok())
                .filter(|c| c.starts_with("artificer-matmu"))
                .count();
            if live == want || std::time::Instant::now() >= deadline {
                return live;
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// Runs `parallel_rows` on its own thread with panicking (or 1 ms sleep) farm jobs; `None`
    /// on a 10 s timeout (a hang).
    fn run_farm_with_timeout(panic_jobs: bool) -> Option<Result<usize>> {
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            static WEIGHT: [u8; 1] = [0];
            let done = AtomicUsize::new(0);
            let done_ptr: *const AtomicUsize = &done;
            let mut out = vec![0.0f32; 256];
            let kind = move || {
                if panic_jobs {
                    JobKind::TestPanic
                } else {
                    JobKind::TestSleep {
                        delay_ms: 1,
                        done: done_ptr,
                    }
                }
            };
            let make_farm = |row_base, n_rows, out_ptr, k, remaining, error| Job {
                weight: WEIGHT.as_ptr(),
                weight_len: WEIGHT.len(),
                k,
                kind: kind(),
                out: out_ptr,
                row_base,
                n_rows,
                remaining,
                error,
            };
            let n_threads = pool_workers().min(4);
            let r = parallel_rows(1, &mut out, n_threads, kind(), make_farm, |_, _| Ok(()));
            let _ = tx.send(r.map(|()| done.load(Ordering::SeqCst)));
        });
        rx.recv_timeout(std::time::Duration::from_secs(10)).ok()
    }

    /// A panicking farm job must turn into an error (not a hang), and every worker must survive.
    #[test]
    fn parallel_rows_worker_panic_returns_error_and_pool_survives() {
        assert!(
            pool_workers().min(4) >= 2,
            "test needs at least 2 pool workers"
        );
        let r = run_farm_with_timeout(true).expect("parallel_rows hung after a worker panic");
        assert!(matches!(r, Err(CoreError::Backend(ref m)) if m.contains("worker panicked")));
        #[cfg(target_os = "linux")]
        assert_eq!(live_pool_threads(pool_workers()), pool_workers());
        let r = run_farm_with_timeout(false).expect("pool hung after a worker panic");
        assert!(r.expect("normal call after the panic must succeed") >= 1);
    }

    #[test]
    fn parallel_rows_local_error_waits_for_queued_jobs() {
        check_local_error_waits_for_workers(false);
    }

    #[test]
    fn parallel_rows_aligned_local_error_waits_for_queued_jobs() {
        check_local_error_waits_for_workers(true);
    }
}
