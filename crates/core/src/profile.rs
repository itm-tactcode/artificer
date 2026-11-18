//! Optional decode-step wall timers (env `ARTIFICER_PROFILE=1`).
//!
//! Cheap when disabled (one atomic load). Accumulators are process-global so a
//! short bench run can dump % of attributed forward time by bucket.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static ACTIVE: AtomicBool = AtomicBool::new(true);
static INIT: AtomicBool = AtomicBool::new(false);

macro_rules! buckets {
    ($($name:ident),* $(,)?) => {
        $(
            static $name: AtomicU64 = AtomicU64::new(0);
        )*
    };
}

buckets! {
    // By quant type (matmul_vec wall)
    NS_Q5_0,
    NS_Q4_K,
    NS_Q6_K,
    NS_Q8_0,
    NS_MATMUL_OTHER,
    // By role (matmul_wt / ssm matmul wall; overlaps type buckets)
    NS_ROLE_SSM_IN,
    NS_ROLE_SSM_OUT,
    NS_ROLE_FFN_UP,
    NS_ROLE_FFN_DOWN,
    NS_ROLE_ATTN,
    NS_ROLE_LM_HEAD,
    NS_ROLE_OTHER_MM,
    // Non-matmul sections
    NS_SSM_NON_MM,
    NS_ATTN_NON_MM,
    NS_NORM,
    NS_EMBD,
    NS_FORWARD_OTHER,
    // Outer decode accounting
    NS_SAMPLE_TOK,
    NS_DECODE_WALL,
    N_DECODE_STEPS,
}

fn ensure_init() {
    if INIT.swap(true, Ordering::Relaxed) {
        return;
    }
    let on = std::env::var_os("ARTIFICER_PROFILE")
        .map(|v| v != "0" && v != "false" && !v.is_empty())
        .unwrap_or(false);
    ENABLED.store(on, Ordering::Relaxed);
}

#[inline]
pub fn enabled() -> bool {
    ensure_init();
    ENABLED.load(Ordering::Relaxed) && ACTIVE.load(Ordering::Relaxed)
}

/// Gate timers without clearing accumulators (e.g. mute prefill).
#[inline]
pub fn set_active(active: bool) {
    ensure_init();
    ACTIVE.store(active, Ordering::Relaxed);
}

#[inline]
pub fn env_enabled() -> bool {
    ensure_init();
    ENABLED.load(Ordering::Relaxed)
}

#[inline]
fn add(bucket: &AtomicU64, ns: u64) {
    bucket.fetch_add(ns, Ordering::Relaxed);
}

/// RAII timer that adds elapsed ns to `bucket` on drop when profiling is on.
pub struct ScopeTimer {
    start: Option<Instant>,
    bucket: &'static AtomicU64,
}

impl ScopeTimer {
    #[inline]
    pub fn new(bucket: &'static AtomicU64) -> Self {
        if enabled() {
            Self {
                start: Some(Instant::now()),
                bucket,
            }
        } else {
            Self {
                start: None,
                bucket,
            }
        }
    }
}

impl Drop for ScopeTimer {
    fn drop(&mut self) {
        if let Some(t0) = self.start.take() {
            add(self.bucket, t0.elapsed().as_nanos() as u64);
        }
    }
}

#[inline]
pub fn time_q5() -> ScopeTimer {
    ScopeTimer::new(&NS_Q5_0)
}
#[inline]
pub fn time_q4k() -> ScopeTimer {
    ScopeTimer::new(&NS_Q4_K)
}
#[inline]
pub fn time_q6k() -> ScopeTimer {
    ScopeTimer::new(&NS_Q6_K)
}
#[inline]
pub fn time_q80() -> ScopeTimer {
    ScopeTimer::new(&NS_Q8_0)
}
#[inline]
pub fn time_mm_other() -> ScopeTimer {
    ScopeTimer::new(&NS_MATMUL_OTHER)
}
#[inline]
pub fn time_ssm_non_mm() -> ScopeTimer {
    ScopeTimer::new(&NS_SSM_NON_MM)
}
#[inline]
pub fn time_attn_non_mm() -> ScopeTimer {
    ScopeTimer::new(&NS_ATTN_NON_MM)
}
#[inline]
pub fn time_norm() -> ScopeTimer {
    ScopeTimer::new(&NS_NORM)
}
#[inline]
pub fn time_embd() -> ScopeTimer {
    ScopeTimer::new(&NS_EMBD)
}
#[inline]
#[allow(dead_code)]
pub fn time_sample_tok() -> ScopeTimer {
    ScopeTimer::new(&NS_SAMPLE_TOK)
}

/// Classify a GGUF tensor name into a role bucket and time it.
#[inline]
pub fn time_role(name: &str) -> ScopeTimer {
    let b = if name.contains("ssm_in") {
        &NS_ROLE_SSM_IN
    } else if name.contains("ssm_out") {
        &NS_ROLE_SSM_OUT
    } else if name.contains("ffn_up") || name.contains("ffn_gate") {
        &NS_ROLE_FFN_UP
    } else if name.contains("ffn_down") {
        &NS_ROLE_FFN_DOWN
    } else if name.contains("attn_") {
        &NS_ROLE_ATTN
    } else if name == "output.weight" || name.ends_with("output.weight") || name == "output" {
        &NS_ROLE_LM_HEAD
    } else {
        &NS_ROLE_OTHER_MM
    };
    ScopeTimer::new(b)
}

#[inline]
pub fn note_decode_step(wall_ns: u64) {
    if !enabled() {
        return;
    }
    add(&NS_DECODE_WALL, wall_ns);
    N_DECODE_STEPS.fetch_add(1, Ordering::Relaxed);
}

fn load(b: &AtomicU64) -> u64 {
    b.load(Ordering::Relaxed)
}

/// Reset all accumulators (call before a timed decode window).
pub fn reset() {
    ensure_init();
    for b in [
        &NS_Q5_0,
        &NS_Q4_K,
        &NS_Q6_K,
        &NS_Q8_0,
        &NS_MATMUL_OTHER,
        &NS_ROLE_SSM_IN,
        &NS_ROLE_SSM_OUT,
        &NS_ROLE_FFN_UP,
        &NS_ROLE_FFN_DOWN,
        &NS_ROLE_ATTN,
        &NS_ROLE_LM_HEAD,
        &NS_ROLE_OTHER_MM,
        &NS_SSM_NON_MM,
        &NS_ATTN_NON_MM,
        &NS_NORM,
        &NS_EMBD,
        &NS_FORWARD_OTHER,
        &NS_SAMPLE_TOK,
        &NS_DECODE_WALL,
        &N_DECODE_STEPS,
    ] {
        b.store(0, Ordering::Relaxed);
    }
}

/// Human-readable dump to stderr. Percentages are vs sum of exclusive leaf
/// buckets (types + non-mm + embd + norm + sample), not vs overlapping roles.
pub fn dump_stderr(label: &str) {
    if !enabled() {
        return;
    }
    let steps = load(&N_DECODE_STEPS).max(1);
    let q5 = load(&NS_Q5_0);
    let q4 = load(&NS_Q4_K);
    let q6 = load(&NS_Q6_K);
    let q8 = load(&NS_Q8_0);
    let mm_o = load(&NS_MATMUL_OTHER);
    let ssm_nm = load(&NS_SSM_NON_MM);
    let attn_nm = load(&NS_ATTN_NON_MM);
    let norm = load(&NS_NORM);
    let embd = load(&NS_EMBD);
    let sample = load(&NS_SAMPLE_TOK);
    let leaf = q5 + q4 + q6 + q8 + mm_o + ssm_nm + attn_nm + norm + embd + sample;
    let leaf = leaf.max(1);
    let pct = |ns: u64| 100.0 * ns as f64 / leaf as f64;
    let ms_step = |ns: u64| (ns as f64 / steps as f64) / 1e6;

    eprintln!("=== ARTIFICER_PROFILE [{label}] steps={steps} ===");
    eprintln!("-- exclusive leaves (sum≈100% of attributed) --");
    eprintln!(
        "  matmul Q5_0:     {:6.2}%  {:7.3} ms/step",
        pct(q5),
        ms_step(q5)
    );
    eprintln!(
        "  matmul Q4_K:     {:6.2}%  {:7.3} ms/step",
        pct(q4),
        ms_step(q4)
    );
    eprintln!(
        "  matmul Q6_K:     {:6.2}%  {:7.3} ms/step",
        pct(q6),
        ms_step(q6)
    );
    eprintln!(
        "  matmul Q8_0:     {:6.2}%  {:7.3} ms/step",
        pct(q8),
        ms_step(q8)
    );
    eprintln!(
        "  matmul other:    {:6.2}%  {:7.3} ms/step",
        pct(mm_o),
        ms_step(mm_o)
    );
    eprintln!(
        "  SSM non-matmul:  {:6.2}%  {:7.3} ms/step",
        pct(ssm_nm),
        ms_step(ssm_nm)
    );
    eprintln!(
        "  attn non-matmul: {:6.2}%  {:7.3} ms/step",
        pct(attn_nm),
        ms_step(attn_nm)
    );
    eprintln!(
        "  norms:           {:6.2}%  {:7.3} ms/step",
        pct(norm),
        ms_step(norm)
    );
    eprintln!(
        "  embd:            {:6.2}%  {:7.3} ms/step",
        pct(embd),
        ms_step(embd)
    );
    eprintln!(
        "  sample/tok:      {:6.2}%  {:7.3} ms/step",
        pct(sample),
        ms_step(sample)
    );
    let mm_sum = q5 + q4 + q6 + q8 + mm_o;
    eprintln!(
        "  (all matmul):    {:6.2}%  {:7.3} ms/step",
        pct(mm_sum),
        ms_step(mm_sum)
    );

    let r_in = load(&NS_ROLE_SSM_IN);
    let r_out = load(&NS_ROLE_SSM_OUT);
    let r_up = load(&NS_ROLE_FFN_UP);
    let r_dn = load(&NS_ROLE_FFN_DOWN);
    let r_at = load(&NS_ROLE_ATTN);
    let r_lm = load(&NS_ROLE_LM_HEAD);
    let r_ot = load(&NS_ROLE_OTHER_MM);
    let role = (r_in + r_out + r_up + r_dn + r_at + r_lm + r_ot).max(1);
    let rp = |ns: u64| 100.0 * ns as f64 / role as f64;
    eprintln!("-- matmul by role (% of matmul role sum) --");
    eprintln!(
        "  ssm_in:   {:6.2}%  {:7.3} ms/step",
        rp(r_in),
        ms_step(r_in)
    );
    eprintln!(
        "  ssm_out:  {:6.2}%  {:7.3} ms/step",
        rp(r_out),
        ms_step(r_out)
    );
    eprintln!(
        "  ffn_up:   {:6.2}%  {:7.3} ms/step",
        rp(r_up),
        ms_step(r_up)
    );
    eprintln!(
        "  ffn_down: {:6.2}%  {:7.3} ms/step",
        rp(r_dn),
        ms_step(r_dn)
    );
    eprintln!(
        "  attn_*:   {:6.2}%  {:7.3} ms/step",
        rp(r_at),
        ms_step(r_at)
    );
    eprintln!(
        "  lm_head:  {:6.2}%  {:7.3} ms/step",
        rp(r_lm),
        ms_step(r_lm)
    );
    eprintln!(
        "  other_mm: {:6.2}%  {:7.3} ms/step",
        rp(r_ot),
        ms_step(r_ot)
    );

    let wall = load(&NS_DECODE_WALL);
    if wall > 0 {
        let accounted = leaf;
        eprintln!(
            "-- decode wall vs attributed: wall={:.3} ms/step  attributed={:.3} ms/step  coverage={:.1}%",
            ms_step(wall),
            ms_step(accounted),
            100.0 * accounted as f64 / wall as f64
        );
    }
}
