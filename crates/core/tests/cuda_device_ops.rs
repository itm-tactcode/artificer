//! CUDA device-op tests (`--features cuda`).
//! Skips cleanly when no device / cudart / cublas is available.

#![cfg(feature = "cuda")]

use artificer_core::{create_backend, Backend, BackendKind, CpuBackend, CudaBackend};
use gguf::TensorType;

fn skip_no_device() -> Option<CudaBackend> {
    match CudaBackend::try_new().and_then(|mut b| {
        b.init_device()?;
        Ok(b)
    }) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skip: no usable CUDA device ({e})");
            None
        }
    }
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-3 * (1.0 + a.abs())
}

fn make_f32_weight(n: usize, k: usize, seed: u32) -> Vec<u8> {
    let mut w = vec![0.0f32; n * k];
    let mut s = seed;
    for v in w.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((s >> 16) as f32 / 65535.0) * 2.0 - 1.0;
    }
    w.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[test]
fn proof_device_gemv_vs_cpu_odd_shapes_and_counters() {
    let Some(b) = skip_no_device() else {
        return;
    };
    let shapes = [
        (1usize, 1usize),
        (3, 5),
        (17, 33),
        (64, 64),
        (127, 63),
        (256, 128),
    ];
    let mut launches = b.gemv_launch_count();
    for &(k, n) in &shapes {
        let bytes = make_f32_weight(n, k, (k * 1000 + n) as u32);
        let mut x = vec![0.0f32; k];
        let mut s = 7u32;
        for v in x.iter_mut() {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            *v = ((s >> 16) as f32 / 65535.0) - 0.5;
        }
        let mut out_cpu = vec![0.0f32; n];
        let mut out_gpu = vec![0.0f32; n];
        CpuBackend
            .matmul_vec(&bytes, TensorType::F32, k, n, &x, &mut out_cpu)
            .unwrap();
        b.matmul_vec(&bytes, TensorType::F32, k, n, &x, &mut out_gpu)
            .expect("device gemv");
        for j in 0..n {
            assert!(
                approx(out_cpu[j], out_gpu[j]),
                "gemv {k}x{n} j={j}: {} vs {}",
                out_cpu[j],
                out_gpu[j]
            );
        }
        let after = b.gemv_launch_count();
        assert!(
            after > launches,
            "gemv_launch_count must increase (silent host fallback?)"
        );
        launches = after;
    }
    assert!(b.resident_weight_bytes() > 0);
    eprintln!(
        "proof GEMV ok on {} launches={} last_ms={:.4} weight_MiB={:.2}",
        b.device_name(),
        b.gemv_launch_count(),
        b.last_gemv_elapsed_ms(),
        b.resident_weight_bytes() as f64 / (1024.0 * 1024.0)
    );
}

#[test]
fn proof_device_gemm_vs_cpu_odd_shapes_and_counters() {
    let Some(b) = skip_no_device() else {
        return;
    };
    let mut launches = b.gemm_launch_count();
    for &(m, n, k) in &[
        (2usize, 3, 4),
        (7, 5, 9),
        (16, 16, 16),
        (31, 17, 13),
        (1, 64, 33),
    ] {
        let mut a = vec![0.0f32; m * k];
        let mut bb = vec![0.0f32; k * n];
        let mut s = 123u32;
        for v in a.iter_mut().chain(bb.iter_mut()) {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = ((s >> 16) as f32 / 65535.0) * 0.5;
        }
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for t in 0..k {
                    sum += a[i * k + t] * bb[t * n + j];
                }
                expected[i * n + j] = sum;
            }
        }
        let mut out = vec![0.0f32; m * n];
        b.gemm_f32(m, n, k, &a, &bb, &mut out).expect("device gemm");
        for i in 0..m * n {
            assert!(
                approx(expected[i], out[i]),
                "gemm {m}x{n}x{k} i={i}: {} vs {}",
                expected[i],
                out[i]
            );
        }
        let after = b.gemm_launch_count();
        assert!(after > launches, "gemm_launch_count must increase");
        launches = after;
    }
    eprintln!(
        "proof GEMM ok on {} launches={}",
        b.device_name(),
        b.gemm_launch_count()
    );
}

#[test]
fn proof_create_backend_cuda_counters() {
    let b = match create_backend(BackendKind::Cuda) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: create_backend(Cuda) failed ({e})");
            return;
        }
    };
    let cuda = b.as_cuda().expect("as_cuda");
    let bytes = make_f32_weight(8, 8, 1);
    let x = vec![1.0f32; 8];
    let mut out = vec![0.0f32; 8];
    let g0 = cuda.gemv_launch_count();
    b.matmul_vec(&bytes, TensorType::F32, 8, 8, &x, &mut out)
        .unwrap();
    assert!(cuda.gemv_launch_count() > g0);
    assert!(cuda.resident_weight_bytes() > 0);
}
