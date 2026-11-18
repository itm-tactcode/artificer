//! End-to-end: tiny-vicuna temp=0 tokens match CPU vs CUDA; counters must move.
#![cfg(feature = "cuda")]

mod common;

use artificer_core::{attention_launch_count, Backend, BackendKind};
use artificer_runtime::{Engine, EngineLoadOptions, GenerateConfig};

fn model_path() -> Option<std::path::PathBuf> {
    common::test_model("tiny-vicuna-1b.q4_k_m.gguf")
}

#[test]
fn tiny_vicuna_temp0_cpu_vs_cuda_tokens() {
    let Some(path) = model_path() else {
        return;
    };

    let cfg = GenerateConfig {
        max_tokens: 8,
        temperature: 0.0,
        ..GenerateConfig::default()
    };

    let mut cpu = match Engine::load_with(
        &path,
        EngineLoadOptions {
            auto_context: false,
            max_context: 256,
            kv_f16: false,
            backend: BackendKind::Cpu,
            ..EngineLoadOptions::default()
        },
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skip: CPU engine load failed ({e})");
            return;
        }
    };

    let mut gpu = match Engine::load_with(
        &path,
        EngineLoadOptions {
            auto_context: false,
            max_context: 256,
            kv_f16: false,
            backend: BackendKind::Cuda,
            ..EngineLoadOptions::default()
        },
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skip: CUDA engine load failed ({e})");
            return;
        }
    };

    let prompt = "What color is the sky?";
    let attn_before = attention_launch_count();
    let quant_before = Backend::as_cuda(gpu.backend())
        .map(|c| c.quant_matvec_launch_count())
        .unwrap_or(0);
    let softmax_before = Backend::as_cuda(gpu.backend())
        .map(|c| c.softmax_launch_count())
        .unwrap_or(0);
    let gemm_before = Backend::as_cuda(gpu.backend())
        .map(|c| c.gemm_launch_count())
        .unwrap_or(0);
    let fused_before = Backend::as_cuda(gpu.backend())
        .map(|c| c.fused_attention_launch_count())
        .unwrap_or(0);

    let out_cpu = cpu.generate(prompt, &cfg).expect("CPU generate");
    let out_gpu = match gpu.generate(prompt, &cfg) {
        Ok(o) => o,
        Err(e) => panic!("CUDA generate failed (device error): {e}"),
    };

    assert_eq!(
        out_cpu.completion_tokens, out_gpu.completion_tokens,
        "temp0 tokens diverge\nCPU text={:?}\nGPU text={:?}",
        out_cpu.text, out_gpu.text
    );

    let cuda = Backend::as_cuda(gpu.backend()).expect("cuda backend");
    assert!(
        cuda.quant_matvec_launch_count() > quant_before,
        "quant_matvec_launch_count did not increase — quantized kernels not used?"
    );
    assert!(
        cuda.softmax_launch_count() > softmax_before,
        "softmax_launch_count did not increase — softmax not on device?"
    );
    assert!(
        // Legacy path: cuBLAS gemm per head; device-resident path: fused attention kernel.
        cuda.gemm_launch_count() > gemm_before
            || cuda.fused_attention_launch_count() > fused_before,
        "neither gemm nor fused attention launch count increased — attention not on device?"
    );
    assert!(
        attention_launch_count() > attn_before,
        "attention_launch_count did not increase"
    );
    assert!(cuda.resident_weight_bytes() > 0);
    let file_bytes = std::fs::metadata(&path).unwrap().len();
    assert!(
        cuda.resident_weight_bytes() <= file_bytes,
        "resident {} > GGUF file {} — weights expanded instead of quantized-resident",
        cuda.resident_weight_bytes(),
        file_bytes
    );
    eprintln!(
        "e2e token match ok — tokens={:?} quant={} softmax={} gemv={} gemm={} attn={} weight_MiB={:.1} last_gemv_ms={:.4}",
        out_gpu.completion_tokens,
        cuda.quant_matvec_launch_count(),
        cuda.softmax_launch_count(),
        cuda.gemv_launch_count(),
        cuda.gemm_launch_count(),
        attention_launch_count(),
        cuda.resident_weight_bytes() as f64 / (1024.0 * 1024.0),
        cuda.last_gemv_elapsed_ms()
    );
}
