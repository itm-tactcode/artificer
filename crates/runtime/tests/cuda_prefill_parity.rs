//! Batched-prefill parity tests for `LlamaModel::prefill` (f16 weight tiles + cuBLAS GemmEx).
//!
//! For Nemotron3-Nano-4B Q4_K_M (28-token sky prompt) and tiny-vicuna (50 tokens):
//! This checks parity of Artificer's rendered prompt fed raw to Ollama, not parity with Ollama's own template (27 tokens).
//!   A. CPU, token-by-token `forward()` over the prompt (reference)
//!   B. CUDA, token-by-token `forward()` over the prompt
//!   C. CUDA, batched `LlamaModel::prefill(tokens, 0, cache, ssm, scratch, backend)`
//! then compares last-position logits (C vs B, C vs A) and 32 free-running greedy decode tokens.
//!
//! Tolerances (same rule as the 27/32 → 32/32 near-tie analysis):
//!   * logits: max |Δ| and max |Δ|/max(1,|ref|) are printed. Asserted bounds:
//!     C vs B ≤ `TOL_ABS_VS_CUDA` (batched GEMM / dequantized tiles vs per-token matvec),
//!     C vs A ≤ `TOL_ABS_VS_CPU` (CPU quantizes activations to Q8_K/Q8_0; measured ≈0.2
//!     per step on tiny-vicuna and asserts < 1.0).
//!   * first greedy token and every decode token must match, UNLESS the reference top-1/top-2
//!     margin (either side) at the first differing step is < `NEAR_TIE_MARGIN` (0.06): then it is
//!     reported as a near-tie and comparison stops for that pair (not a failure).
//!
//! Run (GPU must be free; release because the CPU path of hybrid models has debug-only length
//! asserts on oversized scratch):
//!   ARTIFICER_TEST_MODELS=/dir/with/ggufs CARGO_TARGET_DIR=target/cuda \
//!     cargo test --release -p artificer-runtime --features cuda \
//!     --test cuda_prefill_parity -- --nocapture --test-threads 1
//! (`ARTIFICER_TEST_MODELS` holds `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` and
//! `tiny-vicuna-1b.q4_k_m.gguf`; unset or missing files skip.)
//!
//! Enabled: `batched_prefill()` calls `LlamaModel::prefill` (signature as proposed, no adaptation).
#![cfg(feature = "cuda")]

mod common;

use artificer_core::{
    render_user_prompt, Backend, CpuBackend, CudaBackend, ForwardScratch, KvCache, LlamaModel,
    SsmCache,
};

const SKY: &str = "Explain in two sentences why the sky is blue.";
const DECODE_TOKENS: usize = 32;
const NEAR_TIE_MARGIN: f32 = 0.06;
const TOL_ABS_VS_CUDA: f32 = 0.5;
const TOL_ABS_VS_CPU: f32 = 1.0;
const N_CTX: u32 = 256;

fn model_path(name: &str) -> Option<std::path::PathBuf> {
    common::test_model(name)
}

fn cuda() -> Option<CudaBackend> {
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

/// (argmax, top1 - top2)
fn top2(l: &[f32]) -> (u32, f32) {
    let (mut i1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > v1 {
            v2 = v1;
            v1 = v;
            i1 = i;
        } else if v > v2 {
            v2 = v;
        }
    }
    (i1 as u32, v1 - v2)
}

fn diffs(a: &[f32], reference: &[f32]) -> (f32, f32) {
    a.iter()
        .zip(reference)
        .fold((0.0f32, 0.0f32), |(ma, mr), (&x, &y)| {
            let d = (x - y).abs();
            (ma.max(d), mr.max(d / y.abs().max(1.0)))
        })
}

struct Lane {
    cache: KvCache,
    ssm: Option<SsmCache>,
    scratch: ForwardScratch,
}

impl Lane {
    fn new(model: &LlamaModel) -> Self {
        Self {
            cache: KvCache::new(&model.config),
            ssm: model
                .config
                .ssm
                .map(|_| SsmCache::from_model(model).expect("SSM statics")),
            scratch: ForwardScratch::new(model),
        }
    }
}

/// Token-by-token prefill (what runtime/lib.rs does today); returns last-position logits.
fn token_prefill(
    model: &LlamaModel,
    lane: &mut Lane,
    tokens: &[u32],
    be: &dyn Backend,
) -> Vec<f32> {
    let last = tokens.len() - 1;
    let mut out = Vec::new();
    for (pos, &t) in tokens.iter().enumerate() {
        let r = model
            .forward(
                t,
                pos,
                &mut lane.cache,
                lane.ssm.as_mut(),
                &mut lane.scratch,
                pos == last,
                be,
            )
            .expect("forward");
        if pos == last {
            out = r.expect("logits").to_vec();
        }
    }
    out
}

/// Batched prefill through the batched entry (`LlamaModel::prefill`).
fn batched_prefill(
    model: &LlamaModel,
    lane: &mut Lane,
    tokens: &[u32],
    be: &dyn Backend,
) -> Option<Vec<f32>> {
    Some(
        model
            .prefill(
                tokens,
                0,
                &mut lane.cache,
                lane.ssm.as_mut(),
                &mut lane.scratch,
                be,
            )
            .expect("batched prefill")
            .to_vec(),
    )
}

/// Free-running greedy decode of `n` tokens from `first` logits at position `pos0`.
/// Returns (token, margin) per step, step 0 = the token chosen from `first`.
fn greedy(
    model: &LlamaModel,
    lane: &mut Lane,
    first: &[f32],
    pos0: usize,
    n: usize,
    be: &dyn Backend,
) -> Vec<(u32, f32)> {
    let mut out = vec![top2(first)];
    for i in 1..n {
        let tok = out[i - 1].0;
        let l = model
            .forward(
                tok,
                pos0 + i - 1,
                &mut lane.cache,
                lane.ssm.as_mut(),
                &mut lane.scratch,
                true,
                be,
            )
            .expect("decode")
            .expect("logits");
        out.push(top2(l));
    }
    out
}

/// Compare two greedy streams under the near-tie rule. Returns a one-line verdict; panics on a
/// real (non-near-tie) divergence.
fn compare_streams(name: &str, what: &str, a: &[(u32, f32)], b: &[(u32, f32)]) -> String {
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        if x.0 != y.0 {
            let m = x.1.min(y.1);
            assert!(
                m < NEAR_TIE_MARGIN,
                "{name} {what}: greedy diverges at step {k} ({} vs {}) with margins {:.4}/{:.4} >= {NEAR_TIE_MARGIN} — not a near-tie",
                x.0, y.0, x.1, y.1
            );
            return format!(
                "{what}: {k}/{} match, then NEAR-TIE at step {k} ({} vs {}, margins {:.4}/{:.4})",
                a.len(),
                x.0,
                y.0,
                x.1,
                y.1
            );
        }
    }
    format!("{what}: {}/{} match", a.len(), a.len())
}

fn parity(name: &str, expect_prompt_tokens: usize) {
    let Some(path) = model_path(name) else {
        return;
    };
    let Some(cb) = cuda() else { return };
    let mut model = LlamaModel::load(&path).expect("load");
    model.config.n_ctx = N_CTX;
    let (text, _) = render_user_prompt(&model.tokenizer, SKY).expect("template");
    let tokens = model.tokenizer.encode(&text, true).expect("encode");
    eprintln!("{name}: prompt tokens {}", tokens.len());
    assert_eq!(
        tokens.len(),
        expect_prompt_tokens,
        "{name}: prompt token count (Ollama parity)"
    );
    let cpu_ok = !(model.config.architecture.is_hybrid() && cfg!(debug_assertions));
    if !cpu_ok {
        eprintln!("{name}: CPU legs skipped in debug builds (run with --release)");
    }

    let mut lb = Lane::new(&model);
    let Some(l_batch) = batched_prefill(&model, &mut lb, &tokens, &cb) else {
        eprintln!("skip: {name}: LlamaModel::prefill not available yet");
        return;
    };
    let mut lt = Lane::new(&model);
    let l_tok = token_prefill(&model, &mut lt, &tokens, &cb);

    let (da, dr) = diffs(&l_batch, &l_tok);
    let (tb, mb) = top2(&l_batch);
    let (tt, mt) = top2(&l_tok);
    eprintln!(
        "{name}: last-pos logits batched vs token CUDA: max|Δ| {da:.4e} max rel {dr:.4e}; top1 {tb}/{tt} margins {mb:.4}/{mt:.4}"
    );
    assert!(
        da <= TOL_ABS_VS_CUDA,
        "{name}: batched vs token CUDA logits |Δ| {da}"
    );
    let n = tokens.len();
    let gb = greedy(&model, &mut lb, &l_batch, n, DECODE_TOKENS, &cb);
    let gt = greedy(&model, &mut lt, &l_tok, n, DECODE_TOKENS, &cb);
    let v1 = compare_streams(name, "batched-CUDA vs token-CUDA", &gb, &gt);

    let mut v2 = "batched-CUDA vs CPU: skipped (debug build)".to_string();
    if cpu_ok {
        let cpu = CpuBackend;
        let mut lc = Lane::new(&model);
        let l_cpu = token_prefill(&model, &mut lc, &tokens, &cpu);
        let (da, dr) = diffs(&l_batch, &l_cpu);
        let (tc, mc) = top2(&l_cpu);
        eprintln!(
            "{name}: last-pos logits batched CUDA vs CPU: max|Δ| {da:.4e} max rel {dr:.4e}; top1 {tb}/{tc} margins {mb:.4}/{mc:.4}"
        );
        assert!(
            da <= TOL_ABS_VS_CPU,
            "{name}: batched CUDA vs CPU logits |Δ| {da}"
        );
        let gc = greedy(&model, &mut lc, &l_cpu, n, DECODE_TOKENS, &cpu);
        v2 = compare_streams(name, "batched-CUDA vs CPU", &gb, &gc);
    }
    eprintln!("{name}: {v1}; {v2}");
}

#[test]
fn prefill_parity_nemotron_q4_k_m() {
    parity("NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf", 28);
}

#[test]
fn prefill_parity_tiny_vicuna() {
    parity("tiny-vicuna-1b.q4_k_m.gguf", 50);
}
