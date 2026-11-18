//! Stop during a long CUDA batched prefill (Nemotron, 1024-row chunks).
//! Needs `--features cuda`, a CUDA device and
//! `$ARTIFICER_TEST_MODELS/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`; skips otherwise.
//! Run with `--test-threads=1` like the other CUDA tests.
#![cfg(feature = "cuda")]

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use artificer_core::{
    Backend, CudaBackend, ForwardScratch, KvCache, KvElem, LlamaModel, PrefillStatus, SsmCache,
};
use artificer_runtime::{BackendKind, Engine, EngineLoadOptions, FinishReason, GenerateConfig};

fn greedy(max_tokens: usize) -> GenerateConfig {
    GenerateConfig {
        max_tokens,
        temperature: 0.0,
        seed: Some(1),
        ..GenerateConfig::default()
    }
}

#[test]
fn cuda_stop_ends_long_prefill_within_one_chunk() {
    let Some(p) = common::test_model("NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf") else {
        return;
    };
    let opts = EngineLoadOptions {
        auto_context: false,
        max_context: 4096,
        backend: BackendKind::Cuda,
        ..EngineLoadOptions::default()
    };
    let mut engine = match Engine::load_with(&p, opts) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skip: CUDA engine load failed ({e})");
            return;
        }
    };
    let text =
        "Rayleigh scattering makes the daytime sky look blue, while sunsets look red. ".repeat(200);
    let mut long = engine.model().tokenizer.encode(&text, true).unwrap();
    long.truncate(3072); // 3 chunks of 1024 rows
    assert_eq!(long.len(), 3072);
    let short = engine
        .model()
        .tokenizer
        .encode("The capital of France is", true)
        .unwrap();

    let reference = engine.generate_tokens(&short, &greedy(8)).unwrap();

    // Untriggered: same tokens as the plain API; 3 chunks → 3 prefill polls.
    let mut polls = 0usize;
    let full = engine
        .generate_tokens_cancellable(
            &long,
            &greedy(1),
            |_| {},
            || {
                polls += 1;
                false
            },
        )
        .unwrap();
    let plain = engine.generate_tokens(&long, &greedy(1)).unwrap();
    assert_eq!(full.completion_tokens, plain.completion_tokens);
    assert_eq!(full.timings.prefill_tokens, 3072);
    assert!(polls > 3, "polls {polls}");
    let full_ms = plain.timings.prefill_ms;
    let chunk_ms = full_ms / 3.0;
    eprintln!("full prefill {full_ms:.0} ms (≈{chunk_ms:.0} ms per 1024-row chunk)");

    // Stop from another thread mid-prefill (≈ 40% in, during chunk 2).
    let flag = Arc::new(AtomicBool::new(false));
    let setter = {
        let flag = flag.clone();
        let delay = Duration::from_secs_f64(full_ms * 0.4 / 1000.0);
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            flag.store(true, Ordering::SeqCst);
            Instant::now()
        })
    };
    let out = engine
        .generate_tokens_cancellable(&long, &greedy(8), |_| {}, || flag.load(Ordering::SeqCst))
        .unwrap();
    let returned = Instant::now();
    let set_at = setter.join().unwrap();
    let latency_ms = returned.saturating_duration_since(set_at).as_secs_f64() * 1000.0;
    eprintln!(
        "Stop: processed {} / 3072 tokens, returned {latency_ms:.0} ms after the flag",
        out.timings.prefill_tokens
    );
    assert_eq!(out.finish_reason, FinishReason::Cancelled);
    assert!(out.completion_tokens.is_empty());
    assert!(
        out.timings.prefill_tokens == 1024 || out.timings.prefill_tokens == 2048,
        "whole chunks only: {}",
        out.timings.prefill_tokens
    );
    // Within one chunk of GPU work (plus slack for a noisy desktop GPU).
    assert!(
        latency_ms < chunk_ms * 1.5 + 150.0,
        "latency {latency_ms:.0} ms vs chunk {chunk_ms:.0} ms"
    );

    // Session reusable: identical greedy tokens afterwards.
    let again = engine.generate_tokens(&short, &greedy(8)).unwrap();
    assert_eq!(again.completion_tokens, reference.completion_tokens);
}

/// An untriggered Stop check is numerically free on the real model: Nemotron, 3072 prompt
/// tokens (3 batched chunks of 1024 rows) at the bench shape (n_ctx 10240, f16 KV), plain
/// `prefill` vs `prefill_cancellable` with a check that never fires. Last-position logits must
/// be bit-identical, and so must 16 greedy decode steps (tokens and logits) from each cache.
#[test]
fn cuda_untriggered_check_bit_identical_nemotron() {
    let Some(p) = common::test_model("NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf") else {
        return;
    };
    let cb = match CudaBackend::try_new().and_then(|mut b| {
        b.init_device()?;
        Ok(b)
    }) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip: no usable CUDA device ({e})");
            return;
        }
    };
    let mut model = LlamaModel::load(&p).unwrap();
    model.config.n_ctx = 10240;
    let text =
        "Rayleigh scattering makes the daytime sky look blue, while sunsets look red. ".repeat(200);
    let mut toks = model.tokenizer.encode(&text, true).unwrap();
    toks.truncate(3072);
    assert_eq!(toks.len(), 3072);

    let argmax = |l: &[f32]| {
        l.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 {
                    (i, v)
                } else {
                    b
                }
            })
            .0 as u32
    };
    let mut cache = KvCache::with_elem(&model.config, KvElem::F16);
    let mut ssm = model
        .config
        .ssm
        .map(|_| SsmCache::from_model(&model).unwrap());
    let mut hs = ForwardScratch::new(&model);
    // Returns (prefill logits, 16 greedy tokens, logits of each decode step).
    let mut run = |checked: bool, polls: &mut usize| -> (Vec<f32>, Vec<u32>, Vec<Vec<f32>>) {
        cache.clear();
        if let Some(s) = ssm.as_mut() {
            s.clear();
        }
        if checked {
            let mut never = || {
                *polls += 1;
                false
            };
            let st = model
                .prefill_cancellable(
                    &toks,
                    0,
                    &mut cache,
                    ssm.as_mut(),
                    &mut hs,
                    &cb,
                    Some(&mut never),
                )
                .unwrap();
            assert_eq!(st, PrefillStatus::Complete);
        } else {
            model
                .prefill(&toks, 0, &mut cache, ssm.as_mut(), &mut hs, &cb)
                .unwrap();
        }
        assert_eq!(cache.len, toks.len());
        let first = hs.logits.clone();
        let mut next = argmax(&first);
        let (mut out, mut steps) = (Vec::new(), Vec::new());
        for i in 0..16 {
            out.push(next);
            let l = model
                .forward(
                    next,
                    toks.len() + i,
                    &mut cache,
                    ssm.as_mut(),
                    &mut hs,
                    true,
                    &cb,
                )
                .unwrap()
                .unwrap()
                .to_vec();
            next = argmax(&l);
            steps.push(l);
        }
        (first, out, steps)
    };
    let mut polls = 0usize;
    let (l_plain, t_plain, s_plain) = run(false, &mut polls);
    let (l_chk, t_chk, s_chk) = run(true, &mut polls);
    let max_d = l_plain
        .iter()
        .zip(&l_chk)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "untriggered check: {polls} polls, prefill logits max|d| {max_d:e}, tokens plain {t_plain:?} / checked {t_chk:?}"
    );
    assert_eq!(polls, 3, "check before chunk 1 and after chunks 1-2");
    assert!(
        l_plain
            .iter()
            .zip(&l_chk)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "prefill logits not bit-identical (max |d| {max_d:e})"
    );
    assert_eq!(t_plain, t_chk, "greedy tokens differ");
    for (i, (a, b)) in s_plain.iter().zip(&s_chk).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "decode step {i} logits not bit-identical"
        );
    }
}
