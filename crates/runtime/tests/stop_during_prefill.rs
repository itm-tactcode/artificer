//! Stop during prefill (CPU token-by-token path).
//! Loads `$ARTIFICER_TEST_MODELS/tiny-vicuna-1b.q4_k_m.gguf` and skips when it is unset or the
//! file is absent. The CUDA batched path is covered by `cuda_stop_during_prefill.rs`
//! (`--features cuda`).

mod common;

use artificer_core::render_user_chat_prompt;
use artificer_runtime::{BackendKind, Engine, EngineLoadOptions, FinishReason, GenerateConfig};

fn engine() -> Option<Engine> {
    let p = common::test_model("tiny-vicuna-1b.q4_k_m.gguf")?;
    let opts = EngineLoadOptions {
        auto_context: false,
        max_context: 1024,
        backend: BackendKind::Cpu,
        ..EngineLoadOptions::default()
    };
    Some(Engine::load_with(&p, opts).unwrap())
}

fn greedy(max_tokens: usize) -> GenerateConfig {
    GenerateConfig {
        max_tokens,
        temperature: 0.0,
        seed: Some(1),
        ..GenerateConfig::default()
    }
}

#[test]
fn stop_during_prefill_cpu_ends_within_one_token_and_session_is_reusable() {
    let Some(mut engine) = engine() else { return };
    let short = render_user_chat_prompt(
        &engine.model().tokenizer,
        "What color is the sky on a clear day? Answer in one short sentence.",
    )
    .unwrap()
    .encode(&engine.model().tokenizer, true)
    .unwrap();
    let long_text = "The quick brown fox jumps over the lazy dog. ".repeat(40);
    let long = engine.model().tokenizer.encode(&long_text, true).unwrap();
    assert!(long.len() > 200, "long prompt has {} tokens", long.len());

    // Reference: the non-cancellable API.
    let reference = engine.generate_tokens(&short, &greedy(8)).unwrap();
    assert!(!reference.completion_tokens.is_empty());

    // An untriggered check is free: same tokens as the non-cancellable path.
    let mut polls = 0usize;
    let out = engine
        .generate_tokens_cancellable(
            &short,
            &greedy(8),
            |_| {},
            || {
                polls += 1;
                false
            },
        )
        .unwrap();
    assert_eq!(out.completion_tokens, reference.completion_tokens);
    assert_eq!(out.finish_reason, reference.finish_reason);
    // Polled once per prompt token during prefill, then before each sampling step.
    assert!(
        polls >= short.len() + out.completion_tokens.len(),
        "polls {polls}"
    );

    // Stop on the 9th poll: 8 prompt tokens processed, then a clean return with no
    // completion — the rest of the long prompt is never run.
    let stop_at = 9usize;
    let mut polls = 0usize;
    let mut streamed = 0usize;
    let out = engine
        .generate_tokens_cancellable(
            &long,
            &greedy(8),
            |_| streamed += 1,
            || {
                polls += 1;
                polls >= stop_at
            },
        )
        .unwrap();
    assert_eq!(out.finish_reason, FinishReason::Cancelled);
    assert!(out.completion_tokens.is_empty() && out.text.is_empty());
    assert_eq!(streamed, 0);
    assert_eq!(polls, stop_at, "no polls after the Stop");
    assert_eq!(out.timings.prefill_tokens, stop_at - 1);
    assert_eq!(out.timings.decode_tokens, 0);
    assert_eq!(out.prompt_tokens, long);

    // Stop before any prompt token runs.
    let out = engine
        .generate_tokens_cancellable(&long, &greedy(8), |_| {}, || true)
        .unwrap();
    assert_eq!(out.finish_reason, FinishReason::Cancelled);
    assert_eq!(out.timings.prefill_tokens, 0);

    // The session is reusable: the next generation matches the reference exactly.
    let again = engine.generate_tokens(&short, &greedy(8)).unwrap();
    assert_eq!(again.completion_tokens, reference.completion_tokens);
    assert_eq!(again.text, reference.text);
}
