//! Finish reasons, cancellation and thinking-token accounting
//! (CPU only). Engine tests load `$ARTIFICER_TEST_MODELS/<file>` and skip when it is unset or
//! the file is absent.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use artificer_core::render_user_chat_prompt;
use artificer_runtime::{Engine, FinishReason, GenerateConfig, ThinkTracker};

fn model(name: &str) -> Option<PathBuf> {
    common::test_model(name)
}

fn greedy(max_tokens: usize) -> GenerateConfig {
    GenerateConfig {
        max_tokens,
        temperature: 0.0,
        seed: Some(1),
        ..GenerateConfig::default()
    }
}

const OPEN: u32 = 12;
const CLOSE: u32 = 13;

fn track(prompt: &[u32], completion: &[u32]) -> artificer_runtime::ThinkingStats {
    let mut t = ThinkTracker::with_ids(Some(OPEN), Some(CLOSE), prompt);
    for &id in completion {
        t.observe(id);
    }
    let start = Instant::now();
    t.finish(start, start + Duration::from_millis(50))
}

#[test]
fn think_tracker_counts_through_close_inclusive() {
    // Prompt ends inside `<think>` (Nemotron generation cue).
    let s = track(&[1, 10, OPEN, 5], &[100, 101, 102, CLOSE, 200, 201]);
    assert_eq!(s.tokens, 4);
    assert!(!s.unterminated);
    assert!(s.ms.is_some());
}

#[test]
fn think_tracker_cap_mid_think_counts_everything() {
    let s = track(&[1, OPEN], &[100, 101, 102]);
    assert_eq!(s.tokens, 3);
    assert!(s.unterminated);
    assert_eq!(s.ms, Some(50.0));
}

#[test]
fn think_tracker_no_thinking() {
    // No think tokens at all, and thinking disabled in the prompt (`<think></think>`).
    for prompt in [&[1u32, 10, 5][..], &[1, OPEN, CLOSE][..]] {
        let s = track(prompt, &[100, 101]);
        assert_eq!(s, artificer_runtime::ThinkingStats::default());
    }
    // Vocab without think tokens.
    let mut t = ThinkTracker::with_ids(None, None, &[1, 2]);
    t.observe(5);
    let now = Instant::now();
    assert_eq!(t.finish(now, now).tokens, 0);
}

#[test]
fn think_tracker_model_opened_and_last_close_wins() {
    let s = track(&[1, 10], &[100, OPEN, 101, CLOSE, 102, OPEN, CLOSE, 103]);
    assert_eq!(s.tokens, 7);
    assert!(!s.unterminated);
}

#[test]
fn tiny_vicuna_length_stop_and_cancel() {
    let Some(path) = model("tiny-vicuna-1b.q4_k_m.gguf") else {
        return;
    };
    let mut engine = Engine::load(&path).unwrap();
    let prompt = render_user_chat_prompt(
        &engine.model().tokenizer,
        "What color is the sky on a clear day? Answer in one short sentence.",
    )
    .unwrap();

    let out = engine.generate_prompt(&prompt, &greedy(3)).unwrap();
    assert_eq!(out.finish_reason, FinishReason::Length);
    assert_eq!(out.completion_tokens.len(), 3);
    assert_eq!(out.thinking.tokens, 0);
    assert!(!out.limit_hit_in_thinking());

    let out = engine.generate_prompt(&prompt, &greedy(64)).unwrap();
    assert_eq!(out.finish_reason, FinishReason::Stop);
    assert!(out.completion_tokens.len() < 64);
    assert!(out.text.to_lowercase().contains("blue"), "{:?}", out.text);

    // Cancel after two streamed tokens. (The check is also polled during prefill, so
    // count streamed tokens rather than polls.)
    let streamed = std::cell::Cell::new(0usize);
    let out = engine
        .generate_prompt_cancellable(
            &prompt,
            &greedy(64),
            |_| streamed.set(streamed.get() + 1),
            || streamed.get() >= 2,
        )
        .unwrap();
    assert_eq!(out.finish_reason, FinishReason::Cancelled);
    assert_eq!(out.completion_tokens.len(), 2);
}

/// Nemotron's generation cue opens `<think>`; a small cap ends mid-think.
#[test]
fn nemotron_cap_hit_mid_think() {
    let Some(path) = model("NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf") else {
        return;
    };
    let mut engine = Engine::load(&path).unwrap();
    let prompt = render_user_chat_prompt(&engine.model().tokenizer, "What is 17 * 24?").unwrap();
    assert!(prompt.text().ends_with("<think>\n"));
    let out = engine.generate_prompt(&prompt, &greedy(4)).unwrap();
    assert_eq!(out.finish_reason, FinishReason::Length);
    assert_eq!(out.completion_tokens.len(), 4);
    assert_eq!(out.thinking.tokens, 4);
    assert!(out.thinking.unterminated);
    assert!(out.limit_hit_in_thinking());
    assert!(out.thinking.ms.is_some());
}
