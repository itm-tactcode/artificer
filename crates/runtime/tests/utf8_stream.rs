//! Streamed text must hold back partial UTF-8 characters: a CJK / emoji character split
//! across tokens (byte fallback `<0xE4>…`, byte-level BPE) must never stream as U+FFFD.
//! Uses `ARTIFICER_UTF8_TEST_MODEL` (a GGUF file path) when set, else
//! `$ARTIFICER_TEST_MODELS/tiny-vicuna-1b.q4_k_m.gguf`; skips when neither is present.

mod common;

use std::path::PathBuf;

use artificer_core::render_user_chat_prompt;
use artificer_runtime::{
    BackendKind, Engine, EngineLoadOptions, FinishReason, GenerateConfig, StreamEvent,
};

fn engine() -> Option<Engine> {
    // Precedence: `ARTIFICER_UTF8_TEST_MODEL` (documented override, a GGUF file path), then
    // `$ARTIFICER_TEST_MODELS/tiny-vicuna-1b.q4_k_m.gguf`, then skip.
    let p = match std::env::var_os("ARTIFICER_UTF8_TEST_MODEL") {
        Some(v) => {
            let p = PathBuf::from(v);
            if !p.exists() {
                eprintln!("skip: {} not present", p.display());
                return None;
            }
            p
        }
        None => common::test_model("tiny-vicuna-1b.q4_k_m.gguf")?,
    };
    let opts = EngineLoadOptions {
        auto_context: false,
        max_context: 1024,
        backend: BackendKind::Cpu,
        ..EngineLoadOptions::default()
    };
    Some(Engine::load_with(&p, opts).unwrap())
}

#[test]
fn streamed_chinese_text_has_no_replacement_chars_and_matches_output() {
    let Some(mut engine) = engine() else { return };
    let tok = &engine.model().tokenizer;
    let prompt = render_user_chat_prompt(tok, "请用中文写两句话介绍一下北京。")
        .unwrap()
        .encode(tok, true)
        .unwrap();
    let cfg = GenerateConfig {
        max_tokens: 48,
        temperature: 0.0,
        seed: Some(1),
        stop_strings: vec!["\nUSER:".into()],
        ..GenerateConfig::default()
    };
    let mut streamed = String::new();
    let out = engine
        .generate_tokens_with_callback(&prompt, &cfg, |ev| {
            let StreamEvent::Token { text, .. } = ev;
            streamed.push_str(&text);
        })
        .unwrap();
    let one_shot = engine
        .model()
        .tokenizer
        .decode(&out.completion_tokens, true);
    eprintln!("streamed: {streamed:?}\none-shot: {one_shot:?}");
    assert!(
        streamed
            .chars()
            .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c)),
        "no CJK in {streamed:?}"
    );
    if !one_shot.contains('\u{FFFD}') {
        assert!(!streamed.contains('\u{FFFD}'), "{streamed:?}");
        assert!(!out.text.contains('\u{FFFD}'), "{:?}", out.text);
    }
    if out.finish_reason == FinishReason::Length {
        // Nothing was cut at a stop: the streamed concatenation is the output text.
        assert_eq!(streamed, out.text);
    }
}
