//! F2: Qwen1.5-MoE (`qwen2moe`) on the real GGUF: config, loader shape checks and tokenizer.
//! No forward pass: only GGUF metadata and tensor descriptors are read (weights stay mmapped
//! and untouched). Loads `$ARTIFICER_TEST_MODELS/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf`; prints a
//! `skip:` line and passes when the variable is unset or the file is absent.

mod common;

use artificer_core::{render_chat_prompt, Architecture, ChatMessage, LlamaModel, ModelConfig};

const QM: &str = "Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf";

fn load() -> Option<gguf::GgufModel> {
    let p = common::test_model(QM)?;
    Some(gguf::GgufModel::load(&p).unwrap())
}

#[test]
fn qwen15_moe_config_matches_llama_cpp_defaults() {
    let Some(g) = load() else { return };
    let c = ModelConfig::from_gguf(&g).unwrap();
    assert_eq!(c.architecture, Architecture::Qwen);
    assert_eq!(c.arch_str, "qwen2moe");
    assert_eq!((c.n_embd, c.n_layer, c.n_ff), (2048, 24, 5632));
    assert_eq!((c.n_head, c.n_head_kv, c.head_dim), (16, 16, 128));
    let m = c.moe.as_ref().expect("qwen2moe must parse as MoE");
    assert_eq!(
        (m.n_expert, m.n_expert_used, m.norm_topk_prob),
        (60, 4, false)
    );
    // No expert_*feed_forward_length keys in this GGUF: llama.cpp falls back to
    // n_ff / n_expert_used and n_ff.
    assert_eq!((m.n_ff_exp, m.n_ff_shexp), (Some(1408), Some(5632)));
    assert!(
        c.check_moe_gate(false).is_err(),
        "qwen2moe must stay behind the MoE flag"
    );
    assert!(c.check_moe_gate(true).unwrap().is_some());
    assert!(
        !c.add_bos_token,
        "no add_bos_token key + qwen2 pre: no BOS (llama.cpp parity)"
    );
}

#[test]
fn qwen15_moe_loads_routed_and_shared_expert_tensors() {
    let Some(g) = load() else { return };
    let model = LlamaModel::from_gguf(g).unwrap();
    assert_eq!(model.layers.len(), 24);
    assert!(model.cuda_refusal.is_some(), "MoE layers are CPU-only");
    for (i, l) in model.layers.iter().enumerate() {
        let moe = l
            .moe
            .as_ref()
            .unwrap_or_else(|| panic!("blk.{i}: no MoE weights"));
        assert_eq!(moe.n_ff_exp, 1408, "blk.{i}");
        assert_eq!(moe.gate_inp.shape, vec![2048, 60], "blk.{i}");
        assert_eq!(moe.gate_exps.shape, vec![2048, 1408, 60], "blk.{i}");
        assert_eq!(moe.down_exps.shape, vec![1408, 2048, 60], "blk.{i}");
        let sh = moe
            .shexp
            .as_ref()
            .unwrap_or_else(|| panic!("blk.{i}: no shared expert"));
        assert_eq!(sh.n_ff_shexp, 5632, "blk.{i}");
        assert_eq!(sh.gate_inp_shexp.shape, vec![2048], "blk.{i}");
        assert_eq!(sh.gate_shexp.shape, vec![2048, 5632], "blk.{i}");
        assert_eq!(sh.up_shexp.shape, vec![2048, 5632], "blk.{i}");
        assert_eq!(sh.down_shexp.shape, vec![5632, 2048], "blk.{i}");
        assert!(
            l.ffn_up.is_none() && l.ffn_down.is_none(),
            "blk.{i}: dense FFN on MoE layer"
        );
        assert!(
            l.attn_q_b.is_some() && l.attn_k_b.is_some() && l.attn_v_b.is_some(),
            "blk.{i}"
        );
        assert!(
            l.attn_q_norm.is_none() && l.attn_k_norm.is_none(),
            "blk.{i}"
        );
    }
}

#[test]
fn qwen15_moe_tokenizer_has_no_bos_and_chatml_prompt() {
    let Some(g) = load() else { return };
    let tok = artificer_core::Tokenizer::from_gguf(&g).unwrap();
    assert!(!tok.add_bos);
    assert_eq!((tok.bos_id, tok.eos_id), (151643, 151645));
    let ids = tok.encode("Hello", true).unwrap();
    assert!(!ids.is_empty() && ids[0] != tok.bos_id, "{ids:?}");
    let msgs = vec![ChatMessage::new("user", "Hi")];
    let p = render_chat_prompt(&tok, &msgs, true).unwrap();
    let text = p.text();
    assert!(
        text.contains("<|im_start|>user\nHi<|im_end|>\n"),
        "{text:?}"
    );
    assert!(text.ends_with("<|im_start|>assistant\n"), "{text:?}");
    let ids = p.encode(&tok, true).unwrap();
    assert_ne!(ids.first().copied(), Some(tok.bos_id), "{ids:?}");
}
