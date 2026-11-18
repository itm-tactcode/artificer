//! Reusable activation buffers for the forward pass.

use crate::model::LlamaModel;

/// Scratch buffers reused across tokens to limit allocator pressure.
pub struct ForwardScratch {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub logits: Vec<f32>,
    /// Temporary KV row when cache stores f16.
    pub kv_row: Vec<f32>,
    /// Hybrid Mamba-2 workspaces (empty for dense models).
    pub ssm_zx: Vec<f32>,
    pub ssm_xbc: Vec<f32>,
    pub ssm_y: Vec<f32>,
    pub ssm_tmp: Vec<f32>,
    pub(crate) norm_w: Vec<f32>,
    pub(crate) norm_b: Vec<f32>,
    /// MoE router logits → probabilities `[n_expert]` (empty for dense models).
    pub(crate) moe_logits: Vec<f32>,
    /// MoE weighted expert sum / one expert's down output `[n_embd]` (empty for dense).
    pub(crate) moe_acc: Vec<f32>,
    pub(crate) moe_out: Vec<f32>,
    /// Selected `(expert, weight)` pairs for the current token.
    pub(crate) moe_sel: Vec<(usize, f32)>,
}

impl ForwardScratch {
    pub fn new(model: &LlamaModel) -> Self {
        let cfg = &model.config;
        let d = cfg.n_embd as usize;
        let n_ff_exp = model
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref().map(|m| m.n_ff_exp))
            .max()
            .unwrap_or(0);
        // Qwen2-MoE shared expert runs through hb / hb2 too (may be wider than n_ff).
        let n_ff_shexp = model
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref()?.shexp.as_ref().map(|s| s.n_ff_shexp))
            .max()
            .unwrap_or(0);
        let n_ff = (cfg.n_ff as usize).max(n_ff_exp).max(n_ff_shexp).max(1);
        let (moe_logits, moe_acc, moe_out) = match cfg.moe {
            Some(m) => (vec![0.0; m.n_expert as usize], vec![0.0; d], vec![0.0; d]),
            None => (Vec::new(), Vec::new(), Vec::new()),
        };
        let n_vocab = cfg.n_vocab as usize;
        let n_ctx = cfg.n_ctx as usize;
        let n_head = cfg.n_head.max(1) as usize;
        let n_kv = cfg.n_embd_kv().max(1) as usize;
        // Attention Q may be larger than n_embd for hybrid (e.g. 40*128=5120 vs embd 3136).
        let q_dim = (cfg.n_head as usize)
            .saturating_mul(cfg.head_dim as usize)
            .max(d);

        let (ssm_zx, ssm_xbc, ssm_y, ssm_tmp) = if let Some(ssm) = cfg.ssm {
            let d_in = ssm.d_in_proj() as usize;
            let conv_dim = ssm.conv_dim() as usize;
            let d_inner = ssm.d_inner as usize;
            (
                vec![0.0; d_in],
                vec![0.0; conv_dim],
                vec![0.0; d_inner],
                vec![0.0; ssm.n_head.max(1) as usize],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        // norm weights may be ssm_norm size up to d_inner
        let norm_cap = cfg.ssm.map(|s| s.d_inner as usize).unwrap_or(d).max(d);

        Self {
            x: vec![0.0; d],
            xb: vec![0.0; d.max(q_dim)],
            xb2: vec![0.0; q_dim],
            q: vec![0.0; q_dim],
            k: vec![0.0; n_kv],
            v: vec![0.0; n_kv],
            att: vec![0.0; n_head * n_ctx],
            hb: vec![0.0; n_ff],
            hb2: vec![0.0; n_ff],
            logits: vec![0.0; n_vocab],
            kv_row: vec![0.0; n_kv],
            ssm_zx,
            ssm_xbc,
            ssm_y,
            ssm_tmp,
            norm_w: vec![0.0; norm_cap],
            norm_b: vec![0.0; d],
            moe_logits,
            moe_acc,
            moe_out,
            moe_sel: Vec::new(),
        }
    }
}
