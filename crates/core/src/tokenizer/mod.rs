//! GGUF-embedded tokenizers: SentencePiece BPE (`llama`) and GPT-2 BPE (`gpt2`).
//!
//! Both paths first partition the input on the vocab's special tokens
//! (GGUF `token_type` CONTROL = 3 / USER_DEFINED = 4), longest first, exactly like
//! llama.cpp's `tokenizer_st_partition`; each special becomes one id and only the
//! text between them goes through SPM / BPE.

mod bpe;
mod gpt2_bytes;
pub mod pre;
mod utf8_stream;

use std::collections::HashMap;

use gguf::{GgufModel, MetadataValue};

use crate::error::{CoreError, Result};

use bpe::{decode_bpe_pieces, encode_bpe};
use gpt2_bytes::byte_decode_piece_bytes;
use pre::PreType;

pub use utf8_stream::Utf8StreamDecoder;

/// Which algorithm encodes text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// SentencePiece (`tokenizer.ggml.model = llama`, …). GGUF `llama` vocabs are
    /// SentencePiece-BPE: scores are merge priorities, so encoding uses llama.cpp's
    /// greedy highest-score bigram merge (not a unigram log-prob DP). The variant
    /// name is kept for API compatibility.
    Unigram,
    /// GPT-2 BPE with merges (`tokenizer.ggml.model = gpt2`).
    Bpe,
}

/// Tokenizer loaded from GGUF metadata.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub model: String,
    pub kind: TokenizerKind,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub token_types: Vec<i32>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub add_bos: bool,
    pub add_eos: bool,
    /// Optional HuggingFace-style Jinja chat template from GGUF metadata.
    pub chat_template: Option<String>,
    /// BPE pre-tokenizer scheme (meaningful when `kind == Bpe`).
    pub pre: PreType,
    /// piece → id (for encoding)
    piece_to_id: HashMap<String, u32>,
    /// byte value → token id for `<0xXX>` fallbacks (unigram) or raw bytes
    byte_fallback: [Option<u32>; 256],
    /// BPE merge ranks: (left, right) → rank (lower merges first).
    bpe_ranks: HashMap<(String, String), i32>,
    /// Special tokens matched as whole strings before SPM/BPE (CONTROL / USER_DEFINED
    /// types), sorted longest-first (ties: lower id first).
    special_tokens: Vec<(String, u32)>,
    /// SentencePiece: prepend `▁` to a text fragment that starts the input or follows
    /// a special token (`tokenizer.ggml.add_space_prefix`, default true for `llama`).
    add_space_prefix: bool,
}

/// GGUF / llama.cpp token types.
pub const TOKEN_TYPE_NORMAL: i32 = 1;
pub const TOKEN_TYPE_UNKNOWN: i32 = 2;
pub const TOKEN_TYPE_CONTROL: i32 = 3;
pub const TOKEN_TYPE_USER_DEFINED: i32 = 4;
pub const TOKEN_TYPE_UNUSED: i32 = 5;
pub const TOKEN_TYPE_BYTE: i32 = 6;

/// A piece of input text after special-token partitioning.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}

/// [`Fragment`] after merging adjacent text across segments.
enum OwnedFragment {
    Text(String),
    Special(u32),
}

/// One piece of a prompt for [`Tokenizer::encode_segments`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Segment<'a> {
    /// Trusted template text: special-token strings in it become special ids.
    Template(&'a str),
    /// Untrusted text (user / system / assistant / tool content): always plain pieces.
    Text(&'a str),
}

/// `add_bos` when the GGUF has no `tokenizer.ggml.add_bos_token` key. Artificer's default is
/// `true`. llama.cpp (pinned 8642541, `llama_vocab::impl::load`) starts every BPE vocab at
/// `add_bos = false` and only the llama3-style pre-tokenizers turn it on; Qwen1.5-MoE GGUFs
/// (`gpt2` + pre `qwen2`) omit the key, so llama.cpp prepends no BOS there. Scoped to that
/// case (F2): `gpt2` + `qwen2` gets llama.cpp's `false`; every other vocab keeps `true`.
pub(crate) fn default_add_bos(tokenizer_model: Option<&str>, pre: Option<&str>) -> bool {
    !(tokenizer_model == Some("gpt2") && pre == Some("qwen2"))
}

impl Tokenizer {
    pub fn from_gguf(model: &GgufModel) -> Result<Self> {
        let meta = model.metadata();
        let model_name = meta
            .get_str("tokenizer.ggml.model")?
            .unwrap_or("unknown")
            .to_owned();

        let tokens_val = meta
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| CoreError::MissingMetadata("tokenizer.ggml.tokens".into()))?;
        let tokens = extract_string_array(tokens_val, "tokenizer.ggml.tokens")?;

        let scores = match meta.get("tokenizer.ggml.scores") {
            Some(v) => extract_f32_array(v, "tokenizer.ggml.scores")?,
            None => vec![0.0; tokens.len()],
        };
        if scores.len() != tokens.len() && !scores.is_empty() {
            return Err(CoreError::Tokenizer(format!(
                "scores len {} != tokens len {}",
                scores.len(),
                tokens.len()
            )));
        }
        let scores = if scores.len() == tokens.len() {
            scores
        } else {
            vec![0.0; tokens.len()]
        };

        let token_types = match meta.get("tokenizer.ggml.token_type") {
            Some(v) => extract_i32_array(v, "tokenizer.ggml.token_type")?,
            None => vec![1; tokens.len()],
        };

        let bos_id = meta.get_u64("tokenizer.ggml.bos_token_id")?.unwrap_or(1) as u32;
        let eos_id = meta.get_u64("tokenizer.ggml.eos_token_id")?.unwrap_or(2) as u32;
        let add_bos = match meta.get_bool("tokenizer.ggml.add_bos_token")? {
            Some(b) => b,
            None => default_add_bos(
                meta.get_str("tokenizer.ggml.model")?,
                meta.get_str("tokenizer.ggml.pre")?,
            ),
        };
        let add_eos = meta
            .get_bool("tokenizer.ggml.add_eos_token")?
            .unwrap_or(false);

        let chat_template = meta.get_str("tokenizer.chat_template")?.map(str::to_owned);

        let add_space_prefix = meta
            .get_bool("tokenizer.ggml.add_space_prefix")?
            .unwrap_or(true);

        let pre_str = meta.get_str("tokenizer.ggml.pre")?;
        let pre = PreType::from_gguf_pre(pre_str);

        let kind = match model_name.as_str() {
            "gpt2" | "bpe" => TokenizerKind::Bpe,
            "llama" | "spm" | "" | "unknown" => {
                // If merges exist, prefer BPE even when model string is odd.
                if meta.get("tokenizer.ggml.merges").is_some() {
                    TokenizerKind::Bpe
                } else {
                    TokenizerKind::Unigram
                }
            }
            other if other.contains("bpe") => TokenizerKind::Bpe,
            _ => {
                if meta.get("tokenizer.ggml.merges").is_some() {
                    TokenizerKind::Bpe
                } else {
                    TokenizerKind::Unigram
                }
            }
        };

        let mut piece_to_id = HashMap::with_capacity(tokens.len());
        let mut byte_fallback = [None; 256];

        for (i, piece) in tokens.iter().enumerate() {
            piece_to_id.entry(piece.clone()).or_insert(i as u32);

            // Byte fallback pieces look like <0x0A>
            if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&piece[3..5], 16) {
                    byte_fallback[b as usize] = Some(i as u32);
                }
            }
        }

        // GPT-2: each raw byte has a single mapped unicode token in the vocab.
        if kind == TokenizerKind::Bpe {
            for b in 0u8..=255 {
                let mapped = gpt2_bytes::byte_encode_word_bytes(&[b]);
                if let Some(&id) = piece_to_id.get(&mapped) {
                    byte_fallback[b as usize] = Some(id);
                }
            }
        }

        let mut bpe_ranks = HashMap::new();
        if kind == TokenizerKind::Bpe {
            if let Some(merges_val) = meta.get("tokenizer.ggml.merges") {
                let merges = extract_string_array(merges_val, "tokenizer.ggml.merges")?;
                for (rank, line) in merges.iter().enumerate() {
                    // Format: "left right" (single space separating pieces; pieces may contain spaces? usually not)
                    if let Some((left, right)) = line.split_once(' ') {
                        bpe_ranks.insert((left.to_owned(), right.to_owned()), rank as i32);
                    }
                }
            } else {
                return Err(CoreError::MissingMetadata(
                    "tokenizer.ggml.merges (required for gpt2 BPE)".into(),
                ));
            }
        }

        // Special tokens (llama.cpp: CONTROL | USER_DEFINED are parsed out of the text
        // when `parse_special`; UNKNOWN is excluded so a literal "<unk>" stays text).
        let mut special_tokens: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                let ty = token_types.get(*i).copied().unwrap_or(TOKEN_TYPE_NORMAL);
                !t.is_empty() && (ty == TOKEN_TYPE_CONTROL || ty == TOKEN_TYPE_USER_DEFINED)
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        Ok(Self {
            model: model_name,
            kind,
            tokens,
            scores,
            token_types,
            bos_id,
            eos_id,
            add_bos,
            add_eos,
            chat_template,
            pre,
            piece_to_id,
            byte_fallback,
            bpe_ranks,
            special_tokens,
            add_space_prefix,
        })
    }

    /// Special-token strings (CONTROL / USER_DEFINED) that `encode` maps to single ids.
    pub fn special_tokens(&self) -> impl Iterator<Item = (&str, u32)> {
        self.special_tokens.iter().map(|(s, id)| (s.as_str(), *id))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Lookup a vocab piece by exact string (e.g. `<|im_end|>`).
    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.piece_to_id.get(piece).copied()
    }

    /// True for BOS/EOS and GGUF control tokens (token_type == 3), plus ChatML-style
    /// `<|…|>` markers even if typed as normal.
    pub fn is_control_token(&self, id: u32) -> bool {
        if id == self.bos_id || id == self.eos_id {
            return true;
        }
        let ty = self.token_types.get(id as usize).copied().unwrap_or(1);
        // ggml: 1=normal, 2=unknown, 3=control, 4=user_defined, 5=unused, 6=byte
        if ty == TOKEN_TYPE_CONTROL {
            return true;
        }
        if let Some(tok) = self.id_to_token(id) {
            if tok.starts_with("<|") && tok.ends_with("|>") {
                return true;
            }
        }
        false
    }

    /// Decode a single token id to its surface string, lossily on its own. A token
    /// that holds only part of a multi-byte character (`<0xE4>`, a byte-level BPE
    /// piece) yields U+FFFD here; streaming callers should use
    /// [`decode_token_bytes`](Self::decode_token_bytes) with a [`Utf8StreamDecoder`].
    pub fn decode_token(&self, id: u32, skip_special: bool) -> String {
        String::from_utf8_lossy(&self.decode_token_bytes(id, skip_special)).into_owned()
    }

    /// Raw surface bytes of one token (for streaming): `<0xXX>` byte fallback → that
    /// byte; SentencePiece `▁` → space; GPT-2 byte-level pieces → their bytes. Not
    /// necessarily valid UTF-8 on its own; concatenating the bytes of a token run and
    /// decoding once gives the same text as [`decode`](Self::decode).
    pub fn decode_token_bytes(&self, id: u32, skip_special: bool) -> Vec<u8> {
        if skip_special && self.is_control_token(id) {
            return Vec::new();
        }
        if let Some(tok) = self.id_to_token(id) {
            if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                    return vec![b];
                }
            }
            return match self.kind {
                TokenizerKind::Unigram => tok.replace('▁', " ").into_bytes(),
                TokenizerKind::Bpe => byte_decode_piece_bytes(tok),
            };
        }
        Vec::new()
    }

    pub fn bos_token_str(&self) -> String {
        self.id_to_token(self.bos_id).unwrap_or("<s>").to_owned()
    }

    pub fn eos_token_str(&self) -> String {
        self.id_to_token(self.eos_id).unwrap_or("</s>").to_owned()
    }

    /// Encode **trusted** text (template output, CLI `--raw`) to token ids. Special /
    /// control token strings in `text` (e.g. `<|im_start|>`) become single ids, as in
    /// llama.cpp / Ollama with special parsing on. `add_special` adds BOS/EOS per the
    /// GGUF flags. Never pass user-typed text here: use [`encode_with`](Self::encode_with)
    /// with `parse_special = false`, or a chat [`RenderedPrompt`](crate::RenderedPrompt).
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        self.encode_with(text, add_special, true)
    }

    /// Like [`encode`](Self::encode); `parse_special = false` treats CONTROL / UNKNOWN
    /// token strings as plain text but still splits out USER_DEFINED tokens, like
    /// llama.cpp `parse_special = false`.
    pub fn encode_with(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<u32>> {
        if parse_special {
            return self.encode_segments(&[Segment::Template(text)], add_special);
        }
        // llama.cpp `tokenizer_st_partition` (llama-vocab.cpp L3638-3641): with
        // `parse_special = false` only CONTROL / UNKNOWN specials are skipped; USER_DEFINED
        // tokens (e.g. OLMo's whitespace runs) are still split out before BPE / SPM.
        let fragments: Vec<OwnedFragment> = self
            .partition_special_where(text, |id| {
                self.token_types.get(id as usize).copied() == Some(TOKEN_TYPE_USER_DEFINED)
            })
            .into_iter()
            .map(|f| match f {
                Fragment::Text(t) => OwnedFragment::Text(t.to_owned()),
                Fragment::Special(id) => OwnedFragment::Special(id),
            })
            .collect();
        self.encode_fragments(&fragments, add_special)
    }

    /// Encode a prompt made of trusted template pieces and untrusted text pieces.
    ///
    /// Special-token strings are matched only *inside* individual
    /// [`Segment::Template`] pieces; [`Segment::Text`] is always plain text, and a
    /// special can never form across a segment boundary. Adjacent text (from either
    /// kind of segment) is BPE/SPM-encoded as one run, so when no text segment contains
    /// a special string the ids equal `encode(concat, add_special)`.
    pub fn encode_segments(&self, segments: &[Segment<'_>], add_special: bool) -> Result<Vec<u32>> {
        // Partition, then merge adjacent text so tokenization matches the whole string.
        let mut fragments: Vec<OwnedFragment> = Vec::new();
        fn push_text(frags: &mut Vec<OwnedFragment>, t: &str) {
            if t.is_empty() {
                return;
            }
            match frags.last_mut() {
                Some(OwnedFragment::Text(prev)) => prev.push_str(t),
                _ => frags.push(OwnedFragment::Text(t.to_owned())),
            }
        }
        for seg in segments {
            match *seg {
                Segment::Template(t) => {
                    for frag in self.partition_special(t) {
                        match frag {
                            Fragment::Text(x) => push_text(&mut fragments, x),
                            Fragment::Special(id) => fragments.push(OwnedFragment::Special(id)),
                        }
                    }
                }
                Segment::Text(t) => push_text(&mut fragments, t),
            }
        }
        self.encode_fragments(&fragments, add_special)
    }

    /// BOS / EOS + SPM / BPE over already-partitioned fragments.
    fn encode_fragments(&self, fragments: &[OwnedFragment], add_special: bool) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            out.push(self.bos_id);
        }

        // llama.cpp: the SPM space prefix goes on the first fragment and on any text
        // fragment that directly follows a special token.
        let mut prev_special = true;
        for frag in fragments {
            match frag {
                OwnedFragment::Special(id) => {
                    out.push(*id);
                    prev_special = true;
                }
                OwnedFragment::Text(t) => {
                    match self.kind {
                        TokenizerKind::Unigram => {
                            out.extend(self.encode_spm(t, self.add_space_prefix && prev_special)?);
                        }
                        TokenizerKind::Bpe => {
                            out.extend(encode_bpe(
                                t,
                                self.pre,
                                &self.bpe_ranks,
                                &self.piece_to_id,
                                &self.byte_fallback,
                            )?);
                        }
                    }
                    prev_special = false;
                }
            }
        }

        if add_special && self.add_eos {
            out.push(self.eos_id);
        }
        Ok(out)
    }

    /// Decode token ids to a string (strips specials optionally).
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        match self.kind {
            TokenizerKind::Unigram => {
                // Collect raw bytes so `<0xXX>` byte-fallback runs form valid UTF-8.
                let mut bytes = Vec::new();
                for &id in ids {
                    if skip_special && self.is_control_token(id) {
                        continue;
                    }
                    if let Some(tok) = self.id_to_token(id) {
                        if tok.starts_with("<0x") && tok.ends_with('>') && tok.len() == 6 {
                            if let Ok(b) = u8::from_str_radix(&tok[3..5], 16) {
                                bytes.push(b);
                                continue;
                            }
                        }
                        bytes.extend_from_slice(tok.as_bytes());
                    }
                }
                String::from_utf8_lossy(&bytes).replace('▁', " ")
            }
            TokenizerKind::Bpe => {
                let mut pieces = Vec::new();
                for &id in ids {
                    if skip_special && self.is_control_token(id) {
                        continue;
                    }
                    // Skip obvious special / bracketed control markers when skip_special
                    if skip_special {
                        if let Some(tok) = self.id_to_token(id) {
                            if ((tok.starts_with('<') && tok.ends_with('>'))
                                || (tok.starts_with('[') && tok.ends_with(']'))
                                || tok.starts_with("<|")
                                || tok.starts_with("<SPECIAL"))
                                && tok != "Ġ"
                                && !tok.chars().all(|c| c.is_alphanumeric())
                                && self.token_types.get(id as usize).copied().unwrap_or(1) != 1
                            {
                                continue;
                            }
                        }
                    }
                    if let Some(tok) = self.id_to_token(id) {
                        pieces.push(tok.to_owned());
                    }
                }
                decode_bpe_pieces(&pieces)
            }
        }
    }

    /// Split `text` on special-token strings (llama.cpp `tokenizer_st_partition`):
    /// for each special, longest first, cut every occurrence out of the remaining
    /// text fragments. Empty text fragments are dropped.
    fn partition_special<'a>(&self, text: &'a str) -> Vec<Fragment<'a>> {
        self.partition_special_where(text, |_| true)
    }

    /// [`partition_special`](Self::partition_special) restricted to the specials whose id
    /// passes `keep` (same longest-first order).
    fn partition_special_where<'a>(
        &self,
        text: &'a str,
        keep: impl Fn(u32) -> bool,
    ) -> Vec<Fragment<'a>> {
        let mut frags = vec![Fragment::Text(text)];
        if text.is_empty() {
            return frags;
        }
        for (special, id) in &self.special_tokens {
            if !keep(*id) || !text.contains(special.as_str()) {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len() + 2);
            for frag in frags {
                match frag {
                    Fragment::Text(t) if t.contains(special.as_str()) => {
                        let mut rest = t;
                        while let Some(pos) = rest.find(special.as_str()) {
                            if pos > 0 {
                                next.push(Fragment::Text(&rest[..pos]));
                            }
                            next.push(Fragment::Special(*id));
                            rest = &rest[pos + special.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(Fragment::Text(rest));
                        }
                    }
                    other => next.push(other),
                }
            }
            frags = next;
        }
        frags
    }

    /// SentencePiece-BPE encode of one text fragment (llama.cpp `llm_tokenizer_spm`):
    /// spaces → `▁`, optional leading `▁`, then repeatedly merge the adjacent pair whose
    /// concatenation is a vocab piece with the highest score (ties: leftmost), and
    /// finally map symbols to ids (re-splitting via recorded merges, then byte fallback).
    fn encode_spm(&self, text: &str, add_space_prefix: bool) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut normalized = String::with_capacity(text.len() + 4);
        if add_space_prefix {
            normalized.push('▁');
        }
        for ch in text.chars() {
            normalized.push(if ch == ' ' { '▁' } else { ch });
        }
        self.spm_merge(&normalized)
    }

    fn spm_merge(&self, text: &str) -> Result<Vec<u32>> {
        use std::collections::BinaryHeap;

        #[derive(Clone, Copy)]
        struct Sym {
            start: usize,
            len: usize,
            prev: Option<usize>,
            next: Option<usize>,
        }
        struct Bigram {
            score: f32,
            left: usize,
            right: usize,
            size: usize,
        }
        impl PartialEq for Bigram {
            fn eq(&self, o: &Self) -> bool {
                self.cmp(o) == std::cmp::Ordering::Equal
            }
        }
        impl Eq for Bigram {}
        impl PartialOrd for Bigram {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Bigram {
            // Max-heap: higher score first; on ties the lower left index first.
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                self.score
                    .total_cmp(&o.score)
                    .then_with(|| o.left.cmp(&self.left))
            }
        }

        let mut syms: Vec<Sym> = Vec::new();
        for (start, ch) in text.char_indices() {
            let i = syms.len();
            syms.push(Sym {
                start,
                len: ch.len_utf8(),
                prev: i.checked_sub(1),
                next: None,
            });
            if i > 0 {
                syms[i - 1].next = Some(i);
            }
        }

        let mut heap = BinaryHeap::new();
        // merged text → (left text, right text), to re-split pieces that are not in vocab.
        let mut rev_merge: HashMap<&str, (&str, &str)> = HashMap::new();
        #[allow(clippy::too_many_arguments)]
        fn try_add<'t>(
            tok: &Tokenizer,
            text: &'t str,
            heap: &mut BinaryHeap<Bigram>,
            rev: &mut HashMap<&'t str, (&'t str, &'t str)>,
            syms: &[Sym],
            left: Option<usize>,
            right: Option<usize>,
        ) {
            let (Some(l), Some(r)) = (left, right) else {
                return;
            };
            let (ls, rs) = (syms[l], syms[r]);
            let merged = &text[ls.start..ls.start + ls.len + rs.len];
            if let Some(&id) = tok.piece_to_id.get(merged) {
                let score = tok.scores.get(id as usize).copied().unwrap_or(0.0);
                heap.push(Bigram {
                    score,
                    left: l,
                    right: r,
                    size: merged.len(),
                });
                rev.insert(
                    merged,
                    (
                        &text[ls.start..ls.start + ls.len],
                        &text[rs.start..rs.start + rs.len],
                    ),
                );
            }
        }
        for i in 1..syms.len() {
            try_add(
                self,
                text,
                &mut heap,
                &mut rev_merge,
                &syms,
                Some(i - 1),
                Some(i),
            );
        }
        while let Some(bg) = heap.pop() {
            let (l, r) = (bg.left, bg.right);
            if syms[l].len == 0 || syms[r].len == 0 || syms[l].len + syms[r].len != bg.size {
                continue; // stale
            }
            syms[l].len += syms[r].len;
            syms[r].len = 0;
            syms[l].next = syms[r].next;
            if let Some(n) = syms[r].next {
                syms[n].prev = Some(l);
            }
            let (p, n) = (syms[l].prev, syms[l].next);
            try_add(self, text, &mut heap, &mut rev_merge, &syms, p, Some(l));
            try_add(self, text, &mut heap, &mut rev_merge, &syms, Some(l), n);
        }

        let mut ids = Vec::new();
        let mut cur = if syms.is_empty() { None } else { Some(0) };
        while let Some(i) = cur {
            let s = syms[i];
            if s.len > 0 {
                self.spm_resegment(&text[s.start..s.start + s.len], &rev_merge, &mut ids)?;
            }
            cur = s.next;
        }
        Ok(ids)
    }

    fn spm_resegment(
        &self,
        piece: &str,
        rev_merge: &HashMap<&str, (&str, &str)>,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        if let Some(&id) = self.piece_to_id.get(piece) {
            out.push(id);
            return Ok(());
        }
        if let Some(&(l, r)) = rev_merge.get(piece) {
            self.spm_resegment(l, rev_merge, out)?;
            self.spm_resegment(r, rev_merge, out)?;
            return Ok(());
        }
        for &b in piece.as_bytes() {
            let id = self.byte_fallback[b as usize].ok_or_else(|| {
                CoreError::Tokenizer(format!("no vocab piece and no byte fallback for 0x{b:02X}"))
            })?;
            out.push(id);
        }
        Ok(())
    }
}

fn extract_string_array(v: &MetadataValue, key: &str) -> Result<Vec<String>> {
    let arr = v.as_array().ok_or_else(|| CoreError::BadMetadata {
        key: key.into(),
        msg: "expected array".into(),
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for el in &arr.values {
        let s = el.as_str().ok_or_else(|| CoreError::BadMetadata {
            key: key.into(),
            msg: "expected string elements".into(),
        })?;
        out.push(s.to_owned());
    }
    Ok(out)
}

fn extract_f32_array(v: &MetadataValue, key: &str) -> Result<Vec<f32>> {
    let arr = v.as_array().ok_or_else(|| CoreError::BadMetadata {
        key: key.into(),
        msg: "expected array".into(),
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for el in &arr.values {
        let f = el.as_f64().ok_or_else(|| CoreError::BadMetadata {
            key: key.into(),
            msg: "expected float elements".into(),
        })? as f32;
        out.push(f);
    }
    Ok(out)
}

fn extract_i32_array(v: &MetadataValue, key: &str) -> Result<Vec<i32>> {
    let arr = v.as_array().ok_or_else(|| CoreError::BadMetadata {
        key: key.into(),
        msg: "expected array".into(),
    })?;
    let mut out = Vec::with_capacity(arr.len());
    for el in &arr.values {
        let i = el.as_i64().ok_or_else(|| CoreError::BadMetadata {
            key: key.into(),
            msg: "expected int elements".into(),
        })? as i32;
        out.push(i);
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use gguf::MetadataValue;
    use gguf::{MetadataArray, ValueType, GGUF_MAGIC};

    fn write_u32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_u64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fn write_string(buf: &mut Vec<u8>, s: &str) {
        write_u64(buf, s.len() as u64);
        buf.extend_from_slice(s.as_bytes());
    }
    fn write_kv(buf: &mut Vec<u8>, key: &str, val: &MetadataValue) {
        write_string(buf, key);
        write_u32(buf, val.value_type() as u32);
        match val {
            MetadataValue::String(s) => write_string(buf, s),
            MetadataValue::U32(v) => write_u32(buf, *v),
            MetadataValue::Bool(b) => buf.push(u8::from(*b)),
            MetadataValue::Array(a) => {
                write_u32(buf, a.element_type as u32);
                write_u64(buf, a.values.len() as u64);
                for el in &a.values {
                    match el {
                        MetadataValue::String(s) => write_string(buf, s),
                        MetadataValue::F32(f) => write_u32(buf, f.to_bits()),
                        MetadataValue::I32(i) => write_u32(buf, *i as u32),
                        _ => panic!("test helper incomplete"),
                    }
                }
            }
            _ => panic!("test helper incomplete"),
        }
    }

    #[test]
    fn control_token_helpers() {
        let tokens = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "▁hi"];
        let scores = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0];
        let types = [2, 3, 3, 3, 3, 1];

        let mut buf = Vec::new();
        write_u32(&mut buf, GGUF_MAGIC);
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 0);
        write_u64(&mut buf, 6);

        write_kv(
            &mut buf,
            "tokenizer.ggml.model",
            &MetadataValue::String("llama".into()),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.tokens",
            &MetadataValue::Array(MetadataArray {
                element_type: ValueType::String,
                values: tokens
                    .iter()
                    .map(|t| MetadataValue::String((*t).into()))
                    .collect(),
            }),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.scores",
            &MetadataValue::Array(MetadataArray {
                element_type: ValueType::F32,
                values: scores.iter().map(|f| MetadataValue::F32(*f)).collect(),
            }),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.token_type",
            &MetadataValue::Array(MetadataArray {
                element_type: ValueType::I32,
                values: types.iter().map(|i| MetadataValue::I32(*i)).collect(),
            }),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.bos_token_id",
            &MetadataValue::U32(1),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.eos_token_id",
            &MetadataValue::U32(2),
        );

        let model = gguf::GgufModel::from_bytes(buf).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        assert_eq!(tok.token_id("<|im_start|>"), Some(3));
        assert_eq!(tok.token_id("<|im_end|>"), Some(4));
        assert!(tok.is_control_token(3));
        assert!(tok.is_control_token(4));
        assert!(!tok.is_control_token(5));
        assert_eq!(tok.decode_token(3, true), "");
        assert_eq!(tok.decode_token(3, false), "<|im_start|>");
        assert!(tok.decode_token(5, true).contains("hi"));
    }

    #[test]
    fn encode_decode_roundtrip_simple() {
        // SPM-BPE vocabs contain every intermediate merge; "hi" is needed to reach "▁hi".
        let tokens = ["<unk>", "<s>", "</s>", "▁hi", "▁", "h", "i", "hi"];
        let scores = [0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -3.0, -1.5];
        let types = [2, 3, 3, 1, 1, 1, 1, 1];

        let mut buf = Vec::new();
        write_u32(&mut buf, GGUF_MAGIC);
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 0);
        write_u64(&mut buf, 6);

        write_kv(
            &mut buf,
            "tokenizer.ggml.model",
            &MetadataValue::String("llama".into()),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.tokens",
            &MetadataValue::Array(MetadataArray {
                element_type: ValueType::String,
                values: tokens
                    .iter()
                    .map(|t| MetadataValue::String((*t).into()))
                    .collect(),
            }),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.scores",
            &MetadataValue::Array(MetadataArray {
                element_type: ValueType::F32,
                values: scores.iter().map(|f| MetadataValue::F32(*f)).collect(),
            }),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.token_type",
            &MetadataValue::Array(MetadataArray {
                element_type: ValueType::I32,
                values: types.iter().map(|i| MetadataValue::I32(*i)).collect(),
            }),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.bos_token_id",
            &MetadataValue::U32(1),
        );
        write_kv(
            &mut buf,
            "tokenizer.ggml.eos_token_id",
            &MetadataValue::U32(2),
        );

        let model = gguf::GgufModel::from_bytes(buf).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        assert_eq!(tok.kind, TokenizerKind::Unigram);
        let ids = tok.encode("hi", true).unwrap();
        assert_eq!(ids[0], 1);
        assert!(ids.contains(&3), "ids={ids:?}");
        let text = tok.decode(&ids, true);
        assert!(text.contains("hi"), "decoded={text:?}");
    }

    /// Build a tokenizer from an in-memory GGUF with the given vocab.
    pub(crate) fn build_tok(
        model: &str,
        tokens: &[&str],
        scores: Option<&[f32]>,
        types: &[i32],
        merges: Option<&[&str]>,
        extra_bool: &[(&str, bool)],
    ) -> Tokenizer {
        build_tok_with(model, tokens, scores, types, merges, extra_bool, &[])
    }

    /// [`build_tok`] plus extra string keys (e.g. `tokenizer.ggml.pre`).
    pub(crate) fn build_tok_with(
        model: &str,
        tokens: &[&str],
        scores: Option<&[f32]>,
        types: &[i32],
        merges: Option<&[&str]>,
        extra_bool: &[(&str, bool)],
        extra_str: &[(&str, &str)],
    ) -> Tokenizer {
        let mut kvs: Vec<(String, MetadataValue)> = vec![
            (
                "tokenizer.ggml.model".into(),
                MetadataValue::String(model.into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetadataValue::Array(MetadataArray {
                    element_type: ValueType::String,
                    values: tokens
                        .iter()
                        .map(|t| MetadataValue::String((*t).into()))
                        .collect(),
                }),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetadataValue::Array(MetadataArray {
                    element_type: ValueType::I32,
                    values: types.iter().map(|i| MetadataValue::I32(*i)).collect(),
                }),
            ),
            ("tokenizer.ggml.bos_token_id".into(), MetadataValue::U32(1)),
            ("tokenizer.ggml.eos_token_id".into(), MetadataValue::U32(2)),
        ];
        if let Some(sc) = scores {
            kvs.push((
                "tokenizer.ggml.scores".into(),
                MetadataValue::Array(MetadataArray {
                    element_type: ValueType::F32,
                    values: sc.iter().map(|f| MetadataValue::F32(*f)).collect(),
                }),
            ));
        }
        if let Some(m) = merges {
            kvs.push((
                "tokenizer.ggml.merges".into(),
                MetadataValue::Array(MetadataArray {
                    element_type: ValueType::String,
                    values: m
                        .iter()
                        .map(|t| MetadataValue::String((*t).into()))
                        .collect(),
                }),
            ));
        }
        for (k, b) in extra_bool {
            kvs.push(((*k).into(), MetadataValue::Bool(*b)));
        }
        for (k, s) in extra_str {
            kvs.push(((*k).into(), MetadataValue::String((*s).into())));
        }
        let mut buf = Vec::new();
        write_u32(&mut buf, GGUF_MAGIC);
        write_u32(&mut buf, 3);
        write_u64(&mut buf, 0);
        write_u64(&mut buf, kvs.len() as u64);
        for (k, v) in &kvs {
            write_kv(&mut buf, k, v);
        }
        let model = gguf::GgufModel::from_bytes(buf).unwrap();
        Tokenizer::from_gguf(&model).unwrap()
    }

    /// Small SentencePiece vocab: scores are merge priorities (higher merges first),
    /// like real GGUF `llama` vocabs (score = -rank).
    pub(crate) fn spm_tok() -> Tokenizer {
        let tokens = [
            "<unk>",
            "<s>",
            "</s>",
            "<|im_start|>",
            "<|im_end|>",
            "<think>", // 0-5
            "▁",
            "c",
            "h",
            "a",
            "t",
            "i", // 6-11
            "▁c",
            "at",
            "▁ch",
            "▁chat",
            "ch",
            "hi",
            "▁hi", // 12-18
            "<0x0A>",
            "<0xC3>",
            "<0xA9>", // 19-21
        ];
        let scores = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            -0.1, -0.5, -0.5, -0.5, -0.5, -0.5, //
            -1.0, -2.0, -3.0, -10.0, -5.0, -4.0, -6.0, //
            0.0, 0.0, 0.0,
        ];
        let types = [
            2, 3, 3, 3, 3, 4, //
            1, 1, 1, 1, 1, 1, //
            1, 1, 1, 1, 1, 1, 1, //
            6, 6, 6,
        ];
        build_tok("llama", &tokens, Some(&scores), &types, None, &[])
    }

    #[test]
    fn spm_uses_merge_priority_not_unigram_dp() {
        // Unigram DP over scores-as-log-probs would pick ▁c+h+at (-3.5) over ▁chat (-10),
        // which is the tiny-vicuna over-segmentation bug (96 vs Ollama's 50).
        let tok = spm_tok();
        assert_eq!(tok.encode("chat", false).unwrap(), vec![15]);
        assert_eq!(tok.encode("chat", true).unwrap(), vec![1, 15]);
        assert_eq!(tok.encode("hi chat", false).unwrap(), vec![18, 15]);
    }

    #[test]
    fn spm_byte_fallback_and_decode_roundtrip() {
        let tok = spm_tok();
        let ids = tok.encode("hi\né", false).unwrap();
        assert_eq!(ids, vec![18, 19, 20, 21]);
        // Leading space is the SPM space prefix (llama.cpp strips it on detokenize).
        assert_eq!(tok.decode(&ids, false), " hi\né");
    }

    /// Streamed detokenization: per-token bytes through [`Utf8StreamDecoder`].
    fn stream_decode(tok: &Tokenizer, ids: &[u32]) -> String {
        let mut d = Utf8StreamDecoder::new();
        let mut out = String::new();
        for &id in ids {
            out.push_str(&d.push(&tok.decode_token_bytes(id, false)));
        }
        out.push_str(&d.finish());
        out
    }

    /// SPM vocab whose CJK text exists only as `<0xXX>` byte-fallback tokens.
    fn spm_cjk_byte_tok() -> Tokenizer {
        let tokens = [
            "<unk>", "<s>", "</s>", "▁", "h", "i", "▁hi", // 0-6
            "<0xE4>", "<0xBD>", "<0xA0>", "<0xE5>", "<0xA5>", // 7-11
        ];
        let scores = [
            0.0, 0.0, 0.0, -0.1, -0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let types = [2, 3, 3, 1, 1, 1, 1, 6, 6, 6, 6, 6];
        build_tok("llama", &tokens, Some(&scores), &types, None, &[])
    }

    #[test]
    fn spm_byte_fallback_tokens_stream_to_whole_char() {
        let tok = spm_cjk_byte_tok();
        // <0xE4><0xBD><0xA0> = 你
        assert_eq!(stream_decode(&tok, &[7, 8, 9]), "你");
        // Per-token lossy decode is what used to be streamed: one U+FFFD per byte.
        let lossy: String = [7u32, 8, 9]
            .iter()
            .map(|&i| tok.decode_token(i, false))
            .collect();
        assert_eq!(lossy, "\u{FFFD}\u{FFFD}\u{FFFD}");

        // 你好 (E4 BD A0 E5 A5 BD) through the encoder's byte fallback.
        let ids = tok.encode("hi你好", false).unwrap();
        assert_eq!(ids, vec![3, 4, 5, 7, 8, 9, 10, 11, 8]);
        assert_eq!(stream_decode(&tok, &ids), " hi你好");
        assert_eq!(stream_decode(&tok, &ids), tok.decode(&ids, false));
        // Incomplete tail at end of stream: lossy flush, same as a one-shot decode.
        assert_eq!(stream_decode(&tok, &[6, 7, 8]), " hi\u{FFFD}");
        assert_eq!(
            stream_decode(&tok, &[6, 7, 8]),
            tok.decode(&[6, 7, 8], false)
        );
    }

    #[test]
    fn bpe_byte_level_pieces_stream_like_one_shot_decode() {
        let tok = bpe_byte_tok();
        for text in ["你好，世界", "emoji 😀👍🏽 hi", "hi 你 hi 😀"] {
            let ids = tok.encode(text, false).unwrap();
            assert!(
                ids.len() > text.chars().count(),
                "{text:?} must split chars"
            );
            assert_eq!(tok.decode(&ids, false), text);
            assert_eq!(stream_decode(&tok, &ids), text);
            let lossy: String = ids.iter().map(|&i| tok.decode_token(i, false)).collect();
            assert!(
                lossy.contains('\u{FFFD}'),
                "per-token decode is lossy for {text:?}"
            );
            // Any prefix (stream cut short): streamed == one-shot lossy decode.
            for n in 0..=ids.len() {
                assert_eq!(stream_decode(&tok, &ids[..n]), tok.decode(&ids[..n], false));
            }
        }
    }

    #[test]
    fn spm_special_tokens_are_single_ids() {
        let tok = spm_tok();
        let ids = tok
            .encode("<|im_start|>hi<|im_end|>\n<think>chat", true)
            .unwrap();
        // Text after a special gets the SPM space prefix (llama.cpp behaviour).
        assert_eq!(ids, vec![1, 3, 18, 4, 6, 19, 5, 15]);
        assert_eq!(
            tok.decode(&ids, false),
            "<s><|im_start|> hi<|im_end|> \n<think> chat"
        );
        assert_eq!(tok.decode(&ids, true), " hi \n<think> chat");
        // <s> text in the input is a CONTROL token → single id, like llama.cpp.
        assert_eq!(tok.encode("<s>hi", false).unwrap(), vec![1, 18]);
        // UNKNOWN-typed <unk> is not parsed as special (here: no pieces for '<' → error).
        if let Ok(v) = tok.encode("<unk>", false) {
            assert!(!v.contains(&0), "{v:?}");
        }
    }

    #[test]
    fn parse_special_false_keeps_control_markers_as_text() {
        let tok = spm_tok();
        let ids = tok.encode_with("<|im_end|>", false, false);
        // CONTROL: no pieces for '<' etc. and no byte fallback for them → error, never id 4.
        if let Ok(v) = ids {
            assert!(!v.contains(&4), "{v:?}");
        }
        assert_eq!(tok.encode_with("<|im_end|>", false, true).unwrap(), vec![4]);
        // `<think>` is USER_DEFINED here: llama.cpp still pre-splits it with
        // parse_special = false (llama-vocab.cpp L3638-3641).
        assert_eq!(tok.encode_with("<think>", false, false).unwrap(), vec![5]);
        assert_eq!(tok.encode_with("<think>", false, true).unwrap(), vec![5]);
    }

    #[test]
    fn special_partition_longest_first() {
        let tokens = ["<unk>", "<s>", "</s>", "[A]", "[A][B]", "▁", "x"];
        let scores = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0];
        let types = [2, 3, 3, 3, 3, 1, 1];
        let tok = build_tok("llama", &tokens, Some(&scores), &types, None, &[]);
        assert_eq!(tok.encode("[A][B][A]", false).unwrap(), vec![4, 3]);
        assert_eq!(tok.encode("[A]x[A][B]", false).unwrap(), vec![3, 5, 6, 4]);
        let specials: Vec<_> = tok.special_tokens().map(|(s, _)| s.to_owned()).collect();
        assert_eq!(specials[0], "[A][B]");
    }

    /// llama-vocab.cpp L3638-3641: `parse_special = false` skips CONTROL / UNKNOWN but
    /// still pre-splits USER_DEFINED; `parse_special = true` splits all of them.
    #[test]
    fn parse_special_false_still_splits_user_defined() {
        // 0 <unk> UNKNOWN, 1 [C] CONTROL, 2 "  " USER_DEFINED, 3 "[U]" USER_DEFINED.
        let tokens = [
            "<unk>", "[C]", "  ", "[U]", "▁", "x", "[", "]", "C", "U", "<", ">", "u", "n", "k",
        ];
        let scores = [
            0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ];
        let types = [2, 3, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        let tok = build_tok("llama", &tokens, Some(&scores), &types, None, &[]);
        // USER_DEFINED split in both modes; CONTROL only with parse_special.
        assert_eq!(
            tok.encode_with("x[U]x", false, false).unwrap(),
            vec![4, 5, 3, 4, 5]
        );
        assert_eq!(
            tok.encode_with("x[U]x", false, true).unwrap(),
            vec![4, 5, 3, 4, 5]
        );
        assert_eq!(
            tok.encode_with("x[C]x", false, true).unwrap(),
            vec![4, 5, 1, 4, 5]
        );
        let plain = tok.encode_with("x[C]x", false, false).unwrap();
        assert!(!plain.contains(&1), "{plain:?}");
        assert_eq!(plain, vec![4, 5, 6, 8, 7, 5]);
        // SPM space prefix after a special, as in llama.cpp.
        assert_eq!(
            tok.encode_with("x  x", false, false).unwrap(),
            vec![4, 5, 2, 4, 5]
        );
        // UNKNOWN is never split.
        assert!(!tok.encode_with("<unk>", false, false).unwrap().contains(&0));
    }

    #[test]
    fn bpe_parse_special_false_splits_user_defined_only() {
        let tok = bpe_byte_tok();
        let ud = tok.encode_with("a</think>b", false, false).unwrap();
        assert_eq!(ud, tok.encode_with("a</think>b", false, true).unwrap());
        assert!(ud.contains(&5), "{ud:?}");
        let ctl = tok.encode_with("a<|im_end|>b", false, false).unwrap();
        assert!(!ctl.contains(&4), "{ctl:?}");
        assert!(tok
            .encode_with("a<|im_end|>b", false, true)
            .unwrap()
            .contains(&4));
        // Chat Segment::Text stays plain for every special type (template-only specials).
        let seg = tok
            .encode_segments(&[Segment::Text("a</think>b")], false)
            .unwrap();
        assert!(!seg.contains(&5), "{seg:?}");
    }

    /// GPT-2 BPE vocab with byte-level pieces for every char used below.
    pub(crate) fn bpe_tok() -> Tokenizer {
        let tokens = [
            "<unk>",
            "<s>",
            "</s>",
            "<|im_start|>",
            "<|im_end|>",
            "</think>", // 0-5
            "h",
            "i",
            "<",
            "|",
            "m",
            "_",
            "s",
            "t",
            "a",
            "r",
            ">",
            "Ċ",
            "Ġ",
            "e",
            "n",
            "d",
            "/", // 6-22
            "hi",
            "Ġhi",
            "im", // 23-25
        ];
        let types: Vec<i32> = (0..tokens.len())
            .map(|i| match i {
                0 => 2,
                1..=4 => 3,
                5 => 4,
                _ => 1,
            })
            .collect();
        let merges = ["h i", "Ġ hi", "i m"];
        build_tok(
            "gpt2",
            &tokens,
            None,
            &types,
            Some(&merges),
            &[("tokenizer.ggml.add_bos_token", false)],
        )
    }

    /// GPT-2 BPE vocab: `<unk> <s> </s> <|im_start|> <|im_end|> </think>` (ids 0-5,
    /// `</think>` USER_DEFINED) + one piece per byte, a couple of merges. Encodes any text.
    pub(crate) fn bpe_byte_tok() -> Tokenizer {
        let mut tokens: Vec<String> = [
            "<unk>",
            "<s>",
            "</s>",
            "<|im_start|>",
            "<|im_end|>",
            "</think>",
        ]
        .map(String::from)
        .to_vec();
        for b in 0u8..=255 {
            tokens.push(gpt2_bytes::byte_encode_word_bytes(&[b]));
        }
        tokens.push("hi".into());
        tokens.push("Ġhi".into());
        let types: Vec<i32> = (0..tokens.len())
            .map(|i| match i {
                0 => 2,
                1..=4 => 3,
                5 => 4,
                _ => 1,
            })
            .collect();
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        build_tok(
            "gpt2",
            &refs,
            None,
            &types,
            Some(&["h i", "Ġ hi"]),
            &[("tokenizer.ggml.add_bos_token", false)],
        )
    }

    /// F2: key-absent BOS default. llama.cpp gives BPE (`gpt2`) vocabs with pre `qwen2` no BOS
    /// (Qwen1.5-MoE GGUFs omit `add_bos_token`); everything else keeps Artificer's `true`, and
    /// an explicit key always wins.
    #[test]
    fn add_bos_default_is_false_only_for_keyless_gpt2_qwen2() {
        assert!(!default_add_bos(Some("gpt2"), Some("qwen2")));
        assert!(default_add_bos(Some("gpt2"), None));
        assert!(default_add_bos(Some("gpt2"), Some("llama-bpe")));
        assert!(default_add_bos(Some("gpt2"), Some("qwen35")));
        assert!(default_add_bos(Some("llama"), Some("qwen2")));
        assert!(default_add_bos(None, None));

        // gpt2 BPE requires merges (from_gguf refuses a merge-less gpt2 vocab).
        let tokens = ["<unk>", "<s>", "</s>", "a", "b", "ab"];
        let types = [2, 3, 3, 1, 1, 1];
        let merge_list = ["a b"];
        let merges = Some(&merge_list[..]);
        let qwen2 = [("tokenizer.ggml.pre", "qwen2")];
        let tok = build_tok_with("gpt2", &tokens, None, &types, merges, &[], &qwen2);
        assert!(
            !tok.add_bos,
            "gpt2 + qwen2 without the key must not add BOS"
        );
        let tok = build_tok_with(
            "gpt2",
            &tokens,
            None,
            &types,
            merges,
            &[("tokenizer.ggml.add_bos_token", true)],
            &qwen2,
        );
        assert!(tok.add_bos, "explicit add_bos_token = true must win");
        let llama_bpe = [("tokenizer.ggml.pre", "llama-bpe")];
        let tok = build_tok_with("gpt2", &tokens, None, &types, merges, &[], &llama_bpe);
        assert!(tok.add_bos, "other BPE pre-tokenizers keep the old default");
        let tok = build_tok("gpt2", &tokens, None, &types, merges, &[]);
        assert!(tok.add_bos, "no pre key keeps the old default");
    }

    #[test]
    fn bpe_special_tokens_are_single_ids_and_roundtrip() {
        let tok = bpe_tok();
        let text = "<|im_start|>hi hi<|im_end|>\n</think>hi";
        let ids = tok.encode(text, true).unwrap();
        assert_eq!(ids, vec![3, 23, 24, 4, 17, 5, 23]);
        assert_eq!(tok.decode(&ids, false), text);
        // parse_special = false: markers are spelled with ordinary pieces.
        let plain = tok.encode_with("<|im_start|>", false, false).unwrap();
        assert_eq!(plain, vec![8, 9, 25, 11, 12, 13, 14, 15, 13, 9, 16]);
        assert_eq!(tok.decode(&plain, false), "<|im_start|>");
    }

    #[test]
    fn bpe_text_without_specials_unchanged() {
        // Regression: no special strings → identical to the pre-partition BPE path.
        let tok = bpe_tok();
        for text in [
            "hi",
            "hi hi\nhi",
            " hi",
            "<|",
            "im_start",
            "<hi>",
            "hi\n\nhi  hi",
        ] {
            let direct = encode_bpe(
                text,
                tok.pre,
                &tok.bpe_ranks,
                &tok.piece_to_id,
                &tok.byte_fallback,
            )
            .unwrap();
            assert_eq!(tok.encode(text, false).unwrap(), direct, "text={text:?}");
            assert_eq!(tok.decode(&direct, false), text);
        }
    }

    fn repo_file(name: &str) -> Option<std::path::PathBuf> {
        crate::backend::test_model(name)
    }

    /// OLMoE has USER_DEFINED whitespace-run tokens (e.g. id 50276 = two spaces). The
    /// perplexity path (`encode_with(text, true, false)`, one trailing '\n' dropped) on the
    /// cleaned SCROLL.md copy (git blob 123de70e) must reproduce llama.cpp 8642541b's full id
    /// sequence.
    /// Reference provenance (sha256 + generating command): `testdata/README.md`.
    /// Loads `$ARTIFICER_TEST_MODELS/olmoe-1b-7b-0924-instruct-q4_k_m.gguf` (GGUF metadata
    /// only) via `repo_file` / `backend::test_model`; skips when the variable is unset/empty
    /// or the file is missing. No repo-root lookup.
    #[test]
    fn real_gguf_olmoe_scroll_matches_llama_cpp() {
        let Some(path) = repo_file("olmoe-1b-7b-0924-instruct-q4_k_m.gguf") else {
            return;
        };
        let model = gguf::GgufModel::load(path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        let text = include_str!("testdata/scroll-123de70e.md");
        let text = text.strip_suffix('\n').unwrap_or(text);
        let want: Vec<u32> = include_str!("testdata/olmoe-scroll-123de70e.llama-8642541b.ids")
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        assert_eq!(want.len(), 10956);
        let got = tok.encode_with(text, true, false).unwrap();
        if let Some(i) = got.iter().zip(&want).position(|(a, b)| a != b) {
            panic!("first divergence at {i}: got {} want {}", got[i], want[i]);
        }
        assert_eq!(got.len(), want.len());
    }

    /// llama.cpp `tests/test-tokenizer-0` vectors (`models/ggml-vocab-NAME.gguf` + `.inp` /
    /// `.out`; add_special = false, parse_special = false). Vocab-only GGUFs (no tensors) load
    /// through `GgufModel::load` + `Tokenizer::from_gguf`. Runs only when
    /// `ARTIFICER_LLAMA_VOCAB_DIR` points at a llama.cpp `models/` dir (checked against
    /// 8642541b); otherwise skips. Only vocab types that pass every vector are asserted here.
    #[test]
    fn llama_cpp_vocab_vectors() {
        let Some(dir) = std::env::var_os("ARTIFICER_LLAMA_VOCAB_DIR") else {
            eprintln!("skip: ARTIFICER_LLAMA_VOCAB_DIR not set");
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        for name in ["llama-spm", "phi-3", "mpt", "gpt-2"] {
            let g = dir.join(format!("ggml-vocab-{name}.gguf"));
            let model = gguf::GgufModel::load(&g).unwrap();
            let tok = Tokenizer::from_gguf(&model).unwrap();
            let raw = std::fs::read_to_string(g.with_extension("gguf.inp")).unwrap();
            let out = std::fs::read_to_string(g.with_extension("gguf.out")).unwrap();
            let mut inputs: Vec<&str> = raw.split("\n__ggml_vocab_test__\n").collect();
            if inputs.last() == Some(&"") {
                inputs.pop();
            }
            let outputs: Vec<&str> = out.lines().collect();
            assert_eq!(inputs.len(), outputs.len(), "{name}");
            for (text, ids) in inputs.iter().zip(&outputs) {
                let want: Vec<u32> = ids.split_whitespace().map(|t| t.parse().unwrap()).collect();
                assert_eq!(
                    tok.encode_with(text, false, false).unwrap(),
                    want,
                    "{name} {text:?}"
                );
            }
        }
    }

    fn repo_prompt(name: &str) -> String {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/tokenizer/testdata/ollama")
            .join(name);
        std::fs::read_to_string(p).unwrap()
    }

    /// Ids Ollama 0.40.5 produced for the same GGUF + raw prompt (context field,
    /// BOS prepended). Prompt fixtures: `testdata/README.md`.
    #[test]
    fn real_gguf_matches_ollama_tiny_vicuna() {
        let Some(path) = repo_file("tiny-vicuna-1b.q4_k_m.gguf") else {
            return;
        };
        let model = gguf::GgufModel::load(path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        let ids = tok
            .encode(&repo_prompt("tinyvicuna-sky.prompt.txt"), true)
            .unwrap();
        let expected: [u32; 50] = [
            1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450,
            20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915,
            29879, 5155, 29889, 13, 11889, 29901, 12027, 7420, 297, 1023, 25260, 2020, 278, 14744,
            338, 7254, 29889, 13, 22933, 9047, 13566, 29901,
        ];
        assert_eq!(ids, expected);
    }

    #[test]
    fn real_gguf_matches_ollama_nemotron() {
        let Some(path) = repo_file("NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf") else {
            return;
        };
        let model = gguf::GgufModel::load(path).unwrap();
        let tok = Tokenizer::from_gguf(&model).unwrap();
        let ids = tok
            .encode(&repo_prompt("nemotron-sky.prompt.txt"), true)
            .unwrap();
        let expected: [u32; 29] = [
            1, 1010, 10, 25708, 1010, 11, 1010, 10, 3263, 1010, 2763, 44122, 1294, 2295, 35699,
            5782, 1278, 21283, 1395, 10991, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 1010,
        ];
        assert_eq!(ids, expected);
    }
}
