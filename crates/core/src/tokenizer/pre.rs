//! BPE pre-tokenizers (regex-style split before merges).
//!
//! Hand-rolled approximations of llama.cpp / HuggingFace schemes — enough for
//! GPT-2, Llama-3 family (`pixtral`, …), and Qwen2-style pre-types.

/// Which pre-tokenizer scheme to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    /// Classic GPT-2: `'s|'t|…| ?\p{L}+| ?\p{N}+| …`
    Gpt2,
    /// Llama-3 / pixtral / Falcon-H1 style (case-insensitive contractions, digit groups of 1–3).
    Llama3,
    /// Qwen2 / StableLM2: like Llama-3 but single digits (`\p{N}` not `{1,3}`).
    Qwen2,
}

impl PreType {
    pub fn from_gguf_pre(pre: Option<&str>) -> Self {
        match pre.unwrap_or("default") {
            "default" | "gpt2" | "gpt-2" | "mpt" | "olmo" | "jais" | "trillion"
            | "granite-docling" => Self::Gpt2,
            "llama3" | "llama-3" | "llama-v3" | "robotrons" | "dbrx" | "smaug" | "pixtral"
            | "falcon3" | "falcon-h1" | "midm-2.0" | "lfm2" | "jina-v5-nano" | "chatglm-bpe"
            | "chatglm4" => Self::Llama3,
            "qwen2" | "stablelm2" | "hunyuan" | "solar-open" | "qwen2vl" => Self::Qwen2,
            // Unknown → Llama3 is a safe modern default for many gpt2+pre models.
            _ => Self::Llama3,
        }
    }

    /// Whether whole-word vocab hits should skip BPE merges (Llama-3 / pixtral).
    pub fn ignore_merges(self) -> bool {
        matches!(self, Self::Llama3)
    }
}

/// Split `text` into pre-tokens (before GPT-2 byte encoding).
pub fn pre_tokenize(text: &str, pre: PreType) -> Vec<String> {
    match pre {
        PreType::Gpt2 => split_gpt2(text),
        PreType::Llama3 => split_llama3(text, true),
        PreType::Qwen2 => split_llama3(text, false),
    }
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_number(c: char) -> bool {
    c.is_ascii_digit() || c.is_numeric()
}
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn split_gpt2(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        // 's|'t|'re|'ve|'m|'ll|'d
        if chars[i] == '\'' && i + 1 < n {
            let c1 = chars[i + 1];
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                out.push(chars[i..=i + 1].iter().collect());
                i += 2;
                continue;
            }
            if i + 2 < n {
                let c2 = chars[i + 2];
                if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')
                {
                    out.push(chars[i..=i + 2].iter().collect());
                    i += 3;
                    continue;
                }
            }
        }

        // ?\p{L}+
        if chars[i] == ' ' && i + 1 < n && is_letter(chars[i + 1]) {
            let mut j = i + 1;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if is_letter(chars[i]) {
            let mut j = i;
            while j < n && is_letter(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        // ?\p{N}+
        if chars[i] == ' ' && i + 1 < n && is_number(chars[i + 1]) {
            let mut j = i + 1;
            while j < n && is_number(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if is_number(chars[i]) {
            let mut j = i;
            while j < n && is_number(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        // ?[^\s\p{L}\p{N}]+
        if chars[i] == ' '
            && i + 1 < n
            && !is_space(chars[i + 1])
            && !is_letter(chars[i + 1])
            && !is_number(chars[i + 1])
        {
            let mut j = i + 1;
            while j < n && !is_space(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if !is_space(chars[i]) && !is_letter(chars[i]) && !is_number(chars[i]) {
            let mut j = i;
            while j < n && !is_space(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        // whitespace runs
        if is_space(chars[i]) {
            let mut j = i;
            while j < n && is_space(chars[j]) {
                j += 1;
            }
            // \s+(?!\S) keeps last space with following non-space — GPT-2 custom:
            // if multiple spaces and more text, emit all but last, then continue.
            if j - i > 1 && j < n {
                out.push(chars[i..j - 1].iter().collect());
                i = j - 1;
            } else {
                out.push(chars[i..j].iter().collect());
                i = j;
            }
            continue;
        }

        out.push(chars[i].to_string());
        i += 1;
    }
    out
}

/// Llama-3 style (and Qwen2 when `digit_groups` is false).
fn split_llama3(text: &str, digit_groups: bool) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        // case-insensitive contractions
        if chars[i] == '\'' && i + 1 < n {
            let c1 = chars[i + 1].to_ascii_lowercase();
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                out.push(chars[i..=i + 1].iter().collect());
                i += 2;
                continue;
            }
            if i + 2 < n {
                let c2 = chars[i + 2].to_ascii_lowercase();
                let c1l = chars[i + 1].to_ascii_lowercase();
                if (c1l == 'r' && c2 == 'e')
                    || (c1l == 'v' && c2 == 'e')
                    || (c1l == 'l' && c2 == 'l')
                {
                    out.push(chars[i..=i + 2].iter().collect());
                    i += 3;
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        let start = i;
        if !matches!(chars[i], '\r' | '\n')
            && !is_letter(chars[i])
            && !is_number(chars[i])
            && i + 1 < n
            && is_letter(chars[i + 1])
        {
            i += 1;
            while i < n && is_letter(chars[i]) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }
        if is_letter(chars[i]) {
            while i < n && is_letter(chars[i]) {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
            continue;
        }

        // \p{N}{1,3} or \p{N}
        if is_number(chars[i]) {
            let limit = if digit_groups { 3 } else { usize::MAX };
            let mut j = i;
            let mut count = 0usize;
            while j < n && is_number(chars[j]) && count < limit {
                j += 1;
                count += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        if chars[i] == ' '
            && i + 1 < n
            && !is_space(chars[i + 1])
            && !is_letter(chars[i + 1])
            && !is_number(chars[i + 1])
        {
            let mut j = i + 1;
            while j < n && !is_space(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                j += 1;
            }
            while j < n && matches!(chars[j], '\r' | '\n') {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        if !is_space(chars[i]) && !is_letter(chars[i]) && !is_number(chars[i]) {
            let mut j = i;
            while j < n && !is_space(chars[j]) && !is_letter(chars[j]) && !is_number(chars[j]) {
                j += 1;
            }
            while j < n && matches!(chars[j], '\r' | '\n') {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }

        // \s*[\r\n]+
        if is_space(chars[i]) {
            let mut j = i;
            while j < n && is_space(chars[j]) && !matches!(chars[j], '\r' | '\n') {
                j += 1;
            }
            if j < n && matches!(chars[j], '\r' | '\n') {
                while j < n && matches!(chars[j], '\r' | '\n') {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            // \s+(?!\S) / \s+
            let mut k = i;
            while k < n && is_space(chars[k]) {
                k += 1;
            }
            if k - i > 1 && k < n {
                out.push(chars[i..k - 1].iter().collect());
                i = k - 1;
            } else {
                out.push(chars[i..k].iter().collect());
                i = k;
            }
            continue;
        }

        out.push(chars[i].to_string());
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_splits_words() {
        let p = pre_tokenize("Hello world!", PreType::Gpt2);
        assert!(p.iter().any(|s| s.contains("Hello")));
        assert!(p.iter().any(|s| s.contains("world")));
    }

    /// llama.cpp names the GPT-2 pre-tokenizer "gpt-2" (`ggml-vocab-gpt-2.gguf`); without the
    /// alias it fell through to the Llama-3 default (digit groups, `\p{N}` without leading space).
    #[test]
    fn gpt_2_alias_resolves_like_gpt2() {
        assert_eq!(
            PreType::from_gguf_pre(Some("gpt-2")),
            PreType::from_gguf_pre(Some("gpt2"))
        );
        assert_eq!(PreType::from_gguf_pre(Some("gpt-2")), PreType::Gpt2);
        assert_eq!(
            pre_tokenize("ied 4 ½ months 3333", PreType::from_gguf_pre(Some("gpt-2"))),
            pre_tokenize("ied 4 ½ months 3333", PreType::Gpt2)
        );
    }

    #[test]
    fn llama3_digit_groups() {
        let p = pre_tokenize("12345", PreType::Llama3);
        // 1–3 digit chunks
        assert!(p.iter().all(|s| s.chars().all(|c| c.is_ascii_digit())));
        let joined: String = p.concat();
        assert_eq!(joined, "12345");
        assert!(p.iter().all(|s| s.len() <= 3));
    }
}
