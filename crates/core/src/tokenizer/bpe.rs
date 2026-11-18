//! GPT-2 style BPE merge encode (priority-queue bigram merges).

use std::collections::{BinaryHeap, HashMap};

use super::gpt2_bytes::{byte_decode_piece, byte_encode_word};
use super::pre::{pre_tokenize, PreType};
use crate::error::{CoreError, Result};

#[derive(Clone)]
struct Symbol {
    /// Inclusive start / exclusive end into the encoded word's char list.
    start: usize,
    end: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Eq, PartialEq)]
struct Bigram {
    rank: i32,
    left: usize,
    right: usize,
    /// Combined piece string for staleness check.
    text: String,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // lower rank = higher priority
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Run BPE merges on a GPT-2-byte-encoded word; return final piece strings.
pub fn bpe_merge_word(
    encoded_word: &str,
    ranks: &HashMap<(String, String), i32>,
    piece_to_id: &HashMap<String, u32>,
    ignore_merges: bool,
) -> Vec<String> {
    if encoded_word.is_empty() {
        return Vec::new();
    }
    if ignore_merges && piece_to_id.contains_key(encoded_word) {
        return vec![encoded_word.to_owned()];
    }

    let chars: Vec<char> = encoded_word.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    // One symbol per unicode char of the encoded word.
    let mut symbols: Vec<Symbol> = chars
        .iter()
        .enumerate()
        .map(|(i, _)| Symbol {
            start: i,
            end: i + 1,
            prev: if i == 0 { None } else { Some(i - 1) },
            next: if i + 1 == chars.len() {
                None
            } else {
                Some(i + 1)
            },
        })
        .collect();

    let piece_of = |sym: &Symbol| -> String { chars[sym.start..sym.end].iter().collect() };

    let mut heap = BinaryHeap::new();
    let add_bigram =
        |heap: &mut BinaryHeap<Bigram>, symbols: &[Symbol], left: usize, right: usize| {
            let left_s = piece_of(&symbols[left]);
            let right_s = piece_of(&symbols[right]);
            if let Some(&rank) = ranks.get(&(left_s.clone(), right_s.clone())) {
                heap.push(Bigram {
                    rank,
                    left,
                    right,
                    text: format!("{left_s}{right_s}"),
                });
            }
        };

    for i in 0..symbols.len().saturating_sub(1) {
        add_bigram(&mut heap, &symbols, i, i + 1);
    }

    while let Some(bigram) = heap.pop() {
        let left = bigram.left;
        let right = bigram.right;
        if symbols[left].end == symbols[left].start || symbols[right].end == symbols[right].start {
            continue;
        }
        let left_s = piece_of(&symbols[left]);
        let right_s = piece_of(&symbols[right]);
        if format!("{left_s}{right_s}") != bigram.text {
            continue; // stale
        }
        // Merge right into left.
        symbols[left].end = symbols[right].end;
        symbols[right].start = symbols[right].end; // mark dead
        let next = symbols[right].next;
        symbols[left].next = next;
        if let Some(n) = next {
            symbols[n].prev = Some(left);
        }
        if let Some(p) = symbols[left].prev {
            add_bigram(&mut heap, &symbols, p, left);
        }
        if let Some(n) = symbols[left].next {
            add_bigram(&mut heap, &symbols, left, n);
        }
    }

    let mut out = Vec::new();
    let mut i = Some(0usize);
    // Find first live symbol
    while let Some(idx) = i {
        if symbols[idx].end > symbols[idx].start {
            break;
        }
        i = symbols[idx].next;
    }
    while let Some(idx) = i {
        if symbols[idx].end > symbols[idx].start {
            out.push(piece_of(&symbols[idx]));
        }
        i = symbols[idx].next;
    }
    out
}

/// Encode text → token ids with BPE + pre-tokenizer + GPT-2 byte encoding.
pub fn encode_bpe(
    text: &str,
    pre: PreType,
    ranks: &HashMap<(String, String), i32>,
    piece_to_id: &HashMap<String, u32>,
    byte_fallback: &[Option<u32>; 256],
) -> Result<Vec<u32>> {
    let ignore = pre.ignore_merges();
    let words = pre_tokenize(text, pre);
    let mut ids = Vec::new();
    for word in words {
        let encoded = byte_encode_word(&word);
        let pieces = bpe_merge_word(&encoded, ranks, piece_to_id, ignore);
        for piece in pieces {
            if let Some(&id) = piece_to_id.get(&piece) {
                ids.push(id);
            } else {
                // Byte fallback: decode mapped piece to raw bytes, then <0xXX> or raw byte tokens.
                let raw = byte_decode_piece(&piece);
                for &b in raw.as_bytes() {
                    if let Some(id) = byte_fallback[b as usize] {
                        ids.push(id);
                    } else {
                        // Try single mapped char as token
                        let enc_b = byte_encode_word(std::str::from_utf8(&[b]).unwrap_or(""));
                        if let Some(&id) = piece_to_id.get(&enc_b) {
                            ids.push(id);
                        } else {
                            return Err(CoreError::Tokenizer(format!(
                                "BPE unknown piece {piece:?} and no byte fallback for 0x{b:02X}"
                            )));
                        }
                    }
                }
            }
        }
    }
    Ok(ids)
}

/// Decode BPE token pieces (GPT-2 mapped) back to a UTF-8 string.
pub fn decode_bpe_pieces(pieces: &[String]) -> String {
    let mut mapped = String::new();
    for p in pieces {
        mapped.push_str(p);
    }
    byte_decode_piece(&mapped)
}

#[cfg(test)]
mod tests {
    use super::super::gpt2_bytes::byte_encode_word;
    use super::*;

    #[test]
    fn merge_simple_pair() {
        let mut ranks = HashMap::new();
        // After byte encode, "ab" stays "ab"
        ranks.insert(("a".into(), "b".into()), 0);
        let mut piece_to_id = HashMap::new();
        piece_to_id.insert("a".into(), 1);
        piece_to_id.insert("b".into(), 2);
        piece_to_id.insert("ab".into(), 3);
        let enc = byte_encode_word("ab");
        let pieces = bpe_merge_word(&enc, &ranks, &piece_to_id, false);
        assert_eq!(pieces, vec!["ab".to_string()]);
    }
}
