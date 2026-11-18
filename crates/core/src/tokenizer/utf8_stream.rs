//! Incremental UTF-8 decoding for streamed detokenization.
//!
//! One multi-byte character can be split across tokens (SentencePiece `<0xE4>`
//! byte-fallback pieces, GPT-2 byte-level BPE pieces). Decoding each token's bytes on
//! its own turns every partial piece into U+FFFD. [`Utf8StreamDecoder`] holds back an
//! incomplete trailing sequence until the next bytes complete it, so the concatenated
//! output equals `String::from_utf8_lossy` of all bytes at once.

/// Streaming UTF-8 decoder: feed raw piece bytes with [`push`](Self::push), call
/// [`finish`](Self::finish) at end of stream.
///
/// Bytes that can never become valid (`Utf8Error::error_len() == Some(_)`) become one
/// U+FFFD per invalid sequence, exactly like `String::from_utf8_lossy`. Only a
/// possibly-incomplete *trailing* sequence (at most 3 bytes) is held back.
#[derive(Debug, Default, Clone)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
}

impl Utf8StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `bytes` and return the text that is now complete (may be empty).
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // The first `valid_up_to` bytes are valid UTF-8, so this never falls back.
                    out.push_str(
                        std::str::from_utf8(&self.pending[start..start + valid])
                            .unwrap_or_default(),
                    );
                    match e.error_len() {
                        Some(n) => {
                            // Invalid sequence that more bytes cannot fix.
                            out.push(char::REPLACEMENT_CHARACTER);
                            start += valid + n;
                        }
                        None => {
                            // Incomplete trailing sequence: keep it for the next push.
                            start += valid;
                            break;
                        }
                    }
                }
            }
        }
        self.pending.drain(..start);
        out
    }

    /// True while an incomplete trailing sequence is held back.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// End of stream: return leftover bytes decoded lossily (U+FFFD for the incomplete
    /// sequence, as a one-shot `from_utf8_lossy` would show) and reset. Empty when
    /// nothing is pending.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }

    /// Drop any held-back bytes without emitting them.
    pub fn reset(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(pieces: &[&[u8]]) -> String {
        let mut d = Utf8StreamDecoder::new();
        let mut out = String::new();
        for p in pieces {
            out.push_str(&d.push(p));
        }
        out.push_str(&d.finish());
        out
    }

    const SAMPLES: &[&str] = &[
        "你好，世界！今天天气很好。",
        "😀🎉👍🏽 family: 👨‍👩‍👧‍👦",
        "mixed: héllo 你 😀 ok",
    ];

    #[test]
    fn one_byte_pieces_roundtrip() {
        for s in SAMPLES {
            let bytes = s.as_bytes();
            let pieces: Vec<&[u8]> = bytes.chunks(1).collect();
            assert_eq!(stream(&pieces), *s);
        }
    }

    #[test]
    fn every_single_split_point_roundtrips() {
        for s in SAMPLES {
            let b = s.as_bytes();
            for i in 0..=b.len() {
                assert_eq!(stream(&[&b[..i], &b[i..]]), *s, "split at {i}");
            }
        }
    }

    #[test]
    fn partial_char_is_held_back() {
        let mut d = Utf8StreamDecoder::new();
        // 你 = E4 BD A0
        assert_eq!(d.push(&[0xE4]), "");
        assert!(d.has_pending());
        assert_eq!(d.push(&[0xBD]), "");
        assert_eq!(d.push(&[0xA0, b'!']), "你!");
        assert!(!d.has_pending());
        assert_eq!(d.finish(), "");
    }

    #[test]
    fn random_splits_roundtrip() {
        // Tiny deterministic xorshift; no deps.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for s in SAMPLES {
            let b = s.as_bytes();
            for _ in 0..500 {
                let mut pieces = Vec::new();
                let mut i = 0;
                while i < b.len() {
                    let n = 1 + (next() % 5) as usize;
                    let j = (i + n).min(b.len());
                    pieces.push(&b[i..j]);
                    i = j;
                }
                assert_eq!(stream(&pieces), *s);
            }
        }
    }

    #[test]
    fn invalid_bytes_match_lossy_under_any_split() {
        let cases: &[&[u8]] = &[
            b"a\xFFb",
            b"\xE4\xBDx",        // truncated 3-byte seq then ASCII
            b"\xC0\x80ok",       // overlong
            b"\xED\xA0\x80",     // surrogate
            b"\xF4\x90\x80\x80", // > U+10FFFF
            b"\x80\x80\xE4\xBD\xA0",
            b"\xF0\x9F\x98", // truncated emoji at end (flush)
            "你".as_bytes(),
        ];
        for c in cases {
            let want = String::from_utf8_lossy(c).into_owned();
            let ones: Vec<&[u8]> = c.chunks(1).collect();
            assert_eq!(stream(&ones), want, "1-byte {c:?}");
            for i in 0..=c.len() {
                assert_eq!(stream(&[&c[..i], &c[i..]]), want, "{c:?} split {i}");
            }
        }
    }

    #[test]
    fn finish_flushes_incomplete_tail_lossily() {
        let mut d = Utf8StreamDecoder::new();
        assert_eq!(d.push("ok".as_bytes()), "ok");
        assert_eq!(d.push(&[0xF0, 0x9F]), "");
        assert_eq!(d.finish(), "\u{FFFD}");
        assert!(!d.has_pending());
        // Decoder is reusable after finish.
        assert_eq!(d.push("你".as_bytes()), "你");
        assert_eq!(d.finish(), "");
    }
}
