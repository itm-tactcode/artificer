//! GPT-2 style byte ↔ unicode mapping used by many GGUF BPE tokenizers.
//!
//! Printable bytes keep their codepoint; the rest map into U+0100+ so every
//! byte becomes a single Unicode character (e.g. space → U+0120 "Ġ").

use std::sync::OnceLock;

struct Tables {
    /// byte → UTF-8 of mapped codepoint
    byte_to_utf8: [String; 256],
    /// UTF-8 of mapped codepoint → byte
    utf8_to_byte: std::collections::HashMap<String, u8>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut occupied = [false; 256];
        let mut byte_to_cpt = [0u32; 256];
        for ch in 0x21u8..=0x7E {
            occupied[ch as usize] = true;
            byte_to_cpt[ch as usize] = ch as u32;
        }
        for ch in 0xA1u8..=0xAC {
            occupied[ch as usize] = true;
            byte_to_cpt[ch as usize] = ch as u32;
        }
        for ch in 0xAEu8..=0xFF {
            occupied[ch as usize] = true;
            byte_to_cpt[ch as usize] = ch as u32;
        }
        let mut n = 0u32;
        for ch in 0u16..256 {
            if !occupied[ch as usize] {
                byte_to_cpt[ch as usize] = 256 + n;
                n += 1;
            }
        }

        let mut byte_to_utf8: [String; 256] = std::array::from_fn(|_| String::new());
        let mut utf8_to_byte = std::collections::HashMap::with_capacity(256);
        for b in 0..256 {
            let s = cpt_to_utf8(byte_to_cpt[b]);
            utf8_to_byte.insert(s.clone(), b as u8);
            byte_to_utf8[b] = s;
        }
        Tables {
            byte_to_utf8,
            utf8_to_byte,
        }
    })
}

fn cpt_to_utf8(cpt: u32) -> String {
    let mut s = String::new();
    if cpt <= 0x7F {
        s.push(cpt as u8 as char);
    } else if cpt <= 0x7FF {
        s.push(char::from_u32(cpt).unwrap_or('\u{FFFD}'));
    } else {
        s.push(char::from_u32(cpt).unwrap_or('\u{FFFD}'));
    }
    s
}

/// Encode raw UTF-8 bytes of a pre-token with the GPT-2 byte map.
pub fn byte_encode_word(word: &str) -> String {
    byte_encode_word_bytes(word.as_bytes())
}

/// Encode an arbitrary byte slice (each byte → GPT-2 mapped unicode char).
pub fn byte_encode_word_bytes(bytes: &[u8]) -> String {
    let t = tables();
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push_str(&t.byte_to_utf8[b as usize]);
    }
    out
}

/// Inverse of [`byte_encode_word`]: mapped unicode string → original bytes as UTF-8 lossily.
pub fn byte_decode_piece(piece: &str) -> String {
    String::from_utf8_lossy(&byte_decode_piece_bytes(piece)).into_owned()
}

/// Inverse of [`byte_encode_word`] as raw bytes (no UTF-8 validation): one piece may
/// hold only part of a multi-byte character, so streaming callers join bytes first.
pub fn byte_decode_piece_bytes(piece: &str) -> Vec<u8> {
    let t = tables();
    let mut bytes = Vec::with_capacity(piece.len());
    // Walk UTF-8 characters of the piece; each char is one mapped byte.
    for ch in piece.chars() {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        if let Some(&b) = t.utf8_to_byte.get(s) {
            bytes.push(b);
        } else {
            // Fallback: keep the char as UTF-8
            bytes.extend_from_slice(s.as_bytes());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_maps_to_g_style() {
        let enc = byte_encode_word(" t");
        // space is non-printable → U+0100+; 't' stays 't'
        assert!(enc.ends_with('t'));
        assert_ne!(enc.chars().next().unwrap(), ' ');
        let back = byte_decode_piece(&enc);
        assert_eq!(back, " t");
    }

    #[test]
    fn ascii_letters_stable() {
        assert_eq!(byte_encode_word("Hello"), "Hello");
        assert_eq!(byte_decode_piece("Hello"), "Hello");
    }
}
