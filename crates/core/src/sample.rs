//! Token sampling strategies.

use std::cmp::Ordering;

/// Argmax over logits (greedy).
pub fn sample_greedy(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Sampling parameters.
#[derive(Debug, Clone, Copy)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// Sample next token from logits.
///
/// - `temperature <= 0` → greedy
/// - else apply temperature, optional top-k, nucleus top-p, then multinomial
pub fn sample_token(logits: &[f32], params: SampleParams, rng: &mut impl Rng) -> u32 {
    if params.temperature <= 0.0 {
        return sample_greedy(logits);
    }

    let mut scored: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v / params.temperature))
        .collect();

    // top-k
    if params.top_k > 0 && (params.top_k as usize) < scored.len() {
        let k = params.top_k as usize;
        scored.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
    }

    // softmax on remaining
    let max = scored
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in scored.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for (_, v) in scored.iter_mut() {
        *v /= sum;
    }

    // top-p (nucleus)
    if params.top_p > 0.0 && params.top_p < 1.0 {
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let mut cum = 0.0f32;
        let mut cut = scored.len();
        for (i, (_, p)) in scored.iter().enumerate() {
            cum += *p;
            if cum >= params.top_p {
                cut = i + 1;
                break;
            }
        }
        scored.truncate(cut.max(1));
        // renorm
        let s: f32 = scored.iter().map(|(_, p)| *p).sum();
        if s > 0.0 {
            for (_, p) in scored.iter_mut() {
                *p /= s;
            }
        }
    }

    // multinomial
    let r = rng.next_f32();
    let mut cum = 0.0f32;
    for (id, p) in &scored {
        cum += *p;
        if r <= cum {
            return *id;
        }
    }
    scored.last().map(|(id, _)| *id).unwrap_or(0)
}

/// Minimal RNG trait so we don't need an external rand crate for basic sampling.
pub trait Rng {
    fn next_u32(&mut self) -> u32;
    fn next_f32(&mut self) -> f32 {
        // (0, 1]
        let u = self.next_u32() >> 8; // 24 bits
        (u as f32) / (1u32 << 24) as f32
    }
}

/// xorshift64* PRNG (fast, good enough for sampling).
#[derive(Debug, Clone)]
pub struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x4d595df4d0f33173 } else { seed },
        }
    }

    pub fn from_entropy() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xDEAD_BEEF);
        Self::new(seed)
    }
}

impl Rng for XorShiftRng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x).wrapping_mul(0x2545_F491_4F6C_DD1D) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        assert_eq!(sample_greedy(&[1.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn temp_samples_in_range() {
        let mut rng = XorShiftRng::new(42);
        let logits = [0.0f32; 10];
        let id = sample_token(
            &logits,
            SampleParams {
                temperature: 1.0,
                top_k: 5,
                top_p: 0.9,
            },
            &mut rng,
        );
        assert!(id < 10);
    }
}
