//! IEEE-754 binary16 helpers.

/// f32 → IEEE-754 binary16, round to nearest, ties to even (the same result as llama.cpp /
/// ggml `GGML_FP32_TO_FP16`, F16C `vcvtps2ph` with RNE, and PTX `cvt.rn.f16.f32`).
///
/// - Normal results: round bit = the highest dropped bit, sticky = the OR of the rest; a carry out
///   of the 10-bit mantissa moves into the exponent, and a carry out of exponent 30 gives ±inf
///   (so every |x| ≥ 65520 maps to ±inf, and 65504 stays the largest finite value).
/// - f16 subnormal results (|x| < 2^-14) are rounded on the 2^-24 grid with the same rule; rounding
///   up out of the subnormal range gives the smallest normal (0x0400) naturally. |x| ≤ 2^-25 → ±0
///   (2^-25 exactly is a tie to even zero). f32 subnormals → ±0.
/// - ±0 and ±inf are preserved; NaN stays NaN: sign kept, quiet bit (0x0200) forced, top payload
///   bits kept, so a payload living only in the low 13 bits can never turn into ±inf.
#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;

    if exp == 0xFF {
        if man == 0 {
            return sign | 0x7C00; // ±inf
        }
        return sign | 0x7C00 | 0x0200 | ((man >> 13) as u16 & 0x03FF); // quiet NaN
    }
    if exp == 0 {
        return sign; // ±0 and f32 subnormals (< 2^-126) → ±0
    }
    // f16 biased exponent of the (unrounded) value.
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00; // |x| ≥ 65536 → ±inf
    }
    if e <= 0 {
        // f16 subnormal (or zero) result: value / 2^-24 = m >> shift (plus rounding).
        let m = man | 0x80_0000; // 24-bit significand with the implicit bit
        let shift = (14 - e) as u32; // ≥ 14
        if shift > 24 {
            return sign; // |x| < 2^-25 → ±0
        }
        let mut h = m >> shift;
        let round = (m >> (shift - 1)) & 1;
        let sticky = m & ((1u32 << (shift - 1)) - 1);
        if round == 1 && (sticky != 0 || (h & 1) == 1) {
            h += 1; // may carry into 0x0400 = smallest normal: correct encoding
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let round = (man >> 12) & 1;
    let sticky = man & 0x0FFF;
    if round == 1 && (sticky != 0 || (h & 1) == 1) {
        h += 1; // carry may move into the exponent; out of e = 30 it gives 0x7C00 = inf
    }
    sign | h as u16
}

/// IEEE-754 binary16 → f32.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;

    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (frac << 13)
    } else {
        let e = exp + (127 - 15);
        (sign << 31) | (e << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::{f16_to_f32, f32_to_f16};

    fn h(x: f32) -> u16 {
        f32_to_f16(x)
    }
    fn fb(b: u32) -> f32 {
        f32::from_bits(b)
    }

    #[test]
    fn every_f16_value_round_trips() {
        for b in 0..=u16::MAX {
            let v = f16_to_f32(b);
            if v.is_nan() {
                continue;
            }
            assert_eq!(h(v), b, "{b:#06x}");
        }
    }

    /// Exhaustive over every adjacent finite pair (lo, lo + 1), subnormals and every exponent
    /// carry included: the exact midpoint ties to the even one, one f32 ulp below rounds down,
    /// one above rounds up; both signs.
    #[test]
    fn ties_to_even_and_neighbours_exhaustive() {
        for lo in 0u16..0x7BFF {
            let (a, b) = (f16_to_f32(lo) as f64, f16_to_f32(lo + 1) as f64);
            let mid = ((a + b) / 2.0) as f32;
            assert_eq!(mid as f64, (a + b) / 2.0, "midpoint exact in f32");
            let even = if lo & 1 == 0 { lo } else { lo + 1 };
            let below = fb(mid.to_bits() - 1);
            let above = fb(mid.to_bits() + 1);
            for (sg, sb) in [(1.0f32, 0u16), (-1.0, 0x8000)] {
                assert_eq!(h(sg * mid), even | sb, "tie {lo:#06x}");
                assert_eq!(h(sg * below), lo | sb, "below tie {lo:#06x}");
                assert_eq!(h(sg * above), (lo + 1) | sb, "above tie {lo:#06x}");
            }
        }
    }

    #[test]
    fn named_cases() {
        let u = 2f32.powi(-10); // f16 ulp at 1.0
                                // ties to even, both ways
        assert_eq!(h(1.0 + u / 2.0), 0x3C00, "tie down to even");
        assert_eq!(h(1.0 + 1.5 * u), 0x3C02, "tie up to even");
        assert_eq!(h(-(1.0 + 1.5 * u)), 0xBC02);
        // round up / down off the tie (truncation would give 0x3C00 for the first)
        assert_eq!(h(1.0 + 0.6 * u), 0x3C01, "round up");
        assert_eq!(h(1.0 + 0.4 * u), 0x3C00, "round down");
        assert_eq!(h(0.1), 0x2E66);
        assert_eq!(h(0.01), 0x211F, "0.01 rounds up (truncation gave 0x211E)");
        // mantissa carry into the next exponent
        assert_eq!(
            h(fb(0x3F7F_F000)),
            0x3C00,
            "1 - 2^-12 (tie, odd) carries to 1.0"
        );
        assert_eq!(h(fb(0x3F7F_EFFF)), 0x3BFF);
        assert_eq!(h(fb(0x477F_E000)), 0x7BFF, "65504 exact");
        // max finite and overflow at the boundary
        assert_eq!(h(65504.0), 0x7BFF);
        assert_eq!(h(fb(65520f32.to_bits() - 1)), 0x7BFF, "just below 65520");
        assert_eq!(h(65520.0), 0x7C00, "65520 ties (odd) up to inf");
        assert_eq!(h(-65520.0), 0xFC00);
        assert_eq!(h(65536.0), 0x7C00);
        assert_eq!(h(f32::MAX), 0x7C00);
        assert_eq!(h(f32::MIN), 0xFC00);
        // subnormals
        let s = 2f32.powi(-24); // smallest f16 subnormal
        assert_eq!(h(s), 0x0001, "smallest subnormal");
        assert_eq!(h(-s), 0x8001);
        assert_eq!(h(s / 2.0), 0x0000, "2^-25: tie to even zero");
        assert_eq!(h(-s / 2.0), 0x8000);
        assert_eq!(h(fb((s / 2.0).to_bits() + 1)), 0x0001, "just above 2^-25");
        assert_eq!(h(1.5 * s), 0x0002, "subnormal tie up to even");
        assert_eq!(h(2.5 * s), 0x0002, "subnormal tie down to even");
        assert_eq!(h(0.7 * s), 0x0001);
        assert_eq!(h(1023.0 * s), 0x03FF, "largest subnormal");
        assert_eq!(
            h(1023.5 * s),
            0x0400,
            "rounds up out of the subnormal range"
        );
        assert_eq!(h(fb((1023.5 * s).to_bits() - 1)), 0x03FF);
        assert_eq!(h(2f32.powi(-14)), 0x0400, "smallest normal");
        assert_eq!(
            h(fb(2f32.powi(-14).to_bits() - 1)),
            0x0400,
            "rounds into the normal range"
        );
        // underflow to ±0, f32 subnormals
        assert_eq!(h(1e-10), 0x0000);
        assert_eq!(h(-1e-10), 0x8000);
        assert_eq!(h(fb(0x0000_0001)), 0x0000);
        assert_eq!(h(fb(0x8000_0001)), 0x8000);
        assert_eq!(h(fb(0x007F_FFFF)), 0x0000);
        // ±0, ±inf
        assert_eq!(h(0.0), 0x0000);
        assert_eq!(h(-0.0), 0x8000);
        assert_eq!(h(f32::INFINITY), 0x7C00);
        assert_eq!(h(f32::NEG_INFINITY), 0xFC00);
    }

    #[test]
    fn nan_stays_quiet_nan_with_sign() {
        for b in [
            0x7FC0_0000u32,
            0x7F80_0001,
            0x7F80_1FFF,
            0x7FBF_FFFF,
            0x7FFF_FFFF,
            0xFFC0_0000,
            0xFF80_0001,
            0xFF80_1FFF,
        ] {
            let r = h(fb(b));
            assert_eq!(r & 0x7C00, 0x7C00, "{b:#010x}");
            assert_ne!(r & 0x03FF, 0, "{b:#010x} must not become inf");
            assert_eq!(r & 0x0200, 0x0200, "{b:#010x} quiet");
            assert_eq!(r >> 15, (b >> 31) as u16, "{b:#010x} sign");
            assert!(f16_to_f32(r).is_nan());
        }
    }

    /// Reference fixture from numpy `float32.astype(float16)` (round to nearest even), numpy 2.x,
    /// generator `gen_f16_fixture.py` (sha in the fixture header). NaN rows: numpy's NaN payload
    /// rule differs (it may return a signalling 0x7C01), so they only check NaN-ness and sign.
    #[test]
    fn matches_numpy_float16_fixture() {
        let txt = include_str!("testdata/f16_rne_numpy.txt");
        let mut n = 0usize;
        for line in txt.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.split_whitespace();
            let x = u32::from_str_radix(it.next().unwrap(), 16).unwrap();
            let want = u16::from_str_radix(it.next().unwrap(), 16).unwrap();
            let got = h(fb(x));
            if fb(x).is_nan() {
                assert!(f16_to_f32(got).is_nan(), "{x:#010x}");
                assert_eq!(got & 0x8000, want & 0x8000, "{x:#010x} sign");
            } else {
                assert_eq!(got, want, "{x:#010x}: got {got:#06x} numpy {want:#06x}");
            }
            n += 1;
        }
        assert!(n >= 4000, "fixture rows {n}");
    }

    /// Nearest-even property against exact f64 arithmetic (f32 values and f16 neighbours are all
    /// exact in f64, so differences are exact).
    fn check_nearest_even(x: f32) {
        let r = h(x);
        assert_eq!(
            r & 0x8000,
            if x.is_sign_negative() { 0x8000 } else { 0 },
            "{x:e} sign"
        );
        let ax = (x as f64).abs();
        let m = r & 0x7FFF;
        if m == 0x7C00 {
            assert!(ax >= 65520.0, "{x:e} -> inf");
            return;
        }
        assert!(m < 0x7C00, "{x:e} -> {r:#06x}");
        let v = f16_to_f32(m) as f64;
        let lo = if m > 0 {
            f16_to_f32(m - 1) as f64
        } else {
            -(f16_to_f32(1) as f64)
        };
        let hi = if m < 0x7BFF {
            f16_to_f32(m + 1) as f64
        } else {
            65536.0
        };
        let (d, dl, dh) = ((ax - v).abs(), ax - lo, hi - ax);
        if m == 0x7BFF {
            assert!(ax < 65520.0, "{x:e}");
        }
        assert!(d <= dl && d <= dh, "{x:e} -> {r:#06x} not nearest");
        if d != 0.0 && (d == dl || d == dh) {
            assert_eq!(m & 1, 0, "{x:e} tie must go to even");
        }
    }

    #[test]
    fn strided_sweep_all_f32_is_nearest_even() {
        let mut b: u64 = 0;
        let mut n = 0u64;
        while b <= u32::MAX as u64 {
            let x = fb(b as u32);
            if !x.is_nan() {
                check_nearest_even(x);
                n += 1;
            }
            b += 4099;
        }
        assert!(n > 1_000_000);
    }

    /// Every f32 bit pattern (4.3e9). Slow: `cargo test --release -- --ignored f16_exhaustive`.
    #[test]
    #[ignore]
    fn f16_exhaustive_all_f32_is_nearest_even() {
        for b in 0..=u32::MAX {
            let x = fb(b);
            if !x.is_nan() {
                check_nearest_even(x);
            }
        }
    }
}
