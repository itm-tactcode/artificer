//! GGML block sizes and Q4_K / Q5_K scale packing.

pub(crate) const QK_K: usize = 256;
pub(crate) const QK4_0: usize = 32;
pub(crate) const QK5_0: usize = 32;
pub(crate) const QK8_0: usize = 32;

pub(crate) const BLOCK_Q4_0: usize = 18; // 2 + 16
pub(crate) const BLOCK_Q4_1: usize = 20; // d + m + 16
pub(crate) const BLOCK_Q5_0: usize = 22; // 2 + 4 + 16
pub(crate) const BLOCK_Q5_1: usize = 24; // d + m + qh + qs
pub(crate) const BLOCK_Q8_0: usize = 34; // 2 + 32
pub(crate) const BLOCK_Q2_K: usize = 84;
pub(crate) const BLOCK_Q3_K: usize = 110;
pub(crate) const BLOCK_Q4_K: usize = 144;
pub(crate) const BLOCK_Q5_K: usize = 176;
pub(crate) const BLOCK_Q6_K: usize = 210;
pub(crate) const BLOCK_Q8_K: usize = 292; // f32 d + i8[256] + i16[16] bsums

/// Unpack 6-bit scale/min pair for super-block index `j` in a Q4_K block.
#[inline]
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}
