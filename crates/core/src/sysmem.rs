//! Host memory detection and context-size budgeting for ~16GB-class machines.

/// Snapshot of physical RAM available to the process (best-effort).
#[derive(Debug, Clone, Copy)]
pub struct SystemMemory {
    /// Total physical RAM in bytes (0 if unknown).
    pub total_bytes: u64,
    /// Approximate available RAM in bytes (0 if unknown).
    pub available_bytes: u64,
}

impl SystemMemory {
    /// Linux: `/proc/meminfo`. Windows: `GlobalMemoryStatusEx` (total and available).
    /// macOS: `sysctl hw.memsize` (total only). Anything else, or a failed read, returns
    /// zeros and the caller falls back (no RAM budget; the context is capped by `--ctx`).
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        let found = detect_linux();
        #[cfg(windows)]
        let found = detect_windows();
        #[cfg(target_os = "macos")]
        let found = detect_macos();
        #[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
        let found: Option<SystemMemory> = None;
        found.unwrap_or(Self {
            total_bytes: 0,
            available_bytes: 0,
        })
    }

    /// Prefer available; if missing, use a fraction of total; if unknown, `None`.
    pub fn budget_bytes(&self, fraction: f32) -> Option<u64> {
        let frac = fraction.clamp(0.05, 0.95);
        if self.available_bytes > 0 {
            Some((self.available_bytes as f64 * frac as f64) as u64)
        } else if self.total_bytes > 0 {
            Some((self.total_bytes as f64 * frac as f64) as u64)
        } else {
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn detect_linux() -> Option<SystemMemory> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    let mut free_kb = 0u64;
    let mut buffers_kb = 0u64;
    let mut cached_kb = 0u64;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let key = parts.next()?;
        let val: u64 = parts.next()?.parse().ok()?;
        match key {
            "MemTotal:" => total_kb = val,
            "MemAvailable:" => avail_kb = val,
            "MemFree:" => free_kb = val,
            "Buffers:" => buffers_kb = val,
            "Cached:" => cached_kb = val,
            _ => {}
        }
    }
    if avail_kb == 0 {
        avail_kb = free_kb.saturating_add(buffers_kb).saturating_add(cached_kb);
    }
    Some(SystemMemory {
        total_bytes: total_kb.saturating_mul(1024),
        available_bytes: avail_kb.saturating_mul(1024),
    })
}

#[cfg(windows)]
fn detect_windows() -> Option<SystemMemory> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: `status` is a writable MEMORYSTATUSEX with `dwLength` set, as the call requires.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 || status.ullTotalPhys == 0 {
        return None;
    }
    Some(SystemMemory {
        total_bytes: status.ullTotalPhys,
        available_bytes: status.ullAvailPhys,
    })
}

#[cfg(target_os = "macos")]
fn detect_macos() -> Option<SystemMemory> {
    let mut total: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: the name is NUL-terminated; `total`/`len` describe a writable u64; nothing is set.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == std::mem::size_of::<u64>() && total > 0).then_some(SystemMemory {
        total_bytes: total,
        available_bytes: 0, // total only: the budget uses a fraction of it
    })
}

/// Bytes for one position of K+V across all layers (`elem_size` is 4 for f32, 2 for f16).
pub fn kv_bytes_for_ctx(n_layer: u32, n_embd_kv: u32, n_ctx: u32, elem_size: usize) -> u64 {
    let per_pos = (n_layer as u64)
        .saturating_mul(n_embd_kv as u64)
        .saturating_mul(2) // K and V
        .saturating_mul(elem_size as u64);
    per_pos.saturating_mul(n_ctx as u64)
}

/// Choose an effective context length that fits in `budget_bytes` for the KV cache.
///
/// Keeps at least `min_ctx` and never exceeds `requested_ctx` or `hard_cap`.
pub fn clamp_context_len(
    requested_ctx: u32,
    n_layer: u32,
    n_embd_kv: u32,
    budget_bytes: u64,
    elem_size: usize,
    min_ctx: u32,
    hard_cap: u32,
) -> u32 {
    let req = requested_ctx.min(hard_cap).max(min_ctx);
    let need = kv_bytes_for_ctx(n_layer, n_embd_kv, req, elem_size);
    if need <= budget_bytes || budget_bytes == 0 {
        return req;
    }
    let per_pos = kv_bytes_for_ctx(n_layer, n_embd_kv, 1, elem_size).max(1);
    let fit = (budget_bytes / per_pos) as u32;
    fit.clamp(min_ctx, req)
}

/// Current process peak RSS in bytes (Linux `VmHWM`), if available.
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Current process RSS in bytes (Linux `VmRSS`), if available.
pub fn current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_reduces_huge_ctx() {
        // 80 layers, 1024 kv embd, f32 → ~640 KiB per position
        let n_layer = 80;
        let n_embd_kv = 1024;
        let budget = 512 * 1024 * 1024; // 512 MiB for KV
        let capped = clamp_context_len(1_000_000, n_layer, n_embd_kv, budget, 4, 512, 128_000);
        assert!(capped < 1_000_000);
        assert!(capped >= 512);
        let used = kv_bytes_for_ctx(n_layer, n_embd_kv, capped, 4);
        assert!(used <= budget);
    }

    #[test]
    fn unknown_ram_has_no_budget() {
        let unknown = SystemMemory {
            total_bytes: 0,
            available_bytes: 0,
        };
        assert_eq!(unknown.budget_bytes(0.45), None);
        let total_only = SystemMemory {
            total_bytes: 16 << 30,
            available_bytes: 0,
        };
        assert_eq!(total_only.budget_bytes(0.5), Some(8 << 30));
    }

    #[cfg(any(target_os = "linux", windows, target_os = "macos"))]
    #[test]
    fn detect_reads_host_ram() {
        let m = SystemMemory::detect();
        assert!(m.total_bytes >= 256 << 20, "{m:?}");
        assert!(m.available_bytes <= m.total_bytes, "{m:?}");
        assert!(m.budget_bytes(0.45).is_some());
    }

    #[test]
    fn clamp_keeps_small_ctx() {
        let c = clamp_context_len(2048, 22, 512, 8 * 1024 * 1024 * 1024, 4, 512, 128_000);
        assert_eq!(c, 2048);
    }
}
