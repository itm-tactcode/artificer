//! Lightweight server metrics for concurrent load / usage accounting.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Process-wide counters (shared across requests).
#[derive(Debug, Default)]
pub struct ServerMetrics {
    pub requests_started: AtomicU64,
    pub requests_completed: AtomicU64,
    pub requests_failed: AtomicU64,
    pub prompt_tokens_total: AtomicU64,
    pub completion_tokens_total: AtomicU64,
    /// Callers currently waiting on or holding the engine mutex.
    pub in_flight: AtomicUsize,
    pub peak_in_flight: AtomicUsize,
}

impl ServerMetrics {
    pub fn begin_request(&self) -> RequestGuard<'_> {
        self.requests_started.fetch_add(1, Ordering::Relaxed);
        let n = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        // Update peak.
        let mut peak = self.peak_in_flight.load(Ordering::Relaxed);
        while n > peak {
            match self.peak_in_flight.compare_exchange_weak(
                peak,
                n,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
        RequestGuard { metrics: self }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            requests_started: self.requests_started.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_failed: self.requests_failed.load(Ordering::Relaxed),
            prompt_tokens_total: self.prompt_tokens_total.load(Ordering::Relaxed),
            completion_tokens_total: self.completion_tokens_total.load(Ordering::Relaxed),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            peak_in_flight: self.peak_in_flight.load(Ordering::Relaxed),
        }
    }
}

pub struct RequestGuard<'a> {
    metrics: &'a ServerMetrics,
}

impl RequestGuard<'_> {
    pub fn success(self, prompt_tokens: u64, completion_tokens: u64) {
        self.metrics
            .requests_completed
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .prompt_tokens_total
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.metrics
            .completion_tokens_total
            .fetch_add(completion_tokens, Ordering::Relaxed);
        // Drop decrements in_flight.
    }

    pub fn fail(self) {
        self.metrics.requests_failed.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for RequestGuard<'_> {
    fn drop(&mut self) {
        self.metrics.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub requests_started: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    pub in_flight: usize,
    pub peak_in_flight: usize,
}
