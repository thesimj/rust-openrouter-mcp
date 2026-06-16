//! In-memory usage statistics for the server process.
//!
//! Updated whenever a generation job finishes, and exposed via `get_usage_stats`
//! (read-only) / `reset_usage_stats` (destructive). Per process, not persisted —
//! a stdio MCP server is normally one process per client session.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use tokio::sync::Mutex;

#[derive(Default)]
struct ModelStats {
    requests: u64,
    images_generated: u64,
    actual_cost_usd: f64,
    unknown_cost_count: u64,
}

struct Inner {
    started_at: DateTime<Utc>,
    requests_total: u64,
    requests_failed: u64,
    image_generations: u64,
    images_generated: u64,
    text_generations: u64,
    actual_cost_usd: f64,
    unknown_cost_count: u64,
    by_model: BTreeMap<String, ModelStats>,
}

impl Inner {
    fn new() -> Self {
        Self {
            started_at: Utc::now(),
            requests_total: 0,
            requests_failed: 0,
            image_generations: 0,
            images_generated: 0,
            text_generations: 0,
            actual_cost_usd: 0.0,
            unknown_cost_count: 0,
            by_model: BTreeMap::new(),
        }
    }
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Process-local usage counters, cheaply cloneable (shared `Arc`).
#[derive(Clone)]
pub struct UsageStats {
    inner: Arc<Mutex<Inner>>,
}

impl Default for UsageStats {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageStats {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::new())),
        }
    }

    /// Record one finished job: `variants` image requests, of which `images`
    /// succeeded; `cost` is the summed USD `usage.cost` and `unknown_cost` is the
    /// number of successful images whose cost was not reported.
    pub async fn record_job(
        &self,
        model: &str,
        variants: u64,
        images: u64,
        cost: f64,
        unknown_cost: u64,
    ) {
        let mut s = self.inner.lock().await;
        s.requests_total += variants;
        s.requests_failed += variants.saturating_sub(images);
        s.image_generations += variants;
        s.images_generated += images;
        s.actual_cost_usd += cost;
        s.unknown_cost_count += unknown_cost;
        let m = s.by_model.entry(model.to_string()).or_default();
        m.requests += variants;
        m.images_generated += images;
        m.actual_cost_usd += cost;
        m.unknown_cost_count += unknown_cost;
    }

    /// Record one text/vision request (e.g. describe_image). `cost` is the
    /// reported USD `usage.cost`, if any; a failed request only bumps the
    /// failure counters.
    pub async fn record_text(&self, model: &str, success: bool, cost: Option<f64>) {
        let mut s = self.inner.lock().await;
        s.requests_total += 1;
        if !success {
            s.requests_failed += 1;
            s.by_model.entry(model.to_string()).or_default().requests += 1;
            return;
        }
        s.text_generations += 1;
        match cost {
            Some(c) => s.actual_cost_usd += c,
            None => s.unknown_cost_count += 1,
        }
        let m = s.by_model.entry(model.to_string()).or_default();
        m.requests += 1;
        match cost {
            Some(c) => m.actual_cost_usd += c,
            None => m.unknown_cost_count += 1,
        }
    }

    /// A JSON snapshot of the current counters.
    pub async fn snapshot(&self) -> Value {
        let s = self.inner.lock().await;
        let uptime = (Utc::now() - s.started_at).num_seconds().max(0);
        let by_model: serde_json::Map<String, Value> = s
            .by_model
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    json!({
                        "requests": v.requests,
                        "images_generated": v.images_generated,
                        "actual_cost_usd": round4(v.actual_cost_usd),
                        "unknown_cost_count": v.unknown_cost_count,
                    }),
                )
            })
            .collect();
        json!({
            "started_at": s.started_at.to_rfc3339(),
            "uptime_seconds": uptime,
            "requests_total": s.requests_total,
            "requests_failed": s.requests_failed,
            "image_generations": s.image_generations,
            "images_generated": s.images_generated,
            "text_generations": s.text_generations,
            "actual_cost_usd": round4(s.actual_cost_usd),
            "unknown_cost_count": s.unknown_cost_count,
            "by_model": by_model,
        })
    }

    /// Reset all counters (and the start time).
    pub async fn reset(&self) {
        *self.inner.lock().await = Inner::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn record_and_snapshot_aggregates_by_model() {
        let stats = UsageStats::new();
        // A 4-variant job: 3 succeeded (one without cost), 1 failed.
        stats.record_job("model-a", 4, 3, 0.20, 1).await;
        stats.record_job("model-b", 1, 1, 0.04, 0).await;

        let s = stats.snapshot().await;
        assert_eq!(s["requests_total"], 5);
        assert_eq!(s["requests_failed"], 1);
        assert_eq!(s["images_generated"], 4);
        assert_eq!(s["unknown_cost_count"], 1);
        assert_eq!(s["actual_cost_usd"], 0.24);
        assert_eq!(s["by_model"]["model-a"]["images_generated"], 3);
        assert_eq!(s["by_model"]["model-b"]["actual_cost_usd"], 0.04);
    }

    #[tokio::test]
    async fn record_text_tracks_describe_calls() {
        let stats = UsageStats::new();
        stats.record_text("vision-a", true, Some(0.002)).await;
        stats.record_text("vision-a", true, None).await; // success, cost unknown
        stats.record_text("vision-b", false, None).await; // failed

        let s = stats.snapshot().await;
        assert_eq!(s["requests_total"], 3);
        assert_eq!(s["requests_failed"], 1);
        assert_eq!(s["text_generations"], 2);
        assert_eq!(s["unknown_cost_count"], 1);
        assert_eq!(s["actual_cost_usd"], 0.002);
        assert_eq!(s["by_model"]["vision-a"]["requests"], 2);
    }

    #[tokio::test]
    async fn reset_clears_counters() {
        let stats = UsageStats::new();
        stats.record_job("m", 2, 2, 0.1, 0).await;
        stats.reset().await;
        let s = stats.snapshot().await;
        assert_eq!(s["requests_total"], 0);
        assert_eq!(s["actual_cost_usd"], 0.0);
    }
}
