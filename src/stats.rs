//! In-memory usage statistics for the server process.
//!
//! Updated whenever a generation job finishes, and exposed via `get_usage_stats`
//! (read-only) / `reset_usage_stats` (destructive). Per process, not persisted -
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
    videos_generated: u64,
    audio_files: u64,
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
    video_generations: u64,
    videos_generated: u64,
    audio_generations: u64,
    audio_files: u64,
    actual_cost_usd: f64,
    unknown_cost_count: u64,
    by_model: BTreeMap<String, ModelStats>,
}

impl Default for Inner {
    fn default() -> Self {
        Self {
            started_at: Utc::now(),
            requests_total: 0,
            requests_failed: 0,
            image_generations: 0,
            images_generated: 0,
            text_generations: 0,
            video_generations: 0,
            videos_generated: 0,
            audio_generations: 0,
            audio_files: 0,
            actual_cost_usd: 0.0,
            unknown_cost_count: 0,
            by_model: BTreeMap::new(),
        }
    }
}

impl Inner {
    /// Add a reported (or unreported) cost to both the global and a per-model
    /// counter: known costs accumulate in `actual_cost_usd`, unknown ones bump
    /// `unknown_cost_count`. Returns the per-model entry for further updates.
    fn account_cost(&mut self, model: &str, cost: Option<f64>) -> &mut ModelStats {
        match cost {
            Some(c) => self.actual_cost_usd += c,
            None => self.unknown_cost_count += 1,
        }
        let m = self.by_model.entry(model.to_string()).or_default();
        match cost {
            Some(c) => m.actual_cost_usd += c,
            None => m.unknown_cost_count += 1,
        }
        m
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
            inner: Arc::new(Mutex::new(Inner::default())),
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
        s.account_cost(model, cost).requests += 1;
    }

    /// Record one video request and its saved clip count. An accepted job keeps
    /// its receipt even if delivery fails. Rejected submissions have no receipt.
    pub async fn record_video(
        &self,
        model: &str,
        clips: u64,
        receipt: Option<&crate::billing::Receipt>,
    ) {
        let mut s = self.inner.lock().await;
        s.requests_total += 1;
        s.video_generations += 1;
        if clips == 0 {
            s.requests_failed += 1;
        }
        s.videos_generated += clips;
        let m = match receipt {
            Some(receipt) => s.account_cost(model, receipt.cost),
            None => s.by_model.entry(model.to_string()).or_default(),
        };
        m.requests += 1;
        m.videos_generated += clips;
    }

    /// Record one finished text-to-speech request. `cost` is typically `None`
    /// (the speech endpoint returns no inline usage.cost), so it lands in
    /// `unknown_cost_count`.
    pub async fn record_audio(&self, model: &str, success: bool, cost: Option<f64>) {
        let mut s = self.inner.lock().await;
        s.requests_total += 1;
        s.audio_generations += 1;
        if !success {
            s.requests_failed += 1;
            s.by_model.entry(model.to_string()).or_default().requests += 1;
            return;
        }
        s.audio_files += 1;
        let m = s.account_cost(model, cost);
        m.requests += 1;
        m.audio_files += 1;
    }

    /// Account the receipt carried by a failed request, if the provider had
    /// already answered (and so billed) before the local failure. A receipt
    /// without usage counts as an unknown cost, not as free.
    pub async fn record_failed_receipt(&self, model: &str, error: &anyhow::Error) {
        if let Some(receipt) = crate::billing::Receipt::from_error(error) {
            self.inner.lock().await.account_cost(model, receipt.cost);
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
                        "videos_generated": v.videos_generated,
                        "audio_files": v.audio_files,
                        "actual_cost_usd": round4(v.actual_cost_usd),
                        "unknown_cost_count": v.unknown_cost_count,
                    }),
                )
            })
            .collect();
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "started_at": s.started_at.to_rfc3339(),
            "uptime_seconds": uptime,
            "requests_total": s.requests_total,
            "requests_failed": s.requests_failed,
            "image_generations": s.image_generations,
            "images_generated": s.images_generated,
            "text_generations": s.text_generations,
            "video_generations": s.video_generations,
            "videos_generated": s.videos_generated,
            "audio_generations": s.audio_generations,
            "audio_files": s.audio_files,
            "actual_cost_usd": round4(s.actual_cost_usd),
            "unknown_cost_count": s.unknown_cost_count,
            "by_model": by_model,
        })
    }

    /// Reset all counters (and the start time).
    pub async fn reset(&self) {
        *self.inner.lock().await = Inner::default();
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
        assert_eq!(s["version"], env!("CARGO_PKG_VERSION"));
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

#[cfg(test)]
mod audit_regression {
    use super::*;
    use crate::billing::Receipt;
    #[tokio::test]
    async fn video_billing_is_per_request_and_survives_failed_delivery() {
        let stats = UsageStats::new();
        stats
            .record_video(
                "video",
                2,
                Some(&Receipt {
                    cost: Some(0.9),
                    generation_id: None,
                }),
            )
            .await;
        stats
            .record_video(
                "video",
                0,
                Some(&Receipt {
                    cost: Some(0.4),
                    generation_id: None,
                }),
            )
            .await;
        stats
            .record_video("video", 0, Some(&Receipt::default()))
            .await;
        let output = stats.snapshot().await;
        assert_eq!(output["requests_total"], 3);
        assert_eq!(output["requests_failed"], 2);
        assert_eq!(output["videos_generated"], 2);
        assert_eq!(output["actual_cost_usd"], 1.3);
        assert_eq!(output["unknown_cost_count"], 1);
        assert_eq!(output["by_model"]["video"]["actual_cost_usd"], 1.3);
    }
    #[tokio::test]
    async fn failed_text_extraction_keeps_receipt_without_double_counting_requests() {
        let stats = UsageStats::new();
        let error = anyhow::anyhow!("empty answer").context(Receipt {
            cost: Some(0.02),
            generation_id: None,
        });
        stats.record_text("chat", false, None).await;
        stats.record_failed_receipt("chat", &error).await;
        let output = stats.snapshot().await;
        assert_eq!(output["requests_total"], 1);
        assert_eq!(output["requests_failed"], 1);
        assert_eq!(output["actual_cost_usd"], 0.02);
    }
}
