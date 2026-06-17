//! Video-generation orchestration over the async OpenRouter video job API.
//!
//! Unlike image generation (synchronous chat-completions), video uses an async
//! job API: submit `POST /api/v1/videos`, poll `GET /api/v1/videos/{id}` until
//! the job completes or fails, then download each clip from the content
//! endpoint. Frame images (first/last) and reference images are reused from the
//! image input pipeline (normalized to PNG data URLs).

use std::path::{Path, PathBuf};

use crate::image_gen;

mod job;

pub(crate) use job::run_job;

/// Default seconds between poll attempts (env `OPENROUTER_VIDEO_POLL_INTERVAL`).
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;
/// Default ceiling on the background poll loop (env `OPENROUTER_VIDEO_POLL_TIMEOUT`).
const DEFAULT_POLL_TIMEOUT_SECS: u64 = 600;

/// A local image used as a video frame (first/last). `frame_type` is
/// `first_frame` or `last_frame`.
#[derive(Debug, Clone)]
pub struct VideoInput {
    pub path: PathBuf,
    pub frame_type: String,
}

/// Inputs for a single video generation (domain struct; the wire body is
/// [`openrouter::VideoSubmitBody`]).
#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    pub model: String,
    pub prompt: String,
    pub duration: Option<u32>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    pub size: Option<String>,
    pub generate_audio: Option<bool>,
    pub seed: Option<u64>,
    /// First/last frames for image-to-video. When present, `references` is ignored.
    pub frames: Vec<VideoInput>,
    /// Reference images for reference-to-video.
    pub references: Vec<PathBuf>,
    pub max_image_dimension: u32,
    pub poll_interval_secs: u64,
    pub poll_timeout_secs: u64,
}

/// Resolve a poll setting (seconds): explicit value, else `env_key`, else
/// `default`; floored at 1 so a zero never busy-loops.
fn resolve_secs(explicit: Option<u64>, env_key: &str, default: u64) -> u64 {
    explicit
        .or_else(|| std::env::var(env_key).ok().and_then(|v| v.parse().ok()))
        .unwrap_or(default)
        .max(1)
}

/// Resolve the poll interval: explicit value, else `OPENROUTER_VIDEO_POLL_INTERVAL`,
/// else [`DEFAULT_POLL_INTERVAL_SECS`].
pub fn resolve_poll_interval(explicit: Option<u64>) -> u64 {
    resolve_secs(
        explicit,
        "OPENROUTER_VIDEO_POLL_INTERVAL",
        DEFAULT_POLL_INTERVAL_SECS,
    )
}

/// Resolve the poll timeout: explicit value, else `OPENROUTER_VIDEO_POLL_TIMEOUT`,
/// else [`DEFAULT_POLL_TIMEOUT_SECS`].
pub fn resolve_poll_timeout(explicit: Option<u64>) -> u64 {
    resolve_secs(
        explicit,
        "OPENROUTER_VIDEO_POLL_TIMEOUT",
        DEFAULT_POLL_TIMEOUT_SECS,
    )
}

/// One saved clip in a job's lean summary.
pub struct VideoSummary {
    pub path: PathBuf,
    pub duration: Option<u32>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    pub has_audio: bool,
    pub mime: String,
    pub cost: Option<f64>,
    /// OpenRouter generation id, recorded in the manifest.
    #[allow(dead_code)]
    pub generation_id: Option<String>,
}

/// Result of a full video job: the saved clips, the manifest path, plus warnings
/// and errors.
pub struct VideoJobSummary {
    pub model: String,
    pub manifest_path: PathBuf,
    pub videos: Vec<VideoSummary>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Sidecar manifest path next to the outputs: `<stem>.manifest.json`.
pub fn manifest_path(base: &Path) -> PathBuf {
    image_gen::manifest_path(base)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_poll_interval_and_timeout_default_and_floor_at_one() {
        assert_eq!(resolve_poll_interval(Some(9)), 9);
        assert_eq!(resolve_poll_interval(Some(0)), 1, "floors at 1");
        assert_eq!(resolve_poll_timeout(Some(120)), 120);
        assert_eq!(resolve_poll_timeout(Some(0)), 1, "floors at 1");
    }
}
