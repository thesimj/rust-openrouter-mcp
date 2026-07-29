//! Video-generation orchestration over the async OpenRouter video job API.
//!
//! Unlike image generation (synchronous chat-completions), video uses an async
//! job API: submit `POST /api/v1/videos`, poll `GET /api/v1/videos/{id}` until
//! the job completes or fails, then download each clip from the content
//! endpoint. Frame images (first/last) and reference images are reused from the
//! image input pipeline (normalized to PNG data URLs).

use std::path::PathBuf;

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

/// Parse a poll setting (seconds) from a raw env value, falling back to
/// `default`; floored at 1 so a zero never busy-loops.
fn parse_secs(raw: Option<&str>, default: u64) -> u64 {
    raw.and_then(|v| v.parse().ok()).unwrap_or(default).max(1)
}

/// Poll interval: `OPENROUTER_VIDEO_POLL_INTERVAL`, else [`DEFAULT_POLL_INTERVAL_SECS`].
pub fn resolve_poll_interval() -> u64 {
    let raw = std::env::var("OPENROUTER_VIDEO_POLL_INTERVAL").ok();
    parse_secs(raw.as_deref(), DEFAULT_POLL_INTERVAL_SECS)
}

/// Poll timeout: `OPENROUTER_VIDEO_POLL_TIMEOUT`, else [`DEFAULT_POLL_TIMEOUT_SECS`].
pub fn resolve_poll_timeout() -> u64 {
    let raw = std::env::var("OPENROUTER_VIDEO_POLL_TIMEOUT").ok();
    parse_secs(raw.as_deref(), DEFAULT_POLL_TIMEOUT_SECS)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_secs_defaults_and_floors_at_one() {
        assert_eq!(parse_secs(None, 5), 5, "unset -> default");
        assert_eq!(parse_secs(Some("9"), 5), 9);
        assert_eq!(parse_secs(Some("0"), 5), 1, "floors at 1, never busy-loops");
        assert_eq!(parse_secs(Some("nope"), 5), 5, "garbage -> default");
    }
}
