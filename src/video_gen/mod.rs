//! Video-generation orchestration over the async OpenRouter video job API.
//!
//! Unlike synchronous image generation, video uses an async
//! job API: submit `POST /api/v1/videos`, poll `GET /api/v1/videos/{id}` until
//! the job completes or fails, then download each clip from the content
//! endpoint. Frame images (first/last) and reference images are reused from the
//! image input pipeline (normalized to PNG data URLs).

use std::path::PathBuf;

use anyhow::{Result, bail};

use crate::openrouter::ProviderOptions;

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
/// [`crate::openrouter::VideoSubmitBody`]).
#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    pub model: String,
    /// Required unless a frame or any reference is present (image-only models
    /// take no text); see [`Self::validate`].
    pub prompt: Option<String>,
    pub duration: Option<u32>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    pub size: Option<String>,
    pub generate_audio: Option<bool>,
    pub seed: Option<u64>,
    /// First/last frames for image-to-video. When present, every reference
    /// kind (`references`, `reference_audio`, `reference_videos`) is ignored.
    pub frames: Vec<VideoInput>,
    /// Reference images for reference-to-video.
    pub references: Vec<PathBuf>,
    /// Reference audio clips: URLs or local paths, resolved at submit time by
    /// [`crate::server::media::resolve_media_reference`].
    pub reference_audio: Vec<String>,
    /// Reference video clips: URLs or local paths, resolved like `reference_audio`.
    pub reference_videos: Vec<String>,
    /// Upscaling models only.
    pub creativity: Option<u32>,
    /// Upscaling models only; must be > 0.
    pub upscale_factor: Option<f64>,
    /// Per-provider passthrough (`provider.options.<slug>`), already validated.
    pub provider: Option<ProviderOptions>,
    pub max_image_dimension: u32,
    pub poll_interval_secs: u64,
    pub poll_timeout_secs: u64,
}

impl VideoGenRequest {
    /// Whether any frame or reference of any kind is present - the condition
    /// under which `prompt` may be omitted.
    pub fn has_visual_or_media_input(&self) -> bool {
        !self.frames.is_empty()
            || !self.references.is_empty()
            || !self.reference_audio.is_empty()
            || !self.reference_videos.is_empty()
    }

    /// The trimmed prompt, or `None` when absent or blank.
    pub fn prompt_text(&self) -> Option<&str> {
        self.prompt
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty())
    }

    /// Invariants the endpoint enforces only after accepting (and billing) the
    /// job, checked here before any HTTP call.
    pub fn validate(&self) -> Result<()> {
        if self.prompt_text().is_none() && !self.has_visual_or_media_input() {
            bail!(
                "prompt is required unless a first_frame/last_frame or a reference \
                 (image, audio or video) is given"
            );
        }
        if let Some(factor) = self.upscale_factor
            && (factor.is_nan() || factor <= 0.0)
        {
            bail!("upscale_factor must be > 0 (got {factor})");
        }
        Ok(())
    }
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

/// Optional wall-clock budget for all clip downloads and retry waits.
/// Zero, unset, or invalid values leave progressing downloads unrestricted.
/// Cap enabled budgets at 30 days to keep timer arithmetic representable.
fn parse_delivery_timeout(raw: Option<&str>) -> Option<std::time::Duration> {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(|seconds| std::time::Duration::from_secs(seconds.min(30 * 24 * 60 * 60)))
}

pub(super) fn resolve_delivery_timeout() -> Option<std::time::Duration> {
    let raw = std::env::var("OPENROUTER_VIDEO_DELIVERY_TIMEOUT").ok();
    parse_delivery_timeout(raw.as_deref())
}

/// One saved clip in a job's lean summary.
pub struct VideoSummary {
    pub path: PathBuf,
    pub duration: Option<u32>,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
    pub has_audio: bool,
    pub mime: String,
}

/// Result of a full video job: the saved clips, the manifest path, plus warnings
/// and errors.
pub struct VideoJobSummary {
    pub job_id: String,
    /// The accepted job's receipt; cost is `None` until usage is reported.
    pub billing: crate::billing::Receipt,
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
    fn delivery_timeout_is_optional_and_bounded() {
        assert_eq!(parse_delivery_timeout(None), None);
        assert_eq!(parse_delivery_timeout(Some("0")), None);
        assert_eq!(parse_delivery_timeout(Some("bad")), None);
        assert_eq!(
            parse_delivery_timeout(Some(" 12 ")),
            Some(std::time::Duration::from_secs(12))
        );
        assert_eq!(
            parse_delivery_timeout(Some(&u64::MAX.to_string())),
            Some(std::time::Duration::from_secs(30 * 24 * 60 * 60))
        );
    }

    #[test]
    fn parse_secs_defaults_and_floors_at_one() {
        assert_eq!(parse_secs(None, 5), 5, "unset -> default");
        assert_eq!(parse_secs(Some("9"), 5), 9);
        assert_eq!(parse_secs(Some("0"), 5), 1, "floors at 1, never busy-loops");
        assert_eq!(parse_secs(Some("nope"), 5), 5, "garbage -> default");
    }

    fn request() -> VideoGenRequest {
        VideoGenRequest {
            model: "m".into(),
            prompt: Some("p".into()),
            duration: None,
            resolution: None,
            aspect_ratio: None,
            size: None,
            generate_audio: None,
            seed: None,
            frames: vec![],
            references: vec![],
            reference_audio: vec![],
            reference_videos: vec![],
            creativity: None,
            upscale_factor: None,
            provider: None,
            max_image_dimension: 800,
            poll_interval_secs: 1,
            poll_timeout_secs: 1,
        }
    }

    /// A prompt may be omitted only when some frame or reference stands in for
    /// it; a blank prompt counts as omitted. upscale_factor must be > 0.
    #[test]
    fn validate_enforces_prompt_or_input_and_a_positive_upscale_factor() {
        assert!(request().validate().is_ok());
        let none = VideoGenRequest {
            prompt: None,
            ..request()
        };
        assert!(none.validate().unwrap_err().to_string().contains("prompt"));
        let blank = VideoGenRequest {
            prompt: Some("  ".into()),
            ..request()
        };
        assert!(blank.validate().is_err());

        let with_frame = VideoGenRequest {
            prompt: None,
            frames: vec![VideoInput {
                path: "f.png".into(),
                frame_type: "first_frame".into(),
            }],
            ..request()
        };
        assert!(with_frame.validate().is_ok());
        for refs in [
            VideoGenRequest {
                prompt: None,
                references: vec!["r.png".into()],
                ..request()
            },
            VideoGenRequest {
                prompt: None,
                reference_audio: vec!["a.mp3".into()],
                ..request()
            },
            VideoGenRequest {
                prompt: None,
                reference_videos: vec!["v.mp4".into()],
                ..request()
            },
        ] {
            assert!(refs.validate().is_ok());
        }

        for bad in [0.0, -1.0, f64::NAN] {
            let req = VideoGenRequest {
                upscale_factor: Some(bad),
                ..request()
            };
            let err = req.validate().unwrap_err().to_string();
            assert!(err.contains("upscale_factor"), "got: {err}");
        }
        let ok = VideoGenRequest {
            upscale_factor: Some(0.5),
            ..request()
        };
        assert!(ok.validate().is_ok());
    }
}
