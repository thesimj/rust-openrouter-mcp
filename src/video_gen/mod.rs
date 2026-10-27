//! Video-generation orchestration over the async OpenRouter video job API.
//!
//! Unlike synchronous image generation, video uses an async
//! job API: submit `POST /api/v1/videos`, poll `GET /api/v1/videos/{id}` until
//! the job completes or fails, then download each clip from the content
//! endpoint. Frame images (first/last) and reference images are reused from the
//! image input pipeline (normalized to data URLs: JPEG, or PNG when they have
//! alpha).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Result, bail};

use crate::openrouter::ProviderOptions;

mod job;

pub(crate) use job::run_job;

/// Default time between poll attempts (env `OPENROUTER_VIDEO_POLL_INTERVAL`, seconds).
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(5);
/// Default ceiling on the background poll loop (env `OPENROUTER_VIDEO_POLL_TIMEOUT`, seconds).
const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_secs(600);
/// A local image used as a video frame (first/last).
#[derive(Debug, Clone)]
pub struct VideoInput {
    pub path: PathBuf,
    pub frame_type: crate::openrouter::FrameType,
}

/// An audio or video reference as the caller named it (`source`, recorded in
/// the manifest; a `data:` URL is recorded by its media type only, see
/// `server::media::reference_label`) and as it goes on the wire (`url`: the URL itself, or a local
/// file inlined as a data URL). The tool layer resolves it, with the same caps
/// and MIME rules as the chat inputs.
#[derive(Debug, Clone)]
pub struct MediaReference {
    pub source: String,
    pub url: String,
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
    /// Reference audio clips, already resolved.
    pub reference_audio: Vec<MediaReference>,
    /// Reference video clips, already resolved.
    pub reference_videos: Vec<MediaReference>,
    /// Upscaling models only.
    pub creativity: Option<u32>,
    /// Upscaling models only; must be > 0.
    pub upscale_factor: Option<f64>,
    /// Per-provider passthrough (`provider.options.<slug>`), already validated.
    pub provider: Option<ProviderOptions>,
    pub max_image_dimension: u32,
    /// Time between status polls; always >= 1 s (from
    /// [`resolve_poll_interval`], the only production source), so the job
    /// applies no floor of its own.
    pub poll_interval: Duration,
    /// Time to wait for the job before giving up; at most [`MAX_TIMER`] (from
    /// [`resolve_poll_timeout`]).
    pub poll_timeout: Duration,
}

impl VideoGenRequest {
    /// Whether any reference of any kind (image, audio, video) is present.
    /// Frames win over references on the wire, so the job warns when both are.
    pub fn has_references(&self) -> bool {
        !self.references.is_empty()
            || !self.reference_audio.is_empty()
            || !self.reference_videos.is_empty()
    }

    /// Whether any frame or reference of any kind is present - the condition
    /// under which `prompt` may be omitted.
    pub fn has_visual_or_media_input(&self) -> bool {
        !self.frames.is_empty() || self.has_references()
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

/// Longest timer budget honored for the operator-supplied poll and delivery
/// timeouts: an unbounded value would overflow `Instant + Duration`.
const MAX_TIMER: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// A poll setting given in seconds, else `default`. Floored at 1 s so a zero
/// never busy-loops (this is the only floor: the job trusts the interval it is
/// given), and capped at [`MAX_TIMER`].
fn poll_setting(seconds: Option<u64>, default: Duration) -> Duration {
    seconds
        .map_or(default, Duration::from_secs)
        .clamp(Duration::from_secs(1), MAX_TIMER)
}

/// Poll interval: `OPENROUTER_VIDEO_POLL_INTERVAL`, else [`DEFAULT_POLL_INTERVAL`].
pub fn resolve_poll_interval() -> Duration {
    poll_setting(
        crate::settings::env_number("OPENROUTER_VIDEO_POLL_INTERVAL"),
        DEFAULT_POLL_INTERVAL,
    )
}

/// Poll timeout: `OPENROUTER_VIDEO_POLL_TIMEOUT`, else [`DEFAULT_POLL_TIMEOUT`].
pub fn resolve_poll_timeout() -> Duration {
    poll_setting(
        crate::settings::env_number("OPENROUTER_VIDEO_POLL_TIMEOUT"),
        DEFAULT_POLL_TIMEOUT,
    )
}

/// Optional wall-clock budget for all clip downloads and retry waits, given in
/// seconds. Zero or unset leaves progressing downloads unrestricted. An enabled
/// budget is capped at [`MAX_TIMER`].
fn delivery_timeout(seconds: Option<u64>) -> Option<Duration> {
    seconds
        .filter(|seconds| *seconds > 0)
        .map(|seconds| Duration::from_secs(seconds).min(MAX_TIMER))
}

fn resolve_delivery_timeout() -> Option<Duration> {
    delivery_timeout(crate::settings::env_number(
        "OPENROUTER_VIDEO_DELIVERY_TIMEOUT",
    ))
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
        assert_eq!(delivery_timeout(None), None);
        assert_eq!(delivery_timeout(Some(0)), None);
        assert_eq!(delivery_timeout(Some(12)), Some(Duration::from_secs(12)));
        assert_eq!(delivery_timeout(Some(u64::MAX)), Some(MAX_TIMER));
    }

    #[test]
    fn poll_setting_defaults_floors_at_one_second_and_caps() {
        let default = Duration::from_secs(5);
        assert_eq!(poll_setting(None, default), default, "unset -> default");
        assert_eq!(poll_setting(Some(9), default), Duration::from_secs(9));
        assert_eq!(
            poll_setting(Some(0), default),
            Duration::from_secs(1),
            "floors at 1 s, never busy-loops"
        );
        assert_eq!(poll_setting(Some(u64::MAX), default), MAX_TIMER);
    }

    /// A text-to-video request for model "m" with prompt "p" and every other
    /// input unset; the job tests build their fixtures on it too.
    pub(super) fn request() -> VideoGenRequest {
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
            poll_interval: Duration::from_secs(1),
            poll_timeout: Duration::from_secs(1),
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
                frame_type: crate::openrouter::FrameType::FirstFrame,
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
                reference_audio: vec![MediaReference {
                    source: "a.mp3".into(),
                    url: "https://cdn/a.mp3".into(),
                }],
                ..request()
            },
            VideoGenRequest {
                prompt: None,
                reference_videos: vec![MediaReference {
                    source: "v.mp4".into(),
                    url: "https://cdn/v.mp4".into(),
                }],
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
