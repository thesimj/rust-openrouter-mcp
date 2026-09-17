//! Video-generation orchestration over the async OpenRouter video job API.
//!
//! Unlike synchronous image generation, video uses an async
//! job API: submit `POST /api/v1/videos`, poll `GET /api/v1/videos/{id}` until
//! the job completes or fails, then download each clip from the content
//! endpoint. Frame images (first/last) and reference images are reused from the
//! image input pipeline (normalized to PNG data URLs).

use std::path::PathBuf;

use anyhow::{Context, Result, bail, ensure};

use crate::openrouter::ProviderOptions;

mod job;

pub(crate) use job::run_job;

/// Default seconds between poll attempts (env `OPENROUTER_VIDEO_POLL_INTERVAL`).
const DEFAULT_POLL_INTERVAL_SECS: u64 = 5;
/// Default ceiling on the background poll loop (env `OPENROUTER_VIDEO_POLL_TIMEOUT`).
const DEFAULT_POLL_TIMEOUT_SECS: u64 = 600;
/// Local byte cap for one audio/video reference read from disk (same cap as an
/// input image). URLs are not fetched, so they are not measured.
const MAX_MEDIA_REFERENCE_BYTES: usize = 20 * 1024 * 1024;

/// MIME type for a local audio/video reference, from its file extension. Only
/// containers the video providers document are mapped; anything else must be
/// passed as a URL.
fn media_mime_for_extension(ext: &str) -> Option<&'static str> {
    Some(match ext.to_ascii_lowercase().as_str() {
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "flac" => "audio/flac",
        "m4a" => "audio/mp4",
        "ogg" => "audio/ogg",
        "aac" => "audio/aac",
        "weba" => "audio/webm",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        _ => return None,
    })
}

/// Resolve one audio/video reference for `input_references`: an `http(s)://`
/// or `data:` URL passes through untouched (providers fetch it), a local path
/// is read (capped at [`MAX_MEDIA_REFERENCE_BYTES`]) and inlined as a data URL
/// whose MIME comes from the extension. Shared by the CLI and the MCP tool.
pub(crate) async fn resolve_media_reference(source: &str) -> Result<String> {
    let source = source.trim();
    ensure!(!source.is_empty(), "a media reference is blank");
    let lower = source.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("data:") {
        return Ok(source.to_string());
    }
    let path = PathBuf::from(source);
    let ext = path.extension().unwrap_or_default().to_string_lossy();
    let mime = media_mime_for_extension(&ext).with_context(|| {
        format!(
            "cannot tell the media type of {} from its extension; use mp3/wav/flac/m4a/ogg/\
             aac/weba for audio or mp4/webm/mov for video, or pass a URL",
            path.display()
        )
    })?;
    crate::resources::run_blocking(move || {
        let bytes = crate::resources::read_file_limited(&path, MAX_MEDIA_REFERENCE_BYTES)?;
        ensure!(!bytes.is_empty(), "{} is empty", path.display());
        Ok(crate::image_io::data_url(&bytes, mime))
    })
    .await
}

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
    /// Reference audio clips: URLs or local paths (see [`resolve_media_reference`]).
    pub reference_audio: Vec<String>,
    /// Reference video clips: URLs or local paths (see [`resolve_media_reference`]).
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
    /// job, checked here before any HTTP call. The single implementation for
    /// the CLI and the MCP tool.
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

    /// URLs pass through untouched; local files become data URLs typed by
    /// extension; an unknown extension is refused rather than guessed.
    #[tokio::test]
    async fn media_reference_passes_urls_through_and_inlines_local_files() {
        for url in [
            "https://cdn/beat.mp3",
            "http://cdn/clip.mp4",
            "data:audio/wav;base64,AAAA",
        ] {
            assert_eq!(resolve_media_reference(url).await.unwrap(), url);
        }
        // Surrounding whitespace is trimmed, not sent.
        assert_eq!(
            resolve_media_reference("  https://cdn/x.mp3 ")
                .await
                .unwrap(),
            "https://cdn/x.mp3"
        );

        let dir = tempfile::tempdir().unwrap();
        let mp3 = dir.path().join("beat.MP3");
        std::fs::write(&mp3, b"ABC").unwrap();
        assert_eq!(
            resolve_media_reference(&mp3.to_string_lossy())
                .await
                .unwrap(),
            "data:audio/mpeg;base64,QUJD"
        );
        let mov = dir.path().join("ref.mov");
        std::fs::write(&mov, b"ABC").unwrap();
        assert_eq!(
            resolve_media_reference(&mov.to_string_lossy())
                .await
                .unwrap(),
            "data:video/quicktime;base64,QUJD"
        );

        let txt = dir.path().join("notes.txt");
        std::fs::write(&txt, b"ABC").unwrap();
        let err = resolve_media_reference(&txt.to_string_lossy())
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("media type"), "got: {err}");
        assert!(resolve_media_reference("   ").await.is_err());
        let missing = dir.path().join("missing.mp4");
        assert!(
            resolve_media_reference(&missing.to_string_lossy())
                .await
                .is_err()
        );
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
