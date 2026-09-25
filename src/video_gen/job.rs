//! Video job orchestration: submit, poll, download each clip, write the sidecar
//! manifest, and return a lean summary.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::image_gen::{self, InputImage};
use crate::manifest::{self, FrameImageMeta, VideoClipMeta, VideoManifest};
use crate::openrouter::{FrameImage, ImageUrl, InputReference, OpenRouterClient, VideoSubmitBody};

use super::{VideoGenRequest, VideoJobSummary, VideoSummary};

/// File extension for a video/audio MIME type. Falls back to `mp4`.
fn extension_for(mime: &str) -> &'static str {
    match mime {
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "video/quicktime" => "mov",
        "audio/mpeg" => "mp3",
        _ => "mp4",
    }
}

/// Whether an ISO base media file (mp4/mov) actually carries an audio track,
/// or `None` when the bytes are not that container and we cannot tell.
///
/// Reports what is in the file rather than what was requested. `generate_audio:
/// false` is not honored by every provider - x-ai/grok-imagine-video-1.5 returns
/// a near-silent AAC track anyway - and echoing the request made the manifest
/// disagree with the file on disk.
///
/// Reads the `hdlr` box, whose handler type sits 12 bytes past the box name:
/// `[size:4][\"hdlr\":4][version+flags:4][pre_defined:4][handler_type:4]`. A
/// `soun` handler means an audio track. Scanning for the box beats a full box
/// walk here - the alternative is an mp4 parsing dependency for one boolean.
fn has_audio_track(bytes: &[u8]) -> Option<bool> {
    // ISO-BMFF starts with a `ftyp` box: [size:4]["ftyp":4]. Anything else
    // (webm, for one) is not something this function can answer for.
    if bytes.len() < 12 || &bytes[4..8] != b"ftyp" {
        return None;
    }
    Some(
        bytes
            .windows(4)
            .enumerate()
            .filter(|(_, w)| *w == b"hdlr")
            .any(|(p, _)| bytes.get(p + 12..p + 16) == Some(&b"soun"[..])),
    )
}

/// Output path for one clip. A single clip uses `base` with the given extension;
/// multiple clips get a `-clip-NNN` suffix.
fn clip_output_path(base: &Path, index_zero_based: usize, total: usize, ext: &str) -> PathBuf {
    crate::output::numbered_path(base, "clip", |index| index, index_zero_based, total, ext)
}

/// Run a video generation job: normalize any frame/reference images, submit the
/// job, poll until terminal, download each clip, save it, write the sidecar
/// manifest, and return a lean summary.
pub async fn run_job(
    client: &OpenRouterClient,
    req: &VideoGenRequest,
    base_output: &Path,
) -> Result<VideoJobSummary> {
    // Invariants upstream only rejects after accepting (and billing) the job.
    req.validate()?;
    let mut warnings = Vec::new();

    // frame_images wins over input_references (image-to-video) - warn if both.
    let use_frames = !req.frames.is_empty();
    if use_frames && req.has_references() {
        warnings.push(
            "both frame_images and references (reference_images/reference_audio/\
             reference_videos) were given; sending only frame_images (image-to-video) \
             and ignoring every reference"
                .to_string(),
        );
    }

    // Build unlabeled InputImages (frames/references carry no per-image label).
    let unlabeled = |paths: Vec<PathBuf>| -> Vec<InputImage> {
        paths
            .into_iter()
            .map(|path| InputImage::from_path(path, None))
            .collect()
    };

    // Normalize frames once, up front (a read/decode failure fails before spend).
    let frame_inputs = unlabeled(req.frames.iter().map(|f| f.path.clone()).collect());
    let frame_prepared =
        image_gen::prepare_inputs_async(&frame_inputs, req.max_image_dimension).await?;
    let mut frame_images = Vec::new();
    let mut frame_meta = Vec::new();
    for (i, (f, p)) in req.frames.iter().zip(&frame_prepared).enumerate() {
        frame_images.push(FrameImage {
            image_url: ImageUrl {
                url: p.data_url.clone(),
            },
            frame_type: f.frame_type.clone(),
        });
        frame_meta.push(FrameImageMeta {
            index: i + 1,
            frame_type: f.frame_type.clone(),
            source: f.path.to_string_lossy().into_owned(),
            normalized_width: p.normalized_width,
            normalized_height: p.normalized_height,
        });
        for w in &p.warnings {
            warnings.push(format!("frame image {}: {w}", i + 1));
        }
    }

    // References are only sent when no frames are present. Order: images,
    // audio, videos. Local audio/video files are inlined as data URLs; URLs
    // pass through.
    let mut input_references = Vec::new();
    let mut reference_meta = Vec::new();
    let mut audio_meta = Vec::new();
    let mut video_meta = Vec::new();
    if !use_frames {
        let ref_inputs = unlabeled(req.references.clone());
        let ref_prepared =
            image_gen::prepare_inputs_async(&ref_inputs, req.max_image_dimension).await?;
        for (p, prep) in req.references.iter().zip(&ref_prepared) {
            input_references.push(InputReference::image(ImageUrl {
                url: prep.data_url.clone(),
            }));
            reference_meta.push(p.to_string_lossy().into_owned());
        }
        for reference in &req.reference_audio {
            input_references.push(InputReference::audio(reference.url.clone()));
            audio_meta.push(reference.source.clone());
        }
        for reference in &req.reference_videos {
            input_references.push(InputReference::video(reference.url.clone()));
            video_meta.push(reference.source.clone());
        }
    }

    let prompt = req.prompt_text().map(str::to_string);
    let body = VideoSubmitBody {
        model: req.model.clone(),
        prompt: prompt.clone(),
        duration: req.duration,
        resolution: req.resolution.clone(),
        aspect_ratio: req.aspect_ratio.clone(),
        size: req.size.clone(),
        frame_images,
        input_references,
        generate_audio: req.generate_audio,
        seed: req.seed,
        creativity: req.creativity,
        upscale_factor: req.upscale_factor,
        provider: req.provider.clone(),
    };

    let submitted = client.submit_video(&body).await?;
    let job_id = submitted.id;
    let manifest = VideoManifest {
        endpoint: "/api/v1/videos",
        job_id: job_id.clone(),
        generation_id: None,
        cost: None,
        model: req.model.clone(),
        prompt,
        prompt_source: crate::manifest::PROMPT_SOURCE,
        duration: req.duration,
        resolution: req.resolution.clone(),
        aspect_ratio: req.aspect_ratio.clone(),
        size: req.size.clone(),
        with_audio: req.generate_audio,
        seed: req.seed,
        max_image_dimension: req.max_image_dimension,
        created_at: chrono::Utc::now().to_rfc3339(),
        frame_images: frame_meta,
        input_references: reference_meta,
        reference_audio: audio_meta,
        reference_videos: video_meta,
        creativity: req.creativity,
        upscale_factor: req.upscale_factor,
        provider: req.provider.clone(),
        clips: Vec::new(),
    };
    let mpath = manifest::path(base_output);
    // Save the upstream ID before polling, so an interruption leaves recovery data.
    if let Err(e) = manifest::write(&mpath, &manifest).await {
        warnings.push(format!(
            "could not persist accepted video job {job_id}: {e:#}"
        ));
    }
    let terminal = wait_for_video(
        &job_id,
        req.poll_interval_secs,
        req.poll_timeout_secs,
        || client.poll_video(&job_id),
    )
    .await;
    let interval = std::time::Duration::from_secs(req.poll_interval_secs);
    let job = job_id.as_str();
    let deadline =
        super::resolve_delivery_timeout().map(|budget| tokio::time::Instant::now() + budget);
    Ok(save_outputs(
        req,
        base_output,
        manifest,
        warnings,
        terminal,
        deadline,
        |index| {
            // The clip is already paid for: a transient failure fetching it must
            // not turn the job into a loss.
            retry_transient(interval, DOWNLOAD_ATTEMPTS, move || {
                client.download_video(job, index)
            })
        },
    )
    .await)
}

/// GET attempts per clip download before giving up on a transient failure.
const DOWNLOAD_ATTEMPTS: u32 = 3;

/// Longest sleep between download retries, however long the poll interval or
/// the upstream's Retry-After: a clip is already paid for, so it is worth
/// retrying, but not after hours of silence.
const MAX_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(300);

/// Run `op` again after a retryable failure (see [`poll_retry_delay`]), up to
/// `attempts` times in total. Permanent failures return immediately.
async fn retry_transient<T, F, Fut>(
    interval: std::time::Duration,
    attempts: u32,
    mut op: F,
) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 1;
    loop {
        match op().await {
            Ok(value) => return Ok(value),
            Err(error) => match poll_retry_delay(&error, interval) {
                Some(delay) if attempt < attempts => {
                    attempt += 1;
                    tokio::time::sleep(delay.min(MAX_RETRY_DELAY)).await;
                }
                _ => return Err(error),
            },
        }
    }
}

/// Bound network work while letting final file and recovery-manifest writes finish.
/// A shared deadline covers every clip and its retries. Never repeat submission.
async fn within_delivery_deadline<T, F, Fut>(
    deadline: Option<tokio::time::Instant>,
    job_id: &str,
    download: F,
) -> Result<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    if let Some(deadline) = deadline {
        let timed_out =
            || format!("video delivery timed out (job {job_id}); recover using the saved job ID");
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(timed_out());
        }
        tokio::time::timeout_at(deadline, download())
            .await
            .with_context(timed_out)?
    } else {
        download().await
    }
}

/// One saved clip and its optional audio warning. Billing belongs to the job.
struct DeliveredClip {
    video: VideoSummary,
    warning: Option<String>,
}

/// Download and save one clip, then inspect its audio track.
/// Failed writes produce neither saved-file metadata nor audio warnings.
async fn deliver_clip<Fut>(
    req: &VideoGenRequest,
    base_output: &Path,
    index: usize,
    total: usize,
    download: Fut,
) -> Result<DeliveredClip>
where
    Fut: std::future::Future<Output = Result<(String, Vec<u8>)>>,
{
    let (mime, bytes) = download.await?;
    let path = clip_output_path(base_output, index, total, extension_for(&mime));
    crate::output::write_bytes(&path, &bytes)
        .await
        .map_err(|e| anyhow::anyhow!("could not write {}: {e}", path.display()))?;

    // Unknown containers retain the requested audio setting as a fallback.
    let probed = has_audio_track(&bytes);
    let has_audio = probed.unwrap_or_else(|| req.generate_audio.unwrap_or(false));
    let warning = match (probed, req.generate_audio) {
        (Some(actual), Some(requested)) if actual != requested => Some(format!(
            "clip {}: requested with_audio={requested} but the file {}",
            index + 1,
            if actual {
                "contains an audio track (the provider ignored the flag)"
            } else {
                "lacks one (the provider ignored the flag or the model \
                 does not support audio)"
            },
        )),
        _ => None,
    };

    Ok(DeliveredClip {
        video: VideoSummary {
            path,
            duration: req.duration,
            resolution: req.resolution.clone(),
            aspect_ratio: req.aspect_ratio.clone(),
            has_audio,
            mime,
        },
        warning,
    })
}

/// Deliver a submitted job without coupling cost to the number of saved clips.
async fn save_outputs<F, Fut>(
    req: &VideoGenRequest,
    base_output: &Path,
    mut manifest: VideoManifest,
    mut warnings: Vec<String>,
    terminal: Result<crate::openrouter::VideoPollResponse>,
    deadline: Option<tokio::time::Instant>,
    mut download: F,
) -> VideoJobSummary
where
    F: FnMut(usize) -> Fut,
    Fut: std::future::Future<Output = Result<(String, Vec<u8>)>>,
{
    let job_id = manifest.job_id.clone();
    let mut videos = Vec::new();
    let mut errors = Vec::new();
    let mut clips = Vec::new();
    // Submission was accepted, so the job is billable even when polling fails.
    // Until usage arrives, its cost remains unknown.
    let (terminal, billing) = match terminal {
        Ok(poll) => {
            let receipt = crate::billing::Receipt {
                cost: poll.usage.as_ref().and_then(|u| u.cost),
                generation_id: poll.generation_id.clone(),
            };
            (Some(poll), receipt)
        }
        Err(error) => {
            let receipt = crate::billing::Receipt::from_error(&error)
                .cloned()
                .unwrap_or_default();
            errors.push(format!("{error:#}"));
            (None, receipt)
        }
    };
    manifest.cost = billing.cost;
    manifest.generation_id = billing.generation_id.clone();

    if let Some(poll) = terminal {
        let total = poll.unsigned_urls.len();
        for index in 0..poll.unsigned_urls.len() {
            let mut meta = VideoClipMeta {
                index: index + 1,
                duration: req.duration,
                resolution: req.resolution.clone(),
                aspect_ratio: req.aspect_ratio.clone(),
                generation_id: poll.generation_id.clone(),
                ..Default::default()
            };
            match deliver_clip(
                req,
                base_output,
                index,
                total,
                within_delivery_deadline(deadline, &job_id, || download(index)),
            )
            .await
            {
                Ok(DeliveredClip { video, warning }) => {
                    meta.path = Some(video.path.to_string_lossy().into_owned());
                    meta.mime_type = Some(video.mime.clone());
                    meta.has_audio = Some(video.has_audio);
                    warnings.extend(warning);
                    videos.push(video);
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    errors.push(format!("clip {}: {msg}", index + 1));
                    meta.error = Some(msg);
                }
            }
            clips.push(meta);
        }
        if poll.unsigned_urls.is_empty() {
            errors.push(format!(
                "video job {job_id} completed but returned no download URLs"
            ));
        }
    }

    manifest.clips = clips;
    let mpath = manifest::path(base_output);
    errors.extend(manifest::write_or_report(&mpath, &manifest).await);

    VideoJobSummary {
        job_id,
        billing,
        model: req.model.clone(),
        manifest_path: mpath,
        videos,
        warnings,
        errors,
    }
}

/// Poll only GET requests again. Never repeat the billable submission.
async fn wait_for_video<F, Fut>(
    job_id: &str,
    interval_secs: u64,
    timeout_secs: u64,
    mut poll: F,
) -> Result<crate::openrouter::VideoPollResponse>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<crate::openrouter::VideoPollResponse>>,
{
    let interval = std::time::Duration::from_secs(interval_secs);
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(timeout_secs.min(super::MAX_TIMER_SECS));
    let timed_out = || format!("video generation timed out after {timeout_secs}s (job {job_id})");
    loop {
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(timed_out());
        }
        let result = tokio::time::timeout_at(deadline, poll())
            .await
            .with_context(timed_out)?;
        let delay = match result {
            Ok(response) => match response.status.as_str() {
                "completed" | "succeeded" => return Ok(response),
                "failed" | "cancelled" | "canceled" | "expired" | "error" => {
                    let error = anyhow::anyhow!(
                        "video generation {} (job {job_id}): {}",
                        response.status,
                        response
                            .error
                            .as_deref()
                            .unwrap_or("no provider explanation")
                    );
                    let receipt = crate::billing::Receipt {
                        cost: response.usage.and_then(|usage| usage.cost),
                        generation_id: response.generation_id,
                    };
                    return Err(receipt.attach(error));
                }
                _ => interval,
            },
            Err(error) => match poll_retry_delay(&error, interval) {
                Some(delay) => delay,
                None => {
                    return Err(error)
                        .with_context(|| format!("video polling failed (job {job_id})"));
                }
            },
        };
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        tokio::time::sleep(delay.min(remaining)).await;
    }
}

fn poll_retry_delay(
    error: &anyhow::Error,
    interval: std::time::Duration,
) -> Option<std::time::Duration> {
    if let Some(http) = error.downcast_ref::<crate::openrouter::HttpFailure>()
        && (http.status.is_server_error() || matches!(http.status.as_u16(), 408 | 429))
    {
        return Some(http.retry_after.unwrap_or(interval).max(interval));
    }
    if error
        .downcast_ref::<reqwest::Error>()
        .is_some_and(|e| e.is_timeout() || e.is_connect() || e.is_body())
    {
        return Some(interval);
    }
    None
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    /// A request that polls fast (sub-second) so the job loop finishes quickly,
    /// with no frames/references for the text-to-video happy path.
    fn text_to_video_request(model: &str) -> VideoGenRequest {
        VideoGenRequest {
            model: model.to_string(),
            prompt: Some("a cat surfing".to_string()),
            duration: Some(4),
            resolution: Some("720p".to_string()),
            aspect_ratio: Some("16:9".to_string()),
            generate_audio: Some(true),
            seed: Some(7),
            poll_timeout_secs: 30,
            ..super::super::tests::request()
        }
    }

    /// Build a minimal ISO-BMFF byte string with the given `hdlr` handler types.
    pub(super) fn fake_mp4(handlers: &[&[u8; 4]]) -> Vec<u8> {
        let mut v = vec![0, 0, 0, 0];
        v.extend_from_slice(b"ftypisom");
        for h in handlers {
            v.extend_from_slice(b"hdlr");
            v.extend_from_slice(&[0; 8]); // version+flags, pre_defined
            v.extend_from_slice(*h);
        }
        v
    }

    /// `has_audio` must describe the file, not the request: grok-imagine-video
    /// returns an AAC track even when generate_audio is false.
    #[test]
    fn has_audio_track_reads_the_container() {
        assert_eq!(has_audio_track(&fake_mp4(&[b"vide", b"soun"])), Some(true));
        assert_eq!(has_audio_track(&fake_mp4(&[b"soun"])), Some(true));
        assert_eq!(has_audio_track(&fake_mp4(&[b"vide"])), Some(false));
        assert_eq!(has_audio_track(&fake_mp4(&[])), Some(false));
        // Not ISO-BMFF (webm here) -> no answer, so the caller keeps its fallback.
        assert_eq!(
            has_audio_track(&[0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0]),
            None
        );
        assert_eq!(has_audio_track(b"short"), None);
        // A bare "soun" outside an hdlr box must not count.
        let mut stray = fake_mp4(&[b"vide"]);
        stray.extend_from_slice(b"soun");
        assert_eq!(has_audio_track(&stray), Some(false));
    }

    #[test]
    fn extension_for_maps_known_mimes_and_falls_back_to_mp4() {
        assert_eq!(extension_for("video/mp4"), "mp4");
        assert_eq!(extension_for("video/webm"), "webm");
        assert_eq!(extension_for("video/quicktime"), "mov");
        assert_eq!(extension_for("application/octet-stream"), "mp4");
    }

    #[test]
    fn clip_output_path_single_uses_base_and_multi_suffixes() {
        let single = clip_output_path(Path::new("out/clip.mp4"), 0, 1, "mp4");
        assert_eq!(single, PathBuf::from("out/clip.mp4"));
        let multi = clip_output_path(Path::new("out/clip.mp4"), 1, 3, "webm");
        assert_eq!(multi, PathBuf::from("out/clip-clip-002.webm"));
    }

    #[tokio::test]
    async fn run_job_submits_polls_downloads_and_saves_the_clip() {
        let server = MockServer::start().await;
        // Submit returns a job id; we verify the body shape we build.
        Mock::given(method("POST"))
            .and(path("/videos"))
            .and(body_partial_json(json!({
                "model": "google/veo-3.1",
                "prompt": "a cat surfing",
                "duration": 4,
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "generate_audio": true,
                "seed": 7
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-job-1",
                "status": "pending"
            })))
            .mount(&server)
            .await;
        // First poll completes with one download URL and a cost.
        Mock::given(method("GET"))
            .and(path("/videos/vid-job-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-job-1",
                "generation_id": "gen-vid-9",
                "status": "completed",
                "unsigned_urls": ["https://cdn/clip-0.mp4"],
                "usage": { "cost": 1.23 }
            })))
            .mount(&server)
            .await;
        // Content download for index 0 returns mp4 bytes.
        Mock::given(method("GET"))
            .and(path("/videos/vid-job-1/content"))
            .and(query_param("index", "0"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "video/mp4")
                    .set_body_bytes(b"FAKE-MP4-BYTES".to_vec()),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = text_to_video_request("google/veo-3.1");
        let base = std::env::temp_dir().join("openrouter-mcp-video-test/clip.mp4");
        let summary = run_job(&client, &req, &base).await.unwrap();

        assert_eq!(summary.model, "google/veo-3.1");
        assert_eq!(summary.videos.len(), 1);
        assert!(summary.errors.is_empty(), "errors: {:?}", summary.errors);
        let v = &summary.videos[0];
        assert_eq!(v.mime, "video/mp4");
        assert!(v.has_audio, "generate_audio=true -> has_audio");
        assert_eq!(summary.billing.cost, Some(1.23));
        // The clip bytes landed on disk at the .mp4 path.
        assert_eq!(std::fs::read(&v.path).unwrap(), b"FAKE-MP4-BYTES");
    }

    /// Mount a submit mock that requires `expected` in the body, a poll that
    /// completes at once with one clip, and that clip's download.
    async fn mount_completed_job(server: &MockServer, job: &str, expected: serde_json::Value) {
        Mock::given(method("POST"))
            .and(path("/videos"))
            .and(body_partial_json(expected))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({ "id": job, "status": "pending" })),
            )
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/videos/{job}")))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": job,
                "status": "completed",
                "unsigned_urls": ["https://cdn/clip-0.mp4"],
                "usage": { "cost": 0.5 }
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/videos/{job}/content")))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "video/mp4")
                    .set_body_bytes(fake_mp4(&[b"vide"])),
            )
            .mount(server)
            .await;
    }

    /// The JSON body of the single POST /videos the mock received.
    async fn submitted_body(server: &MockServer) -> serde_json::Value {
        let posts: Vec<_> = server
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.method == wiremock::http::Method::POST)
            .collect();
        assert_eq!(posts.len(), 1, "exactly one submission");
        serde_json::from_slice(&posts[0].body).unwrap()
    }

    /// A tiny valid PNG on disk, for frame inputs that go through the image
    /// pipeline (which decodes them).
    fn write_png(path: &Path) {
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 120, 200, 255]));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        image::DynamicImage::ImageRgba8(img)
            .save_with_format(path, image::ImageFormat::Png)
            .unwrap();
    }

    /// `provider.options` (opaque, unchanged), `creativity` and `upscale_factor`
    /// reach the wire under their documented names, and the manifest records the
    /// provider block.
    #[tokio::test]
    async fn run_job_forwards_provider_options_creativity_and_upscale_factor() {
        let server = MockServer::start().await;
        mount_completed_job(
            &server,
            "vid-opts",
            json!({
                "provider": { "options": { "google-vertex": { "negativePrompt": "blurry" } } },
                "creativity": 5,
                "upscale_factor": 2.0
            }),
        )
        .await;
        let mut options = std::collections::BTreeMap::new();
        options.insert(
            "google-vertex".to_string(),
            json!({ "negativePrompt": "blurry" }),
        );
        let req = VideoGenRequest {
            creativity: Some(5),
            upscale_factor: Some(2.0),
            provider: Some(crate::openrouter::ProviderOptions { options }),
            ..text_to_video_request("google/veo-3.1")
        };
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("clip.mp4");
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let summary = run_job(&client, &req, &base).await.unwrap();
        assert!(summary.errors.is_empty(), "{:?}", summary.errors);

        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&summary.manifest_path).unwrap()).unwrap();
        assert_eq!(
            manifest["provider"],
            json!({ "options": { "google-vertex": { "negativePrompt": "blurry" } } })
        );
        assert_eq!(manifest["creativity"], 5);
        assert_eq!(manifest["upscale_factor"], 2.0);
    }

    /// Audio and video references become `audio_url` / `video_url` content parts
    /// carrying their resolved URL, while the manifest records each source as
    /// given. Order: images, audio, videos.
    #[tokio::test]
    async fn run_job_sends_audio_and_video_references_as_typed_parts() {
        let dir = tempfile::tempdir().unwrap();
        let reference = |source: &str, url: &str| super::super::MediaReference {
            source: source.to_string(),
            url: url.to_string(),
        };

        let server = MockServer::start().await;
        mount_completed_job(
            &server,
            "vid-refs",
            json!({
                "input_references": [
                    { "type": "audio_url", "audio_url": { "url": "https://cdn/song.mp3" } },
                    { "type": "audio_url", "audio_url": { "url": "data:audio/mpeg;base64,QUJD" } },
                    { "type": "video_url", "video_url": { "url": "data:video/mp4;base64,QUJD" } }
                ]
            }),
        )
        .await;
        let req = VideoGenRequest {
            reference_audio: vec![
                reference("https://cdn/song.mp3", "https://cdn/song.mp3"),
                reference("beat.mp3", "data:audio/mpeg;base64,QUJD"),
            ],
            reference_videos: vec![reference("ref.mp4", "data:video/mp4;base64,QUJD")],
            ..text_to_video_request("bytedance/seedance-2.0")
        };
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let summary = run_job(&client, &req, &dir.path().join("out.mp4"))
            .await
            .unwrap();
        assert!(summary.errors.is_empty(), "{:?}", summary.errors);
        let body = submitted_body(&server).await;
        assert_eq!(body["input_references"].as_array().unwrap().len(), 3);

        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&summary.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["reference_audio"].as_array().unwrap().len(), 2);
        assert_eq!(manifest["reference_videos"], json!(["ref.mp4"]));
    }

    /// Image-only models take no prompt: a frame-only request sends no `prompt`
    /// key at all. Frames still win over every reference kind, with the warning.
    #[tokio::test]
    async fn run_job_sends_no_prompt_for_a_frame_only_request_and_frames_win_over_references() {
        let dir = tempfile::tempdir().unwrap();
        let frame = dir.path().join("first.png");
        write_png(&frame);

        let server = MockServer::start().await;
        mount_completed_job(&server, "vid-frame", json!({ "model": "test/i2v" })).await;
        let req = VideoGenRequest {
            prompt: None,
            aspect_ratio: None,
            frames: vec![super::super::VideoInput {
                path: frame.clone(),
                frame_type: "first_frame".to_string(),
            }],
            reference_audio: vec![super::super::MediaReference {
                source: "https://cdn/song.mp3".to_string(),
                url: "https://cdn/song.mp3".to_string(),
            }],
            ..text_to_video_request("test/i2v")
        };
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let summary = run_job(&client, &req, &dir.path().join("out.mp4"))
            .await
            .unwrap();
        assert!(summary.errors.is_empty(), "{:?}", summary.errors);
        let body = submitted_body(&server).await;
        assert!(body.get("prompt").is_none(), "sent: {body}");
        assert_eq!(body["frame_images"].as_array().unwrap().len(), 1);
        assert!(body.get("input_references").is_none(), "sent: {body}");
        assert!(
            summary
                .warnings
                .iter()
                .any(|w| w.contains("reference") && w.contains("ignoring")),
            "{:?}",
            summary.warnings
        );
    }

    /// Both invariants are checked before any HTTP call: a prompt is required
    /// unless a frame or reference is present, and upscale_factor must be > 0.
    #[tokio::test]
    async fn run_job_rejects_missing_prompt_and_bad_upscale_factor_before_submit() {
        let server = MockServer::start().await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("out.mp4");

        let no_prompt = VideoGenRequest {
            prompt: None,
            ..text_to_video_request("m")
        };
        let err = run_job(&client, &no_prompt, &base)
            .await
            .err()
            .expect("rejected")
            .to_string();
        assert!(err.contains("prompt"), "got: {err}");

        for bad in [0.0, -1.5] {
            let req = VideoGenRequest {
                upscale_factor: Some(bad),
                ..text_to_video_request("m")
            };
            let err = run_job(&client, &req, &base)
                .await
                .err()
                .expect("rejected")
                .to_string();
            assert!(err.contains("upscale_factor"), "got: {err}");
        }
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    /// grok-imagine-video returns an AAC track even for with_audio=false: the
    /// job must warn about the flag being ignored, not just flip has_audio.
    #[tokio::test]
    async fn run_job_warns_when_the_provider_ignores_with_audio_false() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .and(body_partial_json(json!({ "generate_audio": false })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-job-2",
                "status": "pending"
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/videos/vid-job-2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-job-2",
                "status": "completed",
                "unsigned_urls": ["https://cdn/clip-0.mp4"]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/videos/vid-job-2/content"))
            .and(query_param("index", "0"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "video/mp4")
                    .set_body_bytes(fake_mp4(&[b"vide", b"soun"])),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let mut req = text_to_video_request("xai/grok-imagine-video");
        req.generate_audio = Some(false);
        let base = std::env::temp_dir().join("openrouter-mcp-video-warn-test/clip.mp4");
        let summary = run_job(&client, &req, &base).await.unwrap();

        assert!(summary.videos[0].has_audio, "probe reads the file");
        assert!(
            summary
                .warnings
                .iter()
                .any(|w| w.contains("with_audio=false") && w.contains("contains")),
            "warnings: {:?}",
            summary.warnings
        );
    }

    #[tokio::test]
    async fn run_job_records_a_failed_status_as_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "id": "vid-job-2" })))
            .mount(&server)
            .await;
        // The poll reports the job failed: no clips, one error, no panic.
        Mock::given(method("GET"))
            .and(path("/videos/vid-job-2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-job-2",
                "status": "failed",
                "unsigned_urls": []
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = text_to_video_request("google/veo-3.1");
        // Pre-create the output dir so the manifest still writes (a failed job
        // produces no clip, so the dir is otherwise never created): this isolates
        // the assertion to the single "generation failed" error.
        let dir = std::env::temp_dir().join("openrouter-mcp-video-fail");
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("clip.mp4");
        let summary = run_job(&client, &req, &base).await.unwrap();

        assert!(summary.videos.is_empty());
        assert_eq!(summary.errors.len(), 1, "errors: {:?}", summary.errors);
        assert!(summary.errors[0].contains("failed"));
    }

    #[tokio::test]
    async fn run_job_surfaces_a_submit_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .respond_with(ResponseTemplate::new(400).set_body_string("{\"error\":\"bad model\"}"))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = text_to_video_request("nope/model");
        let base = std::env::temp_dir().join("openrouter-mcp-video-submit-err/clip.mp4");
        // A submit failure aborts the whole job before any polling/spend.
        let err = match run_job(&client, &req, &base).await {
            Err(e) => e,
            Ok(_) => panic!("submit error should abort the job"),
        };
        assert!(err.to_string().contains("bad model"));
    }
}

#[cfg(test)]
mod audit_regression {
    use super::*;
    fn request() -> VideoGenRequest {
        VideoGenRequest {
            model: "test/video".into(),
            prompt: Some("test".into()),
            duration: Some(5),
            poll_timeout_secs: 5,
            ..super::super::tests::request()
        }
    }
    fn manifest() -> VideoManifest {
        VideoManifest {
            endpoint: "/api/v1/videos",
            job_id: "accepted-job".into(),
            generation_id: None,
            cost: None,
            model: "test/video".into(),
            prompt: Some("test".into()),
            prompt_source: "test",
            duration: Some(5),
            resolution: None,
            aspect_ratio: None,
            size: None,
            with_audio: None,
            seed: None,
            max_image_dimension: 800,
            created_at: "test".into(),
            frame_images: vec![],
            input_references: vec![],
            reference_audio: vec![],
            reference_videos: vec![],
            creativity: None,
            upscale_factor: None,
            provider: None,
            clips: vec![],
        }
    }
    #[tokio::test(start_paused = true)]
    async fn delivery_deadline_bounds_retry_wait_and_stalled_body() {
        let budget = std::time::Duration::from_secs(5);
        let start = tokio::time::Instant::now();
        let mut calls = 0;
        let error = within_delivery_deadline(Some(start + budget), "paid-job", || {
            retry_transient(std::time::Duration::from_secs(1), DOWNLOAD_ATTEMPTS, || {
                calls += 1;
                std::future::ready(Err::<(), _>(failure(429, Some(u64::MAX))))
            })
        })
        .await
        .unwrap_err();
        assert_eq!(calls, 1);
        assert_eq!(start.elapsed(), budget);
        assert!(format!("{error:#}").contains("paid-job"));
        let start = tokio::time::Instant::now();
        let error =
            within_delivery_deadline::<(), _, _>(Some(start + budget), "slow-body", || async {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            })
            .await
            .unwrap_err();
        assert_eq!(start.elapsed(), budget);
        assert!(format!("{error:#}").contains("slow-body"));
    }

    #[tokio::test(start_paused = true)]
    async fn disabled_delivery_deadline_still_bounds_extreme_retry_hints() {
        let start = tokio::time::Instant::now();
        let mut calls = 0;
        within_delivery_deadline(None, "paid-job", || {
            retry_transient(
                std::time::Duration::from_secs(u64::MAX),
                DOWNLOAD_ATTEMPTS,
                || {
                    calls += 1;
                    std::future::ready(Err::<(), _>(failure(429, Some(u64::MAX))))
                },
            )
        })
        .await
        .unwrap_err();
        assert_eq!(calls, 3);
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(600));
    }

    #[tokio::test(start_paused = true)]
    async fn shared_delivery_deadline_preserves_receipt_and_manifest_for_all_clips() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("clip.mp4");
        let poll = serde_json::from_value(serde_json::json!({
            "status":"completed", "generation_id":"gen-test", "usage":{"cost":0.75},
            "unsigned_urls":["https://cdn/0", "https://cdn/1"]
        }))
        .unwrap();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut downloads = Vec::new();
        let summary = save_outputs(
            &request(),
            &base,
            manifest(),
            vec![],
            Ok(poll),
            Some(deadline),
            |index| {
                downloads.push(index);
                std::future::pending::<Result<(String, Vec<u8>)>>()
            },
        )
        .await;
        assert_eq!(
            downloads,
            vec![0],
            "do not start another download after expiry"
        );
        assert!(summary.videos.is_empty());
        assert_eq!(summary.billing.cost, Some(0.75));
        assert_eq!(summary.billing.generation_id.as_deref(), Some("gen-test"));
        assert_eq!(summary.errors.len(), 2);
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(summary.manifest_path).unwrap()).unwrap();
        assert_eq!(saved["job_id"], "accepted-job");
        assert_eq!(saved["cost"], 0.75);
        assert_eq!(saved["generation_id"], "gen-test");
        for clip in saved["clips"].as_array().unwrap() {
            assert!(
                clip["error"]
                    .as_str()
                    .unwrap()
                    .contains("delivery timed out")
            );
            assert!(clip.get("path").is_none());
        }
    }

    #[tokio::test]
    async fn video_delivery_counts_one_cost_for_multiple_clips_and_failed_writes() {
        for failed_writes in 0..=2 {
            let dir = tempfile::tempdir().unwrap();
            let base = dir.path().join("nested/clip.mp4");
            for index in 0..failed_writes {
                std::fs::create_dir_all(clip_output_path(&base, index, 3, "mp4")).unwrap();
            }
            let mut req = request();
            req.generate_audio = Some(false);
            let bytes = super::tests::fake_mp4(&[b"vide", b"soun"]);
            let poll = serde_json::from_value(serde_json::json!({
                "status":"completed", "generation_id":"gen-test", "usage":{"cost":0.75},
                "unsigned_urls":["https://cdn/0", "https://cdn/1", "https://cdn/2"]
            }))
            .unwrap();
            let mut downloads = Vec::new();
            let summary = save_outputs(&req, &base, manifest(), vec![], Ok(poll), None, |index| {
                downloads.push(index);
                std::future::ready(if index == 2 {
                    Err(anyhow::anyhow!("network failure").context("download failed"))
                } else {
                    Ok(("video/mp4".into(), bytes.clone()))
                })
            })
            .await;
            assert_eq!(downloads, [0, 1, 2]);
            assert_eq!(summary.videos.len(), 2 - failed_writes);
            assert_eq!(summary.billing.cost, Some(0.75));
            assert_eq!(summary.billing.generation_id.as_deref(), Some("gen-test"));
            assert_eq!(summary.errors.len(), failed_writes + 1);
            assert_eq!(
                summary.errors.last().unwrap(),
                "clip 3: download failed: network failure"
            );
            let expected_warnings: Vec<_> = (failed_writes..2)
                .map(|index| format!(
                    "clip {}: requested with_audio=false but the file contains an audio track (the provider ignored the flag)",
                    index + 1
                ))
                .collect();
            assert_eq!(summary.warnings, expected_warnings);
            for (video, index) in summary.videos.iter().zip(failed_writes..2) {
                assert_eq!(video.path, clip_output_path(&base, index, 3, "mp4"));
                assert_eq!(std::fs::read(&video.path).unwrap(), bytes);
                assert!(video.has_audio);
            }
            let saved: serde_json::Value =
                serde_json::from_slice(&std::fs::read(summary.manifest_path).unwrap()).unwrap();
            assert_eq!(saved["cost"], 0.75);
            assert_eq!(saved["generation_id"], "gen-test");
            assert_eq!(saved["job_id"], "accepted-job");
            assert_eq!(saved["clips"].as_array().unwrap().len(), 3);
            assert!(
                saved["clips"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|clip| clip.get("cost").is_none())
            );
            for (index, clip) in saved["clips"].as_array().unwrap().iter().enumerate() {
                assert_eq!(clip["index"], index + 1);
                assert_eq!(clip["generation_id"], "gen-test");
                if index < failed_writes || index == 2 {
                    for key in ["path", "mime_type", "has_audio"] {
                        assert!(clip.get(key).is_none(), "failed clip has {key}");
                    }
                    let error = clip["error"].as_str().unwrap();
                    let error_index = if index == 2 { failed_writes } else { index };
                    assert_eq!(
                        summary.errors[error_index],
                        format!("clip {}: {error}", index + 1)
                    );
                    if index < failed_writes {
                        assert!(error.starts_with(&format!(
                            "could not write {}:",
                            clip_output_path(&base, index, 3, "mp4").display()
                        )));
                    }
                } else {
                    assert_eq!(
                        clip["path"],
                        clip_output_path(&base, index, 3, "mp4")
                            .to_string_lossy()
                            .as_ref()
                    );
                    assert_eq!(clip["mime_type"], "video/mp4");
                    assert_eq!(clip["has_audio"], true);
                    assert!(clip.get("error").is_none());
                }
            }
        }
    }

    #[tokio::test]
    async fn manifest_write_failure_preserves_saved_video_and_billing() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("clip.webm");
        let manifest_path = manifest::path(&base);
        std::fs::create_dir(&manifest_path).unwrap();
        let mut req = request();
        req.generate_audio = Some(true);
        let poll = serde_json::from_value(serde_json::json!({
            "status":"completed", "generation_id":"gen-test", "usage":{"cost":0.75},
            "unsigned_urls":["https://cdn/0"]
        }))
        .unwrap();
        let summary = save_outputs(&req, &base, manifest(), vec![], Ok(poll), None, |_| {
            std::future::ready(Ok(("video/webm".into(), vec![1, 2, 3])))
        })
        .await;

        assert_eq!(summary.job_id, "accepted-job");
        assert_eq!(summary.billing.cost, Some(0.75));
        assert_eq!(summary.billing.generation_id.as_deref(), Some("gen-test"));
        assert_eq!(summary.manifest_path, manifest_path);
        assert_eq!(summary.videos.len(), 1);
        assert_eq!(summary.videos[0].path, base);
        assert_eq!(std::fs::read(&base).unwrap(), [1, 2, 3]);
        assert!(
            summary.videos[0].has_audio,
            "unknown container uses request fallback"
        );
        assert!(summary.warnings.is_empty());
        assert_eq!(summary.errors.len(), 1);
        assert!(summary.errors[0].starts_with("manifest write failed:"));
        assert!(
            summary.errors[0].contains("os error"),
            "the cause is kept: {}",
            summary.errors[0]
        );
        assert!(manifest_path.is_dir());
    }
    #[tokio::test]
    async fn accepted_video_without_usage_keeps_unknown_cost_and_recovery_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("new/clip.mp4");
        let summary = save_outputs(
            &request(),
            &base,
            manifest(),
            vec![],
            Err(anyhow::anyhow!("poll timed out")),
            None,
            |_| std::future::ready(Err(anyhow::anyhow!("must not download"))),
        )
        .await;
        assert!(summary.videos.is_empty());
        assert_eq!(summary.billing.cost, None);
        let saved: serde_json::Value =
            serde_json::from_slice(&std::fs::read(summary.manifest_path).unwrap()).unwrap();
        assert_eq!(saved["job_id"], "accepted-job");
        assert!(saved["cost"].is_null());
    }
    fn response(status: &str) -> crate::openrouter::VideoPollResponse {
        serde_json::from_value(serde_json::json!({"status":status})).unwrap()
    }
    fn failure(status: u16, retry_after: Option<u64>) -> anyhow::Error {
        crate::openrouter::HttpFailure {
            status: reqwest::StatusCode::from_u16(status).unwrap(),
            retry_after: retry_after.map(std::time::Duration::from_secs),
            label: "/videos/job-test".into(),
            body: "synthetic".into(),
        }
        .into()
    }
    #[tokio::test(start_paused = true)]
    async fn poll_retries_transient_failures_and_respects_retry_after() {
        let start = tokio::time::Instant::now();
        let mut responses = std::collections::VecDeque::from([
            Err(failure(429, Some(7))),
            Err(failure(503, None)),
            Ok(response("processing")),
            Ok(response("completed")),
        ]);
        let done = wait_for_video("job-test", 2, 30, || {
            std::future::ready(responses.pop_front().unwrap())
        })
        .await
        .unwrap();
        assert_eq!(done.status, "completed");
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(11));
        assert!(responses.is_empty());
    }
    #[tokio::test(start_paused = true)]
    async fn poll_timeout_never_exceeds_deadline_or_loses_job_id() {
        let start = tokio::time::Instant::now();
        let error = wait_for_video("job-test", 1, 5, || {
            std::future::ready(Err(failure(429, Some(100))))
        })
        .await
        .unwrap_err();
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(5));
        assert!(format!("{error:#}").contains("job-test"));
    }
    #[tokio::test(start_paused = true)]
    async fn stalled_poll_is_bounded_and_permanent_errors_do_not_retry() {
        let error = wait_for_video("stalled", 1, 5, std::future::pending)
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("stalled"));
        for status in [401, 403, 404] {
            let mut calls = 0;
            let error = wait_for_video("permanent", 1, 30, || {
                calls += 1;
                std::future::ready(Err(failure(status, None)))
            })
            .await
            .unwrap_err();
            assert_eq!(calls, 1);
            assert!(format!("{error:#}").contains("permanent"));
        }
    }
    #[tokio::test(start_paused = true)]
    async fn download_retries_transient_failures_but_not_permanent_ones() {
        let interval = std::time::Duration::from_secs(2);
        let mut responses = std::collections::VecDeque::from([
            Err(failure(503, None)),
            Err(failure(429, Some(1))),
            Ok(("video/mp4".to_string(), vec![1u8])),
        ]);
        let start = tokio::time::Instant::now();
        let (mime, bytes) = retry_transient(interval, DOWNLOAD_ATTEMPTS, || {
            std::future::ready(responses.pop_front().unwrap())
        })
        .await
        .unwrap();
        assert_eq!((mime.as_str(), bytes), ("video/mp4", vec![1u8]));
        assert_eq!(start.elapsed(), std::time::Duration::from_secs(4));

        let mut calls = 0;
        let error = retry_transient(interval, DOWNLOAD_ATTEMPTS, || {
            calls += 1;
            std::future::ready(Err::<(), _>(failure(404, None)))
        })
        .await
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(format!("{error:#}").contains("404"));

        let mut calls = 0;
        retry_transient(interval, DOWNLOAD_ATTEMPTS, || {
            calls += 1;
            std::future::ready(Err::<(), _>(failure(503, None)))
        })
        .await
        .unwrap_err();
        assert_eq!(calls, DOWNLOAD_ATTEMPTS);
    }
    #[tokio::test(start_paused = true)]
    async fn absurd_poll_timeout_is_capped_instead_of_overflowing() {
        let error = wait_for_video("capped", 1, u64::MAX, || {
            std::future::ready(Err(failure(401, None)))
        })
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("capped"));
    }
    #[tokio::test(start_paused = true)]
    async fn terminal_failure_preserves_provider_explanation_and_known_cost() {
        let error = wait_for_video("rejected",1,30,|| std::future::ready(Ok(
            serde_json::from_value(serde_json::json!({"status":"failed","error":"provider explanation","usage":{"cost":0.2}})).unwrap()
        ))).await.unwrap_err();
        assert!(format!("{error:#}").contains("provider explanation"));
        assert_eq!(
            error
                .downcast_ref::<crate::billing::Receipt>()
                .unwrap()
                .cost,
            Some(0.2)
        );
    }
}
