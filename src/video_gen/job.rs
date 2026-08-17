//! Video job orchestration: submit, poll, download each clip, write the sidecar
//! manifest, and return a lean summary. Shared by the CLI and the MCP tool.

use std::path::{Path, PathBuf};

use anyhow::Result;

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
    if total <= 1 {
        return base.with_extension(ext);
    }
    let width = 3.max(total.to_string().len());
    image_gen::in_parent_of(
        base,
        format!(
            "{}-clip-{:0width$}.{ext}",
            image_gen::base_stem(base),
            index_zero_based + 1,
            width = width
        ),
    )
}

/// Run a video generation job: normalize any frame/reference images, submit the
/// job, poll until terminal, download each clip, save it, write the sidecar
/// manifest, and return a lean summary. Shared by the CLI and the MCP tool.
pub async fn run_job(
    client: &OpenRouterClient,
    req: &VideoGenRequest,
    base_output: &Path,
    prompt_source: &str,
) -> Result<VideoJobSummary> {
    let mut warnings = Vec::new();

    // frame_images wins over input_references (image-to-video) - warn if both.
    let use_frames = !req.frames.is_empty();
    if use_frames && !req.references.is_empty() {
        warnings.push(
            "both frame_images and reference_images were given; sending only \
             frame_images (image-to-video) and ignoring reference_images"
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
        frame_images.push(FrameImage::new(
            ImageUrl {
                url: p.data_url.clone(),
            },
            f.frame_type.clone(),
        ));
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

    // References are only sent when no frames are present.
    let mut input_references = Vec::new();
    let mut reference_meta = Vec::new();
    if !use_frames {
        let ref_inputs = unlabeled(req.references.clone());
        let ref_prepared =
            image_gen::prepare_inputs_async(&ref_inputs, req.max_image_dimension).await?;
        for (p, prep) in req.references.iter().zip(&ref_prepared) {
            input_references.push(InputReference::new(ImageUrl {
                url: prep.data_url.clone(),
            }));
            reference_meta.push(p.to_string_lossy().into_owned());
        }
    }

    let body = VideoSubmitBody {
        model: req.model.clone(),
        prompt: req.prompt.clone(),
        duration: req.duration,
        resolution: req.resolution.clone(),
        aspect_ratio: req.aspect_ratio.clone(),
        size: req.size.clone(),
        frame_images,
        input_references,
        generate_audio: req.generate_audio,
        seed: req.seed,
    };

    let mut videos = Vec::new();
    let mut errors = Vec::new();
    let mut clips = Vec::new();

    // Submit, then poll until a terminal status or the poll timeout elapses.
    let submitted = client.submit_video(&body).await?;
    let job_id = submitted.id;
    let interval = std::time::Duration::from_secs(req.poll_interval_secs);
    let timeout = std::time::Duration::from_secs(req.poll_timeout_secs);
    let deadline = tokio::time::Instant::now() + timeout;

    let mut terminal: Option<crate::openrouter::VideoPollResponse> = None;
    let mut timed_out = false;
    loop {
        // Poll immediately after submission. Bound the request itself by the
        // overall deadline so a slow network call cannot overrun the job timeout.
        let poll = match tokio::time::timeout_at(deadline, client.poll_video(&job_id)).await {
            Ok(result) => result?,
            Err(_) => {
                timed_out = true;
                break;
            }
        };
        match poll.status.as_str() {
            "completed" | "succeeded" => {
                terminal = Some(poll);
                break;
            }
            "failed" | "cancelled" | "canceled" | "expired" | "error" => {
                errors.push(format!("video generation {}: {}", poll.status, job_id));
                break;
            }
            _ => {
                // pending/processing/queued/running/unknown -> keep waiting.
            }
        }
        let now = tokio::time::Instant::now();
        if now >= deadline {
            timed_out = true;
            break;
        }
        tokio::time::sleep_until((now + interval).min(deadline)).await;
    }
    if timed_out {
        errors.push(format!(
            "video generation timed out after {}s (job {job_id})",
            req.poll_timeout_secs
        ));
    }

    if let Some(poll) = terminal {
        let cost = poll.usage.as_ref().and_then(|u| u.cost);
        let total = poll.unsigned_urls.len().max(1);
        for index in 0..poll.unsigned_urls.len() {
            let mut meta = VideoClipMeta {
                index: index + 1,
                duration: req.duration,
                resolution: req.resolution.clone(),
                aspect_ratio: req.aspect_ratio.clone(),
                generation_id: poll.generation_id.clone(),
                cost,
                ..Default::default()
            };
            match client.download_video(&job_id, index).await {
                Ok((mime, bytes)) => {
                    let ext = extension_for(&mime);
                    let path = clip_output_path(base_output, index, total, ext);
                    match crate::output::write_bytes(&path, &bytes).await {
                        Ok(()) => {
                            // Probe the bytes; fall back to the request only for
                            // a container we cannot read.
                            let has_audio = has_audio_track(&bytes)
                                .unwrap_or_else(|| req.generate_audio.unwrap_or(false));
                            meta.path = Some(path.to_string_lossy().into_owned());
                            meta.mime_type = Some(mime.clone());
                            meta.has_audio = Some(has_audio);
                            videos.push(VideoSummary {
                                path,
                                duration: req.duration,
                                resolution: req.resolution.clone(),
                                aspect_ratio: req.aspect_ratio.clone(),
                                has_audio,
                                mime,
                                cost,
                            });
                        }
                        Err(e) => {
                            let msg = format!("could not write {}: {e}", path.display());
                            errors.push(format!("clip {}: {msg}", index + 1));
                            meta.error = Some(msg);
                        }
                    }
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

    let manifest = VideoManifest {
        endpoint: "/api/v1/videos",
        model: req.model.clone(),
        prompt: req.prompt.clone(),
        prompt_source: prompt_source.to_string(),
        duration: req.duration,
        resolution: req.resolution.clone(),
        aspect_ratio: req.aspect_ratio.clone(),
        size: req.size.clone(),
        generate_audio: req.generate_audio,
        seed: req.seed,
        max_image_dimension: req.max_image_dimension,
        created_at: chrono::Utc::now().to_rfc3339(),
        frame_images: frame_meta,
        input_references: reference_meta,
        clips,
    };
    let mpath = manifest::path(base_output);
    if let Err(e) = manifest::write(&mpath, &manifest).await {
        errors.push(format!("manifest write failed: {e}"));
    }

    Ok(VideoJobSummary {
        model: req.model.clone(),
        manifest_path: mpath,
        videos,
        warnings,
        errors,
    })
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
            prompt: "a cat surfing".to_string(),
            duration: Some(4),
            resolution: Some("720p".to_string()),
            aspect_ratio: Some("16:9".to_string()),
            size: None,
            generate_audio: Some(true),
            seed: Some(7),
            frames: vec![],
            references: vec![],
            max_image_dimension: 800,
            poll_interval_secs: 1,
            poll_timeout_secs: 30,
        }
    }

    /// Build a minimal ISO-BMFF byte string with the given `hdlr` handler types.
    fn fake_mp4(handlers: &[&[u8; 4]]) -> Vec<u8> {
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
        let summary = run_job(&client, &req, &base, "test").await.unwrap();

        assert_eq!(summary.model, "google/veo-3.1");
        assert_eq!(summary.videos.len(), 1);
        assert!(summary.errors.is_empty(), "errors: {:?}", summary.errors);
        let v = &summary.videos[0];
        assert_eq!(v.mime, "video/mp4");
        assert!(v.has_audio, "generate_audio=true -> has_audio");
        assert_eq!(v.cost, Some(1.23));
        // The clip bytes landed on disk at the .mp4 path.
        assert_eq!(std::fs::read(&v.path).unwrap(), b"FAKE-MP4-BYTES");
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
        let summary = run_job(&client, &req, &base, "test").await.unwrap();

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
        let err = match run_job(&client, &req, &base, "test").await {
            Err(e) => e,
            Ok(_) => panic!("submit error should abort the job"),
        };
        assert!(err.to_string().contains("bad model"));
    }
}
