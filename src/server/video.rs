//! The `generate_video` tool, its argument struct, and the video-job result builder.

use std::path::PathBuf;

use rmcp::{
    ErrorData, RoleServer, handler::server::wrapper::Parameters, model::CallToolResult,
    service::RequestContext, tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::image_gen;
use crate::server::naming;
use crate::server::provider::ProviderOptionsArgs;
use crate::server::result::{
    DEFAULT_VIDEO_WAIT_SECONDS, attach_warnings_errors, client_wants_inline_previews,
};
use crate::server::schema::{
    RequireFields, de_lenient, de_opt_bool, de_opt_f64, de_opt_uint, require_all,
    scalarize_nullable,
};
use crate::tasks::TaskKind;
use crate::video_gen::{self, VideoGenRequest, VideoInput};

use super::OpenRouterServer;

/// Arguments for the `generate_video` tool.
///
/// `aspect_ratio` and `prompt` are deliberately NOT in the schema's required
/// list: each is only required for text-to-video (no frame image or reference);
/// see `run_generate_video` and `VideoGenRequest::validate` for the conditional
/// rules.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = RequireFields(&["duration", "with_audio"]))]
pub(crate) struct GenerateVideoArgs {
    /// Video model id, e.g. "google/veo-3.1". Use list_models with
    /// output_modalities="video" to discover them.
    pub model: String,
    /// Prompt text describing the video to generate. Required unless a
    /// first_frame/last_frame or a reference (reference_images, reference_audio,
    /// reference_videos) is given - image-only models take no text.
    #[serde(default)]
    pub prompt: Option<String>,
    /// REQUIRED (no default): clip length in seconds - NOT how long generation
    /// takes (that's 30s to several minutes; see the tool description). Accepted
    /// values are model-specific discrete seconds; check describe_model for what
    /// a given model supports.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub duration: Option<u32>,
    /// Named resolution tier: "480p", "720p", "768p", "1080p", "1K", "2K", or
    /// "4K". For text-to-video, pair with aspect_ratio - or use `size` instead
    /// of resolution+aspect_ratio.
    #[serde(default)]
    pub resolution: Option<String>,
    /// REQUIRED (no default) for text-to-video only: aspect ratio, e.g. "16:9",
    /// "9:16", "1:1". Provide aspect_ratio+resolution OR size for text-to-video;
    /// with first_frame/last_frame provide neither (the output ratio follows
    /// the frame image).
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    /// Explicit pixel dimensions as "WIDTHxHEIGHT" (e.g. "1280x720") - a
    /// text-to-video alternative to resolution+aspect_ratio. This is pixels, not
    /// a tier name (unlike `resolution` or generate_image's `image_size`).
    #[serde(default)]
    pub size: Option<String>,
    /// REQUIRED (no default): true to generate an audio track (for
    /// audio-capable models), false for silent video. This is the audio track
    /// baked into the clip.
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub with_audio: Option<bool>,
    /// Seed for reproducible-ish generation (provider support varies).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub seed: Option<u64>,
    /// Local image path used as the first frame (image-to-video). Adding a frame
    /// makes this image-to-video; every reference kind is then ignored.
    #[serde(default)]
    pub first_frame: Option<String>,
    /// Local image path used as the last frame (image-to-video).
    #[serde(default)]
    pub last_frame: Option<String>,
    /// Local image paths used as references (reference-to-video). Ignored, with a
    /// warning, when first_frame/last_frame are given (first_frame/last_frame win).
    #[serde(default)]
    pub reference_images: Vec<String>,
    /// Reference audio clips (models that honor them, e.g. Seedance gen 2+): each
    /// an https URL (fetched by the provider), a data: URL, or a local path
    /// (mp3/wav/flac/m4a/ogg/aac/webm, inlined as a data URL typed from its
    /// bytes or extension, 20 MiB each). Ignored, with a warning, when a frame
    /// is given.
    #[serde(default)]
    pub reference_audio: Vec<String>,
    /// Reference video clips: each an https URL, a data: URL, or a local path
    /// (mp4/webm/mov/mkv, inlined as a data URL typed from its bytes or
    /// extension, 20 MiB each). Ignored, with a warning, when a frame is given.
    #[serde(default)]
    pub reference_videos: Vec<String>,
    /// Upscaling models only: creativity level (integer; range is model-specific,
    /// see describe_model).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub creativity: Option<u32>,
    /// Upscaling models only: output scale factor, must be > 0 (e.g. 2 for 2x).
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub upscale_factor: Option<f64>,
    /// Provider block for this request: per-provider passthrough only, as
    /// {"options": {"<provider-slug>": {...}}}, sent opaque and unchanged;
    /// describe_model lists each endpoint's allowed_passthrough_parameters.
    /// OpenRouter's docs show BOTH {"google-vertex": {"negativePrompt": "..."}}
    /// and {"google-vertex": {"parameters": {"negativePrompt": "..."}}} - pass
    /// the shape your provider expects. Routing fields are ignored by this endpoint.
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ProviderOptionsArgs,
    /// Longest-side cap (px) for input frame/reference images (default 1536, max 4096).
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(max = 4096))]
    pub max_image_dimension: Option<u32>,
    /// Seconds to wait inline before returning a task_id (1-60, default 20).
    /// Video is slow, so the normal path returns "pending"; poll get_result.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(min = 1, max = 60))]
    pub wait_seconds: Option<u64>,
    /// Output file path (extension corrected to the returned format, e.g. .mp4).
    /// Optional: when omitted, an auto-named file is written under
    /// OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp).
    #[serde(default)]
    pub output: Option<String>,
}

/// Build the lean per-job result object for a video job: kind "video", the saved
/// clip paths and metadata, the manifest pointer, plus warnings/errors.
fn video_job_result_json(summary: &video_gen::VideoJobSummary) -> serde_json::Value {
    let videos: Vec<_> = summary
        .videos
        .iter()
        .map(|v| {
            json!({
                "path": v.path.to_string_lossy(),
                "duration": v.duration,
                "resolution": v.resolution,
                "aspect_ratio": v.aspect_ratio,
                "has_audio": v.has_audio,
                "mime": v.mime,
            })
        })
        .collect();
    let mut result = json!({
        "ok": true,
        "model": summary.model,
        "job_id": summary.job_id,
        "cost": summary.billing.cost,
        "kind": "video",
        "videos": videos,
        "manifest": summary.manifest_path.to_string_lossy(),
    });
    attach_warnings_errors(&mut result, &summary.warnings, &summary.errors);
    result
}

#[tool_router(router = video_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Generate a video with an OpenRouter video model (e.g. google/veo-3.1) and \
        save it to `output`. Video generation is slow (30s to several minutes) and runs \
        asynchronously: it almost always returns status \"pending\" with a task_id after \
        wait_seconds (default 20) - poll get_result until it is \"completed\". \
        For text-to-video, pass a prompt plus (aspect_ratio and/or \
        resolution) OR size. `resolution` is a named tier (480p/720p/768p/1080p/1K/2K/4K); \
        `size` is explicit pixels as \"WIDTHxHEIGHT\" (e.g. \"1280x720\") - an alternative to \
        resolution+aspect_ratio, not interchangeable with the tier vocabulary. For image-to-video, \
        pass first_frame (and optionally last_frame) as local image paths and provide neither \
        aspect_ratio nor size - the output ratio follows the frame image, and some models reject \
        a ratio outright in that mode; for reference-to-video pass reference_images (local paths), \
        reference_audio and/or reference_videos (https URLs or local files, inlined as data URLs) \
        - all references are ignored, with a warning, if a frame is given (first_frame/last_frame \
        win). No defaults for the required fields: model, duration and with_audio must all be \
        specified, or the call fails naming what is missing; `prompt` is required too unless a \
        frame or a reference stands in for it (image-only models take no text). resolution/size, \
        seed, frames, references, creativity, upscale_factor, provider, max_image_dimension, \
        wait_seconds, and output are all optional. `creativity` and `upscale_factor` (> 0) are for \
        upscaling models only. Provider-specific settings go in `provider.options` keyed by \
        provider slug and are sent opaque and unchanged: OpenRouter's docs show both \
        {\"google-vertex\": {\"negativePrompt\": \"...\"}} and \
        {\"google-vertex\": {\"parameters\": {\"negativePrompt\": \"...\"}}}, so pass the shape \
        your provider expects; describe_model lists each endpoint's allowed_passthrough_parameters. \
        `duration` is the clip's length in seconds (model-specific discrete values - see \
        describe_model), not how long generation takes. `with_audio` controls the audio track \
        baked into the clip. `output` is optional - omit it for an auto-named file under \
        OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp). The completed result \
        carries the saved file path in JSON plus a file:// ResourceLink per clip when previews \
        are enabled. Links preserve the media type and require client access to the server \
        filesystem.",
        annotations(
            title = "Generate Video",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn generate_video(
        &self,
        Parameters(args): Parameters<GenerateVideoArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let inline = client_wants_inline_previews(&context);
        self.run_generate_video(args, inline).await
    }

    /// Core of `generate_video`, parameterized on inline media like
    /// [`Self::run_generate`], so tests can drive it without a `RequestContext`.
    pub(crate) async fn run_generate_video(
        &self,
        args: GenerateVideoArgs,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        // No defaults: the agent must choose these explicitly.
        let mut missing: Vec<&str> = Vec::new();
        if args.duration.is_none() {
            missing.push("duration (seconds)");
        }
        // Only demanded for text-to-video. With a first/last frame the output
        // ratio comes from the frame image, and some models reject any ratio at
        // all in that mode: bytedance/seedance-2.5 answers a first_frame call
        // carrying aspect_ratio *or* size with
        // 400 InvalidParameter.TaskTypeConstraint. Requiring one here made
        // image-to-video impossible on that model - every allowed call was
        // rejected by one side or the other.
        let has_frame = args.first_frame.is_some() || args.last_frame.is_some();
        if !has_frame && args.aspect_ratio.is_none() && args.size.is_none() {
            missing.push("aspect_ratio (e.g. \"16:9\", \"9:16\") or size (\"WIDTHxHEIGHT\")");
        }
        if args.with_audio.is_none() {
            missing.push("with_audio (true for an audio track, false for silent video)");
        }
        require_all("generate_video", "video", &missing)?;
        let provider = args.provider.into_options()?;

        let mut frames = Vec::new();
        if let Some(p) = &args.first_frame {
            frames.push(VideoInput {
                path: p.into(),
                frame_type: "first_frame".to_string(),
            });
        }
        if let Some(p) = &args.last_frame {
            frames.push(VideoInput {
                path: p.into(),
                frame_type: "last_frame".to_string(),
            });
        }
        let req = VideoGenRequest {
            model: args.model.clone(),
            prompt: args.prompt,
            duration: args.duration,
            resolution: args.resolution,
            aspect_ratio: args.aspect_ratio,
            size: args.size,
            generate_audio: args.with_audio,
            seed: args.seed,
            frames,
            references: args.reference_images.iter().map(PathBuf::from).collect(),
            reference_audio: args.reference_audio,
            reference_videos: args.reference_videos,
            creativity: args.creativity,
            upscale_factor: args.upscale_factor,
            provider,
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
            poll_interval_secs: video_gen::resolve_poll_interval(),
            poll_timeout_secs: video_gen::resolve_poll_timeout(),
        };
        // Conditional prompt and upscale_factor > 0: the same check run_job
        // makes, surfaced here as invalid params before a job is spawned.
        req.validate()
            .map_err(|e| ErrorData::invalid_params(format!("generate_video: {e:#}"), None))?;

        let wait = args
            .wait_seconds
            .unwrap_or(DEFAULT_VIDEO_WAIT_SECONDS)
            .clamp(1, 60);
        let mut config: Vec<String> = Vec::new();
        if let Some(a) = &req.aspect_ratio {
            config.push(a.clone());
        } else if let Some(s) = &req.size {
            config.push(s.clone());
        }
        if let Some(r) = &req.resolution {
            config.push(r.clone());
        }
        if let Some(d) = req.duration {
            config.push(format!("{d}s"));
        }
        let config_refs: Vec<&str> = config.iter().map(String::as_str).collect();
        let base = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Video,
            &req.model,
            &config_refs,
            req.seed,
        );
        let model = args.model;

        self.spawn_job_and_wait(
            TaskKind::Video,
            wait,
            inline_previews,
            move |ctx| async move {
                match video_gen::run_job(&ctx.client, &req, &base, "inline").await {
                    Ok(summary) => {
                        ctx.stats
                            .record_video(
                                &model,
                                summary.videos.len() as u64,
                                Some(&summary.billing),
                            )
                            .await;
                        if summary.videos.is_empty() {
                            return Err(format!(
                                "video job {} failed: {}; recovery manifest: {}",
                                summary.job_id,
                                summary.errors.join("; "),
                                summary.manifest_path.display()
                            ));
                        }
                        Ok(video_job_result_json(&summary))
                    }
                    Err(e) => {
                        ctx.stats
                            .record_video(&model, 0, crate::billing::Receipt::from_error(&e))
                            .await;
                        Err(format!("{e:#}"))
                    }
                }
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn generate_video_requires_explicit_parameters() {
        // Validation runs before any HTTP call, so the base URL is never used.
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = GenerateVideoArgs {
            model: "m".to_string(),
            prompt: Some("p".to_string()),
            duration: None,
            resolution: None,
            aspect_ratio: None,
            size: None,
            with_audio: None,
            seed: None,
            first_frame: None,
            last_frame: None,
            reference_images: vec![],
            reference_audio: vec![],
            reference_videos: vec![],
            creativity: None,
            upscale_factor: None,
            provider: Default::default(),
            max_image_dimension: None,
            wait_seconds: None,
            output: Some("out.mp4".to_string()),
        };
        let err = server.run_generate_video(args, false).await.unwrap_err();
        assert!(err.message.contains("duration"));
        assert!(err.message.contains("aspect_ratio"));
        assert!(err.message.contains("with_audio"));
        assert!(err.message.contains("no defaults"));
    }

    /// With a frame image the output ratio follows that image, and
    /// bytedance/seedance-2.5 rejects a request carrying aspect_ratio OR size
    /// with 400 InvalidParameter.TaskTypeConstraint. Demanding one here left no
    /// valid call: ratio -> upstream 400, no ratio -> our own error.
    #[tokio::test]
    async fn generate_video_does_not_require_a_ratio_when_a_frame_is_given() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let base = |first_frame: Option<String>| GenerateVideoArgs {
            model: "bytedance/seedance-2.5".to_string(),
            prompt: Some("a slow zoom".to_string()),
            duration: Some(8),
            resolution: Some("720p".to_string()),
            aspect_ratio: None,
            size: None,
            with_audio: Some(true),
            seed: None,
            first_frame,
            last_frame: None,
            reference_images: vec![],
            reference_audio: vec![],
            reference_videos: vec![],
            creativity: None,
            upscale_factor: None,
            provider: Default::default(),
            max_image_dimension: None,
            wait_seconds: None,
            output: Some("out.mp4".to_string()),
        };

        // No frame: still required, so validation stops the call before any HTTP.
        let err = server
            .run_generate_video(base(None), false)
            .await
            .unwrap_err();
        assert!(err.message.contains("aspect_ratio"), "{}", err.message);

        // With a frame: validation passes, so the call is accepted and the job
        // runs. It then fails reading the missing frame file, which is proof it
        // got past the argument check rather than being rejected by it.
        let res = server
            .run_generate_video(base(Some("/nonexistent.png".to_string())), false)
            .await
            .expect("a frame call must not be rejected for a missing ratio");
        let v = tool_result_json(&res);
        let text = v.to_string();
        assert!(!text.contains("aspect_ratio"), "{text}");
        assert!(!text.contains("no defaults"), "{text}");
        assert!(text.contains("/nonexistent.png"), "{text}");
    }

    #[tokio::test]
    async fn unreadable_accepted_video_submission_keeps_unknown_billing_without_a_job_id() {
        for status in [202, 400] {
            let mock = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/videos"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("x-generation-id", "gen-accepted-video")
                        .set_body_string("{"),
                )
                .expect(1)
                .mount(&mock)
                .await;
            let server = server_for(mock.uri());
            let dir = tempfile::tempdir().unwrap();
            let out = dir.path().join("clip.mp4");
            let result = server
                .run_generate_video(
                    GenerateVideoArgs {
                        model: "test/video".into(),
                        prompt: Some("a kite".into()),
                        duration: Some(4),
                        resolution: None,
                        aspect_ratio: Some("16:9".into()),
                        size: None,
                        with_audio: Some(false),
                        seed: None,
                        first_frame: None,
                        last_frame: None,
                        reference_images: vec![],
                        reference_audio: vec![],
                        reference_videos: vec![],
                        creativity: None,
                        upscale_factor: None,
                        provider: Default::default(),
                        max_image_dimension: None,
                        wait_seconds: Some(1),
                        output: Some(out.to_string_lossy().into_owned()),
                    },
                    false,
                )
                .await
                .unwrap();
            let envelope = tool_result_json(&result);
            assert_eq!(envelope["status"], "failed");
            assert!(envelope.get("job_id").is_none());
            let stats = server.stats.snapshot().await;
            assert_eq!(stats["requests_total"], 1);
            assert_eq!(stats["requests_failed"], 1);
            assert_eq!(stats["unknown_cost_count"], u64::from(status == 202));
            assert_eq!(
                stats["by_model"]["test/video"]["unknown_cost_count"],
                u64::from(status == 202)
            );
            assert!(!out.exists());
            assert!(!crate::manifest::path(&out).exists());
            assert_eq!(mock.received_requests().await.unwrap().len(), 1);
            if status == 202 {
                assert!(
                    envelope["error"]
                        .as_str()
                        .unwrap()
                        .contains("gen-accepted-video")
                );
            }
        }
    }

    #[tokio::test]
    async fn generate_video_returns_pending_with_a_task_id() {
        // The submit succeeds but the poll keeps reporting "processing", so the
        // short wait window elapses and the tool returns a pending task to poll.
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .respond_with(ResponseTemplate::new(202).set_body_json(json!({ "id": "vid-pending" })))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/videos/vid-pending"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-pending",
                "status": "processing",
                "unsigned_urls": []
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-video-pending/clip.mp4");
        let args = GenerateVideoArgs {
            model: "google/veo-3.1".to_string(),
            prompt: Some("a kite".to_string()),
            duration: Some(4),
            resolution: None,
            aspect_ratio: Some("16:9".to_string()),
            size: None,
            with_audio: Some(false),
            seed: None,
            first_frame: None,
            last_frame: None,
            reference_images: vec![],
            reference_audio: vec![],
            reference_videos: vec![],
            creativity: None,
            upscale_factor: None,
            provider: Default::default(),
            max_image_dimension: None,
            wait_seconds: Some(1), // clamp floor: return quickly as pending
            output: Some(out.to_string_lossy().into_owned()),
        };
        let res = server.run_generate_video(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["status"], "pending");
        assert_eq!(v["kind"], "video");
        assert!(v["task_id"].is_string());
    }

    /// Text-to-video args that pass the unconditional checks; each test tweaks
    /// the field under test.
    fn valid_args(out: &std::path::Path) -> GenerateVideoArgs {
        GenerateVideoArgs {
            model: "test/video".to_string(),
            prompt: Some("a kite".to_string()),
            duration: Some(4),
            resolution: None,
            aspect_ratio: Some("16:9".to_string()),
            size: None,
            with_audio: Some(false),
            seed: None,
            first_frame: None,
            last_frame: None,
            reference_images: vec![],
            reference_audio: vec![],
            reference_videos: vec![],
            creativity: None,
            upscale_factor: None,
            provider: Default::default(),
            max_image_dimension: None,
            wait_seconds: Some(1),
            output: Some(out.to_string_lossy().into_owned()),
        }
    }

    /// `prompt` is conditional, like aspect_ratio: image-only models take none,
    /// so it is required only when no frame and no reference of any kind is
    /// present. Without any input it is rejected before HTTP; with a reference
    /// URL it is accepted and the job runs.
    #[tokio::test]
    async fn generate_video_requires_a_prompt_only_without_a_frame_or_reference() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("clip.mp4");

        let no_prompt = GenerateVideoArgs {
            prompt: None,
            ..valid_args(&out)
        };
        let err = server
            .run_generate_video(no_prompt, false)
            .await
            .unwrap_err();
        assert!(err.message.contains("prompt"), "{}", err.message);
        // A blank prompt counts as missing.
        let blank = GenerateVideoArgs {
            prompt: Some("   ".to_string()),
            ..valid_args(&out)
        };
        let err = server.run_generate_video(blank, false).await.unwrap_err();
        assert!(err.message.contains("prompt"), "{}", err.message);

        // With a reference the call is accepted; the job then fails on the
        // unreachable server, which proves it got past the argument check.
        let with_ref = GenerateVideoArgs {
            prompt: None,
            reference_audio: vec!["https://cdn/song.mp3".to_string()],
            ..valid_args(&out)
        };
        let res = server
            .run_generate_video(with_ref, false)
            .await
            .expect("a reference call must not be rejected for a missing prompt");
        let text = tool_result_json(&res).to_string();
        assert!(!text.contains("prompt"), "{text}");
    }

    #[tokio::test]
    async fn generate_video_rejects_a_non_positive_upscale_factor() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("clip.mp4");
        for bad in [0.0, -2.0] {
            let args = GenerateVideoArgs {
                upscale_factor: Some(bad),
                ..valid_args(&out)
            };
            let err = server.run_generate_video(args, false).await.unwrap_err();
            assert!(err.message.contains("upscale_factor"), "{}", err.message);
        }
    }

    /// The shared provider fixture reaches POST /videos as `provider.options`,
    /// alongside creativity/upscale_factor; an invalid block is rejected before
    /// any HTTP call.
    #[tokio::test]
    async fn generate_video_forwards_provider_options_and_rejects_bad_ones() {
        let (provider, expected) = crate::server::test_support::provider_options_fixture();
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .and(wiremock::matchers::body_partial_json(json!({
                "provider": expected,
                "creativity": 2,
                "upscale_factor": 2.0
            })))
            .respond_with(ResponseTemplate::new(202).set_body_json(json!({ "id": "vid-prov" })))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/videos/vid-prov"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-prov",
                "status": "processing",
                "unsigned_urls": []
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("clip.mp4");
        let args = GenerateVideoArgs {
            provider,
            creativity: Some(2),
            upscale_factor: Some(2.0),
            ..valid_args(&out)
        };
        let res = server.run_generate_video(args, false).await.unwrap();
        assert_eq!(tool_result_json(&res)["status"], "pending");
        let posts = mock
            .received_requests()
            .await
            .unwrap()
            .into_iter()
            .filter(|r| r.method == wiremock::http::Method::POST)
            .count();
        assert_eq!(posts, 1, "the matching submission happened");

        let mut options = std::collections::BTreeMap::new();
        options.insert("acme".to_string(), json!("not-an-object"));
        let bad = GenerateVideoArgs {
            provider: crate::server::provider::ProviderOptionsArgs { options },
            ..valid_args(&out)
        };
        let err = server.run_generate_video(bad, false).await.unwrap_err();
        assert!(err.message.contains("acme"), "{}", err.message);
    }
}
