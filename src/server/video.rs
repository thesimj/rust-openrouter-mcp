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
use crate::server::result::{
    DEFAULT_VIDEO_WAIT_SECONDS, attach_warnings_errors, client_wants_inline_previews,
};
use crate::server::schema::{de_opt_bool, de_opt_uint, require_all, scalarize_nullable};
use crate::tasks::TaskKind;
use crate::video_gen::{self, VideoGenRequest, VideoInput};

use super::OpenRouterServer;

/// Arguments for the `generate_video` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct GenerateVideoArgs {
    /// Video model id, e.g. "google/veo-3.1". Use list_models with
    /// output_modalities="video" to discover them.
    pub model: String,
    /// Prompt text describing the video to generate.
    pub prompt: String,
    /// REQUIRED (no default): clip duration in seconds.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub duration: Option<u32>,
    /// Resolution, e.g. "480p", "720p", "1080p", "1K", "2K", "4K"
    /// (interchangeable with `size`).
    #[serde(default)]
    pub resolution: Option<String>,
    /// REQUIRED (no default unless `size` is given): aspect ratio,
    /// e.g. "16:9", "9:16", "1:1".
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    /// "WIDTHxHEIGHT" (interchangeable with resolution + aspect_ratio).
    #[serde(default)]
    pub size: Option<String>,
    /// REQUIRED (no default): true to generate an audio track (for
    /// audio-capable models), false for silent video.
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub generate_audio: Option<bool>,
    /// Seed for reproducible-ish generation (provider support varies).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub seed: Option<u64>,
    /// Local image path used as the first frame (image-to-video). Adding a frame
    /// makes this image-to-video; reference_images are then ignored.
    #[serde(default)]
    pub first_frame: Option<String>,
    /// Local image path used as the last frame (image-to-video).
    #[serde(default)]
    pub last_frame: Option<String>,
    /// Local image paths used as references (reference-to-video). Ignored, with a
    /// warning, when first_frame/last_frame are given (frame_images wins).
    #[serde(default)]
    pub reference_images: Vec<String>,
    /// Longest-side cap (px) for input frame/reference images (default 800).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_image_dimension: Option<u32>,
    /// Seconds to wait inline before returning a task_id (1-60, default 20).
    /// Video is slow, so the normal path returns "pending"; poll get_result.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(min = 1, max = 60))]
    pub wait_seconds: Option<u64>,
    /// Output file path (extension corrected to the returned format, e.g. .mp4).
    pub output: String,
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
        save it to `output`. For text-to-video, pass a prompt. For image-to-video, also pass \
        first_frame (and optionally last_frame) as local image paths; for reference-to-video pass \
        reference_images (ignored, with a warning, if a frame is given - frames win). This tool \
        has NO defaults: model, prompt, output, duration, generate_audio, and an aspect_ratio \
        OR size must all be specified, or the call fails naming what is missing. Video generation \
        is slow (30s to several minutes): it runs asynchronously and almost always returns status \
        \"pending\" with a task_id after wait_seconds (default 20) - poll get_result until it is \
        \"completed\". The completed result carries the saved file path in JSON plus, for \
        sandboxed clients, a file:// ResourceLink (mime video/mp4) per clip.",
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
        if args.aspect_ratio.is_none() && args.size.is_none() {
            missing.push("aspect_ratio (e.g. \"16:9\", \"9:16\") or size (\"WIDTHxHEIGHT\")");
        }
        if args.generate_audio.is_none() {
            missing.push("generate_audio (true for an audio track, false for silent video)");
        }
        require_all("generate_video", "video", &missing)?;

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
            generate_audio: args.generate_audio,
            seed: args.seed,
            frames,
            references: args.reference_images.iter().map(PathBuf::from).collect(),
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
            poll_interval_secs: video_gen::resolve_poll_interval(None),
            poll_timeout_secs: video_gen::resolve_poll_timeout(None),
        };

        let wait = args
            .wait_seconds
            .unwrap_or(DEFAULT_VIDEO_WAIT_SECONDS)
            .clamp(1, 60);
        let base = PathBuf::from(&args.output);
        let model = args.model;

        self.spawn_job_and_wait(
            TaskKind::Video,
            wait,
            inline_previews,
            move |ctx| async move {
                match video_gen::run_job(&ctx.client, &req, &base, "inline").await {
                    Ok(summary) if !summary.videos.is_empty() => {
                        let costs: Vec<f64> =
                            summary.videos.iter().filter_map(|v| v.cost).collect();
                        let cost = (!costs.is_empty()).then(|| costs.iter().sum());
                        ctx.stats.record_video(&model, true, cost).await;
                        Ok(video_job_result_json(&summary))
                    }
                    Ok(summary) => {
                        ctx.stats.record_video(&model, false, None).await;
                        Err(format!(
                            "video generation failed: {}",
                            summary.errors.join("; ")
                        ))
                    }
                    Err(e) => {
                        ctx.stats.record_video(&model, false, None).await;
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
            prompt: "p".to_string(),
            duration: None,
            resolution: None,
            aspect_ratio: None,
            size: None,
            generate_audio: None,
            seed: None,
            first_frame: None,
            last_frame: None,
            reference_images: vec![],
            max_image_dimension: None,
            wait_seconds: None,
            output: "out.mp4".to_string(),
        };
        let err = server.run_generate_video(args, false).await.unwrap_err();
        assert!(err.message.contains("duration"));
        assert!(err.message.contains("aspect_ratio"));
        assert!(err.message.contains("generate_audio"));
        assert!(err.message.contains("no defaults"));
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
            prompt: "a kite".to_string(),
            duration: Some(4),
            resolution: None,
            aspect_ratio: Some("16:9".to_string()),
            size: None,
            generate_audio: Some(false),
            seed: None,
            first_frame: None,
            last_frame: None,
            reference_images: vec![],
            max_image_dimension: None,
            wait_seconds: Some(1), // clamp floor: return quickly as pending
            output: out.to_string_lossy().into_owned(),
        };
        let res = server.run_generate_video(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["status"], "pending");
        assert_eq!(v["kind"], "video");
        assert!(v["task_id"].is_string());
    }
}
