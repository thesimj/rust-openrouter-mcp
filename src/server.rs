//! The rmcp stdio MCP server and its tools.

use base64::Engine;
use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        AnnotateAble, CallToolResult, Content, RawAudioContent, RawContent, RawResource,
        ServerCapabilities, ServerInfo,
    },
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::Deserialize;

use std::path::PathBuf;

use serde_json::json;

use crate::audio_gen::{self, SpeechGenRequest};
use crate::image_gen::{self, GenerateRequest};
use crate::openrouter::{ModelsQuery, OpenRouterClient, apply_filters};
use crate::stats::UsageStats;
use crate::tasks::{TaskKind, TaskRegistry, TaskSnapshot};
use crate::video_gen::{self, VideoGenRequest, VideoInput};

/// MCP server wrapping an [`OpenRouterClient`].
#[derive(Clone)]
pub struct OpenRouterServer {
    client: OpenRouterClient,
    tasks: TaskRegistry,
    stats: UsageStats,
    tool_router: ToolRouter<Self>,
}

impl OpenRouterServer {
    pub fn new(client: OpenRouterClient) -> Self {
        Self {
            client,
            tasks: TaskRegistry::new(),
            stats: UsageStats::new(),
            tool_router: Self::tool_router(),
        }
    }
}

/// Default seconds to wait inline before returning a task id for a slow job.
const DEFAULT_WAIT_SECONDS: u64 = 10;

/// Default inline wait for video: video takes 30s-several minutes, so the
/// fast-return window almost always yields `pending` and the caller polls
/// get_result. Kept within the 1-60 clamp.
const DEFAULT_VIDEO_WAIT_SECONDS: u64 = 20;

/// Build the lean per-job result object for an image job (paths, dims, requested
/// vs actual, manifest pointer, plus warnings/errors when present).
fn image_job_result_json(
    summary: &image_gen::JobSummary,
    aspect_ratio: &Option<String>,
    image_size: &Option<String>,
) -> serde_json::Value {
    let images: Vec<_> = summary
        .images
        .iter()
        .map(|img| {
            json!({
                "path": img.path.to_string_lossy(),
                "seed": img.seed,
                "width": img.width,
                "height": img.height,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "actual_aspect_ratio": img.actual_aspect_ratio,
                "actual_image_size": img.actual_image_size,
            })
        })
        .collect();
    let mut result = json!({
        "ok": true,
        "model": summary.model,
        "images": images,
        "manifest": summary.manifest_path.to_string_lossy(),
    });
    if !summary.warnings.is_empty() {
        result["warnings"] = json!(summary.warnings);
    }
    if !summary.errors.is_empty() {
        result["errors"] = json!(summary.errors);
    }
    result
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
    if !summary.warnings.is_empty() {
        result["warnings"] = json!(summary.warnings);
    }
    if !summary.errors.is_empty() {
        result["errors"] = json!(summary.errors);
    }
    result
}

/// Longest-side cap (px) for the inline preview embedded in a tool result. The
/// full-resolution image always stays on disk; this only bounds the base64 copy
/// sent to the client so generated images render without bloating context.
/// ~1568px is Claude's image sweet spot (larger is downsampled client-side).
const PREVIEW_MAX_SIDE: u32 = 1568;

/// Most inline previews embedded in a single tool result. Many-variant jobs can
/// produce up to 16 images; base64-embedding every one would bloat the client's
/// context, so any beyond this cap are reported by path in the JSON block only.
const MAX_INLINE_PREVIEWS: usize = 4;

/// Most inline media blocks (audio / video ResourceLinks) attached to a result.
const MAX_INLINE_MEDIA: usize = 4;

/// Largest audio file embedded inline as a base64 AudioContent block. Larger
/// files bloat the client's context badly, so they are reported by path only
/// (the file is always saved to disk regardless).
const MAX_INLINE_AUDIO_BYTES: u64 = 4 * 1024 * 1024;

/// Collect the on-disk paths of the generated images in a job envelope.
fn envelope_image_paths(env: &serde_json::Value) -> Vec<String> {
    env.get("images")
        .and_then(|v| v.as_array())
        .map(|images| {
            images
                .iter()
                .filter_map(|img| img.get("path").and_then(|p| p.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Collect the on-disk paths of the generated clips in a video job envelope.
fn envelope_video_paths(env: &serde_json::Value) -> Vec<String> {
    env.get("videos")
        .and_then(|v| v.as_array())
        .map(|videos| {
            videos
                .iter()
                .filter_map(|v| v.get("path").and_then(|p| p.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Build a `ResourceLink` content block for each generated clip path. rmcp 1.7
/// has no native video content block, so a sandboxed client gets a `file://`
/// ResourceLink (mime video/mp4, size from the file) rather than an embedded
/// blob; the path is also in the JSON text block. Capped at [`MAX_INLINE_MEDIA`].
///
/// Blocking: does a filesystem stat per path - run via `spawn_blocking`.
fn video_resource_link_blocks(paths: &[String]) -> Vec<Content> {
    paths
        .iter()
        .take(MAX_INLINE_MEDIA)
        .map(|path| {
            let name = std::path::Path::new(path)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.clone());
            let mut r = RawResource::new(format!("file://{path}"), name);
            r.mime_type = Some("video/mp4".to_string());
            r.size = std::fs::metadata(path).ok().map(|m| m.len() as u32);
            Content::resource_link(r)
        })
        .collect()
}

/// True when `bytes` is a PNG whose longest side already fits `max_side`, so it
/// can be sent inline verbatim without a decode/resize/re-encode round-trip.
/// [`crate::image_io::decode_dimensions`] only reads the header, so this is cheap.
fn is_png_within_bound(bytes: &[u8], max_side: u32) -> bool {
    const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    bytes.starts_with(&PNG_MAGIC)
        && crate::image_io::decode_dimensions(bytes)
            .map(|(w, h)| w <= max_side && h <= max_side)
            .unwrap_or(false)
}

/// Turn the generated images' paths into inline image content blocks so the
/// client (e.g. Claude Desktop) renders them, not just their paths. Reads each
/// path from disk, downscales to [`PREVIEW_MAX_SIDE`] (or passes an already-small
/// PNG through untouched), and base64-encodes it as a PNG `image` block. Caps the
/// count at [`MAX_INLINE_PREVIEWS`]; unreadable/undecodable files are skipped (the
/// JSON text block still reports their paths), so this never fails the call.
///
/// Blocking: does disk I/O and image decode/encode - run it via `spawn_blocking`,
/// never directly on the async runtime.
fn encode_preview_blocks(paths: &[String]) -> Vec<Content> {
    paths
        .iter()
        .take(MAX_INLINE_PREVIEWS)
        .filter_map(|path| {
            let bytes = std::fs::read(path).ok()?;
            let png = if is_png_within_bound(&bytes, PREVIEW_MAX_SIDE) {
                bytes
            } else {
                crate::image_io::normalize_to_png(&bytes, PREVIEW_MAX_SIDE).ok()?
            };
            let b64 = base64::engine::general_purpose::STANDARD.encode(png);
            Some(Content::image(b64, "image/png".to_string()))
        })
        .collect()
}

/// Decide whether to embed inline image previews for the connected client.
///
/// Why this is client-dependent: a local CLI (Claude Code) shares the
/// filesystem, so a returned path *is* the image - inline base64 only bloats
/// context. Claude Desktop, by contrast, runs the MCP server in a sandbox whose
/// filesystem the app can't read, so a path is useless and the bytes must be
/// returned inline or the image is stranded.
///
/// `OPENROUTER_MCP_IMAGE_PREVIEWS` overrides detection: `always` / `never`
/// (anything else, or unset, means `auto`). The `.mcpb` connector sets `always`.
/// In `auto` we return previews for every client *except* the local-filesystem
/// CLI (`claude-code`), so the failure-prone case (Desktop) is covered by default.
fn client_wants_inline_previews(ctx: &RequestContext<RoleServer>) -> bool {
    match std::env::var("OPENROUTER_MCP_IMAGE_PREVIEWS")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("always") => return true,
        Some("never") => return false,
        _ => {}
    }
    let client = ctx
        .peer
        .peer_info()
        .map(|info| info.client_info.name.to_ascii_lowercase())
        .unwrap_or_default();
    !client.contains("claude-code")
}

/// Build the full tool result for a job envelope: the JSON metadata as a text
/// block, followed (when `inline_previews` and the job completed) by an inline
/// image preview of each generated image.
async fn job_call_result(
    env: &serde_json::Value,
    inline_previews: bool,
) -> Result<CallToolResult, ErrorData> {
    let body = serde_json::to_string_pretty(env)
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
    let mut blocks = vec![Content::text(body)];

    if inline_previews && env.get("status").and_then(|s| s.as_str()) == Some("completed") {
        match env.get("kind").and_then(|k| k.as_str()) {
            Some("video") => {
                let paths = envelope_video_paths(env);
                // A filesystem stat per clip is blocking I/O; keep it off the worker.
                let links = tokio::task::spawn_blocking(move || video_resource_link_blocks(&paths))
                    .await
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                blocks.extend(links);
            }
            // "image" (and the historical default) get inline image previews.
            _ => {
                let paths = envelope_image_paths(env);
                let total = paths.len();
                // Reading + decoding + resizing + re-encoding images is blocking CPU
                // and disk I/O; run it off the async worker so concurrent tool calls
                // (e.g. get_result polls) aren't stalled behind it.
                let previews = tokio::task::spawn_blocking(move || encode_preview_blocks(&paths))
                    .await
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                let shown = previews.len();
                blocks.extend(previews);
                if total > MAX_INLINE_PREVIEWS {
                    blocks.push(Content::text(format!(
                        "note: showing {shown} of {total} generated images inline (capped at \
                         {MAX_INLINE_PREVIEWS}); every image is saved to disk at the paths above."
                    )));
                }
            }
        }
    }
    Ok(CallToolResult::success(blocks))
}

/// Wrap a task snapshot into the response envelope returned by `generate_image`
/// (fast path) and `get_result`: the completed result, an error, or a pending
/// note - always carrying `task_id`, `status`, and `kind`.
fn snapshot_to_envelope(task_id: &str, snap: &TaskSnapshot) -> serde_json::Value {
    let mut env = match snap.status {
        "completed" => snap.result.clone().unwrap_or_else(|| json!({ "ok": true })),
        "failed" => json!({ "ok": false, "error": snap.error }),
        _ => json!({
            "ok": true,
            "message": format!("still generating - call get_result with task_id \"{task_id}\""),
        }),
    };
    env["task_id"] = json!(task_id);
    env["status"] = json!(snap.status);
    env["kind"] = json!(snap.kind);
    env
}

/// Arguments for the `list_models` tool. These map to OpenRouter's server-side
/// `GET /api/v1/models` query parameters, so filtering happens at the API.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct ListModelsArgs {
    /// Server-side free-text search by model name or slug (e.g. "claude").
    #[serde(default)]
    pub query: Option<String>,
    /// Local case-insensitive filter across id, name, and description
    /// (e.g. "openai"). Applied after the server-side query.
    #[serde(default)]
    pub search: Option<String>,
    /// Filter by output modalities. Comma-separated list of: text, image, audio,
    /// embeddings, video, rerank, speech, transcription - or "all". Defaults to
    /// text on the API when omitted (so pass "all" or a value to see others).
    #[serde(default)]
    pub output_modalities: Option<String>,
    /// Filter by input modalities. Comma-separated list of: text, image, audio, file.
    #[serde(default)]
    pub input_modalities: Option<String>,
    /// Only return models supporting these API parameters. Comma-separated,
    /// e.g. "tools", "structured_outputs", "reasoning".
    #[serde(default)]
    pub supported_parameters: Option<String>,
    /// Sort order: pricing-low-to-high, pricing-high-to-low, context-high-to-low,
    /// throughput-high-to-low, latency-low-to-high, most-popular, top-weekly, newest.
    /// Defaults to "top-weekly" (most used this week) when omitted.
    #[serde(default)]
    pub sort: Option<String>,
    /// Minimum context length in tokens; models with less are excluded.
    #[serde(default)]
    pub min_context: Option<u64>,
    /// Return all matching models. By default only the first 20 are returned to
    /// keep the result compact; set true to get the complete list.
    #[serde(default)]
    pub all: bool,
}

/// A local input image for editing / image-to-image. Order is preserved.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImageInput {
    /// Local file path (png/jpeg/webp/gif).
    pub path: String,
    /// Optional label, surfaced to the model as a reference name.
    #[serde(default)]
    pub label: Option<String>,
}

/// Arguments for the `generate_image` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateImageArgs {
    /// Image model id, e.g. "google/gemini-3.1-flash-image-preview".
    pub model: String,
    /// Prompt text describing the image to generate (or the edit to apply).
    pub prompt: String,
    /// REQUIRED (no default): aspect ratio, e.g. "1:1", "16:9", "9:16"
    /// (maps to image_config.aspect_ratio).
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    /// REQUIRED (no default): resolution tier, e.g. "1K", "2K", "4K"
    /// (maps to image_config.image_size).
    #[serde(default)]
    pub image_size: Option<String>,
    /// Seed for reproducible-ish generation (provider support varies).
    #[serde(default)]
    pub seed: Option<u64>,
    /// REQUIRED (no default): true for image-only-output models
    /// (e.g. Grok/Seedream/FLUX), false for dual text+image models
    /// (e.g. Nano Banana, GPT Image).
    #[serde(default)]
    pub image_only: Option<bool>,
    /// Local input images to edit/condition on (image-to-image / multi-image).
    /// Provide them to edit existing images; omit for plain text-to-image.
    #[serde(default)]
    pub images: Vec<ImageInput>,
    /// Longest-side cap (px) for input images before sending (default 800;
    /// env OPENROUTER_IMAGE_MAX_DIMENSION).
    #[serde(default)]
    pub max_image_dimension: Option<u32>,
    /// Number of variants to generate in parallel (1-16, seed-stepped). Default 1.
    /// With >1, files are named <output>-var-001, -002, ... and one manifest covers all.
    #[serde(default)]
    #[schemars(range(min = 1, max = 16))]
    pub variants: Option<usize>,
    /// Seconds to wait inline before returning a task_id for a slow job (1-60,
    /// default 10). The job keeps running; fetch it later with get_result.
    #[serde(default)]
    #[schemars(range(min = 1, max = 60))]
    pub wait_seconds: Option<u64>,
    /// Output file path (single image, or the base name for variants). The
    /// extension is corrected to the actual returned format.
    pub output: String,
}

/// Arguments for the `describe_image` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DescribeImageArgs {
    /// Vision-capable model id (image input, text output), e.g.
    /// "google/gemini-2.5-flash" or "anthropic/claude-sonnet-4.6".
    pub model: String,
    /// Local image path(s) to describe (at least one required).
    pub images: Vec<ImageInput>,
    /// Instruction or question about the image(s). Defaults to a detailed description.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Longest-side cap (px) for input images before sending (default 800).
    #[serde(default)]
    pub max_image_dimension: Option<u32>,
}

/// Arguments for the `generate_video` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateVideoArgs {
    /// Video model id, e.g. "google/veo-3.1". Use list_models with
    /// output_modalities="video" to discover them.
    pub model: String,
    /// Prompt text describing the video to generate.
    pub prompt: String,
    /// REQUIRED (no default): clip duration in seconds.
    #[serde(default)]
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
    #[serde(default)]
    pub generate_audio: Option<bool>,
    /// Seed for reproducible-ish generation (provider support varies).
    #[serde(default)]
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
    #[serde(default)]
    pub max_image_dimension: Option<u32>,
    /// Seconds to wait inline before returning a task_id (1-60, default 20).
    /// Video is slow, so the normal path returns "pending"; poll get_result.
    #[serde(default)]
    #[schemars(range(min = 1, max = 60))]
    pub wait_seconds: Option<u64>,
    /// Output file path (extension corrected to the returned format, e.g. .mp4).
    pub output: String,
}

/// Arguments for the `generate_audio` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateAudioArgs {
    /// TTS model id, e.g. "openai/gpt-4o-mini-tts" or "hexgrad/kokoro-82m".
    pub model: String,
    /// REQUIRED (no default): the text to synthesize.
    #[serde(default)]
    pub input: Option<String>,
    /// REQUIRED (no default): voice id (varies by model, e.g. "alloy").
    #[serde(default)]
    pub voice: Option<String>,
    /// Output audio format: "mp3" (default) or "pcm".
    #[serde(default)]
    pub response_format: Option<String>,
    /// Playback speed (select models only).
    #[serde(default)]
    pub speed: Option<f64>,
    /// Output file path (extension corrected to the returned format, e.g. .mp3).
    pub output: String,
}

/// Arguments for the `get_result` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetResultArgs {
    /// The task_id returned by generate_image (or a future generate_video).
    pub task_id: String,
}

/// Arguments for the `reset_usage_stats` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResetUsageStatsArgs {
    /// Must be true to confirm - this clears all in-memory usage counters.
    #[serde(default)]
    pub confirm: bool,
}

#[tool_router]
impl OpenRouterServer {
    #[tool(
        description = "List available OpenRouter models with their capabilities \
        (input/output modalities, context length) and pricing. Filtering and sorting \
        happen server-side: search by name (query), filter by output/input modalities \
        or supported parameters, sort by newest/most-popular/pricing/context, and set a \
        minimum context length. Output modalities include text, image, audio, embeddings, \
        video, rerank, speech, transcription (default is text only - pass \
        output_modalities=\"all\" or a specific value to see the rest). Returns the \
        first 20 models by default; set all=true for the complete list.",
        annotations(
            title = "List OpenRouter Models",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn list_models(
        &self,
        Parameters(args): Parameters<ListModelsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let query = ModelsQuery {
            q: args.query,
            output_modalities: args.output_modalities,
            input_modalities: args.input_modalities,
            supported_parameters: args.supported_parameters,
            // Default to most-used-this-week when the caller doesn't specify a sort.
            sort: Some(args.sort.unwrap_or_else(|| "top-weekly".to_string())),
            context: args.min_context,
        };

        let raw = self
            .client
            .list_models(&query)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        let filtered = apply_filters(raw, args.search.as_deref(), args.all);

        let mut json = serde_json::to_string_pretty(&filtered.models)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        if filtered.truncated() > 0 {
            json = format!(
                "// showing {} of {} models; set \"all\": true to get the rest\n{}",
                filtered.models.len(),
                filtered.total,
                json
            );
        }

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        description = "Generate or edit an image with an OpenRouter image model (e.g. \
        google/gemini-3.1-flash-image-preview) and save it to `output`. For text-to-image, \
        pass a prompt. For editing / image-to-image, also pass local `images` (order \
        preserved; optional per-image label) - the prompt becomes the edit instruction. \
        Set variants>1 to generate several in parallel (seed-stepped). Returns a compact \
        result: saved image paths, decoded width/height, requested vs actual \
        aspect_ratio/image_size, seeds, a path to the sidecar manifest, and any mismatch \
        warnings. The output format (PNG or JPEG) is chosen by the provider and the \
        extension is set to match. Set image_only=true for models that only output images \
        (e.g. Grok/FLUX). This tool has NO defaults: model, prompt, output, aspect_ratio, \
        image_size and image_only must all be specified, or the call fails with an error \
        naming what is missing. Runs asynchronously: if the job is still going after \
        wait_seconds (default 10), it returns status \"pending\" with a task_id to poll via \
        get_result; otherwise it returns the completed result inline. To analyze or caption \
        an existing image instead of creating one, use describe_image.",
        annotations(
            title = "Generate Image",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn generate_image(
        &self,
        Parameters(args): Parameters<GenerateImageArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let inline_previews = client_wants_inline_previews(&context);
        self.run_generate(args, inline_previews).await
    }

    /// Core of `generate_image`, parameterized on whether to embed inline image
    /// previews (decided per-client by the tool entrypoint). Separated so tests
    /// can drive it without constructing a transport `RequestContext`.
    async fn run_generate(
        &self,
        args: GenerateImageArgs,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        // No defaults: the agent must choose these explicitly.
        let mut missing: Vec<&str> = Vec::new();
        if args.aspect_ratio.is_none() {
            missing.push("aspect_ratio (e.g. \"1:1\", \"16:9\", \"9:16\")");
        }
        if args.image_size.is_none() {
            missing.push("image_size (e.g. \"1K\", \"2K\", \"4K\")");
        }
        if args.image_only.is_none() {
            missing.push(
                "image_only (true for image-only models e.g. Grok/Seedream/FLUX, \
                 false for dual text+image models e.g. Nano Banana/GPT Image)",
            );
        }
        if !missing.is_empty() {
            return Err(ErrorData::invalid_params(
                format!(
                    "generate_image has no defaults - specify every parameter explicitly. \
                     Missing: {}. (model, prompt and output are also required.) Use list_models \
                     with output_modalities=\"image\" to choose a model.",
                    missing.join("; ")
                ),
                None,
            ));
        }

        let aspect_ratio = args.aspect_ratio.clone();
        let image_size = args.image_size.clone();
        let req = GenerateRequest {
            model: args.model.clone(),
            prompt: args.prompt,
            aspect_ratio: args.aspect_ratio,
            image_size: args.image_size,
            seed: args.seed,
            image_only: args.image_only.unwrap_or(false),
            images: args
                .images
                .into_iter()
                .map(|i| image_gen::InputImage {
                    path: i.path.into(),
                    label: i.label,
                })
                .collect(),
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
        };

        let variants = args.variants.unwrap_or(1).clamp(1, 16);
        let wait = args
            .wait_seconds
            .unwrap_or(DEFAULT_WAIT_SECONDS)
            .clamp(1, 60);
        let base = PathBuf::from(&args.output);

        // Register a task and run the job on a background tokio task so the tool
        // never blocks longer than `wait`. The job keeps running after timeout
        // and stores its result for get_result.
        let task_id = uuid::Uuid::now_v7().to_string();
        self.tasks.insert_pending(&task_id, TaskKind::Image).await;

        let client = self.client.clone();
        let tasks = self.tasks.clone();
        let stats = self.stats.clone();
        let model = args.model.clone();
        let id_bg = task_id.clone();
        let variants_u64 = variants as u64;
        let handle = tokio::spawn(async move {
            match image_gen::run_job(&client, &req, variants, &base, "inline").await {
                Ok(summary) if !summary.images.is_empty() => {
                    let images = summary.images.len() as u64;
                    let cost: f64 = summary.images.iter().filter_map(|i| i.cost).sum();
                    let unknown = summary.images.iter().filter(|i| i.cost.is_none()).count() as u64;
                    stats
                        .record_job(&model, variants_u64, images, cost, unknown)
                        .await;
                    tasks
                        .complete(
                            &id_bg,
                            image_job_result_json(&summary, &aspect_ratio, &image_size),
                        )
                        .await;
                }
                Ok(summary) => {
                    stats.record_job(&model, variants_u64, 0, 0.0, 0).await;
                    tasks
                        .fail(
                            &id_bg,
                            format!(
                                "all {variants} variant(s) failed: {}",
                                summary.errors.join("; ")
                            ),
                        )
                        .await;
                }
                Err(e) => {
                    stats.record_job(&model, variants_u64, 0, 0.0, 0).await;
                    tasks.fail(&id_bg, format!("{e:#}")).await;
                }
            }
        });

        // Fast-return window: wait up to `wait` seconds, then report whatever
        // state the task is in (dropping the handle leaves it running).
        let _ = tokio::time::timeout(std::time::Duration::from_secs(wait), handle).await;
        let snap = self
            .tasks
            .snapshot(&task_id)
            .await
            .expect("task was just inserted");
        let env = snapshot_to_envelope(&task_id, &snap);
        job_call_result(&env, inline_previews).await
    }

    #[tool(
        description = "Fetch the status and result of a generation job by task_id (returned \
        by generate_image when a job is still running after its fast-return window). Returns \
        status pending|completed|failed; when completed, the same lean result (image paths, \
        dimensions, manifest) generate_image would have returned. Tasks are in-memory per \
        server process and are lost if the server restarts.",
        annotations(
            title = "Get Job Result",
            read_only_hint = true,
            destructive_hint = false
        )
    )]
    async fn get_result(
        &self,
        Parameters(args): Parameters<GetResultArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let inline_previews = client_wants_inline_previews(&context);
        self.run_get_result(args.task_id, inline_previews).await
    }

    /// Core of `get_result`, parameterized on inline previews like
    /// [`Self::run_generate`], so tests can call it without a `RequestContext`.
    async fn run_get_result(
        &self,
        task_id: String,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        match self.tasks.snapshot(&task_id).await {
            Some(snap) => {
                let env = snapshot_to_envelope(&task_id, &snap);
                job_call_result(&env, inline_previews).await
            }
            None => Err(ErrorData::invalid_params(
                format!(
                    "unknown task_id \"{task_id}\" (tasks are in-memory per server process and lost on restart)"
                ),
                None,
            )),
        }
    }

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
    async fn run_generate_video(
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
        if !missing.is_empty() {
            return Err(ErrorData::invalid_params(
                format!(
                    "generate_video has no defaults - specify every parameter explicitly. \
                     Missing: {}. (model, prompt and output are also required.) Use list_models \
                     with output_modalities=\"video\" to choose a model.",
                    missing.join("; ")
                ),
                None,
            ));
        }

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

        let task_id = uuid::Uuid::now_v7().to_string();
        self.tasks.insert_pending(&task_id, TaskKind::Video).await;

        let client = self.client.clone();
        let tasks = self.tasks.clone();
        let stats = self.stats.clone();
        let model = args.model.clone();
        let id_bg = task_id.clone();
        let handle = tokio::spawn(async move {
            match video_gen::run_job(&client, &req, &base, "inline").await {
                Ok(summary) if !summary.videos.is_empty() => {
                    let cost: Option<f64> = {
                        let costs: Vec<f64> =
                            summary.videos.iter().filter_map(|v| v.cost).collect();
                        if costs.is_empty() {
                            None
                        } else {
                            Some(costs.iter().sum())
                        }
                    };
                    stats.record_video(&model, true, cost).await;
                    tasks
                        .complete(&id_bg, video_job_result_json(&summary))
                        .await;
                }
                Ok(summary) => {
                    stats.record_video(&model, false, None).await;
                    tasks
                        .fail(
                            &id_bg,
                            format!("video generation failed: {}", summary.errors.join("; ")),
                        )
                        .await;
                }
                Err(e) => {
                    stats.record_video(&model, false, None).await;
                    tasks.fail(&id_bg, format!("{e:#}")).await;
                }
            }
        });

        let _ = tokio::time::timeout(std::time::Duration::from_secs(wait), handle).await;
        let snap = self
            .tasks
            .snapshot(&task_id)
            .await
            .expect("task was just inserted");
        let env = snapshot_to_envelope(&task_id, &snap);
        job_call_result(&env, inline_previews).await
    }

    #[tool(
        description = "Generate speech (text-to-speech) with an OpenRouter TTS model (e.g. \
        openai/gpt-4o-mini-tts or hexgrad/kokoro-82m) and save the audio to `output`. This is a \
        synchronous, fast call (not a background task). This tool has NO defaults: model, input \
        (the text), voice, and output must all be specified, or the call fails naming what is \
        missing. Returns the saved file path in JSON; for sandboxed clients it also returns a \
        native inline audio content block when the file is small enough. response_format defaults \
        to mp3 so the extension is deterministic.",
        annotations(
            title = "Generate Speech",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn generate_audio(
        &self,
        Parameters(args): Parameters<GenerateAudioArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let inline = client_wants_inline_previews(&context);
        self.run_generate_audio(args, inline).await
    }

    /// Core of `generate_audio` (synchronous, mirrors `describe_image`),
    /// parameterized on inline media so tests can drive it directly.
    async fn run_generate_audio(
        &self,
        args: GenerateAudioArgs,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        // No defaults: input and voice are the things agents forget.
        let mut missing: Vec<&str> = Vec::new();
        if args
            .input
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            missing.push("input (the text to synthesize)");
        }
        if args
            .voice
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            missing.push("voice (voice id, varies by model e.g. \"alloy\")");
        }
        if !missing.is_empty() {
            return Err(ErrorData::invalid_params(
                format!(
                    "generate_audio has no defaults - specify every parameter explicitly. \
                     Missing: {}. (model and output are also required.) Use list_models with \
                     output_modalities=\"speech\" to choose a model.",
                    missing.join("; ")
                ),
                None,
            ));
        }

        let model = args.model.clone();
        let req = SpeechGenRequest {
            model: args.model,
            input: args.input.unwrap_or_default(),
            voice: args.voice.unwrap_or_default(),
            response_format: args.response_format,
            speed: args.speed,
        };
        let output = PathBuf::from(&args.output);

        match audio_gen::run_job(&self.client, &req, &output, "inline").await {
            Ok(result) => {
                self.stats.record_audio(&model, true, None).await;
                let mut env = json!({
                    "ok": true,
                    "kind": "audio",
                    "model": result.model,
                    "audio": {
                        "path": result.audio.path.to_string_lossy(),
                        "mime": result.audio.mime,
                        "voice": result.audio.voice,
                        "response_format": result.audio.response_format,
                    },
                    "manifest": result.manifest_path.to_string_lossy(),
                });
                if !result.warnings.is_empty() {
                    env["warnings"] = json!(result.warnings);
                }
                let body = serde_json::to_string_pretty(&env)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                let mut blocks = vec![Content::text(body)];

                // Inline native AudioContent for sandboxed clients, under the cap.
                if inline_previews {
                    let path = result.audio.path.clone();
                    let mime = result.audio.mime.clone();
                    let small = std::fs::metadata(&path)
                        .map(|m| m.len() <= MAX_INLINE_AUDIO_BYTES)
                        .unwrap_or(false);
                    if small {
                        let read = tokio::task::spawn_blocking(move || std::fs::read(&path))
                            .await
                            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                        if let Ok(bytes) = read {
                            let data = base64::engine::general_purpose::STANDARD.encode(bytes);
                            blocks.push(
                                RawContent::Audio(RawAudioContent {
                                    data,
                                    mime_type: mime,
                                })
                                .no_annotation(),
                            );
                        }
                    }
                }
                Ok(CallToolResult::success(blocks))
            }
            Err(e) => {
                self.stats.record_audio(&model, false, None).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }

    #[tool(
        description = "Describe or answer a question about local image(s) using a vision-capable \
        model (image input, text output, e.g. google/gemini-2.5-flash, anthropic/claude-sonnet-4.6, \
        or openai/gpt-5.4). Pass one or more image paths and an optional prompt/question (defaults \
        to a detailed description); returns the model's text. Images are downscaled before sending. \
        To create or edit an image instead, use generate_image.",
        annotations(
            title = "Describe Image",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn describe_image(
        &self,
        Parameters(args): Parameters<DescribeImageArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        if args.images.is_empty() {
            return Err(ErrorData::invalid_params(
                "describe_image requires at least one image".to_string(),
                None,
            ));
        }
        let model = args.model.clone();
        let req = image_gen::DescribeRequest {
            model: args.model,
            prompt: args
                .prompt
                .unwrap_or_else(|| "Describe this image in detail.".to_string()),
            images: args
                .images
                .into_iter()
                .map(|i| image_gen::InputImage {
                    path: i.path.into(),
                    label: i.label,
                })
                .collect(),
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
        };
        match image_gen::describe_image(&self.client, &req).await {
            Ok(result) => {
                self.stats.record_text(&model, true, result.cost).await;
                Ok(CallToolResult::success(vec![Content::text(result.text)]))
            }
            Err(e) => {
                self.stats.record_text(&model, false, None).await;
                Err(ErrorData::internal_error(e.to_string(), None))
            }
        }
    }

    #[tool(
        description = "Return basic information about the OpenRouter API key in use \
        (GET /api/v1/key): label, creator_user_id (the owning user - the closest available \
        owner identity, not a name/email), credit usage (total and daily/weekly/monthly), \
        spending limit and remaining balance in USD (null means unlimited), byok_usage, the \
        is_free_tier / is_provisioning_key / is_management_key flags, and a deprecated \
        rate_limit (requests per interval; -1 means unlimited). This is account/key-level \
        info, not a per-request cost.",
        annotations(
            title = "Get Account Info",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn get_account(&self) -> Result<CallToolResult, ErrorData> {
        let info = self
            .client
            .get_key_info()
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        let body = serde_json::to_string_pretty(&info)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }

    #[tool(
        description = "Return in-memory usage statistics for this server process: started_at, \
        uptime_seconds, requests_total, requests_failed, image_generations, images_generated, \
        text_generations (describe_image calls), actual_cost_usd (summed from usage.cost), \
        unknown_cost_count, and a by_model breakdown. Counters reset when the server restarts.",
        annotations(
            title = "Get Usage Stats",
            read_only_hint = true,
            destructive_hint = false
        )
    )]
    async fn get_usage_stats(&self) -> Result<CallToolResult, ErrorData> {
        let snapshot = self.stats.snapshot().await;
        let body = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(body)]))
    }

    #[tool(
        description = "Reset all in-memory usage statistics to zero (and restart the uptime \
        clock). Destructive: requires confirm=true, otherwise it fails without changing anything.",
        annotations(
            title = "Reset Usage Stats",
            read_only_hint = false,
            destructive_hint = true,
            idempotent_hint = true
        )
    )]
    async fn reset_usage_stats(
        &self,
        Parameters(args): Parameters<ResetUsageStatsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        if !args.confirm {
            return Err(ErrorData::invalid_params(
                "reset_usage_stats requires confirm=true (this clears all usage counters)"
                    .to_string(),
                None,
            ));
        }
        self.stats.reset().await;
        Ok(CallToolResult::success(vec![Content::text(
            "usage stats reset".to_string(),
        )]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for OpenRouterServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "MCP server for OpenRouter. Use `list_models` to discover models, \
                their capabilities, and pricing, then `generate_image` to create \
                images, `generate_video` to create videos (slow, async: it returns \
                status \"pending\" with a task_id - poll `get_result` until \
                \"completed\"), and `generate_audio` for text-to-speech (synchronous). \
                If `generate_image` or `generate_video` returns status \"pending\" with \
                a task_id, poll `get_result` until it is \"completed\". \
                `get_usage_stats` reports this process's spend and counts.",
        )
    }
}

/// Start the stdio MCP server and run until the client disconnects.
pub async fn run() -> anyhow::Result<()> {
    let client = OpenRouterClient::from_env()?;
    let service = OpenRouterServer::new(client).serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    /// Build a server whose client talks to the given mock OpenRouter base URL.
    fn server_for(uri: String) -> OpenRouterServer {
        OpenRouterServer::new(OpenRouterClient::with_base_url(uri, "test-key"))
    }

    /// Extract the JSON the tool wrote into its text content block.
    fn tool_result_json(res: &CallToolResult) -> serde_json::Value {
        let v = serde_json::to_value(res).unwrap();
        let text = v["content"][0]["text"].as_str().expect("text content");
        serde_json::from_str(text).unwrap()
    }

    /// Base64 of a genuinely decodable 2x2 PNG, used wherever a test needs an
    /// image the preview path can decode + re-encode (it stands in for the valid
    /// images real providers return).
    fn valid_png_b64() -> String {
        let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 120, 200, 255]));
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        base64::engine::general_purpose::STANDARD.encode(buf.into_inner())
    }

    #[tokio::test]
    async fn generate_image_runs_async_and_get_result_fetches_it() {
        let mock = MockServer::start().await;
        let data_url = format!("data:image/png;base64,{}", valid_png_b64());
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "images": [ { "image_url": { "url": data_url } } ] } }],
                "usage": { "cost": 0.04 }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-async-test.png");
        let args = GenerateImageArgs {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: Some("1:1".to_string()),
            image_size: Some("1K".to_string()),
            seed: Some(5),
            image_only: Some(true),
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: Some(30),
            output: out.to_string_lossy().into_owned(),
        };
        // Fast mock completes within the wait window -> inline completed result.
        // inline_previews=true mirrors a Claude Desktop client.
        let res = server.run_generate(args, true).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["status"], "completed");
        assert_eq!(v["kind"], "image");
        assert!(v["images"][0]["path"].is_string());
        let task_id = v["task_id"].as_str().unwrap().to_string();

        // The completed result also carries an inline image preview block so
        // the client renders the generated image, not just its path.
        let full = serde_json::to_value(&res).unwrap();
        let img_block = full["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["type"] == "image")
            .expect("an image content block is present");
        assert_eq!(img_block["mimeType"], "image/png");
        assert!(!img_block["data"].as_str().unwrap().is_empty());

        // The same task is retrievable by id, also with an inline preview.
        let res2 = server.run_get_result(task_id.clone(), true).await.unwrap();
        let v2 = tool_result_json(&res2);
        assert_eq!(v2["status"], "completed");
        assert_eq!(v2["task_id"], task_id);
        let full2 = serde_json::to_value(&res2).unwrap();
        assert!(
            full2["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["type"] == "image"),
            "get_result also returns the inline preview"
        );

        // A CLI-style client (inline_previews=false) gets paths only, no image block.
        let res_cli = server.run_get_result(task_id.clone(), false).await.unwrap();
        let full_cli = serde_json::to_value(&res_cli).unwrap();
        assert!(
            !full_cli["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["type"] == "image"),
            "no inline preview when the client doesn't want it"
        );
    }

    #[test]
    fn encode_preview_blocks_reads_existing_and_skips_missing() {
        // A valid PNG on disk yields one image block.
        let png = base64::engine::general_purpose::STANDARD
            .decode(valid_png_b64())
            .unwrap();
        let dir = std::env::temp_dir();
        let good = dir.join("openrouter-mcp-preview-good.png");
        std::fs::write(&good, &png).unwrap();
        let missing = dir.join("openrouter-mcp-preview-missing.png");
        let _ = std::fs::remove_file(&missing);

        let env = json!({
            "images": [
                { "path": good.to_string_lossy() },
                { "path": missing.to_string_lossy() },
            ]
        });
        let blocks = encode_preview_blocks(&envelope_image_paths(&env));
        assert_eq!(blocks.len(), 1, "only the readable image becomes a block");
        let v = serde_json::to_value(&blocks[0]).unwrap();
        assert_eq!(v["type"], "image");
        assert_eq!(v["mimeType"], "image/png");
        assert!(!v["data"].as_str().unwrap().is_empty());

        // No images array -> no paths -> no blocks.
        assert!(envelope_image_paths(&json!({ "status": "completed" })).is_empty());
    }

    #[test]
    fn encode_preview_blocks_caps_at_the_limit() {
        // Write more readable PNGs than the cap; only MAX_INLINE_PREVIEWS render.
        let png = base64::engine::general_purpose::STANDARD
            .decode(valid_png_b64())
            .unwrap();
        let dir = std::env::temp_dir();
        let paths: Vec<String> = (0..MAX_INLINE_PREVIEWS + 3)
            .map(|i| {
                let p = dir.join(format!("openrouter-mcp-cap-{i}.png"));
                std::fs::write(&p, &png).unwrap();
                p.to_string_lossy().into_owned()
            })
            .collect();
        let blocks = encode_preview_blocks(&paths);
        assert_eq!(blocks.len(), MAX_INLINE_PREVIEWS, "preview count is capped");
    }

    #[tokio::test]
    async fn job_call_result_gates_previews_on_the_flag() {
        let good = std::env::temp_dir().join("openrouter-mcp-gate-good.png");
        let png = base64::engine::general_purpose::STANDARD
            .decode(valid_png_b64())
            .unwrap();
        std::fs::write(&good, &png).unwrap();
        let env = json!({
            "status": "completed",
            "images": [ { "path": good.to_string_lossy() } ],
        });

        let has_image = |res: &CallToolResult| {
            serde_json::to_value(res).unwrap()["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["type"] == "image")
        };
        // Desktop-style: previews on. CLI-style: text only.
        assert!(has_image(&job_call_result(&env, true).await.unwrap()));
        assert!(!has_image(&job_call_result(&env, false).await.unwrap()));

        // A pending job never carries a preview, even with previews enabled.
        let pending = json!({ "status": "pending", "images": [] });
        assert!(!has_image(&job_call_result(&pending, true).await.unwrap()));
    }

    #[tokio::test]
    async fn reset_usage_stats_requires_confirm() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let err = server
            .reset_usage_stats(Parameters(ResetUsageStatsArgs { confirm: false }))
            .await
            .unwrap_err();
        assert!(err.message.contains("confirm=true"));

        // get_usage_stats works and starts at zero.
        let res = server.get_usage_stats().await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["requests_total"], 0);
        assert_eq!(v["images_generated"], 0);
    }

    #[tokio::test]
    async fn get_result_unknown_task_errors() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let err = server
            .run_get_result("nope".to_string(), true)
            .await
            .unwrap_err();
        assert!(err.message.contains("unknown task_id"));
    }

    #[tokio::test]
    async fn list_models_tool_returns_model_json() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "openai/gpt", "name": "GPT"}]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .list_models(Parameters(ListModelsArgs::default()))
            .await
            .unwrap();

        // The tool returns the model list as pretty JSON text content.
        let body = serde_json::to_string(&result).unwrap();
        assert!(body.contains("openai/gpt"));
    }

    #[tokio::test]
    async fn generate_image_requires_explicit_parameters() {
        // Validation runs before any HTTP call, so the base URL is never used.
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = GenerateImageArgs {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: None,
            image_size: None,
            seed: None,
            image_only: None,
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: None,
            output: "out.png".to_string(),
        };
        let err = server.run_generate(args, true).await.unwrap_err();
        assert!(err.message.contains("aspect_ratio"));
        assert!(err.message.contains("image_size"));
        assert!(err.message.contains("image_only"));
        assert!(err.message.contains("no defaults"));
    }

    #[tokio::test]
    async fn get_account_tool_returns_key_json() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "label": "sk-or-v1-x",
                    "creator_user_id": "user_42",
                    "usage": 1.5,
                    "limit": null,
                    "is_free_tier": false
                }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server.get_account().await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["label"], "sk-or-v1-x");
        assert_eq!(v["creator_user_id"], "user_42");
        assert_eq!(v["usage"], 1.5);
        assert!(v["limit"].is_null());
    }

    #[tokio::test]
    async fn generate_audio_synthesizes_and_returns_path_json() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-tool/voice.mp3");
        let args = GenerateAudioArgs {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: Some("hello".to_string()),
            voice: Some("alloy".to_string()),
            response_format: None,
            speed: None,
            output: out.to_string_lossy().into_owned(),
        };
        // inline_previews=false -> JSON only, no embedded audio block.
        let res = server.run_generate_audio(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["ok"], true);
        assert_eq!(v["kind"], "audio");
        assert_eq!(v["audio"]["voice"], "alloy");
        assert_eq!(v["audio"]["mime"], "audio/mpeg");
        assert!(v["audio"]["path"].as_str().unwrap().ends_with(".mp3"));

        // The stats counter recorded the audio generation.
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["audio_files"], 1);
    }

    #[tokio::test]
    async fn generate_audio_requires_input_and_voice() {
        // Validation runs before any HTTP call.
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = GenerateAudioArgs {
            model: "m".to_string(),
            input: None,
            voice: Some("  ".to_string()), // blank-after-trim counts as missing
            response_format: None,
            speed: None,
            output: "out.mp3".to_string(),
        };
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert!(err.message.contains("input"));
        assert!(err.message.contains("voice"));
        assert!(err.message.contains("no defaults"));
    }

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

    #[tokio::test]
    async fn list_models_tool_surfaces_upstream_errors() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let err = server
            .list_models(Parameters(ListModelsArgs::default()))
            .await
            .unwrap_err();
        assert!(err.message.contains("error status"));
    }
}
