//! Image tools (`generate_image`, `describe_image`), their argument structs, the
//! shared `ImageInput` type, and the image-job result builder.

use rmcp::{
    ErrorData, RoleServer, handler::server::wrapper::Parameters, model::CallToolResult,
    service::RequestContext, tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::image_gen::{self, GenerateRequest};
use crate::server::naming;
use crate::server::provider::{ImageProviderArgs, ProviderRoutingArgs};
use crate::server::result::{
    DEFAULT_WAIT_SECONDS, attach_warnings_errors, client_wants_inline_previews,
};
use crate::server::schema::{
    AtLeastOneOf, de_lenient, de_opt_f64, de_opt_uint, require_all, scalarize_nullable,
};
use crate::tasks::TaskKind;

use super::OpenRouterServer;
use super::media;

/// An input image for editing / image-to-image / vision. Exactly one of
/// `path`, `url`, or `base64` must be set. Order is preserved.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = AtLeastOneOf(&["path", "url", "base64"]))]
pub(crate) struct ImageInput {
    /// Local file path (png/jpeg/webp/gif/svg). One of path/url/base64.
    #[serde(default)]
    pub path: Option<String>,
    /// HTTP(S) URL to fetch the image from. One of path/url/base64.
    #[serde(default)]
    pub url: Option<String>,
    /// Inline image data: a full `data:` URL or raw base64. One of path/url/base64.
    #[serde(default)]
    pub base64: Option<String>,
    /// Optional label, surfaced to the model as a reference name.
    #[serde(default)]
    pub label: Option<String>,
}

/// Validate that an image spec carries exactly one source (path/url/base64),
/// cheaply and without any network fetch. Lets a caller (e.g. `chat_completion`)
/// surface a malformed-image error before running the network-bound model-
/// capability gate.
pub(crate) fn check_image_input(img: &ImageInput) -> Result<(), ErrorData> {
    media::check_exactly_one(
        media::InputKind::Image,
        img.path.as_deref(),
        img.url.as_deref(),
        img.base64.as_deref(),
    )
}

/// Resolve one tool-level [`ImageInput`] to a generator [`image_gen::InputImage`]
/// through the shared source resolver: a path stays lazy (read in
/// `prepare_inputs`), URLs are fetched (SSRF-guarded) and base64/data-URL
/// inputs decoded, both capped at 20 MiB. Requires exactly one source.
async fn resolve_image_input(img: ImageInput) -> Result<image_gen::InputImage, ErrorData> {
    let label = img.label;
    let resolved = media::resolve_source(
        media::InputKind::Image,
        img.path,
        img.url,
        img.base64,
        crate::resources::MAX_IMAGE_BYTES,
        true,
    )
    .await?;
    Ok(match resolved {
        media::Resolved::Path(p) => image_gen::InputImage::from_path(p, label),
        media::Resolved::Bytes(b) => image_gen::InputImage::inline(b.bytes, b.name, label),
        media::Resolved::Url(_) => {
            return Err(ErrorData::internal_error("image url was not fetched", None));
        }
    })
}

/// Resolve a list of tool-level [`ImageInput`]s to generator inputs, in order.
pub(crate) async fn resolve_image_inputs(
    images: Vec<ImageInput>,
) -> Result<Vec<image_gen::InputImage>, ErrorData> {
    media::check_count(media::InputKind::Image, images.len())?;
    let mut total = 0usize;
    let mut out = Vec::with_capacity(images.len());
    for img in images {
        let input = resolve_image_input(img).await?;
        if let image_gen::ImageSource::Inline { bytes, .. } = &input.source {
            total += bytes.len();
            if total > crate::resources::MAX_IMAGE_TOTAL_BYTES {
                return Err(ErrorData::invalid_params(
                    format!(
                        "input images exceed {} MiB in total",
                        crate::resources::MAX_IMAGE_TOTAL_BYTES / (1024 * 1024)
                    ),
                    None,
                ));
            }
        }
        out.push(input);
    }
    Ok(out)
}

/// Arguments for the `generate_image` tool.
///
/// `aspect_ratio` and `image_size` are required unless `size` is given, so
/// they are conditional like `generate_video.aspect_ratio` and stay out of the
/// schema's unconditional `required` list; `run_generate` enforces the rule.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct GenerateImageArgs {
    /// Image model id, e.g. "google/gemini-3.1-flash-image-preview".
    pub model: String,
    /// Prompt text describing the image to generate (or the edit to apply).
    pub prompt: String,
    /// REQUIRED unless `size` is given (no default): aspect ratio, e.g. "1:1",
    /// "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "2.35:1", "5:2",
    /// "9:19.5", "19.5:9", "9:20", "20:9" (maps to the Images API
    /// `aspect_ratio`). Provider support varies. Cannot be combined with a
    /// pixel-form `size`.
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    /// REQUIRED unless `size` is given (no default): resolution TIER (not
    /// pixel dimensions), e.g. "512" (or "0.5K"), "1K", "2K", "4K" (maps to
    /// the Images API `resolution`). Cannot be combined with a pixel-form
    /// `size`.
    #[serde(default)]
    pub image_size: Option<String>,
    /// Output size as "WIDTHxHEIGHT" pixels (e.g. "2048x2048") or a tier
    /// (e.g. "2K"). Alternative to aspect_ratio + image_size: a pixel-form
    /// size together with either of them is rejected locally, because
    /// OpenRouter returns 400 for that combination. Provider support varies.
    #[serde(default)]
    pub size: Option<String>,
    /// Seed for reproducible-ish generation (provider support varies).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub seed: Option<u64>,
    /// Input images to edit/condition on (image-to-image / multi-image). Each
    /// takes exactly one of: path (local file), url (http/https, fetched), or
    /// base64 (a data: URL or raw base64). Omit for plain text-to-image.
    #[serde(default)]
    pub images: Vec<ImageInput>,
    /// Longest-side cap (px) for input images before sending (default 1536,
    /// max 4096; env OPENROUTER_IMAGE_MAX_DIMENSION).
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(max = 4096))]
    pub max_image_dimension: Option<u32>,
    /// Number of variants to generate in parallel (1-16, seed-stepped). Default 1.
    /// With >1, files are named `<output>-var-<seed>-<index>` (seed zero-padded to
    /// 4 digits, index to 3), or `-var-<index>` when no seed is set; one manifest
    /// covers all variants.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(min = 1, max = 16))]
    pub variants: Option<usize>,
    /// Seconds to wait inline before returning a task_id for a slow job (1-60,
    /// default 10). The job keeps running; fetch it later with get_result.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(min = 1, max = 60))]
    pub wait_seconds: Option<u64>,
    /// Output file path (single image, or the base name for variants). The
    /// extension is corrected to the actual returned format. Optional: when
    /// omitted, an auto-named file is written under OPENROUTER_MCP_OUTPUT_DIR
    /// (default $HOME/Downloads/openrouter-mcp).
    #[serde(default)]
    pub output: Option<String>,
    /// Output quality: "auto", "low", "medium", "high", "xhigh", or "max".
    /// Provider support varies.
    #[serde(default)]
    pub quality: Option<String>,
    /// Output file format: "png", "jpeg", "webp", or "svg". Provider support
    /// varies; the saved file's extension always matches what the provider
    /// actually returns, not this request.
    #[serde(default)]
    pub output_format: Option<String>,
    /// Background: "auto", "transparent", or "opaque". Provider support varies.
    #[serde(default)]
    pub background: Option<String>,
    /// Output compression 0-100 (webp/jpeg only). Provider support varies.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(min = 0, max = 100))]
    pub output_compression: Option<u32>,
    /// Provider block for this request: routing plus per-provider passthrough,
    /// as {"order": [...], "only": [...], "ignore": [...], "allow_fallbacks": bool,
    /// "sort": "price"|"throughput"|"latency"|"exacto", "sort_partition":
    /// "model"|"none", "options": {"<provider-slug>": {...}}}. `options` is
    /// keyed by provider slug and holds that provider's own parameters;
    /// describe_model lists each endpoint's allowed_passthrough_parameters,
    /// e.g. {"options": {"black-forest-labs": {"steps": 28, "guidance": 3.5}}}.
    /// Only the slug that serves the request is forwarded.
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ImageProviderArgs,
}

/// Arguments for the `describe_image` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct DescribeImageArgs {
    /// Vision-capable model id (image input, text output), e.g.
    /// "google/gemini-2.5-flash" or "anthropic/claude-sonnet-4.6".
    pub model: String,
    /// Image(s) to describe (at least one required). Each takes exactly one of:
    /// path (local file), url (http/https), or base64 (data: URL or raw base64).
    #[schemars(length(min = 1))]
    pub images: Vec<ImageInput>,
    /// Instruction or question about the image(s). Defaults to a detailed description.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Longest-side cap (px) for input images before sending (default 1536, max 4096).
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(max = 4096))]
    pub max_image_dimension: Option<u32>,
    /// Optional reasoning effort: "max", "xhigh", "high", "medium", "low",
    /// "minimal" or "none". Omit to keep the model's own default. Accepted
    /// values differ per model - check `reasoning.supported_efforts` from
    /// list_models/describe_model. Plain description needs little thinking;
    /// charts, diagrams and dense text benefit from "high".
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Optional system instruction prepended as a system message.
    #[serde(default)]
    pub system: Option<String>,
    /// Optional sampling temperature.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub temperature: Option<f64>,
    /// Optional maximum number of tokens to generate.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_tokens: Option<u64>,
    /// Provider block for this request: routing only, as {"order": [...],
    /// "only": [...], "ignore": [...], "allow_fallbacks", "require_parameters",
    /// "zdr", "sort", "sort_partition"}. Chat completions have no per-provider
    /// `options` passthrough (describe_model's allowed_passthrough_parameters
    /// do not apply here).
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ProviderRoutingArgs,
}

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
    attach_warnings_errors(&mut result, &summary.warnings, &summary.errors);
    result
}

#[tool_router(router = image_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Generate or edit an image with an OpenRouter image model (e.g. \
        google/gemini-3.1-flash-image-preview) and save it. Runs asynchronously: if the job is \
        still going after wait_seconds (default 10), it returns status \"pending\" with a \
        task_id to poll via get_result; otherwise it returns the completed result inline. \
        `output` is optional - omit it to \
        get an auto-named file (kind_datetime_model_config_seed_hash) under \
        OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp). For text-to-image, \
        pass a prompt. For editing / image-to-image, also pass `images` - each given as a \
        local path, an http(s) url, or base64/data-URL (order preserved; optional per-image \
        label) - the prompt becomes the edit instruction. \
        Set variants>1 to generate several in parallel (seed-stepped). Optional `quality` \
        (auto/low/medium/high/xhigh/max), `output_format` (png/jpeg/webp/svg), `background` \
        (auto/transparent/opaque), and `output_compression` (0-100, webp/jpeg only) are passed \
        straight through to the provider - support for each varies by model, and whatever \
        format actually comes back is what gets saved (the extension always matches the real \
        result, not the request). Provider routing and provider-specific parameters go in \
        `provider`: routing keys order/only/ignore/allow_fallbacks/sort, and `provider.options` \
        keyed by provider slug with that provider's own parameters - describe_model lists each \
        endpoint's `allowed_passthrough_parameters` - e.g. \
        {\"options\": {\"black-forest-labs\": {\"steps\": 28, \"guidance\": 3.5}}}. Returns a compact \
        result: saved image paths, decoded width/height, requested vs actual \
        aspect_ratio/image_size, seeds, a path to the sidecar manifest, and any mismatch \
        warnings. Works with any OpenRouter image model (Nano Banana, Grok, \
        Seedream, FLUX, GPT Image, Recraft, ...) via the dedicated image endpoint. No defaults \
        for the required fields: model, prompt, and either `size` (\"WIDTHxHEIGHT\" pixels \
        such as \"2048x2048\", or a tier) or both aspect_ratio and image_size must be \
        specified, or the call fails with an error naming what is missing; a pixel-form `size` \
        combined with aspect_ratio or image_size is rejected locally because OpenRouter \
        returns 400 for it. Every other param - seed, images (max 16), max_image_dimension, \
        variants, wait_seconds, output, quality, output_format, background, \
        output_compression, provider - is optional. To analyze or caption \
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
    pub(crate) async fn run_generate(
        &self,
        args: GenerateImageArgs,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        // Blank/whitespace-only strings count as absent: better a clear
        // "missing parameter" error here than a confusing provider 400.
        let non_blank = |o: Option<String>| o.filter(|s| !s.trim().is_empty());
        let args = GenerateImageArgs {
            aspect_ratio: non_blank(args.aspect_ratio),
            image_size: non_blank(args.image_size),
            size: non_blank(args.size),
            quality: non_blank(args.quality),
            output_format: non_blank(args.output_format),
            background: non_blank(args.background),
            ..args
        };
        // No defaults: the agent must choose the output geometry explicitly -
        // either `size` alone, or aspect_ratio + image_size.
        if args.size.is_none() {
            let mut missing: Vec<&str> = Vec::new();
            if args.aspect_ratio.is_none() {
                missing.push("aspect_ratio (e.g. \"1:1\", \"16:9\", \"9:16\")");
            }
            if args.image_size.is_none() {
                missing.push("image_size (e.g. \"1K\", \"2K\", \"4K\")");
            }
            require_all("generate_image", "image", &missing)?;
        }
        image_gen::check_size_conflict(
            args.size.as_deref(),
            args.image_size.as_deref(),
            args.aspect_ratio.as_deref(),
        )
        .map_err(|e| ErrorData::invalid_params(e.to_string(), None))?;
        let provider = args.provider.into_image_provider()?;

        let Some(reservation) = self.tasks.reserve(TaskKind::Image) else {
            return Ok(Self::admission_error());
        };
        let aspect_ratio = args.aspect_ratio.clone();
        let image_size = args.image_size.clone();
        let size = args.size.clone();
        let images = resolve_image_inputs(args.images).await?;
        let req = GenerateRequest {
            model: args.model.clone(),
            prompt: args.prompt,
            aspect_ratio: args.aspect_ratio,
            image_size: args.image_size,
            seed: args.seed,
            images,
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
            quality: args.quality,
            output_format: args.output_format,
            background: args.background,
            // The schema range is advisory only (rmcp does not validate), so
            // clamp here the way variants/wait_seconds already do.
            output_compression: args.output_compression.map(|c| c.min(100)),
            size: args.size,
            provider,
        };

        let variants = args.variants.unwrap_or(1).clamp(1, 16);
        let wait = args
            .wait_seconds
            .unwrap_or(DEFAULT_WAIT_SECONDS)
            .clamp(1, 60);
        let mut config: Vec<&str> = Vec::new();
        if let Some(a) = &aspect_ratio {
            config.push(a);
        }
        if let Some(s) = &image_size {
            config.push(s);
        }
        if let Some(s) = &size {
            config.push(s);
        }
        let base = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Image,
            &args.model,
            &config,
            args.seed,
        );
        let model = args.model;
        let variants_u64 = variants as u64;

        self.spawn_reserved_job_and_wait(
            reservation,
            wait,
            inline_previews,
            move |ctx| async move {
                match image_gen::run_job(&ctx.client, &req, variants, &base).await {
                    Ok(summary) => {
                        ctx.stats
                            .record_job(
                                &model,
                                variants_u64,
                                summary.images.len() as u64,
                                summary.billing.cost,
                                summary.billing.unknown,
                            )
                            .await;
                        if summary.images.is_empty() {
                            return Err(format!(
                                "all {variants} variant(s) failed: {}",
                                summary.errors.join("; ")
                            ));
                        }
                        Ok(image_job_result_json(&summary, &aspect_ratio, &image_size))
                    }
                    Err(e) => {
                        ctx.stats.record_job(&model, variants_u64, 0, 0.0, 0).await;
                        Err(format!("{e:#}"))
                    }
                }
            },
        )
        .await
    }

    #[tool(
        description = "Describe or answer a question about image(s) using a vision-capable \
        model (image input, text output, e.g. google/gemini-2.5-flash, anthropic/claude-sonnet-4.6, \
        or openai/gpt-5.4). Pass one or more images (each a local path, an http(s) url, or \
        base64/data-URL) and an optional prompt/question (defaults to a detailed description); \
        returns the model's text as the first content block, then a JSON block with the \
        generation_id (for get_generation) and, when present, reasoning, finish_reason and \
        token counts - the same shape chat_completion returns. Images are downscaled before \
        sending. Optional `system`, \
        `temperature`, `max_tokens` and `reasoning_effort` are passed through; `provider` \
        takes routing fields only (order, only, ignore, allow_fallbacks, require_parameters, \
        zdr, sort) - there is no per-provider `options` passthrough on chat completions. \
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
        let _work = self.admit_work()?;
        if args.images.is_empty() {
            return Err(ErrorData::invalid_params(
                "describe_image requires at least one image".to_string(),
                None,
            ));
        }
        let model = args.model.clone();
        let provider = args.provider.into_routing()?;
        let req = image_gen::DescribeRequest {
            model: args.model,
            prompt: args
                .prompt
                .unwrap_or_else(|| "Describe this image in detail.".to_string()),
            images: resolve_image_inputs(args.images).await?,
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
            reasoning_effort: args.reasoning_effort,
            system: args.system,
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            provider,
        };
        let outcome = image_gen::describe_image(&self.client, &req).await;
        self.finish_chat_call(&model, outcome).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json, valid_png_b64};
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn generate_image_runs_async_and_get_result_fetches_it() {
        let mock = MockServer::start().await;
        // The Images API returns raw base64 bytes in data[].b64_json.
        Mock::given(method("POST"))
            .and(path("/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "created": 1748372400,
                "data": [{ "b64_json": valid_png_b64() }],
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
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: Some(30),
            output: Some(out.to_string_lossy().into_owned()),
            quality: None,
            output_format: None,
            background: None,
            output_compression: None,
            size: None,
            provider: Default::default(),
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
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: None,
            output: Some("out.png".to_string()),
            quality: None,
            output_format: None,
            background: None,
            output_compression: None,
            size: None,
            provider: Default::default(),
        };
        let err = server.run_generate(args, true).await.unwrap_err();
        assert!(err.message.contains("aspect_ratio"));
        assert!(err.message.contains("image_size"));
        assert!(err.message.contains("no defaults"));
    }

    #[tokio::test]
    async fn generate_image_treats_blank_strings_as_missing() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = GenerateImageArgs {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: Some("  ".to_string()),
            image_size: Some("".to_string()),
            seed: None,
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: None,
            output: Some("out.png".to_string()),
            quality: None,
            output_format: None,
            background: None,
            output_compression: None,
            size: None,
            provider: Default::default(),
        };
        let err = server.run_generate(args, true).await.unwrap_err();
        assert!(err.message.contains("aspect_ratio"), "got: {}", err.message);
        assert!(err.message.contains("image_size"), "got: {}", err.message);
    }

    #[tokio::test]
    async fn generate_image_forwards_quality_format_background_compression() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images"))
            .and(wiremock::matchers::body_partial_json(json!({
                "quality": "medium",
                "output_format": "webp",
                "background": "opaque",
                "output_compression": 50
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "b64_json": valid_png_b64(), "media_type": "image/webp" }]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-quality-test.png");
        let args = GenerateImageArgs {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: Some("1:1".to_string()),
            image_size: Some("1K".to_string()),
            seed: None,
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: Some(30),
            output: Some(out.to_string_lossy().into_owned()),
            quality: Some("medium".to_string()),
            output_format: Some("webp".to_string()),
            background: Some("opaque".to_string()),
            output_compression: Some(50),
            size: None,
            provider: Default::default(),
        };
        let res = server.run_generate(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["status"], "completed");
        // The bytes are actually PNG (valid_png_b64) despite the provider
        // declaring "image/webp": sniffing wins, so the file is saved as .png,
        // and the mismatch is surfaced as a warning rather than silently trusted.
        assert!(
            v["images"][0]["path"].as_str().unwrap().ends_with(".png"),
            "got: {v}"
        );
        let warnings = v["warnings"].as_array().expect("a warning is present");
        assert!(
            warnings
                .iter()
                .any(|w| w.as_str().unwrap().contains("image/webp")
                    && w.as_str().unwrap().contains("image/png")),
            "got: {warnings:?}"
        );
    }

    /// The `/images` `provider` block (routing subset + `options` keyed by
    /// slug) and `size` reach the wire nested exactly as OpenRouter documents
    /// them, deserialized the way a client sends them (lenient nested object).
    /// With `size` given, `aspect_ratio` and `image_size` are not required.
    #[tokio::test]
    async fn generate_image_forwards_provider_and_size_to_the_wire() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images"))
            .and(wiremock::matchers::body_partial_json(json!({
                "model": "black-forest-labs/flux.2-pro",
                "size": "2048x2048",
                "provider": {
                    "order": ["black-forest-labs"],
                    "options": {"black-forest-labs": {"steps": 28, "guidance": 3.5}}
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "b64_json": valid_png_b64() }]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-provider-test.png");
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model": "black-forest-labs/flux.2-pro",
            "prompt": "p",
            "size": "2048x2048",
            "wait_seconds": 30,
            "output": out.to_string_lossy(),
            "provider": {
                "order": ["black-forest-labs"],
                "options": {"black-forest-labs": {"steps": 28, "guidance": 3.5}}
            }
        }))
        .unwrap();
        let res = server.run_generate(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["status"], "completed", "got: {v}");
        let body: serde_json::Value =
            serde_json::from_slice(&mock.received_requests().await.unwrap()[0].body).unwrap();
        assert!(body.get("resolution").is_none(), "got: {body}");
        assert!(body.get("aspect_ratio").is_none(), "got: {body}");
        // The manifest records the provider block that was sent.
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(v["manifest"].as_str().unwrap()).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["size"], "2048x2048");
        assert_eq!(manifest["provider"]["order"], json!(["black-forest-labs"]));
    }

    /// A pixel-form `size` together with `image_size` or `aspect_ratio` is the
    /// combination OpenRouter rejects with 400; it is refused locally as
    /// invalid_params before any HTTP call, naming the conflict.
    #[tokio::test]
    async fn generate_image_rejects_pixel_size_combined_with_tier_or_ratio() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = |size: &str, image_size: Option<&str>, aspect_ratio: Option<&str>| {
            serde_json::from_value::<GenerateImageArgs>(json!({
                "model": "m", "prompt": "p", "output": "out.png",
                "size": size, "image_size": image_size, "aspect_ratio": aspect_ratio,
            }))
            .unwrap()
        };
        let err = server
            .run_generate(args("2048x2048", Some("2K"), None), true)
            .await
            .unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("size"), "got: {}", err.message);
        assert!(err.message.contains("image_size"), "got: {}", err.message);
        assert!(err.message.contains("400"), "got: {}", err.message);

        let err = server
            .run_generate(args("1024x768", None, Some("4:3")), true)
            .await
            .unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("aspect_ratio"), "got: {}", err.message);
    }

    /// Without `size`, aspect_ratio and image_size stay required (unchanged
    /// semantics); a blank `size` counts as absent.
    #[tokio::test]
    async fn generate_image_blank_size_keeps_ratio_and_tier_required() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "output": "out.png", "size": "  ",
        }))
        .unwrap();
        let err = server.run_generate(args, true).await.unwrap_err();
        assert!(err.message.contains("aspect_ratio"), "got: {}", err.message);
        assert!(err.message.contains("image_size"), "got: {}", err.message);
    }

    /// More than 16 input images is refused as invalid_params at the tool
    /// boundary, before any source is decoded or fetched.
    #[tokio::test]
    async fn generate_image_rejects_more_than_16_input_images() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let images: Vec<_> = (0..17).map(|_| json!({"base64": "invalid!"})).collect();
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "aspect_ratio": "1:1", "image_size": "1K",
            "output": "out.png", "images": images,
        }))
        .unwrap();
        let err = server.run_generate(args, true).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("16"), "got: {}", err.message);
    }

    /// An invalid provider block (a scalar where a per-slug object is due) is
    /// rejected before any HTTP call.
    #[tokio::test]
    async fn generate_image_rejects_bad_provider_options_before_http() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "aspect_ratio": "1:1", "image_size": "1K",
            "output": "out.png", "provider": {"options": {"acme": true}},
        }))
        .unwrap();
        let err = server.run_generate(args, true).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("acme"), "got: {}", err.message);
    }

    /// `describe_image` no longer hardcodes system/temperature/max_tokens to
    /// `None`: they and the provider routing block reach the wire, arriving the
    /// way a client sends them (lenient nested object).
    #[tokio::test]
    async fn describe_image_forwards_system_sampling_and_provider() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "messages": [{"role": "system", "content": "be brief"}, {"role": "user"}],
                "temperature": 0.2,
                "max_tokens": 50,
                "provider": {"order": ["google-vertex"], "zdr": true}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-desc")
                    .set_body_json(json!({
                        "choices": [{"message": {"content": "a square"}}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 2}
                    })),
            )
            .mount(&mock)
            .await;
        let args: DescribeImageArgs = serde_json::from_value(json!({
            "model": "google/gemini-2.5-flash",
            "images": [{"base64": valid_png_b64()}],
            "system": "be brief",
            "temperature": "0.2",
            "max_tokens": 50,
            "provider": {"order": ["google-vertex"], "zdr": "true"}
        }))
        .unwrap();
        let res = server_for(mock.uri())
            .describe_image(rmcp::handler::server::wrapper::Parameters(args))
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "a square");
        // Same second block as chat_completion: the generation id (for
        // get_generation) and the token counts.
        let meta: serde_json::Value =
            serde_json::from_str(v["content"][1]["text"].as_str().unwrap()).unwrap();
        assert_eq!(meta["generation_id"], "gen-desc");
        assert_eq!(meta["usage"]["prompt_tokens"], 9);

        // An invalid routing block is rejected before any HTTP call.
        let bad: DescribeImageArgs = serde_json::from_value(json!({
            "model": "m", "images": [{"base64": valid_png_b64()}],
            "provider": {"sort_partition": "model"}
        }))
        .unwrap();
        let err = server_for("http://127.0.0.1:9".to_string())
            .describe_image(rmcp::handler::server::wrapper::Parameters(bad))
            .await
            .unwrap_err();
        assert!(
            err.message.contains("sort_partition"),
            "got: {}",
            err.message
        );
    }

    /// Defense in depth: even with a scalar schema, clients that stringify all
    /// arguments must still deserialize (the exact failure mode from the bug report).
    #[test]
    fn generate_image_args_accept_stringified_scalars() {
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model": "x-ai/grok-imagine-image-quality",
            "prompt": "a small test image",
            "aspect_ratio": "1:1",
            "image_size": "1K",
            "seed": "42",
            "variants": "2",
            "output": "out.png",
        }))
        .expect("stringified scalars should deserialize");
        assert_eq!(args.seed, Some(42));
        assert_eq!(args.variants, Some(2));
    }

    /// Native typed values and absent/null optionals still work unchanged.
    #[test]
    fn generate_image_args_accept_native_and_absent_scalars() {
        let native: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "seed": 7, "variants": 3, "output": "o.png",
        }))
        .unwrap();
        assert_eq!(native.seed, Some(7));
        assert_eq!(native.variants, Some(3));

        let absent: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "seed": null, "output": "o.png",
        }))
        .unwrap();
        assert_eq!(absent.seed, None);
        assert_eq!(absent.variants, None);
    }

    /// Garbage strings are rejected with a clear message rather than silently
    /// coerced.
    #[test]
    fn invalid_stringified_scalars_are_rejected() {
        let err = serde_json::from_value::<GenerateImageArgs>(json!({
            "model": "m", "prompt": "p", "seed": "not-a-number", "output": "o.png",
        }))
        .unwrap_err();
        assert!(err.to_string().contains("integer"), "got: {err}");
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[tokio::test]
    async fn rejects_excess_jobs_before_decoding_inputs() {
        let server = super::super::test_support::server_for("http://127.0.0.1:9".into());
        let reservations: Vec<_> = (0..32)
            .map(|_| server.tasks.reserve(TaskKind::Image).unwrap())
            .collect();
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model":"test", "prompt":"test", "aspect_ratio":"1:1", "image_size":"1K",
            "images":[{"base64":"invalid!"}]
        }))
        .unwrap();
        let result = server.run_generate(args, false).await.unwrap();
        assert_eq!(result.is_error, Some(true));
        assert!(
            serde_json::to_string(&result)
                .unwrap()
                .contains("pending generation jobs")
        );
        drop(reservations);
        assert!(server.tasks.reserve(TaskKind::Image).is_some());
    }

    #[tokio::test]
    async fn rejects_image_count_before_resolving_sources() {
        let images = (0..17)
            .map(|_| ImageInput {
                path: None,
                url: None,
                base64: Some("invalid!".into()),
                label: None,
            })
            .collect();
        let error = resolve_image_inputs(images).await.unwrap_err();
        assert!(error.message.contains("at most 16"));
    }

    #[tokio::test]
    async fn invalid_preparation_releases_job_reservation() {
        let server = super::super::test_support::server_for("http://127.0.0.1:9".into());
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model":"test", "prompt":"test", "aspect_ratio":"1:1", "image_size":"1K",
            "images":[{"base64":"invalid!"}]
        }))
        .unwrap();
        assert!(server.run_generate(args, false).await.is_err());
        let reservations: Vec<_> = (0..32)
            .map(|_| server.tasks.reserve(TaskKind::Image).unwrap())
            .collect();
        assert_eq!(reservations.len(), 32);
    }
}
