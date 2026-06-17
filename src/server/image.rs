//! Image tools (`generate_image`, `describe_image`), their argument structs, the
//! shared `ImageInput` type, and the image-job result builder.

use rmcp::{
    ErrorData, RoleServer,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    service::RequestContext,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::image_gen::{self, GenerateRequest};
use crate::server::naming;
use crate::server::result::{
    DEFAULT_WAIT_SECONDS, attach_warnings_errors, client_wants_inline_previews,
};
use crate::server::schema::{de_opt_bool, de_opt_uint, require_all, scalarize_nullable};
use crate::tasks::TaskKind;

use super::OpenRouterServer;

/// A local input image for editing / image-to-image. Order is preserved.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct ImageInput {
    /// Local file path (png/jpeg/webp/gif).
    pub path: String,
    /// Optional label, surfaced to the model as a reference name.
    #[serde(default)]
    pub label: Option<String>,
}

impl From<ImageInput> for image_gen::InputImage {
    fn from(i: ImageInput) -> Self {
        image_gen::InputImage {
            path: i.path.into(),
            label: i.label,
        }
    }
}

/// Map a list of tool-level [`ImageInput`]s to the generator's `InputImage`s.
fn to_input_images(images: Vec<ImageInput>) -> Vec<image_gen::InputImage> {
    images.into_iter().map(Into::into).collect()
}

/// Arguments for the `generate_image` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct GenerateImageArgs {
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
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub seed: Option<u64>,
    /// REQUIRED (no default): true for image-only-output models
    /// (e.g. Grok/Seedream/FLUX), false for dual text+image models
    /// (e.g. Nano Banana, GPT Image).
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub image_only: Option<bool>,
    /// Local input images to edit/condition on (image-to-image / multi-image).
    /// Provide them to edit existing images; omit for plain text-to-image.
    #[serde(default)]
    pub images: Vec<ImageInput>,
    /// Longest-side cap (px) for input images before sending (default 800;
    /// env OPENROUTER_IMAGE_MAX_DIMENSION).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_image_dimension: Option<u32>,
    /// Number of variants to generate in parallel (1-16, seed-stepped). Default 1.
    /// With >1, files are named <output>-var-001, -002, ... and one manifest covers all.
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
}

/// Arguments for the `describe_image` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct DescribeImageArgs {
    /// Vision-capable model id (image input, text output), e.g.
    /// "google/gemini-2.5-flash" or "anthropic/claude-sonnet-4.6".
    pub model: String,
    /// Local image path(s) to describe (at least one required).
    pub images: Vec<ImageInput>,
    /// Instruction or question about the image(s). Defaults to a detailed description.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Longest-side cap (px) for input images before sending (default 800).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_image_dimension: Option<u32>,
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
        google/gemini-3.1-flash-image-preview) and save it. `output` is optional - omit it to \
        get an auto-named file (kind_datetime_model_config_seed_hash) under \
        OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp). For text-to-image, \
        pass a prompt. For editing / image-to-image, also pass local `images` (order \
        preserved; optional per-image label) - the prompt becomes the edit instruction. \
        Set variants>1 to generate several in parallel (seed-stepped). Returns a compact \
        result: saved image paths, decoded width/height, requested vs actual \
        aspect_ratio/image_size, seeds, a path to the sidecar manifest, and any mismatch \
        warnings. The output format (PNG or JPEG) is chosen by the provider and the \
        extension is set to match. Set image_only=true for models that only output images \
        (e.g. Grok/FLUX). This tool has NO defaults: model, prompt, aspect_ratio, \
        image_size and image_only must all be specified, or the call fails with an error \
        naming what is missing (output is optional, see above). Runs asynchronously: if the job is still going after \
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
    pub(crate) async fn run_generate(
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
        require_all("generate_image", "image", &missing)?;

        let aspect_ratio = args.aspect_ratio.clone();
        let image_size = args.image_size.clone();
        let req = GenerateRequest {
            model: args.model.clone(),
            prompt: args.prompt,
            aspect_ratio: args.aspect_ratio,
            image_size: args.image_size,
            seed: args.seed,
            image_only: args.image_only.unwrap_or(false),
            images: to_input_images(args.images),
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
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
        let base = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Image,
            &args.model,
            &config,
            args.seed,
        );
        let model = args.model;
        let variants_u64 = variants as u64;

        self.spawn_job_and_wait(
            TaskKind::Image,
            wait,
            inline_previews,
            move |ctx| async move {
                match image_gen::run_job(&ctx.client, &req, variants, &base, "inline").await {
                    Ok(summary) if !summary.images.is_empty() => {
                        let images = summary.images.len() as u64;
                        let cost: f64 = summary.images.iter().filter_map(|i| i.cost).sum();
                        let unknown =
                            summary.images.iter().filter(|i| i.cost.is_none()).count() as u64;
                        ctx.stats
                            .record_job(&model, variants_u64, images, cost, unknown)
                            .await;
                        Ok(image_job_result_json(&summary, &aspect_ratio, &image_size))
                    }
                    Ok(summary) => {
                        ctx.stats.record_job(&model, variants_u64, 0, 0.0, 0).await;
                        Err(format!(
                            "all {variants} variant(s) failed: {}",
                            summary.errors.join("; ")
                        ))
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
            images: to_input_images(args.images),
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
            output: Some(out.to_string_lossy().into_owned()),
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
            image_only: None,
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: None,
            output: Some("out.png".to_string()),
        };
        let err = server.run_generate(args, true).await.unwrap_err();
        assert!(err.message.contains("aspect_ratio"));
        assert!(err.message.contains("image_size"));
        assert!(err.message.contains("image_only"));
        assert!(err.message.contains("no defaults"));
    }

    /// Defense in depth: even with a scalar schema, clients that stringify all
    /// arguments must still deserialize. `image_only: "true"` is the exact payload
    /// from the bug report.
    #[test]
    fn generate_image_args_accept_stringified_scalars() {
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model": "x-ai/grok-imagine-image-quality",
            "prompt": "a small test image",
            "aspect_ratio": "1:1",
            "image_size": "1K",
            "image_only": "true",
            "seed": "42",
            "variants": "2",
            "output": "out.png",
        }))
        .expect("stringified scalars should deserialize");
        assert_eq!(args.image_only, Some(true));
        assert_eq!(args.seed, Some(42));
        assert_eq!(args.variants, Some(2));
    }

    /// Native typed values and absent/null optionals still work unchanged.
    #[test]
    fn generate_image_args_accept_native_and_absent_scalars() {
        let native: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "image_only": false, "seed": 7, "output": "o.png",
        }))
        .unwrap();
        assert_eq!(native.image_only, Some(false));
        assert_eq!(native.seed, Some(7));

        let absent: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "image_only": null, "output": "o.png",
        }))
        .unwrap();
        assert_eq!(absent.image_only, None);
        assert_eq!(absent.seed, None);
    }

    /// Garbage strings are rejected with a clear message rather than silently
    /// coerced.
    #[test]
    fn invalid_stringified_scalars_are_rejected() {
        let err = serde_json::from_value::<GenerateImageArgs>(json!({
            "model": "m", "prompt": "p", "image_only": "yes", "output": "o.png",
        }))
        .unwrap_err();
        assert!(err.to_string().contains("boolean"), "got: {err}");
    }
}
