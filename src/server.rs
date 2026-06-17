//! The rmcp stdio MCP server and its tools.

use rmcp::{
    ErrorData, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::stdio,
};
use schemars::JsonSchema;
use serde::Deserialize;

use std::path::PathBuf;

use serde_json::json;

use crate::image_gen::{self, GenerateRequest};
use crate::openrouter::{ModelsQuery, OpenRouterClient, apply_filters};
use crate::stats::UsageStats;
use crate::tasks::{TaskKind, TaskRegistry, TaskSnapshot};

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

/// Build the lean per-job result object (paths, dims, requested vs actual,
/// manifest pointer, plus warnings/errors when present).
fn job_result_json(
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

/// Wrap a task snapshot into the response envelope returned by `generate_image`
/// (fast path) and `get_result`: the completed result, an error, or a pending
/// note — always carrying `task_id`, `status`, and `kind`.
fn snapshot_to_envelope(task_id: &str, snap: &TaskSnapshot) -> serde_json::Value {
    let mut env = match snap.status {
        "completed" => snap.result.clone().unwrap_or_else(|| json!({ "ok": true })),
        "failed" => json!({ "ok": false, "error": snap.error }),
        _ => json!({
            "ok": true,
            "message": format!("still generating — call get_result with task_id \"{task_id}\""),
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
    /// embeddings, video, rerank, speech, transcription — or "all". Defaults to
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
    /// With >1, files are named <output>-var-001, -002, … and one manifest covers all.
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

/// Arguments for the `get_result` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetResultArgs {
    /// The task_id returned by generate_image (or a future generate_video).
    pub task_id: String,
}

/// Arguments for the `reset_usage_stats` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResetUsageStatsArgs {
    /// Must be true to confirm — this clears all in-memory usage counters.
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
        video, rerank, speech, transcription (default is text only — pass \
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
        preserved; optional per-image label) — the prompt becomes the edit instruction. \
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
                    "generate_image has no defaults — specify every parameter explicitly. \
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
                            job_result_json(&summary, &aspect_ratio, &image_size),
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
        let body = serde_json::to_string_pretty(&env)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(body)]))
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
    ) -> Result<CallToolResult, ErrorData> {
        match self.tasks.snapshot(&args.task_id).await {
            Some(snap) => {
                let env = snapshot_to_envelope(&args.task_id, &snap);
                let body = serde_json::to_string_pretty(&env)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                Ok(CallToolResult::success(vec![Content::text(body)]))
            }
            None => Err(ErrorData::invalid_params(
                format!(
                    "unknown task_id \"{}\" (tasks are in-memory per server process and lost on restart)",
                    args.task_id
                ),
                None,
            )),
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
        (GET /api/v1/key): label, creator_user_id (the owning user — the closest available \
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
                images with an image-capable model. If `generate_image` returns \
                status \"pending\" with a task_id, poll `get_result` until it is \
                \"completed\". `get_usage_stats` reports this process's spend and counts.",
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

    // 1x1 PNG (header is valid; full decode is not exercised on this path).
    const PNG_1X1_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

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

    #[tokio::test]
    async fn generate_image_runs_async_and_get_result_fetches_it() {
        let mock = MockServer::start().await;
        let data_url = format!("data:image/png;base64,{PNG_1X1_B64}");
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
        let res = server.generate_image(Parameters(args)).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["status"], "completed");
        assert_eq!(v["kind"], "image");
        assert!(v["images"][0]["path"].is_string());
        let task_id = v["task_id"].as_str().unwrap().to_string();

        // The same task is retrievable by id.
        let res2 = server
            .get_result(Parameters(GetResultArgs {
                task_id: task_id.clone(),
            }))
            .await
            .unwrap();
        let v2 = tool_result_json(&res2);
        assert_eq!(v2["status"], "completed");
        assert_eq!(v2["task_id"], task_id);
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
            .get_result(Parameters(GetResultArgs {
                task_id: "nope".to_string(),
            }))
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
        let err = server.generate_image(Parameters(args)).await.unwrap_err();
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
