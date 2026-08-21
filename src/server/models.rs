//! The `list_models` tool and its argument struct.

use rmcp::{
    ErrorData,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::openrouter::{ModelsQuery, apply_filters};
use crate::pricing::{attach_pricing_human, humanize_pricing, models_to_json};
use crate::server::schema::{de_bool, de_opt_uint, scalarize_nullable};

use super::OpenRouterServer;

/// Arguments for the `list_models` tool. These map to OpenRouter's server-side
/// `GET /api/v1/models` query parameters, so filtering happens at the API.
#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct ListModelsArgs {
    /// Server-side free-text search by model name or slug (e.g. "claude").
    #[serde(default)]
    pub query: Option<String>,
    /// Local case-insensitive filter across id, name, and description
    /// (e.g. "openai"). Order of operations: the server-side query/
    /// output_modalities/input_modalities/supported_parameters/sort/min_context
    /// filters run first (one API call); this local filter narrows that result
    /// next; the default 20-result cap (unless all=true) is applied last.
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
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub min_context: Option<u64>,
    /// Return all matching models. By default only the first 20 are returned to
    /// keep the result compact; set true to get the complete list.
    #[serde(default, deserialize_with = "de_bool")]
    pub all: bool,
}

/// Arguments for the `describe_model` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct DescribeModelArgs {
    /// Exact model id ("author/slug"), e.g. "anthropic/claude-opus-4.7". Use
    /// list_models to discover ids.
    pub model: String,
}

#[tool_router(router = models_router, vis = "pub(crate)")]
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

        // Shared enrichment: attach a human-readable pricing_human to each
        // model (same path the CLI uses, so the two never diverge).
        let mut json = serde_json::to_string_pretty(&models_to_json(&filtered.models))
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        if filtered.truncated() > 0 {
            json = format!(
                "// showing {} of {} models; set \"all\": true to get the rest\n{}",
                filtered.models.len(),
                filtered.total,
                json
            );
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
    }

    #[tool(
        description = "Get the full detail for a single OpenRouter model by its exact id \
        (author/slug, e.g. \"anthropic/claude-opus-4.7\" - discover ids with list_models). \
        Returns everything OpenRouter reports for that model as JSON: the model object \
        (description, architecture/modalities, tokenizer, context_length, knowledge_cutoff, \
        benchmarks) plus the per-provider endpoints with their pricing, uptime, status, \
        quantization, max tokens, and supported parameters - richer and more current than the \
        list_models entry (which is a compact subset). For video models, also merges the real \
        pricing under a \"video\" key (pricing_skus, supported resolutions/durations/sizes from \
        /videos/models), since the token-based pricing is 0 and misleading for video. For image \
        models, also merges per-endpoint image capabilities under an \"image\" key \
        (supported_parameters, allowed_passthrough_parameters, pricing, supports_streaming from \
        /images/models/{author}/{slug}/endpoints) - best-effort, omitted if that model has no \
        image endpoint. Fails if the id is unknown.",
        annotations(
            title = "Describe OpenRouter Model",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn describe_model(
        &self,
        Parameters(args): Parameters<DescribeModelArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let model = args.model.trim();
        if model.is_empty() {
            return Err(ErrorData::invalid_params(
                "model is required (an exact id, e.g. \"anthropic/claude-opus-4.7\")".to_string(),
                None,
            ));
        }

        let mut detail = self
            .client
            .describe_model(model)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        // Video models price via a separate SKU endpoint; the token-based
        // pricing on the main record is 0 and misleading. Merge the real
        // pricing_skus + supported resolutions/durations/sizes under "video".
        let outputs_video = detail["architecture"]["output_modalities"]
            .as_array()
            .is_some_and(|m| m.iter().any(|v| v == "video"));
        if outputs_video {
            match self.client.video_model_detail(model).await {
                Ok(Some(mut video)) => {
                    // pricing_skus is the video model's real pricing object.
                    if let Some(human) = video.get("pricing_skus").and_then(humanize_pricing) {
                        video["pricing_skus_human"] = human;
                    }
                    detail["video"] = video;
                }
                Ok(None) => {}
                // Surface the failure instead of silently returning the
                // misleading 0 token pricing as if it were complete.
                Err(e) => {
                    detail["video_pricing_error"] =
                        Value::String(format!("could not fetch /videos/models: {e:#}"));
                }
            }
        }

        // Image models: merge the per-endpoint detail (definitive
        // supported_parameters, allowed_passthrough_parameters, pricing,
        // supports_streaming) from the dedicated image-models endpoint, the
        // same best-effort way the video block above is merged.
        let outputs_image = detail["architecture"]["output_modalities"]
            .as_array()
            .is_some_and(|m| m.iter().any(|v| v == "image"));
        if outputs_image {
            match self.client.image_model_detail(model).await {
                Ok(Some(image)) => {
                    detail["image"] = image;
                }
                Ok(None) => {}
                Err(e) => {
                    detail["image_pricing_error"] =
                        Value::String(format!("could not fetch /images/models: {e:#}"));
                }
            }
        }

        // Normalize every pricing block to human "$X/M tokens" form alongside
        // the raw decimals: the top-level record and each per-provider endpoint.
        attach_pricing_human(&mut detail);
        if let Some(endpoints) = detail.get_mut("endpoints").and_then(Value::as_array_mut) {
            for ep in endpoints {
                attach_pricing_human(ep);
            }
        }

        let json = serde_json::to_string_pretty(&detail)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use rmcp::handler::server::wrapper::Parameters;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

    /// F2: `Model` must not silently drop `supported_voices` - it is what
    /// generate_audio points callers at, and OpenRouter's speech models carry
    /// it on `GET /models?output_modalities=speech` (live-confirmed).
    #[tokio::test]
    async fn list_models_tool_carries_supported_voices_through() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{
                    "id": "hexgrad/kokoro-82m",
                    "name": "Kokoro 82M",
                    "supported_voices": ["af_heart", "af_bella"]
                }]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .list_models(Parameters(ListModelsArgs::default()))
            .await
            .unwrap();

        let v = tool_result_json(&result);
        assert_eq!(v[0]["supported_voices"][0], "af_heart");
        assert_eq!(v[0]["supported_voices"][1], "af_bella");
    }

    #[tokio::test]
    async fn describe_model_tool_returns_full_detail_json() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/anthropic/claude-opus-4.7/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "anthropic/claude-opus-4.7",
                    "endpoints": [{"provider_name": "Anthropic", "context_length": 1000000}]
                }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "anthropic/claude-opus-4.7".to_string(),
            }))
            .await
            .unwrap();

        let body = serde_json::to_string(&result).unwrap();
        assert!(body.contains("anthropic/claude-opus-4.7"));
        assert!(body.contains("Anthropic"));
    }

    #[tokio::test]
    async fn describe_model_tool_merges_video_pricing() {
        let mock = MockServer::start().await;
        // Main detail: a video-output model with the misleading 0 token pricing.
        Mock::given(method("GET"))
            .and(path("/models/bytedance/seedance-2.0/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "bytedance/seedance-2.0",
                    "architecture": {"output_modalities": ["video"]},
                    "endpoints": [{"provider_name": "Seed", "pricing": {"prompt": "0", "completion": "0"}}]
                }
            })))
            .mount(&mock)
            .await;
        // The real pricing lives in /videos/models under pricing_skus.
        Mock::given(method("GET"))
            .and(path("/videos/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {"id": "other/model", "pricing_skus": {"generate": "0.50"}},
                    {"id": "bytedance/seedance-2.0", "pricing_skus": {"video_tokens": "0.000007"}}
                ]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "bytedance/seedance-2.0".to_string(),
            }))
            .await
            .unwrap();

        let body = serde_json::to_string(&result).unwrap();
        // The merged "video" block carries the matching entry's real SKU pricing.
        assert!(body.contains("video_tokens"));
        assert!(body.contains("0.000007"));
        // ...and not some other model's SKU.
        assert!(!body.contains("\"generate\""));
    }

    #[tokio::test]
    async fn describe_model_tool_merges_image_endpoint_detail() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/openai/gpt-image-2/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "openai/gpt-image-2",
                    "architecture": {"output_modalities": ["image"]},
                    "endpoints": [{"provider_name": "OpenAI"}]
                }
            })))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/images/models/openai/gpt-image-2/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "openai/gpt-image-2",
                    "endpoints": [{"supported_parameters": ["quality", "output_format"]}]
                }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "openai/gpt-image-2".to_string(),
            }))
            .await
            .unwrap();

        let v = tool_result_json(&result);
        assert_eq!(
            v["image"]["endpoints"][0]["supported_parameters"][0],
            "quality"
        );
    }

    /// A model with no image endpoint (404) is silently omitted, not a failure.
    #[tokio::test]
    async fn describe_model_tool_omits_image_block_on_404() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/some/image-model/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "some/image-model",
                    "architecture": {"output_modalities": ["image"]}
                }
            })))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/images/models/some/image-model/endpoints"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "some/image-model".to_string(),
            }))
            .await
            .unwrap();

        let body = serde_json::to_string(&result).unwrap();
        assert!(!body.contains("\"image\":"));
        assert!(!body.contains("image_pricing_error"));
    }

    /// A non-404 failure is surfaced inline rather than failing the whole tool.
    #[tokio::test]
    async fn describe_model_tool_surfaces_image_fetch_error_without_failing() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/some/image-model/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "some/image-model",
                    "architecture": {"output_modalities": ["image"]}
                }
            })))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/images/models/some/image-model/endpoints"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "some/image-model".to_string(),
            }))
            .await
            .unwrap();

        let body = serde_json::to_string(&result).unwrap();
        assert!(body.contains("image_pricing_error"));
    }

    #[tokio::test]
    async fn describe_model_tool_requires_model_id() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let err = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "   ".to_string(),
            }))
            .await
            .unwrap_err();
        assert!(err.message.contains("model is required"));
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
        assert!(err.message.contains("500"), "got: {}", err.message);
    }
}
