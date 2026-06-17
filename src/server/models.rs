//! The `list_models` tool and its argument struct.

use rmcp::{
    ErrorData,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::openrouter::{ModelsQuery, apply_filters};
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
        description = "Get the full detail for a single OpenRouter model by its exact id \
        (author/slug, e.g. \"anthropic/claude-opus-4.7\" - discover ids with list_models). \
        Returns everything OpenRouter reports for that model as JSON: the model object \
        (description, architecture/modalities, tokenizer, context_length, knowledge_cutoff, \
        benchmarks) plus the per-provider endpoints with their pricing, uptime, status, \
        quantization, max tokens, and supported parameters - richer and more current than the \
        list_models entry (which is a compact subset). Fails if the id is unknown.",
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

        let detail = self
            .client
            .describe_model(model)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        let json = serde_json::to_string_pretty(&detail)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::server_for;
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
        assert!(err.message.contains("error status"));
    }
}
