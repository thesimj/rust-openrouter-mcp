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

use crate::openrouter::{ModelsQuery, OpenRouterClient};

/// MCP server wrapping an [`OpenRouterClient`].
#[derive(Clone)]
pub struct OpenRouterServer {
    client: OpenRouterClient,
    tool_router: ToolRouter<Self>,
}

impl OpenRouterServer {
    pub fn new(client: OpenRouterClient) -> Self {
        Self {
            client,
            tool_router: Self::tool_router(),
        }
    }
}

/// Arguments for the `list_models` tool. These map to OpenRouter's server-side
/// `GET /api/v1/models` query parameters, so filtering happens at the API.
#[derive(Debug, Deserialize, JsonSchema)]
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

/// Default number of models returned by `list_models` unless `all` is set.
const DEFAULT_MODEL_LIMIT: usize = 20;

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
        first 20 models by default; set all=true for the complete list."
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

        let mut models = self
            .client
            .list_models(&query)
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;

        if let Some(needle) = &args.search {
            models.retain(|m| m.matches_search(needle));
        }

        let total = models.len();
        if !args.all {
            models.truncate(DEFAULT_MODEL_LIMIT);
        }

        let mut json = serde_json::to_string_pretty(&models)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        if !args.all && total > models.len() {
            json = format!(
                "// showing {} of {} models; set \"all\": true to get the rest\n{}",
                models.len(),
                total,
                json
            );
        }

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for OpenRouterServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "MCP server for OpenRouter. Use `list_models` to discover models, \
                their capabilities, and pricing before invoking other tools.",
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
