//! The rmcp stdio MCP server and its tools.
//!
//! The server's tool implementations are split by domain into submodules; each
//! contributes a `#[tool_router]`-generated router that [`OpenRouterServer::new`]
//! combines into the single router the [`ServerHandler`] dispatches through.

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::router::tool::ToolRouter,
    model::{ServerCapabilities, ServerInfo},
    tool_handler,
    transport::stdio,
};

use crate::openrouter::OpenRouterClient;
use crate::stats::UsageStats;
use crate::tasks::TaskRegistry;

use caps::ModelCapsCache;

mod account;
mod audio;
mod caps;
mod chat;
mod image;
mod models;
mod naming;
mod result;
mod schema;
mod video;

#[cfg(test)]
mod test_support;

/// MCP server wrapping an [`OpenRouterClient`].
#[derive(Clone)]
pub struct OpenRouterServer {
    pub(crate) client: OpenRouterClient,
    pub(crate) tasks: TaskRegistry,
    pub(crate) stats: UsageStats,
    /// Cache of per-model input modalities, used to gate `chat_completion` image
    /// inputs against what the target model supports.
    pub(crate) model_caps: ModelCapsCache,
    pub(crate) tool_router: ToolRouter<Self>,
}

impl OpenRouterServer {
    pub fn new(client: OpenRouterClient) -> Self {
        Self {
            client,
            tasks: TaskRegistry::new(),
            stats: UsageStats::new(),
            model_caps: ModelCapsCache::new(),
            tool_router: Self::models_router()
                + Self::image_router()
                + Self::video_router()
                + Self::audio_router()
                + Self::chat_router()
                + Self::account_router(),
        }
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
