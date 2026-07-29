//! The rmcp stdio MCP server and its tools.
//!
//! The server's tool implementations are split by domain into submodules; each
//! contributes a `#[tool_router]`-generated router that [`OpenRouterServer::new`]
//! combines into the single router the [`ServerHandler`] dispatches through.

use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::router::tool::ToolRouter,
    model::{Implementation, ServerCapabilities, ServerInfo},
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
    /// Advertises the protocol version rmcp treats as current
    /// ([`ProtocolVersion::default`], i.e. `LATEST`). rmcp 3.0 also knows
    /// `2026-07-28` (stateless lifecycle, MRTR, tasks extension) but does not
    /// default to it: over stdio those changes buy us nothing, and naming a
    /// version ahead of what clients speak only risks a failed handshake.
    /// Opting in is a deliberate change - see the test below.
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(
                "MCP server for OpenRouter. Use `list_models` to discover models, \
                their capabilities, and pricing, then `generate_image` to create \
                images, `generate_video` to create videos (slow, async: it returns \
                status \"pending\" with a task_id - poll `get_result` until \
                \"completed\"), `generate_audio` for text-to-speech and \
                `transcribe_audio` for speech-to-text (both synchronous). \
                If `generate_image` or `generate_video` returns status \"pending\" with \
                a task_id, poll `get_result` until it is \"completed\". \
                `get_usage_stats` reports this process's spend and counts.",
            );
        // rmcp's default `Implementation::from_build_env()` expands
        // `env!("CARGO_CRATE_NAME")` inside the rmcp crate, so it reports the SDK
        // ("rmcp", at rmcp's version) as the server. Clients show this name, so
        // name ourselves.
        info.server_info = Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        info
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
    use super::*;
    use rmcp::model::ProtocolVersion;

    /// Pins the handshake we advertise. A silent bump here changes what every
    /// client negotiates, so moving to `2026-07-28` must be a deliberate edit
    /// (and an rmcp upgrade must not do it for us).
    #[test]
    fn advertises_the_sdk_default_protocol_version_and_tools_only() {
        let info =
            crate::server::test_support::server_for("http://127.0.0.1:9".to_string()).get_info();
        assert_eq!(info.protocol_version, ProtocolVersion::V_2025_11_25);
        // We identify as ourselves, not as the SDK (rmcp's default).
        assert_eq!(info.server_info.name, "openrouter-mcp");
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
        assert!(info.capabilities.tools.is_some(), "tools are advertised");
        // Nothing else is served, so nothing else may be advertised.
        assert!(info.capabilities.prompts.is_none());
        assert!(info.capabilities.resources.is_none());
        assert!(info.instructions.is_some());
    }
}
