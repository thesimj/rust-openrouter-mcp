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
    work: std::sync::Arc<tokio::sync::Semaphore>,
    pub(crate) stats: UsageStats,
    /// Cache of per-model input modalities, used to gate `chat_completion` image
    /// inputs against what the target model supports.
    pub(crate) model_caps: ModelCapsCache,
    pub(crate) tool_router: ToolRouter<Self>,
}

impl OpenRouterServer {
    pub(crate) fn admit_work(&self) -> Result<tokio::sync::OwnedSemaphorePermit, rmcp::ErrorData> {
        self.work.clone().try_acquire_owned().map_err(|_| {
            rmcp::ErrorData::internal_error(
                "Too many synchronous calls; wait for one to finish before retrying.",
                None,
            )
        })
    }

    pub fn new(client: OpenRouterClient) -> Self {
        Self {
            client,
            tasks: TaskRegistry::new(),
            work: std::sync::Arc::new(tokio::sync::Semaphore::new(8)),
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
    let server = OpenRouterServer::new(client);
    let service = server.serve(stdio()).await?;
    let grace = std::env::var("OPENROUTER_MCP_SHUTDOWN_TIMEOUT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30)
        .min(300);
    supervise(service, std::time::Duration::from_secs(grace)).await
}

async fn supervise(
    service: rmcp::service::RunningService<rmcp::RoleServer, OpenRouterServer>,
    grace: std::time::Duration,
) -> anyhow::Result<()> {
    let tasks = service.service().tasks.clone();
    let work = service.service().work.clone();
    let cancellation = service.cancellation_token();
    let waiting = service.waiting();
    tokio::pin!(waiting);
    let outcome = tokio::select! {
        result = &mut waiting => result.map(|_| ()).map_err(anyhow::Error::from),
        signal = tokio::signal::ctrl_c() => {
            tasks.close_admission();
            work.close();
            cancellation.cancel();
            let result = waiting.await;
            signal.map_err(anyhow::Error::from).and_then(|_| result.map(|_| ()).map_err(anyhow::Error::from))
        }
    };
    work.close();
    tasks.shutdown(grace).await;
    outcome
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

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    #[tokio::test]
    async fn synchronous_capacity_is_shared_and_reopens() {
        let server = test_support::server_for("http://127.0.0.1:9".into());
        let clone = server.clone();
        let mut permits: Vec<_> = (0..8).map(|_| server.admit_work().unwrap()).collect();
        assert!(clone.admit_work().is_err());
        permits.pop();
        assert!(clone.admit_work().is_ok());
    }

    #[tokio::test]
    async fn transport_eof_drains_generation_jobs() {
        let server = test_support::server_for("http://127.0.0.1:9".into());
        let tasks = server.tasks.clone();
        let reservation = tasks.reserve(crate::tasks::TaskKind::Image).unwrap();
        let id = reservation.id.clone();
        let (finish, finish_rx) = tokio::sync::oneshot::channel();
        let done = tasks
            .start(reservation, async move {
                finish_rx.await.unwrap();
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                Ok(serde_json::json!({"saved":true}))
            })
            .unwrap();
        let (client_io, server_io) = tokio::io::duplex(8192);
        let service = tokio::spawn(async move {
            let service = server.serve(server_io).await.unwrap();
            supervise(service, std::time::Duration::from_secs(1))
                .await
                .unwrap();
        });
        let (reader, mut writer) = tokio::io::split(client_io);
        writer.write_all(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"test\",\"version\":\"1\"}}}\n").await.unwrap();
        let mut reader = BufReader::new(reader);
        let mut response = String::new();
        reader.read_line(&mut response).await.unwrap();
        assert!(response.contains("result"));
        writer
            .write_all(b"{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n")
            .await
            .unwrap();
        drop(writer);
        drop(reader);
        finish.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(3), service)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(tasks.snapshot(&id).await.unwrap().status, "completed");
        done.await.unwrap();
        assert!(tasks.reserve(crate::tasks::TaskKind::Image).is_none());
    }
}
