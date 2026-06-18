//! Account/key info, usage-stats tools, and job-result retrieval (`get_result`).

use rmcp::{
    ErrorData, RoleServer,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    service::RequestContext,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::server::result::{client_wants_inline_previews, job_call_result, snapshot_to_envelope};
use crate::server::schema::{de_bool, scalarize_nullable};

use super::OpenRouterServer;

/// Arguments for the `get_result` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct GetResultArgs {
    /// The task_id returned by generate_image (or a future generate_video).
    pub task_id: String,
}

/// Arguments for the `reset_usage_stats` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct ResetUsageStatsArgs {
    /// Must be true to confirm - this clears all in-memory usage counters.
    #[serde(default, deserialize_with = "de_bool")]
    pub confirm: bool,
}

#[tool_router(router = account_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Fetch the status and result of a generation job by task_id (returned \
        by generate_image when a job is still running after its fast-return window). Returns \
        status pending|completed|failed; when completed, the same lean result (image paths, \
        dimensions, manifest) generate_image would have returned. Tasks are in-memory per \
        server process and are lost if the server restarts.",
        annotations(
            title = "Get Job Result",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = false
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
    pub(crate) async fn run_get_result(
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
        description = "Return in-memory usage statistics for this server process: version (the \
        server build version), started_at, \
        uptime_seconds, requests_total, requests_failed, image_generations, images_generated, \
        text_generations (describe_image and chat_completion calls), actual_cost_usd \
        (summed from usage.cost), \
        unknown_cost_count, and a by_model breakdown. Counters reset when the server restarts.",
        annotations(
            title = "Get Usage Stats",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = false
        )
    )]
    pub(crate) async fn get_usage_stats(&self) -> Result<CallToolResult, ErrorData> {
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
            idempotent_hint = true,
            open_world_hint = false
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::audio::GenerateAudioArgs;
    use crate::server::models::ListModelsArgs;
    use crate::server::test_support::{server_for, tool_result_json};
    use crate::server::video::GenerateVideoArgs;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

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

    /// Stringified booleans/floats are also accepted for the other tools.
    #[test]
    fn other_tools_accept_stringified_scalars() {
        let lm: ListModelsArgs = serde_json::from_value(json!({ "all": "true" })).unwrap();
        assert!(lm.all);
        let reset: ResetUsageStatsArgs =
            serde_json::from_value(json!({ "confirm": "false" })).unwrap();
        assert!(!reset.confirm);
        let vid: GenerateVideoArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "generate_audio": "true", "duration": "8", "output": "o.mp4",
        }))
        .unwrap();
        assert_eq!(vid.generate_audio, Some(true));
        assert_eq!(vid.duration, Some(8));
        let aud: GenerateAudioArgs = serde_json::from_value(json!({
            "model": "m", "speed": "1.5", "output": "o.mp3",
        }))
        .unwrap();
        assert_eq!(aud.speed, Some(1.5));
    }
}
