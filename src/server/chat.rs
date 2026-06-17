//! The `chat_completion` text tool and its argument struct.

use rmcp::{
    ErrorData,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::openrouter::{ChatRequest, Content as ChatContent, Message};
use crate::server::schema::{de_opt_f64, de_opt_uint, require_all, scalarize_nullable};

use super::OpenRouterServer;

/// Arguments for the `chat_completion` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct ChatCompletionArgs {
    /// Chat/text model id, e.g. "openai/gpt-5.4" or "anthropic/claude-sonnet-4.6".
    /// Discover ids with list_models.
    pub model: String,
    /// REQUIRED: the user message / prompt text to send to the model.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Optional system instruction prepended as a system message.
    #[serde(default)]
    pub system: Option<String>,
    /// Optional sampling temperature.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub temperature: Option<f64>,
    /// Optional maximum number of tokens to generate.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_tokens: Option<u64>,
}

#[tool_router(router = chat_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Send a prompt to any OpenRouter chat/text model and return the model's text \
        reply (text in, text out). This is a synchronous, fast call (not a background task). Useful \
        to route a sub-task to a DIFFERENT model than the host - e.g. ask a cheaper or specialized \
        model on OpenRouter. Provide `model` (a chat model id; discover with list_models) and \
        `prompt` (the user message); both are required or the call fails naming what is missing. \
        `system` (an optional system instruction), `temperature`, and `max_tokens` are optional. \
        Returns the assistant's text.",
        annotations(
            title = "Chat Completion",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn chat_completion(
        &self,
        Parameters(args): Parameters<ChatCompletionArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.run_chat_completion(args).await
    }

    /// Core of `chat_completion` (synchronous), split out so tests drive it directly.
    pub(crate) async fn run_chat_completion(
        &self,
        args: ChatCompletionArgs,
    ) -> Result<CallToolResult, ErrorData> {
        let mut missing: Vec<&str> = Vec::new();
        if args
            .prompt
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            missing.push("prompt (the user message)");
        }
        require_all("chat_completion", "text", &missing)?;

        let model = args.model.clone();

        let mut messages: Vec<Message> = Vec::new();
        if let Some(system) = args
            .system
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            messages.push(Message {
                role: "system".into(),
                content: ChatContent::Text(system.to_string()),
            });
        }
        messages.push(Message {
            role: "user".into(),
            content: ChatContent::Text(args.prompt.unwrap_or_default()),
        });

        let req = ChatRequest {
            model: model.clone(),
            messages,
            modalities: None,
            image_config: None,
            seed: None,
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            stream: false,
        };

        match self.client.chat_completion(&req).await {
            Ok(resp) => {
                let cost = resp.completion.usage.and_then(|u| u.cost);
                self.stats.record_text(&model, true, cost).await;
                let choice = resp.completion.choices.into_iter().next().ok_or_else(|| {
                    ErrorData::internal_error("OpenRouter returned no choices", None)
                })?;
                let text = choice
                    .message
                    .content
                    .filter(|t| !t.is_empty())
                    .ok_or_else(|| ErrorData::internal_error("model returned no text", None))?;
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => {
                self.stats.record_text(&model, false, None).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn chat_completion_returns_model_text() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            // Confirms system+user messages and that temperature/max_tokens are
            // actually forwarded in the request body (not silently dropped).
            .and(body_partial_json(serde_json::json!({
                "model": "openai/gpt-5.4",
                "temperature": 0.5,
                "max_tokens": 64,
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "say hi"}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "hello back"}}],
                "usage": {"cost": 0.0012}
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let args = ChatCompletionArgs {
            model: "openai/gpt-5.4".to_string(),
            prompt: Some("say hi".to_string()),
            system: Some("be terse".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(64),
        };
        let res = server.run_chat_completion(args).await.unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "hello back");

        // The text generation and its cost were recorded.
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 1);
    }

    #[tokio::test]
    async fn chat_completion_requires_prompt() {
        // Validation runs before any HTTP call.
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = ChatCompletionArgs {
            model: "m".to_string(),
            prompt: Some("   ".to_string()), // blank-after-trim counts as missing
            system: None,
            temperature: None,
            max_tokens: None,
        };
        let err = server.run_chat_completion(args).await.unwrap_err();
        assert!(err.message.contains("prompt"));
        assert!(err.message.contains("no defaults"));
    }
}
