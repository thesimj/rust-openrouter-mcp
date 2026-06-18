//! The `chat_completion` text tool and its argument struct.

use rmcp::{
    ErrorData,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::chat_gen;
use crate::image_gen;
use crate::server::schema::{de_opt_f64, de_opt_uint, require_all, scalarize_nullable};

use super::OpenRouterServer;
use super::image::{ImageInput, check_image_input, resolve_image_inputs};

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
    /// Optional input images for a vision-capable model (image-in / text-out).
    /// Each takes exactly one of: path (local file), url (http/https, fetched),
    /// or base64 (a data: URL or raw base64). Best-effort gated on the model's
    /// declared input modalities: the call is rejected only when the catalog
    /// reports the model does NOT accept image input; if its capabilities can't
    /// be determined the request is sent anyway. Omit for a plain text prompt.
    #[serde(default)]
    pub images: Vec<ImageInput>,
    /// Longest-side cap (px) for input images before sending (default 800,
    /// capped at 800). Ignored when no images are provided.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_image_dimension: Option<u32>,
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
        reply (text out). This is a synchronous, fast call (not a background task). Useful \
        to route a sub-task to a DIFFERENT model than the host - e.g. ask a cheaper or specialized \
        model on OpenRouter. Provide `model` (a chat model id; discover with list_models) and \
        `prompt` (the user message); both are required or the call fails naming what is missing. \
        `system` (an optional system instruction), `temperature`, and `max_tokens` are optional. \
        Optionally pass `images` (path/url/base64) to ask a VISION-capable model about them; the \
        call is rejected only when the model is known not to accept image input (use list_models \
        with input_modalities=image to find one). Returns the assistant's text.",
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

        // Validate each image's shape cheaply (no fetch) so a malformed entry
        // reports the accurate "exactly one of ..." error rather than being
        // masked by the capability gate below. Then gate image input on the
        // model's declared capabilities before any network-bound resolution.
        // Both are skipped for text-only calls.
        if !args.images.is_empty() {
            for img in &args.images {
                check_image_input(img)?;
            }
            self.ensure_image_input_supported(&args.model).await?;
        }
        let images = resolve_image_inputs(args.images).await?;
        // The dimension cap only matters when there are images to normalize.
        let max_dim = if images.is_empty() {
            0
        } else {
            image_gen::resolve_max_dimension(args.max_image_dimension)
        };

        let prompt = args.prompt.unwrap_or_default();
        // Record success only after text is actually extracted (an empty-choices
        // or empty-content response is an error, not a successful generation).
        match chat_gen::complete(
            &self.client,
            &chat_gen::ChatInputs {
                model: &args.model,
                system: args.system.as_deref(),
                prompt: &prompt,
                temperature: args.temperature,
                max_tokens: args.max_tokens,
                images: &images,
                max_image_dimension: max_dim,
            },
        )
        .await
        {
            Ok(result) => {
                self.stats.record_text(&args.model, true, result.cost).await;
                Ok(CallToolResult::success(vec![Content::text(result.text)]))
            }
            Err(e) => {
                self.stats.record_text(&args.model, false, None).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }

    /// Best-effort early rejection when `model` is *known* not to accept image
    /// input. The model's input modalities are looked up via list_models and
    /// cached (after the first call completes; a burst of concurrent first-time
    /// calls for the same model may each fetch).
    ///
    /// This is deliberately fail-open: the lookup is a fuzzy catalog search, so
    /// if it errors (network blip, an id the search doesn't surface, a routing-
    /// suffixed id like `:nitro`/`:floor`) or reports no modality metadata, the
    /// request is allowed through and the actual `/chat/completions` call remains
    /// the authority on compatibility. We reject only when the catalog positively
    /// reports input modalities that don't include images — the common, clear
    /// case (e.g. sending an image to a text-only model).
    async fn ensure_image_input_supported(&self, model: &str) -> Result<(), ErrorData> {
        let modalities = match self.model_caps.get(model).await {
            Some(cached) => cached,
            None => match self.client.model_input_modalities(model).await {
                Ok(modalities) => {
                    // Cache only a definite answer; an empty list means "unknown"
                    // (missing/lagging catalog metadata) and must not be pinned for
                    // the process lifetime.
                    if !modalities.is_empty() {
                        self.model_caps.put(model, modalities.clone()).await;
                    }
                    modalities
                }
                // Capabilities couldn't be verified — don't block a possibly-valid call.
                Err(_) => return Ok(()),
            },
        };
        if !modalities.is_empty() && !modalities.iter().any(|m| m == "image") {
            return Err(ErrorData::invalid_params(
                format!(
                    "model '{model}' does not accept image input (input modalities: [{}]). \
                     Use list_models with input_modalities=image to find a vision-capable model.",
                    modalities.join(", ")
                ),
                None,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json, valid_png_b64};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// One inline-base64 image input.
    fn one_image() -> Vec<ImageInput> {
        vec![ImageInput {
            path: None,
            url: None,
            base64: Some(valid_png_b64()),
            label: None,
        }]
    }

    /// Mock `GET /models` so a single model reports the given input modalities.
    async fn mock_model_modalities(mock: &MockServer, id: &str, input_modalities: &[&str]) {
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{
                    "id": id,
                    "architecture": { "input_modalities": input_modalities }
                }]
            })))
            .mount(mock)
            .await;
    }

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
            images: Vec::new(),
            max_image_dimension: None,
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
            images: Vec::new(),
            max_image_dimension: None,
            temperature: None,
            max_tokens: None,
        };
        let err = server.run_chat_completion(args).await.unwrap_err();
        assert!(err.message.contains("prompt"));
        assert!(err.message.contains("no defaults"));
    }

    #[tokio::test]
    async fn chat_completion_sends_image_to_vision_model() {
        let mock = MockServer::start().await;
        // The model declares image input, so the request is allowed through...
        mock_model_modalities(&mock, "google/gemini-2.5-flash", &["text", "image"]).await;
        // ...and the user message is sent as a text-part + image-part array.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text"},
                        {"type": "image_url"}
                    ]}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "a tiny blue square"}}],
                "usage": {"cost": 0.001}
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let args = ChatCompletionArgs {
            model: "google/gemini-2.5-flash".to_string(),
            prompt: Some("what is this?".to_string()),
            system: None,
            images: one_image(),
            max_image_dimension: None,
            temperature: None,
            max_tokens: None,
        };
        let res = server.run_chat_completion(args).await.unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "a tiny blue square");
    }

    #[tokio::test]
    async fn chat_completion_rejects_image_for_text_only_model() {
        let mock = MockServer::start().await;
        // Only `/models` is mocked: a text-only model must be rejected BEFORE any
        // call to `/chat/completions` (which has no mock and would 404).
        mock_model_modalities(&mock, "openai/gpt-5.4", &["text"]).await;

        let server = server_for(mock.uri());
        let args = ChatCompletionArgs {
            model: "openai/gpt-5.4".to_string(),
            prompt: Some("what is this?".to_string()),
            system: None,
            images: one_image(),
            max_image_dimension: None,
            temperature: None,
            max_tokens: None,
        };
        let err = server.run_chat_completion(args).await.unwrap_err();
        assert!(err.message.contains("does not accept image input"));
        // No generation should have been recorded (rejected before the API call).
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 0);
    }

    #[tokio::test]
    async fn chat_completion_allows_image_when_modalities_unknown() {
        // The catalog entry matches by id but reports no architecture/modalities.
        // The gate must fail open (treat "unknown" as allowed), not reject with [].
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "obscure/vision-model" }]
            })))
            .mount(&mock)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "looks like an owl"}}]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let args = ChatCompletionArgs {
            model: "obscure/vision-model".to_string(),
            prompt: Some("what is this?".to_string()),
            system: None,
            images: one_image(),
            max_image_dimension: None,
            temperature: None,
            max_tokens: None,
        };
        let res = server.run_chat_completion(args).await.unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "looks like an owl");
    }

    #[tokio::test]
    async fn chat_completion_allows_image_when_capability_lookup_misses() {
        // A routing-suffixed / search-missed id isn't surfaced by the `q` search,
        // so model_input_modalities errors. The gate must fail open rather than
        // hard-reject a call the real /chat/completions would accept.
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "some/other-model", "architecture": {"input_modalities": ["text"]} }]
            })))
            .mount(&mock)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "described"}}]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let args = ChatCompletionArgs {
            model: "google/gemini-2.5-flash:nitro".to_string(),
            prompt: Some("what is this?".to_string()),
            system: None,
            images: one_image(),
            max_image_dimension: None,
            temperature: None,
            max_tokens: None,
        };
        let res = server.run_chat_completion(args).await.unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "described");
    }
}
