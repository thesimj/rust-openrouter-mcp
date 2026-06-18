//! `GET /api/v1/models` and `GET /api/v1/models/{id}/endpoints`.

use anyhow::{Context, Result};
use serde_json::Value;

use crate::openrouter::{Model, ModelsQuery, ModelsResponse, OpenRouterClient};

impl OpenRouterClient {
    /// The `input_modalities` declared for a single model id (e.g.
    /// `["text", "image"]`). Searches `/models?q=<id>` and returns the
    /// architecture of the entry whose id matches `model_id` exactly. Errors if
    /// no such model is found. Used to gate multimodal inputs before sending.
    pub async fn model_input_modalities(&self, model_id: &str) -> Result<Vec<String>> {
        let query = ModelsQuery {
            q: Some(model_id.to_string()),
            ..Default::default()
        };
        let model = self
            .list_models(&query)
            .await?
            .into_iter()
            .find(|m| m.id == model_id)
            .with_context(|| format!("model '{model_id}' not found on OpenRouter"))?;
        Ok(model
            .architecture
            .map(|a| a.input_modalities)
            .unwrap_or_default())
    }

    /// `GET /api/v1/models` - every model with capabilities and pricing.
    ///
    /// `query` carries OpenRouter's server-side filters (modalities, sort,
    /// free-text, price/context bounds, ...) so the API does the filtering.
    pub async fn list_models(&self, query: &ModelsQuery) -> Result<Vec<Model>> {
        let resp = self
            .http
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .query(&query.to_pairs())
            .send()
            .await
            .context("request to OpenRouter /models failed")?
            .error_for_status()
            .context("OpenRouter /models returned an error status")?;

        let parsed: ModelsResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /models response")?;
        Ok(parsed.data)
    }

    /// `GET /api/v1/models/{model_id}/endpoints` - the full record for one model:
    /// the model object (id, description, architecture, context) plus the
    /// per-provider endpoints (pricing, uptime, status, quantization, supported
    /// parameters). Returned as raw JSON so the caller surfaces everything
    /// OpenRouter reports without a hand-maintained schema. `model_id` is the
    /// `author/slug` id (e.g. "anthropic/claude-opus-4.7").
    pub async fn describe_model(&self, model_id: &str) -> Result<Value> {
        let resp = self
            .http
            .get(format!("{}/models/{}/endpoints", self.base_url, model_id))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("request to OpenRouter /models/{id}/endpoints failed")?
            .error_for_status()
            .context("OpenRouter /models/{id}/endpoints returned an error status")?;

        let mut body: Value = resp
            .json()
            .await
            .context("failed to decode OpenRouter model-endpoints response")?;
        // Unwrap the `data` envelope (model + endpoints) when present.
        Ok(body.get_mut("data").map(Value::take).unwrap_or(body))
    }

    /// `GET /api/v1/videos/models`, returning the entry whose `id` matches
    /// `model_id` (or `None` if absent). Video models carry their real pricing
    /// here under `pricing_skus` (e.g. `per-video-second`, `video_tokens`) plus
    /// supported resolutions/durations/sizes - none of which appears in the
    /// token-based `/models` pricing object (which is `0` for video).
    pub async fn video_model_detail(&self, model_id: &str) -> Result<Option<Value>> {
        let resp = self
            .http
            .get(format!("{}/videos/models", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("request to OpenRouter /videos/models failed")?
            .error_for_status()
            .context("OpenRouter /videos/models returned an error status")?;

        let body: Value = resp
            .json()
            .await
            .context("failed to decode OpenRouter /videos/models response")?;
        let found = body.get("data").and_then(Value::as_array).and_then(|arr| {
            arr.iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(model_id))
                .cloned()
        });
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{ModelsQuery, OpenRouterClient};

    #[tokio::test]
    async fn list_models_sends_query_params_and_parses_data() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(query_param("q", "openai"))
            .and(query_param("sort", "newest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {"id": "openai/gpt", "name": "GPT", "context_length": 128000}
                ]
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let query = ModelsQuery {
            q: Some("openai".to_string()),
            sort: Some("newest".to_string()),
            ..Default::default()
        };
        let models = client.list_models(&query).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "openai/gpt");
        assert_eq!(models[0].context_length, Some(128_000));
    }

    #[tokio::test]
    async fn list_models_errors_on_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "bad-key");
        let err = client
            .list_models(&ModelsQuery::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("error status"));
    }

    #[tokio::test]
    async fn describe_model_unwraps_data_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/anthropic/claude-opus-4.7/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "anthropic/claude-opus-4.7",
                    "endpoints": [{"provider_name": "Anthropic", "status": 0}]
                }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let detail = client
            .describe_model("anthropic/claude-opus-4.7")
            .await
            .unwrap();
        // The `data` envelope is unwrapped; everything underneath is preserved.
        assert_eq!(detail["id"], "anthropic/claude-opus-4.7");
        assert_eq!(detail["endpoints"][0]["provider_name"], "Anthropic");
    }

    #[tokio::test]
    async fn model_input_modalities_returns_matching_model_capabilities() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(query_param("q", "google/gemini-2.5-flash"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    // A near-match that must be ignored (id differs).
                    {"id": "google/gemini-2.5-flash-lite", "architecture": {"input_modalities": ["text"]}},
                    {"id": "google/gemini-2.5-flash", "architecture": {"input_modalities": ["text", "image"]}}
                ]
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let mods = client
            .model_input_modalities("google/gemini-2.5-flash")
            .await
            .unwrap();
        assert_eq!(mods, vec!["text".to_string(), "image".to_string()]);
    }

    #[tokio::test]
    async fn model_input_modalities_errors_when_id_absent() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "some/other-model", "architecture": {"input_modalities": ["text"]}}]
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let err = client
            .model_input_modalities("missing/model")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn describe_model_errors_on_unknown_id() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/foo/bar/endpoints"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let err = client.describe_model("foo/bar").await.unwrap_err();
        assert!(err.to_string().contains("error status"));
    }
}
