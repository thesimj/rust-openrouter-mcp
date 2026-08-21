//! `GET /api/v1/models` and `GET /api/v1/models/{id}/endpoints`.

use anyhow::{Context, Result};
use serde_json::Value;

use crate::openrouter::{
    Model, ModelsQuery, ModelsResponse, OpenRouterClient, truncate_error_body,
};

impl OpenRouterClient {
    /// The `input_modalities` declared for a single model id (e.g.
    /// `["text", "image"]`). Searches `/models?q=<id>` and returns the
    /// architecture of the entry whose id matches `model_id` exactly. Errors if
    /// no such model is found. Used to gate multimodal inputs before sending.
    ///
    /// `output_modalities=all` is required, not cosmetic: the endpoint defaults
    /// that filter to `text`, which hides every image/audio/video/embeddings
    /// model (135 of 546 as of 2026-08). Without it this lookup reports "not
    /// found" for exactly the non-text models a modality gate exists to check.
    pub async fn model_input_modalities(&self, model_id: &str) -> Result<Vec<String>> {
        let query = ModelsQuery {
            q: Some(model_id.to_string()),
            output_modalities: Some("all".to_string()),
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
        let rb = self
            .http
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .query(query);
        let parsed: ModelsResponse = self.send_json(rb, "/models").await?;
        Ok(parsed.data)
    }

    /// `GET /api/v1/models/{model_id}/endpoints` - the full record for one model:
    /// the model object (id, description, architecture, context) plus the
    /// per-provider endpoints (pricing, uptime, status, quantization, supported
    /// parameters). Returned as raw JSON so the caller surfaces everything
    /// OpenRouter reports without a hand-maintained schema. `model_id` is the
    /// `author/slug` id (e.g. "anthropic/claude-opus-4.7").
    pub async fn describe_model(&self, model_id: &str) -> Result<Value> {
        let rb = self
            .http
            .get(format!("{}/models/{}/endpoints", self.base_url, model_id))
            .bearer_auth(&self.api_key);
        let mut body: Value = self
            .send_json(rb, &format!("/models/{model_id}/endpoints"))
            .await?;
        // Unwrap the `data` envelope (model + endpoints) when present.
        Ok(body.get_mut("data").map(Value::take).unwrap_or(body))
    }

    /// `GET /api/v1/videos/models`, returning the entry whose `id` matches
    /// `model_id` (or `None` if absent). Video models carry their real pricing
    /// here under `pricing_skus` (e.g. `per-video-second`, `video_tokens`) plus
    /// supported resolutions/durations/sizes - none of which appears in the
    /// token-based `/models` pricing object (which is `0` for video).
    pub async fn video_model_detail(&self, model_id: &str) -> Result<Option<Value>> {
        let rb = self
            .http
            .get(format!("{}/videos/models", self.base_url))
            .bearer_auth(&self.api_key);
        let body: Value = self.send_json(rb, "/videos/models").await?;
        let found = body.get("data").and_then(Value::as_array).and_then(|arr| {
            arr.iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(model_id))
                .cloned()
        });
        Ok(found)
    }

    /// `GET /api/v1/images/models/{author}/{slug}/endpoints` - per-endpoint
    /// image capabilities: definitive `supported_parameters`,
    /// `allowed_passthrough_parameters`, pricing, and `supports_streaming`.
    /// `model_id` is the `author/slug` id. A 404 (model has no image endpoint)
    /// returns `None` rather than failing - best-effort, matching
    /// [`video_model_detail`](Self::video_model_detail)'s posture of never
    /// failing the whole `describe_model` call over this enrichment.
    pub async fn image_model_detail(&self, model_id: &str) -> Result<Option<Value>> {
        let label = format!("/images/models/{model_id}/endpoints");
        let resp = self
            .http
            .get(format!("{}{label}", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .with_context(|| format!("request to OpenRouter {label} failed"))?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let status = resp.status();
        if !status.is_success() {
            let body = truncate_error_body(resp.text().await.unwrap_or_default());
            anyhow::bail!("OpenRouter {label} returned {status}: {body}");
        }
        let mut body: Value = resp
            .json()
            .await
            .with_context(|| format!("failed to decode OpenRouter {label} response"))?;
        Ok(Some(body.get_mut("data").map(Value::take).unwrap_or(body)))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param, query_param_is_missing};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{ModelsQuery, OpenRouterClient};

    #[tokio::test]
    async fn list_models_sends_query_params_and_parses_data() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            // Every set field reaches the wire under its API name, with the
            // integer rendered as a plain decimal...
            .and(query_param("q", "openai"))
            .and(query_param("sort", "newest"))
            .and(query_param("output_modalities", "image,text"))
            .and(query_param("supported_parameters", "tools"))
            .and(query_param("context", "128000"))
            // ...and `None` fields are omitted rather than sent empty.
            .and(query_param_is_missing("input_modalities"))
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
            output_modalities: Some("image,text".to_string()),
            supported_parameters: Some("tools".to_string()),
            context: Some(128_000),
            input_modalities: None,
        };
        let models = client.list_models(&query).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "openai/gpt");
        assert_eq!(models[0].context_length, Some(128_000));
    }

    /// Pins that we ask for compression. OpenRouter serves gzip and the payloads
    /// are large, highly compressible JSON - `/models` measures 672KB raw against
    /// 71KB gzipped. The header only appears because `reqwest` is built with the
    /// `gzip` feature, so this fails if that feature is ever trimmed from
    /// `Cargo.toml` alongside the other `default-features = false` opt-ins.
    /// Decompression itself is reqwest's job, not ours, so it isn't retested here.
    #[tokio::test]
    async fn requests_advertise_gzip_support() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [] })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        client.list_models(&ModelsQuery::default()).await.unwrap();

        let requests = server.received_requests().await.unwrap();
        let encodings = requests[0].headers.get("accept-encoding");
        let sent = encodings
            .expect("accept-encoding is sent")
            .to_str()
            .unwrap();
        assert!(sent.contains("gzip"), "got: {sent}");
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
        assert!(err.to_string().contains("401"), "got: {err}");
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
            // Matching on this makes the test fail (mock misses -> 404) if the
            // filter is ever dropped again. The endpoint defaults it to `text`,
            // which hides every non-text-output model from this lookup.
            .and(query_param("output_modalities", "all"))
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
    async fn image_model_detail_unwraps_data_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/models/openai/gpt-image-2/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "openai/gpt-image-2",
                    "endpoints": [{"supported_parameters": ["quality", "output_format"]}]
                }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let detail = client
            .image_model_detail("openai/gpt-image-2")
            .await
            .unwrap()
            .expect("image endpoint detail present");
        assert_eq!(detail["id"], "openai/gpt-image-2");
        assert_eq!(detail["endpoints"][0]["supported_parameters"][0], "quality");
    }

    /// A model with no image endpoint (404) is a miss, not a failure - it must
    /// not fail the whole `describe_model` call.
    #[tokio::test]
    async fn image_model_detail_returns_none_on_404() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/models/anthropic/claude-opus-4.7/endpoints"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let detail = client
            .image_model_detail("anthropic/claude-opus-4.7")
            .await
            .unwrap();
        assert!(detail.is_none());
    }

    #[tokio::test]
    async fn image_model_detail_surfaces_non_404_errors() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/images/models/openai/gpt-image-2/endpoints"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let err = client
            .image_model_detail("openai/gpt-image-2")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("500"), "got: {err}");
    }

    /// F3: an oversized error body must be capped the same way
    /// `send_checked` caps it, not echoed verbatim into the caller's context.
    #[tokio::test]
    async fn image_model_detail_truncates_an_oversized_error_body() {
        let server = MockServer::start().await;
        let huge = "x".repeat(5000);
        Mock::given(method("GET"))
            .and(path("/images/models/openai/gpt-image-2/endpoints"))
            .respond_with(ResponseTemplate::new(500).set_body_string(huge))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let err = client
            .image_model_detail("openai/gpt-image-2")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("[truncated]"), "got: {err}");
        assert!(err.to_string().len() < 1000, "got: {err}");
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
        assert!(err.to_string().contains("404"), "got: {err}");
    }
}
