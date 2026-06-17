//! `GET /api/v1/models` endpoint.

use anyhow::{Context, Result};

use crate::openrouter::{Model, ModelsQuery, ModelsResponse, OpenRouterClient};

impl OpenRouterClient {
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
}
