//! `GET /api/v1/generation?id=` - the billing/latency record of one request.

use anyhow::Result;
use serde_json::Value;

use crate::openrouter::{OpenRouterClient, unwrap_data};

impl OpenRouterClient {
    /// `GET /api/v1/generation?id=<generation_id>` - the stored record for one
    /// request: `total_cost`, `provider_name`, native token counts, `latency`,
    /// `generation_time`, `api_type`, and more. Returned as raw JSON (the
    /// `data` envelope unwrapped) so everything OpenRouter reports reaches the
    /// caller without a hand-maintained schema. An unknown or not-yet-indexed
    /// id is a 404 [`crate::openrouter::HttpFailure`].
    pub async fn get_generation(&self, generation_id: &str) -> Result<Value> {
        let rb = self
            .http
            .get(format!("{}/generation", self.base_url))
            .bearer_auth(&self.api_key)
            .query(&[("id", generation_id)]);
        Ok(unwrap_data(self.send_json(rb, "/generation").await?))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{HttpFailure, OpenRouterClient};

    /// The id travels as the `id` query parameter and the `data` envelope is
    /// unwrapped so the caller gets the record itself.
    #[tokio::test]
    async fn get_generation_sends_id_query_param_and_unwraps_data() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/generation"))
            .and(query_param("id", "gen-abc"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "gen-abc",
                    "total_cost": 0.0042,
                    "provider_name": "OpenAI",
                    "native_tokens_prompt": 12,
                    "native_tokens_completion": 30,
                    "latency": 812,
                    "generation_time": 1_530,
                    "api_type": "chat.completions"
                }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let record = client.get_generation("gen-abc").await.unwrap();
        assert_eq!(record["id"], "gen-abc");
        assert_eq!(record["total_cost"], 0.0042);
        assert_eq!(record["provider_name"], "OpenAI");
        assert_eq!(record["api_type"], "chat.completions");
    }

    /// An unknown (or not yet indexed) id is a 404 the caller can recognize
    /// by status, so the tool layer can turn it into an invalid-params error.
    #[tokio::test]
    async fn get_generation_surfaces_404_as_an_http_failure() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/generation"))
            .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let err = client.get_generation("gen-missing").await.unwrap_err();
        let failure = err.downcast_ref::<HttpFailure>().expect("typed failure");
        assert_eq!(failure.status, reqwest::StatusCode::NOT_FOUND);
        assert!(err.to_string().contains("/generation"), "got: {err}");
    }
}
