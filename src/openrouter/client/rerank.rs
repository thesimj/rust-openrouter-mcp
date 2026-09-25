//! `POST /api/v1/rerank`.

use anyhow::Result;
use reqwest::Method;

use crate::openrouter::{OpenRouterClient, RerankBody, RerankResponse};

impl OpenRouterClient {
    /// `POST /api/v1/rerank` - synchronous document reranking. Returns the
    /// decoded body plus the `X-Generation-Id` header. A 2xx whose body cannot
    /// be decoded keeps a billing receipt on the error; an HTTP failure
    /// surfaces the upstream error body (bounded to `MAX_ERROR_BODY_CHARS`).
    pub async fn rerank(&self, req: &RerankBody) -> Result<(RerankResponse, Option<String>)> {
        let rb = self.request(Method::POST, "/rerank").json(req);
        self.send_json_receipted(rb, "/rerank").await
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{OpenRouterClient, ProviderRouting, RerankBody};

    fn request() -> RerankBody {
        RerankBody {
            model: "cohere/rerank-v3.5".into(),
            query: "rust".into(),
            documents: vec!["go".into(), "rust lang".into()],
            top_n: Some(1),
            provider: Some(ProviderRouting {
                order: vec!["cohere".into()],
                ..Default::default()
            }),
        }
    }

    /// The documented body (query, documents array, top_n, provider.order)
    /// reaches the wire; results and usage come back typed with the
    /// `X-Generation-Id` header.
    #[tokio::test]
    async fn rerank_sends_the_body_and_returns_results_with_generation_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .and(body_partial_json(json!({
                "model": "cohere/rerank-v3.5",
                "query": "rust",
                "documents": ["go", "rust lang"],
                "top_n": 1,
                "provider": {"order": ["cohere"]}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-rr-1")
                    .set_body_json(json!({
                        "model": "cohere/rerank-v3.5",
                        "results": [{"index": 1, "relevance_score": 0.98, "document": {"text": "rust lang"}}],
                        "usage": {"cost": 0.002, "search_units": 1}
                    })),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let (body, generation_id) = client.rerank(&request()).await.unwrap();
        assert_eq!(generation_id.as_deref(), Some("gen-rr-1"));
        assert_eq!(body.results[0].index, 1);
        assert_eq!(body.results[0].relevance_score, 0.98);
        assert_eq!(body.usage.unwrap().cost, Some(0.002));
    }
}
