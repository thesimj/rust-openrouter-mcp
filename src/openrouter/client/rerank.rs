//! `POST /api/v1/rerank`.

use anyhow::Result;

use crate::openrouter::{OpenRouterClient, RerankBody, RerankReply};

impl OpenRouterClient {
    /// `POST /api/v1/rerank` - synchronous document reranking. Returns the
    /// decoded body plus the `X-Generation-Id` header. A 2xx whose body cannot
    /// be decoded keeps a billing receipt on the error; an HTTP failure
    /// surfaces the upstream error body verbatim.
    pub async fn rerank(&self, req: &RerankBody) -> Result<RerankReply> {
        let rb = self
            .http
            .post(format!("{}/rerank", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req);
        let (body, generation_id) = self.send_json_receipted(rb, "/rerank").await?;
        Ok(RerankReply {
            body,
            generation_id,
        })
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
        let result = client.rerank(&request()).await.unwrap();
        assert_eq!(result.generation_id.as_deref(), Some("gen-rr-1"));
        assert_eq!(result.body.results[0].index, 1);
        assert_eq!(result.body.results[0].relevance_score, 0.98);
        assert_eq!(result.body.usage.unwrap().cost, Some(0.002));
    }

    #[tokio::test]
    async fn malformed_success_retains_receipt_but_http_failure_does_not() {
        for status in [200, 402] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/rerank"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("x-generation-id", "gen-paid-rr")
                        .set_body_string("{"),
                )
                .mount(&server)
                .await;
            let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
            let error = client
                .rerank(&request())
                .await
                .expect_err("invalid response");
            let receipt = crate::billing::Receipt::from_error(&error);
            if status == 200 {
                let receipt = receipt.expect("unknown charge survives");
                assert_eq!(receipt.cost, None);
                assert_eq!(receipt.generation_id.as_deref(), Some("gen-paid-rr"));
            } else {
                assert!(receipt.is_none());
                assert!(error.to_string().contains("402"), "got: {error}");
            }
        }
    }
}
