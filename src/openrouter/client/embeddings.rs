//! `POST /api/v1/embeddings`.

use anyhow::Result;

use crate::openrouter::{EmbeddingsBody, EmbeddingsResponse, OpenRouterClient};

impl OpenRouterClient {
    /// `POST /api/v1/embeddings` - synchronous text embeddings. Returns the
    /// decoded body plus the `X-Generation-Id` header. A 2xx whose body cannot
    /// be decoded keeps a billing receipt on the error (the provider may have
    /// charged); an HTTP failure surfaces the upstream error body (bounded to 500 chars).
    pub async fn embeddings(
        &self,
        req: &EmbeddingsBody,
    ) -> Result<(EmbeddingsResponse, Option<String>)> {
        let rb = self
            .http
            .post(format!("{}/embeddings", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req);
        self.send_json_receipted(rb, "/embeddings").await
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{EmbeddingsBody, EmbeddingsInput, OpenRouterClient, ProviderRouting};

    fn request(texts: &[&str]) -> EmbeddingsBody {
        EmbeddingsBody {
            model: "openai/text-embedding-3-small".into(),
            input: EmbeddingsInput::from_texts(texts.iter().map(|s| s.to_string()).collect()),
            dimensions: Some(2),
            input_type: None,
            provider: Some(ProviderRouting {
                order: vec!["openai".into()],
                ..Default::default()
            }),
        }
    }

    /// The documented body reaches the wire (single text as a string, plus
    /// `provider.order`), the vectors come back typed, and the
    /// `X-Generation-Id` header is kept for the cost lookup.
    #[tokio::test]
    async fn embeddings_sends_the_body_and_returns_vectors_with_generation_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(json!({
                "model": "openai/text-embedding-3-small",
                "input": "hello",
                "dimensions": 2,
                "provider": {"order": ["openai"]}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-emb-1")
                    .set_body_json(json!({
                        "model": "openai/text-embedding-3-small",
                        "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1, "cost": 0.00001}
                    })),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let (body, generation_id) = client.embeddings(&request(&["hello"])).await.unwrap();
        assert_eq!(generation_id.as_deref(), Some("gen-emb-1"));
        assert_eq!(body.data[0].embedding, vec![0.1, 0.2]);
        assert_eq!(body.usage.unwrap().cost, Some(0.00001));
    }
}
