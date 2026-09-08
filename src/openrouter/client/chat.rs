//! `POST /api/v1/chat/completions` endpoint (image generation / text / vision).

use anyhow::{Context, Result};

use crate::openrouter::{ChatCompletion, ChatRequest, OpenRouterClient, generation_id};

impl OpenRouterClient {
    /// `POST /api/v1/chat/completions` - used for text and vision (describe)
    /// calls. On a non-2xx status the upstream error body is surfaced verbatim
    /// (OpenRouter wraps provider errors there).
    pub async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatCompletion> {
        let rb = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req);
        let response = self.send_checked(rb, "/chat/completions").await?;
        let receipt = crate::billing::Receipt {
            cost: None,
            generation_id: generation_id(&response),
        };
        response
            .json()
            .await
            .context("failed to decode OpenRouter /chat/completions response")
            .map_err(|error| receipt.attach(error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn malformed_chat_success_retains_receipt_but_http_failure_does_not() {
        for status in [200, 401] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("x-generation-id", "gen-paid-chat")
                        .set_body_string("{"),
                )
                .mount(&server)
                .await;
            let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
            let error = client
                .chat_completion(&ChatRequest {
                    model: "test/chat".into(),
                    messages: vec![],
                    modalities: None,
                    image_config: None,
                    seed: None,
                    temperature: None,
                    max_tokens: None,
                    reasoning: None,
                    stream: false,
                })
                .await
                .expect_err("invalid response");
            let receipt = crate::billing::Receipt::from_error(&error);
            if status == 200 {
                let receipt = receipt.expect("unknown charge survives");
                assert_eq!(receipt.cost, None);
                assert_eq!(receipt.generation_id.as_deref(), Some("gen-paid-chat"));
            } else {
                assert!(receipt.is_none());
            }
        }
    }
}
