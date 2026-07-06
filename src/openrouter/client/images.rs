//! `POST /api/v1/images` - OpenRouter's dedicated image-generation endpoint.
//!
//! This is separate from `/chat/completions`: image-only models (e.g. the OpenAI
//! GPT Image family) are reachable *only* here, and every other image model is
//! served here too, so all generation goes through this endpoint.

use anyhow::{Context, Result};

use crate::openrouter::{ImagesRequest, ImagesResponse, OpenRouterClient, generation_id};

impl OpenRouterClient {
    /// `POST /api/v1/images` - generate image(s) from a prompt (and optional
    /// reference images). Returns the parsed response plus the `X-Generation-Id`
    /// response header when present. On a non-2xx status the upstream error body
    /// is surfaced verbatim (OpenRouter wraps provider errors there).
    pub async fn generate_images(
        &self,
        req: &ImagesRequest,
    ) -> Result<(ImagesResponse, Option<String>)> {
        let resp = self
            .http
            .post(format!("{}/images", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req)
            .send()
            .await
            .context("request to OpenRouter /images failed")?;

        let generation_id = generation_id(&resp);

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter /images returned {status}: {body}");
        }

        let parsed: ImagesResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /images response")?;
        Ok((parsed, generation_id))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{ImagesRequest, OpenRouterClient};

    fn request() -> ImagesRequest {
        ImagesRequest {
            model: "openai/gpt-image-2".to_string(),
            prompt: "an owl".to_string(),
            resolution: Some("1K".to_string()),
            aspect_ratio: Some("1:1".to_string()),
            seed: None,
            n: None,
            input_references: vec![],
        }
    }

    #[tokio::test]
    async fn generate_images_posts_body_and_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images"))
            .and(body_partial_json(json!({
                "model": "openai/gpt-image-2",
                "prompt": "an owl",
                "resolution": "1K",
                "aspect_ratio": "1:1"
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-9")
                    .set_body_json(json!({
                        "created": 1748372400,
                        "data": [{ "b64_json": "AAAA" }],
                        "usage": { "cost": 0.03 }
                    })),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let (resp, gen_id) = client.generate_images(&request()).await.unwrap();
        assert_eq!(resp.data.len(), 1);
        assert_eq!(resp.data[0].b64_json, "AAAA");
        assert_eq!(resp.usage.and_then(|u| u.cost), Some(0.03));
        assert_eq!(gen_id.as_deref(), Some("gen-9"));
    }

    #[tokio::test]
    async fn generate_images_surfaces_error_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images"))
            .respond_with(ResponseTemplate::new(500).set_body_string("{\"error\":\"boom\"}"))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let err = client.generate_images(&request()).await.unwrap_err();
        assert!(err.to_string().contains("boom"));
        assert!(err.to_string().contains("500"));
    }
}
