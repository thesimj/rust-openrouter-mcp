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
        let rb = self
            .http
            .post(format!("{}/images", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req);
        let resp = self.send_checked(rb, "/images").await?;

        let generation_id = generation_id(&resp);

        let parsed: ImagesResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /images response")
            .map_err(|error| {
                crate::billing::Receipt {
                    cost: None,
                    generation_id: generation_id.clone(),
                }
                .attach(error)
            })?;
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
            quality: None,
            output_format: None,
            background: None,
            output_compression: None,
        }
    }

    #[tokio::test]
    async fn successful_image_headers_retain_receipt_on_oversize_or_invalid_json() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for length in [crate::openrouter::MAX_JSON_BYTES + 1, 1] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let sender = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0; 4096];
                assert!(socket.read(&mut request).await.unwrap() > 0);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {length}\r\nX-Generation-Id: gen-paid-image\r\nConnection: close\r\n\r\n{{"
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            });
            let client = OpenRouterClient::with_base_url(url, "test-key");
            let error = client
                .generate_images(&request())
                .await
                .expect_err("invalid body");
            sender.await.unwrap();
            let receipt =
                crate::billing::Receipt::from_error(&error).expect("unknown charge survives");
            assert_eq!(receipt.cost, None);
            assert_eq!(receipt.generation_id.as_deref(), Some("gen-paid-image"));
            assert!(
                error
                    .to_string()
                    .starts_with("failed to decode OpenRouter /images response")
            );
            if length > crate::openrouter::MAX_JSON_BYTES {
                assert!(error.to_string().contains("byte limit"));
            }
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
