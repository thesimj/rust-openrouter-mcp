//! `POST /api/v1/audio/speech` endpoint (synchronous text-to-speech).

use anyhow::{Context, Result};

use crate::openrouter::{OpenRouterClient, SpeechBody, SpeechResult, content_type, generation_id};

impl OpenRouterClient {
    /// `POST /api/v1/audio/speech` - synchronous text-to-speech. Returns the raw
    /// audio bytes (OpenAI-Speech-compatible), the content type, and the
    /// `X-Generation-Id` header when present. On a non-2xx status the upstream
    /// error body is surfaced verbatim.
    pub async fn speech(&self, req: &SpeechBody) -> Result<SpeechResult> {
        let resp = self
            .http
            .post(format!("{}/audio/speech", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req)
            .send()
            .await
            .context("request to OpenRouter /audio/speech failed")?;

        let generation_id = generation_id(&resp);

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter /audio/speech returned {status}: {body}");
        }
        let mime = content_type(&resp, "audio/mpeg");
        let bytes = resp
            .bytes()
            .await
            .context("failed to read speech audio bytes")?
            .to_vec();
        Ok(SpeechResult {
            mime,
            bytes,
            generation_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{OpenRouterClient, SpeechBody};

    #[tokio::test]
    async fn speech_returns_bytes_mime_and_generation_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(body_partial_json(json!({
                "model": "openai/gpt-4o-mini-tts",
                "input": "hi",
                "voice": "alloy"
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .insert_header("x-generation-id", "gen-aud-3")
                    .set_body_bytes(b"MP3".to_vec()),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let body = SpeechBody {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: "hi".to_string(),
            voice: "alloy".to_string(),
            response_format: Some("mp3".to_string()),
            speed: None,
        };
        let result = match client.speech(&body).await {
            Ok(r) => r,
            Err(e) => panic!("speech should succeed: {e}"),
        };
        assert_eq!(result.mime, "audio/mpeg");
        assert_eq!(result.bytes, b"MP3");
        assert_eq!(result.generation_id.as_deref(), Some("gen-aud-3"));
    }

    #[tokio::test]
    async fn speech_surfaces_error_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(ResponseTemplate::new(422).set_body_string("{\"error\":\"bad voice\"}"))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let body = SpeechBody {
            model: "m".to_string(),
            input: "x".to_string(),
            voice: "z".to_string(),
            response_format: None,
            speed: None,
        };
        let err = match client.speech(&body).await {
            Err(e) => e,
            Ok(_) => panic!("provider error should propagate"),
        };
        assert!(err.to_string().contains("bad voice"));
    }
}
