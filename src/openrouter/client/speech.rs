//! Synchronous audio endpoints: `POST /api/v1/audio/speech` (text-to-speech)
//! and `POST /api/v1/audio/transcriptions` (speech-to-text).

use anyhow::{Context, Result};
use serde_json::Value;

use crate::openrouter::{
    OpenRouterClient, SpeechBody, SpeechResult, TranscriptionBody, content_type, generation_id,
};

impl OpenRouterClient {
    /// `POST /api/v1/audio/speech` - synchronous text-to-speech. Returns the raw
    /// audio bytes (OpenAI-Speech-compatible), the content type, and the
    /// `X-Generation-Id` header when present. On a non-2xx status the upstream
    /// error body is surfaced verbatim.
    pub async fn speech(&self, req: &SpeechBody) -> Result<SpeechResult> {
        let rb = self
            .http
            .post(format!("{}/audio/speech", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req);
        let resp = self.send_checked(rb, "/audio/speech").await?;

        let generation_id = generation_id(&resp);
        let mime = content_type(&resp, "audio/mpeg");
        let bytes = resp
            .bytes()
            .await
            .context("failed to read speech audio bytes")
            .map_err(|error| {
                crate::billing::Receipt {
                    cost: None,
                    generation_id: generation_id.clone(),
                }
                .attach(error)
            })?;
        Ok(SpeechResult {
            mime,
            bytes,
            generation_id,
        })
    }

    /// `POST /api/v1/audio/transcriptions` - synchronous speech-to-text.
    /// Returns the raw response JSON: shape varies with `response_format`
    /// (`json` is just `{text, usage}`; `verbose_json` adds `language`,
    /// `duration`, `segments`, `words`, etc.), so the caller reads it untyped.
    pub async fn transcribe(&self, req: &TranscriptionBody) -> Result<Value> {
        let rb = self
            .http
            .post(format!("{}/audio/transcriptions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req);
        let response = self.send_checked(rb, "/audio/transcriptions").await?;
        let receipt = crate::billing::Receipt {
            cost: None,
            generation_id: generation_id(&response),
        };
        response
            .json()
            .await
            .context("failed to decode OpenRouter /audio/transcriptions response")
            .map_err(|error| receipt.attach(error))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::openrouter::{OpenRouterClient, SpeechBody};

    #[tokio::test]
    async fn malformed_transcription_success_retains_receipt_but_http_failure_does_not() {
        for status in [200, 401] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/audio/transcriptions"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("x-generation-id", "gen-paid-transcript")
                        .set_body_string("{"),
                )
                .mount(&server)
                .await;
            let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
            let error = client
                .transcribe(&crate::openrouter::TranscriptionBody {
                    model: "test/transcribe".into(),
                    input_audio: crate::openrouter::InputAudio {
                        data: "AAAA".into(),
                        format: "wav".into(),
                    },
                    language: None,
                    response_format: None,
                    timestamp_granularities: vec![],
                    temperature: None,
                })
                .await
                .expect_err("invalid response");
            let receipt = crate::billing::Receipt::from_error(&error);
            if status == 200 {
                let receipt = receipt.expect("unknown charge survives");
                assert_eq!(receipt.cost, None);
                assert_eq!(
                    receipt.generation_id.as_deref(),
                    Some("gen-paid-transcript")
                );
            } else {
                assert!(receipt.is_none());
            }
        }
    }

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
    async fn speech_body_failures_preserve_unknown_billing_and_generation_id() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // Reject an oversized declaration without allocating that body, then
        // exercise a truncated body without a generation header.
        for (length, generation_id) in [
            (
                crate::openrouter::MAX_MEDIA_BYTES + 1,
                Some("gen-paid-audio"),
            ),
            (10, None),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {length}\r\nContent-Type: audio/mpeg\r\n{}Connection: close\r\n\r\nMP3",
                generation_id
                    .map(|id| format!("X-Generation-Id: {id}\r\n"))
                    .unwrap_or_default()
            );
            let sender = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0; 4096];
                assert!(socket.read(&mut request).await.unwrap() > 0);
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            });
            let client = OpenRouterClient::with_base_url(url, "test-key");
            let error = client
                .speech(&SpeechBody {
                    model: "test/speech".into(),
                    input: "hello".into(),
                    voice: "alloy".into(),
                    response_format: None,
                    speed: None,
                })
                .await
                .err()
                .expect("body failure must propagate");
            sender.await.unwrap();
            let receipt = crate::billing::Receipt::from_error(&error)
                .expect("successful headers establish a possible charge");
            assert_eq!(receipt.cost, None);
            assert_eq!(receipt.generation_id.as_deref(), generation_id);
            assert!(
                error
                    .to_string()
                    .starts_with("failed to read speech audio bytes")
            );
            if length > crate::openrouter::MAX_MEDIA_BYTES {
                assert!(error.to_string().contains("byte limit"));
            }
            let mut totals = crate::billing::Totals::default();
            totals.add(receipt);
            assert_eq!(totals.unknown, 1);
        }
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
