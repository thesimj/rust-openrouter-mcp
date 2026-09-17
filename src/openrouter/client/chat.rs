//! `POST /api/v1/chat/completions` endpoint: text / vision (one JSON result)
//! and audio output (music), which OpenRouter delivers only as an SSE stream.

use anyhow::{Context, Result};
use serde_json::Value;

use crate::image_io::Base64Assembler;
use crate::openrouter::{
    ChatAudioResult, ChatChunk, ChatCompletion, ChatRequest, MAX_MEDIA_BYTES, OpenRouterClient,
    generation_id, truncate_error_body,
};

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

    /// `POST /api/v1/chat/completions` with `stream: true` for an audio-output
    /// model (music: `google/lyria-3-*`). Audio output is stream-only upstream;
    /// this aggregates the stream as it arrives into one [`ChatAudioResult`],
    /// decoding each audio fragment on the spot. `req` must carry
    /// `modalities: ["text", "audio"]` and `stream: true` - the caller
    /// (`music_gen`) builds it that way. An `error` event mid-stream, or an
    /// undecodable chunk, fails at once with the billing receipt attached (the
    /// provider may already have charged); a body that ends before `[DONE]`
    /// is returned with `complete: false` rather than discarded.
    pub async fn chat_completion_audio(&self, req: &ChatRequest) -> Result<ChatAudioResult> {
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

        let mut result = ChatAudioResult {
            generation_id: receipt.generation_id.clone(),
            ..Default::default()
        };
        let mut audio = Base64Assembler::default();
        let streamed = response
            .sse_each(MAX_MEDIA_BYTES, |payload| {
                let chunk: ChatChunk = serde_json::from_str(payload)
                    .context("failed to decode an OpenRouter /chat/completions stream chunk")?;
                if let Some(error) = chunk.error {
                    let message = error
                        .get("message")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| error.to_string());
                    anyhow::bail!(
                        "OpenRouter /chat/completions stream reported an error: {}",
                        truncate_error_body(message)
                    );
                }
                if result.generation_id.is_none() {
                    result.generation_id = chunk.id;
                }
                for delta in chunk.choices.into_iter().flatten().filter_map(|c| c.delta) {
                    if let Some(content) = delta.content {
                        result.text.push_str(&content);
                    }
                    if let Some(fragment) = delta.audio {
                        if let Some(data) = fragment.data {
                            audio
                                .push(&data)
                                .context("the streamed audio is not valid base64")?;
                        }
                        if let Some(transcript) = fragment.transcript {
                            result.transcript.push_str(&transcript);
                        }
                    }
                }
                if let Some(cost) = chunk.usage.and_then(|u| u.cost) {
                    result.cost = Some(cost);
                }
                Ok(())
            })
            .await
            .context("failed to read OpenRouter /chat/completions stream");
        // From here on the provider has answered: every failure keeps the receipt.
        result.complete = streamed.map_err(|error| receipt.clone().attach(error))?;
        result.audio = receipt.wrap(|| {
            audio
                .finish()
                .context("the streamed audio is not valid base64")
        })?;
        Ok(result)
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
                    ..Default::default()
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

    fn audio_request(model: &str) -> ChatRequest {
        ChatRequest {
            model: model.to_string(),
            messages: vec![crate::openrouter::Message {
                role: "user".to_string(),
                content: crate::openrouter::Content::Text("lo-fi loop".to_string()),
            }],
            modalities: Some(vec!["text".to_string(), "audio".to_string()]),
            stream: true,
            ..Default::default()
        }
    }

    /// The live Lyria stream shape (2026-09-13): processing comments, a
    /// `content` chunk, audio fragments, a stop chunk, a `usage` chunk, `[DONE]`.
    #[tokio::test]
    async fn chat_completion_audio_aggregates_audio_text_cost_and_generation_id() {
        let server = MockServer::start().await;
        let body = concat!(
            ": OPENROUTER PROCESSING\n\n",
            "data: {\"id\":\"gen-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"<instru\",\"role\":\"assistant\"}}]}\n\n",
            ": OPENROUTER PROCESSING\n\n",
            "data: {\"id\":\"gen-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"mental>\",\"audio\":{\"data\":\"SUQz\"}}}]}\n\n",
            "data: {\"id\":\"gen-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"audio\":{\"data\":\"AwAA\",\"transcript\":\"la la\"}}}]}\n\n",
            "data: {\"id\":\"gen-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"id\":\"gen-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":4,\"cost\":0.04}}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "model": "google/lyria-3-clip-preview",
                "modalities": ["text", "audio"],
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("x-generation-id", "gen-1")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let result = client
            .chat_completion_audio(&audio_request("google/lyria-3-clip-preview"))
            .await
            .unwrap();
        assert_eq!(result.audio, b"ID3\x03\x00\x00");
        assert!(result.complete);
        assert_eq!(result.text, "<instrumental>");
        assert_eq!(result.transcript, "la la");
        assert_eq!(result.generation_id.as_deref(), Some("gen-1"));
        assert_eq!(result.cost, Some(0.04));
        // The request never sent an `audio` block it was not asked for.
        let sent: serde_json::Value = server.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap();
        assert!(sent.get("audio").is_none(), "sent: {sent}");
    }

    #[tokio::test]
    async fn chat_completion_audio_falls_back_to_the_chunk_id_without_the_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "data: {\"id\":\"gen-from-body\",\"choices\":[{\"delta\":{\"audio\":{\"data\":\"QUJD\"}}}]}\n\ndata: [DONE]\n\n",
            ))
            .mount(&server)
            .await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let result = client
            .chat_completion_audio(&audio_request("m"))
            .await
            .unwrap();
        assert_eq!(result.generation_id.as_deref(), Some("gen-from-body"));
        assert_eq!(result.audio, b"ABC");
        assert_eq!(result.cost, None);
    }

    /// A stream that opened with 200 may still fail: an `error` event, an
    /// undecodable chunk, or audio that is not base64. All keep the receipt;
    /// an HTTP failure has none. An oversized error message is cut like an
    /// HTTP error body is.
    #[tokio::test]
    async fn chat_completion_audio_stream_failures_keep_the_receipt_but_http_failures_do_not() {
        let long_error = format!(
            "data: {{\"error\":{{\"message\":\"{}\"}}}}\n\n",
            "x".repeat(5000)
        );
        for (status, body, expect) in [
            (
                200,
                "data: {\"error\":{\"message\":\"quota exceeded\",\"code\":402}}\n\n",
                Some("quota exceeded"),
            ),
            (200, long_error.as_str(), Some("... [truncated]")),
            (200, "data: {\"id\":\n\n", Some("decode")),
            (
                200,
                "data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\"!!!!\"}}}]}\n\n",
                Some("not valid base64"),
            ),
            (
                402,
                "{\"error\":{\"message\":\"insufficient credits\"}}",
                None,
            ),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/chat/completions"))
                .respond_with(
                    ResponseTemplate::new(status)
                        .insert_header("x-generation-id", "gen-billed")
                        .set_body_string(body),
                )
                .mount(&server)
                .await;
            let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
            let error = client
                .chat_completion_audio(&audio_request("m"))
                .await
                .expect_err("must fail");
            let receipt = crate::billing::Receipt::from_error(&error);
            match expect {
                Some(fragment) => {
                    assert!(error.to_string().contains(fragment), "got: {error:#}");
                    assert!(error.to_string().len() < 2000, "not bounded: {error:#}");
                    let receipt = receipt.expect("stream failure keeps the receipt");
                    assert_eq!(receipt.generation_id.as_deref(), Some("gen-billed"));
                    assert_eq!(receipt.cost, None);
                }
                None => {
                    assert!(error.to_string().contains("insufficient credits"));
                    assert!(receipt.is_none());
                }
            }
        }
    }

    /// Fragments encoded one at a time each carry padding; the audio must be
    /// decoded per fragment, not from the concatenation (which is invalid).
    #[tokio::test]
    async fn chat_completion_audio_decodes_independently_padded_fragments() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
                "data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\"SUQzAwAAAAAvMg==\"}}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\"/w==\"}}}]}\n\n",
                "data: [DONE]\n\n",
            )))
            .mount(&server)
            .await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let result = client
            .chat_completion_audio(&audio_request("m"))
            .await
            .unwrap();
        assert_eq!(result.audio, b"ID3\x03\x00\x00\x00\x00/2\xff");
    }

    /// Shapes a provider may legitimately send that carry no audio must not
    /// abort a stream that was already billed: explicit nulls, an `error: null`,
    /// an empty `data:` line, and a `[DONE]` with trailing whitespace (anything
    /// after the sentinel is ignored).
    #[tokio::test]
    async fn chat_completion_audio_tolerates_null_fields_and_sloppy_sentinels() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(concat!(
                "data: {\"id\":\"gen-1\",\"error\":null,\"choices\":[{\"delta\":{\"audio\":{\"data\":\"QUJD\"}}}]}\n\n",
                "data:\n\n",
                "data: {\"choices\":[{\"delta\":null,\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":null,\"usage\":{\"cost\":0.04}}\n\n",
                "data: [DONE] \n\n",
                "data: this is not json\n\n",
            )))
            .mount(&server)
            .await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let result = client
            .chat_completion_audio(&audio_request("m"))
            .await
            .unwrap();
        assert_eq!(result.audio, b"ABC");
        assert_eq!(result.cost, Some(0.04));
        assert!(result.complete);
    }

    /// A body that closes cleanly before `[DONE]` is not a failure (the audio
    /// so far is kept, and billed), but it is reported as incomplete.
    #[tokio::test]
    async fn chat_completion_audio_reports_a_stream_that_ends_before_done() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\"QUJD\"}}}]}\n\n",
            ))
            .mount(&server)
            .await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let result = client
            .chat_completion_audio(&audio_request("m"))
            .await
            .unwrap();
        assert_eq!(result.audio, b"ABC");
        assert!(!result.complete);
    }
}
