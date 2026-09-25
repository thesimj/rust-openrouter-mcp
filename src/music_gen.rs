//! Music generation over OpenRouter's audio-output chat completions.
//!
//! There is no dedicated music endpoint (verified 2026-09-13: `/audio/models`,
//! `/audio/generations`, `/music/*` all 404 and the OpenAPI spec names none).
//! Music models such as `google/lyria-3-clip-preview` are chat models whose
//! `output_modalities` include `audio`; the audio comes back only as an SSE
//! stream of `delta.audio.data` base64 fragments on `POST /chat/completions`
//! with `modalities: ["text", "audio"]`. Like text-to-speech this is one call
//! with no job id, so it mirrors `audio_gen` (no task registry): save the
//! bytes, write a sidecar manifest, return the paths.
//!
//! The container is not declared anywhere in the stream (Lyria returned MP3 in
//! the live probe), so the saved extension comes from the bytes themselves.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::manifest::{self, AudioOutputMeta, MusicManifest};
use crate::openrouter::{
    AudioConfig, ChatRequest, Content, Message, OpenRouterClient, ProviderRouting,
};

/// Inputs for one music generation (domain struct; the wire body is
/// [`crate::openrouter::ChatRequest`] with `modalities: ["text", "audio"]`).
#[derive(Debug, Clone)]
pub struct MusicGenRequest {
    pub model: String,
    /// The musical description (genre, mood, tempo, instruments, lyrics...).
    pub prompt: String,
    /// Requested container (`audio.format`: wav, mp3, flac, opus, pcm16). Sent
    /// only when set; whether a model honors it is model-specific.
    pub format: Option<String>,
    pub seed: Option<u64>,
    /// Provider routing, already validated. Music is a chat call, so it takes
    /// the routing block and no per-provider `options` passthrough.
    pub provider: Option<ProviderRouting>,
}

/// The saved track plus what the model said about it.
#[derive(Debug)]
pub struct MusicSummary {
    pub path: PathBuf,
    pub mime: String,
    /// The `content` the model streamed alongside the audio - Lyria emits the
    /// lyrics, or `<instrumental>`.
    pub text: Option<String>,
    /// The `audio.transcript` fragments, when the model sends them.
    pub transcript: Option<String>,
}

/// Result of a music job: the saved file, the manifest, the reported cost,
/// and non-fatal warnings (e.g. a manifest write that failed after the audio
/// was saved).
#[derive(Debug)]
pub struct MusicJobResult {
    pub model: String,
    pub manifest_path: PathBuf,
    pub music: MusicSummary,
    /// `usage.cost` from the final stream chunk (Lyria reports its flat
    /// per-clip price here even though `/models` lists it as 0).
    pub cost: Option<f64>,
    pub generation_id: Option<String>,
    pub warnings: Vec<String>,
}

/// Trim + lowercase a requested format; blank counts as absent, so a literal
/// `"  "` never reaches the wire; `audio_gen` builds its speech format on it too.
/// Shared with the
/// MCP tool so its filename token matches what is sent.
pub(crate) fn normalize_format(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
}

fn non_blank(s: String) -> Option<String> {
    (!s.trim().is_empty()).then_some(s)
}

/// Run a music job: stream the completion, decode the audio, save it with the
/// extension its bytes call for, and write the sidecar manifest.
pub async fn run_job(
    client: &OpenRouterClient,
    req: &MusicGenRequest,
    output: &Path,
) -> Result<MusicJobResult> {
    let format = normalize_format(req.format.as_deref());
    let body = ChatRequest {
        model: req.model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: Content::Text(req.prompt.clone()),
        }],
        modalities: Some(vec!["text".to_string(), "audio".to_string()]),
        seed: req.seed,
        provider: req.provider.clone(),
        audio: format.clone().map(|format| AudioConfig { format }),
        ..Default::default()
    };

    let result = client.chat_completion_audio(&body).await?;
    // From here on the provider has answered: every failure keeps the receipt.
    let receipt = crate::billing::Receipt {
        cost: result.cost,
        generation_id: result.generation_id.clone(),
    };
    if result.audio.is_empty() {
        let said = non_blank(result.text)
            .map(|t| {
                format!(
                    " (the model answered with text only: {:?})",
                    crate::openrouter::truncate_error_body(t)
                )
            })
            .unwrap_or_default();
        return Err(receipt.attach(anyhow::anyhow!(
            "model returned no audio{said}; use a model whose output_modalities include \
             \"audio\" (list_models with output_modalities=\"audio\")"
        )));
    }
    let (mime, ext) = crate::audio_container::container_for(&result.audio, None, format.as_deref());
    let path = output.with_extension(ext);
    crate::output::write_bytes(&path, &result.audio)
        .await
        .map_err(|e| receipt.attach(anyhow::anyhow!("could not write {}: {e}", path.display())))?;

    let mut warnings = Vec::new();
    if !result.complete {
        warnings.push(
            "the stream ended before its [DONE] sentinel; the saved track may be cut short"
                .to_string(),
        );
    }
    let text = non_blank(result.text);
    let transcript = non_blank(result.transcript);
    let manifest = MusicManifest {
        endpoint: "/api/v1/chat/completions",
        model: req.model.clone(),
        prompt: req.prompt.clone(),
        prompt_source: crate::manifest::PROMPT_SOURCE,
        format: format.clone(),
        seed: req.seed,
        provider: req.provider.clone(),
        text: text.clone(),
        transcript: transcript.clone(),
        cost: result.cost,
        created_at: chrono::Utc::now().to_rfc3339(),
        output: AudioOutputMeta {
            path: path.to_string_lossy().into_owned(),
            mime_type: mime.to_string(),
            generation_id: result.generation_id.clone(),
        },
    };
    let mpath = manifest::path(output);
    warnings.extend(manifest::write_or_report(&mpath, &manifest).await);

    Ok(MusicJobResult {
        model: req.model.clone(),
        manifest_path: mpath,
        music: MusicSummary {
            path,
            mime: mime.to_string(),
            text,
            transcript,
        },
        cost: result.cost,
        generation_id: result.generation_id,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    /// Base64 of the first bytes Lyria actually returned: an ID3v2.3 header.
    const ID3_B64: &str = "SUQzAwAAAAAvMg==";
    /// Base64 of a minimal RIFF/WAVE header.
    const WAV_B64: &str = "UklGRiQAAABXQVZF";

    #[test]
    fn normalize_format_trims_lowercases_and_drops_blank() {
        assert_eq!(normalize_format(Some(" WAV ")).as_deref(), Some("wav"));
        assert_eq!(normalize_format(Some("   ")), None);
        assert_eq!(normalize_format(None), None);
    }

    fn sse(events: &[serde_json::Value]) -> String {
        let mut body = String::from(": OPENROUTER PROCESSING\n\n");
        for event in events {
            body.push_str(&format!("data: {event}\n\n"));
        }
        body.push_str("data: [DONE]\n\n");
        body
    }

    fn request(format: Option<&str>) -> MusicGenRequest {
        MusicGenRequest {
            model: "google/lyria-3-clip-preview".to_string(),
            prompt: "upbeat lo-fi loop".to_string(),
            format: format.map(str::to_string),
            seed: Some(7),
            provider: None,
        }
    }

    #[tokio::test]
    async fn run_job_streams_music_saves_the_sniffed_container_and_records_cost() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "model": "google/lyria-3-clip-preview",
                "messages": [{"role": "user", "content": "upbeat lo-fi loop"}],
                "modalities": ["text", "audio"],
                "seed": 7,
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("x-generation-id", "gen-music-1")
                    .set_body_string(sse(&[
                        json!({"id":"gen-music-1","choices":[{"delta":{"content":"<instrumental>"}}]}),
                        json!({"id":"gen-music-1","choices":[{"delta":{"content":"","audio":{"data":ID3_B64}}}]}),
                        json!({"id":"gen-music-1","choices":[{"delta":{"content":""},"finish_reason":"stop"}]}),
                        json!({"id":"gen-music-1","choices":[],"usage":{"cost":0.04}}),
                    ])),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        // The caller's extension is a guess; the bytes decide.
        let base = std::env::temp_dir().join("openrouter-mcp-music-test/track.wav");
        let result = run_job(&client, &request(None), &base).await.unwrap();

        assert_eq!(result.model, "google/lyria-3-clip-preview");
        assert_eq!(result.music.mime, "audio/mpeg");
        assert_eq!(result.music.path.extension().unwrap(), "mp3");
        assert_eq!(result.music.text.as_deref(), Some("<instrumental>"));
        assert_eq!(result.music.transcript, None);
        assert_eq!(result.cost, Some(0.04));
        assert_eq!(result.generation_id.as_deref(), Some("gen-music-1"));
        assert!(result.warnings.is_empty());
        assert_eq!(
            std::fs::read(&result.music.path).unwrap(),
            b"ID3\x03\x00\x00\x00\x00/2"
        );
        // No `audio` block was sent, since no format was requested.
        let sent: serde_json::Value = server.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap();
        assert!(sent.get("audio").is_none(), "sent: {sent}");

        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&result.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["endpoint"], "/api/v1/chat/completions");
        assert_eq!(manifest["text"], "<instrumental>");
        assert_eq!(manifest["cost"], 0.04);
        assert_eq!(manifest["output"]["mime_type"], "audio/mpeg");
        assert_eq!(manifest["output"]["generation_id"], "gen-music-1");
        assert!(manifest.get("format").is_none(), "{manifest}");
    }

    #[tokio::test]
    async fn run_job_sends_the_requested_format_and_saves_what_comes_back() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({ "audio": { "format": "wav" } })))
            .respond_with(ResponseTemplate::new(200).set_body_string(sse(&[
                json!({"choices":[{"delta":{"audio":{"data":WAV_B64,"transcript":"la"}}}]}),
                json!({"choices":[{"delta":{"audio":{"transcript":" la"}}}]}),
            ])))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let base = std::env::temp_dir().join("openrouter-mcp-music-wav/track.mp3");
        let result = run_job(&client, &request(Some(" WAV ")), &base)
            .await
            .unwrap();
        assert_eq!(result.music.mime, "audio/wav");
        assert_eq!(result.music.path.extension().unwrap(), "wav");
        assert_eq!(result.music.text, None);
        assert_eq!(result.music.transcript.as_deref(), Some("la la"));
        assert_eq!(result.cost, None);
    }

    /// A text-only answer (wrong model, or a refusal) is a failure that still
    /// carries the receipt: the provider answered and may bill for it.
    #[tokio::test]
    async fn run_job_fails_with_the_receipt_when_no_audio_arrives() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-text-only")
                    .set_body_string(sse(&[
                        json!({"choices":[{"delta":{"content":"I cannot make music."}}]}),
                        json!({"choices":[{"delta":{"content":"x".repeat(5000)}}]}),
                        json!({"choices":[],"usage":{"cost":0.001}}),
                    ])),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let base = std::env::temp_dir().join("openrouter-mcp-music-none/track.mp3");
        let error = run_job(&client, &request(None), &base)
            .await
            .expect_err("no audio is an error");
        let message = format!("{error:#}");
        assert!(message.contains("no audio"), "got: {message}");
        assert!(message.contains("I cannot make music."), "got: {message}");
        assert!(
            message.contains("[truncated]"),
            "echo not bounded: {message}"
        );
        assert!(message.len() < 2000, "echo not bounded: {}", message.len());
        let receipt = crate::billing::Receipt::from_error(&error).expect("receipt kept");
        assert_eq!(receipt.generation_id.as_deref(), Some("gen-text-only"));
        assert_eq!(receipt.cost, Some(0.001));
        assert!(!base.exists(), "nothing is written without audio");
    }

    /// A body that closes before `[DONE]` still yields the (billed) audio, but
    /// the result says it may be cut short.
    #[tokio::test]
    async fn run_job_warns_when_the_stream_ends_before_done() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(format!(
                "data: {}\n\n",
                json!({"choices":[{"delta":{"audio":{"data":ID3_B64}}}]})
            )))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let base = std::env::temp_dir().join("openrouter-mcp-music-cut/track.mp3");
        let result = run_job(&client, &request(None), &base).await.unwrap();
        assert_eq!(result.music.path.extension().unwrap(), "mp3");
        assert_eq!(result.warnings.len(), 1, "{:?}", result.warnings);
        assert!(
            result.warnings[0].contains("[DONE]"),
            "{:?}",
            result.warnings
        );
    }

    #[tokio::test]
    async fn run_job_surfaces_a_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_string("{\"error\":{\"message\":\"modalities not supported\"}}"),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let base = std::env::temp_dir().join("openrouter-mcp-music-err/track.mp3");
        let error = run_job(&client, &request(None), &base)
            .await
            .expect_err("provider error should propagate");
        assert!(error.to_string().contains("modalities not supported"));
        assert!(crate::billing::Receipt::from_error(&error).is_none());
    }
}
