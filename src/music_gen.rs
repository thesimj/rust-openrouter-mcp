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
/// `"  "` never reaches the wire (same rule as `audio_gen`). Shared with the
/// MCP tool so its filename token matches what is sent.
pub(crate) fn normalize_format(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase)
}

/// `(mime, extension)` read from an audio container's magic bytes. Only
/// signatures long enough to be unambiguous live here; a bare MPEG frame is
/// [`looks_like_mpeg_frame`], which callers apply with more care.
pub(crate) fn sniff_container(bytes: &[u8]) -> Option<(&'static str, &'static str)> {
    if bytes.starts_with(b"ID3") {
        Some(("audio/mpeg", "mp3"))
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        Some(("audio/wav", "wav"))
    } else if bytes.starts_with(b"fLaC") {
        Some(("audio/flac", "flac"))
    } else if bytes.starts_with(b"OggS") {
        Some(("audio/ogg", "ogg"))
    } else {
        None
    }
}

/// Whether `bytes` open with a plausible MPEG audio frame header (an MP3
/// with no ID3 tag): 11 sync bits, then no reserved version/layer, bitrate,
/// or sample-rate index. Raw PCM has no header and can start with the same
/// bytes, so this is only trusted when PCM was not requested.
pub(crate) fn looks_like_mpeg_frame(bytes: &[u8]) -> bool {
    let [0xFF, second, third, ..] = bytes else {
        return false;
    };
    second & 0xE0 == 0xE0
        && second & 0x18 != 0x08 // version: 01 is reserved
        && second & 0x06 != 0x00 // layer: 00 is reserved
        && third & 0xF0 != 0xF0 // bitrate index 1111 is invalid
        && third & 0x0C != 0x0C // sample-rate index 11 is reserved
}

/// `(mime, extension)` implied by the requested `audio.format`, for bytes
/// whose container could not be sniffed. Defaults to MP3, which is what the
/// only music provider returns today.
fn requested_container(format: Option<&str>) -> (&'static str, &'static str) {
    match format {
        Some("wav") => ("audio/wav", "wav"),
        Some("flac") => ("audio/flac", "flac"),
        Some("opus") => ("audio/opus", "opus"),
        Some("pcm16") | Some("pcm") => ("audio/pcm", "pcm"),
        _ => ("audio/mpeg", "mp3"),
    }
}

/// The container to save as: sniffed from the bytes when recognizable,
/// otherwise the requested format's, otherwise MP3. A bare MPEG frame sync
/// counts as recognizable unless PCM was requested: raw samples carry no
/// header, so a sync pattern there is the audio, not a container.
pub(crate) fn container_for(bytes: &[u8], format: Option<&str>) -> (&'static str, &'static str) {
    if let Some(container) = sniff_container(bytes) {
        return container;
    }
    let requested = requested_container(format);
    if requested.1 != "pcm" && looks_like_mpeg_frame(bytes) {
        return ("audio/mpeg", "mp3");
    }
    requested
}

fn non_blank(s: String) -> Option<String> {
    (!s.trim().is_empty()).then_some(s)
}

/// Run a music job: stream the completion, decode the audio, save it with the
/// extension its bytes call for, and write the sidecar manifest. Shared by the
/// CLI and the MCP tool.
pub async fn run_job(
    client: &OpenRouterClient,
    req: &MusicGenRequest,
    output: &Path,
    prompt_source: &str,
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
        stream: true,
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
    let (mime, ext) = container_for(&result.audio, format.as_deref());
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
        prompt_source: prompt_source.to_string(),
        format: format.clone(),
        seed: req.seed,
        provider: req.provider.clone(),
        text: text.clone(),
        transcript: transcript.clone(),
        cost: result.cost,
        created_at: chrono::Utc::now().to_rfc3339(),
        output: AudioOutputMeta {
            path: Some(path.to_string_lossy().into_owned()),
            mime_type: Some(mime.to_string()),
            generation_id: result.generation_id.clone(),
            error: None,
        },
    };
    let mpath = manifest::path(output);
    if let Err(e) = manifest::write(&mpath, &manifest).await {
        warnings.push(format!("manifest write failed: {e}"));
    }

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
    fn container_for_sniffs_known_headers_then_trusts_the_requested_format() {
        assert_eq!(container_for(b"ID3\x03\x00", None), ("audio/mpeg", "mp3"));
        assert_eq!(
            container_for(b"\xFF\xFB\x90\x00", Some("wav")),
            ("audio/mpeg", "mp3")
        );
        // Raw PCM can open with the same bytes as an MPEG frame sync; when PCM
        // was asked for, the sync pattern is samples, not a container. A real
        // ID3 tag still wins, since a provider may ignore the request.
        assert_eq!(
            container_for(b"\xFF\xFB\x90\x00", Some("pcm16")),
            ("audio/pcm", "pcm")
        );
        assert_eq!(
            container_for(b"ID3\x03\x00", Some("pcm")),
            ("audio/mpeg", "mp3")
        );
        // A sync with reserved header fields is not an MPEG frame.
        assert!(looks_like_mpeg_frame(b"\xFF\xFB\x90\x00"));
        assert!(
            !looks_like_mpeg_frame(b"\xFF\xE8\x90\x00"),
            "reserved version"
        );
        assert!(
            !looks_like_mpeg_frame(b"\xFF\xF9\x90\x00"),
            "reserved layer"
        );
        assert!(
            !looks_like_mpeg_frame(b"\xFF\xFB\xF0\x00"),
            "invalid bitrate"
        );
        assert!(!looks_like_mpeg_frame(b"\xFF\xFB\x9C\x00"), "reserved rate");
        assert!(!looks_like_mpeg_frame(b"\xFF\xFB"), "too short");
        assert_eq!(
            container_for(b"\xFF\xE8\x90\x00", None),
            ("audio/mpeg", "mp3"),
            "default"
        );
        assert_eq!(
            container_for(b"RIFF\x24\x00\x00\x00WAVEfmt ", None),
            ("audio/wav", "wav")
        );
        assert_eq!(container_for(b"fLaC\x00", None), ("audio/flac", "flac"));
        assert_eq!(container_for(b"OggS\x00", None), ("audio/ogg", "ogg"));
        // Unrecognized bytes: the requested format decides, else mp3.
        assert_eq!(
            container_for(b"\x00\x01\x02", Some("wav")),
            ("audio/wav", "wav")
        );
        assert_eq!(
            container_for(b"\x00\x01\x02", Some("pcm16")),
            ("audio/pcm", "pcm")
        );
        assert_eq!(container_for(b"\x00\x01\x02", None), ("audio/mpeg", "mp3"));
        assert_eq!(container_for(b"", Some("nonsense")), ("audio/mpeg", "mp3"));
    }

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
        let result = run_job(&client, &request(None), &base, "test")
            .await
            .unwrap();

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
        let result = run_job(&client, &request(Some(" WAV ")), &base, "test")
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
        let error = run_job(&client, &request(None), &base, "test")
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
        let result = run_job(&client, &request(None), &base, "test")
            .await
            .unwrap();
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
        let error = run_job(&client, &request(None), &base, "test")
            .await
            .expect_err("provider error should propagate");
        assert!(error.to_string().contains("modalities not supported"));
        assert!(crate::billing::Receipt::from_error(&error).is_none());
    }
}
