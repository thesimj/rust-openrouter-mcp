//! Audio orchestration over the synchronous OpenRouter audio APIs: text-to-speech
//! (`POST /api/v1/audio/speech`) and transcription
//! (`POST /api/v1/audio/transcriptions`).
//!
//! Unlike video generation (async job API), both return in one fast call - so
//! these mirror the synchronous `describe_image` path (no task registry).
//! Speech saves a file plus a sidecar manifest; transcription just returns text.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use base64::Engine;

use crate::manifest::{self, AudioManifest, AudioOutputMeta};
use crate::openrouter::{InputAudio, OpenRouterClient, SpeechBody, TranscriptionBody};

/// Audio container formats the transcription endpoint accepts, as the
/// `input_audio.format` values it expects. Keyed by file extension - which is
/// the same string for every format we support.
const TRANSCRIBE_FORMATS: [&str; 7] = ["wav", "mp3", "flac", "m4a", "ogg", "webm", "aac"];

/// Largest audio payload the transcription endpoint accepts (25 MB). Checked
/// before upload so an oversized file fails locally with a clear message rather
/// than after a long transfer.
const MAX_TRANSCRIBE_BYTES: u64 = 25 * 1024 * 1024;

/// The `input_audio.format` value for a file extension, if it is one the
/// endpoint accepts. Case-insensitive.
fn transcribe_format(ext: &str) -> Option<&'static str> {
    let ext = ext.to_ascii_lowercase();
    TRANSCRIBE_FORMATS.into_iter().find(|f| *f == ext)
}

/// Inputs for one transcription request.
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    pub model: String,
    /// Raw base64 audio bytes (no `data:` prefix - upstream rejects those).
    pub data: String,
    /// Container format: one of [`TRANSCRIBE_FORMATS`].
    pub format: String,
    /// Optional ISO-639-1 language hint (e.g. "en").
    pub language: Option<String>,
}

/// A transcript plus the reported USD cost, when present.
pub struct TranscribeResult {
    pub text: String,
    pub cost: Option<f64>,
}

/// Read a local audio file into `(base64, format)` for [`TranscribeRequest`],
/// deriving the format from the file extension and enforcing the upstream size
/// cap. `format_override` wins when the extension is absent or misleading.
pub async fn read_audio_file(
    path: &Path,
    format_override: Option<&str>,
) -> Result<(String, String)> {
    let format = match format_override {
        Some(f) => {
            transcribe_format(f).with_context(|| format!("unsupported audio format {f:?}"))?
        }
        None => {
            let ext = path.extension().unwrap_or_default().to_string_lossy();
            transcribe_format(&ext).with_context(|| {
                format!(
                    "could not infer the audio format from {}; pass format explicitly (one of: {})",
                    path.display(),
                    TRANSCRIBE_FORMATS.join(", ")
                )
            })?
        }
    };

    let size = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("could not read audio file {}", path.display()))?
        .len();
    if size > MAX_TRANSCRIBE_BYTES {
        bail!(
            "audio file is {size} bytes; the transcription endpoint accepts at most \
             {MAX_TRANSCRIBE_BYTES}"
        );
    }

    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("could not read audio file {}", path.display()))?;
    Ok((
        base64::engine::general_purpose::STANDARD.encode(bytes),
        format.to_string(),
    ))
}

/// Transcribe audio to text. Requires already-encoded base64 `data` (see
/// [`read_audio_file`]) so the caller decides where the bytes came from.
pub async fn transcribe(
    client: &OpenRouterClient,
    req: &TranscribeRequest,
) -> Result<TranscribeResult> {
    let body = TranscriptionBody {
        model: req.model.clone(),
        input_audio: InputAudio {
            data: req.data.clone(),
            format: req.format.clone(),
        },
        language: req.language.clone(),
    };
    let resp = client.transcribe(&body).await?;
    if resp.text.trim().is_empty() {
        bail!("model returned an empty transcript");
    }
    Ok(TranscribeResult {
        text: resp.text,
        cost: resp.usage.and_then(|u| u.cost),
    })
}

/// Inputs for a single text-to-speech request (domain struct; the wire body is
/// [`openrouter::SpeechBody`]).
#[derive(Debug, Clone)]
pub struct SpeechGenRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    /// `mp3` or `pcm`; defaults to `mp3` so the file extension is deterministic.
    pub response_format: Option<String>,
    pub speed: Option<f64>,
}

/// The saved audio file plus the metadata worth recording.
pub struct AudioSummary {
    pub path: PathBuf,
    pub mime: String,
    pub voice: String,
    pub response_format: String,
}

/// Result of a TTS job: the saved file plus any non-fatal warnings (e.g. a
/// manifest-write failure that did not lose the audio).
pub struct AudioJobResult {
    pub model: String,
    pub manifest_path: PathBuf,
    pub audio: AudioSummary,
    pub warnings: Vec<String>,
}

/// File extension for an audio MIME type, falling back to the requested
/// `response_format` (mp3/pcm) and finally `mp3`.
fn extension_for(mime: &str, response_format: &str) -> &'static str {
    match mime {
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/pcm" | "audio/l16" => "pcm",
        "audio/flac" => "flac",
        "audio/ogg" => "ogg",
        // application/octet-stream and unknown types: trust the requested format.
        _ => match response_format {
            "pcm" => "pcm",
            "wav" => "wav",
            _ => "mp3",
        },
    }
}

/// Run a TTS job: synthesize the speech, save the bytes (extension from the
/// content-type / requested format), and write the sidecar manifest. Shared by
/// the CLI and the MCP tool.
pub async fn run_job(
    client: &OpenRouterClient,
    req: &SpeechGenRequest,
    output: &Path,
    input_source: &str,
) -> Result<AudioJobResult> {
    // Default response_format to mp3 so the extension is deterministic.
    let response_format = req
        .response_format
        .clone()
        .unwrap_or_else(|| "mp3".to_string());

    let body = SpeechBody {
        model: req.model.clone(),
        input: req.input.clone(),
        voice: req.voice.clone(),
        response_format: Some(response_format.clone()),
        speed: req.speed,
    };

    let result = client.speech(&body).await?;
    let ext = extension_for(&result.mime, &response_format);
    let path = output.with_extension(ext);
    crate::output::write_bytes(&path, &result.bytes)
        .await
        .map_err(|e| anyhow::anyhow!("could not write {}: {e}", path.display()))?;

    let mut warnings = Vec::new();
    let manifest = AudioManifest {
        endpoint: "/api/v1/audio/speech",
        model: req.model.clone(),
        input: req.input.clone(),
        input_source: input_source.to_string(),
        voice: req.voice.clone(),
        response_format: response_format.clone(),
        speed: req.speed,
        created_at: chrono::Utc::now().to_rfc3339(),
        output: AudioOutputMeta {
            path: Some(path.to_string_lossy().into_owned()),
            mime_type: Some(result.mime.clone()),
            generation_id: result.generation_id,
            error: None,
        },
    };
    let mpath = manifest::path(output);
    if let Err(e) = manifest::write(&mpath, &manifest).await {
        warnings.push(format!("manifest write failed: {e}"));
    }

    Ok(AudioJobResult {
        model: req.model.clone(),
        manifest_path: mpath,
        audio: AudioSummary {
            path,
            mime: result.mime,
            voice: req.voice.clone(),
            response_format,
        },
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[test]
    fn extension_for_prefers_mime_then_requested_format() {
        assert_eq!(extension_for("audio/mpeg", "mp3"), "mp3");
        assert_eq!(extension_for("audio/wav", "mp3"), "wav");
        // Unknown/opaque content type falls back to the requested format.
        assert_eq!(extension_for("application/octet-stream", "pcm"), "pcm");
        assert_eq!(extension_for("application/octet-stream", "mp3"), "mp3");
    }

    #[tokio::test]
    async fn run_job_synthesizes_speech_and_saves_the_audio() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            // Verify the wire body we build (response_format defaults to mp3).
            .and(body_partial_json(json!({
                "model": "openai/gpt-4o-mini-tts",
                "input": "hello world",
                "voice": "alloy",
                "response_format": "mp3"
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .insert_header("x-generation-id", "gen-audio-1")
                    .set_body_bytes(b"ID3-FAKE-MP3".to_vec()),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = SpeechGenRequest {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: "hello world".to_string(),
            voice: "alloy".to_string(),
            response_format: None,
            speed: None,
        };
        // Pass an output with the "wrong" extension; the saved file is corrected.
        let base = std::env::temp_dir().join("openrouter-mcp-audio-test/speech.wav");
        let result = run_job(&client, &req, &base, "test").await.unwrap();

        assert_eq!(result.model, "openai/gpt-4o-mini-tts");
        assert_eq!(result.audio.mime, "audio/mpeg");
        assert_eq!(result.audio.voice, "alloy");
        assert_eq!(result.audio.response_format, "mp3");
        // content-type audio/mpeg -> .mp3 extension regardless of the input path.
        assert_eq!(result.audio.path.extension().unwrap(), "mp3");
        assert_eq!(std::fs::read(&result.audio.path).unwrap(), b"ID3-FAKE-MP3");
    }

    #[tokio::test]
    async fn run_job_surfaces_a_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(
                ResponseTemplate::new(400).set_body_string("{\"error\":\"unknown voice\"}"),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = SpeechGenRequest {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: "hi".to_string(),
            voice: "not-a-voice".to_string(),
            response_format: None,
            speed: None,
        };
        let base = std::env::temp_dir().join("openrouter-mcp-audio-err/speech.mp3");
        let err = match run_job(&client, &req, &base, "test").await {
            Err(e) => e,
            Ok(_) => panic!("provider error should propagate"),
        };
        assert!(err.to_string().contains("unknown voice"));
    }
}
