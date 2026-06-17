//! Text-to-speech orchestration over the synchronous OpenRouter speech API.
//!
//! Unlike video generation (async job API), `POST /api/v1/audio/speech` returns
//! the raw audio bytes in one fast call - so this mirrors the synchronous
//! `describe_image` path (no task registry): one call, save the file, write a
//! sidecar manifest, return a lean result.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::manifest::{self, AudioManifest, AudioOutputMeta};
use crate::openrouter::{OpenRouterClient, SpeechBody};

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
    /// `/audio/speech` does not return inline usage.cost, so this is always `None`.
    #[allow(dead_code)]
    pub cost: Option<f64>,
    /// OpenRouter generation id, recorded in the manifest.
    #[allow(dead_code)]
    pub generation_id: Option<String>,
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

/// Sidecar manifest path next to the output: `<stem>.manifest.json`.
pub fn manifest_path(base: &Path) -> PathBuf {
    crate::image_gen::manifest_path(base)
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
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, &result.bytes)
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
            generation_id: result.generation_id.clone(),
            error: None,
        },
    };
    let mpath = manifest_path(output);
    if let Err(e) = manifest::write_audio(&mpath, &manifest) {
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
            cost: None,
            generation_id: result.generation_id,
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
