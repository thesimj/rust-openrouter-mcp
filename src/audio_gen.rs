//! Audio orchestration over the synchronous OpenRouter audio APIs: text-to-speech
//! (`POST /api/v1/audio/speech`) and transcription
//! (`POST /api/v1/audio/transcriptions`).
//!
//! Unlike video generation (async job API), both return in one fast call - so
//! these mirror the synchronous `describe_image` path (no task registry).
//! Speech saves a file plus a sidecar manifest; transcription returns the text
//! (or the whole `verbose_json` object when that format was requested).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use base64::Engine;

use crate::manifest::{self, AudioManifest, AudioOutputMeta};
use crate::openrouter::{
    InputAudio, OpenRouterClient, ProviderOptions, SpeechBody, SpeechInputReference,
    SpeechReferenceAudio, TranscriptionBody,
};

/// `input_audio.format` values accepted locally for every audio input: the
/// transcription source, chat `input_audio` parts, and the speech voice
/// reference. OpenRouter's examples for transcription (wav, mp3, flac, m4a,
/// ogg, webm, aac) plus what its chat audio guide adds (aiff, pcm16, pcm24);
/// which ones a model takes still varies by provider. The format string
/// doubles as the file extension, except for the raw PCM formats.
pub(crate) const INPUT_AUDIO_FORMATS: [&str; 10] = [
    "wav", "mp3", "flac", "m4a", "ogg", "webm", "aac", "aiff", "pcm16", "pcm24",
];

/// Local decoded audio limit (25 MiB) for every audio input except the voice
/// reference, applied to files and inline data. This bounds memory use;
/// OpenRouter JSON requests may support larger inputs.
const MAX_INPUT_AUDIO_BYTES: u64 = 25 * 1024 * 1024;

/// Decoded cap for a voice-cloning reference sample sent as
/// `input_references[].input_audio` on `/audio/speech` (OpenRouter documents
/// 15 MiB decoded / 20 MiB base64).
pub const MAX_VOICE_REFERENCE_BYTES: u64 = 15 * 1024 * 1024;

/// Documented cap on the `text` part of a voice reference (the transcript of
/// the sample).
pub const MAX_VOICE_REFERENCE_TEXT_CHARS: usize = 10_000;

/// The `input_audio.format` value for a file extension or MIME subtype, if it
/// is one of [`INPUT_AUDIO_FORMATS`]. Case-insensitive.
fn input_audio_format(ext: &str) -> Option<&'static str> {
    let ext = ext.trim().to_ascii_lowercase();
    let ext = match ext.as_str() {
        "mpeg" => "mp3",
        "mp4" | "x-m4a" => "m4a",
        "x-wav" | "wave" => "wav",
        "x-flac" => "flac",
        "aif" | "x-aiff" => "aiff",
        other => other,
    };
    INPUT_AUDIO_FORMATS.into_iter().find(|f| *f == ext)
}

/// Inputs for one transcription request.
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    pub model: String,
    /// The audio, already checked by [`validate_inline_audio`]: raw base64 (no
    /// `data:` prefix - upstream rejects those) and a normalized format.
    pub audio: InputAudio,
    /// Optional ISO-639-1 language hint (e.g. "en").
    pub language: Option<String>,
    /// "json" (default, when `None`) or "verbose_json".
    pub response_format: Option<String>,
    /// "segment"/"word"; verbose_json + OpenAI-compatible providers only.
    pub timestamp_granularities: Vec<String>,
    pub temperature: Option<f64>,
    /// Per-provider passthrough (`provider.options.<slug>`), already validated.
    pub provider: Option<ProviderOptions>,
}

/// A transcript plus the reported USD cost, when present. `verbose` carries the
/// full response object (language/duration/segments/words/...) only when
/// `response_format` was "verbose_json"; the default path leaves it `None`.
pub struct TranscribeResult {
    pub text: String,
    pub cost: Option<f64>,
    /// OpenRouter's `X-Generation-Id` header, when the response carried one.
    pub generation_id: Option<String>,
    pub verbose: Option<serde_json::Value>,
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
            input_audio_format(f).with_context(|| format!("unsupported audio format {f:?}"))?
        }
        None => {
            let ext = path.extension().unwrap_or_default().to_string_lossy();
            input_audio_format(&ext).with_context(|| {
                format!(
                    "could not infer the audio format from {}; pass format explicitly (one of: {})",
                    path.display(),
                    INPUT_AUDIO_FORMATS.join(", ")
                )
            })?
        }
    };

    let path = path.to_path_buf();
    let format = format.to_string();
    crate::resources::run_blocking(move || {
        let bytes = crate::resources::read_file_limited(&path, MAX_INPUT_AUDIO_BYTES as usize)?;
        Ok((
            base64::engine::general_purpose::STANDARD.encode(bytes),
            format,
        ))
    })
    .await
}

/// Validate encoded input before upload, including MIME aliases from data URLs.
/// Whitespace inside the base64 (line-wrapping encoders) is removed so the
/// payload sent upstream is the compact form; padding is optional. Shared
/// with the chat `input_audio` parts (`server::media`).
pub(crate) fn validate_inline_audio(data: &str, format: &str) -> Result<(String, String)> {
    let (data, format) =
        validate_inline_audio_within(data, Some(format), MAX_INPUT_AUDIO_BYTES, "input audio")?;
    Ok((data, format.unwrap_or_default()))
}

/// The shared inline-audio check behind [`validate_inline_audio`] and
/// [`VoiceReference::new`]: compact the base64, cap it at `limit` decoded bytes
/// (checked on the encoded length first so an oversized payload is never
/// decoded), reject empty/invalid data, and normalize the container format
/// when one is given (`format` is optional for the speech reference, required
/// for transcription - the caller decides). `what` names the limit in errors.
fn validate_inline_audio_within(
    data: &str,
    format: Option<&str>,
    limit: u64,
    what: &str,
) -> Result<(String, Option<String>)> {
    let data = crate::base64_codec::compact_base64(data);
    let format = format
        .map(|f| input_audio_format(f).with_context(|| format!("unsupported audio format {f:?}")))
        .transpose()?;
    if data.len() > crate::base64_codec::max_base64_len(limit as usize) {
        bail!("audio exceeds the local {what} limit of {limit} decoded bytes");
    }
    let bytes = crate::base64_codec::decode_base64(&data).context("invalid base64 audio")?;
    if bytes.is_empty() {
        bail!("audio is empty");
    }
    if bytes.len() as u64 > limit {
        bail!("audio exceeds the local {what} limit of {limit} decoded bytes");
    }
    Ok((data.into_owned(), format.map(str::to_string)))
}

/// A validated stateless voice-cloning reference for `/audio/speech`: one
/// audio sample plus an optional transcript of it. Built through [`Self::new`]
/// so every voice reference obeys the documented caps.
#[derive(Debug, Clone)]
pub struct VoiceReference {
    /// Raw base64 of the sample (no `data:` prefix).
    data: String,
    /// Container format when known; omitted on the wire otherwise.
    format: Option<String>,
    /// Transcript of the sample, when given.
    text: Option<String>,
}

impl VoiceReference {
    /// Validate a reference: base64 sample under [`MAX_VOICE_REFERENCE_BYTES`]
    /// decoded, an optional container format (aliases normalized like
    /// transcription), and a transcript under
    /// [`MAX_VOICE_REFERENCE_TEXT_CHARS`] - blank text counts as none.
    pub fn new(data: &str, format: Option<&str>, text: Option<&str>) -> Result<Self> {
        let format = format.map(str::trim).filter(|f| !f.is_empty());
        let (data, format) = validate_inline_audio_within(
            data,
            format,
            MAX_VOICE_REFERENCE_BYTES,
            "voice reference",
        )?;
        let text = text
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string);
        if let Some(t) = &text
            && t.chars().count() > MAX_VOICE_REFERENCE_TEXT_CHARS
        {
            bail!(
                "voice reference text exceeds the {MAX_VOICE_REFERENCE_TEXT_CHARS}-character \
                 limit ({} characters)",
                t.chars().count()
            );
        }
        Ok(Self { data, format, text })
    }

    /// The wire parts, in the documented order: the audio sample, then the
    /// transcript when there is one.
    fn into_parts(self) -> Vec<SpeechInputReference> {
        let mut parts = vec![SpeechInputReference::InputAudio {
            input_audio: SpeechReferenceAudio {
                data: self.data,
                format: self.format,
            },
        }];
        if let Some(text) = self.text {
            parts.push(SpeechInputReference::Text { text });
        }
        parts
    }
}

/// Transcribe audio to text. `req.audio` is already resolved and checked by
/// the caller (see [`validate_inline_audio`]), wherever the bytes came from.
pub async fn transcribe(
    client: &OpenRouterClient,
    req: &TranscribeRequest,
) -> Result<TranscribeResult> {
    // Normalize blank values and case before checking the format:
    // trim+lowercase response_format so "Verbose_json"/" json " both work, and
    // trim+lowercase+drop-blank timestamp_granularities entries the same way
    // before they reach the wire ("Word"/" segment " both work too).
    let response_format = req
        .response_format
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_ascii_lowercase);
    let timestamp_granularities: Vec<String> = req
        .timestamp_granularities
        .iter()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    let body = TranscriptionBody {
        model: req.model.clone(),
        input_audio: req.audio.clone(),
        language: req.language.clone(),
        response_format: response_format.clone(),
        timestamp_granularities,
        temperature: req.temperature,
        provider: req.provider.clone(),
    };
    let (raw, generation_id) = client.transcribe(&body).await?;
    let cost = raw
        .get("usage")
        .and_then(|u| u.get("cost"))
        .and_then(serde_json::Value::as_f64);
    let receipt = crate::billing::Receipt {
        cost,
        generation_id: generation_id.clone(),
    };
    let text = raw
        .get("text")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string();
    if text.trim().is_empty() {
        return Err(receipt.attach(anyhow::anyhow!("model returned an empty transcript")));
    }
    let mut verbose = (response_format.as_deref() == Some("verbose_json")).then_some(raw);
    // A provider that silently ignores verbose_json returns the bare json
    // shape; say so instead of handing back an object missing the promised
    // fields with no signal.
    if let Some(v) = verbose.as_mut()
        && v.get("segments").is_none()
        && v.get("words").is_none()
        && v.get("duration").is_none()
        && v.get("language").is_none()
        && let Some(map) = v.as_object_mut()
    {
        let msg = serde_json::Value::String(
            "verbose_json was requested but the response carries none of \
             segments/words/duration/language - the provider likely ignored it \
             (verbose_json needs an OpenAI-compatible provider)"
                .to_string(),
        );
        // The repo-wide warnings shape is a JSON array; append rather than
        // clobber in case a provider ever emits its own warnings field.
        match map.get_mut("warnings") {
            Some(serde_json::Value::Array(a)) => a.push(msg),
            Some(_) => {} // foreign non-array field: leave it untouched
            None => {
                map.insert("warnings".to_string(), serde_json::Value::Array(vec![msg]));
            }
        }
    }
    Ok(TranscribeResult {
        text,
        cost,
        generation_id,
        verbose,
    })
}

/// Inputs for a single text-to-speech request (domain struct; the wire body is
/// [`crate::openrouter::SpeechBody`]).
#[derive(Debug, Clone)]
pub struct SpeechGenRequest {
    pub model: String,
    pub input: String,
    /// Model-specific voice id. Provider-dependent: most models need one and
    /// have no default, voice-cloning models take none. Blank counts as unset.
    pub voice: Option<String>,
    /// `mp3` or `pcm`; defaults to `mp3` so the file extension is deterministic.
    pub response_format: Option<String>,
    pub speed: Option<f64>,
    /// Stateless voice-cloning sample, sent as `input_references`.
    pub voice_reference: Option<VoiceReference>,
    /// Per-provider passthrough (`provider.options.<slug>`), already validated.
    pub provider: Option<ProviderOptions>,
}

/// The saved audio file plus the metadata worth recording.
pub struct AudioSummary {
    pub path: PathBuf,
    pub mime: String,
    /// The voice sent, when one was.
    pub voice: Option<String>,
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

/// The `response_format` that goes on the wire: trimmed and lowercased,
/// defaulting to `mp3` so the file extension is deterministic. Blank counts as
/// absent, so a literal `"  "` is never sent. Shared with the MCP tool so its
/// filename token matches what is sent.
pub(crate) fn normalize_response_format(raw: Option<&str>) -> String {
    crate::music_gen::normalize_format(raw).unwrap_or_else(|| "mp3".to_string())
}

/// The `voice` that goes on the wire and into the manifest: a blank voice is
/// "no voice" (voice-cloning models take none), so it is omitted rather than
/// sent as `"  "`. Shared with the MCP tool for the same reason.
pub(crate) fn normalize_voice(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Run a TTS job: synthesize the speech, save the bytes (typed by
/// [`crate::audio_container::container_for`]), and write the sidecar manifest.
pub async fn run_job(
    client: &OpenRouterClient,
    req: &SpeechGenRequest,
    output: &Path,
) -> Result<AudioJobResult> {
    let response_format = normalize_response_format(req.response_format.as_deref());
    let voice = normalize_voice(req.voice.as_deref());
    let input_references = req
        .voice_reference
        .clone()
        .map(VoiceReference::into_parts)
        .unwrap_or_default();

    let body = SpeechBody {
        model: req.model.clone(),
        input: req.input.clone(),
        voice: voice.clone(),
        response_format: Some(response_format.clone()),
        speed: req.speed,
        input_references,
        provider: req.provider.clone(),
    };

    let result = client.speech(&body).await?;
    let (mime, ext) = crate::audio_container::container_for(
        &result.bytes,
        Some(&result.mime),
        Some(&response_format),
    );
    let path = output.with_extension(ext);
    crate::output::write_bytes(&path, &result.bytes)
        .await
        .map_err(|e| {
            // The speech endpoint returns bytes, not usage: billed, amount unknown.
            let receipt = crate::billing::Receipt {
                cost: None,
                generation_id: result.generation_id.clone(),
            };
            receipt.attach(anyhow::anyhow!("could not write {}: {e}", path.display()))
        })?;

    let mut warnings = Vec::new();
    let manifest = AudioManifest {
        endpoint: "/api/v1/audio/speech",
        model: req.model.clone(),
        input: req.input.clone(),
        input_source: crate::manifest::PROMPT_SOURCE,
        voice: voice.clone(),
        voice_reference: req.voice_reference.is_some(),
        response_format: response_format.clone(),
        speed: req.speed,
        provider: req.provider.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
        output: AudioOutputMeta {
            path: path.to_string_lossy().into_owned(),
            mime_type: mime.to_string(),
            generation_id: result.generation_id,
        },
    };
    let mpath = manifest::path(output);
    warnings.extend(manifest::write_or_report(&mpath, &manifest).await);

    Ok(AudioJobResult {
        model: req.model.clone(),
        manifest_path: mpath,
        audio: AudioSummary {
            path,
            mime: mime.to_string(),
            voice,
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

    /// The saved file and the recorded mime agree: the bytes decide first,
    /// then a content type we know, then the requested format.
    #[tokio::test]
    async fn run_job_types_the_audio_from_its_bytes_then_its_content_type() {
        for (content_type, bytes, requested, mime, ext) in [
            (
                "application/octet-stream",
                &b"ID3\x03\x00"[..],
                None,
                "audio/mpeg",
                "mp3",
            ),
            ("audio/wav", &b"ID3\x03\x00"[..], None, "audio/mpeg", "mp3"),
            ("audio/aac", &b"\x00\x01"[..], None, "audio/aac", "aac"),
            ("audio/x-wav", &b"\x00\x01"[..], None, "audio/wav", "wav"),
            (
                "application/octet-stream",
                &b"\xFF\xFB\x90\x00"[..],
                Some("pcm"),
                "audio/pcm",
                "pcm",
            ),
            (
                "application/octet-stream",
                &b"\x00\x01"[..],
                None,
                "audio/mpeg",
                "mp3",
            ),
        ] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/audio/speech"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .insert_header("content-type", content_type)
                        .set_body_bytes(bytes.to_vec()),
                )
                .mount(&server)
                .await;
            let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
            let req = SpeechGenRequest {
                model: "m".to_string(),
                input: "hi".to_string(),
                voice: None,
                response_format: requested.map(str::to_string),
                speed: None,
                voice_reference: None,
                provider: None,
            };
            let dir = tempfile::tempdir().unwrap();
            let result = run_job(&client, &req, &dir.path().join("speech"))
                .await
                .unwrap();
            let case = format!("{content_type} {bytes:?} {requested:?}");
            assert_eq!(result.audio.mime, mime, "{case}");
            assert_eq!(result.audio.path.extension().unwrap(), ext, "{case}");
        }
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
            voice: Some("alloy".to_string()),
            response_format: None,
            speed: None,
            voice_reference: None,
            provider: None,
        };
        // Pass an output with the "wrong" extension; the saved file is corrected.
        let base = std::env::temp_dir().join("openrouter-mcp-audio-test/speech.wav");
        let result = run_job(&client, &req, &base).await.unwrap();

        assert_eq!(result.model, "openai/gpt-4o-mini-tts");
        assert_eq!(result.audio.mime, "audio/mpeg");
        assert_eq!(result.audio.voice.as_deref(), Some("alloy"));
        assert_eq!(result.audio.response_format, "mp3");
        // content-type audio/mpeg -> .mp3 extension regardless of the input path.
        assert_eq!(result.audio.path.extension().unwrap(), "mp3");
        assert_eq!(std::fs::read(&result.audio.path).unwrap(), b"ID3-FAKE-MP3");
    }

    /// A blank/whitespace-only response_format must
    /// behave exactly like `None` - the mp3 default - not reach the wire as a
    /// literal "  ".
    #[tokio::test]
    async fn run_job_treats_blank_response_format_as_omitted() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(body_partial_json(json!({ "response_format": "mp3" })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE-MP3".to_vec()),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = SpeechGenRequest {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: "hello world".to_string(),
            voice: Some("alloy".to_string()),
            response_format: Some("   ".to_string()),
            speed: None,
            voice_reference: None,
            provider: None,
        };
        let base = std::env::temp_dir().join("openrouter-mcp-audio-blank-format/speech.mp3");
        let result = run_job(&client, &req, &base).await.unwrap();
        assert_eq!(result.audio.response_format, "mp3");
    }

    /// `provider.options` and the voice-cloning `input_references` reach the
    /// wire nested exactly as OpenRouter documents them: the audio part first,
    /// the transcript part second, `voice` absent because none was given.
    #[tokio::test]
    async fn run_job_forwards_provider_options_and_voice_reference() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(body_partial_json(json!({
                "model": "fish-audio/s1",
                "input": "hello world",
                "input_references": [
                    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "mp3"}},
                    {"type": "text", "text": "sample words"}
                ],
                "provider": {"options": {"openai": {"instructions": "speak cheerfully"}}}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE-MP3".to_vec()),
            )
            .mount(&server)
            .await;

        let mut options = std::collections::BTreeMap::new();
        options.insert(
            "openai".to_string(),
            json!({"instructions": "speak cheerfully"}),
        );
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = SpeechGenRequest {
            model: "fish-audio/s1".to_string(),
            input: "hello world".to_string(),
            voice: None,
            response_format: None,
            speed: None,
            voice_reference: Some(
                VoiceReference::new("QUJD", Some("mpeg"), Some(" sample words ")).unwrap(),
            ),
            provider: Some(ProviderOptions { options }),
        };
        let base = std::env::temp_dir().join("openrouter-mcp-audio-ref/speech.mp3");
        let result = run_job(&client, &req, &base).await.unwrap();
        assert_eq!(result.audio.voice, None);

        let sent: serde_json::Value = server.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap();
        assert!(sent.get("voice").is_none(), "sent: {sent}");

        // The manifest records the provider block and that a reference went out.
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&result.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["voice_reference"], true);
        assert_eq!(
            manifest["provider"],
            json!({"options": {"openai": {"instructions": "speak cheerfully"}}})
        );
        assert!(manifest.get("voice").is_none(), "manifest: {manifest}");
    }

    /// A blank voice behaves like none: nothing on the wire, nothing in the
    /// manifest, and no `input_references`/`provider` keys either.
    #[tokio::test]
    async fn run_job_omits_voice_references_and_provider_when_unset() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE-MP3".to_vec()),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = SpeechGenRequest {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: "hello".to_string(),
            voice: Some("   ".to_string()),
            response_format: None,
            speed: None,
            voice_reference: None,
            provider: None,
        };
        let base = std::env::temp_dir().join("openrouter-mcp-audio-novoice/speech.mp3");
        let result = run_job(&client, &req, &base).await.unwrap();
        assert_eq!(result.audio.voice, None);

        let sent: serde_json::Value = server.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap();
        for key in ["voice", "input_references", "provider"] {
            assert!(sent.get(key).is_none(), "{key} sent: {sent}");
        }
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&result.manifest_path).unwrap()).unwrap();
        assert!(manifest.get("voice_reference").is_none(), "{manifest}");
        assert!(manifest.get("provider").is_none(), "{manifest}");
    }

    /// The reference sample obeys the documented caps locally: 15 MiB decoded
    /// audio and a 10000-character transcript. Blank text is dropped, the
    /// format is optional, and a format alias is normalized like transcription.
    #[test]
    fn voice_reference_enforces_the_documented_caps() {
        let ok = VoiceReference::new(" QUJD\r\nRA== ", None, Some("  ")).unwrap();
        assert_eq!(
            ok.into_parts(),
            vec![SpeechInputReference::InputAudio {
                input_audio: SpeechReferenceAudio {
                    data: "QUJDRA==".into(),
                    format: None,
                },
            }]
        );
        let with_text = VoiceReference::new("QUJD", Some("x-wav"), Some("words")).unwrap();
        assert_eq!(
            with_text.into_parts(),
            vec![
                SpeechInputReference::InputAudio {
                    input_audio: SpeechReferenceAudio {
                        data: "QUJD".into(),
                        format: Some("wav".into()),
                    },
                },
                SpeechInputReference::Text {
                    text: "words".into()
                },
            ]
        );

        let long = "x".repeat(MAX_VOICE_REFERENCE_TEXT_CHARS + 1);
        let err = VoiceReference::new("QUJD", None, Some(&long)).unwrap_err();
        assert!(err.to_string().contains("10000"), "got: {err}");
        assert!(
            VoiceReference::new("QUJD", None, Some(&long[..MAX_VOICE_REFERENCE_TEXT_CHARS]))
                .is_ok()
        );

        let too_large = "A".repeat((MAX_VOICE_REFERENCE_BYTES.div_ceil(3) * 4 + 4) as usize);
        let err = VoiceReference::new(&too_large, None, None).unwrap_err();
        assert!(err.to_string().contains("voice reference"), "got: {err}");
        assert!(
            err.to_string()
                .contains(&MAX_VOICE_REFERENCE_BYTES.to_string()),
            "got: {err}"
        );
        assert!(VoiceReference::new("", None, None).is_err());
        assert!(VoiceReference::new("not base64!", None, None).is_err());
        assert!(VoiceReference::new("QUJD", Some("exe"), None).is_err());
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
            voice: Some("not-a-voice".to_string()),
            response_format: None,
            speed: None,
            voice_reference: None,
            provider: None,
        };
        let base = std::env::temp_dir().join("openrouter-mcp-audio-err/speech.mp3");
        let err = match run_job(&client, &req, &base).await {
            Err(e) => e,
            Ok(_) => panic!("provider error should propagate"),
        };
        assert!(err.to_string().contains("unknown voice"));
    }

    fn transcribe_req(response_format: Option<&str>, granularities: &[&str]) -> TranscribeRequest {
        TranscribeRequest {
            model: "openai/whisper-1".to_string(),
            audio: InputAudio {
                data: "QUJD".to_string(),
                format: "mp3".to_string(),
            },
            language: None,
            response_format: response_format.map(str::to_string),
            timestamp_granularities: granularities.iter().map(|s| s.to_string()).collect(),
            temperature: None,
            provider: None,
        }
    }

    /// Response_format gating and the wire value are case/whitespace
    /// insensitive - "Verbose_json" (and " verbose_json ") behave exactly like
    /// "verbose_json".
    #[tokio::test]
    async fn transcribe_normalizes_response_format_case_and_whitespace() {
        for input in [" Verbose_JSON ", "verbose_json", "VERBOSE_JSON"] {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/audio/transcriptions"))
                .and(body_partial_json(
                    json!({ "response_format": "verbose_json" }),
                ))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "text": "hi", "language": "english"
                })))
                .mount(&server)
                .await;

            let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
            let result = transcribe(&client, &transcribe_req(Some(input), &[]))
                .await
                .unwrap();
            // The gate reads the same normalized value, so verbose output is
            // returned regardless of how the caller spelled/cased it.
            assert!(
                result.verbose.is_some(),
                "input {input:?} got: {:?}",
                result.verbose
            );
            assert_eq!(result.verbose.unwrap()["language"], "english");
        }
    }

    /// A blank/whitespace-only response_format is treated as absent (the
    /// default "json" path), not sent to the wire as an empty string.
    #[tokio::test]
    async fn transcribe_drops_blank_response_format() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "text": "hi" })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let result = transcribe(&client, &transcribe_req(Some("   "), &[]))
            .await
            .unwrap();
        assert!(result.verbose.is_none());

        let sent: serde_json::Value = server.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap();
        assert!(sent.get("response_format").is_none(), "sent: {sent}");
    }

    /// Blank entries in timestamp_granularities are dropped before the wire.
    #[tokio::test]
    async fn transcribe_drops_blank_timestamp_granularities_entries() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            // "Word" is lowercased to "word", like response_format, and the
            // blank/whitespace-only entries are dropped entirely.
            .and(body_partial_json(
                json!({ "timestamp_granularities": ["word", "segment"] }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "text": "hi" })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        transcribe(
            &client,
            &transcribe_req(None, &["  ", "Word", "", " Segment "]),
        )
        .await
        .unwrap();
    }

    /// A provider that silently ignores verbose_json returns the bare json
    /// shape; the verbose object must then carry an explicit warning.
    #[tokio::test]
    async fn transcribe_warns_when_the_provider_ignores_verbose_json() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "text": "hi", "usage": { "cost": 0.001 }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let result = transcribe(&client, &transcribe_req(Some("verbose_json"), &[]))
            .await
            .unwrap();
        let v = result.verbose.expect("verbose object still returned");
        let w = v["warnings"][0].as_str().expect("warnings array injected");
        assert!(w.contains("verbose_json"), "got: {w}");
        assert!(w.contains("ignored"), "got: {w}");
        assert!(w.contains("language"), "got: {w}");
    }
}

#[cfg(test)]
mod audit_regression {
    use super::*;
    #[test]
    fn inline_audio_validates_format_encoding_and_local_size_limit() {
        assert_eq!(
            validate_inline_audio(" QUJD ", "mpeg").unwrap(),
            ("QUJD".into(), "mp3".into())
        );
        assert!(validate_inline_audio("invalid!", "mp3").is_err());
        // Line-wrapped and unpadded base64 are compacted, not rejected.
        assert_eq!(
            validate_inline_audio("QUJD\r\nRA==", "wav").unwrap().0,
            "QUJDRA=="
        );
        assert_eq!(validate_inline_audio("QUJDRA", "wav").unwrap().0, "QUJDRA");
        assert!(validate_inline_audio("QUJD", "exe").is_err());
        assert!(validate_inline_audio("", "wav").is_err());
        // Formats the chat audio guide lists beyond the transcription examples.
        for (format, expected) in [
            ("aiff", "aiff"),
            ("x-aiff", "aiff"),
            ("pcm16", "pcm16"),
            ("pcm24", "pcm24"),
        ] {
            assert_eq!(validate_inline_audio("QUJD", format).unwrap().1, expected);
        }
        let too_large = "A".repeat((MAX_INPUT_AUDIO_BYTES.div_ceil(3) * 4 + 4) as usize);
        assert!(
            validate_inline_audio(&too_large, "wav")
                .unwrap_err()
                .to_string()
                .contains("local input audio limit")
        );
    }
}
