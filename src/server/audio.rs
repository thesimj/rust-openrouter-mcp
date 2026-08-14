//! The `generate_audio` text-to-speech tool and its argument struct.

use anyhow::{Context, bail};
use base64::Engine;
use rmcp::{
    ErrorData, RoleServer,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    service::RequestContext,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::audio_gen::{self, SpeechGenRequest};
use crate::server::naming;
use crate::server::result::{MAX_INLINE_AUDIO_BYTES, client_wants_inline_previews};
use crate::server::schema::{de_opt_f64, require_all, scalarize_nullable};

use super::OpenRouterServer;

/// Arguments for the `transcribe_audio` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct TranscribeAudioArgs {
    /// Speech-to-text model id, e.g. "openai/gpt-4o-mini-transcribe" or
    /// "openai/whisper-1". Discover them with list_models using
    /// output_modalities="transcription".
    pub model: String,
    /// Local audio file to transcribe. One of path/base64.
    #[serde(default)]
    pub path: Option<String>,
    /// Inline audio as base64 (a `data:` URL is also accepted). One of
    /// path/base64. Requires `format` unless the data URL names one.
    #[serde(default)]
    pub base64: Option<String>,
    /// Container format: wav, mp3, flac, m4a, ogg, webm, or aac. Inferred from
    /// the file extension when `path` is used.
    #[serde(default)]
    pub format: Option<String>,
    /// Optional ISO-639-1 language hint (e.g. "en", "ja"); improves accuracy.
    #[serde(default)]
    pub language: Option<String>,
}

/// Arguments for the `generate_audio` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct GenerateAudioArgs {
    /// TTS model id, e.g. "hexgrad/kokoro-82m". Voice ids are model-specific, so
    /// pair this with a voice the model actually declares - `list_models` with
    /// output_modalities=speech reports each model's `supported_voices`.
    pub model: String,
    /// REQUIRED (no default): the text to synthesize.
    #[serde(default)]
    pub input: Option<String>,
    /// REQUIRED (no default): voice id, valid only for the chosen model
    /// (e.g. "af_heart" for hexgrad/kokoro-82m).
    #[serde(default)]
    pub voice: Option<String>,
    /// Output audio format: "mp3" (default) or "pcm".
    #[serde(default)]
    pub response_format: Option<String>,
    /// Playback speed (select models only).
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub speed: Option<f64>,
    /// Output file path (extension corrected to the returned format, e.g. .mp3).
    /// Optional: when omitted, an auto-named file is written under
    /// OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp).
    #[serde(default)]
    pub output: Option<String>,
}

#[tool_router(router = audio_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Generate speech (text-to-speech) with an OpenRouter TTS model (e.g. \
        hexgrad/kokoro-82m with voice af_heart) and save the audio to `output`. This is a \
        synchronous, fast call (not a background task). This tool has NO defaults: model, input \
        (the text), and voice must all be specified, or the call fails naming what is \
        missing. Voice ids are model-specific and are not interchangeable between models - \
        call list_models with output_modalities=speech to see each model's supported_voices. `output` is optional - omit it for an auto-named file under \
        OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp). Returns the saved file path in JSON; for sandboxed clients it also returns a \
        native inline audio content block when the file is small enough. response_format defaults \
        to mp3 so the extension is deterministic.",
        annotations(
            title = "Generate Speech",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn generate_audio(
        &self,
        Parameters(args): Parameters<GenerateAudioArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let inline = client_wants_inline_previews(&context);
        self.run_generate_audio(args, inline).await
    }

    /// Core of `generate_audio` (synchronous, mirrors `describe_image`),
    /// parameterized on inline media so tests can drive it directly.
    pub(crate) async fn run_generate_audio(
        &self,
        args: GenerateAudioArgs,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        // No defaults: input and voice are the things agents forget.
        let mut missing: Vec<&str> = Vec::new();
        if args
            .input
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            missing.push("input (the text to synthesize)");
        }
        if args
            .voice
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            missing.push(
                "voice (model-specific voice id, e.g. \"af_heart\" for hexgrad/kokoro-82m; \
                 see supported_voices in list_models)",
            );
        }
        require_all("generate_audio", "speech", &missing)?;

        let model = args.model.clone();
        let req = SpeechGenRequest {
            model: args.model,
            input: args.input.unwrap_or_default(),
            voice: args.voice.unwrap_or_default(),
            response_format: args.response_format,
            speed: args.speed,
        };
        let fmt = req.response_format.as_deref().unwrap_or("mp3");
        let output = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Audio,
            &model,
            &[req.voice.as_str(), fmt],
            None,
        );

        match audio_gen::run_job(&self.client, &req, &output, "inline").await {
            Ok(result) => {
                self.stats.record_audio(&model, true, None).await;
                let mut env = json!({
                    "ok": true,
                    "kind": "audio",
                    "model": result.model,
                    "audio": {
                        "path": result.audio.path.to_string_lossy(),
                        "mime": result.audio.mime,
                        "voice": result.audio.voice,
                        "response_format": result.audio.response_format,
                    },
                    "manifest": result.manifest_path.to_string_lossy(),
                });
                if !result.warnings.is_empty() {
                    env["warnings"] = json!(result.warnings);
                }
                let body = serde_json::to_string_pretty(&env)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                let mut blocks = vec![ContentBlock::text(body)];

                // Inline native AudioContent for sandboxed clients, under the cap.
                if inline_previews {
                    let path = result.audio.path.clone();
                    let mime = result.audio.mime.clone();
                    let small = std::fs::metadata(&path)
                        .map(|m| m.len() <= MAX_INLINE_AUDIO_BYTES)
                        .unwrap_or(false);
                    if small {
                        let read = tokio::task::spawn_blocking(move || std::fs::read(&path))
                            .await
                            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                        if let Ok(bytes) = read {
                            let data = base64::engine::general_purpose::STANDARD.encode(bytes);
                            blocks.push(ContentBlock::audio(data, mime));
                        }
                    }
                }
                Ok(CallToolResult::success(blocks))
            }
            Err(e) => {
                self.stats.record_audio(&model, false, None).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }

    #[tool(
        description = "Transcribe speech to text with an OpenRouter STT model (e.g. \
        openai/gpt-4o-mini-transcribe, openai/whisper-1, or a Voxtral/Chirp model). This is a \
        synchronous, fast call (not a background task). Pass the audio as `path` (a local file, \
        format inferred from its extension) or `base64` (inline data, with `format`); accepted \
        formats are wav, mp3, flac, m4a, ogg, webm, aac, up to 25 MB. An optional `language` \
        hint (ISO-639-1, e.g. \"en\") improves accuracy. Returns the transcript text. Discover \
        STT models with list_models using output_modalities=\"transcription\" - they are not in \
        the default model list. To create speech from text instead, use generate_audio.",
        annotations(
            title = "Transcribe Audio",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    pub(crate) async fn transcribe_audio(
        &self,
        Parameters(args): Parameters<TranscribeAudioArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let model = args.model.clone();
        let req = resolve_transcribe_request(args)
            .await
            .map_err(|e| ErrorData::invalid_params(format!("{e:#}"), None))?;

        match audio_gen::transcribe(&self.client, &req).await {
            Ok(result) => {
                self.stats.record_text(&model, true, result.cost).await;
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    result.text,
                )]))
            }
            Err(e) => {
                self.stats.record_text(&model, false, None).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }
}

/// Resolve `transcribe_audio` arguments to a [`audio_gen::TranscribeRequest`]:
/// exactly one source, base64 decoded from a `data:` URL when given as one, and
/// the format taken from the argument, the data URL, or the file extension.
async fn resolve_transcribe_request(
    args: TranscribeAudioArgs,
) -> anyhow::Result<audio_gen::TranscribeRequest> {
    let path = args.path.filter(|s| !s.trim().is_empty());
    let inline = args.base64.filter(|s| !s.trim().is_empty());
    let (data, format) = match (path, inline) {
        (Some(p), None) => {
            audio_gen::read_audio_file(std::path::Path::new(&p), args.format.as_deref()).await?
        }
        (None, Some(b64)) => {
            // Tolerate a `data:audio/mp3;base64,...` URL: upstream wants the raw
            // bytes, and the subtype is a usable format when none was passed.
            let (from_url, data) = match b64.strip_prefix("data:").and_then(|r| r.split_once(',')) {
                Some((meta, data)) => (
                    meta.split(';')
                        .next()
                        .and_then(|m| m.rsplit('/').next())
                        .map(str::to_string),
                    data.trim().to_string(),
                ),
                None => (None, b64.trim().to_string()),
            };
            let format = args.format.or(from_url).context(
                "base64 audio needs an explicit format (wav, mp3, flac, m4a, ogg, webm, aac)",
            )?;
            (data, format)
        }
        _ => bail!("transcribe_audio needs exactly one of: path or base64"),
    };

    Ok(audio_gen::TranscribeRequest {
        model: args.model,
        data,
        format,
        language: args.language.filter(|s| !s.trim().is_empty()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn generate_audio_synthesizes_and_returns_path_json() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-tool/voice.mp3");
        let args = GenerateAudioArgs {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: Some("hello".to_string()),
            voice: Some("alloy".to_string()),
            response_format: None,
            speed: None,
            output: Some(out.to_string_lossy().into_owned()),
        };
        // inline_previews=false -> JSON only, no embedded audio block.
        let res = server.run_generate_audio(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["ok"], true);
        assert_eq!(v["kind"], "audio");
        assert_eq!(v["audio"]["voice"], "alloy");
        assert_eq!(v["audio"]["mime"], "audio/mpeg");
        assert!(v["audio"]["path"].as_str().unwrap().ends_with(".mp3"));

        // The stats counter recorded the audio generation.
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["audio_files"], 1);
    }

    #[tokio::test]
    async fn generate_audio_embeds_inline_audio_block_for_sandboxed_clients() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-inline/voice.mp3");
        let args = GenerateAudioArgs {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: Some("hello".to_string()),
            voice: Some("alloy".to_string()),
            response_format: None,
            speed: None,
            output: Some(out.to_string_lossy().into_owned()),
        };
        // inline_previews=true (a sandboxed client like Claude Desktop): the
        // small file is embedded as a native audio content block alongside JSON.
        let res = server.run_generate_audio(args, true).await.unwrap();
        let full = serde_json::to_value(&res).unwrap();
        let audio_block = full["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["type"] == "audio")
            .expect("an audio content block is present");
        assert_eq!(audio_block["mimeType"], "audio/mpeg");
        assert!(!audio_block["data"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn transcribe_audio_sends_base64_and_returns_the_transcript() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            // The wire shape: raw base64 under input_audio.data, plus the format
            // and the language hint.
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "model": "openai/whisper-1",
                "input_audio": { "data": "QUJD", "format": "mp3" },
                "language": "en"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "hello there",
                "usage": { "seconds": 1.5, "tokens": 4, "cost": 0.0004 }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server
            .transcribe_audio(Parameters(TranscribeAudioArgs {
                model: "openai/whisper-1".to_string(),
                path: None,
                // A data: URL is tolerated and its subtype supplies the format.
                base64: Some("data:audio/mp3;base64,QUJD".to_string()),
                format: None,
                language: Some("en".to_string()),
            }))
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "hello there");

        // The transcription and its cost were recorded as a text generation.
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 1);
        assert_eq!(stats["actual_cost_usd"], 0.0004);
    }

    #[tokio::test]
    async fn transcribe_audio_rejects_bad_sources_before_any_call() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = |path: Option<&str>, b64: Option<&str>, format: Option<&str>| {
            Parameters(TranscribeAudioArgs {
                model: "m".to_string(),
                path: path.map(str::to_string),
                base64: b64.map(str::to_string),
                format: format.map(str::to_string),
                language: None,
            })
        };

        // Neither source, and both sources, are equally invalid.
        let err = server
            .transcribe_audio(args(None, None, None))
            .await
            .unwrap_err();
        assert!(
            err.message.contains("exactly one of"),
            "got: {}",
            err.message
        );
        let err = server
            .transcribe_audio(args(Some("a.mp3"), Some("QUJD"), None))
            .await
            .unwrap_err();
        assert!(err.message.contains("exactly one of"));

        // Inline audio with no format is unusable (nothing to decode it as).
        let err = server
            .transcribe_audio(args(None, Some("QUJD"), None))
            .await
            .unwrap_err();
        assert!(
            err.message.contains("explicit format"),
            "got: {}",
            err.message
        );

        // An extension the endpoint doesn't accept is caught locally.
        let err = server
            .transcribe_audio(args(Some("note.txt"), None, None))
            .await
            .unwrap_err();
        assert!(
            err.message.contains("could not infer"),
            "got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn transcribe_audio_reads_a_local_file_and_infers_its_format() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                // "ABC" base64-encoded, with the format taken from the .flac extension.
                "input_audio": { "data": "QUJD", "format": "flac" }
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({ "text": "from disk" })),
            )
            .mount(&mock)
            .await;

        let file = std::env::temp_dir().join("openrouter-mcp-transcribe.flac");
        std::fs::write(&file, b"ABC").unwrap();

        let server = server_for(mock.uri());
        let res = server
            .transcribe_audio(Parameters(TranscribeAudioArgs {
                model: "openai/whisper-1".to_string(),
                path: Some(file.to_string_lossy().into_owned()),
                base64: None,
                format: None,
                language: None,
            }))
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "from disk");
    }

    #[tokio::test]
    async fn generate_audio_requires_input_and_voice() {
        // Validation runs before any HTTP call.
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = GenerateAudioArgs {
            model: "m".to_string(),
            input: None,
            voice: Some("  ".to_string()), // blank-after-trim counts as missing
            response_format: None,
            speed: None,
            output: Some("out.mp3".to_string()),
        };
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert!(err.message.contains("input"));
        assert!(err.message.contains("voice"));
        assert!(err.message.contains("no defaults"));
    }
}
