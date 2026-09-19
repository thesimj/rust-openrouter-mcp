//! The `generate_audio` text-to-speech tool and its argument struct.

use anyhow::Context;
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
use crate::server::provider::ProviderOptionsArgs;
use crate::server::result::{client_wants_inline_previews, inline_audio_block};
use crate::server::schema::{
    RequireFields, de_lenient, de_opt_f64, require_all, scalarize_nullable,
};

use super::OpenRouterServer;
use super::media;

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
    /// "json" (default) or "verbose_json" (adds language/duration/segments/words -
    /// OpenAI-compatible providers only; others reject it with a 400).
    #[serde(default)]
    pub response_format: Option<String>,
    /// "segment" and/or "word": per-segment/word timestamps. Only honored with
    /// response_format="verbose_json" on an OpenAI-compatible provider.
    #[serde(default)]
    pub timestamp_granularities: Vec<String>,
    /// Sampling temperature (select providers only).
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub temperature: Option<f64>,
    /// Provider block for this request: per-provider passthrough only, as
    /// {"options": {"<provider-slug>": {...}}}; describe_model lists each
    /// endpoint's allowed_passthrough_parameters. Only the slug that serves the
    /// request is forwarded. Speaker diarization:
    /// {"options": {"deepgram": {"diarize": true}}} or
    /// {"options": {"azure": {"diarization": {"enabled": true}}}}; with
    /// response_format="verbose_json" the segments/words then carry a "speaker"
    /// index. Groq takes vocabulary hints as {"options": {"groq": {"prompt": "..."}}}.
    /// Routing fields are ignored by this endpoint.
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ProviderOptionsArgs,
}

/// Arguments for the `generate_audio` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = RequireFields(&["input"]))]
pub(crate) struct GenerateAudioArgs {
    /// TTS model id, e.g. "hexgrad/kokoro-82m". Voice ids are model-specific, so
    /// pair this with a voice the model actually declares - `list_models` with
    /// output_modalities=speech reports each model's `supported_voices`.
    pub model: String,
    /// REQUIRED (no default): the text to synthesize.
    #[serde(default)]
    pub input: Option<String>,
    /// Voice id, valid only for the chosen model (e.g. "af_heart" for
    /// hexgrad/kokoro-82m). Provider-dependent: most TTS models have NO default
    /// voice and fail without one, so pass it unless the model clones a voice
    /// from `voice_reference` instead (e.g. fish-audio).
    #[serde(default)]
    pub voice: Option<String>,
    /// Output audio format: "mp3" (default) or "pcm".
    #[serde(default)]
    pub response_format: Option<String>,
    /// Playback speed (select models only).
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub speed: Option<f64>,
    /// Stateless voice cloning: the audio sample whose voice to imitate, as
    /// {"path": "<local file>"} (wav, mp3, flac, m4a, ogg, webm, aac; format
    /// inferred from the extension) or {"base64": "<base64 or data: URL>",
    /// "format"?: "wav"} (15 MiB decoded max; the format is optional and
    /// omitted from the request when unknown). Omit for no cloning. Sent as
    /// `input_references`; needs no `voice`.
    #[serde(default, deserialize_with = "de_lenient")]
    pub voice_reference: media::AudioInput,
    /// Transcript of the voice reference sample (max 10000 characters);
    /// improves cloning fidelity on models that use it. Needs `voice_reference`.
    #[serde(default)]
    pub voice_reference_text: Option<String>,
    /// Provider block for this request: per-provider passthrough only, as
    /// {"options": {"<provider-slug>": {...}}}; describe_model lists each
    /// endpoint's allowed_passthrough_parameters. Only the slug that serves the
    /// request is forwarded. Examples: OpenAI speaking-style instructions
    /// {"options": {"openai": {"instructions": "speak like a calm narrator"}}};
    /// Azure style {"options": {"azure": {"style": "cheerful", "styledegree": 1.0}}}.
    /// Routing fields are ignored by this endpoint.
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ProviderOptionsArgs,
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
        synchronous call that waits for the provider response. No defaults for the required fields: \
        model and input (the text) must be specified, or the call fails naming what is missing. \
        `voice` is provider-dependent: most TTS models have no default voice and fail without \
        one, so pass it unless the model clones a voice instead. Voice ids are model-specific and \
        are not interchangeable between models - call list_models with output_modalities=speech \
        to see each model's supported_voices. Stateless voice cloning (e.g. fish-audio models): \
        pass the sample as `voice_reference` ({\"path\": ...} for a local file, or \
        {\"base64\": ..., \"format\"?: ...} for inline data or a data: URL; 15 MiB decoded max), \
        optionally with `voice_reference_text` (its transcript, max 10000 characters); it is \
        sent as `input_references` and needs no `voice`. Provider-specific \
        settings go in `provider.options` keyed by provider slug, e.g. \
        {\"openai\": {\"instructions\": \"speak like a calm narrator\"}} for OpenAI speaking-style \
        instructions or {\"azure\": {\"style\": \"cheerful\", \"styledegree\": 1.0}} for Azure \
        styles. `output` is optional - omit it for an auto-named file under \
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
        let _work = self.admit_work()?;
        // No defaults: input is the thing agents forget. `voice` is optional
        // because it is provider-dependent (cloning models take none); a model
        // that needs one rejects the request upstream with its own message.
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
        require_all("generate_audio", "speech", &missing)?;

        let provider = args.provider.into_options()?;
        let voice_reference =
            resolve_voice_reference(args.voice_reference, args.voice_reference_text).await?;

        let model = args.model.clone();
        let req = SpeechGenRequest {
            model: args.model,
            input: args.input.unwrap_or_default(),
            voice: args.voice,
            response_format: args.response_format,
            speed: args.speed,
            voice_reference,
            provider,
        };
        // Same normalization run_job applies to the wire values, so the
        // auto-filename tokens never diverge from what is actually sent.
        let fmt = req
            .response_format
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| "mp3".to_string());
        let mut tokens: Vec<&str> = Vec::new();
        if let Some(voice) = req.voice.as_deref().map(str::trim)
            && !voice.is_empty()
        {
            tokens.push(voice);
        }
        tokens.push(fmt.as_str());
        let output = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Audio,
            &model,
            &tokens,
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
                    blocks.extend(
                        inline_audio_block(result.audio.path.clone(), result.audio.mime.clone())
                            .await?,
                    );
                }
                Ok(CallToolResult::success(blocks))
            }
            Err(e) => {
                self.stats.record_audio(&model, false, None).await;
                self.stats.record_failed_receipt(&model, &e).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }

    #[tool(
        description = "Transcribe speech to text with an OpenRouter STT model (e.g. \
        openai/gpt-4o-mini-transcribe, openai/whisper-1, or a Voxtral/Chirp model). This is a \
        synchronous call that waits for the provider response. Pass the audio as `path` (a local file, \
        format inferred from its extension) or `base64` (inline data, with `format`); accepted \
        formats are wav, mp3, flac, m4a, ogg, webm, aac, with a local 25 MiB limit for both paths and inline data. An optional `language` \
        hint (ISO-639-1, e.g. \"en\") improves accuracy. Returns the transcript text by default \
        (response_format=\"json\"). Set response_format=\"verbose_json\" to get the full \
        response object (language, duration, segments, words, ...) instead - this needs an \
        OpenAI-compatible provider; other providers reject it with a 400. \
        timestamp_granularities (\"segment\"/\"word\") is only honored alongside verbose_json on \
        an OpenAI-compatible provider. Provider-specific settings go in `provider.options` keyed \
        by provider slug - e.g. speaker diarization with {\"deepgram\": {\"diarize\": true}} or \
        {\"azure\": {\"diarization\": {\"enabled\": true}}}; when the provider diarizes, \
        verbose_json segments and words carry a \"speaker\" index. Discover \
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
        let _work = self.admit_work()?;
        let model = args.model.clone();
        let req = resolve_transcribe_request(args)
            .await
            .map_err(|e| ErrorData::invalid_params(format!("{e:#}"), None))?;

        match audio_gen::transcribe(&self.client, &req).await {
            Ok(result) => {
                self.stats.record_text(&model, true, result.cost).await;
                let body = match result.verbose {
                    // verbose_json: return the full response object, not just text.
                    Some(v) => serde_json::to_string_pretty(&v)
                        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?,
                    None => result.text,
                };
                Ok(CallToolResult::success(vec![ContentBlock::text(body)]))
            }
            Err(e) => {
                self.stats.record_text_failure(&model, &e).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }
}

/// Resolve `transcribe_audio` arguments to a [`audio_gen::TranscribeRequest`]:
/// exactly one source, base64 decoded from a `data:` URL when given as one, and
/// the format taken from the argument, the data URL, or the file extension -
/// through the same [`media::load_audio_input`] the chat `audio` parts use.
async fn resolve_transcribe_request(
    args: TranscribeAudioArgs,
) -> anyhow::Result<audio_gen::TranscribeRequest> {
    let source = media::AudioInput {
        path: args.path,
        base64: args.base64,
        format: args.format,
    };
    let (data, format) = media::load_audio_input(source)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e.message))?;
    let format = format
        .context("base64 audio needs an explicit format (wav, mp3, flac, m4a, ogg, webm, aac)")?;

    Ok(audio_gen::TranscribeRequest {
        model: args.model,
        data,
        format,
        language: args.language.filter(|s| !s.trim().is_empty()),
        response_format: args.response_format,
        timestamp_granularities: args.timestamp_granularities,
        temperature: args.temperature,
        provider: args
            .provider
            .into_options()
            .map_err(|e| anyhow::anyhow!("{}", e.message))?,
    })
}

/// Resolve the `voice_reference` object (plus `voice_reference_text`) of
/// `generate_audio` to a validated [`audio_gen::VoiceReference`], or `None`
/// when the object is empty. The sample is loaded like any other audio input
/// ([`media::load_audio_input`]); its format may stay unknown for inline data
/// (the endpoint does not require it) and [`audio_gen::VoiceReference::new`]
/// applies the 15 MiB / 10000-character caps.
pub(crate) async fn resolve_voice_reference(
    reference: media::AudioInput,
    text: Option<String>,
) -> Result<Option<audio_gen::VoiceReference>, ErrorData> {
    let text = text.filter(|t| !t.trim().is_empty());
    if reference.is_empty() {
        if text.is_some() {
            return Err(ErrorData::invalid_params(
                "voice_reference_text needs a sample: pass voice_reference with exactly one \
                 of path or base64",
                None,
            ));
        }
        return Ok(None);
    }
    let (data, format) = media::load_audio_input(reference).await?;
    audio_gen::VoiceReference::new(&data, format.as_deref(), text.as_deref())
        .map(Some)
        .map_err(|e| ErrorData::invalid_params(format!("{e:#}"), None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// The classic generate_audio call: text + voice, everything else unset.
    fn speech_args(out: &std::path::Path) -> GenerateAudioArgs {
        GenerateAudioArgs {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: Some("hello".to_string()),
            voice: Some("alloy".to_string()),
            response_format: None,
            speed: None,
            voice_reference: Default::default(),
            voice_reference_text: None,
            provider: Default::default(),
            output: Some(out.to_string_lossy().into_owned()),
        }
    }

    /// A mock speech endpoint that returns a tiny MP3 for any body.
    async fn mock_speech() -> MockServer {
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
        mock
    }

    #[tokio::test]
    async fn generate_audio_synthesizes_and_returns_path_json() {
        let mock = mock_speech().await;
        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-tool/voice.mp3");
        let args = speech_args(&out);
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
        let mock = mock_speech().await;
        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-inline/voice.mp3");
        let args = speech_args(&out);
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
                response_format: None,
                timestamp_granularities: vec![],
                temperature: None,
                provider: Default::default(),
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
                response_format: None,
                timestamp_granularities: vec![],
                temperature: None,
                provider: Default::default(),
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
    async fn transcribe_audio_verbose_json_returns_the_full_response_object() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "model": "openai/whisper-1",
                "input_audio": { "data": "QUJD", "format": "mp3" },
                "response_format": "verbose_json",
                "timestamp_granularities": ["word", "segment"]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "hello there",
                "language": "english",
                "duration": 1.5,
                "segments": [{"id": 0, "text": "hello there"}],
                "words": [{"word": "hello", "start": 0.0, "end": 0.4}],
                "usage": { "cost": 0.0004 }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server
            .transcribe_audio(Parameters(TranscribeAudioArgs {
                model: "openai/whisper-1".to_string(),
                path: None,
                base64: Some("data:audio/mp3;base64,QUJD".to_string()),
                format: None,
                language: None,
                response_format: Some("verbose_json".to_string()),
                timestamp_granularities: vec!["word".to_string(), "segment".to_string()],
                temperature: None,
                provider: Default::default(),
            }))
            .await
            .unwrap();
        let v = tool_result_json(&res);
        // The full verbose object reaches the caller, not just bare text.
        assert_eq!(v["text"], "hello there");
        assert_eq!(v["language"], "english");
        assert_eq!(v["duration"], 1.5);
        assert!(v["segments"].is_array());
        assert!(v["words"].is_array());

        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["actual_cost_usd"], 0.0004);
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
                response_format: None,
                timestamp_granularities: vec![],
                temperature: None,
                provider: Default::default(),
            }))
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "from disk");
    }

    /// The diarization recipe from the tool description reaches the wire as
    /// `provider.options.<slug>`, nested exactly as OpenRouter documents it.
    #[tokio::test]
    async fn transcribe_audio_forwards_provider_options_to_the_wire() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "model": "deepgram/nova-3",
                "provider": { "options": { "deepgram": { "diarize": true } } }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "text": "hello there",
                "segments": [{"id": 0, "text": "hello there", "speaker": 0}]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        // Deserialized the way a client sends it, so the nested object goes
        // through the lenient path rather than a hand-built struct.
        let args: TranscribeAudioArgs = serde_json::from_value(serde_json::json!({
            "model": "deepgram/nova-3",
            "base64": "data:audio/mp3;base64,QUJD",
            "provider": { "options": { "deepgram": { "diarize": true } } }
        }))
        .unwrap();
        let res = server.transcribe_audio(Parameters(args)).await.unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "hello there");
    }

    /// The shared fixture later phases reuse produces exactly the block it
    /// promises, and an invalid block is rejected before any HTTP call.
    #[tokio::test]
    async fn transcribe_audio_uses_the_shared_provider_fixture_and_rejects_bad_options() {
        let (provider, expected) = crate::server::test_support::provider_options_fixture();
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({ "provider": expected }),
            ))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "ok" })),
            )
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let args = |provider| TranscribeAudioArgs {
            model: "m".to_string(),
            path: None,
            base64: Some("data:audio/mp3;base64,QUJD".to_string()),
            format: None,
            language: None,
            response_format: None,
            timestamp_granularities: vec![],
            temperature: None,
            provider,
        };
        server
            .transcribe_audio(Parameters(args(provider)))
            .await
            .unwrap();

        let mut options = std::collections::BTreeMap::new();
        options.insert("deepgram".to_string(), serde_json::json!("diarize"));
        let err = server
            .transcribe_audio(Parameters(args(
                crate::server::provider::ProviderOptionsArgs { options },
            )))
            .await
            .unwrap_err();
        assert!(err.message.contains("deepgram"), "got: {}", err.message);
        assert_eq!(mock.received_requests().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn generate_audio_requires_input_but_not_voice() {
        // Validation runs before any HTTP call.
        let server = server_for("http://127.0.0.1:9".to_string());
        let mut args = speech_args(std::path::Path::new("out.mp3"));
        args.input = None;
        args.voice = Some("  ".to_string()); // blank voice is simply "no voice"
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert!(err.message.contains("input"));
        assert!(!err.message.contains("voice"), "got: {}", err.message);
        assert!(err.message.contains("no defaults"));
    }

    /// The `provider.options` recipe from the tool description reaches the
    /// wire under `provider`, via the shared fixture every options-only tool uses.
    #[tokio::test]
    async fn generate_audio_forwards_provider_options_to_the_wire() {
        let (provider, expected) = crate::server::test_support::provider_options_fixture();
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({ "voice": "alloy", "provider": expected }),
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-provider/voice.mp3");
        let mut args = speech_args(&out);
        args.provider = provider;
        let res = server.run_generate_audio(args, false).await.unwrap();
        assert_eq!(tool_result_json(&res)["ok"], true);

        // An invalid block is rejected as invalid params before any HTTP call.
        let mut options = std::collections::BTreeMap::new();
        options.insert("openai".to_string(), serde_json::json!("cheerful"));
        let mut args = speech_args(&out);
        args.provider = crate::server::provider::ProviderOptionsArgs { options };
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("openai"), "got: {}", err.message);
        assert_eq!(mock.received_requests().await.unwrap().len(), 1);
    }

    /// Stateless voice cloning: a base64 sample (data URL tolerated, its
    /// subtype supplying the format) plus a transcript become the two
    /// `input_references` parts, and no `voice` is sent when none is given.
    #[tokio::test]
    async fn generate_audio_sends_a_voice_reference_without_a_voice() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "model": "fish-audio/s1",
                "input": "hello",
                "input_references": [
                    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}},
                    {"type": "text", "text": "the sample words"}
                ]
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-clone/voice.mp3");
        // Deserialized the way a client sends it: no voice key at all.
        let args: GenerateAudioArgs = serde_json::from_value(serde_json::json!({
            "model": "fish-audio/s1",
            "input": "hello",
            "voice_reference": {"base64": "data:audio/wav;base64,QUJD"},
            "voice_reference_text": "the sample words",
            "output": out.to_string_lossy(),
        }))
        .unwrap();
        let res = server.run_generate_audio(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["ok"], true);
        assert!(v["audio"]["voice"].is_null(), "got: {v}");

        let sent: serde_json::Value = mock.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap();
        assert!(sent.get("voice").is_none(), "sent: {sent}");
    }

    /// A reference read from disk infers its format from the extension, like
    /// transcribe_audio (`voice_reference.format` would override it). The
    /// nested object arrives the way a stringifying client sends it.
    #[tokio::test]
    async fn generate_audio_reads_a_voice_reference_file() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "input_references": [
                    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "flac"}}
                ]
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let file = std::env::temp_dir().join("openrouter-mcp-voice-ref.flac");
        std::fs::write(&file, b"ABC").unwrap();
        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-clone-file/voice.mp3");
        let mut args = speech_args(&out);
        args.voice_reference = serde_json::from_value(serde_json::json!({
            "path": file.to_string_lossy()
        }))
        .unwrap();
        server.run_generate_audio(args, false).await.unwrap();
        // Exactly one part: no transcript was given, so no text part is sent.
        let sent: serde_json::Value = mock.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap();
        assert_eq!(sent["input_references"].as_array().unwrap().len(), 1);
    }

    /// Reference validation is a caller error (invalid params), caught before
    /// any HTTP call: an over-long transcript, two sources, a transcript or
    /// format with no sample, and inline data with no usable format is fine.
    #[tokio::test]
    async fn generate_audio_rejects_bad_voice_references_as_invalid_params() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let out = std::path::Path::new("out.mp3");

        let mut args = speech_args(out);
        args.voice_reference.base64 = Some("QUJD".to_string());
        args.voice_reference_text = Some("x".repeat(10_001));
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("10000"), "got: {}", err.message);

        let mut args = speech_args(out);
        args.voice_reference.base64 = Some("QUJD".to_string());
        args.voice_reference.path = Some("sample.wav".to_string());
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(
            err.message.contains("exactly one of"),
            "got: {}",
            err.message
        );

        let mut args = speech_args(out);
        args.voice_reference_text = Some("orphan transcript".to_string());
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(
            err.message.contains("voice_reference"),
            "got: {}",
            err.message
        );

        // A format with no sample is a malformed reference, not an absent one.
        let mut args = speech_args(out);
        args.voice_reference.format = Some("wav".to_string());
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(
            err.message.contains("exactly one of"),
            "got: {}",
            err.message
        );

        let mut args = speech_args(out);
        args.voice_reference.base64 = Some("not base64!".to_string());
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }
}
