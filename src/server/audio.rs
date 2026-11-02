//! The `generate_audio` (text-to-speech) and `transcribe_audio` (speech-to-text)
//! tools and their argument structs.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rmcp::{
    ErrorData, RoleServer,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    service::RequestContext,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::audio_gen::{self, SpeechGenRequest, TranscribeResult};
use crate::billing::Receipt;
use crate::openrouter::HttpFailure;
use crate::server::naming;
use crate::server::provider::AudioProviderArgs;
use crate::server::result::{
    attach_warnings_errors, client_wants_inline_previews, inline_audio_block, internal_error_from,
    invalid_params_from, json_text_result,
};
use crate::server::schema::{
    RequireFields, de_bool, de_lenient, de_opt_f64, non_blank, require_text, scalarize_nullable,
};
use crate::{output, transcript};

use super::OpenRouterServer;
use super::media;

/// Arguments for the `transcribe_audio` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct TranscribeAudioArgs {
    /// Speech-to-text model id, e.g. "openai/gpt-4o-mini-transcribe" or
    /// "openai/whisper-1". Discover them with list_models using
    /// output_modalities="transcription".
    #[serde(deserialize_with = "crate::server::schema::de_model_id")]
    pub model: String,
    /// Local audio file to transcribe. One of path/base64.
    #[serde(default)]
    pub path: Option<String>,
    /// Inline audio as base64 (a `data:` URL is also accepted). One of
    /// path/base64. Requires `format` unless the data URL names one.
    #[serde(default)]
    pub base64: Option<String>,
    /// Container format: wav, mp3, flac, m4a, ogg, webm, aac, aiff, pcm16 or
    /// pcm24 (support varies by provider). Inferred from the file extension
    /// when `path` is used.
    #[serde(default)]
    pub format: Option<String>,
    /// Optional ISO-639-1 language hint (e.g. "en", "ja"); improves accuracy.
    #[serde(default)]
    pub language: Option<String>,
    /// "json" (default) or "verbose_json" (adds language/duration/segments/words).
    /// Support varies by model: e.g. microsoft/mai-transcribe-2 returns it, while
    /// openai/gpt-4o-transcribe and microsoft/mai-transcribe-1.5 reject it with a
    /// 400, as do providers without structured output.
    #[serde(default)]
    pub response_format: Option<String>,
    /// "segment" and/or "word": per-segment/word timestamps. Ignored unless
    /// response_format="verbose_json"; which fields come back varies by provider.
    #[serde(default)]
    pub timestamp_granularities: Vec<String>,
    /// Sampling temperature (select providers only).
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub temperature: Option<f64>,
    /// Provider block for this request: {"zdr": bool, "data_collection":
    /// "allow"|"deny", "options": {"<provider-slug>": {...}}}. The slug is the
    /// endpoint `tag` that describe_model reports (e.g. "azure", "deepgram",
    /// "elevenlabs"). Only the slug that serves the request is forwarded, under
    /// that provider's own field names. Speaker diarization:
    /// {"options": {"deepgram": {"diarize": true}}},
    /// {"options": {"azure": {"diarization": {"enabled": true}}}} or
    /// {"options": {"elevenlabs": {"diarize": true}}}; with
    /// response_format="verbose_json" the segments and/or words then carry a
    /// "speaker" index, depending on the provider (ElevenLabs labels words only).
    /// Groq takes vocabulary hints as {"options": {"groq": {"prompt": "..."}}}.
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: AudioProviderArgs,
    /// Save the full response body to this path as pretty-printed JSON (the
    /// extension is forced to .json) and return a compact summary in place of
    /// the response. When the call fails, the error is saved beside it as
    /// <stem>.error.json. Optional: without it, a response larger than
    /// OPENROUTER_MCP_TRANSCRIPT_INLINE_MAX bytes (default 65536) is auto-saved
    /// under OPENROUTER_MCP_OUTPUT_DIR and summarized the same way; smaller ones
    /// are returned inline.
    #[serde(default)]
    pub output: Option<String>,
    /// With a saved response, also return the plain transcript text inline as
    /// its own content block. Default false.
    #[serde(default, deserialize_with = "de_bool")]
    pub include_text: bool,
    /// Add a "turns" array of {speaker, start, end, text} to the response,
    /// built from its words: a new turn starts when the speaker changes or
    /// after a pause longer than 4 s. Needs word timestamps
    /// (response_format="verbose_json"). Default false.
    #[serde(default, deserialize_with = "de_bool")]
    pub normalize: bool,
}

/// Default of env `OPENROUTER_MCP_TRANSCRIPT_INLINE_MAX`.
const DEFAULT_TRANSCRIPT_INLINE_MAX_BYTES: usize = 64 * 1024;

/// When `transcribe_audio` saves a response the caller gave no `output` for,
/// and where. Read once when the server starts.
#[derive(Debug, Clone)]
pub(crate) struct TranscriptAutoSave {
    /// A larger inline reply is saved and summarized instead.
    pub inline_max_bytes: usize,
    pub dir: PathBuf,
}

impl TranscriptAutoSave {
    pub(crate) fn from_env() -> Self {
        Self {
            inline_max_bytes: crate::settings::env_number("OPENROUTER_MCP_TRANSCRIPT_INLINE_MAX")
                .unwrap_or(DEFAULT_TRANSCRIPT_INLINE_MAX_BYTES),
            dir: naming::output_dir(),
        }
    }
}

/// Arguments for the `generate_audio` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = RequireFields(&["input"]))]
pub(crate) struct GenerateAudioArgs {
    /// TTS model id, e.g. "hexgrad/kokoro-82m". Voice ids are model-specific, so
    /// pair this with a voice the model actually declares - `list_models` with
    /// output_modalities=speech reports each model's `supported_voices`.
    #[serde(deserialize_with = "crate::server::schema::de_model_id")]
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
    /// {"path": "<local file>"} (wav, mp3, flac, m4a, ogg, webm, aac, aiff; format
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
    /// Provider block for this request: {"zdr": bool, "data_collection":
    /// "allow"|"deny", "options": {"<provider-slug>": {...}}}; describe_model lists each
    /// endpoint's allowed_passthrough_parameters. Only the slug that serves the
    /// request is forwarded. Examples: OpenAI speaking-style instructions
    /// {"options": {"openai": {"instructions": "speak like a calm narrator"}}};
    /// Azure style {"options": {"azure": {"style": "cheerful", "styledegree": 1.0}}}.
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: AudioProviderArgs,
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
        sent as `input_references` and needs no `voice`. `provider.zdr` and \
        `provider.data_collection` (\"allow\"/\"deny\") restrict routing by data policy. \
        Provider-specific settings go in `provider.options` keyed by provider slug, e.g. \
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
        let input = require_text(
            "generate_audio",
            "speech",
            "input (the text to synthesize)",
            args.input,
        )?;

        let provider = args.provider.into_audio_provider()?;
        let voice_reference =
            resolve_voice_reference(args.voice_reference, args.voice_reference_text).await?;

        let model = args.model.clone();
        let req = SpeechGenRequest {
            model: args.model,
            input,
            voice: args.voice,
            response_format: args.response_format,
            speed: args.speed,
            voice_reference,
            provider,
        };
        // The filename tokens are the normalized wire values, so they never
        // diverge from what run_job actually sends.
        let fmt = audio_gen::normalize_response_format(req.response_format.as_deref());
        let voice = audio_gen::normalize_voice(req.voice.as_deref());
        let mut tokens: Vec<&str> = Vec::new();
        tokens.extend(voice.as_deref());
        tokens.push(fmt.as_str());
        let output = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Audio,
            &model,
            &tokens,
            None,
        );

        match audio_gen::run_job(&self.client, req, &output).await {
            Ok(result) => {
                self.stats.record_audio(&model, None);
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
                attach_warnings_errors(&mut env, &result.warnings, &[]);
                let mut res = json_text_result(&env)?;
                // Inline native AudioContent for sandboxed clients, under the cap.
                if inline_previews {
                    res.content
                        .extend(inline_audio_block(result.audio.path, result.audio.mime).await?);
                }
                Ok(res)
            }
            Err(e) => {
                self.stats.record_audio_failure(&model, &e);
                Err(internal_error_from(&e))
            }
        }
    }

    #[tool(
        description = "Transcribe speech to text with an OpenRouter STT model (e.g. \
        openai/gpt-4o-mini-transcribe, openai/whisper-1, or a Voxtral/Chirp model). This is a \
        synchronous call that waits for the provider response. Pass the audio as `path` (a local file, \
        format inferred from its extension) or `base64` (inline data, with `format`); accepted \
        formats are wav, mp3, flac, m4a, ogg, webm, aac, aiff, pcm16 and pcm24 (support varies \
        by provider), with a local 25 MiB limit for both paths and inline data. An optional `language` \
        hint (ISO-639-1, e.g. \"en\") improves accuracy. Returns the transcript text by default \
        (response_format=\"json\"). Set response_format=\"verbose_json\" to get the full \
        response object (language, duration, segments, words, usage, ...) instead; support \
        varies by model (e.g. microsoft/mai-transcribe-2 returns it, while \
        openai/gpt-4o-transcribe and microsoft/mai-transcribe-1.5 reject it with a 400). \
        microsoft/mai-transcribe-2 accepts only wav, mp3 and flac. OpenRouter documents a \
        60-second upstream timeout per request, so split long recordings. When OpenRouter \
        reports a generation id, a second block carries {\"generation_id\"} for \
        get_generation. \
        Set `output` to save the full response body as pretty JSON (extension forced to .json; a \
        failed call leaves <stem>.error.json beside it) and get a compact summary back instead \
        (saved path, bytes, cost, word and speaker counts, speaker_seconds, first/last word times, \
        max_gap_s, a preview), plus the plain text when include_text=true; without `output`, a \
        response over OPENROUTER_MCP_TRANSCRIPT_INLINE_MAX bytes (default 65536) is auto-saved \
        under OPENROUTER_MCP_OUTPUT_DIR and summarized the same way. normalize=true adds a \
        `turns` array ({speaker, start, end, text}; a new turn on a speaker change or a pause over \
        4 s) built from the verbose_json words. \
        timestamp_granularities (\"segment\"/\"word\") is ignored unless \
        response_format=\"verbose_json\". `provider.zdr` and `provider.data_collection` \
        (\"allow\"/\"deny\") restrict routing by data policy. Provider-specific settings go \
        in `provider.options` keyed \
        by provider slug (the endpoint `tag` from describe_model) - e.g. speaker diarization \
        with {\"deepgram\": {\"diarize\": true}}, {\"azure\": {\"diarization\": {\"enabled\": \
        true}}} or {\"elevenlabs\": {\"diarize\": true}}; when the provider diarizes, \
        verbose_json segments and/or words carry a \"speaker\" index (Azure labels both, \
        ElevenLabs only words). describe_model's supports_diarization is false for some models \
        that do diarize (e.g. deepgram/nova-3, google/gemini-3.5-transcribe, \
        x-ai/grok-stt-1.0), so do not rely on it. Discover \
        STT models with list_models using output_modalities=\"transcription\" - they are not in \
        the default model list. To create speech from text instead, use generate_audio.",
        annotations(
            title = "Transcribe Audio",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    pub(crate) async fn transcribe_audio(
        &self,
        Parameters(mut args): Parameters<TranscribeAudioArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let _work = self.admit_work()?;
        let output = non_blank(args.output.take());
        let options = TranscriptOptions {
            include_text: args.include_text,
            normalize: args.normalize,
        };
        let req = resolve_transcribe_request(args).await?;

        let model = req.model.clone();
        let output = output.map(|o| self.transcript_path(Some(o), &model));
        let mut unsaved_error = None;
        let outcome = match audio_gen::transcribe(&self.client, req).await {
            Ok(result) => {
                let receipt = Receipt {
                    cost: result.cost,
                    generation_id: result.generation_id.clone(),
                };
                self.deliver_transcript(result, &model, output, options)
                    .await
                    .map_err(|error| receipt.attach(error))
            }
            Err(error) => {
                if let Some(path) = &output {
                    unsaved_error = save_error_report(path, &error).await.err();
                }
                Err(error)
            }
        };
        self.finish_text_call(
            &model,
            outcome,
            |reply| reply.cost,
            |reply| Ok(reply.result),
        )
        .await
        .map_err(|mut error| {
            if let Some(unsaved) = unsaved_error {
                error.message = format!("{}; {unsaved:#}", error.message).into();
            }
            error
        })
    }
}

/// How `transcribe_audio` shapes a reply, beyond where it saves it.
#[derive(Debug, Clone, Copy)]
struct TranscriptOptions {
    include_text: bool,
    normalize: bool,
}

/// A delivered transcription and its cost for the usage stats.
struct TranscriptReply {
    cost: Option<f64>,
    result: CallToolResult,
}

const NO_WORDS_FOR_TURNS: &str = "normalize was requested but the response has no words to \
     build turns from; request response_format=\"verbose_json\" (and word timestamps where the \
     provider needs them)";

impl OpenRouterServer {
    /// The .json path a transcription is saved to: `output` as given, else an
    /// auto-named file under the auto-save directory.
    fn transcript_path(&self, output: Option<String>, model: &str) -> PathBuf {
        naming::resolve_output_base_in(
            &self.transcript_auto_save.dir,
            output,
            naming::MediaKind::Transcript,
            model,
            &[],
            None,
        )
        .with_extension("json")
    }

    /// Save the response body and return its summary when the caller gave
    /// `output` or the inline reply would exceed the auto-save limit;
    /// otherwise return the reply inline as before.
    async fn deliver_transcript(
        &self,
        result: TranscribeResult,
        model: &str,
        output: Option<PathBuf>,
        options: TranscriptOptions,
    ) -> anyhow::Result<TranscriptReply> {
        let TranscribeResult {
            text,
            cost,
            generation_id,
            mut body,
            verbose,
            mut warnings,
        } = result;
        if options.normalize {
            add_turns(&mut body, &mut warnings);
        }
        let (path, saved, note) = match output {
            Some(path) => (path, serde_json::to_vec_pretty(&body)?, None),
            None => {
                let pretty_body = verbose
                    .then(|| serde_json::to_vec_pretty(&body))
                    .transpose()?;
                let inline_bytes = pretty_body.as_ref().map_or(text.len(), Vec::len);
                let limit = self.transcript_auto_save.inline_max_bytes;
                if inline_bytes <= limit {
                    let result = inline_transcript(text, body, verbose, warnings, generation_id)?;
                    return Ok(TranscriptReply { cost, result });
                }
                let saved = match pretty_body {
                    Some(saved) => saved,
                    None => serde_json::to_vec_pretty(&body)?,
                };
                let note = format!(
                    "auto-saved: the {inline_bytes}-byte response is larger than \
                     OPENROUTER_MCP_TRANSCRIPT_INLINE_MAX ({limit} bytes); pass output to \
                     choose the path"
                );
                (self.transcript_path(None, model), saved, Some(note))
            }
        };
        output::write_bytes(&path, &saved)
            .await
            .with_context(|| format!("could not save the transcription to {}", path.display()))?;

        let mut summary = transcript::summarize(&body);
        summary.insert("saved".into(), path.to_string_lossy().into());
        summary.insert("bytes".into(), saved.len().into());
        summary.insert("model".into(), model.into());
        if let Some(id) = &generation_id {
            summary.insert("generation_id".into(), id.as_str().into());
        }
        if !warnings.is_empty() {
            summary.insert("warnings".into(), warnings.into());
        }
        if let Some(note) = note {
            summary.insert("note".into(), note.into());
        }
        let mut content = vec![ContentBlock::text(serde_json::to_string_pretty(&summary)?)];
        content.extend(generation_id_block(generation_id, Vec::new()));
        if options.include_text {
            content.push(ContentBlock::text(text));
        }
        Ok(TranscriptReply {
            cost,
            result: CallToolResult::success(content),
        })
    }
}

/// Add the `turns` built from the body's words, or say why there are none.
fn add_turns(body: &mut Value, warnings: &mut Vec<String>) {
    let turns = transcript::turns(body);
    match body.as_object_mut() {
        Some(fields) if !turns.is_empty() => {
            fields.insert("turns".into(), turns.into());
        }
        _ => warnings.push(NO_WORDS_FOR_TURNS.to_string()),
    }
}

/// The reply when nothing is saved: the transcript text, or the whole body
/// for verbose_json, then the generation id block. Warnings ride in the
/// verbose body, else in the generation id block.
fn inline_transcript(
    text: String,
    mut body: Value,
    verbose: bool,
    warnings: Vec<String>,
    generation_id: Option<String>,
) -> anyhow::Result<CallToolResult> {
    let (first, block_warnings) = if verbose {
        append_warnings(&mut body, warnings);
        (serde_json::to_string_pretty(&body)?, Vec::new())
    } else {
        (text, warnings)
    };
    let mut content = vec![ContentBlock::text(first)];
    content.extend(generation_id_block(generation_id, block_warnings));
    Ok(CallToolResult::success(content))
}

/// Add `warnings` to the body's `warnings` array, creating it when absent.
/// A provider's own non-array `warnings` field is left untouched.
fn append_warnings(body: &mut Value, warnings: Vec<String>) {
    if warnings.is_empty() {
        return;
    }
    let Some(fields) = body.as_object_mut() else {
        return;
    };
    if let Value::Array(existing) = fields
        .entry("warnings")
        .or_insert_with(|| Value::Array(Vec::new()))
    {
        existing.extend(warnings.into_iter().map(Value::from));
    }
}

/// The `{"generation_id"}` block for get_generation, carrying `warnings`
/// when there are any; `None` when there is nothing to carry.
fn generation_id_block(
    generation_id: Option<String>,
    warnings: Vec<String>,
) -> Option<ContentBlock> {
    let mut meta = Map::new();
    if let Some(id) = generation_id {
        meta.insert("generation_id".into(), id.into());
    }
    if !warnings.is_empty() {
        meta.insert("warnings".into(), warnings.into());
    }
    (!meta.is_empty()).then(|| ContentBlock::text(Value::Object(meta).to_string()))
}

/// Write `<stem>.error.json` beside the transcript path: the error message,
/// plus the HTTP status and (truncated) body when OpenRouter answered with
/// an error status.
async fn save_error_report(transcript: &Path, error: &anyhow::Error) -> anyhow::Result<()> {
    let mut report = json!({ "error": format!("{error:#}") });
    if let Some(failure) = error.downcast_ref::<HttpFailure>() {
        report["status"] = failure.status.as_u16().into();
        report["body"] = failure.body.as_str().into();
    }
    let path = transcript.with_extension("error.json");
    output::write_bytes(&path, &serde_json::to_vec_pretty(&report)?)
        .await
        .with_context(|| format!("could not save the error to {}", path.display()))
}

/// Resolve `transcribe_audio` arguments to a [`audio_gen::TranscribeRequest`]:
/// exactly one source, base64 decoded from a `data:` URL when given as one, and
/// the format taken from the argument, the data URL, or the file extension -
/// through the same [`media::resolve_audio_input`] the chat `audio` parts use,
/// so bad input is rejected here, before any call.
async fn resolve_transcribe_request(
    args: TranscribeAudioArgs,
) -> Result<audio_gen::TranscribeRequest, ErrorData> {
    let source = media::AudioInput {
        path: args.path,
        base64: args.base64,
        format: args.format,
    };
    let audio = media::resolve_audio_input(source, "base64 audio needs an explicit format").await?;

    Ok(audio_gen::TranscribeRequest {
        model: args.model,
        audio,
        language: non_blank(args.language),
        response_format: args.response_format,
        timestamp_granularities: args.timestamp_granularities,
        temperature: args.temperature,
        provider: args.provider.into_audio_provider()?,
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
    let text = non_blank(text);
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
    // Decoding up to 15 MiB of base64 to check it belongs off the async runtime.
    crate::resources::run_blocking(move || {
        audio_gen::VoiceReference::new(&data, format.as_deref(), text.as_deref())
    })
    .await
    .map(Some)
    .map_err(|e| invalid_params_from(&e))
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
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-stt")
                    .set_body_json(serde_json::json!({
                        "text": "hello there",
                        "usage": { "seconds": 1.5, "tokens": 4, "cost": 0.0004 }
                    })),
            )
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
                output: None,
                include_text: false,
                normalize: false,
            }))
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "hello there");
        // The generation id follows in its own block, as for chat_completion.
        let meta: serde_json::Value =
            serde_json::from_str(v["content"][1]["text"].as_str().unwrap()).unwrap();
        assert_eq!(meta["generation_id"], "gen-stt");

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
                output: None,
                include_text: false,
                normalize: false,
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

        // Bad inline data or an unknown format is the caller's error, caught
        // before any call, exactly as the same file-based input would be.
        for (b64, format, expected) in [
            ("not base64!", "mp3", "invalid base64"),
            ("QUJD", "exe", "unsupported audio format"),
        ] {
            let err = server
                .transcribe_audio(args(None, Some(b64), Some(format)))
                .await
                .unwrap_err();
            assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS, "{b64:?}");
            assert!(err.message.contains(expected), "got: {}", err.message);
        }
        let stats = server.stats.snapshot();
        assert_eq!(stats["requests_failed"], 0, "{stats}");
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
                output: None,
                include_text: false,
                normalize: false,
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
                output: None,
                include_text: false,
                normalize: false,
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

    /// The data-policy keys reach `/audio/transcriptions` next to `options`.
    #[tokio::test]
    async fn transcribe_audio_forwards_data_policy_to_the_wire() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "provider": { "zdr": true, "data_collection": "deny" }
            })))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "text": "ok" })),
            )
            .mount(&mock)
            .await;

        let args: TranscribeAudioArgs = serde_json::from_value(serde_json::json!({
            "model": "openai/whisper-1",
            "base64": "data:audio/mp3;base64,QUJD",
            "provider": { "zdr": "true", "data_collection": "deny" }
        }))
        .unwrap();
        server_for(mock.uri())
            .transcribe_audio(Parameters(args))
            .await
            .expect("zdr and data_collection must reach the wire");
    }

    /// The data-policy keys reach `/audio/speech` and the manifest records them.
    #[tokio::test]
    async fn generate_audio_forwards_data_policy_to_the_wire() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "provider": { "zdr": true, "data_collection": "allow" }
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let out = std::env::temp_dir().join("openrouter-mcp-audio-policy/voice.mp3");
        let mut args = speech_args(&out);
        args.provider = serde_json::from_value(serde_json::json!({
            "zdr": true, "data_collection": "Allow"
        }))
        .unwrap();
        let res = server_for(mock.uri())
            .run_generate_audio(args, false)
            .await
            .expect("zdr and data_collection must reach the wire");
        let manifest_path = tool_result_json(&res)["manifest"]
            .as_str()
            .unwrap()
            .to_string();
        let manifest: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["provider"]["data_collection"], "allow");
    }

    /// The shared fixture later phases reuse produces exactly the block it
    /// promises, and an invalid block is rejected before any HTTP call.
    #[tokio::test]
    async fn transcribe_audio_uses_the_shared_provider_fixture_and_rejects_bad_options() {
        let (options, expected) = crate::server::test_support::provider_options_fixture();
        let provider = AudioProviderArgs {
            options,
            ..Default::default()
        };
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
            output: None,
            include_text: false,
            normalize: false,
        };
        server
            .transcribe_audio(Parameters(args(provider)))
            .await
            .unwrap();

        let mut options = std::collections::BTreeMap::new();
        options.insert("deepgram".to_string(), serde_json::json!("diarize"));
        let err = server
            .transcribe_audio(Parameters(args(
                crate::server::provider::AudioProviderArgs {
                    options,
                    ..Default::default()
                },
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
    /// wire under `provider`, via the shared options fixture.
    #[tokio::test]
    async fn generate_audio_forwards_provider_options_to_the_wire() {
        let (options, expected) = crate::server::test_support::provider_options_fixture();
        let provider = AudioProviderArgs {
            options,
            ..Default::default()
        };
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
        args.provider = crate::server::provider::AudioProviderArgs {
            options,
            ..Default::default()
        };
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

    /// A mock transcription endpoint that answers every request with `body`
    /// and the generation id "gen-stt".
    async fn mock_transcription(body: serde_json::Value) -> MockServer {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-stt")
                    .set_body_json(body),
            )
            .mount(&mock)
            .await;
        mock
    }

    /// verbose_json arguments for inline MP3 audio, saved to `output` when given.
    fn verbose_args(output: Option<&std::path::Path>) -> TranscribeAudioArgs {
        TranscribeAudioArgs {
            model: "elevenlabs/scribe-v2".to_string(),
            path: None,
            base64: Some("data:audio/mp3;base64,QUJD".to_string()),
            format: None,
            language: None,
            response_format: Some("verbose_json".to_string()),
            timestamp_granularities: vec![],
            temperature: None,
            provider: Default::default(),
            output: output.map(|p| p.to_string_lossy().into_owned()),
            include_text: false,
            normalize: false,
        }
    }

    /// The ElevenLabs Scribe shape: speakers on the words only, one segment
    /// with no speaker.
    fn scribe_body() -> serde_json::Value {
        serde_json::json!({
            "text": "hello there friend",
            "language": "eng",
            "duration": 4.0,
            "segments": [{"id": 0, "start": 0.0, "end": 4.0, "text": "hello there friend"}],
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "speaker": 0, "type": "word"},
                {"word": "there", "start": 0.6, "end": 1.0, "speaker": 0, "type": "word"},
                {"word": "friend", "start": 3.0, "end": 4.0, "speaker": 1, "type": "word"}
            ],
            "usage": {"cost": 0.0717, "seconds": 4.0}
        })
    }

    /// The content blocks of a tool result, as JSON.
    fn content_blocks(res: &CallToolResult) -> Vec<serde_json::Value> {
        serde_json::to_value(res).unwrap()["content"]
            .as_array()
            .unwrap()
            .clone()
    }

    fn read_json(path: &std::path::Path) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    #[tokio::test]
    async fn transcribe_audio_with_output_writes_the_raw_response_and_returns_a_summary() {
        let mock = mock_transcription(scribe_body()).await;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("chunk0.scribe.json");
        let server = server_for(mock.uri());
        let res = server
            .transcribe_audio(Parameters(verbose_args(Some(&out))))
            .await
            .unwrap();

        assert_eq!(read_json(&out), scribe_body());
        let summary = tool_result_json(&res);
        assert_eq!(summary["saved"], out.to_string_lossy().as_ref());
        assert_eq!(summary["bytes"], std::fs::metadata(&out).unwrap().len());
        assert_eq!(summary["model"], "elevenlabs/scribe-v2");
        assert_eq!(summary["generation_id"], "gen-stt");
        assert_eq!(summary["cost"], 0.0717);
        assert_eq!(summary["usage_seconds"], 4.0);
        assert_eq!(summary["words"], 3, "a count, not the word list");
        assert_eq!(summary["segments"], 1);
        assert_eq!(summary["speakers"], serde_json::json!([0, 1]));
        assert!(summary.get("text").is_none(), "{summary}");
        let blocks = content_blocks(&res);
        assert_eq!(blocks.len(), 2, "{blocks:?}");
        assert_eq!(blocks[1]["text"], r#"{"generation_id":"gen-stt"}"#);

        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 1);
        assert_eq!(stats["actual_cost_usd"], 0.0717);
    }

    /// The saved file keeps the provider's key order, not alphabetical order,
    /// so it reads like the response OpenRouter sent.
    #[tokio::test]
    async fn transcribe_audio_output_keeps_the_provider_key_order() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                r#"{"text":"hi","words":[{"word":"hi","start":0.0,"end":0.5}],"usage":{"seconds":0.5,"cost":0.01}}"#,
                "application/json",
            ))
            .mount(&mock)
            .await;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("ordered.json");
        server_for(mock.uri())
            .transcribe_audio(Parameters(verbose_args(Some(&out))))
            .await
            .unwrap();

        let saved = std::fs::read_to_string(&out).unwrap();
        let expected = r#"{
  "text": "hi",
  "words": [
    {
      "word": "hi",
      "start": 0.0,
      "end": 0.5
    }
  ],
  "usage": {
    "seconds": 0.5,
    "cost": 0.01
  }
}"#;
        assert_eq!(saved, expected);
    }

    #[tokio::test]
    async fn transcribe_audio_summary_reads_speakers_from_words_when_segments_have_none() {
        let mock = mock_transcription(scribe_body()).await;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("scribe.json");
        let res = server_for(mock.uri())
            .transcribe_audio(Parameters(verbose_args(Some(&out))))
            .await
            .unwrap();
        let summary = tool_result_json(&res);
        assert_eq!(summary["speakers"], serde_json::json!([0, 1]));
        assert_eq!(
            summary["speaker_seconds"],
            serde_json::json!({"0": 0.9, "1": 1.0})
        );
        assert_eq!(summary["first_word_s"], 0.0);
        assert_eq!(summary["last_word_s"], 4.0);
        assert_eq!(summary["max_gap_s"], 2.0);
    }

    #[tokio::test]
    async fn transcribe_audio_summary_reads_speakers_from_segments() {
        let mock = mock_transcription(serde_json::json!({
            "text": "hello there",
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.5, "text": "hello", "speaker": 0},
                {"id": 1, "start": 1.5, "end": 4.0, "text": "there", "speaker": 1}
            ],
            "words": [
                {"word": "hello", "start": 0.0, "end": 1.5},
                {"word": "there", "start": 1.5, "end": 4.0}
            ]
        }))
        .await;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("azure.json");
        let res = server_for(mock.uri())
            .transcribe_audio(Parameters(verbose_args(Some(&out))))
            .await
            .unwrap();
        let summary = tool_result_json(&res);
        assert_eq!(summary["speakers"], serde_json::json!([0, 1]));
        assert_eq!(
            summary["speaker_seconds"],
            serde_json::json!({"0": 1.5, "1": 2.5})
        );
    }

    #[tokio::test]
    async fn transcribe_audio_summary_omits_speakers_without_diarization() {
        let mock = mock_transcription(serde_json::json!({
            "text": "hello",
            "segments": [{"id": 0, "start": 0.0, "end": 0.5, "text": "hello"}],
            "words": [{"word": "hello", "start": 0.0, "end": 0.5}]
        }))
        .await;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("plain.json");
        let res = server_for(mock.uri())
            .transcribe_audio(Parameters(verbose_args(Some(&out))))
            .await
            .unwrap();
        let summary = tool_result_json(&res);
        assert_eq!(summary["words"], 1);
        assert!(summary.get("speakers").is_none(), "{summary}");
        assert!(summary.get("speaker_seconds").is_none(), "{summary}");
    }

    #[tokio::test]
    async fn transcribe_audio_output_creates_parent_dirs_and_forces_json_extension() {
        let mock = mock_transcription(scribe_body()).await;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("a/b/chunk0.txt");
        let res = server_for(mock.uri())
            .transcribe_audio(Parameters(verbose_args(Some(&out))))
            .await
            .unwrap();
        let saved = dir.path().join("a/b/chunk0.json");
        assert_eq!(
            tool_result_json(&res)["saved"],
            saved.to_string_lossy().as_ref()
        );
        assert_eq!(read_json(&saved), scribe_body());
        assert!(!out.exists());
    }

    #[tokio::test]
    async fn transcribe_audio_include_text_returns_text_inline_with_the_summary() {
        let mock = mock_transcription(scribe_body()).await;
        let dir = tempfile::tempdir().unwrap();
        let mut args = verbose_args(Some(&dir.path().join("chunk.json")));
        args.include_text = true;
        let res = server_for(mock.uri())
            .transcribe_audio(Parameters(args))
            .await
            .unwrap();
        let blocks = content_blocks(&res);
        assert_eq!(blocks.len(), 3, "{blocks:?}");
        assert!(tool_result_json(&res)["saved"].is_string());
        assert_eq!(blocks[1]["text"], r#"{"generation_id":"gen-stt"}"#);
        assert_eq!(blocks[2]["text"], "hello there friend");
    }

    #[tokio::test]
    async fn transcribe_audio_error_with_output_saves_error_json_and_returns_the_error() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(
                ResponseTemplate::new(503)
                    .set_body_string(r#"{"error":{"message":"upstream overloaded"}}"#),
            )
            .mount(&mock)
            .await;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("chunk0.scribe.json");
        let server = server_for(mock.uri());
        let err = server
            .transcribe_audio(Parameters(verbose_args(Some(&out))))
            .await
            .unwrap_err();
        assert!(err.message.contains("503"), "got: {}", err.message);
        assert!(
            !err.message.contains("could not save"),
            "got: {}",
            err.message
        );

        assert!(!out.exists());
        let report = read_json(&dir.path().join("chunk0.scribe.error.json"));
        assert_eq!(report["error"], err.message.as_ref());
        assert_eq!(report["status"], 503);
        assert_eq!(
            report["body"],
            r#"{"error":{"message":"upstream overloaded"}}"#
        );
        assert_eq!(server.stats.snapshot()["requests_failed"], 1);
    }

    #[tokio::test]
    async fn transcribe_audio_error_keeps_the_original_error_when_the_error_file_fails() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
            .mount(&mock)
            .await;
        let dir = tempfile::tempdir().unwrap();
        // A regular file where the parent directory should be.
        let blocker = dir.path().join("blocker");
        std::fs::write(&blocker, b"").unwrap();
        let server = server_for(mock.uri());
        let err = server
            .transcribe_audio(Parameters(verbose_args(Some(&blocker.join("chunk.json")))))
            .await
            .unwrap_err();
        assert!(
            err.message
                .starts_with("OpenRouter /audio/transcriptions returned 503"),
            "got: {}",
            err.message
        );
        assert!(err.message.contains("overloaded"), "got: {}", err.message);
        assert!(
            err.message.contains("chunk.error.json"),
            "got: {}",
            err.message
        );
        assert_eq!(server.stats.snapshot()["requests_failed"], 1);
    }

    #[tokio::test]
    async fn transcribe_audio_without_output_stays_inline_below_the_limit() {
        let mock = mock_transcription(serde_json::json!({
            "text": "short clip", "usage": {"cost": 0.001}
        }))
        .await;
        let dir = tempfile::tempdir().unwrap();
        let mut server = server_for(mock.uri());
        server.transcript_auto_save = TranscriptAutoSave {
            inline_max_bytes: "short clip".len(),
            dir: dir.path().to_path_buf(),
        };
        let mut args = verbose_args(None);
        args.response_format = None;
        let res = server.transcribe_audio(Parameters(args)).await.unwrap();
        let blocks = content_blocks(&res);
        assert_eq!(blocks[0]["text"], "short clip");
        assert_eq!(blocks[1]["text"], r#"{"generation_id":"gen-stt"}"#);
        assert_eq!(blocks.len(), 2);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn transcribe_audio_without_output_auto_saves_above_the_limit() {
        let body = serde_json::json!({ "text": "short clip", "usage": {"cost": 0.001} });
        let mock = mock_transcription(body.clone()).await;
        let dir = tempfile::tempdir().unwrap();
        let mut server = server_for(mock.uri());
        server.transcript_auto_save = TranscriptAutoSave {
            inline_max_bytes: "short clip".len() - 1,
            dir: dir.path().to_path_buf(),
        };
        let mut args = verbose_args(None);
        args.response_format = None;
        let res = server.transcribe_audio(Parameters(args)).await.unwrap();
        let summary = tool_result_json(&res);
        let saved = std::path::PathBuf::from(summary["saved"].as_str().unwrap());
        assert_eq!(saved.parent(), Some(dir.path()));
        let name = saved.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("transcript_"), "{name}");
        assert!(name.ends_with(".json"), "{name}");
        assert_eq!(read_json(&saved), body);
        let note = summary["note"].as_str().unwrap();
        assert!(note.contains("auto-saved"), "{note}");
        assert!(
            note.contains("OPENROUTER_MCP_TRANSCRIPT_INLINE_MAX"),
            "{note}"
        );
    }

    #[tokio::test]
    async fn transcribe_audio_saves_cyrillic_text_unescaped() {
        let body = serde_json::json!({
            "text": "Чи були твого агента",
            "words": [{"word": "Чи", "start": 0.0, "end": 0.3, "speaker": 0}]
        });
        let mock = mock_transcription(body.clone()).await;
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("uk.json");
        let res = server_for(mock.uri())
            .transcribe_audio(Parameters(verbose_args(Some(&out))))
            .await
            .unwrap();
        let raw = std::fs::read_to_string(&out).unwrap();
        assert!(raw.contains("Чи були твого агента"), "{raw}");
        assert!(!raw.contains("\\u"), "{raw}");
        assert_eq!(read_json(&out), body);
        assert_eq!(tool_result_json(&res)["preview"], "Чи були твого агента");
    }

    /// The ignored-verbose_json warning stays on the inline object, as before,
    /// but goes to the summary, not the file, when the body is saved.
    #[tokio::test]
    async fn transcribe_audio_puts_the_ignored_verbose_json_warning_beside_the_body() {
        let body = serde_json::json!({ "text": "hi", "usage": {"cost": 0.001} });
        let mock = mock_transcription(body.clone()).await;
        let server = server_for(mock.uri());

        let inline = tool_result_json(
            &server
                .transcribe_audio(Parameters(verbose_args(None)))
                .await
                .unwrap(),
        );
        assert_eq!(inline["text"], "hi");
        assert!(inline["warnings"][0].as_str().unwrap().contains("ignored"));

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("hi.json");
        let summary = tool_result_json(
            &server
                .transcribe_audio(Parameters(verbose_args(Some(&out))))
                .await
                .unwrap(),
        );
        assert!(summary["warnings"][0].as_str().unwrap().contains("ignored"));
        assert_eq!(read_json(&out), body);
    }

    #[tokio::test]
    async fn transcribe_audio_normalize_adds_turns_to_the_saved_file_and_the_inline_object() {
        let mock = mock_transcription(scribe_body()).await;
        let server = server_for(mock.uri());
        let expected = serde_json::json!([
            {"speaker": 0, "start": 0.0, "end": 1.0, "text": "hello there"},
            {"speaker": 1, "start": 3.0, "end": 4.0, "text": "friend"}
        ]);

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("turns.json");
        let mut args = verbose_args(Some(&out));
        args.normalize = true;
        let summary = tool_result_json(&server.transcribe_audio(Parameters(args)).await.unwrap());
        assert!(summary.get("warnings").is_none(), "{summary}");
        assert_eq!(read_json(&out)["turns"], expected);

        let mut args = verbose_args(None);
        args.normalize = true;
        let inline = tool_result_json(&server.transcribe_audio(Parameters(args)).await.unwrap());
        assert_eq!(inline["turns"], expected);
        assert!(inline["words"].is_array());
    }

    #[tokio::test]
    async fn transcribe_audio_normalize_without_words_warns() {
        let mock = mock_transcription(serde_json::json!({ "text": "hi" })).await;
        let server = server_for(mock.uri());

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("no-words.json");
        let mut args = verbose_args(Some(&out));
        args.response_format = None;
        args.normalize = true;
        let summary = tool_result_json(&server.transcribe_audio(Parameters(args)).await.unwrap());
        assert!(
            summary["warnings"][0]
                .as_str()
                .unwrap()
                .contains("normalize")
        );
        assert!(read_json(&out).get("turns").is_none());

        let mut args = verbose_args(None);
        args.response_format = None;
        args.normalize = true;
        let res = server.transcribe_audio(Parameters(args)).await.unwrap();
        let blocks = content_blocks(&res);
        assert_eq!(blocks[0]["text"], "hi");
        let meta: serde_json::Value =
            serde_json::from_str(blocks[1]["text"].as_str().unwrap()).unwrap();
        assert_eq!(meta["generation_id"], "gen-stt");
        assert!(meta["warnings"][0].as_str().unwrap().contains("normalize"));
    }
}
