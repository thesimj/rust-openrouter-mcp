//! The `chat_completion` text tool, its argument struct, and the conversions
//! from tool arguments to the chat wire controls (`response_format`,
//! `plugins`, `web_search_options`, `reasoning`). The CLI `chat` subcommand
//! reuses those conversions so the two never normalize differently.

use std::collections::BTreeMap;

use rmcp::{
    ErrorData,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::chat_gen;
use crate::image_gen;
use crate::openrouter::{JsonSchemaSpec, PdfOptions, Plugin, ResponseFormat, WebSearchOptions};
use crate::server::provider::ProviderRoutingArgs;
use crate::server::schema::{
    RequireFields, de_lenient, de_opt_bool, de_opt_f64, de_opt_uint, require_all,
    scalarize_nullable,
};

use super::OpenRouterServer;
use super::image::{ImageInput, check_image_input, resolve_image_inputs};
use super::media::{
    AudioInput, FileInput, InputKind, VideoInput, check_audio_input, check_file_input,
    check_video_input, resolve_audio_inputs, resolve_file_inputs, resolve_video_inputs,
};

/// Accepted `pdf_engine` values (the file-parser plugin's `pdf.engine`).
const PDF_ENGINES: [&str; 3] = ["mistral-ocr", "cloudflare-ai", "native"];

/// Arguments for the `chat_completion` tool.
#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = RequireFields(&["prompt"]))]
pub(crate) struct ChatCompletionArgs {
    /// Chat/text model id, e.g. "openai/gpt-5.4" or "anthropic/claude-sonnet-4.6".
    /// Discover ids with list_models.
    pub model: String,
    /// REQUIRED: the user message / prompt text to send to the model.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Optional system instruction prepended as a system message.
    #[serde(default)]
    pub system: Option<String>,
    /// Optional input images for a vision-capable model (image-in / text-out).
    /// Each takes exactly one of: path (local file), url (http/https, fetched),
    /// or base64 (a data: URL or raw base64). Best-effort gated on the model's
    /// declared input modalities: the call is rejected only when the catalog
    /// reports the model does NOT accept image input; if its capabilities can't
    /// be determined the request is sent anyway. Omit for a plain text prompt.
    #[serde(default)]
    pub images: Vec<ImageInput>,
    /// Longest-side cap (px) for input images before sending (default 1536,
    /// max 4096). Ignored when no images are provided.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(max = 4096))]
    pub max_image_dimension: Option<u32>,
    /// Optional documents (PDF etc.) for a model with file input. Each takes
    /// exactly one of path / url (fetched) / base64 (needs `filename`); sent
    /// as `file` parts (data URL, 20 MiB each). Pair with `pdf_engine` to
    /// choose the parser. Gated like `images` on the model's input_modalities
    /// ("file").
    #[serde(default)]
    pub files: Vec<FileInput>,
    /// Optional audio clips for a model with audio input. Each takes exactly
    /// one of path / base64 (+ `format` unless inferable); sent as
    /// `input_audio` parts (25 MiB each). Gated on input_modalities ("audio").
    #[serde(default)]
    pub audio: Vec<AudioInput>,
    /// Optional videos for a model with video input. Each takes exactly one of
    /// url (passed through for the provider to fetch) / path / base64 (data
    /// URL, 20 MiB each), plus an optional `processing` hint; sent as
    /// `video_url` parts. Gated on input_modalities ("video").
    #[serde(default)]
    pub videos: Vec<VideoInput>,
    /// Optional sampling temperature.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub temperature: Option<f64>,
    /// Optional maximum number of tokens to generate.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_tokens: Option<u64>,
    /// Seed for reproducible-ish sampling (provider support varies).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub seed: Option<u64>,
    /// Nucleus sampling cutoff (0-1].
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub top_p: Option<f64>,
    /// Top-k sampling cutoff (>= 1; not supported by every provider).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub top_k: Option<u32>,
    /// Stop sequences: generation halts when any of them appears (up to 4).
    #[serde(default)]
    pub stop: Vec<String>,
    /// Frequency penalty (-2 to 2): penalize tokens by how often they appeared.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub frequency_penalty: Option<f64>,
    /// Presence penalty (-2 to 2): penalize tokens that appeared at all.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub presence_penalty: Option<f64>,
    /// Response verbosity: "low", "medium" or "high" (models that support it).
    #[serde(default)]
    pub verbosity: Option<String>,
    /// true = ask for a JSON object reply (`response_format: {"type":
    /// "json_object"}`). Mutually exclusive with `json_schema`. The prompt
    /// should still say "reply in JSON".
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub json_mode: Option<bool>,
    /// A JSON Schema object the reply must conform to (structured outputs):
    /// sent as `response_format: {"type": "json_schema", "json_schema": {"name",
    /// "strict": true, "schema"}}`. The schema's top-level "title" becomes the
    /// name (default "response"). Mutually exclusive with `json_mode`. Check
    /// `supported_parameters` for "structured_outputs" / "response_format".
    #[serde(default, deserialize_with = "de_lenient")]
    pub json_schema: BTreeMap<String, serde_json::Value>,
    /// Web search: set `enabled: true` (or any other field) to attach the
    /// OpenRouter web plugin; `search_context_size` sets `web_search_options`.
    /// Results come back as `annotations` (url_citation) in the tool result.
    #[serde(default, deserialize_with = "de_lenient")]
    pub web_search: WebSearchArgs,
    /// PDF parsing engine for file inputs: "mistral-ocr" (OCR, paid per page),
    /// "cloudflare-ai" (free, text-based PDFs), or "native" (the model's own
    /// file input). Attaches the file-parser plugin.
    #[serde(default)]
    pub pdf_engine: Option<String>,
    /// Optional reasoning effort: "max", "xhigh", "high", "medium", "low",
    /// "minimal" or "none". Omit to keep the model's own default. Accepted
    /// values differ per model - check `reasoning.supported_efforts` from
    /// list_models/describe_model. Reasoning tokens bill as output tokens, so
    /// lower effort is cheaper and faster. Not together with reasoning_max_tokens.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Reasoning token budget (`reasoning.max_tokens`), for models that take a
    /// budget instead of an effort. Not together with reasoning_effort.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub reasoning_max_tokens: Option<u64>,
    /// true = the model reasons but the reasoning text is left out of the
    /// response (`reasoning.exclude`).
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub reasoning_exclude: Option<bool>,
    /// Provider block for this request: routing only, as {"order": [...],
    /// "only": [...], "ignore": [...], "allow_fallbacks", "require_parameters",
    /// "zdr", "sort", "sort_partition"}. Chat completions have no per-provider
    /// `options` passthrough (describe_model's allowed_passthrough_parameters
    /// do not apply here).
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ProviderRoutingArgs,
}

/// The `web_search` block: controls for the OpenRouter web plugin
/// (`plugins: [{"id": "web", ...}]`) plus `web_search_options`.
#[derive(Debug, Default, Clone, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct WebSearchArgs {
    /// true = attach the web plugin with its defaults. Setting any other field
    /// also attaches it; false with other fields set is an error.
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub enabled: Option<bool>,
    /// Search engine: "native" (the provider's own built-in search) or "exa".
    /// Omit to let OpenRouter pick (native when the model has one, else exa).
    #[serde(default)]
    pub engine: Option<String>,
    /// Plugin mode, passed through as documented by OpenRouter (e.g. "auto").
    #[serde(default)]
    pub mode: Option<String>,
    /// Maximum number of search results to feed the model (default 5 upstream).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_results: Option<u32>,
    /// Custom prompt that introduces the search results to the model.
    #[serde(default)]
    pub search_prompt: Option<String>,
    /// Only search these domains (exa engine).
    #[serde(default)]
    pub include_domains: Vec<String>,
    /// Never search these domains (exa engine).
    #[serde(default)]
    pub exclude_domains: Vec<String>,
    /// "low", "medium" or "high": how much search context the model gets
    /// (`web_search_options.search_context_size`; affects price).
    #[serde(default)]
    pub search_context_size: Option<String>,
}

/// Trim and drop blank strings (the repo-wide "blank means absent" rule).
fn non_blank(s: Option<String>) -> Option<String> {
    s.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn clean_list(items: Vec<String>) -> Vec<String> {
    items
        .into_iter()
        .filter_map(|s| non_blank(Some(s)))
        .collect()
}

impl WebSearchArgs {
    /// The web plugin entry and `web_search_options` this block asks for;
    /// `(None, None)` when nothing is set.
    pub(crate) fn into_wire(self) -> Result<(Option<Plugin>, Option<WebSearchOptions>), ErrorData> {
        let engine = non_blank(self.engine);
        let mode = non_blank(self.mode);
        let search_prompt = non_blank(self.search_prompt);
        let include_domains = clean_list(self.include_domains);
        let exclude_domains = clean_list(self.exclude_domains);
        let search_context_size = non_blank(self.search_context_size);
        let any_set = engine.is_some()
            || mode.is_some()
            || self.max_results.is_some()
            || search_prompt.is_some()
            || !include_domains.is_empty()
            || !exclude_domains.is_empty()
            || search_context_size.is_some();
        match self.enabled {
            Some(false) if any_set => {
                return Err(ErrorData::invalid_params(
                    "web_search.enabled is false but other web_search fields are set; \
                     drop them or set enabled to true",
                    None,
                ));
            }
            Some(false) => return Ok((None, None)),
            None if !any_set => return Ok((None, None)),
            _ => {}
        }
        let plugin = Plugin::Web {
            engine,
            mode,
            max_results: self.max_results,
            search_prompt,
            include_domains,
            exclude_domains,
        };
        let options = search_context_size.map(|search_context_size| WebSearchOptions {
            search_context_size,
        });
        Ok((Some(plugin), options))
    }
}

/// `json_mode` / `json_schema` -> `response_format`. A non-empty schema wins
/// its name from a top-level "title" (else "response") and is always strict;
/// asking for both is a contradiction and is rejected.
pub(crate) fn response_format(
    json_mode: Option<bool>,
    json_schema: BTreeMap<String, serde_json::Value>,
) -> Result<Option<ResponseFormat>, ErrorData> {
    if json_schema.is_empty() {
        return Ok(json_mode
            .unwrap_or(false)
            .then_some(ResponseFormat::JsonObject));
    }
    if json_mode == Some(true) {
        return Err(ErrorData::invalid_params(
            "json_mode and json_schema are mutually exclusive: json_schema already asks \
             for structured JSON output",
            None,
        ));
    }
    let name = json_schema
        .get("title")
        .and_then(|t| t.as_str())
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .unwrap_or("response")
        .to_string();
    Ok(Some(ResponseFormat::JsonSchema {
        json_schema: JsonSchemaSpec {
            name,
            strict: true,
            schema: serde_json::Value::Object(json_schema.into_iter().collect()),
        },
    }))
}

/// `pdf_engine` -> the file-parser plugin entry; `None` when unset.
pub(crate) fn file_parser_plugin(pdf_engine: Option<String>) -> Result<Option<Plugin>, ErrorData> {
    let Some(engine) = non_blank(pdf_engine).map(|e| e.to_ascii_lowercase()) else {
        return Ok(None);
    };
    if !PDF_ENGINES.contains(&engine.as_str()) {
        return Err(ErrorData::invalid_params(
            format!(
                "pdf_engine must be one of {}; got {engine:?}",
                PDF_ENGINES.join(", ")
            ),
            None,
        ));
    }
    Ok(Some(Plugin::FileParser {
        pdf: PdfOptions { engine },
    }))
}

/// `reasoning.effort` and `reasoning.max_tokens` are two ways to say the same
/// thing; OpenRouter documents them as exclusive, so reject both up front.
pub(crate) fn check_reasoning(
    effort: Option<&str>,
    max_tokens: Option<u64>,
) -> Result<(), ErrorData> {
    if effort.is_some_and(|e| !e.trim().is_empty()) && max_tokens.is_some() {
        return Err(ErrorData::invalid_params(
            "reasoning_effort and reasoning_max_tokens are mutually exclusive; pass one",
            None,
        ));
    }
    Ok(())
}

/// The typed view of one response annotation for the tool result: a
/// `url_citation` is flattened to its documented fields, anything else is
/// passed through raw.
fn typed_annotation(raw: &serde_json::Value) -> serde_json::Value {
    if raw["type"] != "url_citation" {
        return raw.clone();
    }
    let citation = &raw["url_citation"];
    let mut out = json!({"type": "url_citation"});
    for key in ["url", "title", "content", "start_index", "end_index"] {
        if let Some(v) = citation.get(key).filter(|v| !v.is_null()) {
            out[key] = v.clone();
        }
    }
    out
}

/// The second content block a chat-family tool returns, as pretty JSON:
/// the generation id (what `get_generation` takes), plus - when present -
/// the reasoning text, the typed annotations (web-search citations), a
/// finish_reason other than "stop" (truncation, filtering), with the token
/// counts riding along. `None` only when the response carried none of those,
/// so a result from an upstream that sends no id stays the single text block
/// it always was.
pub(crate) fn result_meta(result: &chat_gen::ChatResult) -> Option<String> {
    let reasoning = result
        .reasoning
        .as_deref()
        .map(str::trim)
        .filter(|r| !r.is_empty());
    let truncated = result.finish_reason.as_deref().is_some_and(|f| f != "stop");
    if result.generation_id.is_none()
        && reasoning.is_none()
        && result.annotations.is_empty()
        && !truncated
    {
        return None;
    }
    let mut meta = json!({});
    if let Some(id) = &result.generation_id {
        meta["generation_id"] = json!(id);
    }
    if let Some(r) = reasoning {
        meta["reasoning"] = json!(r);
    }
    if !result.annotations.is_empty() {
        meta["annotations"] = json!(
            result
                .annotations
                .iter()
                .map(typed_annotation)
                .collect::<Vec<_>>()
        );
    }
    if let Some(f) = &result.finish_reason {
        meta["finish_reason"] = json!(f);
    }
    if result.prompt_tokens.is_some() || result.completion_tokens.is_some() {
        meta["usage"] = json!({});
        if let Some(p) = result.prompt_tokens {
            meta["usage"]["prompt_tokens"] = json!(p);
        }
        if let Some(c) = result.completion_tokens {
            meta["usage"]["completion_tokens"] = json!(c);
        }
    }
    serde_json::to_string_pretty(&meta).ok()
}

#[tool_router(router = chat_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Send a prompt to any OpenRouter chat/text model and return the model's text \
        reply (text out). This call waits for the provider response. Useful \
        to route a sub-task to a DIFFERENT model than the host - e.g. ask a cheaper or specialized \
        model on OpenRouter. Provide `model` (a chat model id; discover with list_models) and \
        `prompt` (the user message); both are required or the call fails naming what is missing. \
        `system` (an optional system instruction), `temperature`, `max_tokens`, `seed`, `top_p`, \
        `top_k`, `stop`, `frequency_penalty`, `presence_penalty` and `verbosity` are optional \
        sampling controls passed straight through. Structured output: `json_mode: true` asks for \
        a JSON object; `json_schema` (a JSON Schema object) enforces that schema with strict \
        mode - not both. Web search: `web_search` ({\"enabled\": true} or engine/max_results/\
        search_prompt/include_domains/exclude_domains/search_context_size) attaches the \
        OpenRouter web plugin; citations come back as `annotations`. `pdf_engine` (mistral-ocr, \
        cloudflare-ai, native) attaches the file-parser plugin for PDF inputs. Reasoning: \
        `reasoning_effort` OR `reasoning_max_tokens` (not both), plus `reasoning_exclude` to \
        keep the reasoning text out of the reply. `provider` takes routing fields only (order, \
        only, ignore, allow_fallbacks, require_parameters, zdr, sort) - chat completions have \
        no per-provider `options` passthrough. Multimodal input: `images` (path/url/base64) \
        for a VISION-capable model, `files` (PDF and other documents; path/url/base64 + \
        filename; use `pdf_engine` to pick the parser), `audio` (path/base64 + format) and \
        `videos` (url passed through, or path/base64 as a data URL). Parts are sent in the \
        order text, images, files, audio, videos. Each kind is gated on the model's declared \
        input_modalities: the call is rejected only when the catalog says the model does NOT \
        accept that kind (find models with list_models input_modalities=image|file|audio|\
        video); unknown capabilities are let through. Returns the assistant's text as the \
        first content block, then a second JSON block with the generation_id (pass it to \
        get_generation for the recorded cost) plus, when the response carries them, \
        reasoning, annotations (url_citation: url, title, content, start_index, \
        end_index), a finish_reason other than \"stop\" (e.g. \"length\" = truncated), and \
        the prompt/completion token counts. Not \
        exposed on purpose: tools/tool_choice, logit_bias, logprobs, prediction, fallback \
        models, min_p/top_a/repetition_penalty.",
        annotations(
            title = "Chat Completion",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn chat_completion(
        &self,
        Parameters(args): Parameters<ChatCompletionArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        self.run_chat_completion(args).await
    }

    /// Core of `chat_completion` (synchronous), split out so tests drive it directly.
    pub(crate) async fn run_chat_completion(
        &self,
        args: ChatCompletionArgs,
    ) -> Result<CallToolResult, ErrorData> {
        let _work = self.admit_work()?;
        let mut missing: Vec<&str> = Vec::new();
        if args
            .prompt
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            missing.push("prompt (the user message)");
        }
        require_all("chat_completion", "text", &missing)?;

        // Argument-level contradictions and vocabulary, all before any network.
        check_reasoning(args.reasoning_effort.as_deref(), args.reasoning_max_tokens)?;
        let response_format = response_format(args.json_mode, args.json_schema)?;
        let (web_plugin, web_search_options) = args.web_search.into_wire()?;
        let pdf_plugin = file_parser_plugin(args.pdf_engine)?;
        let plugins: Vec<Plugin> = web_plugin.into_iter().chain(pdf_plugin).collect();
        let provider = args.provider.into_routing()?;

        // Validate each input's shape cheaply (no fetch) so a malformed entry
        // reports the accurate "exactly one of ..." error rather than being
        // masked by the capability gate below. Then gate each present kind on
        // the model's declared capabilities before any network-bound
        // resolution. All of it is skipped for text-only calls.
        if !args.images.is_empty() {
            for img in &args.images {
                check_image_input(img)?;
            }
            self.ensure_input_modality(&args.model, InputKind::Image)
                .await?;
        }
        if !args.files.is_empty() {
            for f in &args.files {
                check_file_input(f)?;
            }
            self.ensure_input_modality(&args.model, InputKind::File)
                .await?;
        }
        if !args.audio.is_empty() {
            for a in &args.audio {
                check_audio_input(a)?;
            }
            self.ensure_input_modality(&args.model, InputKind::Audio)
                .await?;
        }
        if !args.videos.is_empty() {
            for v in &args.videos {
                check_video_input(v)?;
            }
            self.ensure_input_modality(&args.model, InputKind::Video)
                .await?;
        }
        let images = resolve_image_inputs(args.images).await?;
        let files = resolve_file_inputs(args.files).await?;
        let audio = resolve_audio_inputs(args.audio).await?;
        let videos = resolve_video_inputs(args.videos).await?;
        // The dimension cap only matters when there are images to normalize.
        let max_dim = if images.is_empty() {
            0
        } else {
            image_gen::resolve_max_dimension(args.max_image_dimension)
        };

        let prompt = args.prompt.unwrap_or_default();
        // Record success only after text is actually extracted (an empty-choices
        // or empty-content response is an error, not a successful generation).
        match chat_gen::complete(
            &self.client,
            &chat_gen::ChatInputs {
                model: &args.model,
                system: args.system.as_deref(),
                prompt: &prompt,
                temperature: args.temperature,
                max_tokens: args.max_tokens,
                images: &images,
                max_image_dimension: max_dim,
                files: &files,
                audio: &audio,
                videos: &videos,
                reasoning_effort: args.reasoning_effort.as_deref(),
                reasoning_max_tokens: args.reasoning_max_tokens,
                reasoning_exclude: args.reasoning_exclude,
                seed: args.seed,
                top_p: args.top_p,
                top_k: args.top_k,
                stop: &args.stop,
                frequency_penalty: args.frequency_penalty,
                presence_penalty: args.presence_penalty,
                verbosity: args.verbosity.as_deref(),
                response_format,
                plugins,
                web_search_options,
                provider,
            },
        )
        .await
        {
            Ok(result) => {
                self.stats.record_text(&args.model, true, result.cost).await;
                let mut blocks = vec![ContentBlock::text(result.text.clone())];
                if let Some(meta) = result_meta(&result) {
                    blocks.push(ContentBlock::text(meta));
                }
                Ok(CallToolResult::success(blocks))
            }
            Err(e) => {
                self.stats.record_text(&args.model, false, None).await;
                self.stats.record_failed_receipt(&args.model, &e).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }

    /// Best-effort early rejection when `model` is *known* not to accept
    /// `kind` input (image, file, audio, video). The model's input modalities
    /// are looked up via list_models and cached (after the first call
    /// completes; a burst of concurrent first-time calls for the same model
    /// may each fetch).
    ///
    /// This is deliberately fail-open: the lookup is a fuzzy catalog search, so
    /// if it errors (network blip, an id the search doesn't surface, a routing-
    /// suffixed id like `:nitro`/`:floor`) or reports no modality metadata, the
    /// request is allowed through and the actual `/chat/completions` call remains
    /// the authority on compatibility. We reject only when the catalog positively
    /// reports input modalities that don't include `kind` — the common, clear
    /// case (e.g. sending an image to a text-only model).
    async fn ensure_input_modality(&self, model: &str, kind: InputKind) -> Result<(), ErrorData> {
        let modalities = match self.model_caps.get(model).await {
            Some(cached) => cached,
            None => match self.client.model_input_modalities(model).await {
                Ok(modalities) => {
                    // Cache only a definite answer; an empty list means "unknown"
                    // (missing/lagging catalog metadata) and must not be pinned for
                    // the process lifetime.
                    if !modalities.is_empty() {
                        self.model_caps.put(model, modalities.clone()).await;
                    }
                    modalities
                }
                // Capabilities couldn't be verified — don't block a possibly-valid call.
                Err(_) => return Ok(()),
            },
        };
        let noun = kind.noun();
        if !modalities.is_empty() && !modalities.iter().any(|m| m == noun) {
            return Err(ErrorData::invalid_params(
                format!(
                    "model '{model}' does not accept {noun} input (input modalities: [{}]). \
                     Use list_models with input_modalities={noun} to find a model that does.",
                    modalities.join(", ")
                ),
                None,
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json, valid_png_b64};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// The minimal valid call; tests override what they exercise.
    fn args(model: &str, prompt: &str) -> ChatCompletionArgs {
        ChatCompletionArgs {
            model: model.to_string(),
            prompt: Some(prompt.to_string()),
            ..Default::default()
        }
    }

    /// One inline-base64 image input.
    fn one_image() -> Vec<ImageInput> {
        vec![ImageInput {
            path: None,
            url: None,
            base64: Some(valid_png_b64()),
            label: None,
        }]
    }

    /// Mock `POST /chat/completions` answering "ok" only to a body that
    /// contains `expected` (partial match), so a missing/misnested field fails
    /// the call rather than passing silently.
    async fn mock_chat(expected: serde_json::Value) -> MockServer {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(expected))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "ok"}}]
            })))
            .mount(&mock)
            .await;
        mock
    }

    /// Run `args` against a permissive mock and hand back the body it sent.
    async fn sent_body(args: ChatCompletionArgs) -> serde_json::Value {
        let mock = mock_chat(serde_json::json!({})).await;
        server_for(mock.uri())
            .run_chat_completion(args)
            .await
            .unwrap();
        mock.received_requests().await.unwrap()[0]
            .body_json()
            .unwrap()
    }

    /// Mock `GET /models` so a single model reports the given input modalities.
    async fn mock_model_modalities(mock: &MockServer, id: &str, input_modalities: &[&str]) {
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{
                    "id": id,
                    "architecture": { "input_modalities": input_modalities }
                }]
            })))
            .mount(mock)
            .await;
    }

    #[tokio::test]
    async fn chat_completion_returns_model_text() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            // Confirms system+user messages and that temperature/max_tokens are
            // actually forwarded in the request body (not silently dropped).
            .and(body_partial_json(serde_json::json!({
                "model": "openai/gpt-5.4",
                "temperature": 0.5,
                "max_tokens": 64,
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "say hi"}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "hello back"}, "finish_reason": "stop"}],
                "usage": {"cost": 0.0012, "prompt_tokens": 5, "completion_tokens": 2}
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server
            .run_chat_completion(ChatCompletionArgs {
                system: Some("be terse".to_string()),
                temperature: Some(0.5),
                max_tokens: Some(64),
                ..args("openai/gpt-5.4", "say hi")
            })
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "hello back");
        // A plain completed answer is the single text block it always was.
        assert_eq!(v["content"].as_array().unwrap().len(), 1, "{v}");

        // The text generation and its cost were recorded.
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 1);
    }

    /// The `reasoning` object is sent only when the caller asks for an effort.
    /// Omitting it is what the official OpenRouter SDK does: the model then
    /// applies its own catalog `default_effort`, which is NOT the same as off.
    ///
    /// The request body is captured and inspected rather than matched with
    /// `body_partial_json`, because a partial match cannot prove a key is
    /// *absent* - the default case is exactly the absence of `reasoning`.
    #[tokio::test]
    async fn reasoning_effort_is_forwarded_and_omitted_by_default() {
        for effort in [None, Some("none"), Some("high")] {
            let body = sent_body(ChatCompletionArgs {
                reasoning_effort: effort.map(str::to_string),
                ..args("openai/gpt-5.6-sol", "hi")
            })
            .await;
            match effort {
                None => assert!(body.get("reasoning").is_none(), "sent: {body}"),
                Some(e) => assert_eq!(body["reasoning"]["effort"], e),
            }
        }
    }

    /// The budget form: `max_tokens` + `exclude` go out under `reasoning`, with
    /// no `effort` key; effort and a budget together are refused before any call.
    #[tokio::test]
    async fn reasoning_budget_and_exclude_are_forwarded_and_effort_conflict_is_rejected() {
        let body = sent_body(ChatCompletionArgs {
            reasoning_max_tokens: Some(2000),
            reasoning_exclude: Some(true),
            ..args("anthropic/claude-sonnet-4.6", "hi")
        })
        .await;
        assert_eq!(
            body["reasoning"],
            serde_json::json!({"max_tokens": 2000, "exclude": true}),
            "sent: {body}"
        );

        let server = server_for("http://127.0.0.1:9".to_string());
        let err = server
            .run_chat_completion(ChatCompletionArgs {
                reasoning_effort: Some("high".to_string()),
                reasoning_max_tokens: Some(2000),
                ..args("m", "hi")
            })
            .await
            .unwrap_err();
        assert!(
            err.message.contains("reasoning_effort")
                && err.message.contains("reasoning_max_tokens"),
            "got: {}",
            err.message
        );
    }

    /// Every sampling knob reaches the wire under its documented name; `stop`
    /// is omitted when empty (partial match cannot prove absence, so the body
    /// is inspected for that half).
    #[tokio::test]
    async fn chat_completion_forwards_sampling_controls() {
        let mock = mock_chat(serde_json::json!({
            "seed": 42,
            "top_p": 0.9,
            "top_k": 40,
            "stop": ["END", "STOP"],
            "frequency_penalty": 0.5,
            "presence_penalty": -0.25,
            "verbosity": "low"
        }))
        .await;
        server_for(mock.uri())
            .run_chat_completion(ChatCompletionArgs {
                seed: Some(42),
                top_p: Some(0.9),
                top_k: Some(40),
                stop: vec!["END".to_string(), "STOP".to_string()],
                frequency_penalty: Some(0.5),
                presence_penalty: Some(-0.25),
                verbosity: Some("low".to_string()),
                ..args("m", "hi")
            })
            .await
            .unwrap();

        let body = sent_body(args("m", "hi")).await;
        for key in [
            "seed",
            "top_p",
            "top_k",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "verbosity",
            "response_format",
            "plugins",
            "web_search_options",
            "provider",
        ] {
            assert!(body.get(key).is_none(), "{key} leaked into: {body}");
        }
    }

    /// Routing goes out as `provider` with the exact keys OpenRouter documents
    /// for chat; the block arrives the way a client sends it (lenient path).
    #[tokio::test]
    async fn chat_completion_forwards_provider_routing() {
        let mock = mock_chat(serde_json::json!({
            "provider": {"order": ["anthropic", "google-vertex"], "allow_fallbacks": false}
        }))
        .await;
        let args: ChatCompletionArgs = serde_json::from_value(serde_json::json!({
            "model": "anthropic/claude-sonnet-4.6",
            "prompt": "hi",
            "provider": {"order": ["anthropic", "google-vertex"], "allow_fallbacks": "false"}
        }))
        .unwrap();
        server_for(mock.uri())
            .run_chat_completion(args)
            .await
            .unwrap();

        // An invalid block is rejected before any HTTP call.
        let bad: ChatCompletionArgs = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "hi", "provider": {"sort": "cheapest"}
        }))
        .unwrap();
        let err = server_for("http://127.0.0.1:9".to_string())
            .run_chat_completion(bad)
            .await
            .unwrap_err();
        assert!(err.message.contains("sort"), "got: {}", err.message);
    }

    #[tokio::test]
    async fn json_schema_and_json_mode_set_response_format_and_are_exclusive() {
        // A schema without a title is named "response" and is strict.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        });
        let mock = mock_chat(serde_json::json!({
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": true, "schema": schema}
            }
        }))
        .await;
        let args_json = serde_json::json!({"model": "m", "prompt": "hi", "json_schema": schema});
        server_for(mock.uri())
            .run_chat_completion(serde_json::from_value(args_json).unwrap())
            .await
            .unwrap();

        // A top-level title becomes the name.
        let body = sent_body(
            serde_json::from_value(serde_json::json!({
                "model": "m", "prompt": "hi",
                "json_schema": {"title": "Answer", "type": "object"}
            }))
            .unwrap(),
        )
        .await;
        assert_eq!(body["response_format"]["json_schema"]["name"], "Answer");

        // json_mode alone is the json_object form.
        let body = sent_body(ChatCompletionArgs {
            json_mode: Some(true),
            ..args("m", "hi")
        })
        .await;
        assert_eq!(
            body["response_format"],
            serde_json::json!({"type": "json_object"})
        );
        // json_mode: false sends nothing.
        let body = sent_body(ChatCompletionArgs {
            json_mode: Some(false),
            ..args("m", "hi")
        })
        .await;
        assert!(body.get("response_format").is_none(), "sent: {body}");

        // Both together are a contradiction, refused before any call.
        let err = server_for("http://127.0.0.1:9".to_string())
            .run_chat_completion(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "prompt": "hi", "json_mode": true,
                    "json_schema": {"type": "object"}
                }))
                .unwrap(),
            )
            .await
            .unwrap_err();
        assert!(
            err.message.contains("json_mode") && err.message.contains("json_schema"),
            "got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn web_search_attaches_the_web_plugin_and_search_options() {
        let mock = mock_chat(serde_json::json!({
            "plugins": [{
                "id": "web",
                "engine": "exa",
                "max_results": 3,
                "include_domains": ["example.com"]
            }],
            "web_search_options": {"search_context_size": "high"}
        }))
        .await;
        server_for(mock.uri())
            .run_chat_completion(
                serde_json::from_value(serde_json::json!({
                    "model": "m", "prompt": "hi",
                    "web_search": {
                        "engine": "exa", "max_results": "3",
                        "include_domains": ["example.com", " "],
                        "search_context_size": "high"
                    }
                }))
                .unwrap(),
            )
            .await
            .unwrap();

        // `enabled: true` alone is the bare plugin, with no search options.
        let body = sent_body(ChatCompletionArgs {
            web_search: WebSearchArgs {
                enabled: Some(true),
                ..Default::default()
            },
            ..args("m", "hi")
        })
        .await;
        assert_eq!(body["plugins"], serde_json::json!([{"id": "web"}]));
        assert!(body.get("web_search_options").is_none(), "sent: {body}");

        // `enabled: false` with knobs set is a contradiction.
        let err = server_for("http://127.0.0.1:9".to_string())
            .run_chat_completion(ChatCompletionArgs {
                web_search: WebSearchArgs {
                    enabled: Some(false),
                    engine: Some("exa".to_string()),
                    ..Default::default()
                },
                ..args("m", "hi")
            })
            .await
            .unwrap_err();
        assert!(err.message.contains("enabled"), "got: {}", err.message);
    }

    #[tokio::test]
    async fn pdf_engine_attaches_the_file_parser_plugin_and_is_validated() {
        let mock = mock_chat(serde_json::json!({
            "plugins": [{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}]
        }))
        .await;
        server_for(mock.uri())
            .run_chat_completion(ChatCompletionArgs {
                pdf_engine: Some(" Mistral-OCR ".to_string()),
                ..args("m", "hi")
            })
            .await
            .unwrap();

        // Both plugins together, web first.
        let body = sent_body(ChatCompletionArgs {
            pdf_engine: Some("native".to_string()),
            web_search: WebSearchArgs {
                enabled: Some(true),
                ..Default::default()
            },
            ..args("m", "hi")
        })
        .await;
        assert_eq!(
            body["plugins"],
            serde_json::json!([{"id": "web"}, {"id": "file-parser", "pdf": {"engine": "native"}}])
        );

        let err = server_for("http://127.0.0.1:9".to_string())
            .run_chat_completion(ChatCompletionArgs {
                pdf_engine: Some("tesseract".to_string()),
                ..args("m", "hi")
            })
            .await
            .unwrap_err();
        assert!(
            err.message.contains("pdf_engine") && err.message.contains("mistral-ocr"),
            "got: {}",
            err.message
        );
    }

    /// Reasoning text, web citations, a truncation finish_reason and the token
    /// counts come back in a second JSON block; the reply stays block 0.
    #[tokio::test]
    async fn chat_completion_surfaces_reasoning_annotations_and_truncation() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).insert_header("x-generation-id", "gen-c1").set_body_json(serde_json::json!({
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "content": "The answer",
                        "reasoning": "Let me think.",
                        "annotations": [
                            {"type": "url_citation", "url_citation": {
                                "url": "https://example.com/a", "title": "A", "content": "snippet",
                                "start_index": 0, "end_index": 3, "extra": "dropped"
                            }},
                            {"type": "file", "file": {"hash": "abc"}}
                        ]
                    }
                }],
                "usage": {"cost": 0.01, "prompt_tokens": 12, "completion_tokens": 34}
            })))
            .mount(&mock)
            .await;
        let res = server_for(mock.uri())
            .run_chat_completion(args("m", "hi"))
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "The answer");
        let meta: serde_json::Value =
            serde_json::from_str(v["content"][1]["text"].as_str().unwrap()).unwrap();
        assert_eq!(meta["generation_id"], "gen-c1");
        assert_eq!(meta["reasoning"], "Let me think.");
        assert_eq!(meta["finish_reason"], "length");
        assert_eq!(meta["usage"]["prompt_tokens"], 12);
        assert_eq!(meta["usage"]["completion_tokens"], 34);
        assert_eq!(
            meta["annotations"][0],
            serde_json::json!({
                "type": "url_citation", "url": "https://example.com/a", "title": "A",
                "content": "snippet", "start_index": 0, "end_index": 3
            })
        );
        // Non-citation annotations pass through raw.
        assert_eq!(meta["annotations"][1]["file"]["hash"], "abc");
    }

    /// A plain completed reply that carries a generation id still gets the
    /// second block, holding just that id: it is what `get_generation` takes
    /// to close the cost loop, so it cannot be dropped when nothing else is
    /// noteworthy.
    #[tokio::test]
    async fn chat_completion_reports_the_generation_id_even_for_a_plain_reply() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-plain")
                    .set_body_json(serde_json::json!({
                        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
                    })),
            )
            .mount(&mock)
            .await;
        let res = server_for(mock.uri())
            .run_chat_completion(args("m", "hi"))
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "ok");
        let meta: serde_json::Value =
            serde_json::from_str(v["content"][1]["text"].as_str().unwrap()).unwrap();
        assert_eq!(
            meta,
            serde_json::json!({"generation_id": "gen-plain", "finish_reason": "stop"})
        );
    }

    /// chat_completion writes nothing (like describe_image/transcribe_audio), so
    /// its tools/list annotation must say so (S11).
    #[test]
    fn chat_completion_is_annotated_read_only() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let tool = server
            .tool_router
            .list_all()
            .into_iter()
            .find(|t| t.name == "chat_completion")
            .expect("chat_completion is registered");
        assert_eq!(tool.annotations.and_then(|a| a.read_only_hint), Some(true));
    }

    #[tokio::test]
    async fn chat_completion_requires_prompt() {
        // Validation runs before any HTTP call.
        let server = server_for("http://127.0.0.1:9".to_string());
        // blank-after-trim counts as missing
        let err = server
            .run_chat_completion(args("m", "   "))
            .await
            .unwrap_err();
        assert!(err.message.contains("prompt"));
        assert!(err.message.contains("no defaults"));
    }

    #[tokio::test]
    async fn chat_completion_sends_image_to_vision_model() {
        let mock = MockServer::start().await;
        // The model declares image input, so the request is allowed through...
        mock_model_modalities(&mock, "google/gemini-2.5-flash", &["text", "image"]).await;
        // ...and the user message is sent as a text-part + image-part array.
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text"},
                        {"type": "image_url"}
                    ]}
                ]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "a tiny blue square"}}],
                "usage": {"cost": 0.001}
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server
            .run_chat_completion(ChatCompletionArgs {
                images: one_image(),
                ..args("google/gemini-2.5-flash", "what is this?")
            })
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "a tiny blue square");
    }

    #[tokio::test]
    async fn chat_completion_rejects_image_for_text_only_model() {
        let mock = MockServer::start().await;
        // Only `/models` is mocked: a text-only model must be rejected BEFORE any
        // call to `/chat/completions` (which has no mock and would 404).
        mock_model_modalities(&mock, "openai/gpt-5.4", &["text"]).await;

        let server = server_for(mock.uri());
        let err = server
            .run_chat_completion(ChatCompletionArgs {
                images: one_image(),
                ..args("openai/gpt-5.4", "what is this?")
            })
            .await
            .unwrap_err();
        assert!(err.message.contains("does not accept image input"));
        // No generation should have been recorded (rejected before the API call).
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 0);
    }

    #[tokio::test]
    async fn chat_completion_allows_image_when_modalities_unknown() {
        // The catalog entry matches by id but reports no architecture/modalities.
        // The gate must fail open (treat "unknown" as allowed), not reject with [].
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "obscure/vision-model" }]
            })))
            .mount(&mock)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "looks like an owl"}}]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server
            .run_chat_completion(ChatCompletionArgs {
                images: one_image(),
                ..args("obscure/vision-model", "what is this?")
            })
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "looks like an owl");
    }

    #[tokio::test]
    async fn chat_completion_allows_image_when_capability_lookup_misses() {
        // A routing-suffixed / search-missed id isn't surfaced by the `q` search,
        // so model_input_modalities errors. The gate must fail open rather than
        // hard-reject a call the real /chat/completions would accept.
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "id": "some/other-model", "architecture": {"input_modalities": ["text"]} }]
            })))
            .mount(&mock)
            .await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "described"}}]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server
            .run_chat_completion(ChatCompletionArgs {
                images: one_image(),
                ..args("google/gemini-2.5-flash:nitro", "what is this?")
            })
            .await
            .unwrap();
        let v = serde_json::to_value(&res).unwrap();
        assert_eq!(v["content"][0]["text"], "described");
    }

    /// Mock `GET /models` so `id` reports every input modality.
    async fn mock_omni_model(mock: &MockServer, id: &str) {
        mock_model_modalities(mock, id, &["text", "image", "file", "audio", "video"]).await;
    }

    /// Every multimodal kind reaches the wire as its documented part, in the
    /// fixed order text, images, files, audio, videos; local bytes become data
    /// URLs with the right MIME, a video URL passes through untouched.
    #[tokio::test]
    async fn chat_completion_sends_file_audio_and_video_parts_in_order() {
        let mock = MockServer::start().await;
        mock_omni_model(&mock, "google/gemini-2.5-pro").await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "summarize"},
                    {"type": "image_url"},
                    {"type": "file", "file": {"filename": "doc.pdf"}},
                    {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "mp3"}},
                    {"type": "video_url", "video_url": {
                        "url": "https://example.com/v.mp4", "processing": "low"}}
                ]}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "done"}}]
            })))
            .mount(&mock)
            .await;
        let pdf_b64 =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, b"%PDF-1.4 fake");
        let args: ChatCompletionArgs = serde_json::from_value(serde_json::json!({
            "model": "google/gemini-2.5-pro",
            "prompt": "summarize",
            "images": [{"base64": valid_png_b64()}],
            "files": [{"base64": pdf_b64, "filename": "doc.pdf"}],
            "audio": [{"base64": "data:audio/mp3;base64,QUJD"}],
            "videos": [{"url": "https://example.com/v.mp4", "processing": "low"}]
        }))
        .unwrap();
        let res = server_for(mock.uri())
            .run_chat_completion(args)
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(&res).unwrap()["content"][0]["text"],
            "done"
        );
        let sent: serde_json::Value = mock
            .received_requests()
            .await
            .unwrap()
            .iter()
            .find(|r| r.url.path() == "/chat/completions")
            .unwrap()
            .body_json()
            .unwrap();
        let parts = sent["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts.len(), 5, "{parts:?}");
        assert!(
            parts[2]["file"]["file_data"]
                .as_str()
                .unwrap()
                .starts_with("data:application/pdf;base64,"),
            "{}",
            parts[2]
        );
    }

    /// The capability gate is per kind: a model whose catalog entry lists only
    /// text+image rejects file, audio and video inputs before any chat call,
    /// each naming the modality to look for.
    #[tokio::test]
    async fn chat_completion_gates_each_input_modality() {
        let mock = MockServer::start().await;
        mock_model_modalities(&mock, "openai/gpt-5.4", &["text", "image"]).await;
        let server = server_for(mock.uri());
        let cases = [
            (
                serde_json::json!({"files": [{"base64": "JVBERi0=", "filename": "a.pdf"}]}),
                "file",
            ),
            (
                serde_json::json!({"audio": [{"base64": "QUJD", "format": "mp3"}]}),
                "audio",
            ),
            (
                serde_json::json!({"videos": [{"url": "https://example.com/v.mp4"}]}),
                "video",
            ),
        ];
        for (extra, kind) in cases {
            let mut v = serde_json::json!({"model": "openai/gpt-5.4", "prompt": "hi"});
            for (k, val) in extra.as_object().unwrap() {
                v[k] = val.clone();
            }
            let args: ChatCompletionArgs = serde_json::from_value(v).unwrap();
            let err = server.run_chat_completion(args).await.unwrap_err();
            assert!(
                err.message
                    .contains(&format!("does not accept {kind} input")),
                "{kind}: {}",
                err.message
            );
            assert!(
                err.message.contains(&format!("input_modalities={kind}")),
                "{kind}: {}",
                err.message
            );
        }
        // Nothing reached /chat/completions.
        assert!(
            mock.received_requests()
                .await
                .unwrap()
                .iter()
                .all(|r| r.url.path() == "/models")
        );
        // A malformed entry is reported before the gate runs (no fetch, no catalog).
        let bad: ChatCompletionArgs = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "hi", "files": [{"filename": "a.pdf"}]
        }))
        .unwrap();
        let err = server_for("http://127.0.0.1:9".to_string())
            .run_chat_completion(bad)
            .await
            .unwrap_err();
        assert!(
            err.message.contains("exactly one of"),
            "got: {}",
            err.message
        );
    }

    /// The nested `web_search` and `provider` blocks are optional `$ref`s on
    /// the root, `json_schema` is an open object (never a bare `true`), and
    /// the multimodal lists are arrays of their own `$defs` types.
    #[test]
    fn nested_blocks_are_optional_refs_and_json_schema_is_an_open_object() {
        let schema = crate::server::schema::schema_json::<ChatCompletionArgs>();
        assert_eq!(
            schema["properties"]["web_search"]["$ref"],
            serde_json::json!("#/$defs/WebSearchArgs")
        );
        assert_eq!(
            schema["properties"]["provider"]["$ref"],
            serde_json::json!("#/$defs/ProviderRoutingArgs")
        );
        for (field, def) in [
            ("images", "ImageInput"),
            ("files", "FileInput"),
            ("audio", "AudioInput"),
            ("videos", "VideoInput"),
        ] {
            let prop = &schema["properties"][field];
            assert_eq!(prop["type"], "array", "{field}: {prop}");
            assert_eq!(
                prop["items"]["$ref"],
                serde_json::json!(format!("#/$defs/{def}")),
                "{field}: {prop}"
            );
            assert!(schema["$defs"][def].is_object(), "{def} missing");
        }
        assert_eq!(schema["properties"]["json_schema"]["type"], "object");
        assert_eq!(
            schema["properties"]["json_schema"]["additionalProperties"],
            true
        );
        let required = schema["required"].as_array().cloned().unwrap_or_default();
        for name in ["web_search", "provider", "json_schema", "stop"] {
            assert!(!required.contains(&serde_json::json!(name)), "{required:?}");
        }
        crate::server::schema::assert_client_safe_schema(&schema, "ChatCompletionArgs");
    }
}
