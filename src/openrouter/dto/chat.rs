//! DTOs for `POST /api/v1/chat/completions` (text/vision/image generation).
//!
//! [`ImageUrl`] is the canonical chat/image reference; it is also reused by the
//! video DTOs (`FrameImage`/`InputReference`) via the flat `dto::*` re-export.

use serde::{Deserialize, Serialize};

use super::provider::ProviderRouting;

/// A chat-completions request. Every optional control is omitted when unset
/// (`None` / empty), so the bare request is exactly `model`, `messages`,
/// `stream`. `stream` is `false` for text/vision calls (one complete result)
/// and `true` for audio output, which OpenRouter only delivers as a stream.
#[derive(Debug, Default, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    /// Output modalities; omitted for plain text-output (vision/describe) calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_config: Option<ImageConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Sampling temperature; omitted when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Max tokens to generate; omitted when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Stop sequences; omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    /// "low" | "medium" | "high" on models that support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    /// Structured-output request (`json_object` or a strict `json_schema`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// OpenRouter plugins (`web`, `file-parser`); omitted when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<Plugin>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    /// Reasoning controls; omitted when `None` so the model keeps its own
    /// `default_effort` from the models catalog.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// Provider routing. Chat completions accept routing fields only (the
    /// schema is closed: no per-provider `options` passthrough here).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderRouting>,
    /// Audio-output options (container format); only meaningful together with
    /// `modalities: ["text", "audio"]`. Omitted when `None` so a provider that
    /// has no such knob (Lyria returns MP3 regardless) never sees it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfig>,
    pub stream: bool,
}

/// The `audio` request block for audio-output chat models. OpenRouter documents
/// `format` values such as `wav`, `mp3`, `flac`, `opus`, `pcm16`; which are
/// honored varies by model.
#[derive(Debug, Serialize)]
pub struct AudioConfig {
    pub format: String,
}

/// The `reasoning` request object. OpenRouter normalizes `effort` per provider
/// (Anthropic `budget_tokens`, Gemini `thinkingLevel`, OpenAI `reasoning_effort`)
/// and documents `effort` and `max_tokens` as mutually exclusive; the tool
/// layer enforces that before the request is built. Every field is omitted
/// when unset. `context`/`mode` (GPT-5.6+ only) are not exposed.
#[derive(Debug, Default, Serialize)]
pub struct Reasoning {
    /// One of: max, xhigh, high, medium, low, minimal, none. Accepted values
    /// vary per model - see `reasoning.supported_efforts` in list_models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// Reasoning token budget, for models that take one instead of an effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Reason internally but leave the reasoning text out of the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// `response_format`: `{"type": "json_object"}` or
/// `{"type": "json_schema", "json_schema": {name, strict, schema}}`. The
/// `text`/`grammar`/`python` forms are not exposed.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaSpec },
}

/// The `json_schema` body of a structured-output request.
#[derive(Debug, Clone, Serialize)]
pub struct JsonSchemaSpec {
    pub name: String,
    pub strict: bool,
    /// The JSON Schema itself, passed through untouched.
    pub schema: serde_json::Value,
}

/// One `plugins[]` entry, discriminated by `id`.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "id")]
pub enum Plugin {
    /// The web-search plugin. Every knob is optional upstream; unset ones are
    /// omitted so `{"id": "web"}` alone means "defaults".
    #[serde(rename = "web")]
    Web {
        #[serde(skip_serializing_if = "Option::is_none")]
        engine: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mode: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_results: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_prompt: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        include_domains: Vec<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        exclude_domains: Vec<String>,
    },
    /// The PDF parser plugin, `pdf.engine`: mistral-ocr | cloudflare-ai | native.
    #[serde(rename = "file-parser")]
    FileParser { pdf: PdfOptions },
}

#[derive(Debug, Clone, Serialize)]
pub struct PdfOptions {
    pub engine: String,
}

/// `web_search_options`: `search_context_size` is low | medium | high.
#[derive(Debug, Clone, Serialize)]
pub struct WebSearchOptions {
    pub search_context_size: String,
}

#[derive(Debug, Serialize)]
pub struct Message {
    pub role: String,
    pub content: Content,
}

/// Message content: either a plain string or an ordered list of parts
/// (text-first, then images) for editing/multi-image requests.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// `image_config` block controlling aspect ratio and resolution tier.
#[derive(Debug, Serialize)]
pub struct ImageConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_size: Option<String>,
}

/// A `{ "url": ... }` image reference, used both in requests (data URLs) and
/// in responses (generated-image data URLs).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletion {
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
    /// OpenAI-normalized: "stop", "length", "content_filter", "tool_calls", "error".
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Assistant message in a text/vision response (`chat_completion`, `describe_image`).
/// Image generation now uses the dedicated `/images` endpoint, so no image field.
#[derive(Debug, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    /// The reasoning text, when the model exposes it (and `exclude` is not set).
    #[serde(default)]
    pub reasoning: Option<String>,
    /// Kept raw: `url_citation` entries from web search (`{type, url_citation:
    /// {url, title, content, start_index, end_index}}`) and `file` entries from
    /// the PDF parser; the tool layer types what it surfaces.
    #[serde(default)]
    pub annotations: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    /// Actual cost in USD reported by OpenRouter, when available.
    #[serde(default)]
    pub cost: Option<f64>,
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
}

/// One `chat.completion.chunk` of a streamed completion. Only the fields the
/// audio path aggregates are typed; everything else is ignored. Fields a
/// provider may send as an explicit `null` are `Option`s: `#[serde(default)]`
/// alone only covers a missing key.
#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    #[serde(default)]
    pub id: Option<String>,
    /// OpenRouter reports a mid-stream failure as a chunk carrying `error`
    /// (with a `message`) instead of `choices`.
    #[serde(default)]
    pub error: Option<serde_json::Value>,
    #[serde(default)]
    pub choices: Option<Vec<ChunkChoice>>,
    /// Present on the final chunk only, carrying the request's `cost`.
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub delta: Option<Delta>,
}

/// The incremental assistant message. Music models put lyrics or
/// `<instrumental>` in `content` and the audio itself in `audio.data`.
#[derive(Debug, Default, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub audio: Option<DeltaAudio>,
}

/// A streamed audio fragment: raw base64 (no `data:` prefix) plus, for speech
/// models, the words spoken in it.
#[derive(Debug, Deserialize)]
pub struct DeltaAudio {
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub transcript: Option<String>,
}

/// A streamed audio-output completion, aggregated over all chunks. `audio` is
/// every `delta.audio.data` fragment decoded as it arrived (see
/// [`crate::image_io::Base64Assembler`] for why fragments are not concatenated
/// first). `text` gathers `delta.content`, `transcript` gathers
/// `delta.audio.transcript`.
#[derive(Debug, Default)]
pub struct ChatAudioResult {
    pub audio: Vec<u8>,
    pub text: String,
    pub transcript: String,
    /// The completion id (`gen-...`), also sent as `X-Generation-Id`.
    pub generation_id: Option<String>,
    /// From the final chunk's `usage.cost`, when reported.
    pub cost: Option<f64>,
    /// Whether the stream ended with its `[DONE]` sentinel. `false` means the
    /// body closed cleanly before it, so `audio` may be cut short.
    pub complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bare_request() -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            messages: vec![Message {
                role: "user".into(),
                content: Content::Text("hi".into()),
            }],
            ..Default::default()
        }
    }

    /// Serde lock: every new optional control is omitted when unset, so the
    /// default request is byte-for-byte what it was before (no `null`s, no
    /// empty arrays that a strict provider might reject).
    #[test]
    fn chat_request_omits_every_unset_control() {
        let v = serde_json::to_value(bare_request()).unwrap();
        assert_eq!(
            v,
            json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            })
        );
    }

    #[test]
    fn chat_request_serializes_sampling_provider_and_stop() {
        let req = ChatRequest {
            provider: Some(ProviderRouting {
                order: vec!["anthropic".into()],
                ..Default::default()
            }),
            top_p: Some(0.9),
            top_k: Some(40),
            stop: vec!["END".into()],
            frequency_penalty: Some(0.5),
            presence_penalty: Some(-0.5),
            verbosity: Some("low".into()),
            ..bare_request()
        };
        let v = serde_json::to_value(req).unwrap();
        assert_eq!(v["provider"], json!({"order": ["anthropic"]}));
        assert_eq!(v["top_p"], 0.9);
        assert_eq!(v["top_k"], 40);
        assert_eq!(v["stop"], json!(["END"]));
        assert_eq!(v["frequency_penalty"], 0.5);
        assert_eq!(v["presence_penalty"], -0.5);
        assert_eq!(v["verbosity"], "low");
    }

    #[test]
    fn response_format_serializes_json_object_and_json_schema() {
        assert_eq!(
            serde_json::to_value(ResponseFormat::JsonObject).unwrap(),
            json!({"type": "json_object"})
        );
        let schema = ResponseFormat::JsonSchema {
            json_schema: JsonSchemaSpec {
                name: "response".into(),
                strict: true,
                schema: json!({"type": "object", "properties": {"a": {"type": "string"}}}),
            },
        };
        assert_eq!(
            serde_json::to_value(schema).unwrap(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": true,
                    "schema": {"type": "object", "properties": {"a": {"type": "string"}}}
                }
            })
        );
    }

    #[test]
    fn plugins_serialize_with_their_ids_and_omit_unset_fields() {
        let web = Plugin::Web {
            engine: None,
            mode: None,
            max_results: Some(3),
            search_prompt: None,
            include_domains: vec!["example.com".into()],
            exclude_domains: vec![],
        };
        assert_eq!(
            serde_json::to_value(web).unwrap(),
            json!({"id": "web", "max_results": 3, "include_domains": ["example.com"]})
        );
        let bare_web = Plugin::Web {
            engine: None,
            mode: None,
            max_results: None,
            search_prompt: None,
            include_domains: vec![],
            exclude_domains: vec![],
        };
        assert_eq!(
            serde_json::to_value(bare_web).unwrap(),
            json!({"id": "web"})
        );
        let pdf = Plugin::FileParser {
            pdf: PdfOptions {
                engine: "mistral-ocr".into(),
            },
        };
        assert_eq!(
            serde_json::to_value(pdf).unwrap(),
            json!({"id": "file-parser", "pdf": {"engine": "mistral-ocr"}})
        );
        let req = ChatRequest {
            plugins: vec![Plugin::Web {
                engine: Some("exa".into()),
                mode: None,
                max_results: None,
                search_prompt: None,
                include_domains: vec![],
                exclude_domains: vec![],
            }],
            web_search_options: Some(WebSearchOptions {
                search_context_size: "high".into(),
            }),
            ..bare_request()
        };
        let v = serde_json::to_value(req).unwrap();
        assert_eq!(v["plugins"], json!([{"id": "web", "engine": "exa"}]));
        assert_eq!(
            v["web_search_options"],
            json!({"search_context_size": "high"})
        );
    }

    #[test]
    fn reasoning_serializes_only_the_set_fields() {
        let effort = Reasoning {
            effort: Some("high".into()),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(effort).unwrap(),
            json!({"effort": "high"})
        );
        let budget = Reasoning {
            effort: None,
            max_tokens: Some(2000),
            exclude: Some(true),
            enabled: Some(true),
        };
        assert_eq!(
            serde_json::to_value(budget).unwrap(),
            json!({"max_tokens": 2000, "exclude": true, "enabled": true})
        );
    }

    /// The response side keeps reasoning, raw annotations, finish_reason and
    /// token counts; a minimal `{"content": ...}` message still parses.
    #[test]
    fn chat_completion_parses_reasoning_annotations_finish_reason_and_tokens() {
        let full: ChatCompletion = serde_json::from_value(json!({
            "choices": [{
                "finish_reason": "length",
                "message": {
                    "content": "answer",
                    "reasoning": "thinking...",
                    "annotations": [{
                        "type": "url_citation",
                        "url_citation": {"url": "https://x.test", "title": "X", "content": "c",
                                         "start_index": 0, "end_index": 3}
                    }]
                }
            }],
            "usage": {"cost": 0.01, "prompt_tokens": 12, "completion_tokens": 34}
        }))
        .unwrap();
        let choice = &full.choices[0];
        assert_eq!(choice.finish_reason.as_deref(), Some("length"));
        assert_eq!(choice.message.reasoning.as_deref(), Some("thinking..."));
        assert_eq!(choice.message.annotations.len(), 1);
        assert_eq!(choice.message.annotations[0]["type"], "url_citation");
        let usage = full.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(12));
        assert_eq!(usage.completion_tokens, Some(34));

        let minimal: ChatCompletion =
            serde_json::from_value(json!({"choices": [{"message": {"content": "x"}}]})).unwrap();
        assert!(minimal.choices[0].finish_reason.is_none());
        assert!(minimal.choices[0].message.reasoning.is_none());
        assert!(minimal.choices[0].message.annotations.is_empty());
    }
}
