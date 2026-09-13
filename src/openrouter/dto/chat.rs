//! DTOs for `POST /api/v1/chat/completions` (text/vision/image generation).
//!
//! [`ImageUrl`] is the canonical chat/image reference; it is also reused by the
//! video DTOs (`FrameImage`/`InputReference`) via the flat `dto::*` re-export.

use serde::{Deserialize, Serialize};

/// A chat-completions request. `image_config`/`seed`/`audio` are omitted when
/// `None`. `stream` is `false` for text/vision calls (one complete result) and
/// `true` for audio output, which OpenRouter only delivers as a stream.
#[derive(Debug, Serialize)]
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
    /// Reasoning controls; omitted when `None` so the model keeps its own
    /// `default_effort` from the models catalog.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
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
/// (Anthropic `budget_tokens`, Gemini `thinkingLevel`, OpenAI `reasoning_effort`).
/// Only `effort` is exposed: the per-provider token budgets it derives are the
/// documented behavior, and a raw `max_tokens` needs different bounds per family.
#[derive(Debug, Serialize)]
pub struct Reasoning {
    /// One of: max, xhigh, high, medium, low, minimal, none. Accepted values
    /// vary per model - see `reasoning.supported_efforts` in list_models.
    pub effort: String,
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
}

/// Assistant message in a text/vision response (`chat_completion`, `describe_image`).
/// Image generation now uses the dedicated `/images` endpoint, so no image field.
#[derive(Debug, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    /// Actual cost in USD reported by OpenRouter, when available.
    #[serde(default)]
    pub cost: Option<f64>,
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
