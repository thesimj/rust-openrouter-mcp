//! DTOs for `POST /api/v1/chat/completions` (text/vision/image generation).
//!
//! [`ImageUrl`] is the canonical chat/image reference; it is also reused by the
//! video DTOs (`FrameImage`/`InputReference`) via the flat `dto::*` re-export.

use serde::{Deserialize, Serialize};

/// A chat-completions request. `image_config`/`seed` are omitted when `None`.
/// `stream` is always sent as `false` (MCP tools return one complete result).
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
    pub stream: bool,
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
