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
    pub stream: bool,
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
pub struct ChatResponse {
    pub completion: ChatCompletion,
    pub generation_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletion {
    #[serde(default)]
    pub id: Option<String>,
    // `model`/`finish_reason` are not yet surfaced; `provider` feeds the manifest.
    #[allow(dead_code)]
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
    #[allow(dead_code)]
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Assistant message in a response. `content` is null for image-only output;
/// generated images arrive in `images`.
#[derive(Debug, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub images: Vec<OutImage>,
}

#[derive(Debug, Deserialize)]
pub struct OutImage {
    pub image_url: ImageUrl,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    /// Actual cost in USD reported by OpenRouter, when available.
    #[serde(default)]
    pub cost: Option<f64>,
}
