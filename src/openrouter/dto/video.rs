//! DTOs for the asynchronous video-generation endpoints (`/videos`,
//! `/videos/models`, `/videos/{id}`).
//!
//! `FrameImage`/`InputReference` reuse the canonical [`ImageUrl`] from
//! `dto::chat`, reachable here as `super::ImageUrl` via the flat re-export.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::ImageUrl;

#[derive(Debug, Deserialize)]
pub struct VideoModelsResponse {
    pub data: Vec<VideoModel>,
}

/// A video-generation model from `/videos/models`. `pricing_skus` maps a SKU
/// name (e.g. `duration_seconds_with_audio`, `video_tokens`) to a price string.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VideoModel {
    pub id: String,
    #[serde(default)]
    pub pricing_skus: BTreeMap<String, String>,
}

/// Request body for `POST /api/v1/videos`. Optional fields are omitted when
/// unset (named `*Body` to avoid colliding with the domain `video_gen` struct).
#[derive(Debug, Serialize)]
pub struct VideoSubmitBody {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub frame_images: Vec<FrameImage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub input_references: Vec<InputReference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// A first/last frame for image-to-video (`frame_type` is `first_frame` or
/// `last_frame`), sent as a data-URL `image_url`.
#[derive(Debug, Serialize)]
pub struct FrameImage {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub image_url: ImageUrl,
    pub frame_type: String,
}

impl FrameImage {
    pub fn new(image_url: ImageUrl, frame_type: String) -> Self {
        Self {
            kind: "image_url",
            image_url,
            frame_type,
        }
    }
}

/// A reference image for reference-to-video, sent as a data-URL `image_url`.
#[derive(Debug, Serialize)]
pub struct InputReference {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub image_url: ImageUrl,
}

impl InputReference {
    pub fn new(image_url: ImageUrl) -> Self {
        Self {
            kind: "image_url",
            image_url,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct VideoSubmitResponse {
    pub id: String,
    // Captured for completeness; we poll by id rather than following these.
    #[allow(dead_code)]
    #[serde(default)]
    pub polling_url: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct VideoPollResponse {
    #[allow(dead_code)]
    pub id: String,
    #[serde(default)]
    pub generation_id: Option<String>,
    pub status: String,
    #[serde(default)]
    pub unsigned_urls: Vec<String>,
    #[serde(default)]
    pub usage: Option<VideoUsage>,
}

#[derive(Debug, Deserialize)]
pub struct VideoUsage {
    #[serde(default)]
    pub cost: Option<f64>,
    // Parsed defensively; not surfaced today.
    #[allow(dead_code)]
    #[serde(default)]
    pub is_byok: Option<bool>,
}
