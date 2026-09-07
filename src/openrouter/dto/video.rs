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
    #[serde(default, deserialize_with = "null_pricing")]
    pub pricing_skus: BTreeMap<String, String>,
}

fn null_pricing<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<BTreeMap<String, String>, D::Error> {
    Ok(Option::<BTreeMap<String, String>>::deserialize(deserializer)?.unwrap_or_default())
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

/// The submit ack. Everything else it returns (`polling_url`, `status`) is
/// ignored: we poll by id.
#[derive(Debug, Deserialize)]
pub struct VideoSubmitResponse {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct VideoPollResponse {
    #[serde(default)]
    pub generation_id: Option<String>,
    pub status: String,
    /// Documented as a string; tolerate an object (`{code, message}`) too, since
    /// a decode failure here would lose the terminal status and its receipt.
    #[serde(default, deserialize_with = "lenient_error")]
    pub error: Option<String>,
    #[serde(default)]
    pub unsigned_urls: Vec<String>,
    #[serde(default)]
    pub usage: Option<VideoUsage>,
}

fn lenient_error<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(|v| match v {
        serde_json::Value::Null => None,
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Object(ref map) => Some(
            map.get("message")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| v.to_string()),
        ),
        other => Some(other.to_string()),
    }))
}

#[derive(Debug, Deserialize)]
pub struct VideoUsage {
    #[serde(default)]
    pub cost: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Locks the exact `/videos` image-to-video wire shape documented by OpenRouter:
    // each frame is an OpenAI-style content part with a `type`, an `image_url`
    // OBJECT, and the `frame_type` discriminator. Guards against regressing to a
    // bare-string / wrong-key element (the shape upstream rejects with a ZodError).
    #[test]
    fn frame_image_serializes_to_documented_image_url_part() {
        let fi = FrameImage::new(
            ImageUrl {
                url: "data:image/png;base64,AAAA".to_string(),
            },
            "first_frame".to_string(),
        );
        assert_eq!(
            serde_json::to_value(&fi).unwrap(),
            json!({
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,AAAA" },
                "frame_type": "first_frame"
            })
        );
    }

    // input_references use the same content-part shape, minus `frame_type`.
    #[test]
    fn input_reference_serializes_to_documented_image_url_part() {
        let ir = InputReference::new(ImageUrl {
            url: "https://example.com/ref.png".to_string(),
        });
        assert_eq!(
            serde_json::to_value(&ir).unwrap(),
            json!({
                "type": "image_url",
                "image_url": { "url": "https://example.com/ref.png" }
            })
        );
    }
}

#[cfg(test)]
mod audit_regression {
    use super::*;
    #[test]
    fn catalog_accepts_null_missing_and_populated_pricing_together() {
        let response: VideoModelsResponse = serde_json::from_value(serde_json::json!({"data":[
            {"id":"a","pricing_skus":null}, {"id":"b"}, {"id":"c","pricing_skus":{"duration_seconds":"0.1"}}
        ]})).unwrap();
        assert_eq!(response.data.len(), 3);
        assert!(response.data[0].pricing_skus.is_empty());
        assert!(response.data[1].pricing_skus.is_empty());
        assert_eq!(response.data[2].pricing_skus["duration_seconds"], "0.1");
    }
    #[test]
    fn poll_error_accepts_string_object_and_null() {
        let parse = |error: serde_json::Value| -> VideoPollResponse {
            serde_json::from_value(serde_json::json!({"status": "failed", "error": error})).unwrap()
        };
        assert_eq!(parse("policy".into()).error.as_deref(), Some("policy"));
        assert_eq!(
            parse(serde_json::json!({"code": 500, "message": "provider down"}))
                .error
                .as_deref(),
            Some("provider down")
        );
        assert_eq!(
            parse(serde_json::json!({"code": 500})).error.as_deref(),
            Some(r#"{"code":500}"#)
        );
        assert_eq!(parse(serde_json::Value::Null).error, None);
    }
}
