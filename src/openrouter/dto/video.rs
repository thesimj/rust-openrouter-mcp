//! DTOs for the asynchronous video-generation endpoints (`/videos`,
//! `/videos/{id}`).
//!
//! `FrameImage`/`InputReference` reuse the canonical [`ImageUrl`] from
//! `dto::chat`, reachable here as `super::ImageUrl` via the flat re-export.

use serde::{Deserialize, Serialize};

use super::ImageUrl;
use super::provider::ProviderOptions;

/// Request body for `POST /api/v1/videos`. Optional fields are omitted when
/// unset (named `*Body` to avoid colliding with the domain `video_gen` struct).
#[derive(Debug, Serialize)]
pub struct VideoSubmitBody {
    pub model: String,
    /// Optional upstream: image-only models (frame or reference in, no text)
    /// take none. The domain layer enforces "prompt or an input".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
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
    /// Upscaling models only: creativity level (integer, model-specific range).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub creativity: Option<u32>,
    /// Upscaling models only: output scale, > 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upscale_factor: Option<f64>,
    /// Per-provider passthrough (`options.<slug>`), sent opaque and unchanged;
    /// routing fields are ignored by this endpoint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderOptions>,
}

/// A `{ "url": ... }` audio or video reference (https or data URL).
#[derive(Debug, Serialize)]
pub struct MediaUrl {
    pub url: String,
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

/// A reference for reference-to-video: an image (data-URL `image_url`), or an
/// audio / video clip by URL (Seedance gen 2+ honors these). Each variant is a
/// content part tagged by `type` and keyed by the same name.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum InputReference {
    #[serde(rename = "image_url")]
    Image { image_url: ImageUrl },
    #[serde(rename = "audio_url")]
    Audio { audio_url: MediaUrl },
    #[serde(rename = "video_url")]
    Video { video_url: MediaUrl },
}

impl InputReference {
    pub fn new(image_url: ImageUrl) -> Self {
        Self::Image { image_url }
    }

    pub fn audio(url: String) -> Self {
        Self::Audio {
            audio_url: MediaUrl { url },
        }
    }

    pub fn video(url: String) -> Self {
        Self::Video {
            video_url: MediaUrl { url },
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
    use std::collections::BTreeMap;

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

    // Audio and video references (Seedance gen 2+) carry the same content-part
    // shape under their own type tag and key: `audio_url` / `video_url`.
    #[test]
    fn audio_and_video_references_serialize_to_their_documented_parts() {
        let audio = InputReference::audio("https://example.com/beat.mp3".to_string());
        assert_eq!(
            serde_json::to_value(&audio).unwrap(),
            json!({
                "type": "audio_url",
                "audio_url": { "url": "https://example.com/beat.mp3" }
            })
        );
        let video = InputReference::video("data:video/mp4;base64,AAAA".to_string());
        assert_eq!(
            serde_json::to_value(&video).unwrap(),
            json!({
                "type": "video_url",
                "video_url": { "url": "data:video/mp4;base64,AAAA" }
            })
        );
    }

    fn minimal_body() -> VideoSubmitBody {
        VideoSubmitBody {
            model: "m".to_string(),
            prompt: None,
            duration: None,
            resolution: None,
            aspect_ratio: None,
            size: None,
            frame_images: vec![],
            input_references: vec![],
            generate_audio: None,
            seed: None,
            creativity: None,
            upscale_factor: None,
            provider: None,
        }
    }

    /// Serde lock: `prompt` (optional for image-only models), `creativity`,
    /// `upscale_factor` and `provider` are omitted entirely when unset - a bare
    /// `null` is not what the endpoint documents - and sent verbatim when set.
    #[test]
    fn submit_body_omits_optional_fields_and_sends_them_verbatim_when_set() {
        let none = serde_json::to_value(minimal_body()).unwrap();
        for key in ["prompt", "creativity", "upscale_factor", "provider"] {
            assert!(none.get(key).is_none(), "{key} sent: {none}");
        }

        let mut options = BTreeMap::new();
        options.insert(
            "google-vertex".to_string(),
            json!({"negativePrompt": "blurry"}),
        );
        let some = serde_json::to_value(VideoSubmitBody {
            prompt: Some("a kite".to_string()),
            creativity: Some(3),
            upscale_factor: Some(2.0),
            provider: Some(ProviderOptions { options }),
            ..minimal_body()
        })
        .unwrap();
        assert_eq!(some["prompt"], "a kite");
        assert_eq!(some["creativity"], 3);
        assert_eq!(some["upscale_factor"], 2.0);
        // The options block is opaque: sent unchanged, no extra nesting added.
        assert_eq!(
            some["provider"],
            json!({"options": {"google-vertex": {"negativePrompt": "blurry"}}})
        );
    }
}

#[cfg(test)]
mod audit_regression {
    use super::*;
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
