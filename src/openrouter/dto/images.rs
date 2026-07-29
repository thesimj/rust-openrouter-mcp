//! DTOs for the dedicated image-generation endpoint (`POST /api/v1/images`).
//!
//! Unlike chat-completions image output (which returns a data URL inside
//! `choices[].message.images`), this endpoint returns base64 image bytes
//! directly in `data[].b64_json`. `InputReference` (and its `ImageUrl`) and
//! `Usage` are reused from the sibling DTO modules via the flat re-export.

use serde::{Deserialize, Serialize};

use super::{InputReference, Usage};

/// Request body for `POST /api/v1/images`. Optional fields are omitted when
/// unset. `resolution`/`aspect_ratio` map from the tool's `image_size`/
/// `aspect_ratio`; input images become `input_references`.
#[derive(Debug, Serialize)]
pub struct ImagesRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Images per call. Left `None` (defaults to 1 upstream): variants are a
    /// parallel fan-out of single-image calls, since most models cap `n` at 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub input_references: Vec<InputReference>,
}

/// Response from `POST /api/v1/images`.
#[derive(Debug, Deserialize)]
pub struct ImagesResponse {
    #[serde(default)]
    pub data: Vec<ImageData>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// One generated image: base64 bytes plus an optional MIME. `media_type` is only
/// present for vector outputs (e.g. SVG); when absent the format is a raster one
/// to sniff from the bytes.
#[derive(Debug, Deserialize)]
pub struct ImageData {
    pub b64_json: String,
    #[serde(default)]
    pub media_type: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter::ImageUrl;
    use serde_json::json;

    /// The request omits empty/unset optionals and renders `input_references`
    /// as the documented `image_url` content-part shape.
    #[test]
    fn images_request_serializes_minimally() {
        let req = ImagesRequest {
            model: "openai/gpt-image-2".to_string(),
            prompt: "an owl".to_string(),
            resolution: Some("1K".to_string()),
            aspect_ratio: Some("1:1".to_string()),
            seed: None,
            n: None,
            input_references: vec![InputReference::new(ImageUrl {
                url: "data:image/png;base64,AAAA".to_string(),
            })],
        };
        assert_eq!(
            serde_json::to_value(&req).unwrap(),
            json!({
                "model": "openai/gpt-image-2",
                "prompt": "an owl",
                "resolution": "1K",
                "aspect_ratio": "1:1",
                "input_references": [
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } }
                ]
            })
        );
    }

    /// A raster response has no `media_type`; a vector one does.
    #[test]
    fn images_response_parses_raster_and_vector() {
        let raster: ImagesResponse = serde_json::from_value(json!({
            "created": 1748372400,
            "data": [{ "b64_json": "AAAA" }],
            "usage": { "cost": 0.04 }
        }))
        .unwrap();
        assert_eq!(raster.data[0].b64_json, "AAAA");
        assert_eq!(raster.data[0].media_type, None);
        assert_eq!(raster.usage.and_then(|u| u.cost), Some(0.04));

        let vector: ImagesResponse = serde_json::from_value(json!({
            "data": [{ "b64_json": "BBBB", "media_type": "image/svg+xml" }]
        }))
        .unwrap();
        assert_eq!(vector.data[0].media_type.as_deref(), Some("image/svg+xml"));
    }
}
