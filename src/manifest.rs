//! The sidecar `*.manifest.json` record written alongside generated images.
//!
//! Holds the full request settings, per-input-image normalization metadata, and
//! per-variant output details (including failures), so the lean tool response
//! can stay minimal while the complete record lives on disk.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

/// The complete record for one generation job.
#[derive(Debug, Serialize)]
pub struct Manifest {
    pub endpoint: &'static str,
    pub model: String,
    pub prompt: String,
    /// `inline`, `file`, or `stdin`.
    pub prompt_source: String,
    pub modalities: Vec<String>,
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
    pub base_seed: Option<u64>,
    pub variants_requested: usize,
    pub max_image_dimension: u32,
    pub created_at: String,
    pub input_images: Vec<InputImageMeta>,
    pub variants: Vec<VariantMeta>,
}

/// Normalization metadata for one input image (editing/image-to-image).
#[derive(Debug, Serialize)]
pub struct InputImageMeta {
    pub index: usize,
    pub label: Option<String>,
    pub source: String,
    /// Original MIME when the input was not one of the four raster formats
    /// (currently only `image/svg+xml`); omitted for raster inputs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_mime_type: Option<&'static str>,
    pub original_width: u32,
    pub original_height: u32,
    pub normalized_mime_type: &'static str,
    pub normalized_width: u32,
    pub normalized_height: u32,
    pub normalization_max_side: u32,
}

/// Output details for one variant. Fields are absent on failure (see `error`).
#[derive(Debug, Default, Serialize)]
pub struct VariantMeta {
    pub index: usize,
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    pub requested_aspect_ratio: Option<String>,
    pub requested_image_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_image_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    pub duration_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Serialize `value` as pretty JSON and write it to `path`. Shared by all
/// manifest writers so the serialize+write+error wrapping stays in one place.
fn write_value(path: &Path, value: &impl Serialize) -> Result<()> {
    let json = serde_json::to_string_pretty(value).context("could not serialize manifest")?;
    std::fs::write(path, json)
        .with_context(|| format!("could not write manifest {}", path.display()))?;
    Ok(())
}

/// Write the image manifest as pretty JSON to `path`.
pub fn write(path: &Path, manifest: &Manifest) -> Result<()> {
    write_value(path, manifest)
}

/// The complete record for one video-generation job.
#[derive(Debug, Serialize)]
pub struct VideoManifest {
    pub endpoint: &'static str,
    pub model: String,
    pub prompt: String,
    /// `inline`, `file`, or `stdin`.
    pub prompt_source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub max_image_dimension: u32,
    pub created_at: String,
    pub frame_images: Vec<FrameImageMeta>,
    pub input_references: Vec<String>,
    pub clips: Vec<VideoClipMeta>,
}

/// Normalization metadata for one first/last frame sent as image-to-video input.
#[derive(Debug, Serialize)]
pub struct FrameImageMeta {
    pub index: usize,
    pub frame_type: String,
    pub source: String,
    pub normalized_width: u32,
    pub normalized_height: u32,
}

/// Output details for one generated clip. Fields are absent on failure (`error`).
#[derive(Debug, Default, Serialize)]
pub struct VideoClipMeta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Write the video manifest as pretty JSON to `path`.
pub fn write_video(path: &Path, manifest: &VideoManifest) -> Result<()> {
    write_value(path, manifest)
}

/// The complete record for one text-to-speech job.
#[derive(Debug, Serialize)]
pub struct AudioManifest {
    pub endpoint: &'static str,
    pub model: String,
    pub input: String,
    /// `inline`, `file`, or `stdin`.
    pub input_source: String,
    pub voice: String,
    pub response_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
    pub created_at: String,
    pub output: AudioOutputMeta,
}

/// Output details for the saved audio file (or its `error`).
#[derive(Debug, Default, Serialize)]
pub struct AudioOutputMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Write the audio manifest as pretty JSON to `path`.
pub fn write_audio(path: &Path, manifest: &AudioManifest) -> Result<()> {
    write_value(path, manifest)
}
