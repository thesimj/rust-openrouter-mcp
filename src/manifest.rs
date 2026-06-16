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

/// Write the manifest as pretty JSON to `path`.
pub fn write(path: &Path, manifest: &Manifest) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest).context("could not serialize manifest")?;
    std::fs::write(path, json)
        .with_context(|| format!("could not write manifest {}", path.display()))?;
    Ok(())
}
