//! Image-generation orchestration over the OpenRouter chat-completions API.
//!
//! Phase 1: a single text-to-image request. The returned image format is
//! whatever the provider sends (sniffed, not assumed) and the dimensions are
//! decoded from the actual bytes (the requested `image_size` is only a hint).

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::image_io;
use crate::openrouter::{
    ChatRequest, Content, ContentPart, ImageUrl, ImagesRequest, InputReference, Message,
    OpenRouterClient,
};

pub(crate) mod job;

pub(crate) use job::{JobSummary, base_stem, in_parent_of, manifest_path, run_job};

/// Default and hard ceiling for the input-image longest-side cap (px). Larger
/// explicit arguments or env overrides are clamped down to this, so input images
/// are never sent with a longest side above it (keeps request payloads bounded;
/// providers gain nothing from larger inputs here).
const DEFAULT_MAX_DIMENSION: u32 = 800;

/// Where an input image's bytes come from. URL inputs are fetched and `base64`/
/// data-URL inputs are decoded at the tool boundary, so by the time bytes are
/// needed an input is either a local path or already-decoded inline bytes.
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// A local file path, read lazily in [`prepare_inputs`].
    Path(PathBuf),
    /// Already-decoded bytes (from a URL fetch or a base64/data-URL argument),
    /// with a human-readable `name` for prompts/manifest.
    Inline { bytes: Vec<u8>, name: String },
}

/// An image used as input (editing / image-to-image). Order is preserved.
#[derive(Debug, Clone)]
pub struct InputImage {
    pub source: ImageSource,
    pub label: Option<String>,
}

impl InputImage {
    /// An input backed by a local file path.
    pub fn from_path(path: impl Into<PathBuf>, label: Option<String>) -> Self {
        Self {
            source: ImageSource::Path(path.into()),
            label,
        }
    }

    /// An input backed by already-decoded bytes (URL fetch / base64 argument).
    pub fn inline(bytes: Vec<u8>, name: impl Into<String>, label: Option<String>) -> Self {
        Self {
            source: ImageSource::Inline {
                bytes,
                name: name.into(),
            },
            label,
        }
    }

    /// Short name used to reference this image in the prompt (file name, URL, or
    /// inline label).
    pub fn display_name(&self) -> String {
        match &self.source {
            ImageSource::Path(p) => p
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
            ImageSource::Inline { name, .. } => name.clone(),
        }
    }

    /// Source descriptor recorded in the manifest.
    pub fn source_label(&self) -> String {
        match &self.source {
            ImageSource::Path(p) => p.to_string_lossy().into_owned(),
            ImageSource::Inline { name, .. } => name.clone(),
        }
    }
}

/// Inputs for a single image generation.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
    pub seed: Option<u64>,
    /// Local images to edit/condition on. Empty for plain text-to-image.
    pub images: Vec<InputImage>,
    /// Longest-side cap (px) for normalized input images.
    pub max_image_dimension: u32,
}

/// Resolve the input-image dimension cap: explicit value, else the
/// `OPENROUTER_IMAGE_MAX_DIMENSION` env var, else [`DEFAULT_MAX_DIMENSION`]. The
/// result is clamped to `1..=`[`DEFAULT_MAX_DIMENSION`] so an oversized argument
/// or env override can never push input images above the ceiling.
pub fn resolve_max_dimension(explicit: Option<u32>) -> u32 {
    explicit
        .or_else(|| {
            std::env::var("OPENROUTER_IMAGE_MAX_DIMENSION")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(DEFAULT_MAX_DIMENSION)
        .clamp(1, DEFAULT_MAX_DIMENSION)
}

/// A generated image plus the metadata worth recording.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub bytes: Vec<u8>,
    /// MIME type as reported in the response data URL (e.g. `image/png`).
    pub mime: String,
    pub width: u32,
    pub height: u32,
    /// Assistant text, when the model returned any alongside the image.
    pub text: Option<String>,
    /// Actual USD cost from `usage.cost`, when present.
    pub cost: Option<f64>,
    /// OpenRouter generation id, recorded in the manifest.
    pub generation_id: Option<String>,
    /// Provider that served the request (e.g. "Google"), recorded in the manifest.
    pub provider: Option<String>,
}

/// Prepend a labeled reference-image block when any input image has a label, so
/// the model can ground references by order/name (text-first, before images).
fn assemble_prompt(prompt: &str, images: &[InputImage]) -> String {
    if images.iter().all(|i| i.label.is_none()) {
        return prompt.to_string();
    }
    let mut block = String::from("Reference images:\n");
    for (i, img) in images.iter().enumerate() {
        let name = img.display_name();
        let label = img.label.as_deref().unwrap_or("image");
        block.push_str(&format!("{}. {label}: {name}\n", i + 1));
    }
    format!("{block}\nUser prompt:\n{prompt}")
}

/// A normalized input image, computed once and reused across all variant
/// requests and the manifest (avoids re-reading/re-encoding per variant).
pub(crate) struct PreparedInput {
    /// `data:image/png;base64,...` URL sent to the model.
    pub data_url: String,
    pub original_width: u32,
    pub original_height: u32,
    pub normalized_width: u32,
    pub normalized_height: u32,
    /// Source MIME when the input arrived as something other than the four raster
    /// formats (currently only `image/svg+xml`); `None` for raster inputs.
    pub source_mime: Option<&'static str>,
    /// Non-fatal notes about this input (e.g. an SVG containing unrendered text).
    pub warnings: Vec<String>,
}

/// Read each input image **once** and produce the PNG data URL plus original /
/// normalized dimensions. Raster inputs (png/jpeg/webp/gif) are decoded and
/// downscaled to `max_dim`; SVG inputs are rasterized to PNG at the cap (see
/// [`image_io::svg_to_png`]), with the SVG's intrinsic viewBox size recorded as
/// the "original" dimensions.
pub(crate) fn prepare_inputs(images: &[InputImage], max_dim: u32) -> Result<Vec<PreparedInput>> {
    images
        .iter()
        .map(|img| {
            let bytes = match &img.source {
                ImageSource::Path(p) => std::fs::read(p)
                    .with_context(|| format!("could not read input image {}", p.display()))?,
                ImageSource::Inline { bytes, .. } => bytes.clone(),
            };
            if image_io::is_svg(&bytes) {
                let svg = image_io::svg_to_png(&bytes, max_dim)
                    .with_context(|| format!("could not rasterize SVG {}", img.source_label()))?;
                let (normalized_width, normalized_height) = image_io::decode_dimensions(&svg.png)?;
                let mut warnings = Vec::new();
                if svg.has_text {
                    warnings.push(
                        "SVG contains <text> which is not rendered (no fonts are loaded)"
                            .to_string(),
                    );
                }
                Ok(PreparedInput {
                    data_url: image_io::png_data_url(&svg.png),
                    original_width: svg.intrinsic_width,
                    original_height: svg.intrinsic_height,
                    normalized_width,
                    normalized_height,
                    source_mime: Some("image/svg+xml"),
                    warnings,
                })
            } else {
                let (original_width, original_height) = image_io::decode_dimensions(&bytes)?;
                let png = image_io::normalize_to_png(&bytes, max_dim)?;
                let (normalized_width, normalized_height) = image_io::decode_dimensions(&png)?;
                Ok(PreparedInput {
                    data_url: image_io::png_data_url(&png),
                    original_width,
                    original_height,
                    normalized_width,
                    normalized_height,
                    source_mime: None,
                    warnings: Vec::new(),
                })
            }
        })
        .collect()
}

/// Build the message content from already-prepared inputs: a plain string for
/// text-to-image, or a text-first array of parts (text prompt, then each input
/// image data URL) for editing / multi-image requests.
pub(crate) fn build_content(
    prompt: &str,
    images: &[InputImage],
    prepared: &[PreparedInput],
) -> Content {
    if prepared.is_empty() {
        return Content::Text(prompt.to_string());
    }
    let mut parts = vec![ContentPart::Text {
        text: assemble_prompt(prompt, images),
    }];
    for input in prepared {
        parts.push(ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: input.data_url.clone(),
            },
        });
    }
    Content::Parts(parts)
}

/// Pre-built inputs for one generation, computed once and cloned per variant:
/// the assembled prompt and the reference-image data URLs (sent as
/// `input_references`). Mirrors the old pre-built `Content`, adapted to the
/// Images API request shape.
#[derive(Debug, Clone)]
pub(crate) struct GenContent {
    prompt: String,
    reference_urls: Vec<String>,
}

/// Assemble the prompt (with any labeled-reference preamble) and collect the
/// prepared input-image data URLs into a [`GenContent`], once per job.
pub(crate) fn build_gen_content(
    prompt: &str,
    images: &[InputImage],
    prepared: &[PreparedInput],
) -> GenContent {
    GenContent {
        prompt: assemble_prompt(prompt, images),
        reference_urls: prepared.iter().map(|p| p.data_url.clone()).collect(),
    }
}

/// Map the tool's `image_size` tier to the Images API `resolution` value: the
/// half-K tier is spelled `512` upstream; the other tiers pass through as-is.
fn resolution_for(image_size: &str) -> String {
    match image_size.trim() {
        "0.5K" | "0.5k" | "512" => "512".to_string(),
        other => other.to_string(),
    }
}

/// Issue one `POST /api/v1/images` request for the given pre-built `content` and
/// extract the generated image. Shared by single and variant generation so the
/// content (including normalized input images) is built once and reused. Each
/// call asks for a single image (`n` defaults to 1 upstream); variants fan out
/// across multiple calls, since most models cap `n` at 1.
pub(crate) async fn generate_core(
    client: &OpenRouterClient,
    req: &GenerateRequest,
    content: GenContent,
) -> Result<GeneratedImage> {
    let input_references = content
        .reference_urls
        .into_iter()
        .map(|url| InputReference::new(ImageUrl { url }))
        .collect();
    let request = ImagesRequest {
        model: req.model.clone(),
        prompt: content.prompt,
        resolution: req.image_size.as_deref().map(resolution_for),
        aspect_ratio: req.aspect_ratio.clone(),
        seed: req.seed,
        n: None,
        input_references,
    };

    let (resp, generation_id) = client.generate_images(&request).await?;
    let cost = resp.usage.and_then(|u| u.cost);
    let item = resp
        .data
        .into_iter()
        .next()
        .context("model returned no image (it may have refused)")?;

    let bytes = image_io::decode_base64(&item.b64_json)?;
    // Prefer the response-declared MIME (only sent for vector output), else sniff
    // the raster magic bytes, else default to PNG.
    let mime = item
        .media_type
        .or_else(|| image_io::sniff_mime(&bytes).map(str::to_string))
        .unwrap_or_else(|| "image/png".to_string());
    let (width, height) = if mime == "image/svg+xml" {
        image_io::svg_dimensions(&bytes).unwrap_or((0, 0))
    } else {
        image_io::decode_dimensions(&bytes)?
    };

    Ok(GeneratedImage {
        bytes,
        mime,
        width,
        height,
        // The Images API returns image bytes only, no assistant commentary.
        text: None,
        cost,
        generation_id,
        // The Images API response body carries no provider attribution.
        provider: None,
    })
}

/// Inputs for an image-description (vision) request.
#[derive(Debug, Clone)]
pub struct DescribeRequest {
    pub model: String,
    /// Instruction or question about the image(s).
    pub prompt: String,
    pub images: Vec<InputImage>,
    pub max_image_dimension: u32,
}

/// The text a vision model returned, plus its cost when reported.
#[derive(Debug, Clone)]
pub struct DescribeResult {
    pub text: String,
    pub cost: Option<f64>,
}

/// Describe (or answer a question about) one or more images: sends them with an
/// instruction to a vision-capable model and returns its text. Requires at least
/// one input image; this is a plain text-output call (no `modalities`).
pub async fn describe_image(
    client: &OpenRouterClient,
    req: &DescribeRequest,
) -> Result<DescribeResult> {
    if req.images.is_empty() {
        anyhow::bail!("describe_image requires at least one input image");
    }
    let prepared = prepare_inputs(&req.images, req.max_image_dimension)?;
    let content = build_content(&req.prompt, &req.images, &prepared);
    // Description is a plain text-output vision call over /chat/completions (no
    // `modalities`/`image_config`), distinct from the /images generation path.
    let chat = ChatRequest {
        model: req.model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content,
        }],
        modalities: None,
        image_config: None,
        seed: None,
        temperature: None,
        max_tokens: None,
        stream: false,
    };

    let resp = client.chat_completion(&chat).await?;
    let cost = resp.completion.usage.and_then(|u| u.cost);
    let choice = resp
        .completion
        .choices
        .into_iter()
        .next()
        .context("OpenRouter returned no choices")?;
    let text = choice
        .message
        .content
        .filter(|t| !t.is_empty())
        .context("model returned no text (use a vision-capable model with text output)")?;
    Ok(DescribeResult { text, cost })
}

#[cfg(test)]
mod tests;
