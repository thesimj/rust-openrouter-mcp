//! Image-generation orchestration over the OpenRouter chat-completions API.
//!
//! Phase 1: a single text-to-image request. The returned image format is
//! whatever the provider sends (sniffed, not assumed) and the dimensions are
//! decoded from the actual bytes (the requested `image_size` is only a hint).

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::image_io;
use crate::openrouter::{
    ChatRequest, Content, ContentPart, ImageConfig, ImageUrl, Message, OpenRouterClient,
};

pub(crate) mod job;

pub(crate) use job::{JobSummary, base_stem, in_parent_of, manifest_path, run_job};

/// Default longest-side cap (px) for normalized input images.
const DEFAULT_MAX_DIMENSION: u32 = 800;

/// A local image used as input (editing / image-to-image). Order is preserved.
#[derive(Debug, Clone)]
pub struct InputImage {
    pub path: PathBuf,
    pub label: Option<String>,
}

/// Inputs for a single image generation.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
    pub seed: Option<u64>,
    /// Image-only-output models (e.g. Grok/Seedream/FLUX) need `["image"]`;
    /// dual-output models (Nano Banana, GPT Image) use `["image","text"]`.
    pub image_only: bool,
    /// Local images to edit/condition on. Empty for plain text-to-image.
    pub images: Vec<InputImage>,
    /// Longest-side cap (px) for normalized input images.
    pub max_image_dimension: u32,
}

/// Resolve the input-image dimension cap: explicit value, else the
/// `OPENROUTER_IMAGE_MAX_DIMENSION` env var, else [`DEFAULT_MAX_DIMENSION`].
pub fn resolve_max_dimension(explicit: Option<u32>) -> u32 {
    explicit
        .or_else(|| {
            std::env::var("OPENROUTER_IMAGE_MAX_DIMENSION")
                .ok()
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(DEFAULT_MAX_DIMENSION)
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
        let name = img
            .path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
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
            let bytes = std::fs::read(&img.path)
                .with_context(|| format!("could not read input image {}", img.path.display()))?;
            if image_io::is_svg(&bytes) {
                let svg = image_io::svg_to_png(&bytes, max_dim)
                    .with_context(|| format!("could not rasterize SVG {}", img.path.display()))?;
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

/// Build a single-user-message chat request for the given content. Shared by the
/// image-generation and image-description paths so the request envelope is built
/// in one place.
fn user_chat(
    model: &str,
    content: Content,
    modalities: Option<Vec<String>>,
    image_config: Option<ImageConfig>,
    seed: Option<u64>,
) -> ChatRequest {
    ChatRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content,
        }],
        modalities,
        image_config,
        seed,
        stream: false,
    }
}

/// Issue one chat-completions request for the given pre-built `content` and
/// extract the generated image. Shared by single and variant generation so the
/// content (including normalized input images) is built once and reused.
pub(crate) async fn generate_core(
    client: &OpenRouterClient,
    req: &GenerateRequest,
    content: Content,
) -> Result<GeneratedImage> {
    let image_config =
        (req.aspect_ratio.is_some() || req.image_size.is_some()).then(|| ImageConfig {
            aspect_ratio: req.aspect_ratio.clone(),
            image_size: req.image_size.clone(),
        });
    let chat = user_chat(
        &req.model,
        content,
        Some(modalities_for(req.image_only)),
        image_config,
        req.seed,
    );

    let resp = client.chat_completion(&chat).await?;
    let generation_id = resp.generation_id.or(resp.completion.id);
    let provider = resp.completion.provider;
    let cost = resp.completion.usage.and_then(|u| u.cost);

    let choice = resp
        .completion
        .choices
        .into_iter()
        .next()
        .context("OpenRouter returned no choices")?;
    let text = choice.message.content.filter(|t| !t.is_empty());
    let image = choice
        .message
        .images
        .into_iter()
        .next()
        .context("model returned no image (it may be a vision-only model or have refused)")?;

    let (mime, bytes) = image_io::parse_data_url(&image.image_url.url)?;
    let (width, height) = image_io::decode_dimensions(&bytes)?;

    Ok(GeneratedImage {
        bytes,
        mime,
        width,
        height,
        text,
        cost,
        generation_id,
        provider,
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
    let chat = user_chat(&req.model, content, None, None, None);

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

/// Output modalities for the request: image-only models get `["image"]`,
/// dual-output models get `["image","text"]`.
pub(crate) fn modalities_for(image_only: bool) -> Vec<String> {
    if image_only {
        vec!["image".to_string()]
    } else {
        vec!["image".to_string(), "text".to_string()]
    }
}

#[cfg(test)]
mod tests;
