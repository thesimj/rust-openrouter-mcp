//! Image-generation orchestration over the OpenRouter chat-completions API.
//!
//! Phase 1: a single text-to-image request. The returned image format is
//! whatever the provider sends (sniffed, not assumed) and the dimensions are
//! decoded from the actual bytes (the requested `image_size` is only a hint).

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::image_io;
use crate::manifest::{self, InputImageMeta, Manifest, VariantMeta};
use crate::openrouter::{
    ChatRequest, Content, ContentPart, ImageConfig, ImageUrl, Message, OpenRouterClient,
};

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

/// Issue one chat-completions request for the given pre-built `content` and
/// extract the generated image. Shared by single and variant generation so the
/// content (including normalized input images) is built once and reused.
async fn generate_core(
    client: &OpenRouterClient,
    req: &GenerateRequest,
    content: Content,
) -> Result<GeneratedImage> {
    let image_config = match (&req.aspect_ratio, &req.image_size) {
        (None, None) => None,
        (a, s) => Some(ImageConfig {
            aspect_ratio: a.clone(),
            image_size: s.clone(),
        }),
    };
    let chat = ChatRequest {
        model: req.model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content,
        }],
        modalities: Some(modalities_for(req.image_only)),
        image_config,
        seed: req.seed,
        stream: false,
    };

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
    let chat = ChatRequest {
        model: req.model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content,
        }],
        modalities: None,
        image_config: None,
        seed: None,
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

/// Output modalities for the request: image-only models get `["image"]`,
/// dual-output models get `["image","text"]`.
fn modalities_for(image_only: bool) -> Vec<String> {
    if image_only {
        vec!["image".to_string()]
    } else {
        vec!["image".to_string(), "text".to_string()]
    }
}

/// Outcome of one variant generation (an image, or a per-variant error).
pub struct VariantOutcome {
    pub index: usize,
    pub seed: Option<u64>,
    pub duration_ms: u128,
    pub result: Result<GeneratedImage>,
}

/// Generate `variants` images concurrently (all at once). With a base seed each
/// variant uses `base + index` for reproducible, distinct outputs. A failed
/// variant is captured in its `result` without aborting the others. Returned
/// ordered by index.
pub async fn generate_variants(
    client: &OpenRouterClient,
    req: &GenerateRequest,
    variants: usize,
    content: Content,
) -> Vec<VariantOutcome> {
    let base_seed = req.seed;
    // saturating_add: a base seed near u64::MAX must not overflow-panic.
    let seed_for = |i: usize| base_seed.map(|s| s.saturating_add(i as u64));
    let handles: Vec<_> = (0..variants)
        .map(|i| {
            let client = client.clone();
            let mut r = req.clone();
            let seed = seed_for(i);
            r.seed = seed;
            let content = content.clone();
            tokio::spawn(async move {
                let start = Instant::now();
                let result = generate_core(&client, &r, content).await;
                (seed, start.elapsed().as_millis(), result)
            })
        })
        .collect();

    let mut outcomes = Vec::with_capacity(variants);
    for (i, handle) in handles.into_iter().enumerate() {
        let outcome = match handle.await {
            Ok((seed, duration_ms, result)) => VariantOutcome {
                index: i,
                seed,
                duration_ms,
                result,
            },
            Err(e) => VariantOutcome {
                index: i,
                seed: seed_for(i),
                duration_ms: 0,
                result: Err(anyhow::anyhow!("variant task failed: {e}")),
            },
        };
        outcomes.push(outcome);
    }
    outcomes
}

/// Output path for one variant. A single variant uses `base` with the given
/// extension. Multiple variants get a `-var-<seed>` suffix (seed zero-padded to
/// at least 4 digits, so it is self-identifying and reproducible); when no seed
/// is set the variant `index` is used instead, zero-padded so 10+ sort.
/// The file stem of `base`, or `"image"` if it has none.
pub(crate) fn base_stem(base: &Path) -> String {
    base.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_string())
}

/// Join `name` onto `base`'s parent directory (or use it bare when there is none).
pub(crate) fn in_parent_of(base: &Path, name: String) -> PathBuf {
    match base.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.join(name),
        _ => PathBuf::from(name),
    }
}

pub fn variant_output_path(
    base: &Path,
    seed: Option<u64>,
    index_zero_based: usize,
    total: usize,
    ext: &str,
) -> PathBuf {
    if total <= 1 {
        return base.with_extension(ext);
    }
    let marker = match seed {
        Some(s) => format!("{s:04}"),
        None => {
            let width = 3.max(total.to_string().len());
            format!("{:0width$}", index_zero_based + 1, width = width)
        }
    };
    in_parent_of(base, format!("{}-var-{marker}.{ext}", base_stem(base)))
}

/// Sidecar manifest path next to the outputs: `<stem>.manifest.json`.
pub fn manifest_path(base: &Path) -> PathBuf {
    in_parent_of(base, format!("{}.manifest.json", base_stem(base)))
}

/// One saved image in a job's lean summary.
pub struct ImageSummary {
    pub path: PathBuf,
    pub seed: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub actual_aspect_ratio: String,
    pub actual_image_size: &'static str,
    /// Actual USD cost from `usage.cost`, when reported (for usage stats).
    pub cost: Option<f64>,
}

/// Result of a full generation job: the saved images, the manifest path, plus
/// aggregated dimension warnings and per-variant errors.
pub struct JobSummary {
    pub model: String,
    pub manifest_path: PathBuf,
    pub images: Vec<ImageSummary>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

/// Run a generation job: fan out `variants` in parallel, save each output (with
/// the provider's actual format), write the sidecar manifest, and return a lean
/// summary. Shared by the CLI and the MCP tool.
pub async fn run_job(
    client: &OpenRouterClient,
    req: &GenerateRequest,
    variants: usize,
    base_output: &Path,
    prompt_source: &str,
) -> Result<JobSummary> {
    // Normalize input images once, up front - reused for every variant request
    // and for the manifest (a read/decode failure fails the whole job before any
    // generation, so no spend occurs).
    let prepared = prepare_inputs(&req.images, req.max_image_dimension)?;
    let input_images: Vec<InputImageMeta> = req
        .images
        .iter()
        .zip(&prepared)
        .enumerate()
        .map(|(i, (img, p))| InputImageMeta {
            index: i + 1,
            label: img.label.clone(),
            source: img.path.to_string_lossy().into_owned(),
            source_mime_type: p.source_mime,
            original_width: p.original_width,
            original_height: p.original_height,
            normalized_mime_type: "image/png",
            normalized_width: p.normalized_width,
            normalized_height: p.normalized_height,
            normalization_max_side: req.max_image_dimension,
        })
        .collect();
    let content = build_content(&req.prompt, &req.images, &prepared);

    let outcomes = generate_variants(client, req, variants, content).await;

    let mut images = Vec::new();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut variant_metas = Vec::new();

    // Surface per-input notes (e.g. an SVG with unrendered text) alongside the
    // per-variant dimension warnings.
    for (i, p) in prepared.iter().enumerate() {
        for w in &p.warnings {
            warnings.push(format!("input image {}: {w}", i + 1));
        }
    }

    for outcome in outcomes {
        let mut meta = VariantMeta {
            index: outcome.index + 1,
            seed: outcome.seed,
            requested_aspect_ratio: req.aspect_ratio.clone(),
            requested_image_size: req.image_size.clone(),
            duration_ms: outcome.duration_ms,
            ..Default::default()
        };
        match outcome.result {
            Ok(img) => {
                let ext = image_io::extension_for(&img.mime);
                let path =
                    variant_output_path(base_output, outcome.seed, outcome.index, variants, ext);
                if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent).ok();
                }
                // Isolate a write failure to this variant rather than aborting
                // the whole batch (the image was generated and paid for).
                match std::fs::write(&path, &img.bytes) {
                    Ok(()) => {
                        let check = image_io::check_dimensions(
                            img.width,
                            img.height,
                            req.aspect_ratio.as_deref(),
                            req.image_size.as_deref(),
                        );
                        for w in &check.warnings {
                            warnings.push(format!("variant {}: {w}", outcome.index + 1));
                        }
                        meta.path = Some(path.to_string_lossy().into_owned());
                        meta.mime_type = Some(img.mime.clone());
                        meta.width = Some(img.width);
                        meta.height = Some(img.height);
                        meta.actual_aspect_ratio = Some(check.actual_aspect_ratio.clone());
                        meta.actual_image_size = Some(check.actual_image_size.to_string());
                        meta.generation_id = img.generation_id.clone();
                        meta.provider = img.provider.clone();
                        meta.cost = img.cost;
                        meta.text = img.text.clone();
                        images.push(ImageSummary {
                            path,
                            seed: outcome.seed,
                            width: img.width,
                            height: img.height,
                            actual_aspect_ratio: check.actual_aspect_ratio,
                            actual_image_size: check.actual_image_size,
                            cost: img.cost,
                        });
                    }
                    Err(e) => {
                        let msg = format!("could not write {}: {e}", path.display());
                        errors.push(format!("variant {}: {msg}", outcome.index + 1));
                        meta.error = Some(msg);
                    }
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                errors.push(format!("variant {}: {msg}", outcome.index + 1));
                meta.error = Some(msg);
            }
        }
        variant_metas.push(meta);
    }

    let manifest = Manifest {
        endpoint: "/api/v1/chat/completions",
        model: req.model.clone(),
        prompt: req.prompt.clone(),
        prompt_source: prompt_source.to_string(),
        modalities: modalities_for(req.image_only),
        aspect_ratio: req.aspect_ratio.clone(),
        image_size: req.image_size.clone(),
        base_seed: req.seed,
        variants_requested: variants,
        max_image_dimension: req.max_image_dimension,
        created_at: chrono::Utc::now().to_rfc3339(),
        input_images,
        variants: variant_metas,
    };
    let mpath = manifest_path(base_output);
    // A manifest-write failure must not discard already-saved images / spend.
    if let Err(e) = manifest::write(&mpath, &manifest) {
        errors.push(format!("manifest write failed: {e}"));
    }

    Ok(JobSummary {
        model: req.model.clone(),
        manifest_path: mpath,
        images,
        warnings,
        errors,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    // 1x1 transparent PNG.
    const PNG_1X1_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

    /// Single-image generation (prepare inputs, build content, run core) - the
    /// path production drives via run_job/generate_variants, exercised directly.
    async fn generate_image(
        client: &OpenRouterClient,
        req: &GenerateRequest,
    ) -> Result<GeneratedImage> {
        let prepared = prepare_inputs(&req.images, req.max_image_dimension)?;
        let content = build_content(&req.prompt, &req.images, &prepared);
        generate_core(client, req, content).await
    }

    /// Write a small valid PNG to a temp file and return its path.
    fn temp_png(name: &str) -> PathBuf {
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::new(2, 2));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, buf.into_inner()).unwrap();
        path
    }

    #[test]
    fn prepare_inputs_rasterizes_svg_to_png_data_url() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200"><rect width="400" height="200"/></svg>"#;
        let path = std::env::temp_dir().join("openrouter-mcp-test-input.svg");
        std::fs::write(&path, svg).unwrap();

        let images = vec![InputImage { path, label: None }];
        let prepared = prepare_inputs(&images, 800).unwrap();
        let p = &prepared[0];

        // SVG was rasterized to PNG and fit to the 800px cap (400x200 -> 800x400),
        // intrinsic viewBox size recorded as the original, source flagged as SVG.
        assert!(p.data_url.starts_with("data:image/png;base64,"));
        assert_eq!((p.original_width, p.original_height), (400, 200));
        assert_eq!((p.normalized_width, p.normalized_height), (800, 400));
        assert_eq!(p.source_mime, Some("image/svg+xml"));
        assert!(p.warnings.is_empty());
    }

    #[test]
    fn build_content_is_plain_text_without_images() {
        let content = build_content("just text", &[], &[]);
        let v = serde_json::to_value(&content).unwrap();
        assert_eq!(v, serde_json::json!("just text"));
    }

    #[test]
    fn build_content_puts_text_first_then_images() {
        let images = vec![InputImage {
            path: temp_png("openrouter-mcp-test-content.png"),
            label: None,
        }];
        let prepared = prepare_inputs(&images, 800).unwrap();
        let content = build_content("edit this", &images, &prepared);
        let v = serde_json::to_value(&content).unwrap();
        assert!(v.is_array());
        assert_eq!(v[0]["type"], "text");
        assert_eq!(v[0]["text"], "edit this");
        assert_eq!(v[1]["type"], "image_url");
        assert!(
            v[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn build_content_prepends_label_block_when_labeled() {
        let images = vec![
            InputImage {
                path: temp_png("openrouter-mcp-test-bg.png"),
                label: Some("background".to_string()),
            },
            InputImage {
                path: temp_png("openrouter-mcp-test-fg.png"),
                label: Some("product".to_string()),
            },
        ];
        let prepared = prepare_inputs(&images, 800).unwrap();
        let content = build_content("compose them", &images, &prepared);
        let v = serde_json::to_value(&content).unwrap();
        let text = v[0]["text"].as_str().unwrap();
        assert!(text.contains("Reference images:"));
        assert!(text.contains("1. background:"));
        assert!(text.contains("2. product:"));
        assert!(text.contains("compose them"));
        // text part, then two image parts.
        assert_eq!(v.as_array().unwrap().len(), 3);
    }

    #[test]
    fn resolve_max_dimension_defaults_to_800() {
        assert_eq!(resolve_max_dimension(Some(1024)), 1024);
        assert_eq!(resolve_max_dimension(None), 800);
    }

    #[test]
    fn variant_output_path_single_uses_base_with_ext() {
        let p = variant_output_path(Path::new("out/hero.png"), Some(1200), 0, 1, "jpg");
        assert_eq!(p, PathBuf::from("out/hero.jpg"));
    }

    #[test]
    fn variant_output_path_names_by_seed() {
        // base seed 1000 -> variants 1000, 1001, ...
        let p1 = variant_output_path(Path::new("out/hero.png"), Some(1000), 0, 4, "png");
        let p2 = variant_output_path(Path::new("out/hero.png"), Some(1003), 3, 4, "png");
        assert_eq!(p1, PathBuf::from("out/hero-var-1000.png"));
        assert_eq!(p2, PathBuf::from("out/hero-var-1003.png"));
    }

    #[test]
    fn variant_output_path_pads_small_seed_to_four_digits() {
        let p = variant_output_path(Path::new("hero.png"), Some(42), 0, 4, "png");
        assert_eq!(p, PathBuf::from("hero-var-0042.png"));
    }

    #[test]
    fn variant_output_path_falls_back_to_index_without_seed() {
        // No seed (provider randomizes) -> zero-padded index, sorts for 10+.
        let p = variant_output_path(Path::new("hero.png"), None, 9, 12, "png");
        assert_eq!(p, PathBuf::from("hero-var-010.png"));
    }

    #[test]
    fn manifest_path_is_stem_dot_manifest_json() {
        assert_eq!(
            manifest_path(Path::new("out/hero.png")),
            PathBuf::from("out/hero.manifest.json")
        );
    }

    #[tokio::test]
    async fn generate_image_sends_request_and_decodes_response() {
        let server = MockServer::start().await;
        let data_url = format!("data:image/png;base64,{PNG_1X1_B64}");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            // Verify the request shape we build.
            .and(body_partial_json(json!({
                "model": "google/gemini-3.1-flash-image-preview",
                "modalities": ["image", "text"],
                "seed": 1200,
                "stream": false,
                "image_config": { "aspect_ratio": "1:1", "image_size": "1K" }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "gen-abc",
                "model": "google/gemini-3.1-flash-image-preview",
                "provider": "Google",
                "choices": [{
                    "message": { "content": null, "images": [
                        { "type": "image_url", "image_url": { "url": data_url } }
                    ]}
                }],
                "usage": { "cost": 0.0684 }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = GenerateRequest {
            model: "google/gemini-3.1-flash-image-preview".to_string(),
            prompt: "an owl".to_string(),
            aspect_ratio: Some("1:1".to_string()),
            image_size: Some("1K".to_string()),
            seed: Some(1200),
            image_only: false,
            images: vec![],
            max_image_dimension: 800,
        };
        let img = generate_image(&client, &req).await.unwrap();
        assert_eq!((img.width, img.height), (1, 1));
        assert_eq!(img.mime, "image/png");
        assert_eq!(img.cost, Some(0.0684));
        assert_eq!(img.generation_id.as_deref(), Some("gen-abc"));
    }

    #[tokio::test]
    async fn generate_image_surfaces_provider_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(400).set_body_string("{\"error\":\"invalid image_size\"}"),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = GenerateRequest {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: None,
            image_size: Some("0.5K".to_string()),
            seed: None,
            image_only: false,
            images: vec![],
            max_image_dimension: 800,
        };
        let err = generate_image(&client, &req).await.unwrap_err();
        assert!(err.to_string().contains("invalid image_size"));
    }

    #[tokio::test]
    async fn describe_image_sends_image_and_returns_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            // A describe call has no `modalities` and content is an array (text + image).
            .and(body_partial_json(json!({
                "messages": [{ "content": [{ "type": "text", "text": "What is this?" }] }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "content": "A small green lizard." } }],
                "usage": { "cost": 0.002 }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = DescribeRequest {
            model: "google/gemini-2.5-flash".to_string(),
            prompt: "What is this?".to_string(),
            images: vec![InputImage {
                path: temp_png("openrouter-mcp-test-describe.png"),
                label: None,
            }],
            max_image_dimension: 800,
        };
        let result = describe_image(&client, &req).await.unwrap();
        assert_eq!(result.text, "A small green lizard.");
        assert_eq!(result.cost, Some(0.002));
    }

    #[tokio::test]
    async fn describe_image_requires_an_image() {
        let client = OpenRouterClient::with_base_url("http://127.0.0.1:9", "k");
        let req = DescribeRequest {
            model: "m".to_string(),
            prompt: "p".to_string(),
            images: vec![],
            max_image_dimension: 800,
        };
        assert!(describe_image(&client, &req).await.is_err());
    }

    #[tokio::test]
    async fn generate_image_only_uses_image_modality() {
        let server = MockServer::start().await;
        let data_url = format!("data:image/jpeg;base64,{PNG_1X1_B64}");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(json!({ "modalities": ["image"] })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "images": [
                    { "image_url": { "url": data_url } }
                ]}}]
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = GenerateRequest {
            model: "x-ai/grok-imagine-image-quality".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: None,
            image_size: None,
            seed: None,
            image_only: true,
            images: vec![],
            max_image_dimension: 800,
        };
        // mime is sniffed from the data URL prefix, even when the bytes are PNG.
        let img = generate_image(&client, &req).await.unwrap();
        assert_eq!(img.mime, "image/jpeg");
    }
}
