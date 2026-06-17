//! Image-generation job orchestration: fan out variants, save outputs, write the
//! sidecar manifest, and return a lean summary. Shared by the CLI and the MCP tool.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;

use crate::image_io;
use crate::manifest::{self, InputImageMeta, Manifest, VariantMeta};
use crate::openrouter::{Content, OpenRouterClient};

use super::{
    GenerateRequest, GeneratedImage, build_content, generate_core, modalities_for, prepare_inputs,
};

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
    use super::*;

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
}
