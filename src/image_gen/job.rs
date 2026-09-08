//! Image-generation job orchestration: fan out variants, save outputs, write the
//! sidecar manifest, and return a lean summary. Shared by the CLI and the MCP tool.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use tokio::sync::Semaphore;

use crate::image_io;
use crate::manifest::{self, InputImageMeta, Manifest, VariantMeta};
use crate::openrouter::OpenRouterClient;
use crate::output::{base_stem, in_parent_of};

use super::{GenContent, GenerateRequest, GeneratedImage, build_gen_content, generate_core};

/// Avoid multiplying large base64 reference-image request bodies across every
/// requested variant at once. Four still provides useful API parallelism.
const MAX_CONCURRENT_VARIANTS: usize = 4;

/// Outcome of one variant generation (an image, or a per-variant error).
pub struct VariantOutcome {
    pub index: usize,
    pub seed: Option<u64>,
    pub duration_ms: u128,
    pub result: Result<GeneratedImage>,
}

/// Generate `variants` images concurrently (at most [`MAX_CONCURRENT_VARIANTS`]
/// in flight). With a base seed each variant uses `base + index` for
/// reproducible, distinct outputs. A failed variant is captured in its `result`
/// without aborting the others. Returned ordered by index.
pub async fn generate_variants(
    client: &OpenRouterClient,
    req: &GenerateRequest,
    variants: usize,
    content: GenContent,
) -> Vec<VariantOutcome> {
    let base_seed = req.seed;
    // saturating_add: a base seed near u64::MAX must not overflow-panic.
    let seed_for = |i: usize| base_seed.map(|s| s.saturating_add(i as u64));
    // One shared copy of the request and pre-built content; each task builds
    // its request body only once it holds a permit, so at most
    // MAX_CONCURRENT_VARIANTS copies of the reference images exist at a time.
    let req = Arc::new(GenerateRequest {
        model: req.model.clone(),
        prompt: String::new(), // GenContent already contains the assembled prompt.
        aspect_ratio: req.aspect_ratio.clone(),
        image_size: req.image_size.clone(),
        seed: req.seed,
        images: Vec::new(), // Normalized references live in GenContent.
        max_image_dimension: req.max_image_dimension,
        quality: req.quality.clone(),
        output_format: req.output_format.clone(),
        background: req.background.clone(),
        output_compression: req.output_compression,
    });
    let content = Arc::new(content);
    let permits = Arc::new(Semaphore::new(MAX_CONCURRENT_VARIANTS));
    // Dropping this set aborts every outstanding variant, including tasks
    // waiting for a permit, when the caller cancels the job.
    let mut tasks = tokio::task::JoinSet::new();
    let mut indices = std::collections::HashMap::new();
    for i in 0..variants {
        let client = client.clone();
        let req = Arc::clone(&req);
        let content = Arc::clone(&content);
        let permits = Arc::clone(&permits);
        let seed = seed_for(i);
        let handle = tasks.spawn(async move {
            let _permit = permits.acquire().await.expect("semaphore closed");
            let start = Instant::now();
            let result = generate_core(&client, &req, seed, &content).await;
            VariantOutcome {
                index: i,
                seed,
                duration_ms: start.elapsed().as_millis(),
                result,
            }
        });
        indices.insert(handle.id(), i);
    }

    let mut outcomes = Vec::with_capacity(variants);
    while let Some(result) = tasks.join_next().await {
        outcomes.push(match result {
            Ok(outcome) => outcome,
            Err(e) => {
                let index = indices[&e.id()];
                VariantOutcome {
                    index,
                    seed: seed_for(index),
                    duration_ms: 0,
                    result: Err(anyhow::anyhow!("variant task failed: {e}")),
                }
            }
        });
    }
    outcomes.sort_by_key(|outcome| outcome.index);
    outcomes
}

/// Output path for one variant. A single variant uses `base` with the given
/// extension. Multiple variants get a `-var-<seed>-<index>` suffix.
/// Seeds use at least four digits, and indices use at least three digits.
/// Without a seed, the suffix contains only the index.
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
        // The index stays unique even if seed stepping saturates at u64::MAX.
        Some(s) => format!("{s:04}-{:03}", index_zero_based + 1),
        None => {
            let width = 3.max(total.to_string().len());
            format!("{:0width$}", index_zero_based + 1, width = width)
        }
    };
    in_parent_of(base, format!("{}-var-{marker}.{ext}", base_stem(base)))
}

/// One saved image in a job's lean summary.
pub struct ImageSummary {
    pub path: PathBuf,
    pub seed: Option<u64>,
    pub width: u32,
    pub height: u32,
    pub actual_aspect_ratio: String,
    pub actual_image_size: &'static str,
}

/// Result of a full generation job: the saved images, the manifest path, plus
/// aggregated dimension warnings and per-variant errors.
pub struct JobSummary {
    pub model: String,
    pub manifest_path: PathBuf,
    pub images: Vec<ImageSummary>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub billing: crate::billing::Totals,
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
    let prepared = super::prepare_inputs_async(&req.images, req.max_image_dimension).await?;
    let input_images: Vec<InputImageMeta> = req
        .images
        .iter()
        .zip(&prepared)
        .enumerate()
        .map(|(i, (img, p))| InputImageMeta {
            index: i + 1,
            label: img.label.clone(),
            source: img.source_label(),
            source_mime_type: p.source_mime,
            original_width: p.original_width,
            original_height: p.original_height,
            normalized_mime_type: p.normalized_mime,
            normalized_width: p.normalized_width,
            normalized_height: p.normalized_height,
            normalization_max_side: req.max_image_dimension,
        })
        .collect();
    let content = build_gen_content(&req.prompt, &req.images, &prepared);

    let outcomes = generate_variants(client, req, variants, content).await;

    let warnings = prepared
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            p.warnings
                .iter()
                .map(move |w| format!("input image {}: {w}", i + 1))
        })
        .collect();
    save_outcomes(
        req,
        base_output,
        prompt_source,
        input_images,
        warnings,
        outcomes,
    )
    .await
}

/// Delivery and receipt for one image, independent of batch aggregation.
struct VariantDelivery {
    meta: VariantMeta,
    image: Option<ImageSummary>,
    receipt: Option<crate::billing::Receipt>,
    warnings: Vec<String>,
    errors: Vec<String>,
}

async fn deliver_variant(
    req: &GenerateRequest,
    base_output: &Path,
    variants: usize,
    outcome: VariantOutcome,
) -> VariantDelivery {
    let mut image = None;
    let mut receipt_out = None;
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
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
            receipt_out = Some(crate::billing::Receipt {
                cost: img.cost,
                generation_id: img.generation_id.clone(),
            });
            meta.generation_id = img.generation_id.clone();
            meta.provider = img.provider.clone();
            meta.cost = img.cost;
            let ext = image_io::extension_for(&img.mime);
            let path = variant_output_path(base_output, outcome.seed, outcome.index, variants, ext);
            // Isolate a write failure to this variant rather than aborting
            // the whole batch (the image was generated and paid for).
            match crate::output::write_bytes(&path, &img.bytes).await {
                Ok(()) => {
                    let check = image_io::check_dimensions(
                        img.width,
                        img.height,
                        req.aspect_ratio.as_deref(),
                        req.image_size.as_deref(),
                    );
                    for w in check.warnings.iter().chain(&img.warnings) {
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
                    image = Some(ImageSummary {
                        path,
                        seed: outcome.seed,
                        width: img.width,
                        height: img.height,
                        actual_aspect_ratio: check.actual_aspect_ratio,
                        actual_image_size: check.actual_image_size,
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
            if let Some(receipt) = crate::billing::Receipt::from_error(&e) {
                receipt_out = Some(receipt.clone());
                meta.cost = receipt.cost;
                meta.generation_id = receipt.generation_id.clone();
            }
            let msg = format!("{e:#}");
            errors.push(format!("variant {}: {msg}", outcome.index + 1));
            meta.error = Some(msg);
        }
    }
    VariantDelivery {
        meta,
        image,
        receipt: receipt_out,
        warnings,
        errors,
    }
}

/// Persist outcomes independently from generation so delivery failures retain billing.
async fn save_outcomes(
    req: &GenerateRequest,
    base_output: &Path,
    prompt_source: &str,
    input_images: Vec<InputImageMeta>,
    mut warnings: Vec<String>,
    outcomes: Vec<VariantOutcome>,
) -> Result<JobSummary> {
    let variants = outcomes.len();
    let mut billing = crate::billing::Totals::default();
    let mut images = Vec::new();
    let mut errors = Vec::new();
    let mut variant_metas = Vec::new();

    for outcome in outcomes {
        let delivery = deliver_variant(req, base_output, variants, outcome).await;
        if let Some(receipt) = delivery.receipt {
            billing.add(&receipt);
        }
        images.extend(delivery.image);
        warnings.extend(delivery.warnings);
        errors.extend(delivery.errors);
        variant_metas.push(delivery.meta);
    }

    let manifest = Manifest {
        endpoint: "/api/v1/images",
        model: req.model.clone(),
        prompt: req.prompt.clone(),
        prompt_source: prompt_source.to_string(),
        aspect_ratio: req.aspect_ratio.clone(),
        image_size: req.image_size.clone(),
        base_seed: req.seed,
        variants_requested: variants,
        max_image_dimension: req.max_image_dimension,
        quality: req.quality.clone(),
        output_format: req.output_format.clone(),
        background: req.background.clone(),
        output_compression: req.output_compression,
        created_at: chrono::Utc::now().to_rfc3339(),
        input_images,
        variants: variant_metas,
    };
    let mpath = manifest::path(base_output);
    // A manifest-write failure must not discard already-saved images / spend.
    if let Err(e) = manifest::write(&mpath, &manifest).await {
        errors.push(format!("manifest write failed: {e}"));
    }

    Ok(JobSummary {
        model: req.model.clone(),
        manifest_path: mpath,
        images,
        warnings,
        errors,
        billing,
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
        assert_eq!(p1, PathBuf::from("out/hero-var-1000-001.png"));
        assert_eq!(p2, PathBuf::from("out/hero-var-1003-004.png"));
    }

    #[test]
    fn variant_output_path_pads_small_seed_to_four_digits() {
        let p = variant_output_path(Path::new("hero.png"), Some(42), 0, 4, "png");
        assert_eq!(p, PathBuf::from("hero-var-0042-001.png"));
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
            crate::manifest::path(Path::new("out/hero.png")),
            PathBuf::from("out/hero.manifest.json")
        );
    }

    /// F5: the manifest doc says it holds "the full request settings" - the
    /// four new image knobs (quality/output_format/background/
    /// output_compression) must actually round-trip onto disk, not just live in
    /// the request struct.
    #[tokio::test]
    async fn run_job_records_the_new_image_knobs_in_the_manifest() {
        use wiremock::matchers::{method, path as wpath};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        const PNG_1X1_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(wpath("/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "b64_json": PNG_1X1_B64 }]
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let req = GenerateRequest {
            model: "openai/gpt-image-2".to_string(),
            prompt: "an owl".to_string(),
            aspect_ratio: Some("1:1".to_string()),
            image_size: Some("1K".to_string()),
            seed: None,
            images: vec![],
            max_image_dimension: 800,
            quality: Some("high".to_string()),
            output_format: Some("webp".to_string()),
            background: Some("transparent".to_string()),
            output_compression: Some(80),
        };
        let base = std::env::temp_dir().join("openrouter-mcp-manifest-knobs-test/hero.png");
        let summary = run_job(&client, &req, 1, &base, "inline").await.unwrap();

        let manifest_json = std::fs::read_to_string(&summary.manifest_path).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&manifest_json).unwrap();
        assert_eq!(manifest["quality"], "high");
        assert_eq!(manifest["output_format"], "webp");
        assert_eq!(manifest["background"], "transparent");
        assert_eq!(manifest["output_compression"], 80);
    }
}

#[cfg(test)]
mod audit_regression {
    use super::*;
    fn request() -> GenerateRequest {
        GenerateRequest {
            model: "test/image".into(),
            prompt: "test".into(),
            aspect_ratio: None,
            image_size: None,
            seed: None,
            images: vec![],
            max_image_dimension: 800,
            quality: None,
            output_format: None,
            background: None,
            output_compression: None,
        }
    }

    #[tokio::test]
    async fn mixed_image_delivery_keeps_all_receipts_and_manifest_errors() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("outputs/image.png");
        // A directory at one output path forces that write to fail on every OS.
        std::fs::create_dir_all(variant_output_path(&base, None, 0, 4, "png")).unwrap();
        let image = |cost, id: &str| GeneratedImage {
            bytes: vec![1, 2, 3],
            mime: "image/png".into(),
            width: 1,
            height: 1,
            text: None,
            cost,
            generation_id: Some(id.into()),
            provider: Some("test".into()),
            warnings: vec!["provider note".into()],
        };
        let results = vec![
            Ok(image(Some(0.25), "write-failed")),
            Ok(image(None, "saved-unknown")),
            Err(
                anyhow::anyhow!("bad image bytes").context(crate::billing::Receipt {
                    cost: Some(0.5),
                    generation_id: Some("decode-failed".into()),
                }),
            ),
            Err(anyhow::anyhow!("submission rejected")),
        ];
        let outcomes = results
            .into_iter()
            .enumerate()
            .map(|(index, result)| VariantOutcome {
                index,
                seed: None,
                duration_ms: 1,
                result,
            })
            .collect();
        let summary = save_outcomes(&request(), &base, "test", vec![], vec![], outcomes)
            .await
            .unwrap();
        assert_eq!(summary.images.len(), 1);
        assert_eq!(
            std::fs::read(&summary.images[0].path).unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(summary.billing.cost, 0.75);
        assert_eq!(summary.billing.unknown, 1);
        assert_eq!(summary.errors.len(), 3);
        assert_eq!(summary.warnings, vec!["variant 2: provider note"]);
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(summary.manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["variants"][0]["cost"], 0.25);
        assert_eq!(manifest["variants"][0]["generation_id"], "write-failed");
        assert!(manifest["variants"][0]["error"].is_string());
        assert!(manifest["variants"][0].get("path").is_none());
        assert_eq!(manifest["variants"][0]["provider"], "test");
        assert!(manifest["variants"][1]["path"].is_string());
        assert_eq!(manifest["variants"][2]["cost"], 0.5);
        assert_eq!(manifest["variants"][2]["generation_id"], "decode-failed");
    }

    #[test]
    fn extreme_seed_variants_keep_distinct_output_paths() {
        let paths: std::collections::HashSet<_> = (0..16)
            .map(|i| {
                variant_output_path(
                    Path::new("probe.png"),
                    Some(u64::MAX.saturating_add(i as u64)),
                    i,
                    16,
                    "png",
                )
            })
            .collect();
        assert_eq!(paths.len(), 16);
    }
    #[tokio::test]
    async fn cancelling_variants_aborts_waiting_requests() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images"))
            .respond_with(
                ResponseTemplate::new(500).set_delay(std::time::Duration::from_millis(200)),
            )
            .mount(&server)
            .await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let job = tokio::spawn(async move {
            generate_variants(
                &client,
                &request(),
                12,
                GenContent {
                    prompt: "test".into(),
                    reference_urls: vec![],
                },
            )
            .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while server.received_requests().await.unwrap().len() < MAX_CONCURRENT_VARIANTS {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        job.abort();
        assert!(matches!(job.await, Err(error) if error.is_cancelled()));
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        assert_eq!(
            server.received_requests().await.unwrap().len(),
            MAX_CONCURRENT_VARIANTS
        );
    }

    #[tokio::test]
    async fn variant_completion_order_preserves_indices_seeds_and_receipts() {
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        for index in 0..3 {
            Mock::given(method("POST"))
                .and(path("/images"))
                .and(body_partial_json(serde_json::json!({"seed": 10 + index})))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_delay(std::time::Duration::from_millis((2 - index) * 20))
                        .set_body_json(serde_json::json!({
                            "id": format!("gen-{index}"),
                            "data": [{"b64_json": "not an image"}],
                            "usage": {"cost": 0.25}
                        })),
                )
                .expect(1)
                .mount(&server)
                .await;
        }
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let mut req = request();
        req.seed = Some(10);
        let outcomes = generate_variants(
            &client,
            &req,
            3,
            GenContent {
                prompt: "test".into(),
                reference_urls: vec![],
            },
        )
        .await;
        for (index, outcome) in outcomes.into_iter().enumerate() {
            assert_eq!(outcome.index, index);
            assert_eq!(outcome.seed, Some(10 + index as u64));
            let error = outcome.result.unwrap_err();
            assert_eq!(
                crate::billing::Receipt::from_error(&error).unwrap().cost,
                Some(0.25)
            );
        }
    }
}
