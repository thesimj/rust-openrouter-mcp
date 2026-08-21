//! Image tools (`generate_image`, `describe_image`), their argument structs, the
//! shared `ImageInput` type, and the image-job result builder.

use std::net::{IpAddr, SocketAddr, ToSocketAddrs};

use base64::Engine;
use rmcp::{
    ErrorData, RoleServer,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    service::RequestContext,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::image_gen::{self, GenerateRequest};
use crate::server::naming;
use crate::server::result::{
    DEFAULT_WAIT_SECONDS, attach_warnings_errors, client_wants_inline_previews,
};
use crate::server::schema::{
    AtLeastOneOf, RequireFields, de_opt_uint, require_all, scalarize_nullable,
};
use crate::tasks::TaskKind;

use super::OpenRouterServer;

/// Hard ceiling for images fetched from third-party URLs. Input images are
/// downscaled to the resolved dimension cap before use, so accepting
/// arbitrarily large source bodies only increases memory pressure.
const MAX_REMOTE_IMAGE_BYTES: u64 = 20 * 1024 * 1024;

/// Total deadline for one remote-image fetch, sized against the ceiling above:
/// 20 MB inside 30s is ~5 Mbit/s, slower than any host worth waiting for.
const REMOTE_IMAGE_TIMEOUT_SECS: u64 = 30;

/// An input image for editing / image-to-image / vision. Exactly one of
/// `path`, `url`, or `base64` must be set. Order is preserved.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = AtLeastOneOf(&["path", "url", "base64"]))]
pub(crate) struct ImageInput {
    /// Local file path (png/jpeg/webp/gif/svg). One of path/url/base64.
    #[serde(default)]
    pub path: Option<String>,
    /// HTTP(S) URL to fetch the image from. One of path/url/base64.
    #[serde(default)]
    pub url: Option<String>,
    /// Inline image data: a full `data:` URL or raw base64. One of path/url/base64.
    #[serde(default)]
    pub base64: Option<String>,
    /// Optional label, surfaced to the model as a reference name.
    #[serde(default)]
    pub label: Option<String>,
}

/// Decode an inline `base64`/data-URL argument to raw bytes.
fn decode_inline(data: &str) -> Result<Vec<u8>, ErrorData> {
    let data = data.trim();
    if data.starts_with("data:") {
        crate::image_io::parse_data_url(data)
            .map(|(_mime, bytes)| bytes)
            .map_err(|e| ErrorData::invalid_params(format!("invalid data URL: {e}"), None))
    } else {
        base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|e| ErrorData::invalid_params(format!("invalid base64 image data: {e}"), None))
    }
}

/// True for IPs a fetched URL must never reach (SSRF guard): loopback, private
/// (RFC1918), CGNAT (100.64/10), link-local (incl. cloud metadata 169.254.169.254),
/// unspecified, broadcast, documentation, multicast, and IPv6 ULA/link-local.
fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64.0.0/10 (CGNAT)
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(v4));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
        }
    }
}

/// Fetch an image URL's bytes with a plain client. Deliberately does NOT use the
/// OpenRouter-authenticated client, so the API key is never sent to a
/// third-party URL. SSRF-hardened: only http/https; the host is resolved and
/// rejected if it points at a private/loopback/link-local address; redirects are
/// disabled; and the connection is pinned to the validated IP so DNS can't be
/// rebound between the check and the request.
async fn fetch_url(url: &str) -> Result<Vec<u8>, ErrorData> {
    let invalid = |msg: String| ErrorData::invalid_params(msg, None);

    let parsed =
        reqwest::Url::parse(url).map_err(|e| invalid(format!("invalid image url: {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(invalid(format!("image url must be http(s): {url}")));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| invalid("image url has no host".to_string()))?
        .to_string();
    let port = parsed.port_or_known_default().unwrap_or(443);

    // Resolve off the async runtime, then refuse internal/private targets.
    let lookup = host.clone();
    let addrs: Vec<SocketAddr> = tokio::task::spawn_blocking(move || {
        (lookup.as_str(), port)
            .to_socket_addrs()
            .map(|it| it.collect::<Vec<_>>())
    })
    .await
    .map_err(|e| ErrorData::internal_error(format!("dns task failed: {e}"), None))?
    .map_err(|e| invalid(format!("could not resolve image url host: {e}")))?;

    if addrs.is_empty() {
        return Err(invalid("image url host did not resolve".to_string()));
    }
    if addrs.iter().any(|a| is_blocked_ip(a.ip())) {
        return Err(invalid(
            "image url resolves to a private/loopback/link-local address; refused".to_string(),
        ));
    }

    // Pin to the validated IP (no second DNS lookup -> no rebinding) and forbid
    // redirects (a 30x could otherwise bounce to an internal host).
    //
    // A total deadline is right here, unlike the shared OpenRouter client: this
    // fetches a URL the *model* supplied, and the body is capped at
    // MAX_REMOTE_IMAGE_BYTES, so there is no legitimate slow-but-large transfer
    // to protect. Without it, a host that accepts and then dribbles bytes hangs
    // the tool call forever - the size cap never trips on a drip.
    //
    // `no_gzip` because enabling reqwest's `gzip` feature turns auto-decompression
    // on for every client in the process. Decoded responses lose Content-Length,
    // which would silently kill the early size check below; image bytes are
    // already compressed, so there is nothing to win here anyway.
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .resolve(&host, addrs[0])
        .timeout(std::time::Duration::from_secs(REMOTE_IMAGE_TIMEOUT_SECS))
        .no_gzip()
        .build()
        .map_err(|e| ErrorData::internal_error(format!("http client build failed: {e}"), None))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| invalid(format!("could not fetch image url: {e}")))?;
    if resp.status().is_redirection() {
        return Err(invalid(
            "image url returned a redirect; refused (SSRF guard)".to_string(),
        ));
    }
    let mut resp = resp
        .error_for_status()
        .map_err(|e| invalid(format!("image url returned an error: {e}")))?;
    let content_length = resp.content_length();
    if let Some(length) = content_length
        && length > MAX_REMOTE_IMAGE_BYTES
    {
        return Err(invalid(format!(
            "image url body is too large ({length} bytes; maximum is {MAX_REMOTE_IMAGE_BYTES})"
        )));
    }

    // Enforce the limit while streaming as Content-Length may be absent or
    // inaccurate. `Response::bytes()` would buffer an unbounded body first.
    let capacity = content_length.unwrap_or(0).min(MAX_REMOTE_IMAGE_BYTES) as usize;
    let mut bytes = Vec::with_capacity(capacity);
    while let Some(chunk) = resp.chunk().await.map_err(|e| {
        ErrorData::internal_error(format!("could not read image url body: {e}"), None)
    })? {
        let next_len = bytes.len().saturating_add(chunk.len());
        if next_len as u64 > MAX_REMOTE_IMAGE_BYTES {
            return Err(invalid(format!(
                "image url body exceeds the {MAX_REMOTE_IMAGE_BYTES}-byte maximum"
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// Validate that an image spec carries exactly one source (path/url/base64),
/// cheaply and without any network fetch. Lets a caller (e.g. `chat_completion`)
/// surface a malformed-image error before running the network-bound model-
/// capability gate.
pub(crate) fn check_image_input(img: &ImageInput) -> Result<(), ErrorData> {
    let count = [&img.path, &img.url, &img.base64]
        .iter()
        .filter(|o| o.as_ref().is_some_and(|s| !s.trim().is_empty()))
        .count();
    if count != 1 {
        return Err(ErrorData::invalid_params(
            "each image needs exactly one of: path, url, or base64".to_string(),
            None,
        ));
    }
    Ok(())
}

/// Resolve one tool-level [`ImageInput`] to a generator [`image_gen::InputImage`],
/// fetching URLs and decoding base64/data-URL inputs. Requires exactly one source.
async fn resolve_image_input(img: ImageInput) -> Result<image_gen::InputImage, ErrorData> {
    check_image_input(&img)?;
    let label = img.label;
    if let Some(p) = img.path.filter(|s| !s.trim().is_empty()) {
        Ok(image_gen::InputImage::from_path(p, label))
    } else if let Some(b64) = img.base64.filter(|s| !s.trim().is_empty()) {
        Ok(image_gen::InputImage::inline(
            decode_inline(&b64)?,
            "inline",
            label,
        ))
    } else {
        let url = img.url.unwrap();
        let bytes = fetch_url(&url).await?;
        Ok(image_gen::InputImage::inline(bytes, url, label))
    }
}

/// Resolve a list of tool-level [`ImageInput`]s to generator inputs, in order.
pub(crate) async fn resolve_image_inputs(
    images: Vec<ImageInput>,
) -> Result<Vec<image_gen::InputImage>, ErrorData> {
    let mut out = Vec::with_capacity(images.len());
    for img in images {
        out.push(resolve_image_input(img).await?);
    }
    Ok(out)
}

/// Arguments for the `generate_image` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = RequireFields(&["aspect_ratio", "image_size"]))]
pub(crate) struct GenerateImageArgs {
    /// Image model id, e.g. "google/gemini-3.1-flash-image-preview".
    pub model: String,
    /// Prompt text describing the image to generate (or the edit to apply).
    pub prompt: String,
    /// REQUIRED (no default): aspect ratio, e.g. "1:1", "16:9", "9:16"
    /// (maps to image_config.aspect_ratio).
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    /// REQUIRED (no default): resolution TIER (not pixel dimensions), e.g.
    /// "1K", "2K", "4K" (maps to image_config.image_size).
    #[serde(default)]
    pub image_size: Option<String>,
    /// Seed for reproducible-ish generation (provider support varies).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub seed: Option<u64>,
    /// Input images to edit/condition on (image-to-image / multi-image). Each
    /// takes exactly one of: path (local file), url (http/https, fetched), or
    /// base64 (a data: URL or raw base64). Omit for plain text-to-image.
    #[serde(default)]
    pub images: Vec<ImageInput>,
    /// Longest-side cap (px) for input images before sending (default 1536,
    /// max 4096; env OPENROUTER_IMAGE_MAX_DIMENSION).
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(max = 4096))]
    pub max_image_dimension: Option<u32>,
    /// Number of variants to generate in parallel (1-16, seed-stepped). Default 1.
    /// With >1, files are named <output>-var-<seed> (zero-padded to 4 digits), or
    /// -var-<index> when no seed is set; one manifest covers all variants.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(min = 1, max = 16))]
    pub variants: Option<usize>,
    /// Seconds to wait inline before returning a task_id for a slow job (1-60,
    /// default 10). The job keeps running; fetch it later with get_result.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(min = 1, max = 60))]
    pub wait_seconds: Option<u64>,
    /// Output file path (single image, or the base name for variants). The
    /// extension is corrected to the actual returned format. Optional: when
    /// omitted, an auto-named file is written under OPENROUTER_MCP_OUTPUT_DIR
    /// (default $HOME/Downloads/openrouter-mcp).
    #[serde(default)]
    pub output: Option<String>,
    /// Output quality: "auto", "low", "medium", or "high". Provider support varies.
    #[serde(default)]
    pub quality: Option<String>,
    /// Output file format: "png", "jpeg", "webp", or "svg". Provider support
    /// varies; the saved file's extension always matches what the provider
    /// actually returns, not this request.
    #[serde(default)]
    pub output_format: Option<String>,
    /// Background: "auto", "transparent", or "opaque". Provider support varies.
    #[serde(default)]
    pub background: Option<String>,
    /// Output compression 0-100 (webp/jpeg only). Provider support varies.
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(min = 0, max = 100))]
    pub output_compression: Option<u32>,
}

/// Arguments for the `describe_image` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct DescribeImageArgs {
    /// Vision-capable model id (image input, text output), e.g.
    /// "google/gemini-2.5-flash" or "anthropic/claude-sonnet-4.6".
    pub model: String,
    /// Image(s) to describe (at least one required). Each takes exactly one of:
    /// path (local file), url (http/https), or base64 (data: URL or raw base64).
    #[schemars(length(min = 1))]
    pub images: Vec<ImageInput>,
    /// Instruction or question about the image(s). Defaults to a detailed description.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Longest-side cap (px) for input images before sending (default 1536, max 4096).
    #[serde(default, deserialize_with = "de_opt_uint")]
    #[schemars(range(max = 4096))]
    pub max_image_dimension: Option<u32>,
    /// Optional reasoning effort: "max", "xhigh", "high", "medium", "low",
    /// "minimal" or "none". Omit to keep the model's own default. Accepted
    /// values differ per model - check `reasoning.supported_efforts` from
    /// list_models/describe_model. Plain description needs little thinking;
    /// charts, diagrams and dense text benefit from "high".
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// Build the lean per-job result object for an image job (paths, dims, requested
/// vs actual, manifest pointer, plus warnings/errors when present).
fn image_job_result_json(
    summary: &image_gen::JobSummary,
    aspect_ratio: &Option<String>,
    image_size: &Option<String>,
) -> serde_json::Value {
    let images: Vec<_> = summary
        .images
        .iter()
        .map(|img| {
            json!({
                "path": img.path.to_string_lossy(),
                "seed": img.seed,
                "width": img.width,
                "height": img.height,
                "aspect_ratio": aspect_ratio,
                "image_size": image_size,
                "actual_aspect_ratio": img.actual_aspect_ratio,
                "actual_image_size": img.actual_image_size,
            })
        })
        .collect();
    let mut result = json!({
        "ok": true,
        "model": summary.model,
        "images": images,
        "manifest": summary.manifest_path.to_string_lossy(),
    });
    attach_warnings_errors(&mut result, &summary.warnings, &summary.errors);
    result
}

#[tool_router(router = image_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Generate or edit an image with an OpenRouter image model (e.g. \
        google/gemini-3.1-flash-image-preview) and save it. Runs asynchronously: if the job is \
        still going after wait_seconds (default 10), it returns status \"pending\" with a \
        task_id to poll via get_result; otherwise it returns the completed result inline. \
        `output` is optional - omit it to \
        get an auto-named file (kind_datetime_model_config_seed_hash) under \
        OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp). For text-to-image, \
        pass a prompt. For editing / image-to-image, also pass `images` - each given as a \
        local path, an http(s) url, or base64/data-URL (order preserved; optional per-image \
        label) - the prompt becomes the edit instruction. \
        Set variants>1 to generate several in parallel (seed-stepped). Optional `quality` \
        (auto/low/medium/high), `output_format` (png/jpeg/webp/svg), `background` \
        (auto/transparent/opaque), and `output_compression` (0-100, webp/jpeg only) are passed \
        straight through to the provider - support for each varies by model, and whatever \
        format actually comes back is what gets saved (the extension always matches the real \
        result, not the request). Returns a compact \
        result: saved image paths, decoded width/height, requested vs actual \
        aspect_ratio/image_size, seeds, a path to the sidecar manifest, and any mismatch \
        warnings. Works with any OpenRouter image model (Nano Banana, Grok, \
        Seedream, FLUX, GPT Image, Recraft, ...) via the dedicated image endpoint. No defaults \
        for the required fields: model, prompt, aspect_ratio and image_size must all be \
        specified, or the call fails with an error naming what is missing (every other \
        param - seed, images, max_image_dimension, variants, wait_seconds, output, quality, \
        output_format, background, output_compression - is optional). To analyze or caption \
        an existing image instead of creating one, use describe_image.",
        annotations(
            title = "Generate Image",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn generate_image(
        &self,
        Parameters(args): Parameters<GenerateImageArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let inline_previews = client_wants_inline_previews(&context);
        self.run_generate(args, inline_previews).await
    }

    /// Core of `generate_image`, parameterized on whether to embed inline image
    /// previews (decided per-client by the tool entrypoint). Separated so tests
    /// can drive it without constructing a transport `RequestContext`.
    pub(crate) async fn run_generate(
        &self,
        args: GenerateImageArgs,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        // Blank/whitespace-only strings count as absent: better a clear
        // "missing parameter" error here than a confusing provider 400.
        let non_blank = |o: Option<String>| o.filter(|s| !s.trim().is_empty());
        let args = GenerateImageArgs {
            aspect_ratio: non_blank(args.aspect_ratio),
            image_size: non_blank(args.image_size),
            quality: non_blank(args.quality),
            output_format: non_blank(args.output_format),
            background: non_blank(args.background),
            ..args
        };
        // No defaults: the agent must choose these explicitly.
        let mut missing: Vec<&str> = Vec::new();
        if args.aspect_ratio.is_none() {
            missing.push("aspect_ratio (e.g. \"1:1\", \"16:9\", \"9:16\")");
        }
        if args.image_size.is_none() {
            missing.push("image_size (e.g. \"1K\", \"2K\", \"4K\")");
        }
        require_all("generate_image", "image", &missing)?;

        let aspect_ratio = args.aspect_ratio.clone();
        let image_size = args.image_size.clone();
        let images = resolve_image_inputs(args.images).await?;
        let req = GenerateRequest {
            model: args.model.clone(),
            prompt: args.prompt,
            aspect_ratio: args.aspect_ratio,
            image_size: args.image_size,
            seed: args.seed,
            images,
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
            quality: args.quality,
            output_format: args.output_format,
            background: args.background,
            // The schema range is advisory only (rmcp does not validate), so
            // clamp here the way variants/wait_seconds already do.
            output_compression: args.output_compression.map(|c| c.min(100)),
        };

        let variants = args.variants.unwrap_or(1).clamp(1, 16);
        let wait = args
            .wait_seconds
            .unwrap_or(DEFAULT_WAIT_SECONDS)
            .clamp(1, 60);
        let mut config: Vec<&str> = Vec::new();
        if let Some(a) = &aspect_ratio {
            config.push(a);
        }
        if let Some(s) = &image_size {
            config.push(s);
        }
        let base = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Image,
            &args.model,
            &config,
            args.seed,
        );
        let model = args.model;
        let variants_u64 = variants as u64;

        self.spawn_job_and_wait(
            TaskKind::Image,
            wait,
            inline_previews,
            move |ctx| async move {
                match image_gen::run_job(&ctx.client, &req, variants, &base, "inline").await {
                    Ok(summary) if !summary.images.is_empty() => {
                        let images = summary.images.len() as u64;
                        let cost: f64 = summary.images.iter().filter_map(|i| i.cost).sum();
                        let unknown =
                            summary.images.iter().filter(|i| i.cost.is_none()).count() as u64;
                        ctx.stats
                            .record_job(&model, variants_u64, images, cost, unknown)
                            .await;
                        Ok(image_job_result_json(&summary, &aspect_ratio, &image_size))
                    }
                    Ok(summary) => {
                        ctx.stats.record_job(&model, variants_u64, 0, 0.0, 0).await;
                        Err(format!(
                            "all {variants} variant(s) failed: {}",
                            summary.errors.join("; ")
                        ))
                    }
                    Err(e) => {
                        ctx.stats.record_job(&model, variants_u64, 0, 0.0, 0).await;
                        Err(format!("{e:#}"))
                    }
                }
            },
        )
        .await
    }

    #[tool(
        description = "Describe or answer a question about image(s) using a vision-capable \
        model (image input, text output, e.g. google/gemini-2.5-flash, anthropic/claude-sonnet-4.6, \
        or openai/gpt-5.4). Pass one or more images (each a local path, an http(s) url, or \
        base64/data-URL) and an optional prompt/question (defaults to a detailed description); \
        returns the model's text. Images are downscaled before sending. \
        To create or edit an image instead, use generate_image.",
        annotations(
            title = "Describe Image",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn describe_image(
        &self,
        Parameters(args): Parameters<DescribeImageArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        if args.images.is_empty() {
            return Err(ErrorData::invalid_params(
                "describe_image requires at least one image".to_string(),
                None,
            ));
        }
        let model = args.model.clone();
        let req = image_gen::DescribeRequest {
            model: args.model,
            prompt: args
                .prompt
                .unwrap_or_else(|| "Describe this image in detail.".to_string()),
            images: resolve_image_inputs(args.images).await?,
            max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
            reasoning_effort: args.reasoning_effort,
        };
        match image_gen::describe_image(&self.client, &req).await {
            Ok(result) => {
                self.stats.record_text(&model, true, result.cost).await;
                Ok(CallToolResult::success(vec![ContentBlock::text(
                    result.text,
                )]))
            }
            Err(e) => {
                self.stats.record_text(&model, false, None).await;
                Err(ErrorData::internal_error(e.to_string(), None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_gen::ImageSource;
    use crate::server::test_support::{server_for, tool_result_json, valid_png_b64};
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn img_input(path: Option<&str>, url: Option<&str>, base64: Option<&str>) -> ImageInput {
        ImageInput {
            path: path.map(str::to_string),
            url: url.map(str::to_string),
            base64: base64.map(str::to_string),
            label: None,
        }
    }

    #[tokio::test]
    async fn resolve_image_input_decodes_base64_and_data_url() {
        // Raw base64 -> inline bytes.
        let resolved = resolve_image_input(img_input(None, None, Some(&valid_png_b64())))
            .await
            .unwrap();
        match resolved.source {
            ImageSource::Inline { bytes, .. } => assert!(!bytes.is_empty()),
            _ => panic!("expected inline bytes from base64"),
        }

        // A full data: URL also decodes to inline bytes.
        let data_url = format!("data:image/png;base64,{}", valid_png_b64());
        let resolved = resolve_image_input(img_input(None, None, Some(&data_url)))
            .await
            .unwrap();
        assert!(matches!(resolved.source, ImageSource::Inline { .. }));
    }

    #[tokio::test]
    async fn resolve_image_input_keeps_path_and_rejects_bad_input() {
        let resolved = resolve_image_input(img_input(Some("/tmp/a.png"), None, None))
            .await
            .unwrap();
        assert!(matches!(resolved.source, ImageSource::Path(_)));

        // No source -> error.
        let err = resolve_image_input(img_input(None, None, None))
            .await
            .unwrap_err();
        assert!(err.message.contains("exactly one of"));

        // Two sources -> error.
        let err = resolve_image_input(img_input(Some("/tmp/a.png"), None, Some("x")))
            .await
            .unwrap_err();
        assert!(err.message.contains("exactly one of"));

        // Non-http url -> rejected (never sent anywhere).
        let err = resolve_image_input(img_input(None, Some("file:///etc/passwd"), None))
            .await
            .unwrap_err();
        assert!(err.message.contains("http"));
    }

    #[test]
    fn is_blocked_ip_blocks_internal_allows_public() {
        use std::net::{Ipv4Addr, Ipv6Addr};
        // Blocked: loopback, private, link-local (incl. cloud metadata), CGNAT.
        assert!(is_blocked_ip(Ipv4Addr::new(127, 0, 0, 1).into()));
        assert!(is_blocked_ip(Ipv4Addr::new(10, 0, 0, 5).into()));
        assert!(is_blocked_ip(Ipv4Addr::new(192, 168, 1, 1).into()));
        assert!(is_blocked_ip(Ipv4Addr::new(172, 16, 0, 1).into()));
        assert!(is_blocked_ip(Ipv4Addr::new(169, 254, 169, 254).into())); // metadata
        assert!(is_blocked_ip(Ipv4Addr::new(100, 64, 0, 1).into())); // CGNAT
        assert!(is_blocked_ip(Ipv6Addr::LOCALHOST.into()));
        // Allowed: public addresses.
        assert!(!is_blocked_ip(Ipv4Addr::new(8, 8, 8, 8).into()));
        assert!(!is_blocked_ip(Ipv4Addr::new(1, 1, 1, 1).into()));
    }

    #[tokio::test]
    async fn fetch_url_refuses_loopback_and_metadata_targets() {
        // SSRF guard: a loopback URL is refused before any connection.
        let err = resolve_image_input(img_input(None, Some("http://127.0.0.1:9/pic.png"), None))
            .await
            .unwrap_err();
        assert!(err.message.contains("private/loopback"));

        // The cloud metadata endpoint is link-local and likewise refused.
        let err = resolve_image_input(img_input(None, Some("http://169.254.169.254/latest"), None))
            .await
            .unwrap_err();
        assert!(err.message.contains("private/loopback"));
    }

    #[tokio::test]
    async fn generate_image_runs_async_and_get_result_fetches_it() {
        let mock = MockServer::start().await;
        // The Images API returns raw base64 bytes in data[].b64_json.
        Mock::given(method("POST"))
            .and(path("/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "created": 1748372400,
                "data": [{ "b64_json": valid_png_b64() }],
                "usage": { "cost": 0.04 }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-async-test.png");
        let args = GenerateImageArgs {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: Some("1:1".to_string()),
            image_size: Some("1K".to_string()),
            seed: Some(5),
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: Some(30),
            output: Some(out.to_string_lossy().into_owned()),
            quality: None,
            output_format: None,
            background: None,
            output_compression: None,
        };
        // Fast mock completes within the wait window -> inline completed result.
        // inline_previews=true mirrors a Claude Desktop client.
        let res = server.run_generate(args, true).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["status"], "completed");
        assert_eq!(v["kind"], "image");
        assert!(v["images"][0]["path"].is_string());
        let task_id = v["task_id"].as_str().unwrap().to_string();

        // The completed result also carries an inline image preview block so
        // the client renders the generated image, not just its path.
        let full = serde_json::to_value(&res).unwrap();
        let img_block = full["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["type"] == "image")
            .expect("an image content block is present");
        assert_eq!(img_block["mimeType"], "image/png");
        assert!(!img_block["data"].as_str().unwrap().is_empty());

        // The same task is retrievable by id, also with an inline preview.
        let res2 = server.run_get_result(task_id.clone(), true).await.unwrap();
        let v2 = tool_result_json(&res2);
        assert_eq!(v2["status"], "completed");
        assert_eq!(v2["task_id"], task_id);
        let full2 = serde_json::to_value(&res2).unwrap();
        assert!(
            full2["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["type"] == "image"),
            "get_result also returns the inline preview"
        );

        // A CLI-style client (inline_previews=false) gets paths only, no image block.
        let res_cli = server.run_get_result(task_id.clone(), false).await.unwrap();
        let full_cli = serde_json::to_value(&res_cli).unwrap();
        assert!(
            !full_cli["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["type"] == "image"),
            "no inline preview when the client doesn't want it"
        );
    }

    #[tokio::test]
    async fn generate_image_requires_explicit_parameters() {
        // Validation runs before any HTTP call, so the base URL is never used.
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = GenerateImageArgs {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: None,
            image_size: None,
            seed: None,
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: None,
            output: Some("out.png".to_string()),
            quality: None,
            output_format: None,
            background: None,
            output_compression: None,
        };
        let err = server.run_generate(args, true).await.unwrap_err();
        assert!(err.message.contains("aspect_ratio"));
        assert!(err.message.contains("image_size"));
        assert!(err.message.contains("no defaults"));
    }

    #[tokio::test]
    async fn generate_image_treats_blank_strings_as_missing() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = GenerateImageArgs {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: Some("  ".to_string()),
            image_size: Some("".to_string()),
            seed: None,
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: None,
            output: Some("out.png".to_string()),
            quality: None,
            output_format: None,
            background: None,
            output_compression: None,
        };
        let err = server.run_generate(args, true).await.unwrap_err();
        assert!(err.message.contains("aspect_ratio"), "got: {}", err.message);
        assert!(err.message.contains("image_size"), "got: {}", err.message);
    }

    #[tokio::test]
    async fn generate_image_forwards_quality_format_background_compression() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/images"))
            .and(wiremock::matchers::body_partial_json(json!({
                "quality": "medium",
                "output_format": "webp",
                "background": "opaque",
                "output_compression": 50
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{ "b64_json": valid_png_b64(), "media_type": "image/webp" }]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-quality-test.png");
        let args = GenerateImageArgs {
            model: "m".to_string(),
            prompt: "p".to_string(),
            aspect_ratio: Some("1:1".to_string()),
            image_size: Some("1K".to_string()),
            seed: None,
            images: vec![],
            max_image_dimension: None,
            variants: None,
            wait_seconds: Some(30),
            output: Some(out.to_string_lossy().into_owned()),
            quality: Some("medium".to_string()),
            output_format: Some("webp".to_string()),
            background: Some("opaque".to_string()),
            output_compression: Some(50),
        };
        let res = server.run_generate(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["status"], "completed");
        // The bytes are actually PNG (valid_png_b64) despite the provider
        // declaring "image/webp": sniffing wins, so the file is saved as .png,
        // and the mismatch is surfaced as a warning rather than silently trusted.
        assert!(
            v["images"][0]["path"].as_str().unwrap().ends_with(".png"),
            "got: {v}"
        );
        let warnings = v["warnings"].as_array().expect("a warning is present");
        assert!(
            warnings
                .iter()
                .any(|w| w.as_str().unwrap().contains("image/webp")
                    && w.as_str().unwrap().contains("image/png")),
            "got: {warnings:?}"
        );
    }

    /// Defense in depth: even with a scalar schema, clients that stringify all
    /// arguments must still deserialize (the exact failure mode from the bug report).
    #[test]
    fn generate_image_args_accept_stringified_scalars() {
        let args: GenerateImageArgs = serde_json::from_value(json!({
            "model": "x-ai/grok-imagine-image-quality",
            "prompt": "a small test image",
            "aspect_ratio": "1:1",
            "image_size": "1K",
            "seed": "42",
            "variants": "2",
            "output": "out.png",
        }))
        .expect("stringified scalars should deserialize");
        assert_eq!(args.seed, Some(42));
        assert_eq!(args.variants, Some(2));
    }

    /// Native typed values and absent/null optionals still work unchanged.
    #[test]
    fn generate_image_args_accept_native_and_absent_scalars() {
        let native: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "seed": 7, "variants": 3, "output": "o.png",
        }))
        .unwrap();
        assert_eq!(native.seed, Some(7));
        assert_eq!(native.variants, Some(3));

        let absent: GenerateImageArgs = serde_json::from_value(json!({
            "model": "m", "prompt": "p", "seed": null, "output": "o.png",
        }))
        .unwrap();
        assert_eq!(absent.seed, None);
        assert_eq!(absent.variants, None);
    }

    /// Garbage strings are rejected with a clear message rather than silently
    /// coerced.
    #[test]
    fn invalid_stringified_scalars_are_rejected() {
        let err = serde_json::from_value::<GenerateImageArgs>(json!({
            "model": "m", "prompt": "p", "seed": "not-a-number", "output": "o.png",
        }))
        .unwrap_err();
        assert!(err.to_string().contains("integer"), "got: {err}");
    }
}
