use base64::Engine;
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
    let content = build_gen_content(&req.prompt, &req.images, &prepared);
    generate_core(client, req, req.seed, &content).await
}

/// Write a small valid PNG to a temp file and return its path.
fn temp_png(name: &str) -> PathBuf {
    temp_png_sized(name, 2, 2)
}

/// Write an opaque PNG of an exact size to a temp file and return its path.
/// Opaque, not RGBA: it stands in for a photo, which is what input images are.
fn temp_png_sized(name: &str, width: u32, height: u32) -> PathBuf {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(width, height));
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

    let images = vec![InputImage::from_path(path, None)];
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
fn assemble_prompt_is_verbatim_without_labels() {
    // Path-only: assemble_prompt reads labels and file names, never the file.
    let images = vec![InputImage::from_path("content.png", None)];
    assert_eq!(assemble_prompt("edit this", &[]), "edit this");
    assert_eq!(assemble_prompt("edit this", &images), "edit this");
}

#[test]
fn assemble_prompt_prepends_label_block_when_labeled() {
    let images = vec![
        InputImage::from_path("bg.png", Some("background".to_string())),
        InputImage::from_path("fg.png", Some("product".to_string())),
    ];
    let text = assemble_prompt("compose them", &images);
    assert!(text.contains("Reference images:"));
    assert!(text.contains("1. background:"));
    assert!(text.contains("2. product:"));
    assert!(text.contains("compose them"));
}

#[test]
fn resolve_max_dimension_defaults_and_clamps_to_ceiling() {
    // Default when nothing is supplied.
    assert_eq!(resolve_max_dimension(None), 1536);
    // Values at or below the ceiling pass through unchanged.
    assert_eq!(resolve_max_dimension(Some(640)), 640);
    // An explicit value above the default raises the cap. This is the point of
    // the argument: 800px destroys dense text and small-print OCR.
    assert_eq!(resolve_max_dimension(Some(2048)), 2048);
    assert_eq!(resolve_max_dimension(Some(4096)), 4096);
    // Anything above the ceiling is clamped down to it.
    assert_eq!(resolve_max_dimension(Some(8192)), 4096);
    assert_eq!(resolve_max_dimension(Some(u32::MAX)), 4096);
    // A zero cap is unusable; clamp it up to a valid minimum.
    assert_eq!(resolve_max_dimension(Some(0)), 1);
}

#[tokio::test]
async fn generate_image_sends_request_and_decodes_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images"))
        // Verify the Images API request shape we build (image_size -> resolution).
        .and(body_partial_json(json!({
            "model": "google/gemini-3.1-flash-image-preview",
            "prompt": "an owl",
            "seed": 1200,
            "resolution": "1K",
            "aspect_ratio": "1:1"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-generation-id", "gen-abc")
                .set_body_json(json!({
                    "created": 1748372400,
                    "data": [{ "b64_json": PNG_1X1_B64 }],
                    "usage": { "cost": 0.0684 }
                })),
        )
        .mount(&server)
        .await;

    let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
    let req = GenerateRequest {
        model: "google/gemini-3.1-flash-image-preview".to_string(),
        prompt: "an owl".to_string(),
        aspect_ratio: Some("1:1".to_string()),
        image_size: Some("1K".to_string()),
        seed: Some(1200),
        images: vec![],
        max_image_dimension: 800,
        quality: None,
        output_format: None,
        background: None,
        output_compression: None,
    };
    let img = generate_image(&client, &req).await.unwrap();
    assert_eq!((img.width, img.height), (1, 1));
    // No media_type in the response -> the PNG magic bytes are sniffed.
    assert_eq!(img.mime, "image/png");
    assert_eq!(img.cost, Some(0.0684));
    assert_eq!(img.generation_id.as_deref(), Some("gen-abc"));
}

#[tokio::test]
async fn generate_image_maps_half_k_resolution() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images"))
        // The 0.5K tier is spelled `512` on the Images API.
        .and(body_partial_json(json!({ "resolution": "512" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "b64_json": PNG_1X1_B64 }]
        })))
        .mount(&server)
        .await;

    let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
    let req = GenerateRequest {
        model: "m".to_string(),
        prompt: "p".to_string(),
        aspect_ratio: None,
        image_size: Some("0.5K".to_string()),
        seed: None,
        images: vec![],
        max_image_dimension: 800,
        quality: None,
        output_format: None,
        background: None,
        output_compression: None,
    };
    assert!(generate_image(&client, &req).await.is_ok());
}

#[tokio::test]
async fn generate_image_surfaces_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images"))
        .respond_with(
            ResponseTemplate::new(500).set_body_string("{\"error\":\"Internal Server Error\"}"),
        )
        .mount(&server)
        .await;

    let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
    let req = GenerateRequest {
        model: "openai/gpt-image-2".to_string(),
        prompt: "p".to_string(),
        aspect_ratio: None,
        image_size: Some("1K".to_string()),
        seed: None,
        images: vec![],
        max_image_dimension: 800,
        quality: None,
        output_format: None,
        background: None,
        output_compression: None,
    };
    let err = generate_image(&client, &req).await.unwrap_err();
    assert!(err.to_string().contains("Internal Server Error"));
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
        images: vec![InputImage::from_path(
            temp_png("openrouter-mcp-test-describe.png"),
            None,
        )],
        max_image_dimension: 800,
        reasoning_effort: None,
    };
    let result = describe_image(&client, &req).await.unwrap();
    assert_eq!(result.text, "A small green lizard.");
    assert_eq!(result.cost, Some(0.002));
}

/// Decode the single `image_url` data URL out of a captured chat request body
/// and return the MIME type and pixel size of the image that was actually sent.
fn sent_image(body: &serde_json::Value) -> (String, (u32, u32)) {
    let url = body["messages"][0]["content"]
        .as_array()
        .expect("multimodal content is an array")
        .iter()
        .find_map(|part| part["image_url"]["url"].as_str())
        .expect("an image part is present");
    let (mime, bytes) = crate::image_io::parse_data_url(url).unwrap();
    (mime, crate::image_io::decode_dimensions(&bytes).unwrap())
}

/// Run one describe call against a mock and hand back the request body it sent.
/// `source` sizes the input image: only the downscaling test needs a large one,
/// and decoding/resizing it is by far the slowest thing in this suite.
async fn captured_describe_body(
    source: (u32, u32),
    max_image_dimension: u32,
    reasoning_effort: Option<&str>,
) -> serde_json::Value {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{ "message": { "content": "ok" } }]
        })))
        .mount(&server)
        .await;

    let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
    let req = DescribeRequest {
        model: "openai/gpt-5.6-terra".to_string(),
        prompt: "What is this?".to_string(),
        images: vec![InputImage::from_path(
            temp_png_sized(
                &format!("openrouter-mcp-test-{}x{}.png", source.0, source.1),
                source.0,
                source.1,
            ),
            None,
        )],
        max_image_dimension,
        reasoning_effort: reasoning_effort.map(str::to_string),
    };
    describe_image(&client, &req).await.unwrap();
    server.received_requests().await.unwrap()[0]
        .body_json()
        .unwrap()
}

/// The resolved cap reaches the wire: a 2000x1000 input is downscaled to the
/// longest-side cap before it is base64'd into the request. `resolve_max_dimension`
/// feeds the first case so a change to the default is caught here too, not just
/// in its own unit test.
#[tokio::test]
async fn describe_image_downscales_input_to_the_resolved_cap() {
    const SRC: (u32, u32) = (2000, 1000);

    let body = captured_describe_body(SRC, resolve_max_dimension(None), None).await;
    assert_eq!(sent_image(&body).1, (1536, 768));

    // An explicit cap above the old 800 ceiling is applied, not clamped down.
    let body = captured_describe_body(SRC, 1024, None).await;
    assert_eq!(sent_image(&body).1, (1024, 512));

    // A cap larger than the source never upscales it.
    let body = captured_describe_body(SRC, 4096, None).await;
    assert_eq!(sent_image(&body).1, SRC);
}

/// Opaque raster input goes out as JPEG, which is ~4x smaller than PNG at the
/// same pixel count. Transparency forces PNG, because JPEG has no alpha channel.
#[tokio::test]
async fn describe_image_sends_jpeg_unless_the_input_has_alpha() {
    let body = captured_describe_body((64, 64), 1536, None).await;
    assert_eq!(sent_image(&body).0, "image/jpeg");

    let opaque = image::DynamicImage::ImageRgb8(image::RgbImage::new(64, 64));
    let transparent = image::DynamicImage::ImageRgba8(image::RgbaImage::new(64, 64));
    for (img, expected) in [(opaque, "image/jpeg"), (transparent, "image/png")] {
        let mut src = std::io::Cursor::new(Vec::new());
        img.write_to(&mut src, image::ImageFormat::Png).unwrap();
        let (bytes, mime) = image_io::normalize_for_send(src.get_ref(), 1536).unwrap();
        assert_eq!(mime, expected);
        // Whatever the format, the bytes must still decode at the same size.
        assert_eq!(image_io::decode_dimensions(&bytes).unwrap(), (64, 64));
    }
}

/// describe_image reaches the reasoning plumbing, and a caller who passes only
/// whitespace gets no `reasoning` object rather than an empty effort string.
/// The effort matrix itself lives in `server::chat::tests` - both tools funnel
/// through the same three lines in `chat_gen::complete`, so it is proved once.
#[tokio::test]
async fn describe_image_forwards_reasoning_effort() {
    const SRC: (u32, u32) = (8, 8);

    let body = captured_describe_body(SRC, 800, Some("high")).await;
    assert_eq!(body["reasoning"]["effort"], "high");

    for blank in [Some(""), Some("  ")] {
        let body = captured_describe_body(SRC, 800, blank).await;
        assert!(body.get("reasoning").is_none(), "{blank:?} sent: {body}");
    }
}

#[tokio::test]
async fn describe_image_requires_an_image() {
    let client = OpenRouterClient::with_base_url("http://127.0.0.1:9", "k");
    let req = DescribeRequest {
        model: "m".to_string(),
        prompt: "p".to_string(),
        images: vec![],
        max_image_dimension: 800,
        reasoning_effort: None,
    };
    assert!(describe_image(&client, &req).await.is_err());
}

#[tokio::test]
async fn generate_image_honors_declared_media_type_for_vector() {
    // A tiny SVG document (a vector model returns bytes + an explicit media_type).
    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="120" height="60" viewBox="0 0 120 60"><rect width="120" height="60"/></svg>"#;
    let svg_b64 = base64::engine::general_purpose::STANDARD.encode(svg);
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "b64_json": svg_b64, "media_type": "image/svg+xml" }]
        })))
        .mount(&server)
        .await;

    let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
    let req = GenerateRequest {
        model: "recraft/recraft-v4.1-vector".to_string(),
        prompt: "p".to_string(),
        aspect_ratio: None,
        image_size: None,
        seed: None,
        images: vec![],
        max_image_dimension: 800,
        quality: None,
        output_format: None,
        background: None,
        output_compression: None,
    };
    let img = generate_image(&client, &req).await.unwrap();
    // media_type is trusted over sniffing, and SVG dimensions come from the viewBox.
    assert_eq!(img.mime, "image/svg+xml");
    assert_eq!((img.width, img.height), (120, 60));
}

/// N3: `image/jpg` is a real-world alias for `image/jpeg`, which is what
/// `sniff_mime` always reports for JPEG bytes - a declared `image/jpg` must not
/// false-positive as a mismatch against the sniffed type.
#[tokio::test]
async fn generate_image_treats_image_jpg_as_an_alias_of_image_jpeg_no_warning() {
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    let jpeg_b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "b64_json": jpeg_b64, "media_type": "image/jpg" }]
        })))
        .mount(&server)
        .await;

    let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
    let req = GenerateRequest {
        model: "m".to_string(),
        prompt: "p".to_string(),
        aspect_ratio: None,
        image_size: None,
        seed: None,
        images: vec![],
        max_image_dimension: 800,
        quality: None,
        output_format: None,
        background: None,
        output_compression: None,
    };
    let img = generate_image(&client, &req).await.unwrap();
    assert_eq!(img.mime, "image/jpeg");
    assert!(img.warnings.is_empty(), "got: {:?}", img.warnings);
}

#[tokio::test]
async fn generate_image_passes_through_quality_format_background_compression() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/images"))
        .and(body_partial_json(json!({
            "quality": "high",
            "output_format": "webp",
            "background": "transparent",
            "output_compression": 80
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{ "b64_json": PNG_1X1_B64 }]
        })))
        .mount(&server)
        .await;

    let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
    let req = GenerateRequest {
        model: "openai/gpt-image-2".to_string(),
        prompt: "p".to_string(),
        aspect_ratio: None,
        image_size: None,
        seed: None,
        images: vec![],
        max_image_dimension: 800,
        quality: Some("high".to_string()),
        output_format: Some("webp".to_string()),
        background: Some("transparent".to_string()),
        output_compression: Some(80),
    };
    assert!(generate_image(&client, &req).await.is_ok());
}

#[test]
fn canonical_mime_normalizes_case_and_aliases() {
    assert_eq!(super::canonical_mime("image/jpg"), "image/jpeg");
    assert_eq!(super::canonical_mime(" image/JPG "), "image/jpeg");
    assert_eq!(super::canonical_mime("IMAGE/JPEG"), "image/jpeg");
    assert_eq!(super::canonical_mime("image/svg"), "image/svg+xml");
    assert_eq!(super::canonical_mime("image/png"), "image/png");
}
