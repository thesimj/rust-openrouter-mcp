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
fn build_content_is_plain_text_without_images() {
    let content = build_content("just text", &[], &[]);
    let v = serde_json::to_value(&content).unwrap();
    assert_eq!(v, serde_json::json!("just text"));
}

#[test]
fn build_content_puts_text_first_then_images() {
    let images = vec![InputImage::from_path(
        temp_png("openrouter-mcp-test-content.png"),
        None,
    )];
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
        InputImage::from_path(
            temp_png("openrouter-mcp-test-bg.png"),
            Some("background".to_string()),
        ),
        InputImage::from_path(
            temp_png("openrouter-mcp-test-fg.png"),
            Some("product".to_string()),
        ),
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
fn resolve_max_dimension_defaults_and_clamps_to_800() {
    // Default when nothing is supplied.
    assert_eq!(resolve_max_dimension(None), 800);
    // Values at or below the ceiling pass through unchanged.
    assert_eq!(resolve_max_dimension(Some(640)), 640);
    assert_eq!(resolve_max_dimension(Some(800)), 800);
    // Anything above the ceiling is clamped down to 800.
    assert_eq!(resolve_max_dimension(Some(1024)), 800);
    assert_eq!(resolve_max_dimension(Some(u32::MAX)), 800);
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
    };
    let img = generate_image(&client, &req).await.unwrap();
    // media_type is trusted over sniffing, and SVG dimensions come from the viewBox.
    assert_eq!(img.mime, "image/svg+xml");
    assert_eq!((img.width, img.height), (120, 60));
}
