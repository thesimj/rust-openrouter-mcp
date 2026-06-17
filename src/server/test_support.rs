//! Shared test helpers used by the per-domain tool test modules.

use base64::Engine;
use rmcp::model::CallToolResult;

use crate::openrouter::OpenRouterClient;

use super::OpenRouterServer;

/// Build a server whose client talks to the given mock OpenRouter base URL.
pub(crate) fn server_for(uri: String) -> OpenRouterServer {
    OpenRouterServer::new(OpenRouterClient::with_base_url(uri, "test-key"))
}

/// Extract the JSON the tool wrote into its text content block.
pub(crate) fn tool_result_json(res: &CallToolResult) -> serde_json::Value {
    let v = serde_json::to_value(res).unwrap();
    let text = v["content"][0]["text"].as_str().expect("text content");
    serde_json::from_str(text).unwrap()
}

/// Base64 of a genuinely decodable 2x2 PNG, used wherever a test needs an
/// image the preview path can decode + re-encode (it stands in for the valid
/// images real providers return).
pub(crate) fn valid_png_b64() -> String {
    let img = image::RgbaImage::from_pixel(2, 2, image::Rgba([10, 120, 200, 255]));
    let mut buf = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut buf, image::ImageFormat::Png)
        .unwrap();
    base64::engine::general_purpose::STANDARD.encode(buf.into_inner())
}
