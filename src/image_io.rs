//! Image byte helpers: decode the `data:` URLs OpenRouter returns, map MIME
//! types to file extensions, and read image dimensions.
//!
//! The output format is provider-chosen and not stable (the same model has
//! returned both JPEG and PNG for identical requests), so the format is always
//! sniffed from the response rather than assumed.

use anyhow::{Context, Result, bail};
use base64::Engine;

/// Parse a `data:image/<mime>;base64,<data>` URL into `(mime, bytes)`.
pub fn parse_data_url(url: &str) -> Result<(String, Vec<u8>)> {
    let rest = url
        .strip_prefix("data:")
        .context("not a data URL (missing `data:` prefix)")?;
    let (meta, data) = rest
        .split_once(',')
        .context("malformed data URL (missing comma)")?;
    if !meta.contains("base64") {
        bail!("unsupported data URL: not base64-encoded");
    }
    let mime = meta.split(';').next().unwrap_or_default().to_string();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .context("failed to base64-decode data URL")?;
    Ok((mime, bytes))
}

/// Decode an input image (png/jpeg/webp/gif), downscale so its longest side is
/// at most `max_side` (aspect preserved), and re-encode as PNG bytes. PNG gives
/// one predictable internal format and preserves transparency; downscaling cuts
/// request size, cost, and context pressure. This is our default, not an
/// OpenRouter requirement.
pub fn normalize_to_png(bytes: &[u8], max_side: u32) -> Result<Vec<u8>> {
    let img = image::load_from_memory(bytes).context("could not decode input image")?;
    let resized = if img.width() > max_side || img.height() > max_side {
        img.resize(max_side, max_side, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let mut out = std::io::Cursor::new(Vec::new());
    resized
        .write_to(&mut out, image::ImageFormat::Png)
        .context("could not encode normalized PNG")?;
    Ok(out.into_inner())
}

/// Build a `data:image/png;base64,...` URL from PNG bytes (for sending inputs).
pub fn png_data_url(png: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    )
}

/// File extension for an image MIME type. Falls back to `bin` for unknowns.
pub fn extension_for(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        _ => "bin",
    }
}

/// Decode the pixel dimensions of an encoded image, auto-detecting the format
/// (do not assume PNG — the format varies per provider/response).
pub fn decode_dimensions(bytes: &[u8]) -> Result<(u32, u32)> {
    image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .context("could not guess image format")?
        .into_dimensions()
        .context("could not read image dimensions")
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Reduce pixel dimensions to a compact `w:h` ratio (e.g. 2048x2048 -> "1:1").
pub fn aspect_ratio_string(width: u32, height: u32) -> String {
    let g = gcd(width, height);
    if g == 0 {
        return format!("{width}:{height}");
    }
    format!("{}:{}", width / g, height / g)
}

/// Whether `width`/`height` match a requested `"W:H"` ratio within ~4%
/// (the documented per-ratio pixel sizes are not exact, e.g. 16:9 -> 1344x768).
/// `None` if `requested` can't be parsed.
pub fn aspect_matches(requested: &str, width: u32, height: u32) -> Option<bool> {
    let (rw, rh) = requested.split_once(':')?;
    let rw: f64 = rw.trim().parse().ok()?;
    let rh: f64 = rh.trim().parse().ok()?;
    // A degenerate ratio (zero/negative side) is unverifiable, not a mismatch.
    if rw <= 0.0 || rh <= 0.0 || height == 0 {
        return None;
    }
    let requested = rw / rh;
    let actual = f64::from(width) / f64::from(height);
    Some((requested - actual).abs() / requested <= 0.04)
}

/// Nearest standard resolution tier (`0.5K`/`1K`/`2K`/`4K`) for the longest side.
pub fn classify_image_size(longest_side: u32) -> &'static str {
    const TIERS: [(&str, u32); 4] = [("0.5K", 512), ("1K", 1024), ("2K", 2048), ("4K", 4096)];
    TIERS
        .iter()
        .min_by_key(|(_, px)| px.abs_diff(longest_side))
        .map(|(tier, _)| *tier)
        .unwrap_or("1K")
}

/// Result of verifying a generated image's dimensions against the request.
#[derive(Debug)]
pub struct DimensionCheck {
    pub actual_aspect_ratio: String,
    pub actual_image_size: &'static str,
    /// Human-readable mismatches, empty when the output matched the request.
    pub warnings: Vec<String>,
}

/// Verify the decoded `width`/`height` against the requested `aspect_ratio` and
/// `image_size`, reporting the actual values and any mismatch (providers honor
/// these to varying degrees, so this surfaces what really came back).
pub fn check_dimensions(
    width: u32,
    height: u32,
    requested_aspect: Option<&str>,
    requested_size: Option<&str>,
) -> DimensionCheck {
    let actual_aspect_ratio = aspect_ratio_string(width, height);
    let actual_image_size = classify_image_size(width.max(height));
    let mut warnings = Vec::new();

    if let Some(req) = requested_aspect {
        if aspect_matches(req, width, height) == Some(false) {
            warnings.push(format!(
                "requested aspect_ratio {req} but image is {actual_aspect_ratio} ({width}x{height})"
            ));
        }
    }
    if let Some(req) = requested_size {
        if !req.eq_ignore_ascii_case(actual_image_size) {
            warnings.push(format!(
                "requested image_size {req} but image is ~{actual_image_size} ({}px)",
                width.max(height)
            ));
        }
    }

    DimensionCheck {
        actual_aspect_ratio,
        actual_image_size,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 1x1 transparent PNG.
    const PNG_1X1_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

    #[test]
    fn parse_data_url_extracts_mime_and_bytes() {
        let url = format!("data:image/png;base64,{PNG_1X1_B64}");
        let (mime, bytes) = parse_data_url(&url).unwrap();
        assert_eq!(mime, "image/png");
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[1..4], b"PNG");
    }

    #[test]
    fn parse_data_url_rejects_non_data_and_non_base64() {
        assert!(parse_data_url("https://example.com/x.png").is_err());
        assert!(parse_data_url("data:image/png,notbase64").is_err());
    }

    #[test]
    fn extension_for_maps_known_types() {
        assert_eq!(extension_for("image/png"), "png");
        assert_eq!(extension_for("image/jpeg"), "jpg");
        assert_eq!(extension_for("image/webp"), "webp");
        assert_eq!(extension_for("image/gif"), "gif");
        assert_eq!(extension_for("application/octet-stream"), "bin");
    }

    #[test]
    fn decode_dimensions_reads_png() {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(PNG_1X1_B64)
            .unwrap();
        assert_eq!(decode_dimensions(&bytes).unwrap(), (1, 1));
    }

    #[test]
    fn normalize_to_png_downscales_and_reencodes() {
        // A 1000x500 JPEG normalized to max 800 -> 800x400 PNG.
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(1000, 500));
        let mut jpeg = std::io::Cursor::new(Vec::new());
        img.write_to(&mut jpeg, image::ImageFormat::Jpeg).unwrap();
        let png = normalize_to_png(jpeg.get_ref(), 800).unwrap();
        assert_eq!(&png[1..4], b"PNG");
        assert_eq!(decode_dimensions(&png).unwrap(), (800, 400));
    }

    #[test]
    fn normalize_to_png_keeps_images_within_cap() {
        let img = image::DynamicImage::ImageRgba8(image::RgbaImage::new(2, 2));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let png = normalize_to_png(buf.get_ref(), 800).unwrap();
        assert_eq!(decode_dimensions(&png).unwrap(), (2, 2));
    }

    #[test]
    fn png_data_url_has_png_prefix() {
        assert!(png_data_url(&[1, 2, 3]).starts_with("data:image/png;base64,"));
    }

    #[test]
    fn aspect_ratio_string_reduces() {
        assert_eq!(aspect_ratio_string(2048, 2048), "1:1");
        assert_eq!(aspect_ratio_string(1344, 768), "7:4");
        assert_eq!(aspect_ratio_string(1920, 1080), "16:9");
    }

    #[test]
    fn aspect_matches_tolerates_documented_pixel_sizes() {
        assert_eq!(aspect_matches("1:1", 2048, 2048), Some(true));
        // 1344x768 = 1.75 vs 16:9 = 1.778 -> within 4%.
        assert_eq!(aspect_matches("16:9", 1344, 768), Some(true));
        assert_eq!(aspect_matches("1:1", 1024, 512), Some(false));
        assert_eq!(aspect_matches("not-a-ratio", 100, 100), None);
    }

    #[test]
    fn classify_image_size_picks_nearest_tier() {
        assert_eq!(classify_image_size(512), "0.5K");
        assert_eq!(classify_image_size(1024), "1K");
        assert_eq!(classify_image_size(1900), "2K");
        assert_eq!(classify_image_size(4096), "4K");
    }

    #[test]
    fn check_dimensions_flags_size_override_but_not_matching_aspect() {
        // Requested 1:1 / 1K, model produced 2048^2 (Seedream behavior).
        let check = check_dimensions(2048, 2048, Some("1:1"), Some("1K"));
        assert_eq!(check.actual_aspect_ratio, "1:1");
        assert_eq!(check.actual_image_size, "2K");
        assert_eq!(check.warnings.len(), 1);
        assert!(check.warnings[0].contains("image_size"));
    }

    #[test]
    fn check_dimensions_clean_when_request_honored() {
        let check = check_dimensions(1024, 1024, Some("1:1"), Some("1K"));
        assert!(check.warnings.is_empty());
    }
}
