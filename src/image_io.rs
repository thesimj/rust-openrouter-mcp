//! Image byte helpers: map MIME types to file extensions, read image
//! dimensions, and normalize inputs for upload. The base64 and `data:` URL
//! codec they share with audio and file inputs is [`crate::base64_codec`].
//!
//! The output format is provider-chosen and not stable (the same model has
//! returned both JPEG and PNG for identical requests), so the format is always
//! sniffed from the response rather than assumed.

use anyhow::{Context, Result, bail};

/// Decode an input image (png/jpeg/webp/gif), downscale so its longest side is
/// at most `max_side` (aspect preserved), and re-encode as PNG bytes. PNG gives
/// one predictable internal format and preserves transparency; downscaling cuts
/// request size, cost, and context pressure. This is our default, not an
/// OpenRouter requirement.
pub fn normalize_to_png(bytes: &[u8], max_side: u32) -> Result<Vec<u8>> {
    let mut out = std::io::Cursor::new(Vec::new());
    decode_and_fit(bytes, max_side)?
        .write_to(&mut out, image::ImageFormat::Png)
        .context("could not encode normalized PNG")?;
    Ok(out.into_inner())
}

/// JPEG quality for the re-encode. 85 is the usual "no visible loss" point and
/// keeps small text legible. Raise it if OCR on dense scans ever suffers.
const JPEG_QUALITY: u8 = 85;

/// Decode and downscale like [`normalize_to_png`], but re-encode as JPEG and
/// return the bytes with their MIME type. Images with an alpha channel stay PNG,
/// because JPEG has no transparency.
///
/// JPEG is roughly 4x smaller than PNG for photographic input at the same pixel
/// count. Providers bill image input per pixel, not per byte, so this cuts
/// upload time and memory without changing cost or resolution.
pub fn normalize_for_send(bytes: &[u8], max_side: u32) -> Result<(Vec<u8>, &'static str)> {
    let img = decode_and_fit(bytes, max_side)?;
    let mut out = std::io::Cursor::new(Vec::new());
    if img.color().has_alpha() {
        img.write_to(&mut out, image::ImageFormat::Png)
            .context("could not encode normalized PNG")?;
        return Ok((out.into_inner(), "image/png"));
    }
    img.write_with_encoder(image::codecs::jpeg::JpegEncoder::new_with_quality(
        &mut out,
        JPEG_QUALITY,
    ))
    .context("could not encode normalized JPEG")?;
    Ok((out.into_inner(), "image/jpeg"))
}

/// Decode an image and downscale it to fit `max_side` on its longest side.
/// Images already within the cap are returned untouched - never upscaled.
///
/// A decompression bomb (a small file that decodes to an enormous pixel buffer)
/// is already refused here: `load_from_memory` applies `image::Limits::default()`,
/// which caps decode allocation at 512 MB. Do not re-add that limit by hand.
fn decode_and_fit(bytes: &[u8], max_side: u32) -> Result<image::DynamicImage> {
    let img = image::load_from_memory(bytes).context("could not decode input image")?;
    Ok(if img.width() > max_side || img.height() > max_side {
        img.resize(max_side, max_side, image::imageops::FilterType::Lanczos3)
    } else {
        img
    })
}

/// Heuristically detect an SVG document from its leading bytes (the `image`
/// crate can't decode SVG, so these inputs are routed to [`svg_to_png`] instead).
/// Skips a UTF-8 BOM and leading whitespace, then looks for an `<svg` root -
/// directly, or after an `<?xml ...?>` declaration within the first chunk.
pub fn is_svg(bytes: &[u8]) -> bool {
    let head = bytes.get(..1024).unwrap_or(bytes);
    let head = head.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(head);
    let text = String::from_utf8_lossy(head);
    let trimmed = text.trim_start();
    trimmed.starts_with("<svg")
        || (trimmed.starts_with("<?xml") && text.contains("<svg"))
        || (trimmed.starts_with("<!--") && text.contains("<svg"))
}

/// A rasterized SVG: PNG bytes plus the source's intrinsic (viewBox) size and
/// whether it contains `<text>` (which is not rendered - no fonts are loaded).
pub struct RasterizedSvg {
    pub png: Vec<u8>,
    pub intrinsic_width: u32,
    pub intrinsic_height: u32,
    pub has_text: bool,
}

/// Rasterize an SVG to PNG, scaling so its longest side is exactly `max_side`
/// (fit-to-cap: vector upscaling is lossless, so small icons render crisp at the
/// cap rather than at their tiny intrinsic size). The pixmap is bounded by
/// `max_side` on both axes by construction, so a hostile `width`/`viewBox` can't
/// trigger a huge allocation. No fonts are loaded (text is skipped) and no
/// external image paths are resolved (the string resolver rejects them).
pub fn svg_to_png(bytes: &[u8], max_side: u32) -> Result<RasterizedSvg> {
    use resvg::{tiny_skia, usvg};

    let opt = svg_options();
    let tree = usvg::Tree::from_data(bytes, &opt).context("could not parse SVG")?;
    let Some((w, h)) = usable_size(&tree) else {
        let size = tree.size();
        bail!(
            "SVG has a degenerate size ({}x{})",
            size.width(),
            size.height()
        );
    };

    let scale = f64::from(max_side) / f64::from(w.max(h));
    let out_w = ((f64::from(w) * scale).round() as u32).max(1);
    let out_h = ((f64::from(h) * scale).round() as u32).max(1);
    let mut pixmap = tiny_skia::Pixmap::new(out_w, out_h)
        .with_context(|| format!("could not allocate a {out_w}x{out_h} pixmap for the SVG"))?;
    #[allow(clippy::cast_possible_truncation)]
    let transform = tiny_skia::Transform::from_scale(scale as f32, scale as f32);
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    let png = pixmap
        .encode_png()
        .context("could not encode the rasterized SVG as PNG")?;

    Ok(RasterizedSvg {
        png,
        intrinsic_width: w.round() as u32,
        intrinsic_height: h.round() as u32,
        has_text: bytes
            .windows(5)
            .any(|win| win.eq_ignore_ascii_case(b"<text")),
    })
}

/// File extension for an image MIME type. Falls back to `bin` for unknowns.
pub fn extension_for(mime: &str) -> &'static str {
    match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/svg+xml" => "svg",
        _ => "bin",
    }
}

/// Sniff a raster image's MIME type from its magic bytes. Returns `None` for
/// formats the `image` crate can't identify (e.g. SVG), so the caller can fall
/// back to a response-supplied `media_type` or a default.
pub fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    match image::guess_format(bytes).ok()? {
        image::ImageFormat::Png => Some("image/png"),
        image::ImageFormat::Jpeg => Some("image/jpeg"),
        image::ImageFormat::WebP => Some("image/webp"),
        image::ImageFormat::Gif => Some("image/gif"),
        _ => None,
    }
}

/// Intrinsic (viewBox) pixel size of an SVG document, if it parses. Used to
/// record dimensions for vector outputs (e.g. Recraft), which the raster
/// [`decode_dimensions`] cannot read.
pub fn svg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    use resvg::usvg;
    let tree = usvg::Tree::from_data(bytes, &svg_options()).ok()?;
    let (w, h) = usable_size(&tree)?;
    Some((w.round() as u32, h.round() as u32))
}

/// An SVG document's size, when it is finite and positive on both axes.
fn usable_size(tree: &resvg::usvg::Tree) -> Option<(f32, f32)> {
    let size = tree.size();
    let (w, h) = (size.width(), size.height());
    (w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0).then_some((w, h))
}

/// Apply the same resource policy to top-level and embedded SVG documents.
fn svg_options() -> resvg::usvg::Options<'static> {
    resvg::usvg::Options {
        image_href_resolver: resvg::usvg::ImageHrefResolver {
            resolve_string: Box::new(|_, _| None),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Decode the pixel dimensions of an encoded image, auto-detecting the format
/// (do not assume PNG - the format varies per provider/response).
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

/// OpenRouter's image `resolution` tiers (`512`, `768`, `1K`, `2K`, `4K` in
/// its enum) with their longest side in pixels. `512` is labelled `0.5K`,
/// the spelling the tools also accept.
const SIZE_TIERS: [(&str, u32); 5] = [
    ("0.5K", 512),
    ("768", 768),
    ("1K", 1024),
    ("2K", 2048),
    ("4K", 4096),
];

/// Nearest standard resolution tier (see [`SIZE_TIERS`]) for the longest side.
pub fn classify_image_size(longest_side: u32) -> &'static str {
    SIZE_TIERS
        .iter()
        .min_by_key(|(_, px)| px.abs_diff(longest_side))
        .map(|(tier, _)| *tier)
        .unwrap_or("1K")
}

/// The longest side of a requested tier, under either spelling of 512
/// (`"512"` on the wire, `"0.5K"`); `None` for a value that is not a tier.
fn tier_pixels(tier: &str) -> Option<u32> {
    let tier = tier.trim();
    if tier == "512" {
        return Some(512);
    }
    SIZE_TIERS
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(tier))
        .map(|(_, px)| *px)
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

    if let Some(req) = requested_aspect
        && aspect_matches(req, width, height) == Some(false)
    {
        warnings.push(format!(
            "requested aspect_ratio {req} but image is {actual_aspect_ratio} ({width}x{height})"
        ));
    }
    if let Some(req) = requested_size
        && tier_pixels(req) != tier_pixels(actual_image_size)
    {
        warnings.push(format!(
            "requested image_size {req} but image is ~{actual_image_size} ({}px)",
            width.max(height)
        ));
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
    use base64::Engine;

    // A 1x1 transparent PNG.
    const PNG_1X1_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

    const SVG_200X100: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 200 100"><rect width="200" height="100" fill="#1e50a0"/></svg>"##;

    #[test]
    fn is_svg_detects_svg_and_rejects_raster() {
        assert!(is_svg(SVG_200X100.as_bytes()));
        assert!(is_svg(b"  \n  <svg xmlns='...'></svg>"));
        assert!(is_svg(
            br#"<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"></svg>"#
        ));
        let png = base64::engine::general_purpose::STANDARD
            .decode(PNG_1X1_B64)
            .unwrap();
        assert!(!is_svg(&png));
        assert!(!is_svg(b"just some text"));
    }

    #[test]
    fn svg_to_png_fits_longest_side_to_cap_and_reports_intrinsic_size() {
        // 200x100 source, cap 800 -> longest side scaled up to 800 -> 800x400.
        let r = svg_to_png(SVG_200X100.as_bytes(), 800).unwrap();
        assert_eq!(&r.png[1..4], b"PNG");
        assert_eq!(decode_dimensions(&r.png).unwrap(), (800, 400));
        assert_eq!((r.intrinsic_width, r.intrinsic_height), (200, 100));
        assert!(!r.has_text);
    }

    #[test]
    fn svg_to_png_flags_text() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="50" height="20"><text x="0" y="10">hi</text></svg>"#;
        let r = svg_to_png(svg.as_bytes(), 800).unwrap();
        assert!(r.has_text);
    }

    #[test]
    fn svg_to_png_rejects_invalid_svg() {
        assert!(svg_to_png(b"<svg>not closed", 800).is_err());
    }

    #[test]
    fn extension_for_maps_known_types() {
        assert_eq!(extension_for("image/png"), "png");
        assert_eq!(extension_for("image/jpeg"), "jpg");
        assert_eq!(extension_for("image/webp"), "webp");
        assert_eq!(extension_for("image/gif"), "gif");
        assert_eq!(extension_for("image/svg+xml"), "svg");
        assert_eq!(extension_for("application/octet-stream"), "bin");
    }

    #[test]
    fn sniff_mime_identifies_raster_and_skips_svg() {
        let png = crate::base64_codec::decode_base64(PNG_1X1_B64).unwrap();
        assert_eq!(sniff_mime(&png), Some("image/png"));
        // SVG is not a raster format the `image` crate recognizes.
        assert_eq!(sniff_mime(SVG_200X100.as_bytes()), None);
    }

    #[test]
    fn svg_dimensions_reads_viewbox_and_rejects_raster() {
        assert_eq!(svg_dimensions(SVG_200X100.as_bytes()), Some((200, 100)));
        let png = crate::base64_codec::decode_base64(PNG_1X1_B64).unwrap();
        assert_eq!(svg_dimensions(&png), None);
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

    /// Every tier in OpenRouter's `resolution` enum (512, 768, 1K, 2K, 4K),
    /// under either of its spellings, matches an image of that size.
    #[test]
    fn check_dimensions_accepts_every_tier_spelling() {
        for (side, requested) in [
            (512, "512"),
            (512, "0.5K"),
            (768, "768"),
            (1024, "1k"),
            (2048, "2K"),
            (4096, "4K"),
        ] {
            let check = check_dimensions(side, side, None, Some(requested));
            assert!(
                check.warnings.is_empty(),
                "{requested}: {:?}",
                check.warnings
            );
        }
        assert_eq!(classify_image_size(768), "768");
        assert_eq!(classify_image_size(512), "0.5K");
        let check = check_dimensions(512, 512, None, Some("1K"));
        assert_eq!(check.warnings.len(), 1, "a real mismatch still warns");
    }
}

#[cfg(test)]
mod audit_regression {
    use super::*;
    #[test]
    fn svg_rejects_absolute_relative_and_nested_file_references() {
        let dir = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
        let local = dir.path().join("synthetic.svg");
        let red = r##"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"><rect width="4" height="4" fill="#ff0000"/></svg>"##;
        std::fs::write(&local, red).unwrap();
        let wrap = |href: &str| {
            format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="4" height="4"><image width="4" height="4" xlink:href="{href}"/></svg>"#
            )
        };
        let absolute = wrap(local.to_str().unwrap());
        let relative = wrap(
            local
                .strip_prefix(std::env::current_dir().unwrap())
                .unwrap()
                .to_str()
                .unwrap(),
        );
        let nested = wrap(&crate::base64_codec::data_url(
            absolute.as_bytes(),
            "image/svg+xml",
        ));
        for input in [&absolute, &relative, &nested] {
            assert_eq!(svg_dimensions(input.as_bytes()), Some((4, 4)));
            let png = svg_to_png(input.as_bytes(), 4).unwrap().png;
            let pixels = image::load_from_memory(&png).unwrap().to_rgba8();
            assert!(pixels.pixels().all(|p| p.0[3] == 0));
        }
        // Data references remain usable, including nested vector content.
        let embedded = wrap(&crate::base64_codec::data_url(
            red.as_bytes(),
            "image/svg+xml",
        ));
        let png = svg_to_png(embedded.as_bytes(), 4).unwrap().png;
        assert_eq!(
            image::load_from_memory(&png)
                .unwrap()
                .to_rgba8()
                .get_pixel(2, 2)
                .0,
            [255, 0, 0, 255]
        );
    }
}
