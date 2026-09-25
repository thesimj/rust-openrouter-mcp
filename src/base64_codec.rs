//! The base64 and `data:` URL codec shared by every inline input and output:
//! images, audio and files sent as data URLs, OpenRouter's `b64_json`
//! images, and streamed audio fragments. Decoding is lenient about
//! whitespace and padding, since the bytes, not the encoding style, matter.

use anyhow::{Context, Result, bail};
use base64::Engine;

/// Parse a `data:<mime>;base64,<data>` URL into `(mime, bytes)`.
pub fn parse_data_url(url: &str) -> Result<(String, Vec<u8>)> {
    let (mime, data) = split_data_url(url)?;
    let bytes = decode_base64(data).context("failed to base64-decode data URL")?;
    Ok((mime.to_string(), bytes))
}

/// Split a `data:<mime>[;<param>...];base64,<data>` URL into its MIME type and
/// still-encoded payload. Shared by every inline input (image and audio) so
/// they agree on what counts as a base64 data URL.
pub fn split_data_url(url: &str) -> Result<(&str, &str)> {
    let rest = url
        .trim()
        .strip_prefix("data:")
        .context("not a data URL (missing `data:` prefix)")?;
    let (meta, data) = rest
        .split_once(',')
        .context("malformed data URL (missing comma)")?;
    if !meta
        .split(';')
        .any(|part| part.trim().eq_ignore_ascii_case("base64"))
    {
        bail!("unsupported data URL: not base64-encoded");
    }
    let mime = meta.split(';').next().unwrap_or_default().trim();
    Ok((mime, data))
}

/// Build a `data:<mime>;base64,...` URL from bytes (for sending inputs).
pub fn data_url(bytes: &[u8], mime: &str) -> String {
    format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

/// Decode raw base64 bytes (no `data:` prefix). Lenient on purpose: interior
/// whitespace (line-wrapping encoders such as `base64 file`) is ignored and
/// padding is optional, since inline payloads are often hand-assembled and the
/// bytes, not the encoding style, are what matters.
pub fn decode_base64(data: &str) -> Result<Vec<u8>> {
    LENIENT_BASE64
        .decode(compact_base64(data).as_bytes())
        .context("failed to base64-decode data")
}

/// Longest compacted base64 text that can decode to at most `decoded_limit`
/// bytes, so a payload past a byte cap is refused before it is decoded.
pub fn max_base64_len(decoded_limit: usize) -> usize {
    decoded_limit.div_ceil(3) * 4
}

/// Standard alphabet, whitespace already removed by [`compact_base64`],
/// padding accepted but not required.
const LENIENT_BASE64: base64::engine::GeneralPurpose = base64::engine::GeneralPurpose::new(
    &base64::alphabet::STANDARD,
    base64::engine::general_purpose::PAD
        .with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent),
);

/// Reassemble bytes from base64 fragments that arrive one at a time (a
/// streamed `delta.audio.data`), without assuming how the sender split them.
/// A fragment that ends in `=` padding was encoded on its own and is decoded
/// on its own - padding cannot appear mid-string, so concatenating it with the
/// next fragment would fail. An unpadded fragment may be an arbitrary cut of
/// one long encoding, so only whole 4-character groups are decoded and the
/// remainder waits for the next fragment. Decoding as fragments arrive also
/// keeps the buffered form at the size of the bytes, not 4/3 of it.
#[derive(Debug, Default)]
pub struct Base64Assembler {
    /// Characters not yet decodable: fewer than one whole group.
    pending: Vec<u8>,
    bytes: Vec<u8>,
}

impl Base64Assembler {
    /// Append one fragment (whitespace ignored) and decode what is decodable.
    pub fn push(&mut self, fragment: &str) -> Result<()> {
        let fragment = compact_base64(fragment);
        self.pending.extend_from_slice(fragment.as_bytes());
        // A padded fragment is self-contained once it is whole groups; a cut
        // inside the padding run ("Mg=" then "=") waits for the rest of it.
        let decodable = if fragment.ends_with('=') && self.pending.len().is_multiple_of(4) {
            self.pending.len()
        } else {
            self.pending.len() / 4 * 4
        };
        if decodable > 0 {
            self.decode_pending(decodable)?;
        }
        Ok(())
    }

    /// Decode whatever remains (a final partial group, padding optional) and
    /// return every byte assembled so far.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        if !self.pending.is_empty() {
            self.decode_pending(self.pending.len())?;
        }
        Ok(self.bytes)
    }

    fn decode_pending(&mut self, len: usize) -> Result<()> {
        let decoded = LENIENT_BASE64
            .decode(&self.pending[..len])
            .context("failed to base64-decode data")?;
        self.bytes.extend_from_slice(&decoded);
        self.pending.drain(..len);
        Ok(())
    }
}

/// `data` with ASCII whitespace removed, borrowed when there was none.
pub fn compact_base64(data: &str) -> std::borrow::Cow<'_, str> {
    let data = data.trim();
    if data.bytes().any(|b| b.is_ascii_whitespace()) {
        std::borrow::Cow::Owned(data.chars().filter(|c| !c.is_ascii_whitespace()).collect())
    } else {
        std::borrow::Cow::Borrowed(data)
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
        // Parameter order, case and spacing around the base64 marker vary.
        assert_eq!(
            split_data_url(" data:audio/mp3;charset=x; BASE64,QUJD ").unwrap(),
            ("audio/mp3", "QUJD")
        );
    }

    #[test]
    fn decode_base64_reads_raw_png_bytes() {
        let bytes = decode_base64(PNG_1X1_B64).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
        assert!(decode_base64("!!!not base64!!!").is_err());
        // Line-wrapped and unpadded encodings decode to the same bytes.
        assert_eq!(decode_base64("QUJD\nRA==").unwrap(), b"ABCD");
        assert_eq!(decode_base64("QUJDRA").unwrap(), b"ABCD");
        assert_eq!(compact_base64(" QUJD\r\nRA== "), "QUJDRA==");
        assert!(matches!(
            compact_base64("QUJD"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn base64_assembler_handles_padded_fragments_and_arbitrary_cuts() {
        // Independently encoded fragments, each padded: the concatenation
        // "SUQzAwAAAAAvMg==AAAA" is not valid base64, but the bytes are.
        let mut a = Base64Assembler::default();
        a.push("SUQzAwAAAAAvMg==").unwrap();
        a.push("AAAA").unwrap();
        a.push("/w==").unwrap();
        assert_eq!(
            a.finish().unwrap(),
            b"ID3\x03\x00\x00\x00\x00/2\x00\x00\x00\xff"
        );

        // One long encoding cut at arbitrary (non-group) positions.
        let whole = "SUQzAwAAAAAvMg==";
        for cut in 0..whole.len() {
            let mut a = Base64Assembler::default();
            a.push(&whole[..cut]).unwrap();
            a.push(&whole[cut..]).unwrap();
            assert_eq!(
                a.finish().unwrap(),
                b"ID3\x03\x00\x00\x00\x00/2",
                "cut {cut}"
            );
        }

        // Whitespace and a missing final padding are tolerated; garbage is not.
        let mut a = Base64Assembler::default();
        a.push(" QUJD\n").unwrap();
        a.push("RA").unwrap();
        assert_eq!(a.finish().unwrap(), b"ABCD");
        assert_eq!(Base64Assembler::default().finish().unwrap(), b"");
        assert!(Base64Assembler::default().push("!!!!").is_err());
    }

    #[test]
    fn data_url_has_mime_prefix() {
        assert!(data_url(&[1, 2, 3], "image/png").starts_with("data:image/png;base64,"));
        assert!(data_url(&[1, 2, 3], "image/jpeg").starts_with("data:image/jpeg;base64,"));
    }
}
