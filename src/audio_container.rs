//! Audio container typing shared by every path that saves or sends audio
//! bytes: music output, speech output, and audio inputs. The bytes decide
//! first; a declared or requested format only fills in when they cannot.

/// `(mime, extension)` read from an audio container's magic bytes. Only
/// signatures long enough to be unambiguous live here; a bare MPEG frame is
/// [`looks_like_mpeg_frame`], which callers apply with more care.
pub(crate) fn sniff(bytes: &[u8]) -> Option<(&'static str, &'static str)> {
    if bytes.starts_with(b"ID3") {
        Some(("audio/mpeg", "mp3"))
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE") {
        Some(("audio/wav", "wav"))
    } else if bytes.starts_with(b"fLaC") {
        Some(("audio/flac", "flac"))
    } else if bytes.starts_with(b"OggS") {
        Some(("audio/ogg", "ogg"))
    } else {
        None
    }
}

/// Whether `bytes` open with a plausible MPEG audio frame header (an MP3
/// with no ID3 tag): 11 sync bits, then no reserved version/layer, bitrate,
/// or sample-rate index. Raw PCM has no header and can start with the same
/// bytes, so this is only trusted when PCM was not requested.
pub(crate) fn looks_like_mpeg_frame(bytes: &[u8]) -> bool {
    let [0xFF, second, third, ..] = bytes else {
        return false;
    };
    second & 0xE0 == 0xE0
        && second & 0x18 != 0x08 // version: 01 is reserved
        && second & 0x06 != 0x00 // layer: 00 is reserved
        && third & 0xF0 != 0xF0 // bitrate index 1111 is invalid
        && third & 0x0C != 0x0C // sample-rate index 11 is reserved
}

/// `(mime, extension)` for a declared MIME type we know, under any of the
/// aliases providers send; `None` for opaque types such as
/// `application/octet-stream`.
fn known_mime(mime: &str) -> Option<(&'static str, &'static str)> {
    match mime.trim().to_ascii_lowercase().as_str() {
        "audio/mpeg" | "audio/mp3" => Some(("audio/mpeg", "mp3")),
        "audio/wav" | "audio/x-wav" | "audio/wave" => Some(("audio/wav", "wav")),
        "audio/pcm" | "audio/l16" => Some(("audio/pcm", "pcm")),
        "audio/flac" | "audio/x-flac" => Some(("audio/flac", "flac")),
        "audio/ogg" => Some(("audio/ogg", "ogg")),
        "audio/opus" => Some(("audio/opus", "opus")),
        "audio/aac" => Some(("audio/aac", "aac")),
        "audio/webm" => Some(("audio/webm", "webm")),
        "audio/mp4" | "audio/m4a" | "audio/x-m4a" => Some(("audio/mp4", "m4a")),
        _ => None,
    }
}

/// `(mime, extension)` implied by a requested output format (`audio.format`
/// for music, `response_format` for speech). Defaults to MP3, what both
/// endpoints return when nothing else is asked for.
fn requested_container(format: Option<&str>) -> (&'static str, &'static str) {
    match format {
        Some("wav") => ("audio/wav", "wav"),
        Some("flac") => ("audio/flac", "flac"),
        Some("opus") => ("audio/opus", "opus"),
        Some("pcm16") | Some("pcm") => ("audio/pcm", "pcm"),
        _ => ("audio/mpeg", "mp3"),
    }
}

/// The container to save generated audio as: sniffed from the bytes when
/// recognizable, else the `declared` content type when it is one we know,
/// else the requested format's, else MP3. A bare MPEG frame sync counts as
/// recognizable unless PCM was requested: raw samples carry no header, so a
/// sync pattern there is the audio, not a container.
pub(crate) fn container_for(
    bytes: &[u8],
    declared: Option<&str>,
    requested: Option<&str>,
) -> (&'static str, &'static str) {
    if let Some(container) = sniff(bytes) {
        return container;
    }
    if let Some(container) = declared.and_then(known_mime) {
        return container;
    }
    let requested = requested_container(requested);
    let pcm_requested = requested.1 == "pcm";
    if !pcm_requested && looks_like_mpeg_frame(bytes) {
        return ("audio/mpeg", "mp3");
    }
    requested
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_for_sniffs_known_headers_then_trusts_the_requested_format() {
        assert_eq!(
            container_for(b"ID3\x03\x00", None, None),
            ("audio/mpeg", "mp3")
        );
        assert_eq!(
            container_for(b"\xFF\xFB\x90\x00", None, Some("wav")),
            ("audio/mpeg", "mp3")
        );
        // Raw PCM can open with the same bytes as an MPEG frame sync; when PCM
        // was asked for, the sync pattern is samples, not a container. A real
        // ID3 tag still wins, since a provider may ignore the request.
        assert_eq!(
            container_for(b"\xFF\xFB\x90\x00", None, Some("pcm16")),
            ("audio/pcm", "pcm")
        );
        assert_eq!(
            container_for(b"ID3\x03\x00", None, Some("pcm")),
            ("audio/mpeg", "mp3")
        );
        assert_eq!(
            container_for(b"\xFF\xE8\x90\x00", None, None),
            ("audio/mpeg", "mp3"),
            "default"
        );
        assert_eq!(
            container_for(b"RIFF\x24\x00\x00\x00WAVEfmt ", None, None),
            ("audio/wav", "wav")
        );
        assert_eq!(
            container_for(b"fLaC\x00", None, None),
            ("audio/flac", "flac")
        );
        assert_eq!(container_for(b"OggS\x00", None, None), ("audio/ogg", "ogg"));
        // Unrecognized bytes: the requested format decides, else mp3.
        assert_eq!(
            container_for(b"\x00\x01\x02", None, Some("wav")),
            ("audio/wav", "wav")
        );
        assert_eq!(
            container_for(b"\x00\x01\x02", None, Some("pcm16")),
            ("audio/pcm", "pcm")
        );
        assert_eq!(
            container_for(b"\x00\x01\x02", None, None),
            ("audio/mpeg", "mp3")
        );
        assert_eq!(
            container_for(b"", None, Some("nonsense")),
            ("audio/mpeg", "mp3")
        );
    }

    #[test]
    fn container_for_uses_a_known_declared_type_before_the_requested_one() {
        assert_eq!(
            container_for(b"\x00\x01", Some("audio/aac"), Some("mp3")),
            ("audio/aac", "aac")
        );
        assert_eq!(
            container_for(b"\x00\x01", Some(" Audio/X-WAV "), None),
            ("audio/wav", "wav")
        );
        // Opaque types fall through; the bytes still beat any declaration.
        assert_eq!(
            container_for(b"\x00\x01", Some("application/octet-stream"), Some("pcm")),
            ("audio/pcm", "pcm")
        );
        assert_eq!(
            container_for(b"ID3\x03", Some("audio/wav"), None),
            ("audio/mpeg", "mp3")
        );
    }

    #[test]
    fn looks_like_mpeg_frame_rejects_reserved_header_fields() {
        assert!(looks_like_mpeg_frame(b"\xFF\xFB\x90\x00"));
        assert!(
            !looks_like_mpeg_frame(b"\xFF\xE8\x90\x00"),
            "reserved version"
        );
        assert!(
            !looks_like_mpeg_frame(b"\xFF\xF9\x90\x00"),
            "reserved layer"
        );
        assert!(
            !looks_like_mpeg_frame(b"\xFF\xFB\xF0\x00"),
            "invalid bitrate"
        );
        assert!(!looks_like_mpeg_frame(b"\xFF\xFB\x9C\x00"), "reserved rate");
        assert!(!looks_like_mpeg_frame(b"\xFF\xFB"), "too short");
    }
}
