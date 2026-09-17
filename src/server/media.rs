//! Shared resolution of media *sources* (a local `path`, an http(s) `url`, or
//! inline `base64`/data-URL) for every tool input that takes one, plus the
//! typed non-image inputs of `chat_completion`: [`FileInput`], [`AudioInput`],
//! [`VideoInput`].
//!
//! Every modality has a different wire part, byte cap, local processing and
//! capability gate, so each keeps its own input struct (one generic
//! `MediaInput` would reintroduce a schema union). What they share is here:
//! the exactly-one-source check, inline decoding with a size cap, and the
//! SSRF-hardened URL fetch. `server::image` builds its `ImageInput` on the
//! same [`resolve_source`], so images behave exactly as before.

use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};

use base64::Engine;
use rmcp::ErrorData;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::openrouter::{FilePart, InputAudio, VideoUrl};
use crate::server::schema::{AtLeastOneOf, scalarize_nullable};

/// Hard ceiling for one inline or fetched media body (20 MiB), the same cap
/// images have in [`crate::resources::MAX_IMAGE_BYTES`]. Bytes are sent as a
/// data URL, so accepting more only increases memory pressure.
pub(crate) const MAX_MEDIA_BYTES: usize = 20 * 1024 * 1024;

/// At most this many entries per input list (each may be [`MAX_MEDIA_BYTES`]).
pub(crate) const MAX_INPUTS_PER_KIND: usize = crate::resources::MAX_IMAGE_INPUTS;

/// Total deadline for one remote fetch, sized against the ceiling above:
/// 20 MB inside 30s is ~5 Mbit/s, slower than any host worth waiting for.
const REMOTE_FETCH_TIMEOUT_SECS: u64 = 30;

/// Which chat input modality a source belongs to. Drives error wording and
/// the per-kind capability gate (`input_modalities` in the models catalog).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputKind {
    Image,
    File,
    Audio,
    Video,
}

impl InputKind {
    /// The catalog `input_modalities` value and the noun used in messages.
    pub(crate) fn noun(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::File => "file",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }

    /// The source fields this kind's input struct actually has.
    fn sources(self) -> &'static str {
        match self {
            Self::Audio => "path or base64",
            _ => "path, url, or base64",
        }
    }
}

/// Where a source ended up after [`resolve_source`].
#[derive(Debug)]
pub(crate) enum Resolved {
    /// A local path, left unread so the consumer can read it lazily (images)
    /// or with its own cap (see [`into_bytes`]).
    Path(PathBuf),
    /// Already-decoded bytes from a base64/data-URL argument or a URL fetch.
    Bytes(MediaBytes),
    /// A URL passed through untouched for the provider to fetch
    /// (`fetch_urls: false`).
    Url(String),
}

/// Decoded bytes plus what is known about them.
#[derive(Debug)]
pub(crate) struct MediaBytes {
    pub bytes: Vec<u8>,
    /// Human-readable origin: the file name, the URL, or `"inline"`.
    pub name: String,
    /// The MIME type a `data:` URL declared, when the input was one.
    pub declared_mime: Option<String>,
}

fn present(s: Option<&str>) -> Option<&str> {
    s.map(str::trim).filter(|s| !s.is_empty())
}

/// Exactly one non-blank source, cheaply and without any network fetch, so a
/// caller can surface a malformed entry before running a network-bound gate.
pub(crate) fn check_exactly_one(
    kind: InputKind,
    path: Option<&str>,
    url: Option<&str>,
    base64: Option<&str>,
) -> Result<(), ErrorData> {
    let count = [path, url, base64]
        .into_iter()
        .filter(|s| present(*s).is_some())
        .count();
    if count != 1 {
        return Err(ErrorData::invalid_params(
            format!(
                "each {} needs exactly one of: {}",
                kind.noun(),
                kind.sources()
            ),
            None,
        ));
    }
    Ok(())
}

/// Resolve one source: a path stays a path, inline data is decoded (capped at
/// `limit` bytes), a URL is fetched (SSRF-guarded, capped at `limit`) when
/// `fetch_urls` is set and passed through otherwise. Requires exactly one source.
pub(crate) async fn resolve_source(
    kind: InputKind,
    path: Option<String>,
    url: Option<String>,
    base64: Option<String>,
    limit: usize,
    fetch_urls: bool,
) -> Result<Resolved, ErrorData> {
    check_exactly_one(kind, path.as_deref(), url.as_deref(), base64.as_deref())?;
    if let Some(p) = path.filter(|s| !s.trim().is_empty()) {
        return Ok(Resolved::Path(PathBuf::from(p.trim())));
    }
    if let Some(b64) = base64.filter(|s| !s.trim().is_empty()) {
        let (declared_mime, bytes) = crate::resources::run_blocking(move || {
            decode_inline(kind, &b64, limit).map_err(|e| anyhow::anyhow!(e.message))
        })
        .await
        .map_err(|e| ErrorData::invalid_params(e.to_string(), None))?;
        return Ok(Resolved::Bytes(MediaBytes {
            bytes,
            name: "inline".to_string(),
            declared_mime,
        }));
    }
    let url = url.map(|u| u.trim().to_string()).unwrap_or_default();
    if !fetch_urls {
        return Ok(Resolved::Url(url));
    }
    let bytes = fetch_url(kind, &url, limit).await?;
    Ok(Resolved::Bytes(MediaBytes {
        bytes,
        name: url,
        declared_mime: None,
    }))
}

/// Materialize a [`Resolved`] as bytes: a path is read (off the runtime,
/// capped at `limit`), bytes pass through. A pass-through URL has no bytes,
/// so asking for them is a caller bug.
pub(crate) async fn into_bytes(
    kind: InputKind,
    resolved: Resolved,
    limit: usize,
) -> Result<MediaBytes, ErrorData> {
    match resolved {
        Resolved::Bytes(b) => Ok(b),
        Resolved::Path(p) => {
            let name = file_name(&p);
            let bytes = crate::resources::run_blocking(move || {
                crate::resources::read_file_limited(&p, limit)
            })
            .await
            .map_err(|e| {
                ErrorData::invalid_params(
                    format!("could not read {} {name}: {e:#}", kind.noun()),
                    None,
                )
            })?;
            Ok(MediaBytes {
                bytes,
                name,
                declared_mime: None,
            })
        }
        Resolved::Url(_) => Err(ErrorData::internal_error(
            format!("{} url was not fetched", kind.noun()),
            None,
        )),
    }
}

/// The last path segment (or the whole string when there is none).
fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string_lossy().into_owned())
}

/// Decode an inline `base64`/data-URL argument to `(declared mime, bytes)`,
/// refusing anything past `limit` before and after decoding.
fn decode_inline(
    kind: InputKind,
    data: &str,
    limit: usize,
) -> Result<(Option<String>, Vec<u8>), ErrorData> {
    let data = data.trim();
    let too_large = || {
        ErrorData::invalid_params(
            format!(
                "inline {} exceeds {} MiB",
                kind.noun(),
                limit / (1024 * 1024)
            ),
            None,
        )
    };
    let payload = if data.starts_with("data:") {
        data.split_once(',').map(|(_, body)| body).unwrap_or(data)
    } else {
        data
    };
    if payload.len() > limit.div_ceil(3) * 4 {
        return Err(too_large());
    }
    let (mime, bytes) = if data.starts_with("data:") {
        crate::image_io::parse_data_url(data)
            .map(|(mime, bytes)| ((!mime.is_empty()).then_some(mime), bytes))
            .map_err(|e| ErrorData::invalid_params(format!("invalid data URL: {e}"), None))
    } else {
        base64::engine::general_purpose::STANDARD
            .decode(data)
            .map(|bytes| (None, bytes))
            .map_err(|e| {
                ErrorData::invalid_params(format!("invalid base64 {} data: {e}", kind.noun()), None)
            })
    }?;
    if bytes.len() > limit {
        return Err(too_large());
    }
    Ok((mime, bytes))
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
                || v4.is_multicast()
                || o[0] == 0
                || o[0] >= 240
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

/// Fetch a URL's bytes with a plain client. Deliberately does NOT use the
/// OpenRouter-authenticated client, so the API key is never sent to a
/// third-party URL. SSRF-hardened: only http/https; the host is resolved and
/// rejected if it points at a private/loopback/link-local address; redirects are
/// disabled; and the connection is pinned to the validated IP so DNS can't be
/// rebound between the check and the request. The body is capped at `limit`.
async fn fetch_url(kind: InputKind, url: &str, limit: usize) -> Result<Vec<u8>, ErrorData> {
    let noun = kind.noun();
    let limit = limit as u64;
    let invalid = |msg: String| ErrorData::invalid_params(msg, None);

    let parsed =
        reqwest::Url::parse(url).map_err(|e| invalid(format!("invalid {noun} url: {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(invalid(format!("{noun} url must be http(s): {url}")));
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| invalid(format!("{noun} url has no host")))?
        .to_string();
    let host = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .to_string();
    let port = parsed.port_or_known_default().unwrap_or(443);
    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(REMOTE_FETCH_TIMEOUT_SECS);

    // Resolve off the async runtime, then refuse internal/private targets.
    let lookup = host.clone();
    let addrs: Vec<SocketAddr> = tokio::time::timeout_at(
        deadline,
        crate::resources::run_blocking(move || {
            Ok((lookup.as_str(), port)
                .to_socket_addrs()?
                .collect::<Vec<_>>())
        }),
    )
    .await
    .map_err(|_| invalid(format!("{noun} URL DNS lookup timed out")))?
    .map_err(|e| invalid(format!("could not resolve {noun} url host: {e}")))?;

    if addrs.is_empty() {
        return Err(invalid(format!("{noun} url host did not resolve")));
    }
    if addrs.iter().any(|a| is_blocked_ip(a.ip())) {
        return Err(invalid(format!(
            "{noun} url resolves to a private/loopback/link-local address; refused"
        )));
    }

    // Pin to the validated IP (no second DNS lookup -> no rebinding) and forbid
    // redirects (a 30x could otherwise bounce to an internal host).
    //
    // A total deadline is right here, unlike the shared OpenRouter client: this
    // fetches a URL the *model* supplied, and the body is capped, so there is
    // no legitimate slow-but-large transfer to protect. Without it, a host that
    // accepts and then dribbles bytes hangs the tool call forever - the size
    // cap never trips on a drip.
    //
    // `no_gzip` because enabling reqwest's `gzip` feature turns auto-decompression
    // on for every client in the process. Decoded responses lose Content-Length,
    // which would silently kill the early size check below; media bytes are
    // already compressed, so there is nothing to win here anyway.
    let client = reqwest::Client::builder()
        .no_proxy()
        .tls_backend_rustls()
        .redirect(reqwest::redirect::Policy::none())
        .resolve(&host, addrs[0])
        .timeout(deadline.saturating_duration_since(tokio::time::Instant::now()))
        .no_gzip()
        .build()
        .map_err(|e| ErrorData::internal_error(format!("http client build failed: {e}"), None))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| invalid(format!("could not fetch {noun} url: {e}")))?;
    if resp.status().is_redirection() {
        return Err(invalid(format!(
            "{noun} url returned a redirect; refused (SSRF guard)"
        )));
    }
    let mut resp = resp
        .error_for_status()
        .map_err(|e| invalid(format!("{noun} url returned an error: {e}")))?;
    let content_length = resp.content_length();
    if let Some(length) = content_length
        && length > limit
    {
        return Err(invalid(format!(
            "{noun} url body is too large ({length} bytes; maximum is {limit})"
        )));
    }

    // Enforce the limit while streaming as Content-Length may be absent or
    // inaccurate. `Response::bytes()` would buffer an unbounded body first.
    let capacity = content_length.unwrap_or(0).min(limit) as usize;
    let mut bytes = Vec::with_capacity(capacity);
    while let Some(chunk) = resp.chunk().await.map_err(|e| {
        ErrorData::internal_error(format!("could not read {noun} url body: {e}"), None)
    })? {
        let next_len = bytes.len().saturating_add(chunk.len());
        if next_len as u64 > limit {
            return Err(invalid(format!(
                "{noun} url body exceeds the {limit}-byte maximum"
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// Refuse more than [`MAX_INPUTS_PER_KIND`] entries of one kind.
fn check_count(kind: InputKind, len: usize) -> Result<(), ErrorData> {
    if len > MAX_INPUTS_PER_KIND {
        return Err(ErrorData::invalid_params(
            format!(
                "at most {MAX_INPUTS_PER_KIND} {} inputs are supported",
                kind.noun()
            ),
            None,
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Typed chat inputs
// ---------------------------------------------------------------------------

/// A document input (PDF, text, office files) for `chat_completion`. Exactly
/// one of `path`, `url`, or `base64`.
#[derive(Debug, Default, Clone, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = AtLeastOneOf(&["path", "url", "base64"]))]
pub(crate) struct FileInput {
    /// Local file path. One of path/url/base64.
    #[serde(default)]
    pub path: Option<String>,
    /// HTTP(S) URL to fetch the file from. One of path/url/base64.
    #[serde(default)]
    pub url: Option<String>,
    /// Inline file data: a `data:` URL or raw base64. One of path/url/base64.
    /// Needs `filename`.
    #[serde(default)]
    pub base64: Option<String>,
    /// File name sent to the model (its extension tells the parser what it
    /// is). Defaults to the path's / URL's file name; required with base64.
    #[serde(default)]
    pub filename: Option<String>,
}

/// An audio clip input for `chat_completion`. Exactly one of `path`, `base64`.
#[derive(Debug, Default, Clone, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = AtLeastOneOf(&["path", "base64"]))]
pub(crate) struct AudioInput {
    /// Local audio file (wav/mp3/flac/m4a/ogg/webm/aac, max 25 MiB). One of
    /// path/base64.
    #[serde(default)]
    pub path: Option<String>,
    /// Inline audio: a `data:audio/...;base64,` URL or raw base64. One of
    /// path/base64. Raw base64 needs `format`.
    #[serde(default)]
    pub base64: Option<String>,
    /// Container format: wav, mp3, flac, m4a, ogg, webm, or aac. Inferred
    /// from the file extension or the data URL's MIME type when omitted.
    #[serde(default)]
    pub format: Option<String>,
}

/// A video input for `chat_completion`. Exactly one of `url`, `path`, `base64`.
#[derive(Debug, Default, Clone, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = AtLeastOneOf(&["url", "path", "base64"]))]
pub(crate) struct VideoInput {
    /// Video URL (http(s), or a YouTube link on providers that accept one),
    /// passed through untouched for the provider to fetch. One of url/path/base64.
    #[serde(default)]
    pub url: Option<String>,
    /// Local video file (mp4/webm/mov/mkv, max 20 MiB), sent as a data URL.
    /// One of url/path/base64.
    #[serde(default)]
    pub path: Option<String>,
    /// Inline video: a `data:video/...;base64,` URL or raw base64 (max 20 MiB).
    /// One of url/path/base64.
    #[serde(default)]
    pub base64: Option<String>,
    /// Provider processing hint (e.g. Gemini media resolution "low"/"high"),
    /// passed through untouched.
    #[serde(default)]
    pub processing: Option<String>,
}

/// Cheap shape checks (exactly one source), for the pre-gate validation pass.
pub(crate) fn check_file_input(f: &FileInput) -> Result<(), ErrorData> {
    check_exactly_one(
        InputKind::File,
        f.path.as_deref(),
        f.url.as_deref(),
        f.base64.as_deref(),
    )
}

pub(crate) fn check_audio_input(a: &AudioInput) -> Result<(), ErrorData> {
    check_exactly_one(
        InputKind::Audio,
        a.path.as_deref(),
        None,
        a.base64.as_deref(),
    )
}

pub(crate) fn check_video_input(v: &VideoInput) -> Result<(), ErrorData> {
    check_exactly_one(
        InputKind::Video,
        v.path.as_deref(),
        v.url.as_deref(),
        v.base64.as_deref(),
    )
}

/// The MIME type for a `file` part: sniffed (PDF magic, raster images) first,
/// then what a data URL declared, then the file name's extension, else
/// `application/octet-stream`.
fn file_mime(bytes: &[u8], filename: &str, declared: Option<&str>) -> String {
    if bytes.starts_with(b"%PDF") {
        return "application/pdf".to_string();
    }
    if let Some(m) = crate::image_io::sniff_mime(bytes) {
        return m.to_string();
    }
    if let Some(d) = present(declared) {
        return d.to_ascii_lowercase();
    }
    let ext = Path::new(filename)
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "md" => "text/markdown",
        "csv" => "text/csv",
        "json" => "application/json",
        "html" | "htm" => "text/html",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        _ => "application/octet-stream",
    }
    .to_string()
}

/// The MIME type for an inline `video_url`: sniffed (MP4/QuickTime `ftyp`,
/// WebM/Matroska EBML) first, then a declared data-URL type, then the
/// extension, else `video/mp4`.
fn video_mime(bytes: &[u8], name: &str, declared: Option<&str>) -> String {
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        return if bytes[8..].starts_with(b"qt") {
            "video/quicktime"
        } else {
            "video/mp4"
        }
        .to_string();
    }
    if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        let ext = Path::new(name)
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase());
        return if ext.as_deref() == Some("mkv") {
            "video/x-matroska"
        } else {
            "video/webm"
        }
        .to_string();
    }
    if let Some(d) = present(declared) {
        return d.to_ascii_lowercase();
    }
    let ext = Path::new(name)
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "avi" => "video/x-msvideo",
        "mpeg" | "mpg" => "video/mpeg",
        _ => "video/mp4",
    }
    .to_string()
}

/// The MIME type for an inline audio reference: sniffed (RIFF/WAVE, ID3 or an
/// MPEG frame sync, fLaC, OggS, an `ftyp` box, EBML) first, then a declared
/// data-URL type, then the extension, else `audio/mpeg`.
fn audio_mime(bytes: &[u8], name: &str, declared: Option<&str>) -> String {
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        return "audio/wav".to_string();
    }
    let mpeg_sync = bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] & 0xE0 == 0xE0;
    if bytes.starts_with(b"ID3") || mpeg_sync {
        return "audio/mpeg".to_string();
    }
    if bytes.starts_with(b"fLaC") {
        return "audio/flac".to_string();
    }
    if bytes.starts_with(b"OggS") {
        return "audio/ogg".to_string();
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        return "audio/mp4".to_string();
    }
    if bytes.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return "audio/webm".to_string();
    }
    if let Some(d) = present(declared) {
        return d.to_ascii_lowercase();
    }
    let ext = Path::new(name)
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "wav" => "audio/wav",
        "flac" => "audio/flac",
        "m4a" => "audio/mp4",
        "ogg" => "audio/ogg",
        "aac" => "audio/aac",
        "webm" | "weba" => "audio/webm",
        _ => "audio/mpeg",
    }
    .to_string()
}

/// Resolve one audio/video reference for a video job's `input_references`
/// (`kind` is [`InputKind::Audio`] or [`InputKind::Video`]): an `http(s)://`
/// URL passes through untouched (the provider fetches it), a `data:` URL is
/// decoded, capped at [`MAX_MEDIA_BYTES`] and re-issued, and a local path is
/// read (same cap) and inlined as a data URL typed from its bytes, declared
/// type, or extension. Shared by the CLI and the MCP tool through
/// `video_gen`, and with the chat inputs above through [`resolve_source`].
pub(crate) async fn resolve_media_reference(
    kind: InputKind,
    source: &str,
) -> Result<String, ErrorData> {
    let source = source.trim();
    if source.is_empty() {
        return Err(ErrorData::invalid_params(
            format!("a {} reference is blank", kind.noun()),
            None,
        ));
    }
    let lower = source.to_ascii_lowercase();
    let (path, url, base64) = if lower.starts_with("http://") || lower.starts_with("https://") {
        (None, Some(source.to_string()), None)
    } else if lower.starts_with("data:") {
        (None, None, Some(source.to_string()))
    } else {
        (Some(source.to_string()), None, None)
    };
    let resolved = resolve_source(kind, path, url, base64, MAX_MEDIA_BYTES, false).await?;
    let media = match resolved {
        Resolved::Url(url) => return Ok(url),
        other => into_bytes(kind, other, MAX_MEDIA_BYTES).await?,
    };
    if media.bytes.is_empty() {
        return Err(ErrorData::invalid_params(
            format!("{} reference {} is empty", kind.noun(), media.name),
            None,
        ));
    }
    let declared = media.declared_mime.as_deref();
    let mime = match kind {
        InputKind::Audio => audio_mime(&media.bytes, &media.name, declared),
        _ => video_mime(&media.bytes, &media.name, declared),
    };
    Ok(crate::image_io::data_url(&media.bytes, &mime))
}

/// The file name a URL or path implies: its last segment without a query.
fn implied_filename(name: &str) -> Option<String> {
    let trimmed = name.split(['?', '#']).next().unwrap_or(name);
    let last = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed).trim();
    (!last.is_empty() && last != "inline").then(|| last.to_string())
}

async fn resolve_file_input(f: FileInput) -> Result<FilePart, ErrorData> {
    let filename = present(f.filename.as_deref()).map(str::to_string);
    let resolved = resolve_source(
        InputKind::File,
        f.path,
        f.url,
        f.base64,
        MAX_MEDIA_BYTES,
        true,
    )
    .await?;
    let media = into_bytes(InputKind::File, resolved, MAX_MEDIA_BYTES).await?;
    let filename = filename
        .or_else(|| implied_filename(&media.name))
        .ok_or_else(|| {
            ErrorData::invalid_params(
                "files[].filename is required with base64 input (e.g. \"report.pdf\")",
                None,
            )
        })?;
    let mime = file_mime(&media.bytes, &filename, media.declared_mime.as_deref());
    Ok(FilePart {
        filename,
        file_data: crate::image_io::data_url(&media.bytes, &mime),
    })
}

/// Resolve `files` to `file` parts, in order (fetching URLs, reading paths).
pub(crate) async fn resolve_file_inputs(files: Vec<FileInput>) -> Result<Vec<FilePart>, ErrorData> {
    check_count(InputKind::File, files.len())?;
    let mut out = Vec::with_capacity(files.len());
    for f in files {
        out.push(resolve_file_input(f).await?);
    }
    Ok(out)
}

async fn resolve_audio_input(a: AudioInput) -> Result<InputAudio, ErrorData> {
    check_audio_input(&a)?;
    let invalid = |e: anyhow::Error| ErrorData::invalid_params(format!("{e:#}"), None);
    let format = present(a.format.as_deref()).map(str::to_string);
    if let Some(p) = present(a.path.as_deref()) {
        let (data, format) = crate::audio_gen::read_audio_file(Path::new(p), format.as_deref())
            .await
            .map_err(invalid)?;
        return Ok(InputAudio { data, format });
    }
    let b64 = a.base64.unwrap_or_default();
    // Tolerate a `data:audio/mp3;base64,...` URL: upstream wants the raw bytes,
    // and the subtype is a usable format when none was passed.
    let (from_url, data) = if b64.trim().starts_with("data:") {
        let (mime, data) = crate::image_io::split_data_url(&b64).map_err(invalid)?;
        (
            mime.rsplit('/').next().map(str::to_string),
            data.trim().to_string(),
        )
    } else {
        (None, b64.trim().to_string())
    };
    let format = format.or(from_url).ok_or_else(|| {
        ErrorData::invalid_params(
            "audio[].format is required with raw base64 (wav, mp3, flac, m4a, ogg, webm, aac)",
            None,
        )
    })?;
    let (data, format) = crate::resources::run_blocking(move || {
        crate::audio_gen::validate_inline_audio(&data, &format)
    })
    .await
    .map_err(invalid)?;
    Ok(InputAudio { data, format })
}

/// Resolve `audio` to `input_audio` parts, in order.
pub(crate) async fn resolve_audio_inputs(
    audio: Vec<AudioInput>,
) -> Result<Vec<InputAudio>, ErrorData> {
    check_count(InputKind::Audio, audio.len())?;
    let mut out = Vec::with_capacity(audio.len());
    for a in audio {
        out.push(resolve_audio_input(a).await?);
    }
    Ok(out)
}

async fn resolve_video_input(v: VideoInput) -> Result<VideoUrl, ErrorData> {
    let processing = present(v.processing.as_deref()).map(str::to_string);
    let resolved = resolve_source(
        InputKind::Video,
        v.path,
        v.url,
        v.base64,
        MAX_MEDIA_BYTES,
        false,
    )
    .await?;
    let url = match resolved {
        Resolved::Url(url) => url,
        other => {
            let media = into_bytes(InputKind::Video, other, MAX_MEDIA_BYTES).await?;
            let mime = video_mime(&media.bytes, &media.name, media.declared_mime.as_deref());
            crate::image_io::data_url(&media.bytes, &mime)
        }
    };
    Ok(VideoUrl { url, processing })
}

/// Resolve `videos` to `video_url` parts, in order: URLs pass through, local
/// bytes become data URLs.
pub(crate) async fn resolve_video_inputs(
    videos: Vec<VideoInput>,
) -> Result<Vec<VideoUrl>, ErrorData> {
    check_count(InputKind::Video, videos.len())?;
    let mut out = Vec::with_capacity(videos.len());
    for v in videos {
        out.push(resolve_video_input(v).await?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::valid_png_b64;
    use rmcp::handler::server::common::schema_for_type;
    use serde_json::json;

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    async fn image_source(
        path: Option<&str>,
        url: Option<&str>,
        base64: Option<&str>,
    ) -> Result<Resolved, ErrorData> {
        resolve_source(
            InputKind::Image,
            path.map(str::to_string),
            url.map(str::to_string),
            base64.map(str::to_string),
            crate::resources::MAX_IMAGE_BYTES,
            true,
        )
        .await
    }

    #[tokio::test]
    async fn resolve_source_decodes_base64_and_data_url_and_keeps_declared_mime() {
        // Raw base64 -> inline bytes, no declared type.
        match image_source(None, None, Some(&valid_png_b64()))
            .await
            .unwrap()
        {
            Resolved::Bytes(m) => {
                assert!(!m.bytes.is_empty());
                assert_eq!(m.name, "inline");
                assert_eq!(m.declared_mime, None);
            }
            other => panic!("expected inline bytes from base64, got {other:?}"),
        }
        // A full data: URL also decodes, and its MIME is kept for consumers.
        let data_url = format!("data:image/png;base64,{}", valid_png_b64());
        match image_source(None, None, Some(&data_url)).await.unwrap() {
            Resolved::Bytes(m) => assert_eq!(m.declared_mime.as_deref(), Some("image/png")),
            other => panic!("expected inline bytes from data URL, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resolve_source_keeps_path_and_rejects_bad_input() {
        assert!(matches!(
            image_source(Some("/tmp/a.png"), None, None).await.unwrap(),
            Resolved::Path(_)
        ));

        // No source -> error naming the alternatives for the kind.
        let err = image_source(None, None, None).await.unwrap_err();
        assert!(err.message.contains("exactly one of"));
        assert!(err.message.contains("image"), "{}", err.message);

        // Two sources -> error.
        let err = image_source(Some("/tmp/a.png"), None, Some("x"))
            .await
            .unwrap_err();
        assert!(err.message.contains("exactly one of"));

        // Non-http url -> rejected (never sent anywhere).
        let err = image_source(None, Some("file:///etc/passwd"), None)
            .await
            .unwrap_err();
        assert!(err.message.contains("http"));

        // Garbage base64 and an oversized payload are refused before any use.
        let err = image_source(None, None, Some("!!!")).await.unwrap_err();
        assert!(err.message.contains("base64"), "{}", err.message);
        let big = "A".repeat(crate::resources::MAX_IMAGE_BYTES.div_ceil(3) * 4 + 8);
        let err = image_source(None, None, Some(&big)).await.unwrap_err();
        assert!(err.message.contains("exceeds"), "{}", err.message);
    }

    /// Audio has no `url`, so its message lists only the two fields it has.
    #[test]
    fn exactly_one_message_names_the_kind_and_its_sources() {
        let err = check_exactly_one(InputKind::Audio, None, None, None).unwrap_err();
        assert_eq!(
            err.message,
            "each audio needs exactly one of: path or base64"
        );
        assert!(check_exactly_one(InputKind::File, Some(" a.pdf "), None, Some("  ")).is_ok());
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
        let err = image_source(None, Some("http://127.0.0.1:9/pic.png"), None)
            .await
            .unwrap_err();
        assert!(err.message.contains("private/loopback"));

        // The cloud metadata endpoint is link-local and likewise refused.
        let err = image_source(None, Some("http://169.254.169.254/latest"), None)
            .await
            .unwrap_err();
        assert!(err.message.contains("private/loopback"));

        // The same guard protects every kind, not just images.
        let err = resolve_file_inputs(vec![FileInput {
            url: Some("http://127.0.0.1:9/doc.pdf".into()),
            ..Default::default()
        }])
        .await
        .unwrap_err();
        assert!(err.message.contains("file url"), "{}", err.message);
    }

    #[tokio::test]
    async fn file_inputs_become_data_urls_with_sniffed_mime_and_filename() {
        // Inline PDF with an explicit name.
        let parts = resolve_file_inputs(vec![FileInput {
            base64: Some(b64(b"%PDF-1.7 hello")),
            filename: Some(" report.pdf ".into()),
            ..Default::default()
        }])
        .await
        .unwrap();
        assert_eq!(parts[0].filename, "report.pdf");
        assert!(
            parts[0]
                .file_data
                .starts_with("data:application/pdf;base64,"),
            "{}",
            parts[0].file_data
        );

        // A local file: the name defaults to the file name, the type to the
        // sniffed bytes (a PNG named .bin is still a PNG).
        let dir = std::env::temp_dir().join("openrouter-mcp-media-file");
        std::fs::create_dir_all(&dir).unwrap();
        let png_path = dir.join("picture.bin");
        std::fs::write(
            &png_path,
            base64::engine::general_purpose::STANDARD
                .decode(valid_png_b64())
                .unwrap(),
        )
        .unwrap();
        let parts = resolve_file_inputs(vec![FileInput {
            path: Some(png_path.to_string_lossy().into_owned()),
            ..Default::default()
        }])
        .await
        .unwrap();
        assert_eq!(parts[0].filename, "picture.bin");
        assert!(parts[0].file_data.starts_with("data:image/png;base64,"));

        // Unsniffable text: the declared data-URL type wins, else the extension.
        let parts = resolve_file_inputs(vec![
            FileInput {
                base64: Some(format!("data:text/csv;base64,{}", b64(b"a,b"))),
                filename: Some("t.txt".into()),
                ..Default::default()
            },
            FileInput {
                base64: Some(b64(b"# notes")),
                filename: Some("notes.md".into()),
                ..Default::default()
            },
            FileInput {
                base64: Some(b64(b"???")),
                filename: Some("blob".into()),
                ..Default::default()
            },
        ])
        .await
        .unwrap();
        assert!(parts[0].file_data.starts_with("data:text/csv;"));
        assert!(parts[1].file_data.starts_with("data:text/markdown;"));
        assert!(
            parts[2]
                .file_data
                .starts_with("data:application/octet-stream;")
        );

        // base64 without a name has nothing to call the file.
        let err = resolve_file_inputs(vec![FileInput {
            base64: Some(b64(b"%PDF-1.7")),
            ..Default::default()
        }])
        .await
        .unwrap_err();
        assert!(err.message.contains("filename"), "{}", err.message);

        // A missing local file is a clear argument error.
        let err = resolve_file_inputs(vec![FileInput {
            path: Some(dir.join("nope.pdf").to_string_lossy().into_owned()),
            ..Default::default()
        }])
        .await
        .unwrap_err();
        assert!(err.message.contains("nope.pdf"), "{}", err.message);
    }

    #[tokio::test]
    async fn audio_inputs_take_the_format_from_the_argument_extension_or_data_url() {
        // Data URL: format from the MIME subtype, prefix stripped.
        let parts = resolve_audio_inputs(vec![AudioInput {
            base64: Some("data:audio/mp3;base64,QUJD".into()),
            ..Default::default()
        }])
        .await
        .unwrap();
        assert_eq!(parts[0].data, "QUJD");
        assert_eq!(parts[0].format, "mp3");

        // Raw base64 needs a format...
        let err = resolve_audio_inputs(vec![AudioInput {
            base64: Some("QUJD".into()),
            ..Default::default()
        }])
        .await
        .unwrap_err();
        assert!(err.message.contains("format"), "{}", err.message);
        // ...and an unknown one is refused.
        let err = resolve_audio_inputs(vec![AudioInput {
            base64: Some("QUJD".into()),
            format: Some("midi".into()),
            ..Default::default()
        }])
        .await
        .unwrap_err();
        assert!(err.message.contains("unsupported"), "{}", err.message);

        // A local file: format from its extension, bytes base64'd.
        let dir = std::env::temp_dir().join("openrouter-mcp-media-audio");
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("clip.wav");
        std::fs::write(&wav, b"RIFF....WAVE").unwrap();
        let parts = resolve_audio_inputs(vec![AudioInput {
            path: Some(wav.to_string_lossy().into_owned()),
            ..Default::default()
        }])
        .await
        .unwrap();
        assert_eq!(parts[0].format, "wav");
        assert_eq!(parts[0].data, b64(b"RIFF....WAVE"));

        // Both sources at once is the shared exactly-one error.
        let err = resolve_audio_inputs(vec![AudioInput {
            path: Some("a.wav".into()),
            base64: Some("QUJD".into()),
            ..Default::default()
        }])
        .await
        .unwrap_err();
        assert!(err.message.contains("path or base64"), "{}", err.message);
    }

    #[tokio::test]
    async fn video_inputs_pass_urls_through_and_inline_local_bytes() {
        // A URL is not fetched, not validated, not rewritten.
        let parts = resolve_video_inputs(vec![VideoInput {
            url: Some(" https://www.youtube.com/watch?v=abc ".into()),
            processing: Some("low".into()),
            ..Default::default()
        }])
        .await
        .unwrap();
        assert_eq!(parts[0].url, "https://www.youtube.com/watch?v=abc");
        assert_eq!(parts[0].processing.as_deref(), Some("low"));

        // Inline MP4 (ftyp box) -> video/mp4 data URL; WebM EBML -> video/webm.
        let mp4 = [&[0, 0, 0, 0x18][..], b"ftypisom", &[0; 8]].concat();
        let webm = [&[0x1A, 0x45, 0xDF, 0xA3][..], &[0; 8]].concat();
        let parts = resolve_video_inputs(vec![
            VideoInput {
                base64: Some(b64(&mp4)),
                ..Default::default()
            },
            VideoInput {
                base64: Some(b64(&webm)),
                ..Default::default()
            },
        ])
        .await
        .unwrap();
        assert!(
            parts[0].url.starts_with("data:video/mp4;base64,"),
            "{}",
            parts[0].url
        );
        assert!(
            parts[1].url.starts_with("data:video/webm;base64,"),
            "{}",
            parts[1].url
        );
        assert_eq!(parts[0].processing, None);

        // A local file with unsniffable bytes: the extension decides.
        let dir = std::env::temp_dir().join("openrouter-mcp-media-video");
        std::fs::create_dir_all(&dir).unwrap();
        let mov = dir.join("clip.mov");
        std::fs::write(&mov, b"not really a movie").unwrap();
        let parts = resolve_video_inputs(vec![VideoInput {
            path: Some(mov.to_string_lossy().into_owned()),
            ..Default::default()
        }])
        .await
        .unwrap();
        assert!(
            parts[0].url.starts_with("data:video/quicktime;base64,"),
            "{}",
            parts[0].url
        );

        // Too many entries are refused before any is read.
        let many = (0..MAX_INPUTS_PER_KIND + 1)
            .map(|_| VideoInput {
                base64: Some("!!!".into()),
                ..Default::default()
            })
            .collect();
        let err = resolve_video_inputs(many).await.unwrap_err();
        assert!(err.message.contains("at most"), "{}", err.message);
    }

    /// Video-job references (`reference_audio`/`reference_videos`) go through
    /// the same resolver, caps and MIME tables as the chat inputs: URLs pass
    /// through untouched, `data:` URLs are decoded, capped and re-issued, local
    /// files are read and typed from their bytes, then their extension.
    #[tokio::test]
    async fn media_references_pass_urls_through_and_inline_local_and_data_sources() {
        for url in ["https://cdn/beat.mp3", "http://cdn/clip.mp4"] {
            assert_eq!(
                resolve_media_reference(InputKind::Audio, url)
                    .await
                    .unwrap(),
                url
            );
        }
        assert_eq!(
            resolve_media_reference(InputKind::Audio, "  https://cdn/x.mp3 ")
                .await
                .unwrap(),
            "https://cdn/x.mp3"
        );
        // A data URL keeps its declared type once decoded and re-encoded.
        assert_eq!(
            resolve_media_reference(InputKind::Audio, "data:audio/wav;base64,AAAA")
                .await
                .unwrap(),
            "data:audio/wav;base64,AAAA"
        );

        let dir = std::env::temp_dir().join("openrouter-mcp-media-reference");
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let p = dir.join(name);
            std::fs::write(&p, bytes).unwrap();
            p.to_string_lossy().into_owned()
        };
        // Extension-typed audio and video, including containers the old
        // video-only table lacked (mkv, webm audio).
        for (kind, name, expected) in [
            (InputKind::Audio, "beat.MP3", "data:audio/mpeg;base64,QUJD"),
            (
                InputKind::Audio,
                "voice.webm",
                "data:audio/webm;base64,QUJD",
            ),
            (
                InputKind::Video,
                "ref.mov",
                "data:video/quicktime;base64,QUJD",
            ),
            (
                InputKind::Video,
                "ref.mkv",
                "data:video/x-matroska;base64,QUJD",
            ),
        ] {
            let path = write(name, b"ABC");
            assert_eq!(
                resolve_media_reference(kind, &path).await.unwrap(),
                expected,
                "{name}"
            );
        }
        // Sniffed bytes beat a misleading extension.
        let wav = write("sample.bin", b"RIFF   WAVEfmt ");
        assert!(
            resolve_media_reference(InputKind::Audio, &wav)
                .await
                .unwrap()
                .starts_with("data:audio/wav;base64,"),
        );

        // Blank, missing, empty, and oversized inline sources are refused.
        assert!(
            resolve_media_reference(InputKind::Audio, "   ")
                .await
                .is_err()
        );
        let missing = dir.join("missing.mp4");
        assert!(
            resolve_media_reference(InputKind::Video, &missing.to_string_lossy())
                .await
                .is_err()
        );
        let empty = write("empty.mp3", b"");
        let err = resolve_media_reference(InputKind::Audio, &empty)
            .await
            .unwrap_err();
        assert!(err.message.contains("empty"), "{}", err.message);
        let huge = format!(
            "data:video/mp4;base64,{}",
            "A".repeat(MAX_MEDIA_BYTES.div_ceil(3) * 4 + 4)
        );
        let err = resolve_media_reference(InputKind::Video, &huge)
            .await
            .unwrap_err();
        assert!(err.message.contains("exceeds"), "{}", err.message);
    }

    /// Each nested input advertises its own "at least one source" branches
    /// (only on the nested type, never a tool-args root) and stays free of
    /// nullable unions.
    #[test]
    fn nested_inputs_encode_at_least_one_source_and_are_client_safe() {
        let file = schema_for_type::<FileInput>();
        assert_eq!(
            file.get("anyOf").cloned(),
            Some(json!([
                { "type": "object", "required": ["path"] },
                { "type": "object", "required": ["url"] },
                { "type": "object", "required": ["base64"] }
            ]))
        );
        let audio = schema_for_type::<AudioInput>();
        assert_eq!(
            audio.get("anyOf").cloned(),
            Some(json!([
                { "type": "object", "required": ["path"] },
                { "type": "object", "required": ["base64"] }
            ]))
        );
        let video = schema_for_type::<VideoInput>();
        assert_eq!(
            video.get("anyOf").cloned(),
            Some(json!([
                { "type": "object", "required": ["url"] },
                { "type": "object", "required": ["path"] },
                { "type": "object", "required": ["base64"] }
            ]))
        );
        // Optional fields collapse to scalars (the nested type opted in).
        assert_eq!(video["properties"]["processing"]["type"], "string");
        assert_eq!(audio["properties"]["format"]["type"], "string");
        assert_eq!(file["properties"]["filename"]["type"], "string");
    }
}
