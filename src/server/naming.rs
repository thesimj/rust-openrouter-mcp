//! Auto-naming for generated artifacts when the caller omits `output`.
//!
//! Scheme: `<kind>_<YYYYMMDD-HHMMSS>_<model>_<config…>_seed<seed>_<hash4>` (no
//! extension - the job sets the real extension once the format is known). The
//! timestamp sits right after the kind so names sort chronologically within a
//! kind; a 4-hex collision tail (seeded with sub-second nanos) keeps two jobs
//! that share every visible field distinct.
//!
//! Files land under [`output_dir`] (env `OPENROUTER_MCP_OUTPUT_DIR`, else
//! `$HOME/Downloads/openrouter-mcp`, else a temp dir) so they are user-visible
//! regardless of the server process's working directory - which is undefined in
//! the Claude Desktop / `.mcpb` runtime.

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use std::hash::{Hash, Hasher};

/// What kind of artifact a job produced - drives the filename prefix.
#[derive(Clone, Copy)]
pub(crate) enum MediaKind {
    Image,
    Video,
    Audio,
}

impl MediaKind {
    fn prefix(self) -> &'static str {
        match self {
            MediaKind::Image => "img",
            MediaKind::Video => "vid",
            MediaKind::Audio => "aud",
        }
    }
}

/// Lowercase a model id to a filename-safe token: drop the vendor prefix
/// (`google/…`), keep alphanumerics, `.` and `-`, collapse everything else to a
/// single `-`. "google/gemini-3.1-flash-image-preview" -> "gemini-3.1-flash-image-preview".
fn model_token(model: &str) -> String {
    let tail = model.rsplit('/').next().unwrap_or(model);
    sanitize(tail, true)
}

/// Sanitize one config/model token to filesystem-safe characters. Maps `:` to
/// `x` (so "16:9" -> "16x9"), keeps alphanumerics/`.`/`-`, collapses any other
/// run to a single `-`, and trims leading/trailing `-`. Config tokens keep their
/// case ("2K", "720p"); the model token is lowercased.
fn sanitize(s: &str, lowercase: bool) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_dash = false;
    for c in s.chars() {
        let c = if c == ':' { 'x' } else { c };
        if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
            out.push(if lowercase { c.to_ascii_lowercase() } else { c });
            prev_dash = c == '-';
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// 4-char hex nonce making a name collision-proof even when model/config/seed
/// and the second-resolution timestamp all match. Folds in `now`'s sub-second
/// nanos so two jobs in the same second still differ. Not a stable digest.
fn collision_tail(model: &str, config: &[&str], seed: Option<u64>, now: DateTime<Utc>) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model.hash(&mut h);
    config.hash(&mut h);
    seed.hash(&mut h);
    now.timestamp_subsec_nanos().hash(&mut h);
    format!("{:04x}", h.finish() & 0xffff)
}

/// Build the auto-generated base name (no extension). `config` tokens are
/// kind-specific and supplied by the caller (image: aspect, size; video:
/// aspect-or-size, resolution, `<n>s`; audio: voice, format). Empty tokens are
/// dropped; `seed` becomes `seed<n>` when present and is omitted otherwise.
pub(crate) fn auto_base_name(
    kind: MediaKind,
    model: &str,
    config: &[&str],
    seed: Option<u64>,
    now: DateTime<Utc>,
) -> String {
    let mut parts = vec![
        kind.prefix().to_string(),
        now.format("%Y%m%d-%H%M%S").to_string(),
        model_token(model),
    ];
    for t in config {
        let token = sanitize(t, false);
        if !token.is_empty() {
            parts.push(token);
        }
    }
    if let Some(s) = seed {
        parts.push(format!("seed{s}"));
    }
    parts.push(collision_tail(model, config, seed, now));
    parts.join("_")
}

/// The base directory for auto-named outputs: `OPENROUTER_MCP_OUTPUT_DIR` if
/// set, else `$HOME/Downloads/openrouter-mcp`, else a temp dir. User-visible by
/// default so artifacts are findable when the process CWD is undefined.
pub(crate) fn output_dir() -> PathBuf {
    if let Ok(d) = std::env::var("OPENROUTER_MCP_OUTPUT_DIR") {
        if !d.trim().is_empty() {
            return PathBuf::from(d);
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home).join("Downloads").join("openrouter-mcp");
        }
    }
    std::env::temp_dir().join("openrouter-mcp")
}

/// Resolve the output base path: the caller-supplied `output` verbatim when
/// non-empty, else an auto-named file under [`output_dir`]. The returned path
/// has no enforced extension - the job sets it from the returned format.
pub(crate) fn resolve_output_base(
    output: Option<String>,
    kind: MediaKind,
    model: &str,
    config: &[&str],
    seed: Option<u64>,
) -> PathBuf {
    match output {
        Some(o) if !o.trim().is_empty() => PathBuf::from(o),
        _ => output_dir().join(auto_base_name(kind, model, config, seed, Utc::now())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Timelike};

    fn at(nanos: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 17, 17, 22, 45)
            .unwrap()
            .with_nanosecond(nanos)
            .unwrap()
    }

    #[test]
    fn image_name_shape() {
        let n = auto_base_name(
            MediaKind::Image,
            "google/gemini-3.1-flash-image-preview",
            &["16:9", "2K"],
            Some(4242),
            at(0),
        );
        // <kind>_<datetime>_<model>_<aspect>_<size>_seed<n>_<hash4>
        assert!(
            n.starts_with("img_20260617-172245_gemini-3.1-flash-image-preview_16x9_2K_seed4242_")
        );
        let tail = n.rsplit('_').next().unwrap();
        assert_eq!(tail.len(), 4);
        assert!(tail.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn no_seed_drops_seed_field() {
        let n = auto_base_name(
            MediaKind::Video,
            "google/veo-3",
            &["16:9", "720p", "8s"],
            None,
            at(0),
        );
        assert!(n.starts_with("vid_20260617-172245_veo-3_16x9_720p_8s_"));
        assert!(!n.contains("seed"));
    }

    #[test]
    fn audio_keeps_case_drops_empty_config() {
        let n = auto_base_name(
            MediaKind::Audio,
            "openai/gpt-4o-mini-tts",
            &["alloy", "", "mp3"],
            Some(1),
            at(0),
        );
        assert!(n.starts_with("aud_20260617-172245_gpt-4o-mini-tts_alloy_mp3_seed1_"));
    }

    #[test]
    fn sub_second_nanos_break_collisions() {
        let a = auto_base_name(MediaKind::Image, "m", &["1:1"], Some(7), at(0));
        let b = auto_base_name(MediaKind::Image, "m", &["1:1"], Some(7), at(500_000_000));
        // Same visible fields and same second, different nanos -> different tails.
        assert_ne!(a, b);
    }

    #[test]
    fn explicit_output_used_verbatim() {
        let p = resolve_output_base(
            Some("out/hero.png".to_string()),
            MediaKind::Image,
            "m",
            &[],
            None,
        );
        assert_eq!(p, PathBuf::from("out/hero.png"));
    }

    #[test]
    fn blank_output_falls_back_to_auto_name() {
        let p = resolve_output_base(
            Some("   ".to_string()),
            MediaKind::Image,
            "m",
            &["1:1"],
            None,
        );
        assert!(p.file_name().unwrap().to_string_lossy().starts_with("img_"));
    }
}
