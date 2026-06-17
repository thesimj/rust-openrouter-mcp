//! Job-envelope construction, inline preview/media encoding, and the shared
//! background-job spawn/wait flow used by the image and video tools.

use base64::Engine;
use rmcp::{
    ErrorData, RoleServer,
    model::{CallToolResult, Content, RawResource},
    service::RequestContext,
};
use serde_json::json;

use crate::stats::UsageStats;
use crate::tasks::{TaskKind, TaskSnapshot};

use super::OpenRouterServer;

/// Default seconds to wait inline before returning a task id for a slow job.
pub(crate) const DEFAULT_WAIT_SECONDS: u64 = 10;

/// Default inline wait for video: video takes 30s-several minutes, so the
/// fast-return window almost always yields `pending` and the caller polls
/// get_result. Kept within the 1-60 clamp.
pub(crate) const DEFAULT_VIDEO_WAIT_SECONDS: u64 = 20;

/// Longest-side cap (px) for the inline preview embedded in a tool result. The
/// full-resolution image always stays on disk; this only bounds the base64 copy
/// sent to the client so generated images render without bloating context.
/// ~1568px is Claude's image sweet spot (larger is downsampled client-side).
const PREVIEW_MAX_SIDE: u32 = 1568;

/// Most inline previews embedded in a single tool result. Many-variant jobs can
/// produce up to 16 images; base64-embedding every one would bloat the client's
/// context, so any beyond this cap are reported by path in the JSON block only.
const MAX_INLINE_PREVIEWS: usize = 4;

/// Most inline media blocks (audio / video ResourceLinks) attached to a result.
const MAX_INLINE_MEDIA: usize = 4;

/// Largest audio file embedded inline as a base64 AudioContent block. Larger
/// files bloat the client's context badly, so they are reported by path only
/// (the file is always saved to disk regardless).
pub(crate) const MAX_INLINE_AUDIO_BYTES: u64 = 4 * 1024 * 1024;

/// Append non-empty `warnings`/`errors` arrays to a job result object. Shared by
/// the image, video, and audio envelope builders.
pub(crate) fn attach_warnings_errors(
    v: &mut serde_json::Value,
    warnings: &[String],
    errors: &[String],
) {
    if !warnings.is_empty() {
        v["warnings"] = json!(warnings);
    }
    if !errors.is_empty() {
        v["errors"] = json!(errors);
    }
}

/// Collect the on-disk paths of the generated images in a job envelope.
fn envelope_image_paths(env: &serde_json::Value) -> Vec<String> {
    env.get("images")
        .and_then(|v| v.as_array())
        .map(|images| {
            images
                .iter()
                .filter_map(|img| img.get("path").and_then(|p| p.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Collect the on-disk paths of the generated clips in a video job envelope.
fn envelope_video_paths(env: &serde_json::Value) -> Vec<String> {
    env.get("videos")
        .and_then(|v| v.as_array())
        .map(|videos| {
            videos
                .iter()
                .filter_map(|v| v.get("path").and_then(|p| p.as_str()))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Build a `ResourceLink` content block for each generated clip path. rmcp 1.7
/// has no native video content block, so a sandboxed client gets a `file://`
/// ResourceLink (mime video/mp4, size from the file) rather than an embedded
/// blob; the path is also in the JSON text block. Capped at [`MAX_INLINE_MEDIA`].
///
/// Blocking: does a filesystem stat per path - run via `spawn_blocking`.
fn video_resource_link_blocks(paths: &[String]) -> Vec<Content> {
    paths
        .iter()
        .take(MAX_INLINE_MEDIA)
        .map(|path| {
            let name = std::path::Path::new(path)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.clone());
            let mut r = RawResource::new(format!("file://{path}"), name);
            r.mime_type = Some("video/mp4".to_string());
            r.size = std::fs::metadata(path).ok().map(|m| m.len() as u32);
            Content::resource_link(r)
        })
        .collect()
}

/// True when `bytes` is a PNG whose longest side already fits `max_side`, so it
/// can be sent inline verbatim without a decode/resize/re-encode round-trip.
/// [`crate::image_io::decode_dimensions`] only reads the header, so this is cheap.
fn is_png_within_bound(bytes: &[u8], max_side: u32) -> bool {
    const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    bytes.starts_with(&PNG_MAGIC)
        && crate::image_io::decode_dimensions(bytes)
            .map(|(w, h)| w <= max_side && h <= max_side)
            .unwrap_or(false)
}

/// Turn the generated images' paths into inline image content blocks so the
/// client (e.g. Claude Desktop) renders them, not just their paths. Reads each
/// path from disk, downscales to [`PREVIEW_MAX_SIDE`] (or passes an already-small
/// PNG through untouched), and base64-encodes it as a PNG `image` block. Caps the
/// count at [`MAX_INLINE_PREVIEWS`]; unreadable/undecodable files are skipped (the
/// JSON text block still reports their paths), so this never fails the call.
///
/// Blocking: does disk I/O and image decode/encode - run it via `spawn_blocking`,
/// never directly on the async runtime.
fn encode_preview_blocks(paths: &[String]) -> Vec<Content> {
    paths
        .iter()
        .take(MAX_INLINE_PREVIEWS)
        .filter_map(|path| {
            let bytes = std::fs::read(path).ok()?;
            let png = if is_png_within_bound(&bytes, PREVIEW_MAX_SIDE) {
                bytes
            } else {
                crate::image_io::normalize_to_png(&bytes, PREVIEW_MAX_SIDE).ok()?
            };
            let b64 = base64::engine::general_purpose::STANDARD.encode(png);
            Some(Content::image(b64, "image/png".to_string()))
        })
        .collect()
}

/// Decide whether to embed inline image previews for the connected client.
///
/// Why this is client-dependent: a local CLI (Claude Code) shares the
/// filesystem, so a returned path *is* the image - inline base64 only bloats
/// context. Claude Desktop, by contrast, runs the MCP server in a sandbox whose
/// filesystem the app can't read, so a path is useless and the bytes must be
/// returned inline or the image is stranded.
///
/// `OPENROUTER_MCP_IMAGE_PREVIEWS` overrides detection: `always` / `never`
/// (anything else, or unset, means `auto`). The `.mcpb` connector sets `always`.
/// In `auto` we return previews for every client *except* the local-filesystem
/// CLI (`claude-code`), so the failure-prone case (Desktop) is covered by default.
pub(crate) fn client_wants_inline_previews(ctx: &RequestContext<RoleServer>) -> bool {
    match std::env::var("OPENROUTER_MCP_IMAGE_PREVIEWS")
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("always") => return true,
        Some("never") => return false,
        _ => {}
    }
    let client = ctx
        .peer
        .peer_info()
        .map(|info| info.client_info.name.to_ascii_lowercase())
        .unwrap_or_default();
    !client.contains("claude-code")
}

/// Build the full tool result for a job envelope: the JSON metadata as a text
/// block, followed (when `inline_previews` and the job completed) by an inline
/// image preview of each generated image.
pub(crate) async fn job_call_result(
    env: &serde_json::Value,
    inline_previews: bool,
) -> Result<CallToolResult, ErrorData> {
    let body = serde_json::to_string_pretty(env)
        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
    let mut blocks = vec![Content::text(body)];

    if inline_previews && env.get("status").and_then(|s| s.as_str()) == Some("completed") {
        match env.get("kind").and_then(|k| k.as_str()) {
            Some("video") => {
                let paths = envelope_video_paths(env);
                // A filesystem stat per clip is blocking I/O; keep it off the worker.
                let links = tokio::task::spawn_blocking(move || video_resource_link_blocks(&paths))
                    .await
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                blocks.extend(links);
            }
            // "image" (and the historical default) get inline image previews.
            _ => {
                let paths = envelope_image_paths(env);
                let total = paths.len();
                // Reading + decoding + resizing + re-encoding images is blocking CPU
                // and disk I/O; run it off the async worker so concurrent tool calls
                // (e.g. get_result polls) aren't stalled behind it.
                let previews = tokio::task::spawn_blocking(move || encode_preview_blocks(&paths))
                    .await
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                let shown = previews.len();
                blocks.extend(previews);
                if total > MAX_INLINE_PREVIEWS {
                    blocks.push(Content::text(format!(
                        "note: showing {shown} of {total} generated images inline (capped at \
                         {MAX_INLINE_PREVIEWS}); every image is saved to disk at the paths above."
                    )));
                }
            }
        }
    }
    Ok(CallToolResult::success(blocks))
}

/// Wrap a task snapshot into the response envelope returned by `generate_image`
/// (fast path) and `get_result`: the completed result, an error, or a pending
/// note - always carrying `task_id`, `status`, and `kind`.
pub(crate) fn snapshot_to_envelope(task_id: &str, snap: &TaskSnapshot) -> serde_json::Value {
    let mut env = match snap.status {
        "completed" => snap.result.clone().unwrap_or_else(|| json!({ "ok": true })),
        "failed" => json!({ "ok": false, "error": snap.error }),
        _ => json!({
            "ok": true,
            "message": format!("still generating - call get_result with task_id \"{task_id}\""),
        }),
    };
    env["task_id"] = json!(task_id);
    env["status"] = json!(snap.status);
    env["kind"] = json!(snap.kind);
    env
}

/// Outcome of a background generation job, returned by the closure passed to
/// [`OpenRouterServer::spawn_job_and_wait`].
///
/// - `Ok(result_json)` : the job produced output; `result_json` is stored as the
///   completed task result and the closure has already recorded stats.
/// - `Err(message)`    : the job failed (or produced nothing); `message` is stored
///   as the task error and the closure has already recorded stats.
pub(crate) type JobOutcome = Result<serde_json::Value, String>;

impl OpenRouterServer {
    /// Shared background-job flow used by `generate_image` and `generate_video`:
    /// register a pending task of `kind`, run `run` on a background tokio task
    /// (it owns recording stats and returns the lean result JSON or an error
    /// message), wait up to `wait` seconds for it, then snapshot the task and
    /// build the tool result (with inline previews when requested). The job keeps
    /// running past the timeout, storing its result for `get_result`.
    pub(crate) async fn spawn_job_and_wait<F, Fut>(
        &self,
        kind: TaskKind,
        wait: u64,
        inline_previews: bool,
        run: F,
    ) -> Result<CallToolResult, ErrorData>
    where
        F: FnOnce(OpenRouterClientCtx) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = JobOutcome> + Send + 'static,
    {
        let task_id = uuid::Uuid::now_v7().to_string();
        self.tasks.insert_pending(&task_id, kind).await;

        let ctx = OpenRouterClientCtx {
            client: self.client.clone(),
            stats: self.stats.clone(),
        };
        let tasks = self.tasks.clone();
        let id_bg = task_id.clone();
        let handle = tokio::spawn(async move {
            match run(ctx).await {
                Ok(result_json) => tasks.complete(&id_bg, result_json).await,
                Err(message) => tasks.fail(&id_bg, message).await,
            }
        });

        // Fast-return window: wait up to `wait` seconds, then report whatever
        // state the task is in (dropping the handle leaves it running).
        let _ = tokio::time::timeout(std::time::Duration::from_secs(wait), handle).await;
        let snap = self
            .tasks
            .snapshot(&task_id)
            .await
            .expect("task was just inserted");
        let env = snapshot_to_envelope(&task_id, &snap);
        job_call_result(&env, inline_previews).await
    }
}

/// The cloned client + stats handles a background job needs; passed to the job
/// closure so it can record stats and call the generators.
pub(crate) struct OpenRouterClientCtx {
    pub(crate) client: crate::openrouter::OpenRouterClient,
    pub(crate) stats: UsageStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::valid_png_b64;

    #[test]
    fn encode_preview_blocks_reads_existing_and_skips_missing() {
        // A valid PNG on disk yields one image block.
        let png = base64::engine::general_purpose::STANDARD
            .decode(valid_png_b64())
            .unwrap();
        let dir = std::env::temp_dir();
        let good = dir.join("openrouter-mcp-preview-good.png");
        std::fs::write(&good, &png).unwrap();
        let missing = dir.join("openrouter-mcp-preview-missing.png");
        let _ = std::fs::remove_file(&missing);

        let env = json!({
            "images": [
                { "path": good.to_string_lossy() },
                { "path": missing.to_string_lossy() },
            ]
        });
        let blocks = encode_preview_blocks(&envelope_image_paths(&env));
        assert_eq!(blocks.len(), 1, "only the readable image becomes a block");
        let v = serde_json::to_value(&blocks[0]).unwrap();
        assert_eq!(v["type"], "image");
        assert_eq!(v["mimeType"], "image/png");
        assert!(!v["data"].as_str().unwrap().is_empty());

        // No images array -> no paths -> no blocks.
        assert!(envelope_image_paths(&json!({ "status": "completed" })).is_empty());
    }

    #[test]
    fn encode_preview_blocks_caps_at_the_limit() {
        // Write more readable PNGs than the cap; only MAX_INLINE_PREVIEWS render.
        let png = base64::engine::general_purpose::STANDARD
            .decode(valid_png_b64())
            .unwrap();
        let dir = std::env::temp_dir();
        let paths: Vec<String> = (0..MAX_INLINE_PREVIEWS + 3)
            .map(|i| {
                let p = dir.join(format!("openrouter-mcp-cap-{i}.png"));
                std::fs::write(&p, &png).unwrap();
                p.to_string_lossy().into_owned()
            })
            .collect();
        let blocks = encode_preview_blocks(&paths);
        assert_eq!(blocks.len(), MAX_INLINE_PREVIEWS, "preview count is capped");
    }

    #[tokio::test]
    async fn job_call_result_gates_previews_on_the_flag() {
        let good = std::env::temp_dir().join("openrouter-mcp-gate-good.png");
        let png = base64::engine::general_purpose::STANDARD
            .decode(valid_png_b64())
            .unwrap();
        std::fs::write(&good, &png).unwrap();
        let env = json!({
            "status": "completed",
            "images": [ { "path": good.to_string_lossy() } ],
        });

        let has_image = |res: &CallToolResult| {
            serde_json::to_value(res).unwrap()["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["type"] == "image")
        };
        // Desktop-style: previews on. CLI-style: text only.
        assert!(has_image(&job_call_result(&env, true).await.unwrap()));
        assert!(!has_image(&job_call_result(&env, false).await.unwrap()));

        // A pending job never carries a preview, even with previews enabled.
        let pending = json!({ "status": "pending", "images": [] });
        assert!(!has_image(&job_call_result(&pending, true).await.unwrap()));
    }
}
