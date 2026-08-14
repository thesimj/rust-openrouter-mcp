//! Minimal async REST client for the OpenRouter HTTP API.
//!
//! OpenRouter is an OpenAI-compatible JSON REST API. For now this client only
//! covers `GET /api/v1/models`, which lists every available model along with
//! its capabilities (modalities, context length) and pricing.

mod client;
mod dto;
pub(crate) use dto::*;

use anyhow::{Context, Result};

const BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Default app-attribution values sent on every request. OpenRouter uses these
/// to build the app's page and rankings (purely informational; no effect on
/// pricing or responses). Both are overridable via the `OPENROUTER_HTTP_REFERER`
/// and `OPENROUTER_X_TITLE` env vars.
const APP_REFERER: &str = "https://github.com/thesimj/rust-openrouter-mcp";
const APP_TITLE: &str = "rust-openrouter-mcp";

/// Stall detection, not a deadline. `read_timeout` resets after every successful
/// read, so a slow-but-progressing transfer is never cut off - which matters
/// because `download_video` and `transcribe_audio` move tens of megabytes and
/// have no size bound we control. A total `timeout()` would cap those by wall
/// clock and fail a video generation that already succeeded and was paid for.
const READ_TIMEOUT_SECS: u64 = 60;
/// Separate, because a peer that never completes the TCP/TLS handshake never
/// produces a read for `READ_TIMEOUT_SECS` to bound.
const CONNECT_TIMEOUT_SECS: u64 = 10;

/// Build the shared `reqwest::Client`, attaching the OpenRouter app-attribution
/// headers (`HTTP-Referer` / `X-Title`) as defaults so every endpoint inherits
/// them. Falls back to a bare client if header construction fails.
fn build_http_client() -> reqwest::Client {
    use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
    let referer = std::env::var("OPENROUTER_HTTP_REFERER").unwrap_or_else(|_| APP_REFERER.into());
    let title = std::env::var("OPENROUTER_X_TITLE").unwrap_or_else(|_| APP_TITLE.into());
    let mut headers = HeaderMap::new();
    // Header names are case-insensitive on the wire; `from_static` requires
    // lowercase. OpenRouter documents them as `HTTP-Referer` / `X-Title`.
    if let Ok(v) = HeaderValue::from_str(&referer) {
        headers.insert(HeaderName::from_static("http-referer"), v);
    }
    if let Ok(v) = HeaderValue::from_str(&title) {
        headers.insert(HeaderName::from_static("x-title"), v);
    }
    // reqwest applies no timeout of any kind by default. Without one, a provider
    // that accepts the connection and then stalls hangs the MCP tool call
    // forever, and a stalled background job parks a `Pending` entry that
    // `TaskRegistry::prune_terminal` never evicts - so the registry's cap stops
    // holding. Note this client is not the only one: `server::image::fetch_url`
    // builds its own (pinned to a validated IP) and sets its own bounds.
    reqwest::Client::builder()
        .default_headers(headers)
        .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .read_timeout(std::time::Duration::from_secs(READ_TIMEOUT_SECS))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Thin wrapper around `reqwest::Client` carrying the OpenRouter API key.
#[derive(Clone)]
pub struct OpenRouterClient {
    pub(in crate::openrouter) http: reqwest::Client,
    pub(in crate::openrouter) api_key: String,
    pub(in crate::openrouter) base_url: String,
}

/// Extract the `X-Generation-Id` response header when present.
pub(in crate::openrouter) fn generation_id(resp: &reqwest::Response) -> Option<String> {
    resp.headers()
        .get("x-generation-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// Parse the bare `content-type` MIME (stripping any `; charset=...` suffix),
/// falling back to `default` when the header is missing or unparsable.
pub(in crate::openrouter) fn content_type(resp: &reqwest::Response, default: &str) -> String {
    resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(';').next().unwrap_or(s).trim().to_string())
        .unwrap_or_else(|| default.to_string())
}

/// Most characters of an upstream error body kept in an error message. A JSON
/// provider error is well under this; a proxy's HTML error page is not, and these
/// errors surface in an MCP tool result, so an unbounded body would dump the whole
/// page into the caller's context.
const MAX_ERROR_BODY_CHARS: usize = 500;

/// Bound an upstream error body to [`MAX_ERROR_BODY_CHARS`], marking a cut.
fn truncate_error_body(mut body: String) -> String {
    if let Some((cut, _)) = body.char_indices().nth(MAX_ERROR_BODY_CHARS) {
        body.truncate(cut);
        body.push_str("... [truncated]");
    }
    body
}

impl OpenRouterClient {
    /// Build a client, reading the key from `OPENROUTER_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .context("OPENROUTER_API_KEY environment variable is not set")?;
        Ok(Self {
            http: build_http_client(),
            api_key,
            base_url: BASE_URL.to_string(),
        })
    }

    /// Build a client pointed at an arbitrary base URL. Used by tests to target
    /// a local mock server instead of the live OpenRouter API.
    #[cfg(test)]
    pub(crate) fn with_base_url(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: build_http_client(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// Send a prepared request, surfacing the verbatim upstream error body on a
    /// non-2xx status (OpenRouter wraps provider errors there, so discarding it
    /// loses the only useful diagnostic). `label` is the endpoint path rendered
    /// in both the transport-failure context and the non-success error.
    pub(in crate::openrouter) async fn send_checked(
        &self,
        rb: reqwest::RequestBuilder,
        label: &str,
    ) -> Result<reqwest::Response> {
        let resp = rb
            .send()
            .await
            .with_context(|| format!("request to OpenRouter {label} failed"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = truncate_error_body(resp.text().await.unwrap_or_default());
            anyhow::bail!("OpenRouter {label} returned {status}: {body}");
        }
        Ok(resp)
    }

    /// [`send_checked`](Self::send_checked) plus JSON decoding, deriving the
    /// decode-failure context from the same `label`. The endpoints that need the
    /// raw response first (a header or the body bytes) call `send_checked`.
    pub(in crate::openrouter) async fn send_json<T: serde::de::DeserializeOwned>(
        &self,
        rb: reqwest::RequestBuilder,
        label: &str,
    ) -> Result<T> {
        self.send_checked(rb, label)
            .await?
            .json()
            .await
            .with_context(|| format!("failed to decode OpenRouter {label} response"))
    }
}

/// Default number of models returned by list queries unless `all` is requested.
pub const DEFAULT_MODEL_LIMIT: usize = 20;

/// Result of applying the local `search` filter and the default result cap.
/// `models` is what the caller should display; `total` is how many matched
/// before truncation, so callers can render a "showing X of Y" note.
pub struct FilteredModels {
    pub models: Vec<Model>,
    pub total: usize,
}

impl FilteredModels {
    /// How many matching models the default cap omitted (0 when `all` was set
    /// or nothing was truncated).
    pub fn truncated(&self) -> usize {
        self.total - self.models.len()
    }
}

/// Apply the local case-insensitive `search` filter (across id/name/description)
/// and, unless `all`, cap the result at [`DEFAULT_MODEL_LIMIT`]. Returns the
/// models to display plus the pre-truncation match count. Shared by the CLI
/// `models` command and the `list_models` MCP tool so the two never diverge.
pub fn apply_filters(mut models: Vec<Model>, search: Option<&str>, all: bool) -> FilteredModels {
    if let Some(needle) = search {
        models.retain(|m| m.matches_search(needle));
    }
    let total = models.len();
    if !all {
        models.truncate(DEFAULT_MODEL_LIMIT);
    }
    FilteredModels { models, total }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_error_body_bounds_long_bodies_only() {
        // A normal provider error passes through untouched.
        let short = r#"{"error":"unsupported model"}"#;
        assert_eq!(truncate_error_body(short.to_string()), short);

        // An HTML error page is cut to the cap, and says so.
        let long = "x".repeat(MAX_ERROR_BODY_CHARS * 3);
        let cut = truncate_error_body(long);
        assert!(cut.starts_with(&"x".repeat(MAX_ERROR_BODY_CHARS)));
        assert!(cut.ends_with("... [truncated]"));

        // Cutting mid-multibyte-character must not panic or split a char.
        let wide = "é".repeat(MAX_ERROR_BODY_CHARS * 2);
        assert!(truncate_error_body(wide).starts_with(&"é".repeat(MAX_ERROR_BODY_CHARS)));
    }

    /// Build `n` placeholder models with ids `model-0`, `model-1`, ... so list
    /// filtering/truncation can be exercised without hitting the network.
    fn models(n: usize) -> Vec<Model> {
        (0..n)
            .map(|i| Model {
                id: format!("model-{i}"),
                name: None,
                description: None,
                context_length: None,
                architecture: None,
                pricing: None,
            })
            .collect()
    }

    #[test]
    fn apply_filters_caps_at_default_limit_and_reports_total() {
        let filtered = apply_filters(models(25), None, false);
        assert_eq!(filtered.models.len(), DEFAULT_MODEL_LIMIT);
        assert_eq!(filtered.total, 25);
        assert_eq!(filtered.truncated(), 5);
    }

    #[test]
    fn apply_filters_all_returns_everything_with_no_truncation() {
        let filtered = apply_filters(models(25), None, true);
        assert_eq!(filtered.models.len(), 25);
        assert_eq!(filtered.total, 25);
        assert_eq!(filtered.truncated(), 0);
    }

    #[test]
    fn apply_filters_below_limit_is_not_truncated() {
        let filtered = apply_filters(models(3), None, false);
        assert_eq!(filtered.models.len(), 3);
        assert_eq!(filtered.total, 3);
        assert_eq!(filtered.truncated(), 0);
    }

    #[test]
    fn apply_filters_search_runs_before_truncation() {
        // 30 models; only "model-1", "model-1x", "model-1y"... match "model-1".
        let mut all = models(30);
        all[1].name = Some("special".to_string());
        // Search narrows to ids containing "model-2" => model-2, model-20..29 = 11 matches.
        let filtered = apply_filters(all, Some("model-2"), false);
        assert_eq!(filtered.total, 11);
        assert_eq!(filtered.models.len(), 11); // under the cap, so all kept
        assert!(filtered.models.iter().all(|m| m.id.contains("model-2")));
    }

    #[test]
    fn apply_filters_search_then_cap_reports_pre_truncation_total() {
        // "model-" matches all 25; search keeps 25, cap trims to 20.
        let filtered = apply_filters(models(25), Some("MODEL-"), false);
        assert_eq!(filtered.total, 25);
        assert_eq!(filtered.models.len(), DEFAULT_MODEL_LIMIT);
        assert_eq!(filtered.truncated(), 5);
    }
}
