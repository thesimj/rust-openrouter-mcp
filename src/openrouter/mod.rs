//! Minimal async REST client for the OpenRouter HTTP API.
//!
//! Covers model discovery, account information, chat, images, audio, and video.

mod client;
mod dto;
pub(crate) use dto::*;

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

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
/// can take longer than the generation request. A total `timeout()` would cap those by wall
/// clock and fail a video generation that already succeeded and was paid for.
///
/// The caveat that sets the value: `/images` is synchronous and buffered. It
/// sends no bytes at all until the image is finished, so there is nothing for
/// the timer to reset on and it becomes a generation deadline in practice. At
/// 60s that killed real work - bytedance-seed/seedream-5-0-pro took a measured
/// 154s for one 1K image, and the tool reported "operation timed out" for a
/// request the provider went on to complete and bill. 300s covers the slow
/// image models with headroom; a truly dead connection still dies, just later,
/// and generation runs as a background task so nobody is blocked waiting.
const READ_TIMEOUT_SECS: u64 = 300;
/// Separate, because a peer that never completes the TCP/TLS handshake never
/// produces a read for `READ_TIMEOUT_SECS` to bound.
const CONNECT_TIMEOUT_SECS: u64 = 10;

/// Limits apply to decompressed response bytes, including chunked bodies.
const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;
const MAX_MEDIA_BYTES: usize = 256 * 1024 * 1024;
const MAX_ERROR_BYTES: usize = 2048;
const MAX_UPSTREAM_REQUESTS: usize = 16;
/// Bound upstream retry hints before converting them into timer deadlines.
const MAX_RETRY_AFTER_SECS: u64 = 300;

/// Own admission until the body is consumed or the response is dropped.
pub(in crate::openrouter) struct BoundedResponse {
    response: reqwest::Response,
    _permit: OwnedSemaphorePermit,
}

impl BoundedResponse {
    fn headers(&self) -> &reqwest::header::HeaderMap {
        self.response.headers()
    }

    fn status(&self) -> reqwest::StatusCode {
        self.response.status()
    }

    async fn checked(self, label: &str) -> Result<Self> {
        let status = self.status();
        if !status.is_success() {
            let retry_after = self
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(retry_after);
            let body = error_prefix(self.response).await;
            return Err(HttpFailure {
                status,
                retry_after,
                label: label.to_string(),
                body,
            }
            .into());
        }
        Ok(self)
    }

    async fn read(mut self, limit: usize) -> Result<Vec<u8>> {
        if self
            .response
            .content_length()
            .is_some_and(|n| n > limit as u64)
        {
            anyhow::bail!("OpenRouter response exceeds the {limit}-byte limit");
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = self.response.chunk().await? {
            if chunk.len() > limit.saturating_sub(bytes.len()) {
                anyhow::bail!("OpenRouter response exceeds the {limit}-byte limit");
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }

    async fn json<T: serde::de::DeserializeOwned>(self) -> Result<T> {
        let bytes = self.read(MAX_JSON_BYTES).await?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    async fn bytes(self) -> Result<Vec<u8>> {
        self.read(MAX_MEDIA_BYTES).await
    }
}

/// Keep a small diagnostic prefix, without downloading the entire error body.
async fn error_prefix(mut response: reqwest::Response) -> String {
    let mut bytes = Vec::new();
    while bytes.len() < MAX_ERROR_BYTES {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let keep = chunk.len().min(MAX_ERROR_BYTES - bytes.len());
                bytes.extend_from_slice(&chunk[..keep]);
            }
            _ => break,
        }
    }
    truncate_error_body(String::from_utf8_lossy(&bytes).into_owned())
}

/// Build the shared `reqwest::Client`, attaching the OpenRouter app-attribution
/// headers (`HTTP-Referer` / `X-Title`) as defaults so every endpoint inherits
/// them. Invalid optional headers are omitted. Client construction errors propagate.
fn build_http_client() -> Result<reqwest::Client> {
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
        .tls_backend_rustls()
        .default_headers(headers)
        .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
        .read_timeout(std::time::Duration::from_secs(READ_TIMEOUT_SECS))
        .build()
        .context("could not build OpenRouter HTTP client")
}

/// Retain HTTP metadata so polling can distinguish transient errors.
#[derive(Debug)]
pub(crate) struct HttpFailure {
    pub status: reqwest::StatusCode,
    pub retry_after: Option<std::time::Duration>,
    pub(crate) label: String,
    pub(crate) body: String,
}

impl std::fmt::Display for HttpFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OpenRouter {} returned {}: {}",
            self.label, self.status, self.body
        )
    }
}

impl std::error::Error for HttpFailure {}

fn retry_after(value: &str) -> Option<std::time::Duration> {
    if let Ok(seconds) = value.trim().parse::<u64>() {
        return Some(std::time::Duration::from_secs(
            seconds.min(MAX_RETRY_AFTER_SECS),
        ));
    }
    let at = chrono::DateTime::parse_from_rfc2822(value).ok()?;
    Some(
        (at.with_timezone(&chrono::Utc) - chrono::Utc::now())
            .to_std()
            .unwrap_or_default()
            .min(std::time::Duration::from_secs(MAX_RETRY_AFTER_SECS)),
    )
}

/// Thin wrapper around `reqwest::Client` carrying the OpenRouter API key.
#[derive(Clone)]
pub struct OpenRouterClient {
    pub(in crate::openrouter) http: reqwest::Client,
    pub(in crate::openrouter) api_key: String,
    pub(in crate::openrouter) base_url: String,
    requests: Arc<Semaphore>,
}

/// Extract the `X-Generation-Id` response header when present.
pub(in crate::openrouter) fn generation_id(resp: &BoundedResponse) -> Option<String> {
    resp.headers()
        .get("x-generation-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// Parse the bare `content-type` MIME (stripping any `; charset=...` suffix),
/// falling back to `default` when the header is missing or unparsable.
pub(in crate::openrouter) fn content_type(resp: &BoundedResponse, default: &str) -> String {
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
            http: build_http_client()?,
            api_key,
            base_url: BASE_URL.to_string(),
            requests: Arc::new(Semaphore::new(MAX_UPSTREAM_REQUESTS)),
        })
    }

    /// Build a client pointed at an arbitrary base URL. Used by tests to target
    /// a local mock server instead of the live OpenRouter API.
    #[cfg(test)]
    pub(crate) fn with_base_url(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: build_http_client().expect("test HTTP client"),
            api_key: api_key.into(),
            base_url: base_url.into(),
            requests: Arc::new(Semaphore::new(MAX_UPSTREAM_REQUESTS)),
        }
    }

    /// Hold shared admission from request send until body consumption finishes.
    async fn send_response(
        &self,
        rb: reqwest::RequestBuilder,
        label: &str,
    ) -> Result<BoundedResponse> {
        let permit = self
            .requests
            .clone()
            .acquire_owned()
            .await
            .context("OpenRouter request admission closed")?;
        let response = rb
            .send()
            .await
            .with_context(|| format!("request to OpenRouter {label} failed"))?;
        Ok(BoundedResponse {
            response,
            _permit: permit,
        })
    }

    /// Send a prepared request and retain a bounded diagnostic on HTTP failure.
    pub(in crate::openrouter) async fn send_checked(
        &self,
        rb: reqwest::RequestBuilder,
        label: &str,
    ) -> Result<BoundedResponse> {
        self.send_response(rb, label).await?.checked(label).await
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

    #[tokio::test]
    async fn rejects_declared_and_chunked_oversize_bodies() {
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![1; 9]))
            .mount(&server)
            .await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let response = client
            .send_checked(client.http.get(server.uri()), "/test")
            .await
            .unwrap();
        assert!(
            response
                .read(8)
                .await
                .unwrap_err()
                .to_string()
                .contains("8-byte limit")
        );

        // No Content-Length: enforce the cap after accumulating separate chunks.
        let (url, sender) =
            chunked_response("200 OK", vec![b"12345".to_vec(), b"6789".to_vec()], false).await;
        let response = client
            .send_checked(client.http.get(url), "/test")
            .await
            .unwrap();
        assert!(
            response
                .read(8)
                .await
                .unwrap_err()
                .to_string()
                .contains("8-byte limit")
        );
        sender.await.unwrap();
    }

    async fn chunked_response(
        status: &str,
        chunks: Vec<Vec<u8>>,
        stall: bool,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let header = format!(
            "HTTP/1.1 {status}\r\nTransfer-Encoding: chunked\r\nRetry-After: 18446744073709551615\r\nConnection: close\r\n\r\n"
        );
        let sender = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            let received = socket.read(&mut request).await.unwrap();
            assert!(received > 0);
            socket.write_all(header.as_bytes()).await.unwrap();
            for chunk in chunks {
                socket
                    .write_all(format!("{:x}\r\n", chunk.len()).as_bytes())
                    .await
                    .unwrap();
                socket.write_all(&chunk).await.unwrap();
                socket.write_all(b"\r\n").await.unwrap();
            }
            if stall {
                std::future::pending::<()>().await;
            } else {
                socket.write_all(b"0\r\n\r\n").await.unwrap();
            }
        });
        (url, sender)
    }

    #[tokio::test]
    async fn error_prefix_stops_reading_and_retains_retry_metadata() {
        let (url, sender) = chunked_response(
            "429 Too Many Requests",
            vec![vec![b'x'; MAX_ERROR_BYTES]],
            true,
        )
        .await;
        let client = OpenRouterClient::with_base_url(&url, "test-secret");
        let error = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            client.send_checked(client.http.get(url), "/test"),
        )
        .await
        .expect("bounded prefix must not wait for EOF")
        .err()
        .unwrap();
        sender.abort();
        let failure = error.downcast_ref::<HttpFailure>().unwrap();
        assert_eq!(failure.status, reqwest::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            failure.retry_after,
            Some(std::time::Duration::from_secs(300))
        );
        assert!(failure.body.ends_with("[truncated]"));
        assert!(failure.body.len() < 600);
        assert!(!error.to_string().contains("test-secret"));
    }

    #[tokio::test]
    async fn absent_image_endpoint_does_not_wait_for_error_body() {
        let (url, sender) = chunked_response("404 Not Found", vec![], true).await;
        let client = OpenRouterClient::with_base_url(url, "test-key");
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            client.image_model_detail("test/model"),
        )
        .await
        .expect("404 needs no body")
        .unwrap();
        sender.abort();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn admission_is_shared_and_held_until_response_body_is_dropped() {
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok":true})))
            .mount(&server)
            .await;
        let mut client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        client.requests = Arc::new(Semaphore::new(1));
        let first = client
            .send_checked(client.http.get(server.uri()), "/test")
            .await
            .unwrap();
        let clone = client.clone();
        let second = clone.send_json::<serde_json::Value>(clone.http.get(server.uri()), "/test");
        tokio::pin!(second);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(30), &mut second)
                .await
                .is_err()
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        drop(first);
        assert_eq!(second.await.unwrap()["ok"], true);
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
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
                reasoning: None,
                supported_voices: None,
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

#[cfg(test)]
mod audit_regression {
    use super::*;
    #[test]
    fn retry_after_parses_seconds_dates_and_invalid_values() {
        assert_eq!(retry_after("7"), Some(std::time::Duration::from_secs(7)));
        assert_eq!(
            retry_after("Wed, 21 Oct 2015 07:28:00 GMT"),
            Some(std::time::Duration::ZERO)
        );
        assert!(retry_after("invalid").is_none());
    }
    #[test]
    fn rustls_client_builds_with_the_selected_backend() {
        assert!(build_http_client().is_ok());
        let manifest = include_str!("../../Cargo.toml");
        let reqwest = manifest
            .lines()
            .find(|line| line.starts_with("reqwest ="))
            .unwrap();
        assert!(reqwest.contains("\"rustls\""));
        assert!(!reqwest.contains("\"native-tls\""));
    }
}
