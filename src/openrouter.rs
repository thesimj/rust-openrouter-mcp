//! Minimal async REST client for the OpenRouter HTTP API.
//!
//! OpenRouter is an OpenAI-compatible JSON REST API. For now this client only
//! covers `GET /api/v1/models`, which lists every available model along with
//! its capabilities (modalities, context length) and pricing.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Thin wrapper around `reqwest::Client` carrying the OpenRouter API key.
#[derive(Clone)]
pub struct OpenRouterClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl OpenRouterClient {
    /// Build a client, reading the key from `OPENROUTER_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .context("OPENROUTER_API_KEY environment variable is not set")?;
        Ok(Self {
            http: reqwest::Client::new(),
            api_key,
            base_url: BASE_URL.to_string(),
        })
    }

    /// Build a client pointed at an arbitrary base URL. Used by tests to target
    /// a local mock server instead of the live OpenRouter API.
    #[cfg(test)]
    pub(crate) fn with_base_url(base_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    /// `GET /api/v1/models` - every model with capabilities and pricing.
    ///
    /// `query` carries OpenRouter's server-side filters (modalities, sort,
    /// free-text, price/context bounds, ...) so the API does the filtering.
    pub async fn list_models(&self, query: &ModelsQuery) -> Result<Vec<Model>> {
        let resp = self
            .http
            .get(format!("{}/models", self.base_url))
            .bearer_auth(&self.api_key)
            .query(&query.to_pairs())
            .send()
            .await
            .context("request to OpenRouter /models failed")?
            .error_for_status()
            .context("OpenRouter /models returned an error status")?;

        let parsed: ModelsResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /models response")?;
        Ok(parsed.data)
    }
}

impl OpenRouterClient {
    /// `GET /api/v1/videos/models` - video-generation models with `pricing_skus`
    /// (per video-second / per video-token), resolutions, durations, etc.
    pub async fn list_video_models(&self) -> Result<Vec<VideoModel>> {
        let resp = self
            .http
            .get(format!("{}/videos/models", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("request to OpenRouter /videos/models failed")?
            .error_for_status()
            .context("OpenRouter /videos/models returned an error status")?;

        let parsed: VideoModelsResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /videos/models response")?;
        Ok(parsed.data)
    }
}

impl OpenRouterClient {
    /// `GET /api/v1/key` - basic information about the API key in use: its label,
    /// the owning `creator_user_id`, credit usage (total and per period), spending
    /// limit / remaining balance, tier/management flags, and the (deprecated)
    /// rate limit. This is key/account-level info, not the owner's name or email.
    pub async fn get_key_info(&self) -> Result<KeyInfo> {
        let resp = self
            .http
            .get(format!("{}/key", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("request to OpenRouter /key failed")?
            .error_for_status()
            .context("OpenRouter /key returned an error status")?;

        let parsed: KeyInfoResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /key response")?;
        Ok(parsed.data)
    }
}

#[derive(Debug, Deserialize)]
pub struct KeyInfoResponse {
    pub data: KeyInfo,
}

/// Basic information about the API key in use (`GET /api/v1/key`). Every field is
/// optional/defaulted: the upstream schema evolves, and `limit`/`limit_remaining`
/// are `null` for unlimited keys. Fields OpenRouter returns but we don't surface
/// (e.g. `limit_reset`, `expires_at`, BYOK period breakdowns) are ignored.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct KeyInfo {
    /// Human-readable label, usually a masked key (e.g. "sk-or-v1-813...ca1").
    #[serde(default)]
    pub label: Option<String>,
    /// Opaque id of the user who owns the key (closest available "owner" identity).
    #[serde(default)]
    pub creator_user_id: Option<String>,
    /// Whether this is a free-tier key.
    #[serde(default)]
    pub is_free_tier: Option<bool>,
    /// Whether this key can provision (create/manage) other keys.
    #[serde(default)]
    pub is_provisioning_key: Option<bool>,
    /// Whether this is an account management key.
    #[serde(default)]
    pub is_management_key: Option<bool>,
    /// Spending cap in USD; `None` (null upstream) means unlimited.
    #[serde(default)]
    pub limit: Option<f64>,
    /// Remaining balance in USD; `None` means unlimited.
    #[serde(default)]
    pub limit_remaining: Option<f64>,
    /// Total credits consumed (USD).
    #[serde(default)]
    pub usage: Option<f64>,
    /// Credits consumed today (USD).
    #[serde(default)]
    pub usage_daily: Option<f64>,
    /// Credits consumed this week (USD).
    #[serde(default)]
    pub usage_weekly: Option<f64>,
    /// Credits consumed this month (USD).
    #[serde(default)]
    pub usage_monthly: Option<f64>,
    /// Spend on bring-your-own-key providers (USD), not billed as credits.
    #[serde(default)]
    pub byok_usage: Option<f64>,
    /// Legacy rate-limit descriptor (deprecated upstream; kept for completeness).
    #[serde(default)]
    pub rate_limit: Option<RateLimit>,
}

/// Legacy per-key rate limit. `requests` is signed because OpenRouter returns
/// `-1` to mean "no limit"; the field is deprecated and safe to ignore.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RateLimit {
    #[serde(default)]
    pub requests: Option<i64>,
    #[serde(default)]
    pub interval: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct VideoModelsResponse {
    pub data: Vec<VideoModel>,
}

/// A video-generation model from `/videos/models`. `pricing_skus` maps a SKU
/// name (e.g. `duration_seconds_with_audio`, `video_tokens`) to a price string.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VideoModel {
    pub id: String,
    #[serde(default)]
    pub pricing_skus: BTreeMap<String, String>,
}

/// Server-side query parameters for `GET /api/v1/models`. Every field is
/// optional; `None`/empty fields are omitted from the request.
#[derive(Debug, Default)]
pub struct ModelsQuery {
    /// Free-text search by model name or slug (`q`).
    pub q: Option<String>,
    /// Comma list of output modalities: text, image, audio, embeddings, all.
    pub output_modalities: Option<String>,
    /// Comma list of input modalities: text, image, audio, file.
    pub input_modalities: Option<String>,
    /// Comma list of required supported parameters, e.g. "tools".
    pub supported_parameters: Option<String>,
    /// Server-side sort, e.g. "newest", "most-popular", "pricing-low-to-high".
    pub sort: Option<String>,
    /// Minimum context length in tokens.
    pub context: Option<u64>,
}

impl ModelsQuery {
    fn to_pairs(&self) -> Vec<(&'static str, String)> {
        let mut pairs = Vec::new();
        if let Some(v) = &self.q {
            pairs.push(("q", v.clone()));
        }
        if let Some(v) = &self.output_modalities {
            pairs.push(("output_modalities", v.clone()));
        }
        if let Some(v) = &self.input_modalities {
            pairs.push(("input_modalities", v.clone()));
        }
        if let Some(v) = &self.supported_parameters {
            pairs.push(("supported_parameters", v.clone()));
        }
        if let Some(v) = &self.sort {
            pairs.push(("sort", v.clone()));
        }
        if let Some(v) = &self.context {
            pairs.push(("context", v.to_string()));
        }
        pairs
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<Model>,
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

impl OpenRouterClient {
    /// `POST /api/v1/chat/completions` - used for image generation (and, later,
    /// text/vision). Returns the parsed completion plus the `X-Generation-Id`
    /// response header when present. On a non-2xx status the upstream error body
    /// is surfaced verbatim (OpenRouter wraps provider errors there).
    pub async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req)
            .send()
            .await
            .context("request to OpenRouter /chat/completions failed")?;

        let generation_id = resp
            .headers()
            .get("x-generation-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter /chat/completions returned {status}: {body}");
        }

        let completion: ChatCompletion = resp
            .json()
            .await
            .context("failed to decode OpenRouter /chat/completions response")?;
        Ok(ChatResponse {
            completion,
            generation_id,
        })
    }
}

impl OpenRouterClient {
    /// `POST /api/v1/videos` - submit an asynchronous video-generation job. This
    /// is **not** the chat endpoint: it returns `202` with a job id to poll. On a
    /// non-2xx status the upstream error body is surfaced verbatim.
    pub async fn submit_video(&self, req: &VideoSubmitBody) -> Result<VideoSubmitResponse> {
        let resp = self
            .http
            .post(format!("{}/videos", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req)
            .send()
            .await
            .context("request to OpenRouter /videos failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter /videos returned {status}: {body}");
        }
        let parsed: VideoSubmitResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /videos submit response")?;
        Ok(parsed)
    }

    /// `GET /api/v1/videos/{id}` - poll a submitted video job for its status and,
    /// once complete, the (unsigned) download URLs and usage.
    pub async fn poll_video(&self, job_id: &str) -> Result<VideoPollResponse> {
        let resp = self
            .http
            .get(format!("{}/videos/{job_id}", self.base_url))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("request to OpenRouter /videos/{id} failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter /videos/{job_id} returned {status}: {body}");
        }
        let parsed: VideoPollResponse = resp
            .json()
            .await
            .context("failed to decode OpenRouter /videos/{id} response")?;
        Ok(parsed)
    }

    /// `GET /api/v1/videos/{id}/content?index=N` - download one generated clip.
    /// Returns `(content_type, bytes)`; the content type (e.g. `video/mp4`) is
    /// used to choose the file extension.
    pub async fn download_video(&self, job_id: &str, index: usize) -> Result<(String, Vec<u8>)> {
        let resp = self
            .http
            .get(format!("{}/videos/{job_id}/content", self.base_url))
            .query(&[("index", index.to_string())])
            .bearer_auth(&self.api_key)
            .send()
            .await
            .context("request to OpenRouter /videos/{id}/content failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter /videos/{job_id}/content returned {status}: {body}");
        }
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(';').next().unwrap_or(s).trim().to_string())
            .unwrap_or_else(|| "video/mp4".to_string());
        let bytes = resp
            .bytes()
            .await
            .context("failed to read video content bytes")?
            .to_vec();
        Ok((content_type, bytes))
    }

    /// `POST /api/v1/audio/speech` - synchronous text-to-speech. Returns the raw
    /// audio bytes (OpenAI-Speech-compatible), the content type, and the
    /// `X-Generation-Id` header when present. On a non-2xx status the upstream
    /// error body is surfaced verbatim.
    pub async fn speech(&self, req: &SpeechBody) -> Result<SpeechResult> {
        let resp = self
            .http
            .post(format!("{}/audio/speech", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req)
            .send()
            .await
            .context("request to OpenRouter /audio/speech failed")?;

        let generation_id = resp
            .headers()
            .get("x-generation-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter /audio/speech returned {status}: {body}");
        }
        let mime = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(';').next().unwrap_or(s).trim().to_string())
            .unwrap_or_else(|| "audio/mpeg".to_string());
        let bytes = resp
            .bytes()
            .await
            .context("failed to read speech audio bytes")?
            .to_vec();
        Ok(SpeechResult {
            mime,
            bytes,
            generation_id,
        })
    }
}

/// Request body for `POST /api/v1/videos`. Optional fields are omitted when
/// unset (named `*Body` to avoid colliding with the domain `video_gen` struct).
#[derive(Debug, Serialize)]
pub struct VideoSubmitBody {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub frame_images: Vec<FrameImage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub input_references: Vec<InputReference>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_audio: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// A first/last frame for image-to-video (`frame_type` is `first_frame` or
/// `last_frame`), sent as a data-URL `image_url`.
#[derive(Debug, Serialize)]
pub struct FrameImage {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub image_url: ImageUrl,
    pub frame_type: String,
}

impl FrameImage {
    pub fn new(image_url: ImageUrl, frame_type: String) -> Self {
        Self {
            kind: "image_url",
            image_url,
            frame_type,
        }
    }
}

/// A reference image for reference-to-video, sent as a data-URL `image_url`.
#[derive(Debug, Serialize)]
pub struct InputReference {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub image_url: ImageUrl,
}

impl InputReference {
    pub fn new(image_url: ImageUrl) -> Self {
        Self {
            kind: "image_url",
            image_url,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct VideoSubmitResponse {
    pub id: String,
    // Captured for completeness; we poll by id rather than following these.
    #[allow(dead_code)]
    #[serde(default)]
    pub polling_url: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct VideoPollResponse {
    #[allow(dead_code)]
    pub id: String,
    #[serde(default)]
    pub generation_id: Option<String>,
    pub status: String,
    #[serde(default)]
    pub unsigned_urls: Vec<String>,
    #[serde(default)]
    pub usage: Option<VideoUsage>,
}

#[derive(Debug, Deserialize)]
pub struct VideoUsage {
    #[serde(default)]
    pub cost: Option<f64>,
    // Parsed defensively; not surfaced today.
    #[allow(dead_code)]
    #[serde(default)]
    pub is_byok: Option<bool>,
}

/// Request body for `POST /api/v1/audio/speech`. `response_format`/`speed` are
/// omitted when unset.
#[derive(Debug, Serialize)]
pub struct SpeechBody {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f64>,
}

/// Raw audio bytes from `/audio/speech`, constructed from the response (not
/// deserialized): the MIME type, bytes, and optional generation id.
pub struct SpeechResult {
    pub mime: String,
    pub bytes: Vec<u8>,
    pub generation_id: Option<String>,
}

/// A chat-completions request. `image_config`/`seed` are omitted when `None`.
/// `stream` is always sent as `false` (MCP tools return one complete result).
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    /// Output modalities; omitted for plain text-output (vision/describe) calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_config: Option<ImageConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct Message {
    pub role: String,
    pub content: Content,
}

/// Message content: either a plain string or an ordered list of parts
/// (text-first, then images) for editing/multi-image requests.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

/// `image_config` block controlling aspect ratio and resolution tier.
#[derive(Debug, Serialize)]
pub struct ImageConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_size: Option<String>,
}

/// A `{ "url": ... }` image reference, used both in requests (data URLs) and
/// in responses (generated-image data URLs).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub completion: ChatCompletion,
    pub generation_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletion {
    #[serde(default)]
    pub id: Option<String>,
    // `model`/`finish_reason` are not yet surfaced; `provider` feeds the manifest.
    #[allow(dead_code)]
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: ResponseMessage,
    #[allow(dead_code)]
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Assistant message in a response. `content` is null for image-only output;
/// generated images arrive in `images`.
#[derive(Debug, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub images: Vec<OutImage>,
}

#[derive(Debug, Deserialize)]
pub struct OutImage {
    pub image_url: ImageUrl,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    /// Actual cost in USD reported by OpenRouter, when available.
    #[serde(default)]
    pub cost: Option<f64>,
}

/// A single OpenRouter model entry. Fields are optional/defaulted defensively
/// because the upstream schema evolves and varies per provider.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Model {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    #[serde(default)]
    pub architecture: Option<Architecture>,
    #[serde(default)]
    pub pricing: Option<Pricing>,
}

impl Model {
    /// Case-insensitive match of `needle` against the model id, name, and
    /// description. Used by the `search` filter in both the CLI and MCP tool.
    pub fn matches_search(&self, needle: &str) -> bool {
        let needle = needle.to_lowercase();
        self.id.to_lowercase().contains(&needle)
            || self
                .name
                .as_deref()
                .is_some_and(|n| n.to_lowercase().contains(&needle))
            || self
                .description
                .as_deref()
                .is_some_and(|d| d.to_lowercase().contains(&needle))
    }
}

/// Capability descriptor: which input/output modalities a model supports.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Architecture {
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default)]
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub output_modalities: Vec<String>,
    #[serde(default)]
    pub tokenizer: Option<String>,
}

/// Per-unit pricing, reported by OpenRouter as decimal strings (USD per unit).
/// Mirrors the official SDK's `PublicPricing`; all fields beyond prompt/
/// completion are optional and provider-dependent.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Pricing {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
    #[serde(default)]
    pub request: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
    /// Per generated-image cost (also exposed on per-endpoint detail).
    #[serde(default)]
    pub image_output: Option<String>,
    /// Per image-token cost.
    #[serde(default)]
    pub image_token: Option<String>,
    #[serde(default)]
    pub audio: Option<String>,
    /// Per audio-output cost.
    #[serde(default)]
    pub audio_output: Option<String>,
    #[serde(default)]
    pub web_search: Option<String>,
    #[serde(default)]
    pub internal_reasoning: Option<String>,
    #[serde(default)]
    pub input_audio_cache: Option<String>,
    #[serde(default)]
    pub input_cache_read: Option<String>,
    #[serde(default)]
    pub input_cache_write: Option<String>,
    /// Fractional discount applied to the above (numeric, not a price string).
    #[serde(default)]
    pub discount: Option<f64>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[tokio::test]
    async fn list_models_sends_query_params_and_parses_data() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .and(query_param("q", "openai"))
            .and(query_param("sort", "newest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {"id": "openai/gpt", "name": "GPT", "context_length": 128000}
                ]
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let query = ModelsQuery {
            q: Some("openai".to_string()),
            sort: Some("newest".to_string()),
            ..Default::default()
        };
        let models = client.list_models(&query).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "openai/gpt");
        assert_eq!(models[0].context_length, Some(128_000));
    }

    #[tokio::test]
    async fn list_models_errors_on_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "bad-key");
        let err = client
            .list_models(&ModelsQuery::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("error status"));
    }

    #[tokio::test]
    async fn list_video_models_parses_pricing_skus() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/videos/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {"id": "google/veo", "pricing_skus": {"duration_seconds": "0.1"}}
                ]
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let vms = client.list_video_models().await.unwrap();
        assert_eq!(vms.len(), 1);
        assert_eq!(vms[0].id, "google/veo");
        assert_eq!(
            vms[0]
                .pricing_skus
                .get("duration_seconds")
                .map(String::as_str),
            Some("0.1")
        );
    }

    #[tokio::test]
    async fn get_key_info_parses_live_shaped_response() {
        let server = MockServer::start().await;
        // Body mirrors the real GET /api/v1/key payload (extra fields ignored,
        // rate_limit.requests is -1 = unlimited, limit is null = unlimited).
        Mock::given(method("GET"))
            .and(path("/key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "label": "sk-or-v1-813...ca1",
                    "creator_user_id": "user_39wcop8",
                    "is_free_tier": false,
                    "is_provisioning_key": false,
                    "is_management_key": false,
                    "limit": null,
                    "limit_reset": null,
                    "limit_remaining": null,
                    "expires_at": null,
                    "usage": 63.17,
                    "usage_daily": 3.25,
                    "usage_weekly": 10.28,
                    "usage_monthly": 10.46,
                    "byok_usage": 0,
                    "byok_usage_daily": 0,
                    "rate_limit": { "requests": -1, "interval": "10s", "note": "deprecated" }
                }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let info = client.get_key_info().await.unwrap();
        assert_eq!(info.label.as_deref(), Some("sk-or-v1-813...ca1"));
        assert_eq!(info.creator_user_id.as_deref(), Some("user_39wcop8"));
        assert_eq!(info.is_free_tier, Some(false));
        assert_eq!(info.is_management_key, Some(false));
        assert!(info.limit.is_none(), "null limit -> unlimited");
        assert!(info.limit_remaining.is_none());
        assert_eq!(info.usage, Some(63.17));
        assert_eq!(info.usage_monthly, Some(10.46));
        assert_eq!(info.byok_usage, Some(0.0));
        let rl = info.rate_limit.unwrap();
        assert_eq!(rl.requests, Some(-1));
        assert_eq!(rl.interval.as_deref(), Some("10s"));
    }

    #[tokio::test]
    async fn get_key_info_errors_on_non_success_status() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/key"))
            .respond_with(ResponseTemplate::new(401))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "bad-key");
        let err = client.get_key_info().await.unwrap_err();
        assert!(err.to_string().contains("error status"));
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

    #[test]
    fn query_pairs_omit_empty_fields_and_keep_expected_names() {
        let query = ModelsQuery {
            q: Some("openai".to_string()),
            output_modalities: Some("image,text".to_string()),
            input_modalities: None,
            supported_parameters: Some("tools".to_string()),
            sort: Some("newest".to_string()),
            context: Some(128_000),
        };

        assert_eq!(
            query.to_pairs(),
            vec![
                ("q", "openai".to_string()),
                ("output_modalities", "image,text".to_string()),
                ("supported_parameters", "tools".to_string()),
                ("sort", "newest".to_string()),
                ("context", "128000".to_string()),
            ]
        );
    }

    #[test]
    fn matches_search_checks_id_name_and_description_case_insensitively() {
        let model = Model {
            id: "openai/gpt-audio-mini".to_string(),
            name: Some("OpenAI: GPT Audio Mini".to_string()),
            description: Some("A cost-efficient audio model.".to_string()),
            context_length: None,
            architecture: None,
            pricing: None,
        };

        assert!(model.matches_search("OPENAI"));
        assert!(model.matches_search("audio mini"));
        assert!(model.matches_search("cost-efficient"));
        assert!(!model.matches_search("anthropic"));
    }

    #[test]
    fn models_response_decodes_missing_optional_fields() {
        let json = r#"{
          "data": [
            {
              "id": "provider/minimal"
            }
          ]
        }"#;

        let parsed: ModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.data.len(), 1);
        let model = &parsed.data[0];
        assert_eq!(model.id, "provider/minimal");
        assert!(model.name.is_none());
        assert!(model.architecture.is_none());
        assert!(model.pricing.is_none());
    }

    #[test]
    fn models_response_decodes_capabilities_and_pricing() {
        let json = r#"{
          "data": [
            {
              "id": "openai/example",
              "name": "OpenAI Example",
              "description": "Example model",
              "context_length": 400000,
              "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
                "tokenizer": "GPT"
              },
              "pricing": {
                "prompt": "0.00000125",
                "completion": "0.00001",
                "web_search": "0.01",
                "discount": 0.5
              }
            }
          ]
        }"#;

        let parsed: ModelsResponse = serde_json::from_str(json).unwrap();
        let model = &parsed.data[0];
        assert_eq!(model.context_length, Some(400_000));

        let arch = model.architecture.as_ref().unwrap();
        assert_eq!(arch.input_modalities, vec!["text", "image"]);
        assert_eq!(arch.output_modalities, vec!["text"]);
        assert_eq!(arch.tokenizer.as_deref(), Some("GPT"));

        let pricing = model.pricing.as_ref().unwrap();
        assert_eq!(pricing.prompt.as_deref(), Some("0.00000125"));
        assert_eq!(pricing.completion.as_deref(), Some("0.00001"));
        assert_eq!(pricing.web_search.as_deref(), Some("0.01"));
        assert_eq!(pricing.discount, Some(0.5));
        assert!(pricing.image.is_none());
    }

    #[tokio::test]
    async fn submit_video_posts_body_and_parses_job_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .and(body_partial_json(
                json!({ "model": "google/veo-3.1", "prompt": "a dog" }),
            ))
            // The async job API responds 202 with a job id and polling url.
            .respond_with(ResponseTemplate::new(202).set_body_json(json!({
                "id": "vid-1",
                "polling_url": "https://openrouter.ai/api/v1/videos/vid-1",
                "status": "pending"
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let body = VideoSubmitBody {
            model: "google/veo-3.1".to_string(),
            prompt: "a dog".to_string(),
            duration: Some(4),
            resolution: None,
            aspect_ratio: Some("16:9".to_string()),
            size: None,
            frame_images: vec![],
            input_references: vec![],
            generate_audio: Some(false),
            seed: None,
        };
        let resp = client.submit_video(&body).await.unwrap();
        assert_eq!(resp.id, "vid-1");
    }

    #[tokio::test]
    async fn submit_video_surfaces_error_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/videos"))
            .respond_with(ResponseTemplate::new(400).set_body_string("{\"error\":\"unsupported\"}"))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let body = VideoSubmitBody {
            model: "m".to_string(),
            prompt: "p".to_string(),
            duration: None,
            resolution: None,
            aspect_ratio: None,
            size: None,
            frame_images: vec![],
            input_references: vec![],
            generate_audio: None,
            seed: None,
        };
        let err = client.submit_video(&body).await.unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    #[tokio::test]
    async fn poll_video_parses_status_urls_and_usage() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/videos/vid-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "vid-1",
                "generation_id": "gen-7",
                "status": "completed",
                "unsigned_urls": ["https://cdn/0.mp4", "https://cdn/1.mp4"],
                "usage": { "cost": 0.9, "is_byok": false }
            })))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let poll = client.poll_video("vid-1").await.unwrap();
        assert_eq!(poll.status, "completed");
        assert_eq!(poll.generation_id.as_deref(), Some("gen-7"));
        assert_eq!(poll.unsigned_urls.len(), 2);
        assert_eq!(poll.usage.unwrap().cost, Some(0.9));
    }

    #[tokio::test]
    async fn download_video_returns_content_type_and_bytes() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/videos/vid-1/content"))
            .and(query_param("index", "2"))
            .respond_with(
                ResponseTemplate::new(200)
                    // A charset suffix must be stripped to the bare MIME type.
                    .insert_header("content-type", "video/webm; charset=binary")
                    .set_body_bytes(b"WEBM".to_vec()),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let (mime, bytes) = client.download_video("vid-1", 2).await.unwrap();
        assert_eq!(mime, "video/webm");
        assert_eq!(bytes, b"WEBM");
    }

    #[tokio::test]
    async fn speech_returns_bytes_mime_and_generation_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .and(body_partial_json(json!({
                "model": "openai/gpt-4o-mini-tts",
                "input": "hi",
                "voice": "alloy"
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .insert_header("x-generation-id", "gen-aud-3")
                    .set_body_bytes(b"MP3".to_vec()),
            )
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let body = SpeechBody {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: "hi".to_string(),
            voice: "alloy".to_string(),
            response_format: Some("mp3".to_string()),
            speed: None,
        };
        let result = match client.speech(&body).await {
            Ok(r) => r,
            Err(e) => panic!("speech should succeed: {e}"),
        };
        assert_eq!(result.mime, "audio/mpeg");
        assert_eq!(result.bytes, b"MP3");
        assert_eq!(result.generation_id.as_deref(), Some("gen-aud-3"));
    }

    #[tokio::test]
    async fn speech_surfaces_error_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(ResponseTemplate::new(422).set_body_string("{\"error\":\"bad voice\"}"))
            .mount(&server)
            .await;

        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let body = SpeechBody {
            model: "m".to_string(),
            input: "x".to_string(),
            voice: "z".to_string(),
            response_format: None,
            speed: None,
        };
        let err = match client.speech(&body).await {
            Err(e) => e,
            Ok(_) => panic!("provider error should propagate"),
        };
        assert!(err.to_string().contains("bad voice"));
    }
}
