//! DTOs for `GET /api/v1/models`: the query parameters, the model entry, and
//! its capability/pricing descriptors.

use serde::{Deserialize, Serialize};

/// Server-side query parameters for `GET /api/v1/models`. Every field is
/// optional; `None` fields are omitted from the query string by serde_urlencoded
/// (which reqwest's `.query()` uses).
#[derive(Debug, Default, Serialize)]
pub struct ModelsQuery {
    /// Free-text search by model name or slug (`q`).
    pub q: Option<String>,
    /// Comma list of output modalities: text, image, audio, embeddings, video,
    /// rerank, speech, transcription - or all. (`music` is not a value: music
    /// models are the `audio` ones.)
    pub output_modalities: Option<String>,
    /// Comma list of input modalities: text, image, audio, file.
    pub input_modalities: Option<String>,
    /// Comma list of required supported parameters, e.g. "tools".
    pub supported_parameters: Option<String>,
    /// Server-side sort: most-popular, newest, top-weekly, pricing-low-to-high,
    /// pricing-high-to-low, context-high-to-low, throughput-high-to-low,
    /// latency-low-to-high, intelligence-high-to-low, coding-high-to-low,
    /// agentic-high-to-low, design-arena-elo-high-to-low.
    pub sort: Option<String>,
    /// Minimum context length in tokens.
    pub context: Option<u64>,
    /// Use-case category: programming, roleplay, marketing, marketing/seo,
    /// technology, science, translation, legal, finance, health, trivia, academia.
    pub category: Option<String>,
    /// Comma list of hosting provider names (e.g. "OpenAI,Anthropic").
    pub providers: Option<String>,
    /// Comma list of model author slugs (e.g. "openai,anthropic").
    pub model_authors: Option<String>,
    /// Architecture / model family (e.g. "GPT", "Claude", "Gemini", "Llama").
    pub arch: Option<String>,
    /// Prompt price bounds in $/M tokens.
    pub min_price: Option<f64>,
    pub max_price: Option<f64>,
    /// Completion price bounds in $/M tokens.
    pub min_output_price: Option<f64>,
    pub max_output_price: Option<f64>,
    /// Only models with zero-data-retention endpoints. The API accepts only
    /// the string "true", so `Some(false)` is treated as "no filter" and
    /// omitted rather than sent.
    #[serde(skip_serializing_if = "is_not_true")]
    pub zdr: Option<bool>,
    /// Data region of the model's endpoints: "eu" or "us".
    pub region: Option<String>,
    /// "true" = only distillable models, "false" = exclude them.
    pub distillable: Option<bool>,
    /// Model age bounds in days since creation.
    pub min_age_days: Option<u64>,
    pub max_age_days: Option<u64>,
    /// Page size (1..=1000, API default 500) and records to skip. When both
    /// are omitted the API returns the full list.
    pub limit: Option<u64>,
    pub offset: Option<u64>,
    /// Artificial Analysis intelligence index bounds.
    pub min_intelligence_index: Option<f64>,
    pub max_intelligence_index: Option<f64>,
    /// Artificial Analysis coding index bounds.
    pub min_coding_index: Option<f64>,
    pub max_coding_index: Option<f64>,
    /// Artificial Analysis agentic index bounds.
    pub min_agentic_index: Option<f64>,
    pub max_agentic_index: Option<f64>,
    /// Tool-calling success rate bounds as fractions in [0, 1].
    pub min_tool_success_rate: Option<f64>,
    pub max_tool_success_rate: Option<f64>,
}

/// `skip_serializing_if` predicate for [`ModelsQuery::zdr`].
fn is_not_true(v: &Option<bool>) -> bool {
    *v != Some(true)
}

#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<Model>,
    /// Total models matching the query, before this page's limit/offset.
    /// Absent from older responses and most test fixtures.
    #[serde(default)]
    pub total_count: Option<u64>,
    /// Pagination links; `next` is null on the last page.
    #[serde(default)]
    pub links: Option<ModelsLinks>,
}

impl ModelsResponse {
    /// One-line pagination summary for a caller paging with `limit`/`offset`:
    /// the server's `total_count` and, when there is another page, its
    /// `links.next` URL. `None` when the response carries no `total_count`.
    /// Used by the `list_models` tool header.
    pub fn pagination_note(&self) -> Option<String> {
        let total = self.total_count?;
        let mut note =
            format!("server total_count: {total} models match this query before limit/offset");
        if let Some(next) = self.links.as_ref().and_then(|l| l.next.as_deref()) {
            note.push_str(&format!("; next page: {next}"));
        }
        Some(note)
    }
}

/// The `links` block of a paginated `/models` response.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelsLinks {
    /// Relative URL of the next page (e.g. `/api/v1/models?offset=500&limit=500`).
    #[serde(default)]
    pub next: Option<String>,
}

/// A single OpenRouter model entry. Fields are optional/defaulted defensively
/// because the upstream schema evolves and varies per provider.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Model {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub context_length: Option<u64>,
    /// ISO `YYYY-MM-DD` date after which the model may be removed; omitted
    /// from the compact row when the API reports null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expiration_date: Option<String>,
    /// ISO date the training data extends to; omitted when unknown (null).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knowledge_cutoff: Option<String>,
    #[serde(default)]
    pub architecture: Option<Architecture>,
    #[serde(default)]
    pub pricing: Option<Pricing>,
    /// Reasoning capabilities: `supported_efforts`, `default_effort`,
    /// `default_enabled`, `mandatory`. Passed through untyped because the shape
    /// varies per provider and gains fields. Absent for non-reasoning models.
    /// The `reasoning_effort` tool arg points callers here, so it has to survive
    /// the round-trip through this struct.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<serde_json::Value>,
    /// Voice ids the model accepts (speech/TTS models only). Shape varies
    /// per provider, so it is passed through untyped - same rationale as
    /// `reasoning`. Live-confirmed on `GET /models?output_modalities=speech`;
    /// absent for non-speech models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supported_voices: Option<serde_json::Value>,
}

impl Model {
    /// Case-insensitive match of `needle` against the model id, name, and
    /// description. Used by the MCP `search` filter.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_cache_write_1h: Option<String>,
    /// Fractional discount applied to the above (numeric, not a price string).
    #[serde(default)]
    pub discount: Option<f64>,
    /// Tiered/time-window pricing overrides. Shape varies (conditions +
    /// alternate prices), so it is passed through untyped rather than modeled -
    /// same rationale as `Model::reasoning`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overrides: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_search_checks_id_name_and_description_case_insensitively() {
        let model = Model {
            id: "openai/gpt-audio-mini".to_string(),
            name: Some("OpenAI: GPT Audio Mini".to_string()),
            description: Some("A cost-efficient audio model.".to_string()),
            context_length: None,
            architecture: None,
            pricing: None,
            ..Default::default()
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
        // Pre-pagination responses (and mocks) carry no envelope fields.
        assert!(parsed.total_count.is_none());
        assert!(parsed.links.is_none());
        assert!(model.expiration_date.is_none());
        assert!(model.knowledge_cutoff.is_none());
    }

    /// The pagination envelope (`total_count`, `links.next` - null on the last
    /// page) and the two lifecycle dates decode; the dates are omitted again on
    /// serialize when null so compact rows stay lean.
    #[test]
    fn models_response_decodes_pagination_envelope_and_lifecycle_dates() {
        let json = r#"{
          "data": [
            {
              "id": "openai/gpt-5.6-sol",
              "expiration_date": "2027-03-01",
              "knowledge_cutoff": "2026-01-31"
            },
            { "id": "provider/undated", "expiration_date": null, "knowledge_cutoff": null }
          ],
          "total_count": 546,
          "links": { "next": null }
        }"#;

        let parsed: ModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.total_count, Some(546));
        let links = parsed.links.as_ref().unwrap();
        assert!(links.next.is_none());

        let dated = &parsed.data[0];
        assert_eq!(dated.expiration_date.as_deref(), Some("2027-03-01"));
        assert_eq!(dated.knowledge_cutoff.as_deref(), Some("2026-01-31"));
        let out = serde_json::to_value(dated).unwrap();
        assert_eq!(out["expiration_date"], "2027-03-01");
        assert_eq!(out["knowledge_cutoff"], "2026-01-31");

        let undated = serde_json::to_value(&parsed.data[1]).unwrap();
        assert!(undated.get("expiration_date").is_none());
        assert!(undated.get("knowledge_cutoff").is_none());
    }

    /// The one-line pagination note in the MCP header:
    /// nothing without `total_count`, the count alone on the last page, and
    /// the next-page link when the server says there is more.
    #[test]
    fn pagination_note_reports_total_count_and_next_link() {
        let none: ModelsResponse = serde_json::from_str(r#"{"data": []}"#).unwrap();
        assert!(none.pagination_note().is_none());

        let last: ModelsResponse =
            serde_json::from_str(r#"{"data": [], "total_count": 546, "links": {"next": null}}"#)
                .unwrap();
        assert_eq!(
            last.pagination_note().as_deref(),
            Some("server total_count: 546 models match this query before limit/offset")
        );

        let more: ModelsResponse = serde_json::from_str(
            r#"{"data": [], "total_count": 546,
                "links": {"next": "/api/v1/models?offset=20&limit=20"}}"#,
        )
        .unwrap();
        assert_eq!(
            more.pagination_note().as_deref(),
            Some(
                "server total_count: 546 models match this query before limit/offset; \
                 next page: /api/v1/models?offset=20&limit=20"
            )
        );
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

    /// The `reasoning_effort` tool arg tells callers to read
    /// `reasoning.supported_efforts` from list_models, so the block must survive
    /// deserialize -> serialize. A typed struct would drop fields OpenRouter adds.
    #[test]
    fn models_response_round_trips_the_reasoning_block() {
        let json = r#"{
          "data": [
            {
              "id": "openai/gpt-5.6-sol",
              "reasoning": {
                "mandatory": false,
                "default_enabled": true,
                "supported_efforts": ["max", "high", "low", "none"],
                "default_effort": "medium"
              }
            },
            { "id": "provider/no-reasoning" }
          ]
        }"#;

        let parsed: ModelsResponse = serde_json::from_str(json).unwrap();
        let reasoning = parsed.data[0].reasoning.as_ref().unwrap();
        assert_eq!(reasoning["default_effort"], "medium");
        assert_eq!(reasoning["supported_efforts"][3], "none");

        // Serialized back out, the block is intact and non-reasoning models do
        // not sprout a null field.
        let out = serde_json::to_value(&parsed.data[0]).unwrap();
        assert_eq!(out["reasoning"]["supported_efforts"][0], "max");
        let bare = serde_json::to_value(&parsed.data[1]).unwrap();
        assert!(bare.get("reasoning").is_none());
    }

    /// Tiered/time-window pricing (`overrides[]`) round-trips untyped, like
    /// `reasoning`, and a model with flat-only pricing doesn't sprout a null field.
    #[test]
    fn pricing_round_trips_overrides() {
        let json = r#"{
          "data": [
            {
              "id": "openai/gpt-5.6-sol",
              "pricing": {
                "prompt": "0.000001",
                "overrides": [
                  {"condition": "peak_hours", "prompt": "0.000002"}
                ]
              }
            },
            { "id": "provider/flat-pricing", "pricing": { "prompt": "0.000001" } }
          ]
        }"#;

        let parsed: ModelsResponse = serde_json::from_str(json).unwrap();
        let pricing = parsed.data[0].pricing.as_ref().unwrap();
        let overrides = pricing.overrides.as_ref().unwrap();
        assert_eq!(overrides[0]["condition"], "peak_hours");

        let out = serde_json::to_value(&parsed.data[0]).unwrap();
        assert_eq!(out["pricing"]["overrides"][0]["prompt"], "0.000002");
        let flat = serde_json::to_value(&parsed.data[1]).unwrap();
        assert!(flat["pricing"].get("overrides").is_none());
    }
}
