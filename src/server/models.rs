//! The `list_models` and `describe_model` tools, their argument structs, the
//! local search/cap/pagination presentation, and the per-model enrichment.

use rmcp::{
    ErrorData,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::openrouter::{Model, ModelsQuery, ModelsResponse};
use crate::pricing::{attach_pricing_human, humanize_pricing, models_to_json};
use crate::server::result::{internal_error_from, json_text_result};
use crate::server::schema::{
    de_bool, de_opt_bool, de_opt_f64, de_opt_uint, non_blank, scalarize_nullable,
};

use super::OpenRouterServer;

/// Default number of models `list_models` shows unless `all` is requested.
const DEFAULT_MODEL_LIMIT: usize = 20;

/// Result of applying the local `search` filter and the default result cap.
/// `models` is what the tool displays; `total` is how many matched before
/// truncation, for the "showing X of Y" footer.
struct FilteredModels {
    models: Vec<Model>,
    total: usize,
}

impl FilteredModels {
    /// How many matching models the default cap omitted (0 when `all` was set
    /// or nothing was truncated).
    fn truncated(&self) -> usize {
        self.total - self.models.len()
    }
}

/// Apply the local case-insensitive `search` filter (across id/name/description)
/// and, unless `all`, cap the result at [`DEFAULT_MODEL_LIMIT`].
fn apply_filters(mut models: Vec<Model>, search: Option<&str>, all: bool) -> FilteredModels {
    if let Some(needle) = search {
        models.retain(|m| matches_search(m, needle));
    }
    let total = models.len();
    if !all {
        models.truncate(DEFAULT_MODEL_LIMIT);
    }
    FilteredModels { models, total }
}

/// Case-insensitive match of `needle` against a model's id, name and
/// description - the tool's `search` argument.
fn matches_search(model: &Model, needle: &str) -> bool {
    let needle = needle.to_lowercase();
    let has = |s: Option<&str>| s.is_some_and(|s| s.to_lowercase().contains(&needle));
    model.id.to_lowercase().contains(&needle)
        || has(model.name.as_deref())
        || has(model.description.as_deref())
}

/// One-line pagination summary for a caller paging with `limit`/`offset`: the
/// server's `total_count` and, when there is another page, its `links.next`
/// URL. `None` when the response carries no `total_count`.
fn pagination_note(page: &ModelsResponse) -> Option<String> {
    let total = page.total_count?;
    let mut note =
        format!("server total_count: {total} models match this query before limit/offset");
    if let Some(next) = page.links.as_ref().and_then(|l| l.next.as_deref()) {
        note.push_str(&format!("; next page: {next}"));
    }
    Some(note)
}

/// Arguments for the `list_models` tool. These map to OpenRouter's server-side
/// `GET /api/v1/models` query parameters, so filtering happens at the API.
#[derive(Debug, Default, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct ListModelsArgs {
    /// Server-side free-text search by model name or slug (e.g. "claude").
    #[serde(default)]
    pub query: Option<String>,
    /// Local case-insensitive filter across id, name, and description
    /// (e.g. "openai"). Order of operations: every server-side filter (query,
    /// modalities, category, providers, price/index bounds, sort, limit/offset,
    /// ...) runs first (one API call); this local filter narrows that result
    /// next; the default 20-result cap (unless all=true) is applied last.
    #[serde(default)]
    pub search: Option<String>,
    /// Filter by output modalities. Comma-separated list of: text, image, audio
    /// (audio-output chat models, i.e. music such as google/lyria-3-*),
    /// embeddings, video, rerank, speech (text-to-speech), transcription
    /// (speech-to-text), decisions (structured decision models for
    /// make_decisions) - or "all". Defaults to text on the API when omitted
    /// (so pass "all" or a value to see others).
    #[serde(default)]
    pub output_modalities: Option<String>,
    /// Filter by input modalities. Comma-separated list of: text, image, audio, file.
    #[serde(default)]
    pub input_modalities: Option<String>,
    /// Only return models supporting these API parameters. Comma-separated,
    /// e.g. "tools", "structured_outputs", "reasoning".
    #[serde(default)]
    pub supported_parameters: Option<String>,
    /// Sort order: pricing-low-to-high, pricing-high-to-low, context-high-to-low,
    /// throughput-high-to-low, latency-low-to-high, most-popular, top-weekly, newest,
    /// intelligence-high-to-low, coding-high-to-low, agentic-high-to-low,
    /// design-arena-elo-high-to-low (benchmark sorts place unscored models last).
    /// Defaults to "top-weekly" (most used this week) when omitted.
    #[serde(default)]
    pub sort: Option<String>,
    /// Minimum context length in tokens; models with less are excluded.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub min_context: Option<u64>,
    /// Use-case category. One of: programming, roleplay, marketing, marketing/seo,
    /// technology, science, translation, legal, finance, health, trivia, academia.
    #[serde(default)]
    pub category: Option<String>,
    /// Only models hosted by these providers. Comma-separated provider names,
    /// e.g. "OpenAI,Anthropic".
    #[serde(default)]
    pub providers: Option<String>,
    /// Only models by these authors. Comma-separated author slugs, e.g.
    /// "openai,anthropic".
    #[serde(default)]
    pub model_authors: Option<String>,
    /// Architecture / model family, e.g. "GPT", "Claude", "Gemini", "Llama".
    #[serde(default)]
    pub arch: Option<String>,
    /// Minimum prompt (input) price in $ per million tokens.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub min_price: Option<f64>,
    /// Maximum prompt (input) price in $ per million tokens.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub max_price: Option<f64>,
    /// Minimum completion (output) price in $ per million tokens.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub min_output_price: Option<f64>,
    /// Maximum completion (output) price in $ per million tokens.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub max_output_price: Option<f64>,
    /// true = only models with zero-data-retention endpoints. false/omitted =
    /// no ZDR filter (the API has no "exclude ZDR" mode).
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub zdr: Option<bool>,
    /// Data region of the model's endpoints: "eu" or "us".
    #[serde(default)]
    pub region: Option<String>,
    /// true = only distillable models, false = exclude them, omitted = no filter.
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub distillable: Option<bool>,
    /// Minimum model age in days since it was added.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub min_age_days: Option<u64>,
    /// Maximum model age in days since it was added (e.g. 30 = released this month).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub max_age_days: Option<u64>,
    /// Server-side page size, 1..=1000 (API default 500; with limit and offset
    /// both omitted the API returns the full list). The header line reports the
    /// server's total_count so you can page with offset. The local 20-row cap
    /// still applies unless all=true.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub limit: Option<u64>,
    /// Server-side records to skip (pair with limit to page).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub offset: Option<u64>,
    /// Minimum Artificial Analysis intelligence index.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub min_intelligence_index: Option<f64>,
    /// Maximum Artificial Analysis intelligence index.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub max_intelligence_index: Option<f64>,
    /// Minimum Artificial Analysis coding index.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub min_coding_index: Option<f64>,
    /// Maximum Artificial Analysis coding index.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub max_coding_index: Option<f64>,
    /// Minimum Artificial Analysis agentic index.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub min_agentic_index: Option<f64>,
    /// Maximum Artificial Analysis agentic index.
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub max_agentic_index: Option<f64>,
    /// Minimum tool-calling success rate as a fraction in [0, 1] (0.9 = 90% of
    /// requests finish with a tool_calls finish reason).
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub min_tool_success_rate: Option<f64>,
    /// Maximum tool-calling success rate as a fraction in [0, 1].
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub max_tool_success_rate: Option<f64>,
    /// Return all matching models. By default only the first 20 are returned to
    /// keep the result compact; set true to get the complete list.
    #[serde(default, deserialize_with = "de_bool")]
    pub all: bool,
}

impl ListModelsArgs {
    /// The wire query for these args. Field names match one-to-one; the only
    /// normalization is the `top-weekly` default sort.
    fn into_query(self) -> ModelsQuery {
        ModelsQuery {
            q: self.query,
            output_modalities: self.output_modalities,
            input_modalities: self.input_modalities,
            supported_parameters: self.supported_parameters,
            // Default to most-used-this-week when the caller doesn't specify a sort.
            sort: Some(self.sort.unwrap_or_else(|| "top-weekly".to_string())),
            context: self.min_context,
            category: self.category,
            providers: self.providers,
            model_authors: self.model_authors,
            arch: self.arch,
            min_price: self.min_price,
            max_price: self.max_price,
            min_output_price: self.min_output_price,
            max_output_price: self.max_output_price,
            zdr: self.zdr,
            region: self.region,
            distillable: self.distillable,
            min_age_days: self.min_age_days,
            max_age_days: self.max_age_days,
            limit: self.limit,
            offset: self.offset,
            min_intelligence_index: self.min_intelligence_index,
            max_intelligence_index: self.max_intelligence_index,
            min_coding_index: self.min_coding_index,
            max_coding_index: self.max_coding_index,
            min_agentic_index: self.min_agentic_index,
            max_agentic_index: self.max_agentic_index,
            min_tool_success_rate: self.min_tool_success_rate,
            max_tool_success_rate: self.max_tool_success_rate,
        }
    }
}

/// Arguments for the `describe_model` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct DescribeModelArgs {
    /// Exact model id ("author/slug"), e.g. "anthropic/claude-opus-4.7". Use
    /// list_models to discover ids.
    pub model: String,
}

#[tool_router(router = models_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "List available OpenRouter models with their capabilities \
        (input/output modalities, context length, reasoning efforts, TTS voices, \
        knowledge_cutoff, expiration_date) and pricing. Filtering and sorting happen \
        server-side: search by name (query), filter by output/input modalities, supported \
        parameters, use-case category, hosting providers, model_authors, arch (model \
        family), prompt/completion price bounds ($/M tokens), zdr (zero data retention), \
        region (eu|us), distillable, model age in days, and Artificial Analysis \
        intelligence/coding/agentic index or tool_success_rate bounds; sort by \
        newest/most-popular/pricing/context or by the intelligence/coding/agentic/\
        design-arena benchmarks; set a minimum context length; page with limit/offset \
        (the header line reports the server's total_count). Output modalities include \
        text, image, audio (audio-output chat models - music such as google/lyria-3-*), \
        embeddings, video, rerank, speech (text-to-speech), transcription (speech-to-text), \
        decisions (structured decision models such as typesafe/jev-1.13, for make_decisions); \
        the default is text only, so pass output_modalities=\"all\" or a specific value to \
        see the rest. Returns the first 20 models by default; set all=true for the \
        complete list.",
        annotations(
            title = "List OpenRouter Models",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn list_models(
        &self,
        Parameters(mut args): Parameters<ListModelsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        // Local post-processing knobs; everything else is the wire query. A
        // blank search means no filter (the repo-wide rule).
        let search = non_blank(args.search.take());
        let all = args.all;
        let query = args.into_query();

        let page = self
            .client
            .list_models_page(&query)
            .await
            .map_err(|e| internal_error_from(&e))?;
        let pagination = pagination_note(&page);

        let filtered = apply_filters(page.data, search.as_deref(), all);

        // Attach human-readable pricing_human to each model.
        let models = models_to_json(&filtered.models)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        let mut json = serde_json::to_string_pretty(&models)
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        if filtered.truncated() > 0 {
            json = format!(
                "// showing {} of {} models; set \"all\": true to get the rest\n{}",
                filtered.models.len(),
                filtered.total,
                json
            );
        }
        // The server's own count (before limit/offset) and next-page link are
        // what a paging caller needs; the local footer above only knows about
        // this page.
        if let Some(note) = pagination {
            json = format!("// {note}\n{json}");
        }

        Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
    }

    #[tool(
        description = "Get the full detail for a single OpenRouter model by its exact id \
        (author/slug, e.g. \"anthropic/claude-opus-4.7\" - discover ids with list_models). \
        Returns everything OpenRouter reports for that model as JSON: the model object \
        (description, architecture/modalities, tokenizer, context_length, knowledge_cutoff, \
        benchmarks) plus the per-provider endpoints with their pricing, uptime, status, \
        quantization, max tokens, and supported parameters - richer and more current than the \
        list_models entry (which is a compact subset). For video models, also merges the real \
        pricing under a \"video\" key (pricing_skus, supported resolutions/durations/sizes from \
        /videos/models), since the token-based pricing is 0 and misleading for video. For image \
        models, also merges per-endpoint image capabilities under an \"image\" key \
        (supported_parameters, allowed_passthrough_parameters, pricing, supports_streaming from \
        /images/models/{author}/{slug}/endpoints) - best-effort, omitted if that model has no \
        image endpoint. For audio-output models whose token pricing is 0 (music, e.g. \
        google/lyria-3-*), adds an \"audio_pricing_note\": they bill a flat fee per track \
        that appears only in the description. Fails if the id is unknown.",
        annotations(
            title = "Describe OpenRouter Model",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn describe_model(
        &self,
        Parameters(args): Parameters<DescribeModelArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let model = args.model.trim();
        if model.is_empty() {
            return Err(ErrorData::invalid_params(
                "model is required (an exact id, e.g. \"anthropic/claude-opus-4.7\")".to_string(),
                None,
            ));
        }

        let detail = enriched_model_detail(&self.client, model)
            .await
            .map_err(|e| internal_error_from(&e))?;
        json_text_result(&detail)
    }
}

/// Attached to an audio-output model whose token pricing is all zeros.
const AUDIO_PRICING_NOTE: &str = "token pricing is 0 for this audio-output model: it bills a \
    flat fee per generated track, stated only in its description (e.g. \"$0.04 per clip\"); \
    the actual charge is reported after the call as usage.cost (generate_music returns it \
    as cost_usd)";

/// Whether the model record declares `modality` among its output modalities.
fn outputs(detail: &Value, modality: &str) -> bool {
    detail["architecture"]["output_modalities"]
        .as_array()
        .is_some_and(|m| m.iter().any(|v| v == modality))
}

/// Fetch model detail and preserve optional modality enrichment failures inline.
async fn enriched_model_detail(
    client: &crate::openrouter::OpenRouterClient,
    model: &str,
) -> anyhow::Result<Value> {
    let mut detail = client.describe_model(model).await?;

    // Video models price via a separate SKU endpoint; the token-based
    // pricing on the main record is 0 and misleading. Merge the real
    // pricing_skus + supported resolutions/durations/sizes under "video".
    if outputs(&detail, "video") {
        match client.video_model_detail(model).await {
            Ok(Some(mut video)) => {
                // pricing_skus is the video model's real pricing object.
                if let Some(human) = video.get("pricing_skus").and_then(humanize_pricing) {
                    video["pricing_skus_human"] = human;
                }
                detail["video"] = video;
            }
            Ok(None) => {}
            // Surface the failure instead of silently returning the
            // misleading 0 token pricing as if it were complete.
            Err(e) => {
                detail["video_pricing_error"] =
                    Value::String(format!("could not fetch /videos/models: {e:#}"));
            }
        }
    }

    // Image models: merge the per-endpoint detail (definitive
    // supported_parameters, allowed_passthrough_parameters, pricing,
    // supports_streaming) from the dedicated image-models endpoint, the
    // same best-effort way the video block above is merged.
    if outputs(&detail, "image") {
        match client.image_model_detail(model).await {
            Ok(Some(mut image)) => {
                // Same human rendering the video block gets: image pricing
                // lines carry numeric cost_usd, unreadable at 4e-05.
                if let Some(endpoints) = image.get_mut("endpoints").and_then(Value::as_array_mut) {
                    for ep in endpoints {
                        crate::pricing::attach_image_pricing_human(ep);
                    }
                }
                detail["image"] = image;
            }
            Ok(None) => {}
            Err(e) => {
                detail["image_pricing_error"] =
                    Value::String(format!("could not fetch /images/models: {e:#}"));
            }
        }
    }

    // Audio-output chat models (music: google/lyria-3-*) have no dedicated
    // pricing endpoint, and their token pricing is 0 while the real per-track
    // price appears only in the description. Say so, so the zeros are not
    // read as "free" (verified live 2026-09-13: a clip billed usage.cost 0.04).
    if outputs(&detail, "audio") && crate::pricing::is_zero_priced(&detail["pricing"]) {
        detail["audio_pricing_note"] = Value::String(AUDIO_PRICING_NOTE.to_string());
    }

    // Normalize every pricing block to human "$X/M tokens" form alongside
    // the raw decimals: the top-level record and each per-provider endpoint.
    attach_pricing_human(&mut detail);
    if let Some(endpoints) = detail.get_mut("endpoints").and_then(Value::as_array_mut) {
        for ep in endpoints {
            attach_pricing_human(ep);
        }
    }
    Ok(detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use rmcp::handler::server::wrapper::Parameters;
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Build `n` placeholder models with ids `model-0`, `model-1`, ... so the
    /// local filter and cap can be exercised without hitting the network.
    fn models(n: usize) -> Vec<Model> {
        (0..n)
            .map(|i| Model {
                id: format!("model-{i}"),
                ..Default::default()
            })
            .collect()
    }

    /// The cap applies only without `all`, and `total` always reports the
    /// pre-truncation match count.
    #[test]
    fn apply_filters_caps_at_default_limit_unless_all() {
        let filtered = apply_filters(models(25), None, false);
        assert_eq!(filtered.models.len(), DEFAULT_MODEL_LIMIT);
        assert_eq!(filtered.total, 25);
        assert_eq!(filtered.truncated(), 5);

        let filtered = apply_filters(models(25), None, true);
        assert_eq!(filtered.models.len(), 25);
        assert_eq!(filtered.truncated(), 0);

        let filtered = apply_filters(models(3), None, false);
        assert_eq!(filtered.models.len(), 3);
        assert_eq!(filtered.truncated(), 0);
    }

    /// Search narrows first, then the cap trims; `total` counts the matches.
    #[test]
    fn apply_filters_search_runs_before_truncation() {
        // "model-2" matches model-2 and model-20..29 = 11 of 30.
        let filtered = apply_filters(models(30), Some("model-2"), false);
        assert_eq!(filtered.total, 11);
        assert_eq!(filtered.models.len(), 11);
        assert!(filtered.models.iter().all(|m| m.id.contains("model-2")));

        // "MODEL-" matches all 25 (case-insensitive); the cap trims to 20.
        let filtered = apply_filters(models(25), Some("MODEL-"), false);
        assert_eq!(filtered.total, 25);
        assert_eq!(filtered.models.len(), DEFAULT_MODEL_LIMIT);
    }

    #[test]
    fn matches_search_checks_id_name_and_description_case_insensitively() {
        let model = Model {
            id: "openai/gpt-audio-mini".to_string(),
            name: Some("OpenAI: GPT Audio Mini".to_string()),
            description: Some("A cost-efficient audio model.".to_string()),
            ..Default::default()
        };
        assert!(matches_search(&model, "OPENAI"));
        assert!(matches_search(&model, "audio mini"));
        assert!(matches_search(&model, "cost-efficient"));
        assert!(!matches_search(&model, "anthropic"));
    }

    /// The one-line pagination note in the MCP header: nothing without
    /// `total_count`, the count alone on the last page, and the next-page link
    /// when the server says there is more.
    #[test]
    fn pagination_note_reports_total_count_and_next_link() {
        let none: ModelsResponse = serde_json::from_str(r#"{"data": []}"#).unwrap();
        assert!(pagination_note(&none).is_none());

        let last: ModelsResponse =
            serde_json::from_str(r#"{"data": [], "total_count": 546, "links": {"next": null}}"#)
                .unwrap();
        assert_eq!(
            pagination_note(&last).as_deref(),
            Some("server total_count: 546 models match this query before limit/offset")
        );

        let more: ModelsResponse = serde_json::from_str(
            r#"{"data": [], "total_count": 546,
                "links": {"next": "/api/v1/models?offset=20&limit=20"}}"#,
        )
        .unwrap();
        assert_eq!(
            pagination_note(&more).as_deref(),
            Some(
                "server total_count: 546 models match this query before limit/offset; \
                 next page: /api/v1/models?offset=20&limit=20"
            )
        );
    }

    #[tokio::test]
    async fn list_models_tool_returns_model_json() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "openai/gpt", "name": "GPT"}]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .list_models(Parameters(ListModelsArgs::default()))
            .await
            .unwrap();

        // The tool returns the model list as pretty JSON text content.
        let body = serde_json::to_string(&result).unwrap();
        assert!(body.contains("openai/gpt"));
    }

    /// `search` follows the repo-wide blank-means-absent rule: a padded needle
    /// is trimmed before matching and a blank one applies no filter at all.
    #[tokio::test]
    async fn list_models_search_trims_and_treats_blank_as_no_filter() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "openai/gpt"}, {"id": "anthropic/claude"}]
            })))
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let ids = |result: &rmcp::model::CallToolResult| -> Vec<String> {
            tool_result_json(result)
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m["id"].as_str().unwrap().to_string())
                .collect()
        };

        let padded = server
            .list_models(Parameters(ListModelsArgs {
                search: Some("  claude ".into()),
                ..Default::default()
            }))
            .await
            .unwrap();
        assert_eq!(ids(&padded), vec!["anthropic/claude"]);

        let blank = server
            .list_models(Parameters(ListModelsArgs {
                search: Some("   ".into()),
                ..Default::default()
            }))
            .await
            .unwrap();
        assert_eq!(ids(&blank), vec!["openai/gpt", "anthropic/claude"]);
    }

    /// `Model` must not silently drop `supported_voices` - it is what
    /// generate_audio points callers at, and OpenRouter's speech models carry
    /// it on `GET /models?output_modalities=speech` (live-confirmed).
    #[tokio::test]
    async fn list_models_tool_carries_supported_voices_through() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{
                    "id": "hexgrad/kokoro-82m",
                    "name": "Kokoro 82M",
                    "supported_voices": ["af_heart", "af_bella"]
                }]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .list_models(Parameters(ListModelsArgs::default()))
            .await
            .unwrap();

        let v = tool_result_json(&result);
        assert_eq!(v[0]["supported_voices"][0], "af_heart");
        assert_eq!(v[0]["supported_voices"][1], "af_bella");
    }

    /// Every new tool arg maps to the query param of the same name, with the
    /// tolerant scalar deserializers accepting stringified numbers/booleans
    /// (the failure mode the rest of the schema helpers exist for). The mock
    /// only matches when all params are present.
    #[tokio::test]
    async fn list_models_tool_forwards_every_filter_by_its_documented_name() {
        let mock = MockServer::start().await;
        let expected: &[(&str, &str)] = &[
            ("category", "programming"),
            ("providers", "OpenAI,Anthropic"),
            ("model_authors", "openai,anthropic"),
            ("arch", "Claude"),
            ("min_price", "0.5"),
            ("max_price", "2.5"),
            ("min_output_price", "1.5"),
            ("max_output_price", "10.5"),
            ("zdr", "true"),
            ("region", "eu"),
            ("distillable", "false"),
            ("min_age_days", "7"),
            ("max_age_days", "365"),
            ("limit", "50"),
            ("offset", "100"),
            ("min_intelligence_index", "40.5"),
            ("max_intelligence_index", "70.5"),
            ("min_coding_index", "30.5"),
            ("max_coding_index", "60.5"),
            ("min_agentic_index", "20.5"),
            ("max_agentic_index", "50.5"),
            ("min_tool_success_rate", "0.9"),
            ("max_tool_success_rate", "0.99"),
            ("sort", "intelligence-high-to-low"),
        ];
        let mut m = Mock::given(method("GET")).and(path("/models"));
        for (name, value) in expected {
            m = m.and(wiremock::matchers::query_param(*name, *value));
        }
        m.respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [] })))
            .expect(1)
            .mount(&mock)
            .await;

        // Mixed typed and stringified scalars, as real clients send them.
        let args: ListModelsArgs = serde_json::from_value(json!({
            "category": "programming",
            "providers": "OpenAI,Anthropic",
            "model_authors": "openai,anthropic",
            "arch": "Claude",
            "min_price": 0.5,
            "max_price": "2.5",
            "min_output_price": 1.5,
            "max_output_price": "10.5",
            "zdr": "true",
            "region": "eu",
            "distillable": false,
            "min_age_days": "7",
            "max_age_days": 365,
            "limit": 50,
            "offset": "100",
            "min_intelligence_index": 40.5,
            "max_intelligence_index": "70.5",
            "min_coding_index": 30.5,
            "max_coding_index": 60.5,
            "min_agentic_index": "20.5",
            "max_agentic_index": 50.5,
            "min_tool_success_rate": 0.9,
            "max_tool_success_rate": "0.99",
            "sort": "intelligence-high-to-low"
        }))
        .unwrap();
        let server = server_for(mock.uri());
        server.list_models(Parameters(args)).await.unwrap();
    }

    /// The server's `total_count` (matches before this page's limit/offset)
    /// is returned in the header line when present, so a caller paging with
    /// limit/offset knows how far to go. Absent upstream -> no header, and
    /// the body stays plain JSON.
    #[tokio::test]
    async fn list_models_tool_reports_server_total_count_when_present() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"id": "openai/gpt"}],
                "total_count": 546,
                "links": {"next": "/api/v1/models?offset=1&limit=1"}
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .list_models(Parameters(ListModelsArgs {
                limit: Some(1),
                ..Default::default()
            }))
            .await
            .unwrap();
        let v = serde_json::to_value(&result).unwrap();
        let text = v["content"][0]["text"].as_str().unwrap();
        let header = text.lines().next().unwrap();
        assert!(header.starts_with("// "), "got: {header}");
        assert!(header.contains("total_count: 546"), "got: {header}");
        assert!(
            header.contains("next page: /api/v1/models?offset=1&limit=1"),
            "got: {header}"
        );
        assert!(text.contains("openai/gpt"));
    }

    /// The compact row (one `list_models` entry) surfaces the capability
    /// fields callers are pointed at - `reasoning.supported_efforts` /
    /// `mandatory`, `supported_voices`, `expiration_date`, `knowledge_cutoff` -
    /// and only when non-null, so a plain text model's row stays lean.
    #[tokio::test]
    async fn list_models_compact_row_surfaces_capabilities_only_when_present() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {
                        "id": "openai/gpt-5.6-sol",
                        "reasoning": {
                            "mandatory": true,
                            "supported_efforts": ["high", "medium", "low"],
                            "default_effort": "medium"
                        },
                        "expiration_date": "2027-03-01",
                        "knowledge_cutoff": "2026-01-31",
                        "supported_voices": null
                    },
                    {
                        "id": "hexgrad/kokoro-82m",
                        "supported_voices": ["af_heart"],
                        "expiration_date": null,
                        "knowledge_cutoff": null
                    },
                    { "id": "provider/plain" }
                ]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .list_models(Parameters(ListModelsArgs::default()))
            .await
            .unwrap();
        let rows = tool_result_json(&result);

        let reasoning = &rows[0];
        assert_eq!(reasoning["reasoning"]["supported_efforts"][0], "high");
        assert_eq!(reasoning["reasoning"]["mandatory"], true);
        assert_eq!(reasoning["expiration_date"], "2027-03-01");
        assert_eq!(reasoning["knowledge_cutoff"], "2026-01-31");
        assert!(reasoning.get("supported_voices").is_none());

        let speech = &rows[1];
        assert_eq!(speech["supported_voices"][0], "af_heart");
        assert!(speech.get("expiration_date").is_none());
        assert!(speech.get("knowledge_cutoff").is_none());

        let plain = rows[2].as_object().unwrap();
        for key in [
            "reasoning",
            "supported_voices",
            "expiration_date",
            "knowledge_cutoff",
        ] {
            assert!(!plain.contains_key(key), "plain row leaked {key}");
        }
    }

    #[tokio::test]
    async fn describe_model_tool_returns_full_detail_json() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/anthropic/claude-opus-4.7/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "anthropic/claude-opus-4.7",
                    "endpoints": [{"provider_name": "Anthropic", "context_length": 1000000}]
                }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "anthropic/claude-opus-4.7".to_string(),
            }))
            .await
            .unwrap();

        let body = serde_json::to_string(&result).unwrap();
        assert!(body.contains("anthropic/claude-opus-4.7"));
        assert!(body.contains("Anthropic"));
    }

    #[tokio::test]
    async fn describe_model_tool_merges_video_pricing() {
        let mock = MockServer::start().await;
        // Main detail: a video-output model with the misleading 0 token pricing.
        Mock::given(method("GET"))
            .and(path("/models/bytedance/seedance-2.0/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "bytedance/seedance-2.0",
                    "architecture": {"output_modalities": ["video"]},
                    "endpoints": [{"provider_name": "Seed", "pricing": {"prompt": "0", "completion": "0"}}]
                }
            })))
            .mount(&mock)
            .await;
        // The real pricing lives in /videos/models under pricing_skus.
        Mock::given(method("GET"))
            .and(path("/videos/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [
                    {"id": "other/model", "pricing_skus": {"generate": "0.50"}},
                    {"id": "bytedance/seedance-2.0", "pricing_skus": {"video_tokens": "0.000007"}}
                ]
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "bytedance/seedance-2.0".to_string(),
            }))
            .await
            .unwrap();

        let body = serde_json::to_string(&result).unwrap();
        // The merged "video" block carries the matching entry's real SKU pricing.
        assert!(body.contains("video_tokens"));
        assert!(body.contains("0.000007"));
        // ...and not some other model's SKU.
        assert!(!body.contains("\"generate\""));
    }

    #[tokio::test]
    async fn describe_model_tool_merges_image_endpoint_detail() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/openai/gpt-image-2/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "openai/gpt-image-2",
                    "architecture": {"output_modalities": ["image"]},
                    "endpoints": [{"provider_name": "OpenAI"}]
                }
            })))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/images/models/openai/gpt-image-2/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "openai/gpt-image-2",
                    "endpoints": [{
                        "supported_parameters": ["quality", "output_format"],
                        "pricing": [{"billable": "output_image", "unit": "token", "cost_usd": 0.00003}]
                    }]
                }
            })))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "openai/gpt-image-2".to_string(),
            }))
            .await
            .unwrap();

        let v = tool_result_json(&result);
        assert_eq!(
            v["image"]["endpoints"][0]["supported_parameters"][0],
            "quality"
        );
        // Endpoint pricing uses the explicit unit even when the billable name
        // contains "image". Preserve the original unit alongside the rendering.
        assert_eq!(v["image"]["endpoints"][0]["pricing"][0]["unit"], "token");
        assert_eq!(
            v["image"]["endpoints"][0]["pricing_human"][0],
            "output_image: $30/M tokens"
        );
    }

    /// A model with no image endpoint (404) is silently omitted, not a failure.
    #[tokio::test]
    async fn describe_model_tool_omits_image_block_on_404() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/some/image-model/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "some/image-model",
                    "architecture": {"output_modalities": ["image"]}
                }
            })))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/images/models/some/image-model/endpoints"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "some/image-model".to_string(),
            }))
            .await
            .unwrap();

        let body = serde_json::to_string(&result).unwrap();
        assert!(!body.contains("\"image\":"));
        assert!(!body.contains("image_pricing_error"));
    }

    /// A non-404 failure is surfaced inline rather than failing the whole tool.
    #[tokio::test]
    async fn describe_model_tool_surfaces_image_fetch_error_without_failing() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models/some/image-model/endpoints"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {
                    "id": "some/image-model",
                    "architecture": {"output_modalities": ["image"]}
                }
            })))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/images/models/some/image-model/endpoints"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let result = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "some/image-model".to_string(),
            }))
            .await
            .unwrap();

        let body = serde_json::to_string(&result).unwrap();
        assert!(body.contains("image_pricing_error"));
    }

    /// Lyria-style records price at 0 tokens while billing per track; the note
    /// flags exactly those. A speech-style audio model with real per-token
    /// audio prices gets no note.
    #[tokio::test]
    async fn describe_model_tool_flags_zero_priced_audio_output_models() {
        for (id, pricing, expect_note) in [
            (
                "google/lyria-3-clip-preview",
                json!({"prompt": "0", "completion": "0"}),
                true,
            ),
            (
                "openai/gpt-audio",
                json!({"prompt": "0.0000025", "completion": "0.00001", "audio_output": "0.000064"}),
                false,
            ),
        ] {
            let mock = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path(format!("/models/{id}/endpoints")))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "data": {
                        "id": id,
                        "architecture": {"output_modalities": ["text", "audio"]},
                        "pricing": pricing,
                        "endpoints": [{"provider_name": "P", "pricing": pricing}]
                    }
                })))
                .mount(&mock)
                .await;
            let result = server_for(mock.uri())
                .describe_model(Parameters(DescribeModelArgs {
                    model: id.to_string(),
                }))
                .await
                .unwrap();
            let detail = tool_result_json(&result);
            let note = detail.get("audio_pricing_note").and_then(Value::as_str);
            assert_eq!(note.is_some(), expect_note, "{id}: {detail}");
            if let Some(note) = note {
                assert!(note.contains("per generated track"), "{note}");
                assert!(note.contains("cost_usd"), "{note}");
            }
            // No stray video/image enrichment was attempted for an audio model.
            assert!(detail.get("video").is_none());
            assert!(detail.get("image").is_none());
        }
    }

    #[tokio::test]
    async fn describe_model_tool_requires_model_id() {
        let server = server_for("http://127.0.0.1:9".to_string());
        let err = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "   ".to_string(),
            }))
            .await
            .unwrap_err();
        assert!(err.message.contains("model is required"));
    }

    #[tokio::test]
    async fn list_models_tool_surfaces_upstream_errors() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/models"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let err = server
            .list_models(Parameters(ListModelsArgs::default()))
            .await
            .unwrap_err();
        assert!(err.message.contains("500"), "got: {}", err.message);
    }
    /// A body that is not JSON fails with its cause (the parser's message),
    /// not only the outer "failed to decode" line.
    #[tokio::test]
    async fn model_tools_report_the_cause_of_a_decode_failure() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let listed = server
            .list_models(Parameters(ListModelsArgs::default()))
            .await
            .unwrap_err();
        let described = server
            .describe_model(Parameters(DescribeModelArgs {
                model: "test/model".to_string(),
            }))
            .await
            .unwrap_err();
        for err in [listed, described] {
            assert!(err.message.contains("expected"), "got: {}", err.message);
        }
    }

    #[tokio::test]
    async fn describe_model_preserves_both_modalities_and_video_errors() {
        for video_status in [200, 500] {
            let mock = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/models/test/multimodal/endpoints"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "data": {
                        "id": "test/multimodal",
                        "architecture": {"output_modalities": ["video", "image"]},
                        "pricing": {"prompt": "0.000001"},
                        "custom_field": "preserved"
                    }
                })))
                .expect(1)
                .mount(&mock)
                .await;
            Mock::given(method("GET"))
                .and(path("/videos/models"))
                .respond_with(ResponseTemplate::new(video_status).set_body_json(json!({
                    "data": [{"id": "test/multimodal", "pricing_skus": {"video_tokens": "0.000007"}}]
                }))).expect(1).mount(&mock).await;
            Mock::given(method("GET"))
                .and(path("/images/models/test/multimodal/endpoints"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "data": {"endpoints": [{"pricing": [{"billable": "output_image", "unit": "token", "cost_usd": 0.00003}]}]}
                }))).expect(1).mount(&mock).await;
            let result = server_for(mock.uri())
                .describe_model(Parameters(DescribeModelArgs {
                    model: "test/multimodal".into(),
                }))
                .await
                .unwrap();
            let detail = tool_result_json(&result);
            assert_eq!(detail["custom_field"], "preserved");
            assert_eq!(detail["pricing"]["prompt"], "0.000001");
            assert!(detail["pricing_human"].is_object());
            assert_eq!(
                detail["image"]["endpoints"][0]["pricing"][0]["cost_usd"],
                0.00003
            );
            assert_eq!(
                detail["image"]["endpoints"][0]["pricing_human"][0],
                "output_image: $30/M tokens"
            );
            if video_status == 200 {
                assert_eq!(detail["video"]["pricing_skus"]["video_tokens"], "0.000007");
                assert!(detail.get("video_pricing_error").is_none());
            } else {
                assert!(
                    detail["video_pricing_error"]
                        .as_str()
                        .unwrap()
                        .contains("/videos/models")
                );
                assert!(detail.get("video").is_none());
            }
        }
    }
}
