//! The retrieval and accounting tools: `embed_text` (`POST /embeddings`),
//! `rerank_documents` (`POST /rerank`) and `get_generation`
//! (`GET /generation?id=`), with their argument structs.

use rmcp::{
    ErrorData, handler::server::wrapper::Parameters, model::CallToolResult, tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::embed_gen;
use crate::openrouter::{EmbeddingsBody, EmbeddingsInput, HttpFailure, RerankBody};
use crate::server::provider::ProviderRoutingArgs;
use crate::server::result::json_text_result;
use crate::server::schema::{de_lenient, de_opt_uint, scalarize_nullable};

use super::OpenRouterServer;

/// Arguments for the `embed_text` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct EmbedTextArgs {
    /// Embeddings model id, e.g. "openai/text-embedding-3-small". Discover
    /// them with list_models using output_modalities="embeddings".
    pub model: String,
    /// The texts to embed: one or more strings, none blank. One vector comes
    /// back per text, in the same order.
    #[schemars(length(min = 1))]
    pub input: Vec<String>,
    /// Output vector size, for models that support truncation (e.g. 256 for
    /// openai/text-embedding-3-*). Omit for the model's native size.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub dimensions: Option<u32>,
    /// Provider-specific input hint such as "query" or "document" (Cohere,
    /// Voyage). Omit when the model does not distinguish.
    #[serde(default)]
    pub input_type: Option<String>,
    /// Provider block for this request: routing only, as {"order": [slugs],
    /// "only": [slugs], "ignore": [slugs], "allow_fallbacks": bool,
    /// "require_parameters": bool, "zdr": bool, "sort":
    /// "price"|"throughput"|"latency"|"exacto", "sort_partition": "model"|"none"}.
    /// This endpoint has no per-provider `options` passthrough (describe_model's
    /// allowed_passthrough_parameters do not apply here).
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ProviderRoutingArgs,
}

/// Arguments for the `rerank_documents` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct RerankDocumentsArgs {
    /// Rerank model id, e.g. "cohere/rerank-v3.5". Discover them with
    /// list_models using output_modalities="rerank".
    pub model: String,
    /// The query to rank the documents against.
    pub query: String,
    /// The candidate documents as plain text: one or more strings, none
    /// blank. Result `index` values point into this list.
    #[schemars(length(min = 1))]
    pub documents: Vec<String>,
    /// Return only the best N documents. Omit for every document, ranked.
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub top_n: Option<u32>,
    /// Provider block for this request: routing only, as {"order": [slugs],
    /// "only": [slugs], "ignore": [slugs], "allow_fallbacks": bool,
    /// "require_parameters": bool, "zdr": bool, "sort":
    /// "price"|"throughput"|"latency"|"exacto", "sort_partition": "model"|"none"}.
    /// This endpoint has no per-provider `options` passthrough (describe_model's
    /// allowed_passthrough_parameters do not apply here).
    #[serde(default, deserialize_with = "de_lenient")]
    pub provider: ProviderRoutingArgs,
}

/// Arguments for the `get_generation` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct GetGenerationArgs {
    /// The generation id returned by another tool's result (the
    /// `generation_id` field, from OpenRouter's X-Generation-Id header).
    pub generation_id: String,
}

#[tool_router(router = embeddings_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Embed one or more texts with an OpenRouter embeddings model (e.g. \
        openai/text-embedding-3-small, google/gemini-embedding-001, or a Cohere/Voyage/Qwen \
        model) and return the float vectors. This is a synchronous call. `input` is a list of \
        strings (at least one, none blank); one vector comes back per text, in input order. \
        `dimensions` truncates the vectors on models that support it; `input_type` (\"query\" / \
        \"document\") is a hint some providers use. `provider` carries routing only (order, \
        only, ignore, allow_fallbacks, require_parameters, zdr, sort) - this endpoint has no \
        passthrough options. Returns JSON: model, dimensions (vector length), count, \
        embeddings (array of number arrays), usage {prompt_tokens, total_tokens, cost in USD}, \
        and generation_id - pass that to get_generation for the full cost record. Discover \
        embeddings models with list_models using output_modalities=\"embeddings\" - they are \
        not in the default model list.",
        annotations(
            title = "Embed Text",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    pub(crate) async fn embed_text(
        &self,
        Parameters(args): Parameters<EmbedTextArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let _work = self.admit_work()?;
        let model = args.model.clone();
        embed_gen::check_texts(&args.input, "input")
            .map_err(|e| ErrorData::invalid_params(format!("{e:#}"), None))?;
        let body = EmbeddingsBody {
            model: args.model,
            input: EmbeddingsInput::from_texts(args.input),
            dimensions: args.dimensions,
            input_type: args.input_type.filter(|s| !s.trim().is_empty()),
            provider: args.provider.into_routing()?,
        };

        match embed_gen::embed(&self.client, &body).await {
            Ok(result) => {
                self.stats.record_text(&model, result.cost).await;
                json_text_result(&result.to_json())
            }
            Err(e) => {
                self.stats.record_text_failure(&model, &e).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }

    #[tool(
        description = "Rerank candidate documents against a query with an OpenRouter rerank \
        model (e.g. cohere/rerank-v3.5) and return them best-first with a relevance score. \
        This is a synchronous call. `documents` is a list of plain-text strings (at least one, \
        none blank); `top_n` keeps only the best N. `provider` carries routing only (order, \
        only, ignore, allow_fallbacks, require_parameters, zdr, sort) - this endpoint has no \
        passthrough options. Returns JSON: model, results (in the provider's ranked order, \
        each {index into your documents, relevance_score, text}), usage {cost in USD, \
        search_units, total_tokens}, and generation_id - pass that to get_generation for the \
        full cost record. Discover rerank models with list_models using \
        output_modalities=\"rerank\" - they are not in the default model list.",
        annotations(
            title = "Rerank Documents",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    pub(crate) async fn rerank_documents(
        &self,
        Parameters(args): Parameters<RerankDocumentsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let _work = self.admit_work()?;
        let model = args.model.clone();
        let body = RerankBody {
            model: args.model,
            query: args.query,
            documents: args.documents,
            top_n: args.top_n,
            provider: args.provider.into_routing()?,
        };
        embed_gen::validate_rerank(&body)
            .map_err(|e| ErrorData::invalid_params(format!("{e:#}"), None))?;

        match embed_gen::rerank(&self.client, &body).await {
            Ok(result) => {
                self.stats.record_text(&model, result.cost).await;
                json_text_result(&result.to_json())
            }
            Err(e) => {
                self.stats.record_text_failure(&model, &e).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }

    #[tool(
        description = "Look up the stored record of one request by its generation id \
        (GET /api/v1/generation?id=). The id (\"gen-...\", from OpenRouter's X-Generation-Id \
        header) is in the result of chat_completion, describe_image, generate_music, embed_text, \
        rerank_documents and make_decisions, and in the sidecar manifest generate_image, generate_video and \
        generate_audio write; this returns what OpenRouter recorded for it as \
        JSON, verbatim: total_cost (USD actually charged), provider_name, the model, native \
        token counts (native_tokens_prompt / native_tokens_completion / reasoning / cached), \
        latency and generation_time in milliseconds, api_type, finish reason, and more. Use \
        it to close the cost loop when a tool reported usage.cost as null. The record can lag \
        the request by a few seconds: an unknown or not-yet-indexed id is a 404, reported as \
        an invalid-params error - retry shortly. This is a lookup, not a generation: it is \
        counted as a request in get_usage_stats but adds no cost there.",
        annotations(
            title = "Get Generation Record",
            read_only_hint = true,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    pub(crate) async fn get_generation(
        &self,
        Parameters(args): Parameters<GetGenerationArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let id = args.generation_id.trim();
        if id.is_empty() {
            return Err(ErrorData::invalid_params(
                "generation_id is required (the generation_id another tool returned, e.g. \"gen-...\")"
                    .to_string(),
                None,
            ));
        }

        match self.client.get_generation(id).await {
            Ok(record) => {
                self.stats.record_lookup(true).await;
                json_text_result(&record)
            }
            Err(e) => {
                self.stats.record_lookup(false).await;
                let not_found = e
                    .downcast_ref::<HttpFailure>()
                    .is_some_and(|f| f.status == reqwest::StatusCode::NOT_FOUND);
                if not_found {
                    return Err(ErrorData::invalid_params(
                        format!(
                            "generation {id:?} was not found (404): the id is unknown, or its \
                             record is not available yet - records can lag the request by a \
                             few seconds, so retry shortly. Upstream: {e:#}"
                        ),
                        None,
                    ));
                }
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use rmcp::model::ErrorCode;
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn embed_args(v: serde_json::Value) -> Parameters<EmbedTextArgs> {
        Parameters(serde_json::from_value(v).unwrap())
    }

    fn rerank_args(v: serde_json::Value) -> Parameters<RerankDocumentsArgs> {
        Parameters(serde_json::from_value(v).unwrap())
    }

    /// The documented body - array input, dimensions, input_type and
    /// `provider.order` - reaches `/embeddings`; the result is the flattened
    /// envelope and the cost lands in the usage stats as a text generation.
    #[tokio::test]
    async fn embed_text_forwards_the_body_with_provider_order_and_returns_vectors() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(json!({
                "model": "openai/text-embedding-3-small",
                "input": ["a", "b"],
                "dimensions": 2,
                "input_type": "query",
                "provider": {"order": ["openai"]}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-emb")
                    .set_body_json(json!({
                        "model": "openai/text-embedding-3-small",
                        "data": [
                            {"index": 0, "embedding": [0.1, 0.2]},
                            {"index": 1, "embedding": [0.3, 0.4]}
                        ],
                        "usage": {"prompt_tokens": 2, "total_tokens": 2, "cost": 0.0003}
                    })),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        // Deserialized the way a client sends it, so the nested provider block
        // goes through the lenient path rather than a hand-built struct.
        let res = server
            .embed_text(embed_args(json!({
                "model": "openai/text-embedding-3-small",
                "input": ["a", "b"],
                "dimensions": "2",
                "input_type": "query",
                "provider": {"order": ["openai"]}
            })))
            .await
            .unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["model"], "openai/text-embedding-3-small");
        assert_eq!(v["dimensions"], 2);
        assert_eq!(v["count"], 2);
        assert_eq!(v["embeddings"], json!([[0.1, 0.2], [0.3, 0.4]]));
        assert_eq!(v["usage"]["cost"], 0.0003);
        assert_eq!(v["usage"]["prompt_tokens"], 2);
        assert_eq!(v["generation_id"], "gen-emb");

        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 1);
        assert_eq!(stats["actual_cost_usd"], 0.0003);
    }

    /// Exactly one text is sent as a bare string, the shape every provider
    /// accepts; no provider block is sent when none was given.
    #[tokio::test]
    async fn embed_text_sends_a_single_text_as_a_string_and_omits_provider() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(json!({"input": "solo"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": [{"index": 0, "embedding": [1.0]}]
            })))
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let res = server
            .embed_text(embed_args(json!({"model": "m", "input": ["solo"]})))
            .await
            .unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["count"], 1);
        assert_eq!(v["dimensions"], 1);
        // Absent usage is reported as null, not invented.
        assert!(v["usage"]["cost"].is_null());

        let sent: serde_json::Value =
            serde_json::from_slice(&mock.received_requests().await.unwrap()[0].body).unwrap();
        assert!(sent.get("provider").is_none(), "sent: {sent}");
        assert!(sent.get("dimensions").is_none(), "sent: {sent}");
    }

    /// Empty or blank input, and a bad provider block, are invalid params
    /// caught before any HTTP call; nothing is counted as a request.
    #[tokio::test]
    async fn embed_text_rejects_bad_input_before_any_call() {
        let mock = MockServer::start().await;
        let server = server_for(mock.uri());

        let err = server
            .embed_text(embed_args(json!({"model": "m", "input": []})))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("input"), "got: {}", err.message);

        let err = server
            .embed_text(embed_args(json!({"model": "m", "input": ["ok", "  "]})))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("blank"), "got: {}", err.message);

        let err = server
            .embed_text(embed_args(json!({
                "model": "m", "input": ["ok"], "provider": {"sort": "cheapest"}
            })))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("sort"), "got: {}", err.message);

        assert_eq!(mock.received_requests().await.unwrap().len(), 0);
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["requests_total"], 0);
    }

    /// An upstream failure is an internal error that counts as a failed request.
    #[tokio::test]
    async fn embed_text_surfaces_upstream_errors_and_counts_the_failure() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(402).set_body_string("insufficient credits"))
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let err = server
            .embed_text(embed_args(json!({"model": "m", "input": ["x"]})))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::INTERNAL_ERROR);
        assert!(err.message.contains("402"), "got: {}", err.message);
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["requests_total"], 1);
        assert_eq!(stats["requests_failed"], 1);
    }

    /// The documented body reaches `/rerank`, results come back ordered as the
    /// provider ranked them with `text` filled from the echo or the input,
    /// and the cost is recorded.
    #[tokio::test]
    async fn rerank_documents_forwards_the_body_and_returns_ranked_results() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .and(body_partial_json(json!({
                "model": "cohere/rerank-v3.5",
                "query": "rust",
                "documents": ["go", "rust lang", "zig"],
                "top_n": 2,
                "provider": {"order": ["cohere"]}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-rr")
                    .set_body_json(json!({
                        "model": "cohere/rerank-v3.5",
                        "results": [
                            {"index": 1, "relevance_score": 0.98, "document": {"text": "rust lang"}},
                            {"index": 2, "relevance_score": 0.1}
                        ],
                        "usage": {"cost": 0.002, "search_units": 1, "total_tokens": 9}
                    })),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server
            .rerank_documents(rerank_args(json!({
                "model": "cohere/rerank-v3.5",
                "query": "rust",
                "documents": ["go", "rust lang", "zig"],
                "top_n": 2,
                "provider": {"order": ["cohere"]}
            })))
            .await
            .unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["model"], "cohere/rerank-v3.5");
        assert_eq!(
            v["results"],
            json!([
                {"index": 1, "relevance_score": 0.98, "text": "rust lang"},
                {"index": 2, "relevance_score": 0.1, "text": "zig"}
            ])
        );
        assert_eq!(
            v["usage"],
            json!({"cost": 0.002, "search_units": 1, "total_tokens": 9})
        );
        assert_eq!(v["generation_id"], "gen-rr");

        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["text_generations"], 1);
        assert_eq!(stats["actual_cost_usd"], 0.002);
    }

    #[tokio::test]
    async fn rerank_documents_rejects_empty_documents_and_blank_query_before_any_call() {
        let mock = MockServer::start().await;
        let server = server_for(mock.uri());

        let err = server
            .rerank_documents(rerank_args(
                json!({"model": "m", "query": "q", "documents": []}),
            ))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("documents"), "got: {}", err.message);

        let err = server
            .rerank_documents(rerank_args(
                json!({"model": "m", "query": "  ", "documents": ["a"]}),
            ))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("query"), "got: {}", err.message);

        assert_eq!(mock.received_requests().await.unwrap().len(), 0);
    }

    /// The id travels as the `id` query parameter and `data` comes back
    /// verbatim. A lookup is counted as a request but never as a generation
    /// or a cost - the record it returns is the cost of some *other* call.
    #[tokio::test]
    async fn get_generation_sends_the_id_and_returns_the_record_verbatim() {
        let mock = MockServer::start().await;
        let record = json!({
            "id": "gen-1",
            "total_cost": 0.0042,
            "provider_name": "OpenAI",
            "native_tokens_prompt": 12,
            "native_tokens_completion": 30,
            "latency": 812,
            "generation_time": 1530,
            "api_type": "chat.completions",
            "some_future_field": {"nested": true}
        });
        Mock::given(method("GET"))
            .and(path("/generation"))
            .and(query_param("id", "gen-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": record})))
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let res = server
            .get_generation(Parameters(GetGenerationArgs {
                generation_id: " gen-1 ".to_string(),
            }))
            .await
            .unwrap();
        assert_eq!(tool_result_json(&res), record);

        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["requests_total"], 1);
        assert_eq!(stats["requests_failed"], 0);
        assert_eq!(stats["text_generations"], 0);
        assert_eq!(stats["actual_cost_usd"], 0.0);
        assert_eq!(stats["unknown_cost_count"], 0);
    }

    /// A blank id is rejected locally; an upstream 404 (unknown or not yet
    /// indexed id) is an invalid-params error naming the id, not an internal
    /// failure. Other statuses stay internal errors. Both count as requests.
    #[tokio::test]
    async fn get_generation_maps_404_to_invalid_params_and_rejects_a_blank_id() {
        let mock = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/generation"))
            .and(query_param("id", "gen-missing"))
            .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
            .mount(&mock)
            .await;
        Mock::given(method("GET"))
            .and(path("/generation"))
            .and(query_param("id", "gen-boom"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let lookup = |id: &str| {
            server.get_generation(Parameters(GetGenerationArgs {
                generation_id: id.to_string(),
            }))
        };

        let err = lookup("   ").await.unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(
            err.message.contains("generation_id"),
            "got: {}",
            err.message
        );
        assert_eq!(mock.received_requests().await.unwrap().len(), 0);

        let err = lookup("gen-missing").await.unwrap_err();
        assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
        assert!(err.message.contains("gen-missing"), "got: {}", err.message);
        assert!(err.message.contains("404"), "got: {}", err.message);

        let err = lookup("gen-boom").await.unwrap_err();
        assert_eq!(err.code, ErrorCode::INTERNAL_ERROR);
        assert!(err.message.contains("500"), "got: {}", err.message);

        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["requests_total"], 2);
        assert_eq!(stats["requests_failed"], 2);
    }
}
