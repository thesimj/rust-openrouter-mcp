//! Retrieval helpers for the MCP tools: text embeddings
//! (`POST /embeddings`) and document reranking (`POST /rerank`).
//!
//! Like [`crate::audio_gen`], this is the single path both front ends use:
//! input validation, the single-string-vs-array normalization, response
//! flattening and the result JSON all live here once.

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::openrouter::{
    EmbeddingsBody, EmbeddingsInput, OpenRouterClient, ProviderRouting, RerankBody,
};

/// Inputs for one `/embeddings` request.
#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub model: String,
    /// One or more texts; sent as a bare string when there is exactly one.
    pub input: Vec<String>,
    pub dimensions: Option<u32>,
    pub input_type: Option<String>,
    /// Routing block, already validated (this endpoint has no `options`).
    pub provider: Option<ProviderRouting>,
}

/// One vector per input text, in input order, plus usage and the receipt id.
#[derive(Debug)]
pub struct EmbedResult {
    pub model: String,
    pub embeddings: Vec<Vec<f64>>,
    pub prompt_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cost: Option<f64>,
    pub generation_id: Option<String>,
}

impl EmbedResult {
    /// Vector length, taken from the first embedding (0 when none).
    pub fn dimensions(&self) -> usize {
        self.embeddings.first().map_or(0, Vec::len)
    }

    /// The result envelope returned by the MCP tool.
    pub fn to_json(&self) -> Value {
        json!({
            "model": self.model,
            "dimensions": self.dimensions(),
            "count": self.embeddings.len(),
            "embeddings": self.embeddings,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "total_tokens": self.total_tokens,
                "cost": self.cost,
            },
            "generation_id": self.generation_id,
        })
    }
}

/// Inputs for one `/rerank` request.
#[derive(Debug, Clone)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<u32>,
    /// Routing block, already validated (this endpoint has no `options`).
    pub provider: Option<ProviderRouting>,
}

/// One ranked document: its position in the request, the provider's score,
/// and its text (the provider's echo, else the input at `index`).
#[derive(Debug)]
pub struct RankedDocument {
    pub index: usize,
    pub relevance_score: f64,
    pub text: Option<String>,
}

/// Ranked results in the order the provider returned them (best first).
#[derive(Debug)]
pub struct RerankResult {
    pub model: String,
    pub results: Vec<RankedDocument>,
    pub cost: Option<f64>,
    pub search_units: Option<u64>,
    pub total_tokens: Option<u64>,
    pub generation_id: Option<String>,
}

impl RerankResult {
    /// The result envelope returned by the MCP tool.
    pub fn to_json(&self) -> Value {
        let results: Vec<Value> = self
            .results
            .iter()
            .map(|r| {
                json!({
                    "index": r.index,
                    "relevance_score": r.relevance_score,
                    "text": r.text,
                })
            })
            .collect();
        json!({
            "model": self.model,
            "results": results,
            "usage": {
                "cost": self.cost,
                "search_units": self.search_units,
                "total_tokens": self.total_tokens,
            },
            "generation_id": self.generation_id,
        })
    }
}

impl EmbedRequest {
    /// The local checks run before any HTTP call: at least one non-blank text.
    pub fn validate(&self) -> Result<()> {
        check_texts(&self.input, "input")
    }
}

impl RerankRequest {
    /// The local checks run before any HTTP call: a non-blank query and at
    /// least one non-blank document.
    pub fn validate(&self) -> Result<()> {
        if self.query.trim().is_empty() {
            bail!("query must not be blank");
        }
        check_texts(&self.documents, "documents")
    }
}

/// At least one text, none blank (whitespace-only counts as absent). `field`
/// names the argument in the error (`input` / `documents`).
pub fn check_texts(texts: &[String], field: &str) -> Result<()> {
    if texts.is_empty() {
        bail!("{field} must contain at least one text");
    }
    if let Some(pos) = texts.iter().position(|t| t.trim().is_empty()) {
        bail!("{field}[{pos}] is blank - every entry must contain text");
    }
    Ok(())
}

/// Embed one or more texts. Validates locally, sends the documented body, and
/// restores input order from the response `index` when the provider sends it.
pub async fn embed(client: &OpenRouterClient, req: &EmbedRequest) -> Result<EmbedResult> {
    req.validate()?;
    let body = EmbeddingsBody {
        model: req.model.clone(),
        input: EmbeddingsInput::from_texts(req.input.clone()),
        dimensions: req.dimensions,
        input_type: req.input_type.clone(),
        provider: req.provider.clone(),
    };
    let reply = client.embeddings(&body).await?;
    let usage = reply.body.usage.unwrap_or_default();
    let receipt = crate::billing::Receipt {
        cost: usage.cost,
        generation_id: reply.generation_id.clone(),
    };
    if reply.body.data.is_empty() {
        return Err(receipt.attach(anyhow::anyhow!("model returned no embeddings")));
    }
    let mut data = reply.body.data;
    // Providers are supposed to answer in input order; `index` is the
    // authority when present, so a reordered reply still lines up.
    if data.iter().all(|d| d.index.is_some()) {
        data.sort_by_key(|d| d.index);
    }
    Ok(EmbedResult {
        model: reply.body.model.unwrap_or_else(|| req.model.clone()),
        embeddings: data.into_iter().map(|d| d.embedding).collect(),
        prompt_tokens: usage.prompt_tokens,
        total_tokens: usage.total_tokens,
        cost: usage.cost,
        generation_id: reply.generation_id,
    })
}

/// Rerank `documents` against `query`. Validates locally and flattens the
/// reply, filling each result's `text` from the input when the provider does
/// not echo the document.
pub async fn rerank(client: &OpenRouterClient, req: &RerankRequest) -> Result<RerankResult> {
    req.validate()?;
    let body = RerankBody {
        model: req.model.clone(),
        query: req.query.clone(),
        documents: req.documents.clone(),
        top_n: req.top_n,
        provider: req.provider.clone(),
    };
    let reply = client.rerank(&body).await?;
    let usage = reply.body.usage.unwrap_or_default();
    let results = reply
        .body
        .results
        .into_iter()
        .map(|r| RankedDocument {
            index: r.index,
            text: r
                .document
                .and_then(|d| d.text)
                .or_else(|| req.documents.get(r.index).cloned()),
            relevance_score: r.relevance_score,
        })
        .collect();
    Ok(RerankResult {
        model: reply.body.model.unwrap_or_else(|| req.model.clone()),
        results,
        cost: usage.cost,
        search_units: usage.search_units,
        total_tokens: usage.total_tokens,
        generation_id: reply.generation_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter::{OpenRouterClient, ProviderRouting};
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn embed_request(input: &[&str]) -> EmbedRequest {
        EmbedRequest {
            model: "openai/text-embedding-3-small".into(),
            input: strings(input),
            dimensions: None,
            input_type: None,
            provider: None,
        }
    }

    fn rerank_request(documents: &[&str]) -> RerankRequest {
        RerankRequest {
            model: "cohere/rerank-v3.5".into(),
            query: "rust".into(),
            documents: strings(documents),
            top_n: None,
            provider: None,
        }
    }

    /// The one validation rule both tools share: at least one text, none of
    /// them blank. Whitespace-only counts as absent (repo-wide rule).
    #[test]
    fn check_texts_requires_at_least_one_non_blank_entry() {
        assert!(check_texts(&[], "input").is_err());
        let err = check_texts(&strings(&["ok", "   "]), "documents").unwrap_err();
        assert!(err.to_string().contains("documents"), "got: {err}");
        assert!(err.to_string().contains("blank"), "got: {err}");
        check_texts(&strings(&["a"]), "input").unwrap();
        check_texts(&strings(&["a", "b"]), "input").unwrap();
    }

    /// Two texts go as an array, `provider.order` rides along, and the result
    /// is flattened: vectors in input order (by `index`), dims from the first
    /// vector, usage and the generation id kept for the cost lookup.
    #[tokio::test]
    async fn embed_sends_array_input_and_flattens_the_response() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .and(body_partial_json(json!({
                "model": "openai/text-embedding-3-small",
                "input": ["a", "b"],
                "dimensions": 2,
                "input_type": "document",
                "provider": {"order": ["openai"]}
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-e")
                    .set_body_json(json!({
                        "model": "openai/text-embedding-3-small",
                        // Out of order on purpose: the caller gets input order.
                        "data": [
                            {"index": 1, "embedding": [0.3, 0.4]},
                            {"index": 0, "embedding": [0.1, 0.2]}
                        ],
                        "usage": {"prompt_tokens": 2, "total_tokens": 2, "cost": 0.00002}
                    })),
            )
            .mount(&mock)
            .await;

        let client = OpenRouterClient::with_base_url(mock.uri(), "test-key");
        let mut req = embed_request(&["a", "b"]);
        req.dimensions = Some(2);
        req.input_type = Some("document".into());
        req.provider = Some(ProviderRouting {
            order: vec!["openai".into()],
            ..Default::default()
        });
        let result = embed(&client, &req).await.unwrap();
        assert_eq!(result.embeddings, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert_eq!(result.dimensions(), 2);
        assert_eq!(result.cost, Some(0.00002));
        assert_eq!(result.generation_id.as_deref(), Some("gen-e"));

        let v = result.to_json();
        assert_eq!(v["model"], "openai/text-embedding-3-small");
        assert_eq!(v["dimensions"], 2);
        assert_eq!(v["count"], 2);
        assert_eq!(v["embeddings"][1], json!([0.3, 0.4]));
        assert_eq!(
            v["usage"],
            json!({"prompt_tokens": 2, "total_tokens": 2, "cost": 0.00002})
        );
        assert_eq!(v["generation_id"], "gen-e");
    }

    /// Validation runs before any HTTP call; a 2xx with no vectors is an error
    /// that still carries the receipt (the provider may have billed).
    #[tokio::test]
    async fn embed_rejects_bad_input_locally_and_empty_data_with_receipt() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-empty")
                    .set_body_json(json!({"data": [], "usage": {"cost": 0.0}})),
            )
            .mount(&mock)
            .await;
        let client = OpenRouterClient::with_base_url(mock.uri(), "test-key");

        let err = embed(&client, &embed_request(&[])).await.unwrap_err();
        assert!(err.to_string().contains("input"), "got: {err}");
        assert_eq!(mock.received_requests().await.unwrap().len(), 0);

        let err = embed(&client, &embed_request(&["x"])).await.unwrap_err();
        assert!(err.to_string().contains("no embeddings"), "got: {err}");
        let receipt = crate::billing::Receipt::from_error(&err).expect("receipt kept");
        assert_eq!(receipt.generation_id.as_deref(), Some("gen-empty"));
        assert_eq!(receipt.cost, Some(0.0));
    }

    /// Results keep the provider's order; `text` comes from `document.text`
    /// and falls back to the input document at `index` when the provider
    /// echoes nothing; usage counters and the generation id are kept.
    #[tokio::test]
    async fn rerank_sends_documents_and_fills_text_from_the_input_when_absent() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .and(body_partial_json(json!({
                "model": "cohere/rerank-v3.5",
                "query": "rust",
                "documents": ["go", "rust lang", "zig"],
                "top_n": 2
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-r")
                    .set_body_json(json!({
                        "model": "cohere/rerank-v3.5",
                        "results": [
                            {"index": 1, "relevance_score": 0.98, "document": {"text": "rust lang (echoed)"}},
                            {"index": 2, "relevance_score": 0.10}
                        ],
                        "usage": {"cost": 0.002, "search_units": 1, "total_tokens": 9}
                    })),
            )
            .mount(&mock)
            .await;

        let client = OpenRouterClient::with_base_url(mock.uri(), "test-key");
        let mut req = rerank_request(&["go", "rust lang", "zig"]);
        req.top_n = Some(2);
        let result = rerank(&client, &req).await.unwrap();
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].index, 1);
        assert_eq!(
            result.results[0].text.as_deref(),
            Some("rust lang (echoed)")
        );
        assert_eq!(result.results[1].index, 2);
        assert_eq!(result.results[1].text.as_deref(), Some("zig"));
        assert_eq!(result.cost, Some(0.002));
        assert_eq!(result.generation_id.as_deref(), Some("gen-r"));

        let v = result.to_json();
        assert_eq!(v["model"], "cohere/rerank-v3.5");
        assert_eq!(
            v["results"][1],
            json!({"index": 2, "relevance_score": 0.1, "text": "zig"})
        );
        assert_eq!(
            v["usage"],
            json!({"cost": 0.002, "search_units": 1, "total_tokens": 9})
        );
        assert_eq!(v["generation_id"], "gen-r");
    }

    #[tokio::test]
    async fn rerank_rejects_empty_documents_and_blank_query_before_any_call() {
        let mock = MockServer::start().await;
        let client = OpenRouterClient::with_base_url(mock.uri(), "test-key");

        let err = rerank(&client, &rerank_request(&[])).await.unwrap_err();
        assert!(err.to_string().contains("documents"), "got: {err}");

        let mut blank_query = rerank_request(&["a"]);
        blank_query.query = "  ".into();
        let err = rerank(&client, &blank_query).await.unwrap_err();
        assert!(err.to_string().contains("query"), "got: {err}");

        assert_eq!(mock.received_requests().await.unwrap().len(), 0);
    }

    /// An out-of-range `index` from a buggy provider yields no text, not a panic.
    #[tokio::test]
    async fn rerank_tolerates_an_out_of_range_index() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rerank"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": [{"index": 7, "relevance_score": 0.5}]
            })))
            .mount(&mock)
            .await;
        let client = OpenRouterClient::with_base_url(mock.uri(), "test-key");
        let result = rerank(&client, &rerank_request(&["only"])).await.unwrap();
        assert_eq!(result.results[0].index, 7);
        assert_eq!(result.results[0].text, None);
        // The request model is the fallback when the reply names none.
        assert_eq!(result.model, "cohere/rerank-v3.5");
        assert_eq!(result.cost, None);
    }
}
