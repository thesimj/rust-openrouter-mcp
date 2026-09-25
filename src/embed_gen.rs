//! Retrieval helpers for the MCP tools: text embeddings
//! (`POST /embeddings`) and document reranking (`POST /rerank`).
//!
//! Like [`crate::audio_gen`], this is the single path the tools use after
//! their argument checks: the request goes out as the wire body the tool
//! built, and the reply is flattened into the result JSON here once.

use anyhow::{Result, bail};
use serde_json::{Value, json};

use crate::openrouter::{EmbeddingsBody, OpenRouterClient, RerankBody};

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

/// The local checks a rerank body must pass before any HTTP call: a non-blank
/// query and at least one non-blank document.
pub fn validate_rerank(body: &RerankBody) -> Result<()> {
    if body.query.trim().is_empty() {
        bail!("query must not be blank");
    }
    check_texts(&body.documents, "documents")
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

/// Embed one or more texts and restore input order from the response `index`
/// when the provider sends it. The caller has already checked the texts with
/// [`check_texts`].
pub async fn embed(client: &OpenRouterClient, body: &EmbeddingsBody) -> Result<EmbedResult> {
    let (reply, generation_id) = client.embeddings(body).await?;
    let usage = reply.usage.unwrap_or_default();
    let receipt = crate::billing::Receipt {
        cost: usage.cost,
        generation_id: generation_id.clone(),
    };
    if reply.data.is_empty() {
        return Err(receipt.attach(anyhow::anyhow!("model returned no embeddings")));
    }
    let mut data = reply.data;
    // Providers are supposed to answer in input order; `index` is the
    // authority when present, so a reordered reply still lines up.
    let indexed = data.iter().all(|d| d.index.is_some());
    if indexed {
        data.sort_by_key(|d| d.index);
    }
    let inputs = body.input.len();
    let one_per_input = data.len() == inputs
        && (!indexed || data.iter().enumerate().all(|(i, d)| d.index == Some(i)));
    if !one_per_input {
        return Err(receipt.attach(anyhow::anyhow!(
            "model returned {} embeddings for {inputs} inputs (or their indices do not \
             cover 0..{inputs} once each)",
            data.len()
        )));
    }
    Ok(EmbedResult {
        model: reply.model.unwrap_or_else(|| body.model.clone()),
        embeddings: data.into_iter().map(|d| d.embedding).collect(),
        prompt_tokens: usage.prompt_tokens,
        total_tokens: usage.total_tokens,
        cost: usage.cost,
        generation_id,
    })
}

/// Rerank `body.documents` against `body.query` and flatten the reply,
/// filling each result's `text` from the input when the provider does not
/// echo the document. The caller has already run [`validate_rerank`].
pub async fn rerank(client: &OpenRouterClient, body: &RerankBody) -> Result<RerankResult> {
    let (reply, generation_id) = client.rerank(body).await?;
    let usage = reply.usage.unwrap_or_default();
    let results = reply
        .results
        .into_iter()
        .map(|r| RankedDocument {
            index: r.index,
            text: r
                .document
                .and_then(|d| d.text)
                .or_else(|| body.documents.get(r.index).cloned()),
            relevance_score: r.relevance_score,
        })
        .collect();
    Ok(RerankResult {
        model: reply.model.unwrap_or_else(|| body.model.clone()),
        results,
        cost: usage.cost,
        search_units: usage.search_units,
        total_tokens: usage.total_tokens,
        generation_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter::{EmbeddingsInput, OpenRouterClient, ProviderRouting};
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn embed_body(input: &[&str]) -> EmbeddingsBody {
        EmbeddingsBody {
            model: "openai/text-embedding-3-small".into(),
            input: EmbeddingsInput::from_texts(strings(input)),
            dimensions: None,
            input_type: None,
            provider: None,
        }
    }

    fn rerank_body(documents: &[&str]) -> RerankBody {
        RerankBody {
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

    /// The result promises one vector per input, in input order: a reply
    /// that is short, or whose indices skip or repeat, is a billed failure,
    /// not a success that pairs vectors with the wrong texts.
    #[tokio::test]
    async fn embed_rejects_a_reply_that_does_not_match_the_inputs() {
        for data in [
            json!([{"index": 0, "embedding": [0.1]}]),
            json!([{"index": 0, "embedding": [0.1]}, {"index": 0, "embedding": [0.2]}]),
            json!([{"index": 0, "embedding": [0.1]}, {"index": 2, "embedding": [0.2]}]),
            json!([{"embedding": [0.1]}]),
        ] {
            let mock = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/embeddings"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "data": data, "usage": {"cost": 0.00001}
                })))
                .mount(&mock)
                .await;
            let client = OpenRouterClient::with_base_url(mock.uri(), "test-key");
            let err = embed(&client, &embed_body(&["a", "b"]))
                .await
                .expect_err(&format!("must fail: {data}"));
            assert!(err.to_string().contains("2 inputs"), "{data}: {err}");
            let receipt = crate::billing::Receipt::from_error(&err).expect("billed");
            assert_eq!(receipt.cost, Some(0.00001));
        }
    }

    #[test]
    fn validate_rerank_requires_a_query_and_documents() {
        let err = validate_rerank(&rerank_body(&[])).unwrap_err();
        assert!(err.to_string().contains("documents"), "got: {err}");
        let mut blank_query = rerank_body(&["a"]);
        blank_query.query = "  ".into();
        let err = validate_rerank(&blank_query).unwrap_err();
        assert!(err.to_string().contains("query"), "got: {err}");
        validate_rerank(&rerank_body(&["a"])).unwrap();
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
        let mut body = embed_body(&["a", "b"]);
        body.dimensions = Some(2);
        body.input_type = Some("document".into());
        body.provider = Some(ProviderRouting {
            order: vec!["openai".into()],
            ..Default::default()
        });
        let result = embed(&client, &body).await.unwrap();
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

    /// A 2xx with no vectors is an error that still carries the receipt (the
    /// provider may have billed).
    #[tokio::test]
    async fn embed_reports_empty_data_with_the_receipt() {
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
        let err = embed(&client, &embed_body(&["x"])).await.unwrap_err();
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
        let mut body = rerank_body(&["go", "rust lang", "zig"]);
        body.top_n = Some(2);
        let result = rerank(&client, &body).await.unwrap();
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
        let result = rerank(&client, &rerank_body(&["only"])).await.unwrap();
        assert_eq!(result.results[0].index, 7);
        assert_eq!(result.results[0].text, None);
        // The request model is the fallback when the reply names none.
        assert_eq!(result.model, "cohere/rerank-v3.5");
        assert_eq!(result.cost, None);
    }
}
