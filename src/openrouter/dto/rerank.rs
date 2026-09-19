//! DTOs for `POST /api/v1/rerank`.
//!
//! Documents are plain strings here; the `{text, image}` object form is
//! documented upstream but out of scope (skipped on purpose).

use serde::{Deserialize, Serialize};

use super::provider::ProviderRouting;

/// Request body for `POST /api/v1/rerank`. Unset optionals are omitted.
#[derive(Debug, Serialize)]
pub struct RerankBody {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    /// Return only the best `top_n` documents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
    /// Routing-only block; this endpoint's schema rejects `options`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderRouting>,
}

/// Response body: ranked results (best first, as the provider orders them).
#[derive(Debug, Deserialize)]
pub struct RerankResponse {
    pub model: Option<String>,
    pub results: Vec<RerankItem>,
    pub usage: Option<RerankUsage>,
}

/// One ranked document. `index` points into the request's `documents`;
/// `document` is the provider's echo as an object `{text}`, when present.
#[derive(Debug, Deserialize)]
pub struct RerankItem {
    pub index: usize,
    pub relevance_score: f64,
    pub document: Option<RerankDocument>,
}

#[derive(Debug, Deserialize)]
pub struct RerankDocument {
    pub text: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct RerankUsage {
    /// USD charge for the request, when OpenRouter reports it inline.
    pub cost: Option<f64>,
    /// Cohere-style billing unit.
    pub search_units: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter::ProviderRouting;
    use serde_json::json;

    /// Serde lock: `top_n` and `provider` are omitted when unset; documents are
    /// always sent as an array of strings.
    #[test]
    fn request_matches_the_documented_wire_shape() {
        let minimal = RerankBody {
            model: "cohere/rerank-v3.5".into(),
            query: "what is rust".into(),
            documents: vec!["a".into(), "b".into()],
            top_n: None,
            provider: None,
        };
        assert_eq!(
            serde_json::to_value(&minimal).unwrap(),
            json!({"model": "cohere/rerank-v3.5", "query": "what is rust", "documents": ["a", "b"]})
        );
        let full = RerankBody {
            top_n: Some(1),
            provider: Some(ProviderRouting {
                order: vec!["cohere".into()],
                ..Default::default()
            }),
            ..minimal
        };
        let v = serde_json::to_value(&full).unwrap();
        assert_eq!(v["top_n"], 1);
        assert_eq!(v["provider"], json!({"order": ["cohere"]}));
    }

    /// `results[].document` is an object `{text}` (or absent), never the bare
    /// input string; usage carries cost plus provider-specific counters.
    #[test]
    fn response_parses_results_document_objects_and_usage() {
        let r: RerankResponse = serde_json::from_value(json!({
            "model": "cohere/rerank-v3.5",
            "results": [
                {"index": 1, "relevance_score": 0.9, "document": {"text": "b"}},
                {"index": 0, "relevance_score": 0.1}
            ],
            "usage": {"cost": 0.002, "search_units": 1, "total_tokens": 12}
        }))
        .unwrap();
        assert_eq!(r.model.as_deref(), Some("cohere/rerank-v3.5"));
        assert_eq!(r.results[0].index, 1);
        assert_eq!(r.results[0].relevance_score, 0.9);
        assert_eq!(
            r.results[0]
                .document
                .as_ref()
                .and_then(|d| d.text.as_deref()),
            Some("b")
        );
        assert!(r.results[1].document.is_none());
        let usage = r.usage.unwrap();
        assert_eq!(usage.cost, Some(0.002));
        assert_eq!(usage.search_units, Some(1));
        assert_eq!(usage.total_tokens, Some(12));
    }
}
