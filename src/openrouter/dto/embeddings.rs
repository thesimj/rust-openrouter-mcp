//! DTOs for `POST /api/v1/embeddings`.
//!
//! Float vectors only: `encoding_format` is deliberately not sent, because a
//! base64 reply would need an untagged response union to decode. Token-array
//! and multimodal `input` forms are likewise out of scope (documented, skipped).

use serde::{Deserialize, Serialize};

use super::provider::ProviderRouting;

/// Request body for `POST /api/v1/embeddings`. Unset optionals are omitted.
#[derive(Debug, Serialize)]
pub struct EmbeddingsBody {
    pub model: String,
    pub input: EmbeddingsInput,
    /// Output vector size, for models that support truncation (e.g. OpenAI
    /// text-embedding-3-*).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// Provider-specific hint such as "query" / "document" (Cohere, Voyage).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    /// Routing-only block; this endpoint's schema rejects `options`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderRouting>,
}

/// The `input` field: a bare string for one text (the shape every provider
/// accepts), an array for several.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Text(String),
    Texts(Vec<String>),
}

impl EmbeddingsInput {
    /// Exactly one text -> [`Self::Text`]; anything else -> [`Self::Texts`].
    pub fn from_texts(texts: Vec<String>) -> Self {
        match <[String; 1]>::try_from(texts) {
            Ok([text]) => Self::Text(text),
            Err(texts) => Self::Texts(texts),
        }
    }
}

/// Response body: one vector per input, plus token usage and the USD cost.
#[derive(Debug, Deserialize)]
pub struct EmbeddingsResponse {
    pub model: Option<String>,
    pub data: Vec<EmbeddingItem>,
    pub usage: Option<EmbeddingsUsage>,
}

/// One embedding; `index` is the input position (OpenAI-compatible providers
/// send it, so vectors can be restored to input order).
#[derive(Debug, Deserialize)]
pub struct EmbeddingItem {
    pub index: Option<usize>,
    pub embedding: Vec<f64>,
}

#[derive(Debug, Default, Deserialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    /// USD charge for the request, when OpenRouter reports it inline.
    pub cost: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openrouter::ProviderRouting;
    use serde_json::json;

    fn request(input: Vec<&str>, provider: Option<ProviderRouting>) -> EmbeddingsBody {
        EmbeddingsBody {
            model: "openai/text-embedding-3-small".into(),
            input: EmbeddingsInput::from_texts(input.into_iter().map(str::to_string).collect()),
            dimensions: None,
            input_type: None,
            provider,
        }
    }

    /// Serde lock: exactly one text goes on the wire as a bare string (the
    /// shape every provider accepts), more than one as an array; unset
    /// optionals are omitted rather than sent as null.
    #[test]
    fn single_text_is_a_string_and_several_are_an_array() {
        let one = serde_json::to_value(request(vec!["hello"], None)).unwrap();
        assert_eq!(
            one,
            json!({"model": "openai/text-embedding-3-small", "input": "hello"})
        );
        let two = serde_json::to_value(request(vec!["a", "b"], None)).unwrap();
        assert_eq!(two["input"], json!(["a", "b"]));
        assert!(two.get("dimensions").is_none());
        assert!(two.get("input_type").is_none());
        assert!(two.get("provider").is_none());
    }

    #[test]
    fn optionals_and_provider_routing_reach_the_wire_under_their_api_names() {
        let mut req = request(
            vec!["q"],
            Some(ProviderRouting {
                order: vec!["openai".into()],
                ..Default::default()
            }),
        );
        req.dimensions = Some(256);
        req.input_type = Some("query".into());
        let v = serde_json::to_value(req).unwrap();
        assert_eq!(v["dimensions"], 256);
        assert_eq!(v["input_type"], "query");
        assert_eq!(v["provider"], json!({"order": ["openai"]}));
    }

    /// The response keeps float vectors typed; `usage.cost` is the USD charge.
    #[test]
    fn response_parses_vectors_and_usage() {
        let r: EmbeddingsResponse = serde_json::from_value(json!({
            "object": "list",
            "model": "openai/text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.5, -0.25]},
                {"object": "embedding", "index": 0, "embedding": [1.0, 2.0]}
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4, "cost": 0.0001}
        }))
        .unwrap();
        assert_eq!(r.model.as_deref(), Some("openai/text-embedding-3-small"));
        assert_eq!(r.data.len(), 2);
        assert_eq!(r.data[0].index, Some(1));
        assert_eq!(r.data[0].embedding, vec![0.5, -0.25]);
        let usage = r.usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(4));
        assert_eq!(usage.total_tokens, Some(4));
        assert_eq!(usage.cost, Some(0.0001));

        // Minimal shape: no usage, no model, no index.
        let bare: EmbeddingsResponse =
            serde_json::from_value(json!({"data": [{"embedding": [0.1]}]})).unwrap();
        assert!(bare.usage.is_none());
        assert_eq!(bare.data[0].index, None);
    }
}
