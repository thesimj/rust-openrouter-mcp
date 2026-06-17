//! DTOs for `GET /api/v1/models`: the query parameters, the model entry, and
//! its capability/pricing descriptors.

use serde::{Deserialize, Serialize};

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
    pub(crate) fn to_pairs(&self) -> Vec<(&'static str, String)> {
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
    use super::*;

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
}
