//! The `provider` request block OpenRouter accepts on its POST endpoints.
//!
//! Three shapes, because the OpenAPI specs differ per endpoint and the chat
//! family declares its block `additionalProperties: false`, so the *type* has
//! to decide what may be sent:
//!
//! - [`ProviderRouting`] - chat completions, `/embeddings`, `/rerank`: routing
//!   fields only. There is no `options` passthrough on these endpoints.
//! - [`ImageProvider`] - `/images`: the routing subset OpenRouter documents
//!   there (`order`, `only`, `ignore`, `allow_fallbacks`, `sort`) plus `options`.
//! - [`ProviderOptions`] - `/audio/speech`, `/audio/transcriptions`, `/videos`:
//!   `options` only; routing is documented as ignored on these endpoints.
//!
//! `options` is keyed by provider slug (`"deepgram": {"diarize": true}`). Only
//! the slug that serves the request is forwarded; unknown keys are dropped
//! upstream. Values are opaque here on purpose: OpenRouter treats them as
//! opaque too, and each endpoint's `allowed_passthrough_parameters` (surfaced
//! by `describe_model`) is the source of truth for what a provider accepts.
//! For video the docs show both `options.<slug>.<param>` and
//! `options.<slug>.parameters.<param>`; we send whatever the caller gives.
//!
//! Routing vocabulary is the subset users reach for (YAGNI): `order`, `only`,
//! `ignore`, `allow_fallbacks`, `require_parameters`, `zdr`, `sort`. The rest of
//! OpenRouter's `ProviderPreferences` (quantizations, max_price, preferred_*,
//! data_collection, enforce_distillable_text) is added on demand.

use std::collections::BTreeMap;

use serde::Serialize;

/// Passthrough map keyed by provider slug.
pub type ProviderOptionsMap = BTreeMap<String, serde_json::Value>;

/// `sort` is either a bare string (`"price"`) or `{by, partition}` upstream.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ProviderSort {
    By(String),
    Partitioned { by: String, partition: String },
}

/// Routing-only block for chat completions, `/embeddings` and `/rerank`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ProviderRouting {
    /// Provider slugs to try in order; disables load balancing.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<String>,
    /// Allow-list of provider slugs.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub only: Vec<String>,
    /// Deny-list of provider slugs.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ignore: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Route only to providers that support every parameter in the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// Zero-data-retention endpoints only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<ProviderSort>,
}

impl ProviderRouting {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// `None` when nothing is set, so request DTOs can `skip_serializing_if`.
    pub fn non_empty(self) -> Option<Self> {
        (!self.is_empty()).then_some(self)
    }
}

/// The `/images` block: the documented routing subset plus passthrough.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ImageProvider {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub only: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ignore: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<ProviderSort>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub options: ProviderOptionsMap,
}

impl ImageProvider {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub fn non_empty(self) -> Option<Self> {
        (!self.is_empty()).then_some(self)
    }
}

/// Passthrough-only block for the endpoints where OpenRouter ignores routing.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ProviderOptions {
    pub options: ProviderOptionsMap,
}

impl ProviderOptions {
    pub fn non_empty(self) -> Option<Self> {
        (!self.options.is_empty()).then_some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn options(slug: &str, v: serde_json::Value) -> ProviderOptionsMap {
        let mut m = BTreeMap::new();
        m.insert(slug.to_string(), v);
        m
    }

    #[test]
    fn empty_blocks_serialize_to_empty_objects_and_non_empty_drops_them() {
        assert_eq!(
            serde_json::to_value(ProviderRouting::default()).unwrap(),
            json!({})
        );
        assert_eq!(ProviderRouting::default().non_empty(), None);
        assert_eq!(ImageProvider::default().non_empty(), None);
        assert_eq!(ProviderOptions::default().non_empty(), None);
    }

    /// Locks the documented chat/embeddings/rerank wire shape: snake_case
    /// names, string sort, and - because that schema is closed - no `options`.
    #[test]
    fn routing_matches_the_documented_wire_shape() {
        let p = ProviderRouting {
            order: vec!["anthropic".into(), "google-vertex".into()],
            only: vec![],
            ignore: vec!["deepinfra".into()],
            allow_fallbacks: Some(false),
            require_parameters: Some(true),
            zdr: Some(true),
            sort: Some(ProviderSort::By("price".into())),
        };
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(
            v,
            json!({
                "order": ["anthropic", "google-vertex"],
                "ignore": ["deepinfra"],
                "allow_fallbacks": false,
                "require_parameters": true,
                "zdr": true,
                "sort": "price"
            })
        );
        assert!(v.get("options").is_none());
    }

    #[test]
    fn partitioned_sort_serializes_as_an_object() {
        let p = ProviderRouting {
            sort: Some(ProviderSort::Partitioned {
                by: "throughput".into(),
                partition: "none".into(),
            }),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(&p).unwrap(),
            json!({"sort": {"by": "throughput", "partition": "none"}})
        );
    }

    #[test]
    fn image_provider_carries_the_subset_plus_options() {
        let p = ImageProvider {
            order: vec!["black-forest-labs".into()],
            options: options("black-forest-labs", json!({"steps": 28, "guidance": 3.5})),
            ..Default::default()
        };
        assert_eq!(
            serde_json::to_value(&p).unwrap(),
            json!({
                "order": ["black-forest-labs"],
                "options": {"black-forest-labs": {"steps": 28, "guidance": 3.5}}
            })
        );
    }

    #[test]
    fn options_only_block_emits_just_options() {
        let o = ProviderOptions {
            options: options("openai", json!({"instructions": "cheerful"})),
        }
        .non_empty()
        .unwrap();
        assert_eq!(
            serde_json::to_value(&o).unwrap(),
            json!({"options": {"openai": {"instructions": "cheerful"}}})
        );
    }
}
