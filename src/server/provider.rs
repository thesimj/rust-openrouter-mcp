//! Tool-argument shapes for the OpenRouter `provider` request block and their
//! conversion to the wire types in [`crate::openrouter`].
//!
//! This is the one place tool arguments (MCP or CLI `--provider <json>`) become
//! [`ProviderRouting`] / [`ImageProvider`] / [`ProviderOptions`], so the
//! vocabulary checks live here once. Three arg structs mirror the three wire
//! types because the endpoints differ in what they accept (see the DTO docs).
//!
//! Schema safety: these are nested `$defs` types in every tool-args schema, so
//! each carries its own `scalarize_nullable`, every field has
//! `#[serde(default)]` (nothing lands in `required`), `sort` is a plain string
//! plus `sort_partition` (an untagged enum would emit a nested `anyOf`), and
//! `options` is a `BTreeMap<String, Value>` (emits `additionalProperties: true`,
//! not the bare `true` an `Option<Value>` would).

use std::collections::BTreeMap;

use rmcp::ErrorData;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::openrouter::{
    ImageProvider, ProviderOptions, ProviderOptionsMap, ProviderRouting, ProviderSort,
};
use crate::server::schema::{de_lenient, de_opt_bool, scalarize_nullable};

/// Accepted `sort` values (OpenRouter `ProviderPreferences.sort`).
const SORT_VALUES: [&str; 4] = ["price", "throughput", "latency", "exacto"];
/// Accepted `sort_partition` values (`sort.partition` in the object form).
const SORT_PARTITIONS: [&str; 2] = ["model", "none"];

/// Routing-only `provider` block for chat completions, `/embeddings` and
/// `/rerank` - the endpoints whose schema rejects `options`.
#[derive(Debug, Default, Clone, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct ProviderRoutingArgs {
    /// Provider slugs to try in this order (e.g. ["anthropic", "google-vertex"]);
    /// disables load balancing.
    #[serde(default)]
    pub order: Vec<String>,
    /// Allow-list: route only to these provider slugs.
    #[serde(default)]
    pub only: Vec<String>,
    /// Deny-list: never route to these provider slugs.
    #[serde(default)]
    pub ignore: Vec<String>,
    /// Fall back to other providers when the preferred ones fail (default true
    /// upstream). false = fail instead of falling back.
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub allow_fallbacks: Option<bool>,
    /// Route only to providers that support every parameter in the request.
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub require_parameters: Option<bool>,
    /// Zero-data-retention endpoints only.
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub zdr: Option<bool>,
    /// Provider ordering: "price", "throughput", "latency", or "exacto".
    #[serde(default)]
    pub sort: Option<String>,
    /// With `sort`: "model" (default upstream) or "none". Only meaningful
    /// together with `sort`.
    #[serde(default)]
    pub sort_partition: Option<String>,
}

impl ProviderRoutingArgs {
    /// Validate and convert; `Ok(None)` when nothing is set.
    pub(crate) fn into_routing(self) -> Result<Option<ProviderRouting>, ErrorData> {
        let sort = parse_sort(self.sort, self.sort_partition)?;
        Ok(ProviderRouting {
            order: clean_slugs(self.order),
            only: clean_slugs(self.only),
            ignore: clean_slugs(self.ignore),
            allow_fallbacks: self.allow_fallbacks,
            require_parameters: self.require_parameters,
            zdr: self.zdr,
            sort,
        }
        .non_empty())
    }
}

/// The `/images` `provider` block: the routing subset OpenRouter documents
/// there plus per-provider passthrough `options`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed once the images and chat phases expose provider routing"
    )
)]
#[derive(Debug, Default, Clone, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct ImageProviderArgs {
    /// Provider slugs to try in this order; disables load balancing.
    #[serde(default)]
    pub order: Vec<String>,
    /// Allow-list: route only to these provider slugs.
    #[serde(default)]
    pub only: Vec<String>,
    /// Deny-list: never route to these provider slugs.
    #[serde(default)]
    pub ignore: Vec<String>,
    /// Fall back to other providers when the preferred ones fail (default true
    /// upstream). false = fail instead of falling back.
    #[serde(default, deserialize_with = "de_opt_bool")]
    pub allow_fallbacks: Option<bool>,
    /// Provider ordering: "price", "throughput", "latency", or "exacto".
    #[serde(default)]
    pub sort: Option<String>,
    /// With `sort`: "model" (default upstream) or "none".
    #[serde(default)]
    pub sort_partition: Option<String>,
    /// Per-provider passthrough, keyed by provider slug, each value an object of
    /// that provider's parameters (see `allowed_passthrough_parameters` from
    /// describe_model), e.g. {"black-forest-labs": {"steps": 20}}. Only the
    /// slug that serves the request is forwarded.
    #[serde(default, deserialize_with = "de_lenient")]
    pub options: BTreeMap<String, serde_json::Value>,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed once the images and chat phases expose provider routing"
    )
)]
impl ImageProviderArgs {
    /// Validate and convert; `Ok(None)` when nothing is set.
    pub(crate) fn into_image_provider(self) -> Result<Option<ImageProvider>, ErrorData> {
        let sort = parse_sort(self.sort, self.sort_partition)?;
        Ok(ImageProvider {
            order: clean_slugs(self.order),
            only: clean_slugs(self.only),
            ignore: clean_slugs(self.ignore),
            allow_fallbacks: self.allow_fallbacks,
            sort,
            options: validate_options(self.options)?,
        }
        .non_empty())
    }
}

/// Passthrough-only `provider` block for `/audio/speech`,
/// `/audio/transcriptions` and `/videos`, where routing fields are ignored.
#[derive(Debug, Default, Clone, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct ProviderOptionsArgs {
    /// Per-provider passthrough, keyed by provider slug, each value an object of
    /// that provider's parameters, e.g. {"deepgram": {"diarize": true}}. Only
    /// the slug that serves the request is forwarded; unknown keys are dropped
    /// upstream.
    #[serde(default, deserialize_with = "de_lenient")]
    pub options: BTreeMap<String, serde_json::Value>,
}

impl ProviderOptionsArgs {
    /// Validate and convert; `Ok(None)` when nothing is set.
    pub(crate) fn into_options(self) -> Result<Option<ProviderOptions>, ErrorData> {
        Ok(ProviderOptions {
            options: validate_options(self.options)?,
        }
        .non_empty())
    }
}

/// Trim slugs and drop blank entries (the repo-wide "blank means absent" rule).
fn clean_slugs(slugs: Vec<String>) -> Vec<String> {
    slugs
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// `sort` + `sort_partition` -> the wire enum. Blank strings count as unset;
/// vocabulary is checked case-insensitively; a partition without a sort is an
/// error because upstream has nowhere to put it.
fn parse_sort(
    sort: Option<String>,
    partition: Option<String>,
) -> Result<Option<ProviderSort>, ErrorData> {
    let normalize = |s: Option<String>| {
        s.map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
    };
    let sort = normalize(sort);
    let partition = normalize(partition);
    if let Some(s) = &sort
        && !SORT_VALUES.contains(&s.as_str())
    {
        return Err(ErrorData::invalid_params(
            format!(
                "provider.sort must be one of {}; got {s:?}",
                SORT_VALUES.join(", ")
            ),
            None,
        ));
    }
    if let Some(p) = &partition
        && !SORT_PARTITIONS.contains(&p.as_str())
    {
        return Err(ErrorData::invalid_params(
            format!(
                "provider.sort_partition must be one of {}; got {p:?}",
                SORT_PARTITIONS.join(", ")
            ),
            None,
        ));
    }
    Ok(match (sort, partition) {
        (None, None) => None,
        (None, Some(_)) => {
            return Err(ErrorData::invalid_params(
                "provider.sort_partition is only meaningful together with provider.sort",
                None,
            ));
        }
        (Some(by), None) => Some(ProviderSort::By(by)),
        (Some(by), Some(partition)) => Some(ProviderSort::Partitioned { by, partition }),
    })
}

/// Every `options` value must be an object keyed by a non-blank provider slug:
/// OpenRouter forwards `options.<slug>` as that provider's parameter map, so a
/// scalar there is always a caller mistake (`{"deepgram": true}` for
/// `{"deepgram": {"diarize": true}}`). Values are otherwise passed through
/// untouched.
fn validate_options(
    options: BTreeMap<String, serde_json::Value>,
) -> Result<ProviderOptionsMap, ErrorData> {
    let mut out = ProviderOptionsMap::new();
    for (slug, value) in options {
        let slug = slug.trim().to_string();
        if slug.is_empty() {
            return Err(ErrorData::invalid_params(
                "provider.options keys must be provider slugs (e.g. \"deepgram\"), got a blank key",
                None,
            ));
        }
        if !value.is_object() {
            return Err(ErrorData::invalid_params(
                format!(
                    "provider.options.{slug} must be an object of that provider's parameters \
                     (e.g. {{\"{slug}\": {{\"param\": value}}}}), got {value}"
                ),
                None,
            ));
        }
        out.insert(slug, value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::schema::schema_json;
    use serde_json::json;

    fn opts(slug: &str, v: serde_json::Value) -> ProviderOptionsMap {
        let mut m = ProviderOptionsMap::new();
        m.insert(slug.to_string(), v);
        m
    }

    #[test]
    fn empty_args_convert_to_none() {
        assert_eq!(ProviderRoutingArgs::default().into_routing().unwrap(), None);
        assert_eq!(
            ImageProviderArgs::default().into_image_provider().unwrap(),
            None
        );
        assert_eq!(ProviderOptionsArgs::default().into_options().unwrap(), None);
        // Blank-only slugs count as nothing set.
        let blank = ProviderRoutingArgs {
            order: vec!["  ".into(), String::new()],
            ..Default::default()
        };
        assert_eq!(blank.into_routing().unwrap(), None);
    }

    #[test]
    fn routing_args_map_onto_the_wire_type_with_trimmed_slugs() {
        let args = ProviderRoutingArgs {
            order: vec![" anthropic ".into(), "google-vertex".into()],
            only: vec![],
            ignore: vec!["deepinfra".into()],
            allow_fallbacks: Some(false),
            require_parameters: Some(true),
            zdr: Some(true),
            sort: Some("price".into()),
            sort_partition: None,
        };
        let routing = args.into_routing().unwrap().unwrap();
        assert_eq!(
            routing,
            ProviderRouting {
                order: vec!["anthropic".into(), "google-vertex".into()],
                only: vec![],
                ignore: vec!["deepinfra".into()],
                allow_fallbacks: Some(false),
                require_parameters: Some(true),
                zdr: Some(true),
                sort: Some(ProviderSort::By("price".into())),
            }
        );
    }

    #[test]
    fn sort_vocabulary_is_validated_case_insensitively() {
        for ok in ["price", "Throughput", " latency ", "exacto"] {
            let args = ProviderRoutingArgs {
                sort: Some(ok.into()),
                ..Default::default()
            };
            let routing = args.into_routing().unwrap().unwrap();
            assert_eq!(
                routing.sort,
                Some(ProviderSort::By(ok.trim().to_ascii_lowercase()))
            );
        }
        let bad = ProviderRoutingArgs {
            sort: Some("cheapest".into()),
            ..Default::default()
        };
        let err = bad.into_routing().unwrap_err();
        assert!(err.message.contains("sort"), "got: {}", err.message);
        assert!(err.message.contains("price"), "got: {}", err.message);
        // A blank sort is "not set", not an error.
        let blank = ProviderRoutingArgs {
            sort: Some("  ".into()),
            ..Default::default()
        };
        assert_eq!(blank.into_routing().unwrap(), None);
    }

    #[test]
    fn sort_partition_needs_sort_and_a_known_value() {
        let with = ProviderRoutingArgs {
            sort: Some("throughput".into()),
            sort_partition: Some("None".into()),
            ..Default::default()
        };
        assert_eq!(
            with.into_routing().unwrap().unwrap().sort,
            Some(ProviderSort::Partitioned {
                by: "throughput".into(),
                partition: "none".into(),
            })
        );
        let model = ProviderRoutingArgs {
            sort: Some("latency".into()),
            sort_partition: Some("model".into()),
            ..Default::default()
        };
        assert!(matches!(
            model.into_routing().unwrap().unwrap().sort,
            Some(ProviderSort::Partitioned { .. })
        ));

        let orphan = ProviderRoutingArgs {
            sort_partition: Some("model".into()),
            ..Default::default()
        };
        let err = orphan.into_routing().unwrap_err();
        assert!(
            err.message.contains("sort_partition"),
            "got: {}",
            err.message
        );

        let unknown = ProviderRoutingArgs {
            sort: Some("price".into()),
            sort_partition: Some("region".into()),
            ..Default::default()
        };
        let err = unknown.into_routing().unwrap_err();
        assert!(err.message.contains("model"), "got: {}", err.message);
    }

    #[test]
    fn image_provider_args_carry_the_subset_plus_options() {
        let args = ImageProviderArgs {
            order: vec!["black-forest-labs".into()],
            sort: Some("price".into()),
            options: opts("black-forest-labs", json!({"steps": 20})),
            ..Default::default()
        };
        let p = args.into_image_provider().unwrap().unwrap();
        assert_eq!(p.order, vec!["black-forest-labs".to_string()]);
        assert_eq!(p.sort, Some(ProviderSort::By("price".into())));
        assert_eq!(p.options["black-forest-labs"], json!({"steps": 20}));
        // Same sort rules as routing.
        let bad = ImageProviderArgs {
            sort_partition: Some("model".into()),
            ..Default::default()
        };
        assert!(bad.into_image_provider().is_err());
    }

    #[test]
    fn options_values_must_be_objects_keyed_by_a_slug() {
        let ok = ProviderOptionsArgs {
            options: opts("deepgram", json!({"diarize": true})),
        };
        let o = ok.into_options().unwrap().unwrap();
        assert_eq!(o.options["deepgram"], json!({"diarize": true}));

        for bad in [json!(true), json!("diarize"), json!([1]), json!(null)] {
            let args = ProviderOptionsArgs {
                options: opts("deepgram", bad.clone()),
            };
            let err = args.into_options().unwrap_err();
            assert!(err.message.contains("deepgram"), "got: {}", err.message);
            assert!(err.message.contains("object"), "got: {}", err.message);
        }
        let blank_slug = ProviderOptionsArgs {
            options: opts("  ", json!({"k": 1})),
        };
        assert!(blank_slug.into_options().is_err());
        // Image options obey the same rule.
        let img = ImageProviderArgs {
            options: opts("acme", json!(3)),
            ..Default::default()
        };
        assert!(img.into_image_provider().is_err());
    }

    /// Args deserialize leniently: `options` as an object or as a JSON string,
    /// booleans possibly stringified, everything optional.
    #[test]
    fn args_deserialize_leniently() {
        let a: ProviderOptionsArgs =
            serde_json::from_value(json!({"options": {"deepgram": {"diarize": true}}})).unwrap();
        assert_eq!(a.options["deepgram"], json!({"diarize": true}));
        let s: ProviderOptionsArgs =
            serde_json::from_value(json!({"options": "{\"deepgram\":{\"diarize\":true}}"}))
                .unwrap();
        assert_eq!(s.options["deepgram"], json!({"diarize": true}));
        let n: ProviderOptionsArgs = serde_json::from_value(json!({"options": null})).unwrap();
        assert!(n.options.is_empty());
        let e: ProviderOptionsArgs = serde_json::from_value(json!({})).unwrap();
        assert!(e.options.is_empty());

        let r: ProviderRoutingArgs =
            serde_json::from_value(json!({"zdr": "true", "order": ["a"]})).unwrap();
        assert_eq!(r.zdr, Some(true));
        assert_eq!(r.order, vec!["a".to_string()]);
        let img: ImageProviderArgs = serde_json::from_value(json!({})).unwrap();
        assert!(img.options.is_empty());
    }

    /// On a tool-args root the nested block is a `$ref` into `$defs` carrying
    /// the field's doc, never in `required`, and the referenced definition is
    /// present - the shape clients render as an optional nested object.
    #[test]
    fn provider_field_is_an_optional_ref_on_the_tool_args_root() {
        let schema = schema_json::<crate::server::audio::TranscribeAudioArgs>();
        let provider = &schema["properties"]["provider"];
        assert_eq!(
            provider["$ref"],
            json!("#/$defs/ProviderOptionsArgs"),
            "got: {provider}"
        );
        assert!(
            provider["description"]
                .as_str()
                .is_some_and(|d| d.contains("deepgram")),
            "field doc carries the diarization recipe: {provider}"
        );
        assert!(schema["$defs"]["ProviderOptionsArgs"].is_object());
        let required = schema["required"].as_array().cloned().unwrap_or_default();
        assert!(!required.contains(&json!("provider")), "{required:?}");
    }

    /// `options` must advertise a plain open object - the schema shape clients
    /// handle - and none of the args structs may carry a union anywhere.
    #[test]
    fn options_schema_is_an_open_object_without_unions() {
        fn options_schema(mut schema: serde_json::Value) -> serde_json::Value {
            let mut o = schema["properties"]["options"].take();
            // Doc comment and serde default are allowed decorations.
            o.as_object_mut().map(|m| {
                m.remove("description");
                m.remove("default")
            });
            o
        }
        let expected = json!({"type": "object", "additionalProperties": true});
        assert_eq!(
            options_schema(schema_json::<ProviderOptionsArgs>()),
            expected
        );
        assert_eq!(options_schema(schema_json::<ImageProviderArgs>()), expected);
        for (name, schema) in [
            ("ProviderRoutingArgs", schema_json::<ProviderRoutingArgs>()),
            ("ImageProviderArgs", schema_json::<ImageProviderArgs>()),
            ("ProviderOptionsArgs", schema_json::<ProviderOptionsArgs>()),
        ] {
            let text = schema.to_string();
            assert!(!text.contains("anyOf"), "{name}: {text}");
            assert!(!text.contains("oneOf"), "{name}: {text}");
            assert!(
                schema.get("required").is_none()
                    || schema["required"].as_array().unwrap().is_empty(),
                "{name}: nothing may be required: {schema}"
            );
            crate::server::schema::assert_client_safe_schema(&schema, name);
        }
    }
}
