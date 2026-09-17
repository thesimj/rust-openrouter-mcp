//! JSON-Schema normalization and tolerant scalar deserialization helpers shared
//! by every tool-argument struct, plus the shared required-parameter validator.

use rmcp::ErrorData;

/// Recursively rewrite the generated JSON Schema so optional parameters carry a
/// single scalar `"type"` (e.g. `"boolean"`) instead of the JSON-Schema 2020-12
/// nullable union schemars emits for `Option<T>` (e.g. `["boolean", "null"]`).
///
/// Several MCP clients (including some Claude connectors) mishandle union types:
/// rather than send a typed value they stringify it - `"true"` for a boolean,
/// `"10"` for an integer - which then fails strict server-side deserialization
/// (`invalid type: string "true", expected a boolean`). Collapsing to a scalar
/// type makes those clients emit the correctly-typed value. Optionality is still
/// expressed by the parent object's `required` list (these fields are absent from
/// it), so nothing is lost. The contradictory `"default": null` schemars attaches
/// to `Option<T>` is dropped at the same time.
///
/// Applied via `#[schemars(transform = scalarize_nullable)]` on every tool-argument
/// struct *and* on nested types (`ImageInput`): the transform recurses through a
/// type's own subschemas but not into sibling `$defs` entries, so each referenced
/// type must opt in directly. Multi-type unions with more than one non-null member
/// are left untouched.
pub(crate) fn scalarize_nullable(schema: &mut schemars::Schema) {
    use schemars::transform::transform_subschemas;
    if let Some(obj) = schema.as_object_mut()
        && let Some(serde_json::Value::Array(types)) = obj.get("type")
    {
        let non_null: Vec<serde_json::Value> = types
            .iter()
            .filter(|t| t.as_str() != Some("null"))
            .cloned()
            .collect();
        if non_null.len() == 1 {
            obj.insert("type".to_string(), non_null.into_iter().next().unwrap());
            if obj.get("default") == Some(&serde_json::Value::Null) {
                obj.remove("default");
            }
        }
    }
    transform_subschemas(&mut scalarize_nullable, schema);
}

/// Force the container's JSON-Schema `required` array to also list `self.0`,
/// on top of whatever schemars inferred from non-`Option` fields.
///
/// Fields the tool prose calls "REQUIRED (no default)" are kept `Option<T>` so
/// `require_all` can report a friendly per-field error instead of a raw schema
/// rejection - but that makes schemars mark them optional in the schema too, so
/// a schema-trusting client omits them and only fails at runtime. Applied via
/// `#[schemars(transform = RequireFields(&[...]))]` alongside `scalarize_nullable`.
pub(crate) struct RequireFields(pub(crate) &'static [&'static str]);

/// Catch a typo'd/renamed field name at test time, before it silently
/// no-ops (RequireFields) or invalidates every call (AtLeastOneOf).
fn assert_props(obj: &serde_json::Map<String, serde_json::Value>, names: &[&str], ctx: &str) {
    for name in names {
        debug_assert!(
            obj.get("properties").and_then(|p| p.get(*name)).is_some(),
            "{ctx}: {name:?} is not a property of this schema"
        );
    }
    let _ = (obj, names, ctx); // silence release-build unused warnings
}

impl schemars::transform::Transform for RequireFields {
    fn transform(&mut self, schema: &mut schemars::Schema) {
        if let Some(obj) = schema.as_object_mut() {
            assert_props(obj, self.0, "RequireFields");
            let required = obj
                .entry("required")
                .or_insert_with(|| serde_json::Value::Array(Vec::new()));
            if let serde_json::Value::Array(arr) = required {
                for name in self.0 {
                    let v = serde_json::Value::String((*name).to_string());
                    if !arr.contains(&v) {
                        arr.push(v);
                    }
                }
            }
        }
    }
}

/// Advertise "at least one of these fields" as an `anyOf` of object-typed
/// single-`required` branches. NOT `oneOf` ("" counts as present, rejecting
/// placeholder-filling clients the runtime accepts), and NEVER on a tool-args
/// root: real clients reject a root anyOf as a union (nested types only).
pub(crate) struct AtLeastOneOf(pub(crate) &'static [&'static str]);

impl schemars::transform::Transform for AtLeastOneOf {
    fn transform(&mut self, schema: &mut schemars::Schema) {
        if let Some(obj) = schema.as_object_mut() {
            assert_props(obj, self.0, "AtLeastOneOf");
            let branches: Vec<serde_json::Value> = self
                .0
                .iter()
                .map(|name| serde_json::json!({ "type": "object", "required": [name] }))
                .collect();
            obj.insert("anyOf".to_string(), serde_json::Value::Array(branches));
        }
    }
}

/// Coerce a JSON value that is either a real boolean or a stringified one
/// (`"true"`/`"false"`, case- and whitespace-insensitive) into a `bool`. This is
/// the deserialization-side counterpart to [`scalarize_nullable`]: it absorbs the
/// residual stringification from clients that mistype tool arguments even when the
/// schema advertises a scalar type.
fn coerce_bool<E: serde::de::Error>(v: &serde_json::Value) -> Result<bool, E> {
    match v {
        serde_json::Value::Bool(b) => Ok(*b),
        serde_json::Value::String(s) => match s.trim().to_ascii_lowercase().as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(E::custom(format!(
                "expected a boolean or \"true\"/\"false\", got string {other:?}"
            ))),
        },
        other => Err(E::custom(format!("expected a boolean, got {other}"))),
    }
}

/// Deserialize a required `bool`, tolerating stringified booleans.
pub(crate) fn de_bool<'de, D>(d: D) -> Result<bool, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    coerce_bool(&serde_json::Value::deserialize(d)?)
}

/// Deserialize an optional `bool`, tolerating stringified booleans; `null` -> None.
pub(crate) fn de_opt_bool<'de, D>(d: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    match Option::<serde_json::Value>::deserialize(d)? {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => coerce_bool(&v).map(Some),
    }
}

/// Deserialize an optional unsigned integer, tolerating stringified numbers
/// (`"10"`); `null` -> None. Generic over the unsigned target type.
pub(crate) fn de_opt_uint<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: TryFrom<u64>,
    <T as TryFrom<u64>>::Error: std::fmt::Display,
{
    use serde::Deserialize as _;
    use serde::de::Error as _;
    let n: u64 = match Option::<serde_json::Value>::deserialize(d)? {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::Number(num)) => num.as_u64().ok_or_else(|| {
            D::Error::custom(format!("expected a non-negative integer, got {num}"))
        })?,
        Some(serde_json::Value::String(s)) => s
            .trim()
            .parse()
            .map_err(|_| D::Error::custom(format!("expected an integer, got string {s:?}")))?,
        Some(other) => {
            return Err(D::Error::custom(format!(
                "expected an integer, got {other}"
            )));
        }
    };
    T::try_from(n)
        .map(Some)
        .map_err(|e| D::Error::custom(format!("integer {n} out of range: {e}")))
}

/// Deserialize an optional float, tolerating stringified numbers (`"1.5"`);
/// `null` -> None.
pub(crate) fn de_opt_f64<'de, D>(d: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    use serde::de::Error as _;
    match Option::<serde_json::Value>::deserialize(d)? {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(num)) => num
            .as_f64()
            .map(Some)
            .ok_or_else(|| D::Error::custom(format!("expected a number, got {num}"))),
        Some(serde_json::Value::String(s)) => s
            .trim()
            .parse()
            .map(Some)
            .map_err(|_| D::Error::custom(format!("expected a number, got string {s:?}"))),
        Some(other) => Err(D::Error::custom(format!("expected a number, got {other}"))),
    }
}

/// Deserialize an object-valued argument leniently: accept the value itself,
/// a JSON *string* that parses to it (clients that stringify nested objects,
/// the same failure mode [`de_bool`] absorbs for scalars), or `null` / a blank
/// string, both of which yield `T::default()`. Pair with `#[serde(default)]` so
/// an absent field also defaults.
pub(crate) fn de_lenient<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::de::DeserializeOwned + Default,
{
    use serde::Deserialize as _;
    use serde::de::Error as _;
    match serde_json::Value::deserialize(d)? {
        serde_json::Value::Null => Ok(T::default()),
        serde_json::Value::String(s) if s.trim().is_empty() => Ok(T::default()),
        serde_json::Value::String(s) => serde_json::from_str(&s)
            .map_err(|e| D::Error::custom(format!("invalid JSON in string argument: {e}"))),
        other => serde_json::from_value(other).map_err(D::Error::custom),
    }
}

/// Shared "no defaults" validator: if any required-but-absent parameters were
/// collected in `missing`, fail with the standard message naming them and the
/// modality to pass to `list_models`. Returns `Ok(())` when nothing is missing.
pub(crate) fn require_all(tool: &str, modality: &str, missing: &[&str]) -> Result<(), ErrorData> {
    if missing.is_empty() {
        return Ok(());
    }
    Err(ErrorData::invalid_params(
        format!(
            "{tool} has no defaults - specify every parameter explicitly. Missing: {}. \
             Use list_models with output_modalities=\"{modality}\" to choose a model.",
            missing.join("; ")
        ),
        None,
    ))
}

/// The generated tool-args schema of `T` as a plain JSON value (rmcp's
/// `schema_for_type` hands back a shared map).
#[cfg(test)]
pub(crate) fn schema_json<T: schemars::JsonSchema + std::any::Any>() -> serde_json::Value {
    let schema = rmcp::handler::server::common::schema_for_type::<T>();
    serde_json::Value::Object((*schema).clone())
}

/// Walk a generated tool-args schema (root and every `$defs` entry) and
/// panic on the shapes that break real MCP clients: a nullable union
/// (`anyOf`/`oneOf` with a `{"type":"null"}` branch, or a `type` array
/// containing `"null"` - the trap `Option<Struct>` falls into, which
/// `scalarize_nullable` cannot fix) and an untyped property schema (the bare
/// boolean `true`, or `{"default": null}` - both what
/// `Option<serde_json::Value>` emits). A tool-args *root* may carry no
/// `anyOf`/`oneOf` at all. Later phases only add their struct to
/// `tests::every_tool_args_schema_is_client_safe`.
#[cfg(test)]
pub(crate) fn assert_client_safe_schema(root: &serde_json::Value, name: &str) {
    assert!(
        root.get("anyOf").is_none() && root.get("oneOf").is_none(),
        "{name}: tool-args root carries a union"
    );
    walk(root, name);
    for key in ["$defs", "definitions"] {
        if let Some(defs) = root.get(key).and_then(|d| d.as_object()) {
            for (def_name, def) in defs {
                walk(def, &format!("{name}.{key}.{def_name}"));
            }
        }
    }

    fn is_null_branch(v: &serde_json::Value) -> bool {
        v.get("type").and_then(|t| t.as_str()) == Some("null")
    }

    fn walk(node: &serde_json::Value, path: &str) {
        let Some(obj) = node.as_object() else { return };
        for key in ["anyOf", "oneOf"] {
            if let Some(branches) = obj.get(key).and_then(|b| b.as_array()) {
                assert!(
                    !branches.iter().any(is_null_branch),
                    "{path}: {key} has a null branch: {node}"
                );
            }
        }
        if let Some(types) = obj.get("type").and_then(|t| t.as_array()) {
            assert!(
                !types.iter().any(|t| t.as_str() == Some("null")),
                "{path}: nullable type union {types:?}"
            );
        }
        if let Some(props) = obj.get("properties").and_then(|p| p.as_object()) {
            for (prop, schema) in props {
                assert!(
                    !schema.is_boolean(),
                    "{path}.{prop}: bare boolean property schema"
                );
                // `Option<Value>` + `serde(default)` emits `{"default": null}`:
                // the same accept-anything schema as `true`, spelled as an
                // object. Every property must say what it is.
                let typed = ["type", "$ref", "anyOf", "oneOf", "allOf", "enum", "const"]
                    .iter()
                    .any(|k| schema.get(k).is_some());
                assert!(typed, "{path}.{prop}: untyped property schema {schema}");
                walk(schema, &format!("{path}.{prop}"));
            }
        }
        for key in ["items", "additionalProperties", "not"] {
            if let Some(sub) = obj.get(key) {
                walk(sub, &format!("{path}.{key}"));
            }
        }
        for key in ["anyOf", "oneOf", "allOf", "prefixItems"] {
            if let Some(branches) = obj.get(key).and_then(|b| b.as_array()) {
                for (i, b) in branches.iter().enumerate() {
                    walk(b, &format!("{path}.{key}[{i}]"));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::server::audio::GenerateAudioArgs;
    use crate::server::chat::ChatCompletionArgs;
    use crate::server::image::{DescribeImageArgs, GenerateImageArgs, ImageInput};
    use crate::server::models::ListModelsArgs;
    use crate::server::music::GenerateMusicArgs;
    use crate::server::video::GenerateVideoArgs;
    use rmcp::handler::server::common::schema_for_type;
    use schemars::JsonSchema;
    use serde_json::json;

    /// The permanent guard: every tool-arg struct, root and `$defs`, is free of
    /// nullable unions and bare `true` schemas. Add new arg structs here.
    #[test]
    fn every_tool_args_schema_is_client_safe() {
        use super::schema_json;
        use crate::server::account::{GetResultArgs, ResetUsageStatsArgs};
        use crate::server::audio::TranscribeAudioArgs;
        use crate::server::embeddings::{EmbedTextArgs, GetGenerationArgs, RerankDocumentsArgs};
        use crate::server::models::DescribeModelArgs;
        use crate::server::provider::{
            ImageProviderArgs, ProviderOptionsArgs, ProviderRoutingArgs,
        };
        let schemas: Vec<(&str, serde_json::Value)> = vec![
            ("EmbedTextArgs", schema_json::<EmbedTextArgs>()),
            ("RerankDocumentsArgs", schema_json::<RerankDocumentsArgs>()),
            ("GetGenerationArgs", schema_json::<GetGenerationArgs>()),
            ("ChatCompletionArgs", schema_json::<ChatCompletionArgs>()),
            ("DescribeImageArgs", schema_json::<DescribeImageArgs>()),
            ("GenerateImageArgs", schema_json::<GenerateImageArgs>()),
            ("GenerateVideoArgs", schema_json::<GenerateVideoArgs>()),
            ("GenerateAudioArgs", schema_json::<GenerateAudioArgs>()),
            ("TranscribeAudioArgs", schema_json::<TranscribeAudioArgs>()),
            ("GenerateMusicArgs", schema_json::<GenerateMusicArgs>()),
            ("ListModelsArgs", schema_json::<ListModelsArgs>()),
            ("DescribeModelArgs", schema_json::<DescribeModelArgs>()),
            ("GetResultArgs", schema_json::<GetResultArgs>()),
            ("ResetUsageStatsArgs", schema_json::<ResetUsageStatsArgs>()),
            ("ProviderRoutingArgs", schema_json::<ProviderRoutingArgs>()),
            ("ImageProviderArgs", schema_json::<ImageProviderArgs>()),
            ("ProviderOptionsArgs", schema_json::<ProviderOptionsArgs>()),
        ];
        for (name, schema) in &schemas {
            super::assert_client_safe_schema(schema, name);
        }
    }

    /// The retrieval tools' required arrays advertise `minItems: 1` (the prose
    /// says "at least one"), their genuinely required scalars are in
    /// `required`, and the routing block is an optional `$ref` into `$defs`.
    #[test]
    fn retrieval_tool_schemas_require_texts_and_carry_optional_routing() {
        use super::schema_json;
        use crate::server::embeddings::{EmbedTextArgs, GetGenerationArgs, RerankDocumentsArgs};

        let embed = schema_json::<EmbedTextArgs>();
        assert_eq!(embed["properties"]["input"]["minItems"], json!(1));
        assert_eq!(embed["properties"]["dimensions"]["type"], json!("integer"));
        let required = required_fields::<EmbedTextArgs>();
        assert!(required.contains(&"model".to_string()), "{required:?}");
        assert!(required.contains(&"input".to_string()), "{required:?}");
        assert!(!required.contains(&"provider".to_string()), "{required:?}");
        assert_eq!(
            embed["properties"]["provider"]["$ref"],
            json!("#/$defs/ProviderRoutingArgs")
        );
        assert!(embed["$defs"]["ProviderRoutingArgs"].is_object());

        let rerank = schema_json::<RerankDocumentsArgs>();
        assert_eq!(rerank["properties"]["documents"]["minItems"], json!(1));
        assert_eq!(rerank["properties"]["top_n"]["type"], json!("integer"));
        let required = required_fields::<RerankDocumentsArgs>();
        for name in ["model", "query", "documents"] {
            assert!(required.contains(&name.to_string()), "{required:?}");
        }
        assert_eq!(
            rerank["properties"]["provider"]["$ref"],
            json!("#/$defs/ProviderRoutingArgs")
        );

        let required = required_fields::<GetGenerationArgs>();
        assert_eq!(required, vec!["generation_id".to_string()]);
    }

    /// The lint must actually catch the two traps it exists for, or it guards
    /// nothing: `Option<Struct>` (nullable anyOf) and `Option<Value>` (bare true).
    #[test]
    fn client_safe_lint_rejects_option_struct_and_option_value() {
        use serde::Deserialize;
        #[derive(Deserialize, JsonSchema, Default)]
        struct Inner {
            #[allow(dead_code)]
            #[serde(default)]
            k: Option<String>,
        }
        #[derive(Deserialize, JsonSchema)]
        #[schemars(transform = super::scalarize_nullable)]
        struct BadStruct {
            #[allow(dead_code)]
            #[serde(default)]
            inner: Option<Inner>,
        }
        #[derive(Deserialize, JsonSchema)]
        #[schemars(transform = super::scalarize_nullable)]
        struct BadValue {
            #[allow(dead_code)]
            #[serde(default)]
            v: Option<serde_json::Value>,
        }
        let bad_struct = super::schema_json::<BadStruct>();
        let r =
            std::panic::catch_unwind(|| super::assert_client_safe_schema(&bad_struct, "BadStruct"));
        assert!(r.is_err(), "Option<Struct> must be rejected: {bad_struct}");
        let bad_value = super::schema_json::<BadValue>();
        let r =
            std::panic::catch_unwind(|| super::assert_client_safe_schema(&bad_value, "BadValue"));
        assert!(r.is_err(), "Option<Value> must be rejected: {bad_value}");
    }

    /// `de_lenient` accepts the object itself, a JSON string holding it, and
    /// `null`/blank (-> default); an absent field defaults via `serde(default)`.
    #[test]
    fn de_lenient_accepts_object_string_and_null() {
        use std::collections::BTreeMap;
        #[derive(serde::Deserialize, Debug)]
        struct Holder {
            #[serde(default, deserialize_with = "super::de_lenient")]
            opts: BTreeMap<String, serde_json::Value>,
        }
        let direct: Holder =
            serde_json::from_value(json!({"opts": {"deepgram": {"diarize": true}}})).unwrap();
        assert_eq!(direct.opts["deepgram"], json!({"diarize": true}));

        let stringified: Holder =
            serde_json::from_value(json!({"opts": "{\"deepgram\":{\"diarize\":true}}"})).unwrap();
        assert_eq!(stringified.opts["deepgram"], json!({"diarize": true}));

        let null: Holder = serde_json::from_value(json!({"opts": null})).unwrap();
        assert!(null.opts.is_empty());
        let blank: Holder = serde_json::from_value(json!({"opts": "  "})).unwrap();
        assert!(blank.opts.is_empty());
        let absent: Holder = serde_json::from_value(json!({})).unwrap();
        assert!(absent.opts.is_empty());

        // Garbage in the string, or the wrong shape, is still an error.
        assert!(serde_json::from_value::<Holder>(json!({"opts": "{not json"})).is_err());
        assert!(serde_json::from_value::<Holder>(json!({"opts": [1, 2]})).is_err());
    }

    /// Fetch the JSON Schema `type` for a property of a tool-argument struct.
    fn prop_type<T: JsonSchema + std::any::Any>(prop: &str) -> serde_json::Value {
        let schema = schema_for_type::<T>();
        schema
            .get("properties")
            .and_then(|p| p.get(prop))
            .and_then(|p| p.get("type"))
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    }

    /// Fetch a property's raw schema object for a tool-argument struct.
    fn prop<T: JsonSchema + std::any::Any>(name: &str) -> serde_json::Value {
        schema_for_type::<T>()
            .get("properties")
            .and_then(|p| p.get(name))
            .cloned()
            .unwrap_or(serde_json::Value::Null)
    }

    /// F13: a `RequireFields` name that doesn't match any property (a typo, or
    /// a rename that forgot to update the transform) must fail loudly in a
    /// debug build rather than silently no-op in the generated schema.
    /// `debug_assert!` compiles out under `--release`, so this test would
    /// simply not panic there (N1) - gate it on the same cfg the assert itself
    /// depends on rather than failing spuriously in a release run.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is not a property of this schema")]
    fn require_fields_catches_an_unknown_property_name() {
        use serde::Deserialize;
        #[derive(Deserialize, JsonSchema)]
        #[schemars(transform = super::RequireFields(&["does_not_exist"]))]
        struct Bogus {
            #[allow(dead_code)]
            #[serde(default)]
            real_field: Option<String>,
        }
        schema_for_type::<Bogus>();
    }

    /// The `required` array of a tool-argument struct's schema.
    fn required_fields<T: JsonSchema + std::any::Any>() -> Vec<String> {
        schema_for_type::<T>()
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The tools/list schema for each "REQUIRED (no default)" field must actually
    /// list it in `required`, not just say so in prose - otherwise a
    /// schema-trusting client omits it and only fails at runtime (S1).
    #[test]
    fn required_no_default_fields_are_in_the_schema_required_array() {
        let chat = required_fields::<ChatCompletionArgs>();
        assert!(chat.contains(&"prompt".to_string()), "{chat:?}");

        let audio = required_fields::<GenerateAudioArgs>();
        assert!(audio.contains(&"input".to_string()), "{audio:?}");
        assert!(audio.contains(&"voice".to_string()), "{audio:?}");

        let music = required_fields::<GenerateMusicArgs>();
        assert!(music.contains(&"prompt".to_string()), "{music:?}");

        let image = required_fields::<GenerateImageArgs>();
        assert!(image.contains(&"aspect_ratio".to_string()), "{image:?}");
        assert!(image.contains(&"image_size".to_string()), "{image:?}");

        let video = required_fields::<GenerateVideoArgs>();
        assert!(video.contains(&"duration".to_string()), "{video:?}");
        assert!(video.contains(&"with_audio".to_string()), "{video:?}");
        // aspect_ratio is conditional (only required for text-to-video without a
        // frame), so it must stay OUT of the unconditional schema required list.
        assert!(!video.contains(&"aspect_ratio".to_string()), "{video:?}");
    }

    /// `describe_image.images` must declare `minItems: 1` - the prose already
    /// says at least one image is required (S13).
    #[test]
    fn describe_image_images_has_min_items_one() {
        let images = prop::<DescribeImageArgs>("images");
        assert_eq!(images["minItems"], json!(1), "got: {images}");
    }

    /// `max_image_dimension` must cap at 4096 in the schema, matching the prose
    /// cap, on every tool that carries it (S14/F6): chat_completion,
    /// generate_image, describe_image, and generate_video.
    #[test]
    fn max_image_dimension_caps_at_4096_in_schema() {
        for max in [
            prop::<ChatCompletionArgs>("max_image_dimension")["maximum"].clone(),
            prop::<GenerateImageArgs>("max_image_dimension")["maximum"].clone(),
            prop::<DescribeImageArgs>("max_image_dimension")["maximum"].clone(),
            prop::<GenerateVideoArgs>("max_image_dimension")["maximum"].clone(),
        ] {
            assert_eq!(max, json!(4096));
        }
    }

    /// schemars renders `Option<bool>` as the union `["boolean","null"]`, which
    /// some MCP clients stringify to `"true"`. The `scalarize_nullable` transform
    /// must collapse every optional param to a single scalar `type` across all
    /// tool-argument structs (and nested types).
    #[test]
    fn optional_params_use_scalar_types_not_nullable_unions() {
        assert_eq!(prop_type::<GenerateImageArgs>("seed"), json!("integer"));
        assert_eq!(
            prop_type::<GenerateImageArgs>("image_size"),
            json!("string")
        );
        assert_eq!(prop_type::<GenerateImageArgs>("variants"), json!("integer"));
        assert_eq!(prop_type::<ListModelsArgs>("min_context"), json!("integer"));
        assert_eq!(
            prop_type::<DescribeImageArgs>("max_image_dimension"),
            json!("integer")
        );
        assert_eq!(
            prop_type::<GenerateVideoArgs>("with_audio"),
            json!("boolean")
        );
        assert_eq!(prop_type::<GenerateVideoArgs>("duration"), json!("integer"));
        assert_eq!(prop_type::<GenerateAudioArgs>("speed"), json!("number"));
        assert_eq!(prop_type::<GenerateMusicArgs>("seed"), json!("integer"));
        // Nested $defs type must opt in too, or its optional fields keep the union.
        assert_eq!(prop_type::<ImageInput>("label"), json!("string"));
    }

    /// The contradictory `"default": null` schemars attaches to `Option<T>` is
    /// dropped once the type is collapsed to a scalar.
    #[test]
    fn collapsed_optionals_drop_null_default() {
        let schema = schema_for_type::<GenerateImageArgs>();
        let seed = schema
            .get("properties")
            .and_then(|p| p.get("seed"))
            .unwrap();
        assert!(
            seed.get("default").is_none(),
            "expected no `default` on seed, got {seed}"
        );
    }

    /// S16: "needs one of these sources" is schema-encoded as anyOf
    /// object-typed single-required branches on the NESTED ImageInput only.
    /// Tool-args roots must stay plain objects: a real client rejected a root
    /// anyOf as "a union with a non-object branch", so TranscribeAudioArgs
    /// carries no anyOf and relies on its runtime exactly-one check.
    #[test]
    fn at_least_one_of_is_encoded_as_anyof_required_branches() {
        let img = schema_for_type::<ImageInput>();
        assert_eq!(
            img.get("anyOf").cloned(),
            Some(json!([
                { "type": "object", "required": ["path"] },
                { "type": "object", "required": ["url"] },
                { "type": "object", "required": ["base64"] }
            ]))
        );
        let tr = schema_for_type::<crate::server::audio::TranscribeAudioArgs>();
        assert!(tr.get("anyOf").is_none(), "no union at a tool-args root");
        assert!(tr.get("oneOf").is_none());
    }

    /// A typo'd field name in AtLeastOneOf must panic in debug builds instead
    /// of emitting a branch nothing can satisfy (same guard as RequireFields).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "AtLeastOneOf")]
    fn at_least_one_of_catches_an_unknown_property_name() {
        #[derive(serde::Deserialize, JsonSchema)]
        #[schemars(transform = crate::server::schema::AtLeastOneOf(&["nope"]))]
        struct Bogus {
            #[allow(dead_code)]
            real: Option<String>,
        }
        schema_for_type::<Bogus>();
    }
}
