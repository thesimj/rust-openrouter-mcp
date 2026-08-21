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

/// Advertise "at least one of these fields" as an `anyOf` of single-`required`
/// branches. Deliberately NOT `oneOf`: JSON Schema counts `""` as present, so
/// `oneOf` would reject placeholder-filling clients the runtime accepts - the
/// runtime "exactly one of" check owns strictness and the friendly error.
pub(crate) struct AtLeastOneOf(pub(crate) &'static [&'static str]);

impl schemars::transform::Transform for AtLeastOneOf {
    fn transform(&mut self, schema: &mut schemars::Schema) {
        if let Some(obj) = schema.as_object_mut() {
            assert_props(obj, self.0, "AtLeastOneOf");
            let branches: Vec<serde_json::Value> = self
                .0
                .iter()
                .map(|name| serde_json::json!({ "required": [name] }))
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

#[cfg(test)]
mod tests {
    use crate::server::audio::GenerateAudioArgs;
    use crate::server::chat::ChatCompletionArgs;
    use crate::server::image::{DescribeImageArgs, GenerateImageArgs, ImageInput};
    use crate::server::models::ListModelsArgs;
    use crate::server::video::GenerateVideoArgs;
    use rmcp::handler::server::common::schema_for_type;
    use schemars::JsonSchema;
    use serde_json::json;

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
    /// single-required branches on ImageInput (path/url/base64) and transcribe
    /// (path/base64). anyOf, not oneOf: "" counts as present in JSON Schema,
    /// so oneOf would reject placeholder-filling clients the runtime accepts.
    #[test]
    fn at_least_one_of_is_encoded_as_anyof_required_branches() {
        let img = schema_for_type::<ImageInput>();
        assert_eq!(
            img.get("anyOf").cloned(),
            Some(json!([
                { "required": ["path"] },
                { "required": ["url"] },
                { "required": ["base64"] }
            ]))
        );
        let tr = schema_for_type::<crate::server::audio::TranscribeAudioArgs>();
        assert_eq!(
            tr.get("anyOf").cloned(),
            Some(json!([
                { "required": ["path"] },
                { "required": ["base64"] }
            ]))
        );
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
