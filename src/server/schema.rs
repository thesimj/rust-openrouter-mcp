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
    if let Some(obj) = schema.as_object_mut() {
        if let Some(serde_json::Value::Array(types)) = obj.get("type") {
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
    }
    transform_subschemas(&mut scalarize_nullable, schema);
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
             (model, prompt and output are also required.) Use list_models with \
             output_modalities=\"{modality}\" to choose a model.",
            missing.join("; ")
        ),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use crate::server::audio::GenerateAudioArgs;
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

    /// schemars renders `Option<bool>` as the union `["boolean","null"]`, which
    /// some MCP clients stringify to `"true"`. The `scalarize_nullable` transform
    /// must collapse every optional param to a single scalar `type` across all
    /// tool-argument structs (and nested types).
    #[test]
    fn optional_params_use_scalar_types_not_nullable_unions() {
        assert_eq!(
            prop_type::<GenerateImageArgs>("image_only"),
            json!("boolean")
        );
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
            prop_type::<GenerateVideoArgs>("generate_audio"),
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
        let image_only = schema
            .get("properties")
            .and_then(|p| p.get("image_only"))
            .unwrap();
        assert!(
            image_only.get("default").is_none(),
            "expected no `default` on image_only, got {image_only}"
        );
    }
}
