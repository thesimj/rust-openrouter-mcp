//! OpenRouter REST DTOs, split by endpoint family. Re-exported flat so callers
//! (and the parent `openrouter` module) keep referring to them as
//! `crate::openrouter::X` regardless of which submodule they live in.

mod chat;
mod embeddings;
mod images;
mod key;
mod models;
mod provider;
mod rerank;
mod speech;
mod video;

pub(crate) use chat::*;
pub(crate) use embeddings::*;
pub(crate) use images::*;
pub(crate) use key::*;
pub(crate) use models::*;
pub(crate) use provider::*;
pub(crate) use rerank::*;
pub(crate) use speech::*;
pub(crate) use video::*;

/// `deserialize_with` helper for a non-`Option` field a provider may send as
/// an explicit `null`: `#[serde(default)]` alone only covers a missing key,
/// while `Vec`/`BTreeMap` reject `null` outright. Maps `null` to `T::default()`.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + serde::Deserialize<'de>,
{
    Ok(<Option<T> as serde::Deserialize>::deserialize(deserializer)?.unwrap_or_default())
}
