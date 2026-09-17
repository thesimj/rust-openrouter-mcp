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
