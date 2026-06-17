//! OpenRouter REST DTOs, split by endpoint family. Re-exported flat so callers
//! (and the parent `openrouter` module) keep referring to them as
//! `crate::openrouter::X` regardless of which submodule they live in.

mod chat;
mod key;
mod models;
mod speech;
mod video;

pub(crate) use chat::*;
pub(crate) use key::*;
pub(crate) use models::*;
pub(crate) use speech::*;
pub(crate) use video::*;
