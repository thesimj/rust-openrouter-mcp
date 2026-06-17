//! Per-endpoint-family `impl OpenRouterClient` blocks plus their co-located
//! tests. Each submodule adds an inherent-method block to the shared
//! [`OpenRouterClient`](super::OpenRouterClient); none export new items.

mod chat;
mod key;
mod models;
mod speech;
mod video;
