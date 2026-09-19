//! Per-endpoint-family `impl OpenRouterClient` blocks plus their co-located
//! tests. Each submodule adds an inherent-method block to the shared
//! [`OpenRouterClient`](super::OpenRouterClient); none export new items.

mod chat;
mod decisions;
mod embeddings;
mod generation;
mod images;
mod key;
mod models;
mod rerank;
mod speech;
mod video;
