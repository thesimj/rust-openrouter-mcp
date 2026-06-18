//! Process-local cache of model input-modality capabilities.
//!
//! `chat_completion` consults this before sending image inputs so it can reject
//! a model that does not accept images early. After the first lookup for a model
//! id completes, its modalities are cached for the lifetime of the (single-
//! session) server process; the cache is not single-flight, so a burst of
//! concurrent first-time calls for the same id may each issue one `/models`
//! lookup (harmless — they resolve to the same value).

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

/// Cheaply cloneable cache mapping a model id to its declared input modalities
/// (e.g. `["text", "image"]`).
#[derive(Clone, Default)]
pub(crate) struct ModelCapsCache {
    inner: Arc<Mutex<HashMap<String, Vec<String>>>>,
}

impl ModelCapsCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// The cached input modalities for `model`, if it was looked up before.
    pub(crate) async fn get(&self, model: &str) -> Option<Vec<String>> {
        self.inner.lock().await.get(model).cloned()
    }

    /// Record `modalities` for `model`.
    pub(crate) async fn put(&self, model: &str, modalities: Vec<String>) {
        self.inner
            .lock()
            .await
            .insert(model.to_string(), modalities);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn get_returns_none_until_put_then_the_stored_value() {
        let cache = ModelCapsCache::new();
        assert_eq!(cache.get("a/b").await, None);
        cache
            .put("a/b", vec!["text".to_string(), "image".to_string()])
            .await;
        assert_eq!(
            cache.get("a/b").await,
            Some(vec!["text".to_string(), "image".to_string()])
        );
        // Unrelated keys remain absent.
        assert_eq!(cache.get("c/d").await, None);
    }
}
