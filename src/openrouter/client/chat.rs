//! `POST /api/v1/chat/completions` endpoint (image generation / text / vision).

use anyhow::Result;

use crate::openrouter::{ChatCompletion, ChatRequest, OpenRouterClient};

impl OpenRouterClient {
    /// `POST /api/v1/chat/completions` - used for text and vision (describe)
    /// calls. On a non-2xx status the upstream error body is surfaced verbatim
    /// (OpenRouter wraps provider errors there).
    pub async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatCompletion> {
        let rb = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req);
        self.send_json(rb, "/chat/completions").await
    }
}
