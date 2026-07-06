//! `POST /api/v1/chat/completions` endpoint (image generation / text / vision).

use anyhow::{Context, Result};

use crate::openrouter::{ChatCompletion, ChatRequest, ChatResponse, OpenRouterClient};

impl OpenRouterClient {
    /// `POST /api/v1/chat/completions` - used for text and vision (describe)
    /// calls. On a non-2xx status the upstream error body is surfaced verbatim
    /// (OpenRouter wraps provider errors there).
    pub async fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let resp = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(req)
            .send()
            .await
            .context("request to OpenRouter /chat/completions failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("OpenRouter /chat/completions returned {status}: {body}");
        }

        let completion: ChatCompletion = resp
            .json()
            .await
            .context("failed to decode OpenRouter /chat/completions response")?;
        Ok(ChatResponse { completion })
    }
}
