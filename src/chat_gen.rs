//! Shared chat-completion (text in -> text out) used by both the
//! `chat_completion` MCP tool and the `chat` CLI subcommand, so the request
//! envelope and response extraction live in one place (mirrors
//! [`crate::image_gen::describe_image`]).

use anyhow::{Context, Result};

use crate::openrouter::{ChatRequest, Content, Message, OpenRouterClient};

/// A chat reply: the assistant text plus the reported USD cost (when present).
pub struct ChatResult {
    pub text: String,
    pub cost: Option<f64>,
}

/// Build a text-only chat request (optional system message, then the user
/// prompt) and return the model's reply text and cost. Errors if the model
/// returns no choices or empty content. `prompt` is assumed already validated
/// as non-empty by the caller.
pub async fn complete(
    client: &OpenRouterClient,
    model: &str,
    system: Option<&str>,
    prompt: &str,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
) -> Result<ChatResult> {
    let mut messages = Vec::new();
    if let Some(system) = system.map(str::trim).filter(|s| !s.is_empty()) {
        messages.push(Message {
            role: "system".to_string(),
            content: Content::Text(system.to_string()),
        });
    }
    messages.push(Message {
        role: "user".to_string(),
        content: Content::Text(prompt.to_string()),
    });

    let req = ChatRequest {
        model: model.to_string(),
        messages,
        modalities: None,
        image_config: None,
        seed: None,
        temperature,
        max_tokens,
        stream: false,
    };

    let resp = client.chat_completion(&req).await?;
    let cost = resp.completion.usage.and_then(|u| u.cost);
    let choice = resp
        .completion
        .choices
        .into_iter()
        .next()
        .context("OpenRouter returned no choices")?;
    let text = choice
        .message
        .content
        .filter(|t| !t.is_empty())
        .context("model returned no text")?;
    Ok(ChatResult { text, cost })
}
