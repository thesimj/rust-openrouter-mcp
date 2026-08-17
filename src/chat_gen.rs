//! Shared chat-completion (text/vision in -> text out): the single place the
//! `/chat/completions` request envelope is built and its response extracted.
//! Used by the `chat_completion` MCP tool, the `chat` CLI subcommand, and
//! [`crate::image_gen::describe_image`], which delegates here.

use anyhow::{Context, Result};

use crate::image_gen::{self, InputImage};
use crate::openrouter::{
    ChatRequest, Content, ContentPart, ImageUrl, Message, OpenRouterClient, Reasoning,
};

/// A chat reply: the assistant text plus the reported USD cost (when present).
pub struct ChatResult {
    pub text: String,
    pub cost: Option<f64>,
}

/// Everything needed to issue one chat completion. `images` empty => a plain
/// text-in / text-out call; non-empty => a multimodal user message where each
/// image is normalized to a PNG data URL capped at `max_image_dimension` (which
/// is unused — and may be any value — when `images` is empty). The caller is
/// responsible for having verified the model accepts image input. `prompt` is
/// assumed already validated as non-empty.
pub struct ChatInputs<'a> {
    pub model: &'a str,
    pub system: Option<&'a str>,
    pub prompt: &'a str,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub images: &'a [InputImage],
    pub max_image_dimension: u32,
    /// Reasoning effort (max, xhigh, high, medium, low, minimal, none). `None`
    /// sends no `reasoning` object, so the model keeps its catalog default.
    pub reasoning_effort: Option<&'a str>,
}

/// Build a chat request (optional system message, then the user message) and
/// return the model's reply text and cost. Errors if the model returns no
/// choices or empty content.
pub async fn complete(client: &OpenRouterClient, inputs: &ChatInputs<'_>) -> Result<ChatResult> {
    let mut messages = Vec::new();
    if let Some(system) = inputs.system.map(str::trim).filter(|s| !s.is_empty()) {
        messages.push(Message {
            role: "system".to_string(),
            content: Content::Text(system.to_string()),
        });
    }
    let user_content = if inputs.images.is_empty() {
        Content::Text(inputs.prompt.to_string())
    } else {
        // Multimodal user message: `prompt` verbatim, then each normalized input
        // image. Callers wanting the labeled-reference preamble (image_gen's
        // "Reference images:" block) apply `assemble_prompt` themselves - it does
        // not belong in a plain Q&A chat.
        let prepared =
            image_gen::prepare_inputs_async(inputs.images, inputs.max_image_dimension).await?;
        let mut parts = vec![ContentPart::Text {
            text: inputs.prompt.to_string(),
        }];
        // `into_iter`: each data_url is a base64 PNG (hundreds of KB), so move it
        // into the request rather than cloning every input image.
        for input in prepared {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: input.data_url,
                },
            });
        }
        Content::Parts(parts)
    };
    messages.push(Message {
        role: "user".to_string(),
        content: user_content,
    });

    let req = ChatRequest {
        model: inputs.model.to_string(),
        messages,
        modalities: None,
        image_config: None,
        seed: None,
        temperature: inputs.temperature,
        max_tokens: inputs.max_tokens,
        reasoning: inputs
            .reasoning_effort
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|effort| Reasoning {
                effort: effort.to_string(),
            }),
        stream: false,
    };

    let completion = client.chat_completion(&req).await?;
    let cost = completion.usage.and_then(|u| u.cost);
    let choice = completion
        .choices
        .into_iter()
        .next()
        .context("OpenRouter returned no choices")?;
    let text = choice
        .message
        .content
        .filter(|t| !t.is_empty())
        .with_context(|| {
            if inputs.images.is_empty() {
                "model returned no text".to_string()
            } else {
                "model returned no text (it may be an image-output-only model; \
                 use one with text output)"
                    .to_string()
            }
        })?;
    Ok(ChatResult { text, cost })
}
