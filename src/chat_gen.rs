//! Shared chat-completion (text/vision in -> text out): the single place the
//! `/chat/completions` request envelope is built and its response extracted.
//! Used by the `chat_completion` MCP tool and
//! [`crate::image_gen::describe_image`], which delegates here.

use anyhow::{Context, Result};

use crate::image_gen::{self, InputImage};
use crate::openrouter::{
    ChatRequest, Choice, Content, ContentPart, FilePart, ImageUrl, InputAudio, Message,
    OpenRouterClient, Plugin, ProviderRouting, Reasoning, ResponseFormat, VideoUrl,
    WebSearchOptions,
};

/// A chat reply: the assistant text plus what came with it - the reported USD
/// cost, the generation id (for `get_generation`), the reasoning text (when
/// the model exposes it), the raw response `annotations` (web-search
/// citations, parsed-file records), the `finish_reason`, and the token counts.
#[derive(Debug)]
pub struct ChatResult {
    pub text: String,
    pub cost: Option<f64>,
    /// OpenRouter's generation id (`gen-...`): the `X-Generation-Id` header,
    /// else the body `id`. Absent only when the upstream sent neither.
    pub generation_id: Option<String>,
    pub reasoning: Option<String>,
    pub annotations: Vec<serde_json::Value>,
    pub finish_reason: Option<String>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

/// Everything needed to issue one chat completion. `images` empty => a plain
/// text-in / text-out call; non-empty => a multimodal user message where each
/// image is normalized to a PNG data URL capped at `max_image_dimension` (which
/// is unused — and may be any value — when `images` is empty). The caller is
/// responsible for having verified the model accepts image input. `prompt` is
/// assumed already validated as non-empty. Every optional control is passed
/// through as given (blank strings count as unset); contradictions between
/// them (`effort` + `max_tokens`) are the caller's to reject.
#[derive(Default)]
pub struct ChatInputs<'a> {
    pub model: &'a str,
    pub system: Option<&'a str>,
    pub prompt: &'a str,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub images: &'a [InputImage],
    pub max_image_dimension: u32,
    /// Documents, already resolved to `file` parts (data URLs).
    pub files: &'a [FilePart],
    /// Audio clips, already resolved to `input_audio` parts.
    pub audio: &'a [InputAudio],
    /// Videos, already resolved to `video_url` parts (URL or data URL).
    pub videos: &'a [VideoUrl],
    /// Reasoning effort (max, xhigh, high, medium, low, minimal, none). With
    /// `reasoning_max_tokens` and `reasoning_exclude` all unset, no `reasoning`
    /// object is sent, so the model keeps its catalog default.
    pub reasoning_effort: Option<&'a str>,
    pub reasoning_max_tokens: Option<u64>,
    pub reasoning_exclude: Option<bool>,
    pub seed: Option<u64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub stop: &'a [String],
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub verbosity: Option<&'a str>,
    pub response_format: Option<ResponseFormat>,
    pub plugins: Vec<Plugin>,
    pub web_search_options: Option<WebSearchOptions>,
    /// Provider routing, already validated (chat takes routing fields only).
    pub provider: Option<ProviderRouting>,
}

/// Trim and drop a blank optional string.
fn non_blank(s: Option<&str>) -> Option<String> {
    s.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// The `reasoning` block, or `None` when nothing about reasoning was asked for.
fn reasoning(inputs: &ChatInputs<'_>) -> Option<Reasoning> {
    let block = Reasoning {
        effort: non_blank(inputs.reasoning_effort),
        max_tokens: inputs.reasoning_max_tokens,
        exclude: inputs.reasoning_exclude,
    };
    (block.effort.is_some() || block.max_tokens.is_some() || block.exclude.is_some())
        .then_some(block)
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
    let multimodal = !inputs.images.is_empty()
        || !inputs.files.is_empty()
        || !inputs.audio.is_empty()
        || !inputs.videos.is_empty();
    let user_content = if !multimodal {
        Content::Text(inputs.prompt.to_string())
    } else {
        // Multimodal user message: `prompt` verbatim, then each normalized input
        // image, then files, audio and videos - a fixed order so a prompt that
        // says "the second file" means the same thing on every call. Callers
        // wanting the labeled-reference preamble (image_gen's "Reference
        // images:" block) apply `assemble_prompt` themselves - it does not
        // belong in a plain Q&A chat.
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
        parts.extend(
            inputs
                .files
                .iter()
                .cloned()
                .map(|file| ContentPart::File { file }),
        );
        parts.extend(
            inputs
                .audio
                .iter()
                .cloned()
                .map(|input_audio| ContentPart::InputAudio { input_audio }),
        );
        parts.extend(
            inputs
                .videos
                .iter()
                .cloned()
                .map(|video_url| ContentPart::VideoUrl { video_url }),
        );
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
        seed: inputs.seed,
        temperature: inputs.temperature,
        max_tokens: inputs.max_tokens,
        top_p: inputs.top_p,
        top_k: inputs.top_k,
        stop: inputs
            .stop
            .iter()
            .filter(|s| !s.trim().is_empty())
            .cloned()
            .collect(),
        frequency_penalty: inputs.frequency_penalty,
        presence_penalty: inputs.presence_penalty,
        verbosity: non_blank(inputs.verbosity),
        response_format: inputs.response_format.clone(),
        plugins: inputs.plugins.clone(),
        web_search_options: inputs.web_search_options.clone(),
        reasoning: reasoning(inputs),
        provider: inputs.provider.clone(),
        audio: None,
        stream: false,
    };

    let completion = client.chat_completion(&req).await?;
    let usage = completion.usage;
    let cost = usage.as_ref().and_then(|u| u.cost);
    let receipt = crate::billing::Receipt {
        cost,
        generation_id: completion.id,
    };
    let generation_id = receipt.generation_id.clone();
    receipt.wrap(|| {
        let choice = first_choice(completion.choices, !inputs.images.is_empty())?;
        Ok(ChatResult {
            text: choice.text,
            cost,
            generation_id,
            reasoning: choice.reasoning,
            annotations: choice.annotations,
            finish_reason: choice.finish_reason,
            prompt_tokens: usage.as_ref().and_then(|u| u.prompt_tokens),
            completion_tokens: usage.as_ref().and_then(|u| u.completion_tokens),
        })
    })
}

/// The parts of the first choice the result carries.
struct FirstChoice {
    text: String,
    reasoning: Option<String>,
    annotations: Vec<serde_json::Value>,
    finish_reason: Option<String>,
}

/// The first choice with non-empty text, or why there is none.
fn first_choice(choices: Vec<Choice>, has_images: bool) -> Result<FirstChoice> {
    let choice = choices
        .into_iter()
        .next()
        .context("OpenRouter returned no choices")?;
    let text = choice
        .message
        .content
        .filter(|t| !t.is_empty())
        .with_context(|| {
            if has_images {
                "model returned no text (it may be an image-output-only model; \
                 use one with text output)"
            } else {
                "model returned no text"
            }
        })?;
    Ok(FirstChoice {
        text,
        reasoning: choice.message.reasoning,
        annotations: choice.message.annotations,
        finish_reason: choice.finish_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn mock_completion(body: serde_json::Value) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-generation-id", "gen-chat-7")
                    .set_body_json(body),
            )
            .mount(&server)
            .await;
        server
    }

    /// The generation id reaches the result on success and rides on the
    /// billing receipt when the reply is unusable (billed, but no text) - the
    /// two paths `get_generation` and `record_failed_receipt` depend on.
    #[tokio::test]
    async fn complete_carries_the_generation_id_on_success_and_in_the_failure_receipt() {
        let server = mock_completion(serde_json::json!({
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"cost": 0.01}
        }))
        .await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let inputs = ChatInputs {
            model: "m",
            prompt: "hello",
            ..Default::default()
        };
        let result = complete(&client, &inputs).await.unwrap();
        assert_eq!(result.generation_id.as_deref(), Some("gen-chat-7"));
        assert_eq!(result.cost, Some(0.01));

        let server =
            mock_completion(serde_json::json!({"choices": [], "usage": {"cost": 0.01}})).await;
        let client = OpenRouterClient::with_base_url(server.uri(), "test-key");
        let error = complete(&client, &inputs).await.unwrap_err();
        let receipt = crate::billing::Receipt::from_error(&error).expect("billed failure");
        assert_eq!(receipt.generation_id.as_deref(), Some("gen-chat-7"));
        assert_eq!(receipt.cost, Some(0.01));
    }
}
