//! Command-line interface: argument parsing (clap) and command dispatch.
//!
//! `main.rs` parses [`Cli`] then calls [`dispatch`]. The per-command handler
//! logic lives in [`commands`]; the model-table rendering helpers live in
//! [`table`].

mod commands;
mod table;

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};

use crate::image_gen;

#[derive(Parser)]
#[command(
    name = "openrouter-mcp",
    version,
    about = "MCP (stdio) server and CLI for OpenRouter - models, image/video/audio generation & description"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the MCP server over stdio.
    Mcp,
    /// List OpenRouter models with their capabilities and pricing.
    Models(ModelsArgs),
    /// Generate an image from a text prompt and save it to disk.
    Image(ImageArgs),
    /// Generate a video from a prompt (and optional first/last frame or reference images).
    Video(VideoArgs),
    /// Generate speech (text-to-speech) and save it to disk.
    Audio(AudioArgs),
    /// Generate music from a text prompt and save it to disk.
    Music(MusicArgs),
    /// Transcribe a local audio file to text (speech-to-text).
    Transcribe(TranscribeArgs),
    /// Describe local image(s) with a vision-capable model.
    Describe(DescribeArgs),
    /// Send a prompt to any chat/text model and print its reply.
    Chat(ChatArgs),
    /// Embed one or more texts with an embeddings model and print the vectors as JSON.
    Embed(EmbedArgs),
    /// Rerank documents against a query with a rerank model and print the ranking as JSON.
    Rerank(RerankArgs),
    /// Fetch the cost/latency record of one request by its generation id.
    Generation(GenerationArgs),
    /// Show basic info about the API key in use (label, owner, credits, limits).
    Key,
}

/// CLI flags for `describe`, mirroring the `describe_image` MCP tool.
#[derive(clap::Args)]
pub(crate) struct DescribeArgs {
    /// Vision-capable model id (image input, text output),
    /// e.g. google/gemini-2.5-flash or anthropic/claude-sonnet-4.6.
    #[arg(short, long)]
    model: String,
    /// Image to describe (repeatable). Use `label=path` to label a reference.
    #[arg(long = "image")]
    images: Vec<String>,
    /// Instruction/question about the image(s) (default: a detailed description).
    #[arg(short, long)]
    prompt: Option<String>,
    /// Longest-side cap (px) for input images before sending (default 1536, max 4096).
    #[arg(long)]
    max_image_dimension: Option<u32>,
    /// Reasoning effort: max, xhigh, high, medium, low, minimal, none.
    /// Omit to keep the model's own default.
    #[arg(long)]
    reasoning_effort: Option<String>,
    #[command(flatten)]
    provider: ProviderFlags,
}

/// CLI flags for `image`, mirroring the `generate_image` MCP tool.
#[derive(clap::Args)]
pub(crate) struct ImageArgs {
    /// Model id, e.g. google/gemini-3.1-flash-image-preview.
    #[arg(short, long)]
    model: String,
    /// Prompt text. Use --prompt-file to read from a file/stdin instead.
    #[arg(short, long)]
    prompt: Option<String>,
    /// Read the prompt from a file (use '-' for stdin).
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Aspect ratio, e.g. 1:1, 16:9 (Images API `aspect_ratio`).
    #[arg(long)]
    aspect_ratio: Option<String>,
    /// Resolution tier, e.g. 512, 1K, 2K, 4K (Images API `resolution`).
    #[arg(long)]
    image_size: Option<String>,
    /// Output size: WIDTHxHEIGHT pixels (e.g. 2048x2048) or a tier. Alternative
    /// to --aspect-ratio + --image-size; a pixel size combined with either is
    /// rejected (OpenRouter returns 400 for it).
    #[arg(long)]
    size: Option<String>,
    /// Base seed; variant N uses seed+N (provider support varies).
    #[arg(long)]
    seed: Option<u64>,
    /// Input image for editing / image-to-image (repeatable, order preserved).
    /// Use `label=path` to label a reference, e.g. --image product=./p.jpg.
    #[arg(long = "image")]
    images: Vec<String>,
    /// Longest-side cap (px) for input images before sending (default 1536,
    /// max 4096; env OPENROUTER_IMAGE_MAX_DIMENSION).
    #[arg(long)]
    max_image_dimension: Option<u32>,
    /// Number of variants to generate in parallel (1-16, seed-stepped).
    #[arg(long, default_value_t = 1)]
    variants: usize,
    /// Output path (single image, or the base name for variants). The extension
    /// is corrected to the format the provider actually returns.
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Output directory (alternative to --output; use with --output-name).
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Output base name (used with --output-dir).
    #[arg(long)]
    output_name: Option<String>,
    /// Output quality: auto, low, medium, high, xhigh, or max. Provider support
    /// varies.
    #[arg(long)]
    quality: Option<String>,
    /// Output file format: png, jpeg, webp, or svg. Provider support varies.
    #[arg(long)]
    output_format: Option<String>,
    /// Background: auto, transparent, or opaque. Provider support varies.
    #[arg(long)]
    background: Option<String>,
    /// Output compression 0-100 (webp/jpeg only). Provider support varies.
    #[arg(long, value_parser = clap::value_parser!(u32).range(0..=100))]
    output_compression: Option<u32>,
    #[command(flatten)]
    provider: ProviderFlags,
}

/// CLI flags for `video`, mirroring the `generate_video` MCP tool.
#[derive(clap::Args)]
pub(crate) struct VideoArgs {
    /// Model id, e.g. google/veo-3.1.
    #[arg(short, long)]
    model: String,
    /// Prompt text. Use --prompt-file to read from a file/stdin instead. Optional
    /// when a frame or a reference is given (image-only models take no text).
    #[arg(short, long)]
    prompt: Option<String>,
    /// Read the prompt from a file (use '-' for stdin).
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Clip duration in seconds.
    #[arg(long)]
    duration: Option<u32>,
    /// Resolution, e.g. 480p, 720p, 768p, 1080p, 1K, 2K, 4K.
    #[arg(long)]
    resolution: Option<String>,
    /// Aspect ratio, e.g. 16:9, 9:16, 1:1.
    #[arg(long)]
    aspect_ratio: Option<String>,
    /// Size as WIDTHxHEIGHT (interchangeable with resolution + aspect_ratio).
    #[arg(long)]
    size: Option<String>,
    /// Generate an audio track (for audio-capable models).
    #[arg(long)]
    with_audio: bool,
    /// Seed (provider support varies).
    #[arg(long)]
    seed: Option<u64>,
    /// Local image used as the first frame (image-to-video).
    #[arg(long)]
    first_frame: Option<PathBuf>,
    /// Local image used as the last frame (image-to-video).
    #[arg(long)]
    last_frame: Option<PathBuf>,
    /// Reference image (repeatable) for reference-to-video. Ignored, with a
    /// warning, when a first/last frame is given (frames win).
    #[arg(long = "reference-image")]
    reference_images: Vec<String>,
    /// Reference audio clip (repeatable): an https URL or a local file
    /// (mp3/wav/flac/m4a/ogg/aac/weba, inlined as a data URL, 20 MiB each).
    /// Ignored, with a warning, when a frame is given.
    #[arg(long = "reference-audio")]
    reference_audio: Vec<String>,
    /// Reference video clip (repeatable): an https URL or a local file
    /// (mp4/webm/mov, inlined as a data URL, 20 MiB each). Ignored, with a
    /// warning, when a frame is given.
    #[arg(long = "reference-video")]
    reference_videos: Vec<String>,
    /// Upscaling models only: creativity level (model-specific integer range).
    #[arg(long)]
    creativity: Option<u32>,
    /// Upscaling models only: output scale factor, > 0 (e.g. 2 for 2x).
    #[arg(long)]
    upscale_factor: Option<f64>,
    #[command(flatten)]
    provider: ProviderFlags,
    /// Longest-side cap (px) for input frame/reference images (default 1536, max 4096).
    #[arg(long)]
    max_image_dimension: Option<u32>,
    /// Output path (extension corrected to the returned format, e.g. .mp4).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Output directory (alternative to --output; use with --output-name).
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Output base name (used with --output-dir).
    #[arg(long)]
    output_name: Option<String>,
}

/// CLI flags for `audio`, mirroring the `generate_audio` MCP tool.
#[derive(clap::Args)]
pub(crate) struct AudioArgs {
    /// Model id, e.g. hexgrad/kokoro-82m.
    #[arg(short, long)]
    model: String,
    /// Text to synthesize. Use --input-file to read from a file/stdin instead.
    #[arg(short, long)]
    input: Option<String>,
    /// Read the input text from a file (use '-' for stdin).
    #[arg(long)]
    input_file: Option<PathBuf>,
    /// Voice id, valid only for the chosen model (e.g. af_heart for kokoro).
    /// Most models have no default voice; omit only for voice-cloning models
    /// driven by --voice-reference.
    #[arg(long)]
    voice: Option<String>,
    /// Local audio sample whose voice to clone (wav, mp3, flac, m4a, ogg, webm,
    /// aac; 15 MiB decoded max), sent as input_references.
    #[arg(long, value_name = "PATH")]
    voice_reference: Option<PathBuf>,
    /// Transcript of the --voice-reference sample (max 10000 characters).
    #[arg(long, requires = "voice_reference")]
    voice_reference_text: Option<String>,
    /// Output audio format: mp3 (default) or pcm.
    #[arg(long)]
    response_format: Option<String>,
    /// Playback speed (select models only).
    #[arg(long)]
    speed: Option<f64>,
    /// Output path (extension corrected to the returned format, e.g. .mp3).
    #[arg(short, long)]
    output: PathBuf,
    #[command(flatten)]
    provider: ProviderFlags,
}

/// CLI flags for `music`, mirroring the `generate_music` MCP tool.
#[derive(clap::Args)]
pub(crate) struct MusicArgs {
    /// Music model id, e.g. google/lyria-3-clip-preview (30 s clip) or
    /// google/lyria-3-pro-preview (full song).
    #[arg(short, long)]
    model: String,
    /// Musical description (genre, mood, tempo, instruments, lyrics). Use
    /// --prompt-file to read it from a file/stdin instead.
    #[arg(short, long)]
    prompt: Option<String>,
    /// Read the prompt from a file (use '-' for stdin).
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Requested container, sent as audio.format (wav, mp3, flac, opus, pcm16).
    /// Model-specific: Lyria ignores it and returns MP3. The saved extension
    /// follows the bytes returned.
    #[arg(long)]
    format: Option<String>,
    /// Seed for reproducible-ish generation (model support varies).
    #[arg(long)]
    seed: Option<u64>,
    /// Output path (extension corrected to the returned container, e.g. .mp3).
    #[arg(short, long)]
    output: PathBuf,
    #[command(flatten)]
    provider: ProviderFlags,
}

/// The `--provider <json>` flag shared by the subcommands whose MCP tool takes
/// a `provider` block. Parsed with the tool's own lenient path into the
/// matching `*Args` type from [`crate::server::provider`], so the CLI and MCP
/// validate identically and no normalization is duplicated here.
#[derive(clap::Args)]
pub(crate) struct ProviderFlags {
    /// OpenRouter `provider` block as JSON, the same object the matching MCP
    /// tool takes, e.g. '{"options":{"deepgram":{"diarize":true}}}'. Which keys
    /// are accepted depends on the endpoint (see the tool's `provider` docs).
    #[arg(long, value_name = "JSON")]
    provider: Option<String>,
}

impl ProviderFlags {
    /// Parse the flag into a provider args type; an absent flag is `T::default()`.
    pub(crate) fn parse<T>(&self) -> anyhow::Result<T>
    where
        T: serde::de::DeserializeOwned + Default,
    {
        let raw = serde_json::Value::String(self.provider.clone().unwrap_or_default());
        crate::server::schema::de_lenient(raw).context("--provider is not a valid JSON object")
    }
}

/// CLI flags for `transcribe`, mirroring the `transcribe_audio` MCP tool.
#[derive(clap::Args)]
pub(crate) struct TranscribeArgs {
    /// STT model id, e.g. openai/gpt-4o-mini-transcribe or openai/whisper-1.
    #[arg(short, long)]
    model: String,
    /// Local audio file to transcribe (wav/mp3/flac/m4a/ogg/webm/aac, max 25 MB).
    #[arg(short, long)]
    file: PathBuf,
    /// Container format; inferred from the file extension when omitted.
    #[arg(long)]
    format: Option<String>,
    /// ISO-639-1 language hint (e.g. en, ja) to improve accuracy.
    #[arg(short, long)]
    language: Option<String>,
    /// "json" (default) or "verbose_json" (adds language/duration/segments/
    /// words). Needs an OpenAI-compatible provider; others reject it with a 400.
    #[arg(long)]
    response_format: Option<String>,
    /// Comma-separated: segment and/or word. Only honored with
    /// --response-format verbose_json on an OpenAI-compatible provider.
    #[arg(long, value_delimiter = ',')]
    timestamp_granularities: Vec<String>,
    /// Sampling temperature (select providers only).
    #[arg(long)]
    temperature: Option<f64>,
    #[command(flatten)]
    provider: ProviderFlags,
}

/// CLI flags for `embed`, mirroring the `embed_text` MCP tool.
#[derive(clap::Args)]
pub(crate) struct EmbedArgs {
    /// Embeddings model id, e.g. openai/text-embedding-3-small (discover with
    /// `models --output-modalities embeddings`).
    #[arg(short, long)]
    model: String,
    /// Text to embed (repeatable; one flag per text).
    #[arg(short, long = "input", required = true)]
    inputs: Vec<String>,
    /// Output vector size, for models that support truncation.
    #[arg(long)]
    dimensions: Option<u32>,
    /// Provider-specific hint such as "query" or "document".
    #[arg(long)]
    input_type: Option<String>,
    #[command(flatten)]
    provider: ProviderFlags,
}

/// CLI flags for `rerank`, mirroring the `rerank_documents` MCP tool.
#[derive(clap::Args)]
pub(crate) struct RerankArgs {
    /// Rerank model id, e.g. cohere/rerank-v3.5 (discover with
    /// `models --output-modalities rerank`).
    #[arg(short, long)]
    model: String,
    /// The query to rank the documents against.
    #[arg(short, long)]
    query: String,
    /// Document text (repeatable; one flag per document, order preserved).
    #[arg(short, long = "document", required = true)]
    documents: Vec<String>,
    /// Return only the best N documents.
    #[arg(long)]
    top_n: Option<u32>,
    #[command(flatten)]
    provider: ProviderFlags,
}

/// CLI flags for `generation`, mirroring the `get_generation` MCP tool.
#[derive(clap::Args)]
pub(crate) struct GenerationArgs {
    /// The generation id (`X-Generation-Id`) reported by a previous request.
    #[arg(long = "id", value_name = "GENERATION_ID")]
    generation_id: String,
}

/// CLI flags for `chat`, mirroring the `chat_completion` MCP tool.
#[derive(clap::Args)]
pub(crate) struct ChatArgs {
    /// Chat/text model id, e.g. openai/gpt-5.4 or anthropic/claude-sonnet-4.6.
    #[arg(short, long)]
    model: String,
    /// Prompt text. Use --prompt-file to read from a file/stdin instead.
    #[arg(short, long)]
    prompt: Option<String>,
    /// Read the prompt from a file (use '-' for stdin).
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Optional system instruction prepended as a system message.
    #[arg(short, long)]
    system: Option<String>,
    /// Sampling temperature.
    #[arg(long)]
    temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    #[arg(long)]
    max_tokens: Option<u64>,
    /// Reasoning effort: max, xhigh, high, medium, low, minimal, none.
    /// Omit to keep the model's own default.
    #[arg(long)]
    reasoning_effort: Option<String>,
    /// Ask for a JSON object reply (response_format json_object). Not with
    /// --json-schema.
    #[arg(long)]
    json_mode: bool,
    /// JSON Schema the reply must conform to (strict structured output): inline
    /// JSON, or the path of a file holding it. Not with --json-mode.
    #[arg(long, value_name = "FILE-OR-JSON")]
    json_schema: Option<String>,
    /// Attach the OpenRouter web-search plugin with its defaults.
    #[arg(long)]
    web_search: bool,
    /// PDF parsing engine for file inputs: mistral-ocr, cloudflare-ai, or native.
    #[arg(long)]
    pdf_engine: Option<String>,
    /// Local document (PDF etc.) to send as a `file` part (repeatable, order
    /// preserved). Needs a model with file input.
    #[arg(long = "file")]
    files: Vec<PathBuf>,
    /// Local audio clip (wav/mp3/flac/m4a/ogg/webm/aac) to send as an
    /// `input_audio` part (repeatable). Needs a model with audio input.
    #[arg(long = "audio")]
    audio: Vec<PathBuf>,
    /// Video to send as a `video_url` part (repeatable): an http(s) URL is
    /// passed through for the provider to fetch, anything else is a local
    /// file sent as a data URL. Needs a model with video input.
    #[arg(long = "video")]
    videos: Vec<String>,
    #[command(flatten)]
    provider: ProviderFlags,
}

/// `--video` value -> a [`crate::server::media::VideoInput`]: URLs pass
/// through, everything else is a local path.
pub(crate) fn parse_video_arg(value: &str) -> crate::server::media::VideoInput {
    let is_url = value.starts_with("http://") || value.starts_with("https://");
    crate::server::media::VideoInput {
        url: is_url.then(|| value.to_string()),
        path: (!is_url).then(|| value.to_string()),
        ..Default::default()
    }
}

/// Read `--json-schema`: the value itself when it parses as a JSON object,
/// otherwise the contents of the file it names.
fn read_json_schema(
    value: &str,
) -> anyhow::Result<std::collections::BTreeMap<String, serde_json::Value>> {
    let text = if value.trim_start().starts_with('{') {
        value.to_string()
    } else {
        std::fs::read_to_string(value)
            .with_context(|| format!("could not read --json-schema file {value}"))?
    };
    serde_json::from_str(&text).context("--json-schema is not a JSON object")
}

/// CLI flags for `models`, mirroring the `list_models` MCP tool.
#[derive(clap::Args)]
pub(crate) struct ModelsArgs {
    /// Server-side free-text search by model name or slug (OpenRouter `q`).
    #[arg(short, long)]
    query: Option<String>,
    /// Local case-insensitive filter across id, name, and description
    /// (e.g. --search openai). Applied after the server-side query.
    #[arg(short, long)]
    search: Option<String>,
    /// Output modalities (comma-separated): text, image, audio, embeddings,
    /// video, rerank, speech, transcription - or "all".
    #[arg(long)]
    output_modalities: Option<String>,
    /// Input modalities (comma-separated): text, image, audio, file.
    #[arg(long)]
    input_modalities: Option<String>,
    /// Required supported parameters (comma-separated), e.g. "tools".
    #[arg(long)]
    supported_parameters: Option<String>,
    /// Sort order (default: top-weekly): most-popular, newest, top-weekly,
    /// pricing-low-to-high, pricing-high-to-low, context-high-to-low,
    /// throughput-high-to-low, latency-low-to-high, intelligence-high-to-low,
    /// coding-high-to-low, agentic-high-to-low, design-arena-elo-high-to-low.
    #[arg(long)]
    sort: Option<String>,
    /// Minimum context length in tokens.
    #[arg(long)]
    min_context: Option<u64>,
    /// Use-case category: programming, roleplay, marketing, marketing/seo,
    /// technology, science, translation, legal, finance, health, trivia, academia.
    #[arg(long)]
    category: Option<String>,
    /// Hosting providers (comma-separated), e.g. "OpenAI,Anthropic".
    #[arg(long)]
    providers: Option<String>,
    /// Server-side page size (1..=1000). The local 20-row cap still applies
    /// unless --all is given.
    #[arg(long)]
    limit: Option<u64>,
    /// Server-side records to skip (pair with --limit to page).
    #[arg(long)]
    offset: Option<u64>,
    /// Only models with zero-data-retention endpoints.
    #[arg(long)]
    zdr: bool,
    /// Data region of the model's endpoints: eu or us.
    #[arg(long)]
    region: Option<String>,
    /// Return all matching models instead of just the first 20.
    #[arg(long)]
    all: bool,
    /// Print a human-readable table instead of the default JSON output.
    #[arg(long)]
    table: bool,
}

/// Route a parsed [`Cli`] to the matching command handler.
pub(crate) async fn dispatch(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Some(Command::Models(args)) => commands::run_models(args).await,
        Some(Command::Image(args)) => commands::run_image(args).await,
        Some(Command::Video(args)) => commands::run_video(args).await,
        Some(Command::Audio(args)) => commands::run_audio(args).await,
        Some(Command::Music(args)) => commands::run_music(args).await,
        Some(Command::Transcribe(args)) => commands::run_transcribe(args).await,
        Some(Command::Describe(args)) => commands::run_describe(args).await,
        Some(Command::Chat(args)) => commands::run_chat(args).await,
        Some(Command::Embed(args)) => commands::run_embed(args).await,
        Some(Command::Rerank(args)) => commands::run_rerank(args).await,
        Some(Command::Generation(args)) => commands::run_generation(args).await,
        Some(Command::Key) => commands::run_key().await,
        Some(Command::Mcp) | None => crate::server::run().await,
    }
}

/// Resolve the prompt text and its source (`inline`/`file`/`stdin`).
fn resolve_prompt(
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
) -> anyhow::Result<(String, String)> {
    let (text, source) = if let Some(pf) = prompt_file {
        if pf == Path::new("-") {
            use std::io::Read;
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            (s.trim().to_string(), "stdin".to_string())
        } else {
            let s = std::fs::read_to_string(&pf)
                .with_context(|| format!("could not read prompt file {}", pf.display()))?;
            (s.trim().to_string(), "file".to_string())
        }
    } else {
        match prompt {
            Some(p) => (p, "inline".to_string()),
            None => anyhow::bail!("provide --prompt or --prompt-file"),
        }
    };
    if text.trim().is_empty() {
        anyhow::bail!("prompt is empty (an empty --prompt-file / stdin is not allowed)");
    }
    Ok((text, source))
}

/// Resolve the base output path from `--output`, or `--output-dir`+`--output-name`.
fn resolve_base_output(
    output: Option<PathBuf>,
    output_dir: Option<PathBuf>,
    output_name: Option<String>,
) -> anyhow::Result<PathBuf> {
    if let Some(o) = output {
        return Ok(o);
    }
    match (output_dir, output_name) {
        (Some(dir), Some(name)) => Ok(dir.join(name)),
        _ => anyhow::bail!("provide --output, or both --output-dir and --output-name"),
    }
}

/// Parse a CLI `--image` value, which is either `path` or `label=path`.
pub(crate) fn parse_image_arg(value: &str) -> image_gen::InputImage {
    // Only treat `left=right` as a labeled reference when `left` looks like a
    // bare label (alphanumeric/`_`/`-`), so a real path containing '=' (e.g.
    // `./a=b/img.png`) is kept whole instead of being mis-split.
    if let Some((label, path)) = value.split_once('=') {
        let is_label = !label.is_empty()
            && !path.is_empty()
            && label
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '-');
        if is_label {
            return image_gen::InputImage::from_path(path, Some(label.to_string()));
        }
    }
    image_gen::InputImage::from_path(value, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image_gen::ImageSource;

    /// The label/path split has to distinguish a real label from a path that
    /// merely contains '=' - the case the function exists to get right.
    #[test]
    fn parse_image_arg_splits_labels_but_keeps_paths_containing_equals() {
        let labeled = parse_image_arg("product=./p.jpg");
        assert_eq!(labeled.label.as_deref(), Some("product"));
        assert_eq!(labeled.display_name(), "p.jpg");

        // A path with '=' in a directory name is NOT a label - keep it whole.
        let awkward = parse_image_arg("./a=b/img.png");
        assert_eq!(awkward.label, None);
        match &awkward.source {
            ImageSource::Path(p) => assert_eq!(p.to_string_lossy(), "./a=b/img.png"),
            _ => panic!("expected a path source"),
        }

        // Plain path, and the degenerate empty-side forms, stay unlabeled.
        assert_eq!(parse_image_arg("./plain.png").label, None);
        assert_eq!(parse_image_arg("=/leading.png").label, None);
        assert_eq!(parse_image_arg("trailing=").label, None);
    }

    /// `audio` mirrors generate_audio: `--voice` is optional (voice-cloning
    /// models take none), the cloning sample comes from `--voice-reference`
    /// with an optional `--voice-reference-text` that is meaningless without
    /// it, and `--provider` carries the same block the tool takes.
    #[test]
    fn audio_flags_make_voice_optional_and_take_a_voice_reference() {
        use crate::server::provider::ProviderOptionsArgs;
        let cli = Cli::try_parse_from([
            "openrouter-mcp",
            "audio",
            "-m",
            "fish-audio/s1",
            "-i",
            "hello",
            "-o",
            "out.mp3",
            "--voice-reference",
            "sample.wav",
            "--voice-reference-text",
            "the sample words",
            "--provider",
            "{\"options\":{\"openai\":{\"instructions\":\"cheerful\"}}}",
        ])
        .unwrap();
        let Some(Command::Audio(args)) = cli.command else {
            panic!("expected audio")
        };
        assert_eq!(args.voice, None);
        assert_eq!(args.voice_reference, Some(PathBuf::from("sample.wav")));
        assert_eq!(
            args.voice_reference_text.as_deref(),
            Some("the sample words")
        );
        let block = args
            .provider
            .parse::<ProviderOptionsArgs>()
            .unwrap()
            .into_options()
            .unwrap()
            .unwrap();
        assert_eq!(
            block.options["openai"],
            serde_json::json!({"instructions": "cheerful"})
        );

        // The classic form still parses, with the voice carried through.
        let cli = Cli::try_parse_from([
            "openrouter-mcp",
            "audio",
            "-m",
            "hexgrad/kokoro-82m",
            "-i",
            "hi",
            "-o",
            "out.mp3",
            "--voice",
            "af_heart",
        ])
        .unwrap();
        let Some(Command::Audio(args)) = cli.command else {
            panic!("expected audio")
        };
        assert_eq!(args.voice.as_deref(), Some("af_heart"));
        assert_eq!(args.voice_reference, None);

        // A transcript without a sample is rejected at parse time.
        assert!(
            Cli::try_parse_from([
                "openrouter-mcp",
                "audio",
                "-m",
                "m",
                "-i",
                "hi",
                "-o",
                "out.mp3",
                "--voice-reference-text",
                "words",
            ])
            .is_err()
        );
    }

    /// `--provider` takes the same JSON block the MCP tool takes and goes
    /// through the same lenient parse + validation, so the CLI cannot drift.
    #[test]
    fn transcribe_provider_flag_parses_and_validates_like_the_tool() {
        use crate::server::provider::ProviderOptionsArgs;
        let cli = Cli::try_parse_from([
            "openrouter-mcp",
            "transcribe",
            "-m",
            "deepgram/nova-3",
            "-f",
            "a.mp3",
            "--provider",
            "{\"options\":{\"deepgram\":{\"diarize\":true}}}",
        ])
        .unwrap();
        let Some(Command::Transcribe(args)) = cli.command else {
            panic!("expected transcribe")
        };
        let block = args
            .provider
            .parse::<ProviderOptionsArgs>()
            .unwrap()
            .into_options()
            .unwrap()
            .unwrap();
        assert_eq!(
            block.options["deepgram"],
            serde_json::json!({"diarize": true})
        );

        // Absent flag -> nothing sent.
        let cli = Cli::try_parse_from(["openrouter-mcp", "transcribe", "-m", "m", "-f", "a.mp3"])
            .unwrap();
        let Some(Command::Transcribe(args)) = cli.command else {
            panic!("expected transcribe")
        };
        assert!(
            args.provider
                .parse::<ProviderOptionsArgs>()
                .unwrap()
                .into_options()
                .unwrap()
                .is_none()
        );

        // Malformed JSON is a parse error, a non-object slug value a validation error.
        let bad = ProviderFlags {
            provider: Some("{nope".to_string()),
        };
        assert!(bad.parse::<ProviderOptionsArgs>().is_err());
        let bad_value = ProviderFlags {
            provider: Some("{\"options\":{\"deepgram\":true}}".to_string()),
        };
        assert!(
            bad_value
                .parse::<ProviderOptionsArgs>()
                .unwrap()
                .into_options()
                .is_err()
        );
    }

    /// `image --provider` takes the `/images` block (routing subset + options)
    /// and `--size` the pixel/tier string, parsed like the MCP tool.
    #[test]
    fn image_provider_and_size_flags_parse_like_the_tool() {
        use crate::server::provider::ImageProviderArgs;
        let cli = Cli::try_parse_from([
            "openrouter-mcp",
            "image",
            "-m",
            "black-forest-labs/flux.2-pro",
            "-p",
            "an owl",
            "--size",
            "2048x2048",
            "--provider",
            "{\"order\":[\"black-forest-labs\"],\"options\":{\"black-forest-labs\":{\"steps\":28}}}",
        ])
        .unwrap();
        let Some(Command::Image(args)) = cli.command else {
            panic!("expected image")
        };
        assert_eq!(args.size.as_deref(), Some("2048x2048"));
        let block = args
            .provider
            .parse::<ImageProviderArgs>()
            .unwrap()
            .into_image_provider()
            .unwrap()
            .unwrap();
        assert_eq!(block.order, vec!["black-forest-labs".to_string()]);
        assert_eq!(
            block.options["black-forest-labs"],
            serde_json::json!({"steps": 28})
        );

        // Absent flags -> nothing sent.
        let cli = Cli::try_parse_from(["openrouter-mcp", "image", "-m", "m", "-p", "p"]).unwrap();
        let Some(Command::Image(args)) = cli.command else {
            panic!("expected image")
        };
        assert!(args.size.is_none());
        assert!(
            args.provider
                .parse::<ImageProviderArgs>()
                .unwrap()
                .into_image_provider()
                .unwrap()
                .is_none()
        );
    }

    /// `video` takes the new reference kinds, the upscaling knobs and
    /// `--provider`, and no longer demands `--prompt` (image-only models).
    #[test]
    fn video_flags_parse_references_upscaling_knobs_and_provider() {
        use crate::server::provider::ProviderOptionsArgs;
        let cli = Cli::try_parse_from([
            "openrouter-mcp",
            "video",
            "-m",
            "bytedance/seedance-2.0",
            "--reference-audio",
            "beat.mp3",
            "--reference-audio",
            "https://cdn/song.mp3",
            "--reference-video",
            "ref.mp4",
            "--creativity",
            "3",
            "--upscale-factor",
            "2",
            "--provider",
            "{\"options\":{\"google-vertex\":{\"negativePrompt\":\"blurry\"}}}",
        ])
        .unwrap();
        let Some(Command::Video(args)) = cli.command else {
            panic!("expected video")
        };
        assert_eq!(args.prompt, None);
        assert_eq!(
            args.reference_audio,
            vec!["beat.mp3".to_string(), "https://cdn/song.mp3".to_string()]
        );
        assert_eq!(args.reference_videos, vec!["ref.mp4".to_string()]);
        assert_eq!(args.creativity, Some(3));
        assert_eq!(args.upscale_factor, Some(2.0));
        let block = args
            .provider
            .parse::<ProviderOptionsArgs>()
            .unwrap()
            .into_options()
            .unwrap()
            .unwrap();
        assert_eq!(
            block.options["google-vertex"],
            serde_json::json!({"negativePrompt": "blurry"})
        );
    }

    /// The retrieval subcommands mirror their MCP tools: repeatable text
    /// flags, the optional knobs, and the shared `--provider` routing block
    /// parsed through the tool's own args type.
    #[test]
    fn embed_rerank_and_generation_subcommands_parse_like_their_tools() {
        use crate::server::provider::ProviderRoutingArgs;
        let cli = Cli::try_parse_from([
            "openrouter-mcp",
            "embed",
            "-m",
            "openai/text-embedding-3-small",
            "--input",
            "a",
            "--input",
            "b",
            "--dimensions",
            "256",
            "--input-type",
            "query",
            "--provider",
            "{\"order\":[\"openai\"],\"zdr\":true}",
        ])
        .unwrap();
        let Some(Command::Embed(args)) = cli.command else {
            panic!("expected embed")
        };
        assert_eq!(args.inputs, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(args.dimensions, Some(256));
        assert_eq!(args.input_type.as_deref(), Some("query"));
        let routing = args
            .provider
            .parse::<ProviderRoutingArgs>()
            .unwrap()
            .into_routing()
            .unwrap()
            .unwrap();
        assert_eq!(routing.order, vec!["openai".to_string()]);
        assert_eq!(routing.zdr, Some(true));
        // At least one --input is mandatory, like the tool's minItems: 1.
        assert!(Cli::try_parse_from(["openrouter-mcp", "embed", "-m", "m"]).is_err());

        let cli = Cli::try_parse_from([
            "openrouter-mcp",
            "rerank",
            "-m",
            "cohere/rerank-v3.5",
            "-q",
            "rust",
            "--document",
            "go",
            "--document",
            "rust lang",
            "--top-n",
            "1",
        ])
        .unwrap();
        let Some(Command::Rerank(args)) = cli.command else {
            panic!("expected rerank")
        };
        assert_eq!(args.query, "rust");
        assert_eq!(
            args.documents,
            vec!["go".to_string(), "rust lang".to_string()]
        );
        assert_eq!(args.top_n, Some(1));
        assert!(
            args.provider
                .parse::<ProviderRoutingArgs>()
                .unwrap()
                .into_routing()
                .unwrap()
                .is_none()
        );

        let cli = Cli::try_parse_from(["openrouter-mcp", "generation", "--id", "gen-1"]).unwrap();
        let Some(Command::Generation(args)) = cli.command else {
            panic!("expected generation")
        };
        assert_eq!(args.generation_id, "gen-1");
    }

    /// The chat-family subcommands take `--provider` as the routing block, and
    /// `chat` parses its structured-output / plugin flags the way the tool does.
    #[test]
    fn chat_family_flags_parse_like_the_tools() {
        use crate::server::provider::ProviderRoutingArgs;
        let schema_file = std::env::temp_dir().join("openrouter-mcp-cli-schema.json");
        std::fs::write(&schema_file, r#"{"title":"Answer","type":"object"}"#).unwrap();
        let cli = Cli::try_parse_from([
            "openrouter-mcp",
            "chat",
            "-m",
            "openai/gpt-5.4",
            "-p",
            "hi",
            "--json-schema",
            &schema_file.to_string_lossy(),
            "--web-search",
            "--pdf-engine",
            "native",
            "--file",
            "report.pdf",
            "--audio",
            "clip.mp3",
            "--video",
            "https://example.com/v.mp4",
            "--video",
            "local.mp4",
            "--provider",
            "{\"order\":[\"openai\"]}",
        ])
        .unwrap();
        let Some(Command::Chat(args)) = cli.command else {
            panic!("expected chat")
        };
        assert!(args.web_search);
        assert!(!args.json_mode);
        assert_eq!(args.pdf_engine.as_deref(), Some("native"));
        assert_eq!(args.files, vec![PathBuf::from("report.pdf")]);
        assert_eq!(args.audio, vec![PathBuf::from("clip.mp3")]);
        let videos: Vec<_> = args.videos.iter().map(|v| parse_video_arg(v)).collect();
        assert_eq!(videos[0].url.as_deref(), Some("https://example.com/v.mp4"));
        assert_eq!(videos[0].path, None);
        assert_eq!(videos[1].path.as_deref(), Some("local.mp4"));
        assert_eq!(videos[1].url, None);
        let schema = read_json_schema(args.json_schema.as_deref().unwrap()).unwrap();
        assert_eq!(schema["title"], "Answer");
        // Inline JSON works too, and garbage is an error.
        assert_eq!(
            read_json_schema(r#"{"type":"object"}"#).unwrap()["type"],
            "object"
        );
        assert!(read_json_schema("no-such-file.json").is_err());
        let routing = args
            .provider
            .parse::<ProviderRoutingArgs>()
            .unwrap()
            .into_routing()
            .unwrap()
            .unwrap();
        assert_eq!(routing.order, vec!["openai".to_string()]);

        for sub in [
            vec!["describe", "-m", "m", "--image", "a.png"],
            vec!["music", "-m", "m", "-p", "x", "-o", "t.mp3"],
        ] {
            let mut argv = vec!["openrouter-mcp"];
            argv.extend(sub);
            argv.extend(["--provider", "{\"only\":[\"google-vertex\"]}"]);
            let cli = Cli::try_parse_from(argv).unwrap();
            let flags = match cli.command {
                Some(Command::Describe(a)) => a.provider,
                Some(Command::Music(a)) => a.provider,
                _ => panic!("unexpected subcommand"),
            };
            let routing = flags
                .parse::<ProviderRoutingArgs>()
                .unwrap()
                .into_routing()
                .unwrap()
                .unwrap();
            assert_eq!(routing.only, vec!["google-vertex".to_string()]);
        }
    }

    #[test]
    fn resolve_prompt_reads_inline_and_file_and_rejects_empty() {
        // Inline wins and reports its source.
        let (text, source) = resolve_prompt(Some("a kite".to_string()), None).unwrap();
        assert_eq!((text.as_str(), source.as_str()), ("a kite", "inline"));

        // A file is read and trimmed, and reports source "file".
        let path = std::env::temp_dir().join("openrouter-mcp-prompt-test.txt");
        std::fs::write(&path, "  from a file \n").unwrap();
        let (text, source) = resolve_prompt(None, Some(path.clone())).unwrap();
        assert_eq!((text.as_str(), source.as_str()), ("from a file", "file"));

        // Neither source, an empty file, and a missing file are all errors.
        assert!(resolve_prompt(None, None).is_err());
        std::fs::write(&path, "   \n").unwrap();
        let err = resolve_prompt(None, Some(path)).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
        let missing = std::env::temp_dir().join("openrouter-mcp-no-such-prompt.txt");
        assert!(resolve_prompt(None, Some(missing)).is_err());
    }

    #[test]
    fn resolve_base_output_accepts_output_or_dir_plus_name() {
        let direct = resolve_base_output(Some(PathBuf::from("out/hero.png")), None, None).unwrap();
        assert_eq!(direct, PathBuf::from("out/hero.png"));

        let joined =
            resolve_base_output(None, Some(PathBuf::from("out")), Some("hero".to_string()))
                .unwrap();
        assert_eq!(joined, PathBuf::from("out").join("hero"));

        // --output wins when both forms are given.
        let both = resolve_base_output(
            Some(PathBuf::from("explicit.png")),
            Some(PathBuf::from("out")),
            Some("hero".to_string()),
        )
        .unwrap();
        assert_eq!(both, PathBuf::from("explicit.png"));

        // Half of the dir+name pair is not enough.
        assert!(resolve_base_output(None, None, None).is_err());
        assert!(resolve_base_output(None, Some(PathBuf::from("out")), None).is_err());
        assert!(resolve_base_output(None, None, Some("hero".to_string())).is_err());
    }
}
