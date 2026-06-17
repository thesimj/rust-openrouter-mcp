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
    /// Describe local image(s) with a vision-capable model.
    Describe(DescribeArgs),
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
    /// Longest-side cap (px) for input images before sending (default 800).
    #[arg(long)]
    max_image_dimension: Option<u32>,
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
    /// Aspect ratio, e.g. 1:1, 16:9 (mapped to image_config.aspect_ratio).
    #[arg(long)]
    aspect_ratio: Option<String>,
    /// Resolution tier, e.g. 1K, 2K, 4K (mapped to image_config.image_size).
    #[arg(long)]
    image_size: Option<String>,
    /// Base seed; variant N uses seed+N (provider support varies).
    #[arg(long)]
    seed: Option<u64>,
    /// Use image-only output modality (for models that only output images).
    #[arg(long)]
    image_only: bool,
    /// Input image for editing / image-to-image (repeatable, order preserved).
    /// Use `label=path` to label a reference, e.g. --image product=./p.jpg.
    #[arg(long = "image")]
    images: Vec<String>,
    /// Longest-side cap (px) for input images before sending (default 800;
    /// env OPENROUTER_IMAGE_MAX_DIMENSION).
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
}

/// CLI flags for `video`, mirroring the `generate_video` MCP tool.
#[derive(clap::Args)]
pub(crate) struct VideoArgs {
    /// Model id, e.g. google/veo-3.1.
    #[arg(short, long)]
    model: String,
    /// Prompt text. Use --prompt-file to read from a file/stdin instead.
    #[arg(short, long)]
    prompt: Option<String>,
    /// Read the prompt from a file (use '-' for stdin).
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Clip duration in seconds.
    #[arg(long)]
    duration: Option<u32>,
    /// Resolution, e.g. 480p, 720p, 1080p, 1K, 2K, 4K.
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
    generate_audio: bool,
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
    /// Longest-side cap (px) for input frame/reference images (default 800).
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
    /// Model id, e.g. openai/gpt-4o-mini-tts or hexgrad/kokoro-82m.
    #[arg(short, long)]
    model: String,
    /// Text to synthesize. Use --input-file to read from a file/stdin instead.
    #[arg(short, long)]
    input: Option<String>,
    /// Read the input text from a file (use '-' for stdin).
    #[arg(long)]
    input_file: Option<PathBuf>,
    /// Voice id (varies by model, e.g. alloy).
    #[arg(long)]
    voice: String,
    /// Output audio format: mp3 (default) or pcm.
    #[arg(long)]
    response_format: Option<String>,
    /// Playback speed (select models only).
    #[arg(long)]
    speed: Option<f64>,
    /// Output path (extension corrected to the returned format, e.g. .mp3).
    #[arg(short, long)]
    output: PathBuf,
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
    /// Sort order (default: top-weekly). See --help for all values.
    #[arg(long)]
    sort: Option<String>,
    /// Minimum context length in tokens.
    #[arg(long)]
    min_context: Option<u64>,
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
        Some(Command::Describe(args)) => commands::run_describe(args).await,
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
fn parse_image_arg(value: &str) -> image_gen::InputImage {
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
            return image_gen::InputImage {
                path: PathBuf::from(path),
                label: Some(label.to_string()),
            };
        }
    }
    image_gen::InputImage {
        path: PathBuf::from(value),
        label: None,
    }
}
