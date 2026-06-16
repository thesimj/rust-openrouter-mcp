//! `openrouter-mcp` — an MCP (stdio) server for OpenRouter, doubling as a CLI.
//!
//! Run the MCP server with: `openrouter-mcp` (or `openrouter-mcp mcp`)
//! Or use it directly, e.g.: `openrouter-mcp models --output-modalities image --sort newest`
//!
//! Requires the `OPENROUTER_API_KEY` environment variable (or a local `.env`).

mod image_gen;
mod image_io;
mod manifest;
mod openrouter;
mod server;
mod stats;
mod tasks;

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};

use image_gen::GenerateRequest;
use openrouter::{ModelsQuery, OpenRouterClient};

#[derive(Parser)]
#[command(
    name = "openrouter-mcp",
    version,
    about = "MCP (stdio) server and CLI for OpenRouter — models, image generation & description"
)]
struct Cli {
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
    /// Describe local image(s) with a vision-capable model.
    Describe(DescribeArgs),
}

/// CLI flags for `describe`, mirroring the `describe_image` MCP tool.
#[derive(clap::Args)]
struct DescribeArgs {
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
struct ImageArgs {
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

/// Resolve the prompt text and its source (`inline`/`file`/`stdin`).
fn resolve_prompt(
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
) -> anyhow::Result<(String, String)> {
    if let Some(pf) = prompt_file {
        if pf == Path::new("-") {
            use std::io::Read;
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            return Ok((s.trim().to_string(), "stdin".to_string()));
        }
        let s = std::fs::read_to_string(&pf)
            .with_context(|| format!("could not read prompt file {}", pf.display()))?;
        return Ok((s.trim().to_string(), "file".to_string()));
    }
    match prompt {
        Some(p) => Ok((p, "inline".to_string())),
        None => anyhow::bail!("provide --prompt or --prompt-file"),
    }
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
    match value.split_once('=') {
        // Only treat as labeled when the left side looks like a label, not a
        // Windows drive letter (e.g. `C:\...` has no `=`, but guard anyway).
        Some((label, path)) if !label.is_empty() && !path.is_empty() => image_gen::InputImage {
            path: PathBuf::from(path),
            label: Some(label.to_string()),
        },
        _ => image_gen::InputImage {
            path: PathBuf::from(value),
            label: None,
        },
    }
}

/// CLI flags for `models`, mirroring the `list_models` MCP tool.
#[derive(clap::Args)]
struct ModelsArgs {
    /// Server-side free-text search by model name or slug (OpenRouter `q`).
    #[arg(short, long)]
    query: Option<String>,
    /// Local case-insensitive filter across id, name, and description
    /// (e.g. --search openai). Applied after the server-side query.
    #[arg(short, long)]
    search: Option<String>,
    /// Output modalities (comma-separated): text, image, audio, embeddings,
    /// video, rerank, speech, transcription — or "all".
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a local `.env` file if present (does not override real env vars).
    // Key resolution is therefore: real env var > .env entry > error in from_env().
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();
    match cli.command {
        Some(Command::Models(args)) => run_models(args).await,
        Some(Command::Image(args)) => run_image(args).await,
        Some(Command::Describe(args)) => run_describe(args).await,
        Some(Command::Mcp) | None => server::run().await,
    }
}

/// Describe local image(s) with a vision model and print the text to stdout.
async fn run_describe(args: DescribeArgs) -> anyhow::Result<()> {
    let client = OpenRouterClient::from_env()?;
    if args.images.is_empty() {
        anyhow::bail!("provide at least one --image");
    }
    let req = image_gen::DescribeRequest {
        model: args.model,
        prompt: args
            .prompt
            .unwrap_or_else(|| "Describe this image in detail.".to_string()),
        images: args.images.iter().map(|v| parse_image_arg(v)).collect(),
        max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
    };
    let result = image_gen::describe_image(&client, &req).await?;
    println!("{}", result.text);
    if let Some(cost) = result.cost {
        eprintln!("cost: ${cost}");
    }
    Ok(())
}

/// Generate one or more images and save them, plus a sidecar manifest. The CLI
/// blocks until all variants finish (run in parallel).
async fn run_image(args: ImageArgs) -> anyhow::Result<()> {
    let client = OpenRouterClient::from_env()?;
    let (prompt, prompt_source) = resolve_prompt(args.prompt, args.prompt_file)?;
    let base = resolve_base_output(args.output, args.output_dir, args.output_name)?;
    let variants = args.variants.clamp(1, 16);

    let req = GenerateRequest {
        model: args.model,
        prompt,
        aspect_ratio: args.aspect_ratio,
        image_size: args.image_size,
        seed: args.seed,
        image_only: args.image_only,
        images: args.images.iter().map(|v| parse_image_arg(v)).collect(),
        max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
    };

    let summary = image_gen::run_job(&client, &req, variants, &base, &prompt_source).await?;

    for img in &summary.images {
        eprintln!(
            "saved {} ({}x{} = {}{})",
            img.path.display(),
            img.width,
            img.height,
            img.actual_aspect_ratio,
            img.seed.map(|s| format!(", seed {s}")).unwrap_or_default(),
        );
    }
    for warning in &summary.warnings {
        eprintln!("note: {warning}");
    }
    for error in &summary.errors {
        eprintln!("error: {error}");
    }
    eprintln!("manifest: {}", summary.manifest_path.display());

    if summary.images.is_empty() {
        anyhow::bail!("all {variants} variant(s) failed");
    }
    // stdout: the saved paths, for scripting.
    for img in &summary.images {
        println!("{}", img.path.display());
    }
    Ok(())
}

/// Format an OpenRouter per-token price string as USD per 1M tokens.
/// Returns "-" when missing/unparseable and "0" for zero.
fn per_million(price: &Option<String>) -> String {
    match price.as_deref().and_then(|s| s.parse::<f64>().ok()) {
        Some(0.0) => "0".to_string(),
        // Negative values are sentinels (e.g. openrouter/auto uses -1 = "varies").
        Some(p) if p < 0.0 => "-".to_string(),
        Some(p) => {
            // Trim trailing zeros from a fixed-precision render.
            let s = format!("{:.4}", p * 1_000_000.0);
            let s = s.trim_end_matches('0').trim_end_matches('.');
            format!("${s}")
        }
        None => "-".to_string(),
    }
}

/// Human-readable context length, e.g. 131072 -> "128K", 1000000 -> "1M".
/// Prefers exact decimal (÷1000) then exact binary (÷1024), else 1-decimal.
fn human_context(n: Option<u64>) -> String {
    let n = match n {
        Some(n) if n > 0 => n,
        _ => return "-".to_string(),
    };
    let trim = |v: f64, suffix: &str| {
        let s = format!("{v:.2}");
        let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        format!("{s}{suffix}")
    };
    if n >= 1_000_000 {
        if n % 1_000_000 == 0 {
            return format!("{}M", n / 1_000_000);
        }
        if n % (1024 * 1024) == 0 {
            return format!("{}M", n / (1024 * 1024));
        }
        return trim(n as f64 / 1_000_000.0, "M");
    }
    if n % 1000 == 0 {
        return format!("{}K", n / 1000);
    }
    if n % 1024 == 0 {
        return format!("{}K", n / 1024);
    }
    if n >= 1000 {
        return trim(n as f64 / 1000.0, "K");
    }
    n.to_string()
}

/// Trim a float to a compact decimal string (up to 8 places, no trailing zeros).
fn trim_num(v: f64) -> String {
    let s = format!("{v:.8}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Render a list of prices as `$x<unit>` or `$min-max<unit>`.
fn range_str(vals: &[f64], unit: &str) -> String {
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < f64::EPSILON {
        format!("${}{}", trim_num(min), unit)
    } else {
        format!("${}-{}{}", trim_num(min), trim_num(max), unit)
    }
}

/// Derive a concise price for a video model from its heterogeneous
/// `pricing_skus`: dollars-per-second, cents-per-second, or per video-token.
fn video_price(skus: &std::collections::BTreeMap<String, String>) -> String {
    let collect = |pred: &dyn Fn(&str) -> bool| -> Vec<f64> {
        skus.iter()
            .filter(|(k, _)| pred(k))
            .filter_map(|(_, v)| v.parse::<f64>().ok())
            .collect()
    };
    let secs = collect(&|k| k.contains("duration_seconds"));
    if !secs.is_empty() {
        return range_str(&secs, "/s");
    }
    let cents = collect(&|k| k.contains("second"));
    if !cents.is_empty() {
        let dollars: Vec<f64> = cents.iter().map(|c| c / 100.0).collect();
        return range_str(&dollars, "/s");
    }
    let toks = collect(&|k| k.contains("token"));
    if !toks.is_empty() {
        return range_str(&toks, "/vid-tok");
    }
    "-".to_string()
}

/// Format a raw decimal price string as `$<value>`, or "-" if missing/zero.
fn dollars(price: &Option<String>) -> String {
    match price.as_deref().and_then(|s| s.parse::<f64>().ok()) {
        Some(v) if v > 0.0 => format!("${}", price.as_deref().unwrap_or("")),
        _ => "-".to_string(),
    }
}

/// The single output modality a model is bucketed under for the sectioned
/// table. Highest-priority media modality wins; text is the fallback.
fn primary_modality(m: &openrouter::Model) -> &'static str {
    const ORDER: [&str; 7] = [
        "video",
        "image",
        "audio",
        "speech",
        "transcription",
        "embeddings",
        "rerank",
    ];
    let outs = m.architecture.as_ref().map(|a| &a.output_modalities);
    for cat in ORDER {
        if outs.is_some_and(|o| o.iter().any(|x| x == cat)) {
            return cat;
        }
    }
    "text"
}

/// A rendered model table: the `table` body (destined for stdout) and any
/// `notes` (destined for stderr, keeping the table itself pipe-clean).
struct RenderedTable {
    table: String,
    notes: Vec<String>,
}

/// Render models grouped into per-modality sections, each with the columns
/// relevant to that modality. Uses only data from the `/models` response.
/// Returns the table text plus footnotes rather than printing, so the layout
/// can be unit-tested.
fn render_sectioned_table(
    models: &[openrouter::Model],
    video_prices: &std::collections::HashMap<String, String>,
) -> RenderedTable {
    use std::fmt::Write as _;

    // Section order shown to the user.
    const SECTIONS: [&str; 8] = [
        "text",
        "image",
        "video",
        "audio",
        "speech",
        "transcription",
        "embeddings",
        "rerank",
    ];
    let mut buf = String::new();
    let mut video_note = false;
    let mut image_note = false;

    for section in SECTIONS {
        let rows: Vec<&openrouter::Model> = models
            .iter()
            .filter(|m| primary_modality(m) == section)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let _ = writeln!(buf, "\n== {} ==", section.to_uppercase());

        match section {
            // Image models bill two ways: per token (gemini, gpt-image) and/or
            // per generated image. Show both so neither cost is hidden.
            "image" => {
                let _ = writeln!(
                    buf,
                    "{:<44} {:>7}  {:>11}  {:>11}  {:>10}",
                    "ID", "CONTEXT", "IN($/1M)", "OUT($/1M)", "$/IMG"
                );
                for m in rows {
                    let p = m.pricing.as_ref();
                    let in_ = per_million(&p.and_then(|x| x.prompt.clone()));
                    let out = per_million(&p.and_then(|x| x.completion.clone()));
                    // Prefer per-output-image price; fall back to the `image` field.
                    let img = dollars(
                        &p.and_then(|x| x.image_output.clone().or_else(|| x.image.clone())),
                    );
                    // No per-image price AND no real token price => price not in
                    // /models (these are per-image billed; see /endpoints).
                    let no_token =
                        matches!(in_.as_str(), "-" | "0") && matches!(out.as_str(), "-" | "0");
                    if img == "-" && no_token {
                        image_note = true;
                    }
                    let _ = writeln!(
                        buf,
                        "{:<44} {:>7}  {:>11}  {:>11}  {:>10}",
                        m.id,
                        human_context(m.context_length),
                        in_,
                        out,
                        img
                    );
                }
            }
            // Video models: per-second/per-token price from /videos/models.
            "video" => {
                video_note = true;
                let _ = writeln!(buf, "{:<48}  PRICE *", "ID");
                for m in rows {
                    let price = video_prices.get(&m.id).map(String::as_str).unwrap_or("-");
                    let _ = writeln!(buf, "{:<48}  {}", m.id, price);
                }
            }
            // Everything else is token-billed: show in/out per 1M tokens.
            _ => {
                let _ = writeln!(
                    buf,
                    "{:<48} {:>8}  {:>12}  {:>12}",
                    "ID", "CONTEXT", "IN($/1M)", "OUT($/1M)"
                );
                for m in rows {
                    let p = m.pricing.as_ref();
                    let in_ = per_million(&p.and_then(|x| x.prompt.clone()));
                    let out = per_million(&p.and_then(|x| x.completion.clone()));
                    let _ = writeln!(
                        buf,
                        "{:<48} {:>8}  {:>12}  {:>12}",
                        m.id,
                        human_context(m.context_length),
                        in_,
                        out
                    );
                }
            }
        }
    }

    let mut notes = Vec::new();
    if video_note {
        notes.push(
            "* video pricing from /videos/models; units vary: /s = per second, \
             /vid-tok = per video token."
                .to_string(),
        );
    }
    if image_note {
        notes.push(
            "Note: some image models don't expose pricing in /models (shown as -); \
             see the per-endpoint detail or the model page for their per-image rate."
                .to_string(),
        );
    }
    RenderedTable { table: buf, notes }
}

async fn run_models(args: ModelsArgs) -> anyhow::Result<()> {
    let client = OpenRouterClient::from_env()?;
    let query = ModelsQuery {
        q: args.query,
        output_modalities: args.output_modalities,
        input_modalities: args.input_modalities,
        supported_parameters: args.supported_parameters,
        sort: Some(args.sort.unwrap_or_else(|| "top-weekly".to_string())),
        context: args.min_context,
    };

    let raw = client.list_models(&query).await?;
    let filtered = openrouter::apply_filters(raw, args.search.as_deref(), args.all);
    let (models, total) = (filtered.models, filtered.total);

    if !args.table {
        // Default: JSON, matching the `list_models` MCP tool output.
        println!("{}", serde_json::to_string_pretty(&models)?);
        if !args.all && total > models.len() {
            eprintln!(
                "\nshowing {} of {} models; pass --all to see the rest",
                models.len(),
                total
            );
        }
        return Ok(());
    }

    // Enrich the VIDEO section with real per-second pricing from /videos/models
    // (one extra call, only when the result actually contains video models).
    let mut video_prices: std::collections::HashMap<String, String> = Default::default();
    if models.iter().any(|m| primary_modality(m) == "video") {
        match client.list_video_models().await {
            Ok(vms) => {
                for vm in vms {
                    video_prices.insert(vm.id, video_price(&vm.pricing_skus));
                }
            }
            Err(e) => eprintln!("warning: could not fetch video pricing: {e}"),
        }
    }

    let rendered = render_sectioned_table(&models, &video_prices);
    print!("{}", rendered.table);
    for note in &rendered.notes {
        eprintln!("\n{note}");
    }

    if !args.all && total > models.len() {
        eprintln!(
            "\nshowing {} of {} models; pass --all to see the rest",
            models.len(),
            total
        );
    } else {
        eprintln!("\n{} models", models.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::openrouter::{Architecture, Model, Pricing};

    fn model_with_outputs(id: &str, output_modalities: &[&str]) -> Model {
        Model {
            id: id.to_string(),
            name: None,
            description: None,
            context_length: None,
            architecture: Some(Architecture {
                modality: None,
                input_modalities: Vec::new(),
                output_modalities: output_modalities.iter().map(|s| s.to_string()).collect(),
                tokenizer: None,
            }),
            pricing: None,
        }
    }

    #[test]
    fn per_million_formats_prices_and_sentinels() {
        assert_eq!(per_million(&Some("0".to_string())), "0");
        assert_eq!(per_million(&Some("-1".to_string())), "-");
        assert_eq!(per_million(&Some("0.00000075".to_string())), "$0.75");
        assert_eq!(per_million(&Some("0.00003".to_string())), "$30");
        assert_eq!(per_million(&Some("not-a-number".to_string())), "-");
        assert_eq!(per_million(&None), "-");
    }

    #[test]
    fn human_context_uses_compact_units() {
        assert_eq!(human_context(None), "-");
        assert_eq!(human_context(Some(0)), "-");
        assert_eq!(human_context(Some(128_000)), "128K");
        assert_eq!(human_context(Some(131_072)), "128K");
        assert_eq!(human_context(Some(1_050_000)), "1.05M");
        assert_eq!(human_context(Some(1_048_576)), "1M");
        assert_eq!(human_context(Some(999)), "999");
    }

    #[test]
    fn video_price_prefers_seconds_then_cents_then_video_tokens() {
        let mut skus = BTreeMap::new();
        skus.insert("duration_seconds".to_string(), "0.12".to_string());
        skus.insert("video_tokens".to_string(), "0.01".to_string());
        assert_eq!(video_price(&skus), "$0.12/s");

        let mut skus = BTreeMap::new();
        skus.insert("second_with_audio".to_string(), "3".to_string());
        skus.insert("second_without_audio".to_string(), "2".to_string());
        assert_eq!(video_price(&skus), "$0.02-0.03/s");

        let mut skus = BTreeMap::new();
        skus.insert("video_tokens".to_string(), "0.0002".to_string());
        assert_eq!(video_price(&skus), "$0.0002/vid-tok");

        assert_eq!(video_price(&BTreeMap::new()), "-");
    }

    #[test]
    fn trim_num_drops_trailing_zeros_and_caps_precision() {
        assert_eq!(trim_num(1.0), "1");
        assert_eq!(trim_num(1.50), "1.5");
        assert_eq!(trim_num(0.12000000), "0.12");
        // Capped at 8 decimal places.
        assert_eq!(trim_num(0.123456789), "0.12345679");
    }

    #[test]
    fn range_str_collapses_equal_bounds_and_renders_ranges() {
        assert_eq!(range_str(&[0.02], "/s"), "$0.02/s");
        // Equal min/max collapse to a single value.
        assert_eq!(range_str(&[0.5, 0.5], "/s"), "$0.5/s");
        assert_eq!(range_str(&[0.02, 0.03], "/s"), "$0.02-0.03/s");
        // Order-independent: min and max are derived, not positional.
        assert_eq!(range_str(&[0.03, 0.02], "/s"), "$0.02-0.03/s");
    }

    #[test]
    fn dollars_passes_through_positive_and_blanks_the_rest() {
        assert_eq!(dollars(&Some("0.04".to_string())), "$0.04");
        assert_eq!(dollars(&Some("0".to_string())), "-");
        assert_eq!(dollars(&Some("-1".to_string())), "-");
        assert_eq!(dollars(&Some("nope".to_string())), "-");
        assert_eq!(dollars(&None), "-");
    }

    #[test]
    fn per_million_trims_and_handles_fractions() {
        // 0.000001 * 1e6 = 1.0 -> "$1"
        assert_eq!(per_million(&Some("0.000001".to_string())), "$1");
        // Fractional cents survive the 4-decimal render.
        assert_eq!(per_million(&Some("0.0000012345".to_string())), "$1.2345");
    }

    #[test]
    fn human_context_handles_binary_and_fractional_units() {
        assert_eq!(human_context(Some(2048)), "2K");
        assert_eq!(human_context(Some(1500)), "1.5K");
        assert_eq!(human_context(Some(1_000_000)), "1M");
        assert_eq!(human_context(Some(2_500_000)), "2.5M");
    }

    #[test]
    fn video_price_renders_single_cents_value() {
        let mut skus = std::collections::BTreeMap::new();
        skus.insert("second_with_audio".to_string(), "5".to_string());
        // 5 cents/second -> $0.05/s
        assert_eq!(video_price(&skus), "$0.05/s");
    }

    #[test]
    fn render_sectioned_table_groups_sections_and_emits_notes() {
        use std::collections::HashMap;

        // A token-billed text model with real pricing.
        let mut text = model_with_outputs("openai/gpt", &["text"]);
        text.context_length = Some(128_000);
        text.pricing = Some(Pricing {
            prompt: Some("0.000001".to_string()),
            completion: Some("0.000002".to_string()),
            ..Default::default()
        });
        // An image model with no pricing exposed -> should trigger the image note.
        let image = model_with_outputs("blackforest/flux", &["image"]);
        // A video model; its price comes from the side map.
        let video = model_with_outputs("google/veo", &["video"]);

        let mut video_prices = HashMap::new();
        video_prices.insert("google/veo".to_string(), "$0.1/s".to_string());

        let rendered = render_sectioned_table(&[text, image, video], &video_prices);

        // Sections appear, in the data we gave.
        assert!(rendered.table.contains("== TEXT =="));
        assert!(rendered.table.contains("== IMAGE =="));
        assert!(rendered.table.contains("== VIDEO =="));
        // Row content is rendered.
        assert!(rendered.table.contains("openai/gpt"));
        assert!(rendered.table.contains("128K"));
        assert!(rendered.table.contains("$0.1/s"));
        // Both footnotes fire: video is always noted, image has no pricing.
        assert_eq!(rendered.notes.len(), 2);
        assert!(rendered.notes.iter().any(|n| n.contains("video pricing")));
        assert!(
            rendered
                .notes
                .iter()
                .any(|n| n.contains("don't expose pricing"))
        );
    }

    #[test]
    fn primary_modality_prioritizes_media_outputs() {
        assert_eq!(
            primary_modality(&model_with_outputs("text", &["text"])),
            "text"
        );
        assert_eq!(
            primary_modality(&model_with_outputs("image", &["text", "image"])),
            "image"
        );
        assert_eq!(
            primary_modality(&model_with_outputs("video", &["image", "video"])),
            "video"
        );
        assert_eq!(
            primary_modality(&model_with_outputs("rerank", &["rerank"])),
            "rerank"
        );

        let no_arch = Model {
            id: "none".to_string(),
            name: None,
            description: None,
            context_length: None,
            architecture: None,
            pricing: None,
        };
        assert_eq!(primary_modality(&no_arch), "text");
    }
}
