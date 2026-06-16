//! `openrouter-mcp` — an MCP (stdio) server for OpenRouter, doubling as a CLI.
//!
//! Run the MCP server with: `openrouter-mcp` (or `openrouter-mcp mcp`)
//! Or use it directly, e.g.: `openrouter-mcp models --output-modalities image --sort newest`
//!
//! Requires the `OPENROUTER_API_KEY` environment variable (or a local `.env`).

mod openrouter;
mod server;

use clap::{Parser, Subcommand};

use openrouter::{ModelsQuery, OpenRouterClient};

#[derive(Parser)]
#[command(
    name = "openrouter-mcp",
    version,
    about = "MCP (stdio) server for OpenRouter — chat, image & video generation"
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

/// Default number of models returned unless `all` is requested.
const DEFAULT_MODEL_LIMIT: usize = 20;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a local `.env` file if present (does not override real env vars).
    // Key resolution is therefore: real env var > .env entry > error in from_env().
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();
    match cli.command {
        Some(Command::Models(args)) => run_models(args).await,
        Some(Command::Mcp) | None => server::run().await,
    }
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

/// Print models grouped into per-modality sections, each with the columns
/// relevant to that modality. Uses only data from the `/models` response.
fn print_sectioned_table(
    models: &[openrouter::Model],
    video_prices: &std::collections::HashMap<String, String>,
) {
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
        println!("\n== {} ==", section.to_uppercase());

        match section {
            // Image models bill two ways: per token (gemini, gpt-image) and/or
            // per generated image. Show both so neither cost is hidden.
            "image" => {
                println!(
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
                    println!(
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
                println!("{:<48}  PRICE *", "ID");
                for m in rows {
                    let price = video_prices.get(&m.id).map(String::as_str).unwrap_or("-");
                    println!("{:<48}  {}", m.id, price);
                }
            }
            // Everything else is token-billed: show in/out per 1M tokens.
            _ => {
                println!(
                    "{:<48} {:>8}  {:>12}  {:>12}",
                    "ID", "CONTEXT", "IN($/1M)", "OUT($/1M)"
                );
                for m in rows {
                    let p = m.pricing.as_ref();
                    let in_ = per_million(&p.and_then(|x| x.prompt.clone()));
                    let out = per_million(&p.and_then(|x| x.completion.clone()));
                    println!(
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

    if video_note {
        eprintln!(
            "\n* video pricing from /videos/models; units vary: /s = per second, \
             /vid-tok = per video token."
        );
    }
    if image_note {
        eprintln!(
            "\nNote: some image models don't expose pricing in /models (shown as -); \
             see the per-endpoint detail or the model page for their per-image rate."
        );
    }
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

    let mut models = client.list_models(&query).await?;
    if let Some(needle) = &args.search {
        models.retain(|m| m.matches_search(needle));
    }
    let total = models.len();
    if !args.all {
        models.truncate(DEFAULT_MODEL_LIMIT);
    }

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

    print_sectioned_table(&models, &video_prices);

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
    use crate::openrouter::{Architecture, Model};

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
