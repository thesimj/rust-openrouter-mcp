//! Per-command handlers for the CLI subcommands.

use super::table::{primary_modality, render_sectioned_table};
use super::{
    AudioArgs, ChatArgs, DescribeArgs, ImageArgs, ModelsArgs, VideoArgs, parse_image_arg,
    resolve_base_output, resolve_prompt,
};
use crate::image_gen::GenerateRequest;
use crate::openrouter::{ModelsQuery, OpenRouterClient};
use crate::pricing::{models_to_json, video_price};
use crate::{audio_gen, chat_gen, image_gen, openrouter, video_gen};

/// Print the "showing N of total / N models" footer shared by both `run_models`
/// output paths (JSON and table).
fn print_model_count_footer(shown: usize, total: usize, all: bool) {
    if !all && total > shown {
        eprintln!("\nshowing {shown} of {total} models; pass --all to see the rest");
    } else {
        eprintln!("\n{shown} models");
    }
}

/// Print a job's `note:`/`error:` lines then its `manifest:` path to stderr,
/// the trailer shared by the image and video CLI handlers.
fn print_job_notes(warnings: &[String], errors: &[String], manifest: &std::path::Path) {
    for warning in warnings {
        eprintln!("note: {warning}");
    }
    for error in errors {
        eprintln!("error: {error}");
    }
    eprintln!("manifest: {}", manifest.display());
}

/// Fetch and print basic info about the API key in use (`GET /api/v1/key`).
pub(crate) async fn run_key() -> anyhow::Result<()> {
    let client = OpenRouterClient::from_env()?;
    let info = client.get_key_info().await?;

    // `limit`/`limit_remaining` are null for unlimited keys.
    let usd = |o: Option<f64>| o.map_or_else(|| "unlimited".to_string(), |v| format!("${v}"));
    let flag = |o: Option<bool>| if o.unwrap_or(false) { "yes" } else { "no" };

    println!("label:            {}", info.label.as_deref().unwrap_or("-"));
    println!(
        "owner (user id):  {}",
        info.creator_user_id.as_deref().unwrap_or("-")
    );
    println!("free tier:        {}", flag(info.is_free_tier));
    println!("provisioning key: {}", flag(info.is_provisioning_key));
    println!("management key:   {}", flag(info.is_management_key));
    println!("limit:            {}", usd(info.limit));
    println!("limit remaining:  {}", usd(info.limit_remaining));
    println!("usage (total):    ${}", info.usage.unwrap_or(0.0));
    println!(
        "usage day/wk/mo:  ${} / ${} / ${}",
        info.usage_daily.unwrap_or(0.0),
        info.usage_weekly.unwrap_or(0.0),
        info.usage_monthly.unwrap_or(0.0)
    );
    if let Some(byok) = info.byok_usage.filter(|v| *v > 0.0) {
        println!("byok usage:       ${byok}");
    }
    Ok(())
}

/// Describe local image(s) with a vision model and print the text to stdout.
pub(crate) async fn run_describe(args: DescribeArgs) -> anyhow::Result<()> {
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

/// Send a prompt to a chat/text model and print the reply to stdout (cost to
/// stderr). Mirrors the `chat_completion` MCP tool.
pub(crate) async fn run_chat(args: ChatArgs) -> anyhow::Result<()> {
    let client = OpenRouterClient::from_env()?;
    let (prompt, _source) = resolve_prompt(args.prompt, args.prompt_file)?;

    let result = chat_gen::complete(
        &client,
        &args.model,
        args.system.as_deref(),
        &prompt,
        args.temperature,
        args.max_tokens,
    )
    .await?;
    println!("{}", result.text);
    if let Some(cost) = result.cost {
        eprintln!("cost: ${cost}");
    }
    Ok(())
}

/// Generate one or more images and save them, plus a sidecar manifest. The CLI
/// blocks until all variants finish (run in parallel).
pub(crate) async fn run_image(args: ImageArgs) -> anyhow::Result<()> {
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
    print_job_notes(&summary.warnings, &summary.errors, &summary.manifest_path);

    if summary.images.is_empty() {
        anyhow::bail!("all {variants} variant(s) failed");
    }
    // stdout: the saved paths, for scripting.
    for img in &summary.images {
        println!("{}", img.path.display());
    }
    Ok(())
}

/// Generate a video and save it, plus a sidecar manifest. The CLI blocks
/// synchronously through the submit + poll loop (unlike the async MCP tool).
pub(crate) async fn run_video(args: VideoArgs) -> anyhow::Result<()> {
    let client = OpenRouterClient::from_env()?;
    let (prompt, prompt_source) = resolve_prompt(args.prompt, args.prompt_file)?;
    let base = resolve_base_output(args.output, args.output_dir, args.output_name)?;

    let mut frames = Vec::new();
    if let Some(p) = args.first_frame {
        frames.push(video_gen::VideoInput {
            path: p,
            frame_type: "first_frame".to_string(),
        });
    }
    if let Some(p) = args.last_frame {
        frames.push(video_gen::VideoInput {
            path: p,
            frame_type: "last_frame".to_string(),
        });
    }

    let req = video_gen::VideoGenRequest {
        model: args.model,
        prompt,
        duration: args.duration,
        resolution: args.resolution,
        aspect_ratio: args.aspect_ratio,
        size: args.size,
        generate_audio: Some(args.generate_audio),
        seed: args.seed,
        frames,
        references: args
            .reference_images
            .iter()
            .map(std::path::PathBuf::from)
            .collect(),
        max_image_dimension: image_gen::resolve_max_dimension(args.max_image_dimension),
        poll_interval_secs: video_gen::resolve_poll_interval(None),
        poll_timeout_secs: video_gen::resolve_poll_timeout(None),
    };

    let summary = video_gen::run_job(&client, &req, &base, &prompt_source).await?;

    for v in &summary.videos {
        eprintln!(
            "saved {}{}",
            v.path.display(),
            if v.has_audio { " (with audio)" } else { "" },
        );
    }
    print_job_notes(&summary.warnings, &summary.errors, &summary.manifest_path);

    if summary.videos.is_empty() {
        anyhow::bail!("no video clip was produced");
    }
    for v in &summary.videos {
        println!("{}", v.path.display());
    }
    Ok(())
}

/// Generate speech and save it, plus a sidecar manifest.
pub(crate) async fn run_audio(args: AudioArgs) -> anyhow::Result<()> {
    let client = OpenRouterClient::from_env()?;
    let (input, input_source) = resolve_prompt(args.input, args.input_file)?;

    let req = audio_gen::SpeechGenRequest {
        model: args.model,
        input,
        voice: args.voice,
        response_format: args.response_format,
        speed: args.speed,
    };

    let result = audio_gen::run_job(&client, &req, &args.output, &input_source).await?;

    eprintln!("voice: {}", result.audio.voice);
    for warning in &result.warnings {
        eprintln!("note: {warning}");
    }
    eprintln!("manifest: {}", result.manifest_path.display());
    println!("{}", result.audio.path.display());
    Ok(())
}

pub(crate) async fn run_models(args: ModelsArgs) -> anyhow::Result<()> {
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
        // Default: JSON, matching the `list_models` MCP tool output (same
        // shared enrichment, so the two never diverge).
        println!(
            "{}",
            serde_json::to_string_pretty(&models_to_json(&models))?
        );
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

    print_model_count_footer(models.len(), total, args.all);
    Ok(())
}
