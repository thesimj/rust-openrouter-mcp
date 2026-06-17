//! `openrouter-mcp` - an MCP (stdio) server for OpenRouter, doubling as a CLI.
//!
//! Run the MCP server with: `openrouter-mcp` (or `openrouter-mcp mcp`)
//! Or use it directly, e.g.: `openrouter-mcp models --output-modalities image --sort newest`
//!
//! Requires the `OPENROUTER_API_KEY` environment variable (or a local `.env`).

mod audio_gen;
mod cli;
mod image_gen;
mod image_io;
mod manifest;
mod openrouter;
mod pricing;
mod server;
mod stats;
mod tasks;
mod video_gen;

use clap::Parser;

use cli::Cli;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a local `.env` file if present (does not override real env vars).
    // Key resolution is therefore: real env var > .env entry > error in from_env().
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();
    cli::dispatch(cli).await
}
