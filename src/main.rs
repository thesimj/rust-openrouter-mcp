//! `openrouter-mcp` - an MCP (stdio) server for OpenRouter.
//!
//! Run the MCP server with `openrouter-mcp` (or `openrouter-mcp mcp`).
//! Print the version with `openrouter-mcp --version` or `openrouter-mcp -V`.
//!
//! Requires the `OPENROUTER_API_KEY` environment variable (or a local `.env`).

mod audio_gen;
mod billing;
mod chat_gen;
mod embed_gen;
mod image_gen;
mod image_io;
mod manifest;
mod music_gen;
mod openrouter;
mod output;
mod pricing;
mod resources;
mod server;
mod stats;
mod tasks;
mod video_gen;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    match (args.next(), args.next()) {
        // `mcp` is the historical subcommand; clients configured with `args: ["mcp"]` keep working.
        (None, None) => {}
        (Some(arg), None) if arg == "mcp" => {}
        (Some(arg), None) if arg == "--version" || arg == "-V" => {
            println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        _ => {
            eprintln!(
                "Use no arguments (or `mcp`) for MCP, or --version / -V to print the version."
            );
            std::process::exit(2);
        }
    }

    // Load a local `.env` file if present (does not override real env vars).
    // Key resolution is therefore: real env var > .env entry > error in from_env().
    let _ = dotenvy::dotenv();

    server::run().await
}
