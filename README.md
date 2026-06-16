# rust-openrouter-mcp

MCP (stdio) server for OpenRouter: chat with LLMs, generate images, and generate
videos from a single Rust binary.

Version `0.0.1` is the first test slice. It starts the MCP server and exposes
model discovery only, so agents can inspect OpenRouter models, capabilities, and
pricing before generation tools are added.

## Current Features

- MCP stdio server with a `list_models` tool.
- CLI model browser with JSON or table output.
- Server-side OpenRouter filters for query, modality, supported parameters,
  sort order, and minimum context length.
- Local search across model id, name, and description.
- Pricing display for text, image, audio, and video model listings.

## Planned Tools

- `chat_completion` for OpenRouter text and vision models.
- `generate_image` for text-to-image and image editing models.
- `generate_video` and `get_video_status` for async video generation.
- Audio, reranking, validation, and model-detail helpers.

## Install

From crates.io, after publishing:

```bash
cargo install openrouter-mcp
```

From a local checkout:

```bash
cargo install --path . --locked --force
```

## Configuration

Set an OpenRouter API key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

On PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

A local `.env` file is also loaded if present:

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

Do not commit `.env`.

## MCP Usage

Start the MCP stdio server:

```bash
openrouter-mcp
```

The explicit subcommand also works:

```bash
openrouter-mcp mcp
```

Example MCP client config:

```json
{
  "mcpServers": {
    "openrouter": {
      "command": "openrouter-mcp",
      "env": {
        "OPENROUTER_API_KEY": "sk-or-v1-..."
      }
    }
  }
}
```

If your client already provides `OPENROUTER_API_KEY` in the process
environment, the `env` block is optional.

## CLI Usage

List popular text models as JSON:

```bash
openrouter-mcp models
```

Show a table of recent OpenAI models:

```bash
openrouter-mcp models --query openai --sort newest --table
```

List image-capable models:

```bash
openrouter-mcp models --output-modalities image --sort newest --table
```

Return every matching model instead of the first 20:

```bash
openrouter-mcp models --query claude --all
```

Useful filters:

```bash
openrouter-mcp models --supported-parameters tools --min-context 128000
openrouter-mcp models --input-modalities image --output-modalities text
openrouter-mcp models --search codex
```

## MCP Tool

### `list_models`

Lists OpenRouter models with capabilities and pricing.

Arguments:

- `query`: server-side free-text search by model name or slug.
- `search`: local case-insensitive search across id, name, and description.
- `output_modalities`: comma-separated output modalities, such as `text`,
  `image`, `audio`, `embeddings`, `video`, `rerank`, `speech`,
  `transcription`, or `all`.
- `input_modalities`: comma-separated input modalities, such as `text`,
  `image`, `audio`, or `file`.
- `supported_parameters`: comma-separated required parameters, such as `tools`,
  `structured_outputs`, or `reasoning`.
- `sort`: `pricing-low-to-high`, `pricing-high-to-low`,
  `context-high-to-low`, `throughput-high-to-low`, `latency-low-to-high`,
  `most-popular`, `top-weekly`, or `newest`.
- `min_context`: minimum context length in tokens.
- `all`: return all matching models. Defaults to `false`, which returns the
  first 20.

## Development

Run the local checks:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo package --allow-dirty
```

Run a live OpenRouter smoke test:

```bash
cargo run -- models --query openai --sort newest --table
```
