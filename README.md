# rust-openrouter-mcp

MCP (stdio) server **and** CLI for [OpenRouter](https://openrouter.ai), in a
single Rust binary. Discover models, generate and edit images (with parallel
variants and a sidecar manifest), describe images with a vision model, and track
per-process usage — all behind one `openrouter-mcp` executable.

## Features

- **Model discovery** — `list_models` with server-side filters (modality,
  supported params, sort, min context), local search, and pricing.
- **Image generation** — `generate_image`: text-to-image, image editing /
  image-to-image (multiple local inputs), and **parallel variants** (seed-stepped).
  - Input images may be PNG, JPEG, WebP, GIF, or **SVG**. SVG inputs are
    rasterized to PNG (longest side scaled to the dimension cap; transparency
    preserved). Text in SVGs is not rendered (no fonts are loaded) and is flagged
    as a warning.
  - The output format (PNG/JPEG) is **chosen by the provider** — it is sniffed
    from the response and the file extension is set to match.
  - Requested `aspect_ratio` / `image_size` are **verified** against the actual
    decoded pixels; mismatches are surfaced as warnings.
  - Every job writes a `*.manifest.json` sidecar (full settings, per-input and
    per-variant metadata, cost, provider, timing).
  - **Asynchronous**: if a job runs longer than `wait_seconds` (default 10) the
    tool returns a `task_id`; poll `get_result` for completion.
- **Image description** — `describe_image`: image → detailed text via any
  vision-capable model (image input, text output).
- **Account info** — `get_account`: basic info about the API key in use (label,
  owning user id, credit usage with daily/weekly/monthly breakdown, spending
  limit / remaining balance, and tier / key-type flags).
- **Usage stats** — `get_usage_stats` (read-only) and `reset_usage_stats`
  (destructive, requires `confirm: true`): per-process request/cost counters with
  a by-model breakdown.

## Add to Claude Desktop (one-click)

This server ships as a **Claude Desktop extension** (`.mcpb`) for **macOS** — no
terminal or Rust toolchain required.

1. Download `openrouter-mcp.mcpb` from the
   [latest release](https://github.com/thesimj/rust-openrouter-mcp/releases/latest).
2. Double-click it (or drag it into **Claude Desktop → Settings → Extensions**).
3. Click **Install**, paste your
   [OpenRouter API key](https://openrouter.ai/keys), and enable it.

Your API key is stored in the macOS keychain by Claude Desktop and injected into
the server as `OPENROUTER_API_KEY`. See [Privacy Policy](#privacy-policy).

To build the bundle yourself (produces a universal arm64+x86_64 binary):

```bash
scripts/build-mcpb.sh        # -> dist/openrouter-mcp.mcpb
```

## Connect another client (CLI, IDE, agent)

`openrouter-mcp` is a universal local stdio MCP server. **[CONNECT.md](CONNECT.md)**
has copy-paste setup for Claude Code, Codex CLI, Gemini CLI, Antigravity, Cursor,
Windsurf, VS Code (Copilot), Zed, Cline, Roo Code, Continue, Goose, opencode,
Crush, Amp, OpenHands, and more.

## Install (CLI / other MCP clients)

From a local checkout:

```bash
cargo install --path . --locked --force
```

From crates.io (once published):

```bash
cargo install openrouter-mcp
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

A local `.env` file is also loaded if present (real env vars take precedence):

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

Do not commit `.env`.

Optional: `OPENROUTER_IMAGE_MAX_DIMENSION` (default `800`) caps the longest side
of input images before they are sent (raster inputs are downscaled to reduce
request size/cost; SVG inputs are rasterized with their longest side scaled to
this cap).

Optional: `OPENROUTER_MCP_IMAGE_PREVIEWS` controls whether `generate_image` /
`get_result` embed the generated image **inline** (base64) in the tool result, in
addition to saving it to disk:

- `auto` (default) — inline previews for every client **except** the local
  `claude-code` CLI, which shares the filesystem and can open the saved file
  directly.
- `always` — always embed previews. The Claude Desktop connector sets this,
  because Desktop runs the server in a sandboxed filesystem it can't read, so the
  saved path is unreachable and the image must come back inline.
- `never` — paths only, never inline bytes.

Inline previews are downscaled to a 1568px longest side; the full-resolution
image is always the file saved on disk.

## MCP usage

Start the stdio server (`mcp` subcommand is implied when none is given):

```bash
openrouter-mcp        # or: openrouter-mcp mcp
```

Example MCP client config:

```json
{
  "mcpServers": {
    "openrouter": {
      "command": "openrouter-mcp",
      "env": { "OPENROUTER_API_KEY": "sk-or-v1-..." }
    }
  }
}
```

If the client already provides `OPENROUTER_API_KEY` in the environment, the `env`
block is optional.

### MCP tools

| Tool | Kind | Description |
| --- | --- | --- |
| `list_models` | read-only | List models with capabilities and pricing (server-side filters, local search). |
| `generate_image` | write | Generate or edit images; supports `variants`; async with `task_id`. **No defaults** — `model`, `prompt`, `output`, `aspect_ratio`, `image_size`, and `image_only` must all be set. |
| `get_result` | read-only | Fetch a job by `task_id`: `pending` / `completed` / `failed`. |
| `describe_image` | read-only | Describe local image(s) with a vision-capable model; returns text. |
| `get_account` | read-only | Basic info about the API key in use: label, owning user id, credit usage (total + daily/weekly/monthly), limit/remaining, and tier/key-type flags. |
| `get_usage_stats` | read-only | In-memory request/cost counters with a by-model breakdown. |
| `reset_usage_stats` | destructive | Reset all counters (`confirm: true` required). |

`generate_image` returns a lean result — saved paths, decoded width/height,
requested vs. actual aspect/size, seeds, and a pointer to the sidecar manifest;
the full per-variant detail lives in the manifest on disk.

## CLI usage

The same binary is a CLI. Subcommands: `models`, `image`, `describe`, `key`, `mcp`.

Show info about the API key in use:

```bash
openrouter-mcp key
```

Browse models:

```bash
openrouter-mcp models --query openai --sort newest --table
openrouter-mcp models --output-modalities image --sort newest --table
openrouter-mcp models --query openai --search codex
openrouter-mcp models --query claude --all
```

Generate an image:

```bash
openrouter-mcp image \
  --model google/gemini-3.1-flash-image-preview \
  --prompt "a photorealistic owl with one cybernetic eye, starry sky" \
  --aspect-ratio 1:1 --image-size 1K --seed 1200 \
  --output ./out/owl.png
```

Edit / image-to-image (repeatable `--image`, optional `label=path`):

```bash
openrouter-mcp image \
  --model google/gemini-3.1-flash-image-preview \
  --prompt "add a small wizard hat" \
  --image ./out/owl.png \
  --output ./out/owl-hat.png
```

Four parallel variants (files named `*-var-<seed>.<ext>` plus a manifest):

```bash
openrouter-mcp image -m bytedance-seed/seedream-4.5 --image-only \
  --prompt "a cute pixar-style baby dragon" \
  --aspect-ratio 1:1 --image-size 1K --seed 1490 --variants 4 \
  --output ./out/dragon.png
```

Describe an image:

```bash
openrouter-mcp describe -m google/gemini-2.5-flash-lite --image ./out/owl.png
```

The image format the provider returns is not guaranteed; the CLI corrects the
saved file's extension to match what actually came back.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo llvm-cov --summary-only        # coverage (cargo install cargo-llvm-cov)
```

Live smoke tests (require `OPENROUTER_API_KEY`):

```bash
cargo run -- models --query openai --sort newest --table
cargo run -- describe -m google/gemini-2.5-flash-lite --image ./some.png
```

## Privacy Policy

`openrouter-mcp` runs entirely on your machine and collects no telemetry. The
only third party it contacts is [OpenRouter](https://openrouter.ai), and only to
fulfill the requests you make (model discovery, image generation/description).
Your API key is sent solely to OpenRouter to authenticate those calls. Generated
images are written only to paths you specify; usage stats live in memory and are
lost on exit. Full details: [PRIVACY.md](PRIVACY.md).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
