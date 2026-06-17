# rust-openrouter-mcp

MCP (stdio) server **and** CLI for [OpenRouter](https://openrouter.ai), in a
single Rust binary. Discover models, generate and edit images (with parallel
variants and a sidecar manifest), describe images with a vision model, and track
per-process usage - all behind one `openrouter-mcp` executable.

## Features

- **Model discovery** - `list_models` with server-side filters (modality,
  supported params, sort, min context), local search, and pricing; `describe_model`
  returns full detail for one model id (architecture, context, benchmarks, and
  per-provider endpoints with pricing - including real `pricing_skus` for video).
- **Image generation** - `generate_image`: text-to-image, image editing /
  image-to-image (multiple local inputs), and **parallel variants** (seed-stepped).
  - Input images may be PNG, JPEG, WebP, GIF, or **SVG**. SVG inputs are
    rasterized to PNG (longest side scaled to the dimension cap; transparency
    preserved). Text in SVGs is not rendered (no fonts are loaded) and is flagged
    as a warning.
  - The output format (PNG/JPEG) is **chosen by the provider** - it is sniffed
    from the response and the file extension is set to match.
  - Requested `aspect_ratio` / `image_size` are **verified** against the actual
    decoded pixels; mismatches are surfaced as warnings.
  - Every job writes a `*.manifest.json` sidecar (full settings, per-input and
    per-variant metadata, cost, provider, timing).
  - **Asynchronous**: if a job runs longer than `wait_seconds` (default 10) the
    tool returns a `task_id`; poll `get_result` for completion.
- **Video generation** - `generate_video`: text-to-video and image-to-video
  (first/last frame and reference images) with an OpenRouter video model.
  **Asynchronous**: returns a `task_id`; poll `get_result` for the hosted clip.
- **Speech generation** - `generate_audio`: text-to-speech with an OpenRouter
  TTS model (voice/format/speed); saves the audio to disk with a manifest.
- **Image description** - `describe_image`: image -> detailed text via any
  vision-capable model (image input, text output).
- **Chat completion** - `chat_completion`: send a prompt to any OpenRouter
  chat/text model and get its text reply (text in, text out) - route a sub-task
  to a different model (optional `system`, `temperature`, `max_tokens`).
- **Account info** - `get_account`: basic info about the API key in use (label,
  owning user id, credit usage with daily/weekly/monthly breakdown, spending
  limit / remaining balance, and tier / key-type flags).
- **Usage stats** - `get_usage_stats` (read-only) and `reset_usage_stats`
  (destructive, requires `confirm: true`): per-process request/cost counters with
  a by-model breakdown.

## Add to Claude Desktop (one-click)

This server ships as a **Claude Desktop extension** (`.mcpb`) for **macOS** - no
terminal or Rust toolchain required.

1. Download `openrouter-mcp.mcpb` from the
   [latest release](https://github.com/thesimj/rust-openrouter-mcp/releases/latest).
2. Double-click it (or drag it into **Claude Desktop -> Settings -> Extensions**).
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

- `auto` (default) - inline previews for every client **except** the local
  `claude-code` CLI, which shares the filesystem and can open the saved file
  directly.
- `always` - always embed previews. The Claude Desktop connector sets this,
  because Desktop runs the server in a sandboxed filesystem it can't read, so the
  saved path is unreachable and the image must come back inline.
- `never` - paths only, never inline bytes.

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
| `list_models` | read-only | List models with capabilities and pricing (server-side filters, local search; human-readable `$X/M tokens` pricing). |
| `describe_model` | read-only | Full detail for one model id: description, architecture, context, benchmarks, per-provider endpoints, and (for video models) real `pricing_skus`. |
| `generate_image` | write | Generate or edit images; supports `variants`; async with `task_id`. Inputs by `path`/`url`/`base64`. **No defaults** for `model`, `prompt`, `aspect_ratio`, `image_size`, `image_only`; `output` is optional (auto-named under `OPENROUTER_MCP_OUTPUT_DIR`). |
| `generate_video` | write | Text-to-video / image-to-video with an OpenRouter video model; async, poll by `task_id`. |
| `generate_audio` | write | Text-to-speech with an OpenRouter TTS model; saves audio to disk. |
| `chat_completion` | write | Send a prompt to any OpenRouter chat/text model and return its text reply (text in, text out); route a sub-task to a different model. |
| `describe_image` | read-only | Describe image(s) - by `path`, `url`, or `base64`/data-URL - with a vision-capable model; returns text. |
| `get_result` | read-only | Fetch a job by `task_id`: `pending` / `completed` / `failed`. |
| `get_account` | read-only | Basic info about the API key in use: label, owning user id, credit usage (total + daily/weekly/monthly), limit/remaining, and tier/key-type flags. |
| `get_usage_stats` | read-only | In-memory request/cost counters (and server `version`) with a by-model breakdown. |
| `reset_usage_stats` | destructive | Reset all counters (`confirm: true` required). |

`generate_image` returns a lean result - saved paths, decoded width/height,
requested vs. actual aspect/size, seeds, and a pointer to the sidecar manifest;
the full per-variant detail lives in the manifest on disk.

## CLI usage

The same binary is a CLI. Subcommands: `models`, `image`, `video`, `audio`,
`describe`, `chat`, `key`, `mcp`.

Show info about the API key in use:

```bash
openrouter-mcp key
```

Send a prompt to a chat/text model:

```bash
openrouter-mcp chat --model openai/gpt-5.4 --prompt "Summarize MCP in one sentence."
openrouter-mcp chat -m anthropic/claude-sonnet-4.6 -s "Be terse." -p "Why Rust?" --temperature 0.3
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

Generate a video (blocks through submit + poll; text-to-video or
image-to-video via `--first-frame` / `--last-frame`):

```bash
openrouter-mcp video \
  --model bytedance/seedance-2.0 \
  --prompt "a paper boat drifting down a rain-soaked street, cinematic" \
  --duration 8 --resolution 1080p --output ./out/boat.mp4
```

Generate speech (text-to-speech):

```bash
openrouter-mcp audio \
  --model openai/gpt-4o-mini-tts --voice alloy \
  --input "Hello from OpenRouter." --output ./out/hello.mp3
```

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
fulfill the requests you make (model discovery, image/video/speech generation,
image description, and chat completion). Your API key is sent solely to
OpenRouter to authenticate those calls. Generated files are written only to
paths you specify (or an auto-named path under `OPENROUTER_MCP_OUTPUT_DIR`);
usage stats live in memory and are lost on exit. Full details:
[PRIVACY.md](PRIVACY.md).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
