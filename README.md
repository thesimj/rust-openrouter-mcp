# rust-openrouter-mcp

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/thesimj/rust-openrouter-mcp#license)
[![Release MCPB](https://github.com/thesimj/rust-openrouter-mcp/actions/workflows/release-mcpb.yml/badge.svg)](https://github.com/thesimj/rust-openrouter-mcp/actions/workflows/release-mcpb.yml)
[![MSRV](https://img.shields.io/badge/MSRV-1.88-blue.svg)](https://github.com/thesimj/rust-openrouter-mcp/blob/main/Cargo.toml)

<p align="center">
  <img src="assets/hero.jpg" alt="rust-openrouter-mcp - one Rust binary routing an AI assistant to OpenRouter's models" width="100%">
</p>

One small Rust program that gives your AI assistant access to every model on
[OpenRouter](https://openrouter.ai). It runs as an MCP server for Claude
Desktop, Claude Code, Cursor and other clients, and it doubles as a
command-line tool. Bring your own OpenRouter API key.

## What you can do with it

- **Find models.** Search the catalog by what a model can do, what it costs,
  who serves it, and how well it scores.
- **Make images.** Text-to-image and image editing with any image model.
  Several variants in parallel. Each result gets a sidecar manifest.
- **Make video.** Text-to-video, image-to-video from a first or last frame,
  or reference-to-video from images, audio or clips.
- **Make speech and music.** Text-to-speech, voice cloning from a sample, and
  music from a prompt (Google Lyria).
- **Transcribe audio.** Speech-to-text, with speaker labels when the provider
  supports them.
- **Talk to any chat model.** Send a prompt with images, PDFs, audio or video
  attached. Ask for JSON output. Turn on web search.
- **Embed and rerank text.** Vectors for search, and best-first ranking of
  documents against a query.
- **Know what you paid.** Every result carries a generation id, and one tool
  looks up the real charge for it.

Every tool also accepts a `provider` setting. Use it to choose which provider
serves the request, or to pass a provider's own knobs through. See
[Provider routing](#provider-routing-and-passthrough).

## Install

### Claude Desktop, one click

1. Download the bundle for your platform from the
   [latest release](https://github.com/thesimj/rust-openrouter-mcp/releases/latest):
   `openrouter-mcp-macos.mcpb`, `openrouter-mcp-windows.mcpb`, or
   `openrouter-mcp-linux.mcpb`.
2. Double-click it, or drag it into **Claude Desktop -> Settings -> Extensions**.
3. Click **Install** and paste your [OpenRouter API key](https://openrouter.ai/keys).

Claude Desktop stores the key in the OS credential store and hands it to the
server as `OPENROUTER_API_KEY`. No terminal or Rust toolchain needed.

### Other clients

[CONNECT.md](CONNECT.md) has copy-paste setup for Claude Code, Codex CLI,
Gemini CLI, Cursor, Windsurf, VS Code, Zed, Cline, Roo Code, Continue, Goose,
opencode, Crush, Amp, OpenHands, and more.

One-click buttons for two of them (install the binary first, then replace the
placeholder key):

[![Add to Cursor](https://cursor.com/deeplink/mcp-install-dark.svg)](cursor://anysphere.cursor-deeplink/mcp/install?name=openrouter&config=eyJjb21tYW5kIjogIm9wZW5yb3V0ZXItbWNwIiwgImFyZ3MiOiBbIm1jcCJdLCAiZW52IjogeyJPUEVOUk9VVEVSX0FQSV9LRVkiOiAic2stb3ItdjEtWU9VUl9LRVkifX0=)
[![Install in VS Code](https://img.shields.io/badge/VS_Code-Install_Server-0098FF?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=openrouter&config=%7B%22command%22%3A%20%22openrouter-mcp%22%2C%20%22args%22%3A%20%5B%22mcp%22%5D%2C%20%22env%22%3A%20%7B%22OPENROUTER_API_KEY%22%3A%20%22sk-or-v1-YOUR_KEY%22%7D%7D)

### The binary

```bash
cargo install openrouter-mcp            # from crates.io
cargo install --path . --locked --force # from a checkout
```

Then set your key:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."      # bash/zsh
$env:OPENROUTER_API_KEY = "sk-or-v1-..."      # PowerShell
```

A `.env` file in the working directory works too. Do not commit it.

Generic MCP client config:

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

## The tools

| Tool | What it does |
| --- | --- |
| `list_models` | Search the catalog. Filter by input/output modality, supported parameters, category, provider, author, price, region, zero-data-retention, model age, and benchmark scores. Sort, page, and see pricing in dollars per million tokens. |
| `describe_model` | Everything about one model: architecture, context, benchmarks, and each provider endpoint with its pricing and the `allowed_passthrough_parameters` you may send in `provider.options`. |
| `generate_image` | Text-to-image or image editing. Inputs by path, URL or base64 (up to 16). Pick `aspect_ratio` + `image_size`, or one `size`. Optional `quality`, `output_format`, `background`, `output_compression`, `seed`, `variants`. Long jobs return a `task_id`. |
| `generate_video` | Text-, image- or reference-to-video. Required: `model`, `duration`, `with_audio`. `prompt` is optional when a frame or reference is given. Optional `resolution`, `aspect_ratio`, `size`, `seed`, `creativity`, `upscale_factor`. Always returns a `task_id`; poll `get_result`. |
| `generate_audio` | Text-to-speech. `voice` is model-specific. Voice cloning: pass a sample as `voice_reference` plus an optional transcript. |
| `generate_music` | Text-to-music with Google Lyria. Returns the track, the lyrics text, and the per-track cost. |
| `transcribe_audio` | Speech-to-text from a file or base64. Optional `language`, `verbose_json` with timestamps, and speaker labels via `provider.options`. |
| `chat_completion` | Send a prompt to any chat model. Attach `images`, `files`, `audio`, `videos`. Ask for `json_mode` or a `json_schema`. Turn on `web_search`. Control sampling and reasoning. |
| `describe_image` | Describe one or more images with a vision model. |
| `embed_text` | Float vectors for a list of texts. |
| `rerank_documents` | Rank documents against a query, best first. |
| `get_generation` | The stored record for a `generation_id`: real cost, provider, tokens, latency. |
| `get_result` | Fetch an async job by `task_id`. |
| `get_account` | Your API key's label, credits, and limits. |
| `get_usage_stats` | This process's request and cost counters. |
| `reset_usage_stats` | Clear those counters. Needs `confirm: true`. |

Every tool description tells the assistant which `list_models` filter finds
the right kind of model, for example `output_modalities="transcription"`.

## Provider routing and passthrough

OpenRouter can route one model to several providers. The `provider` object on
each tool lets you steer that.

**Routing** (chat, describe_image, generate_music, embed_text, rerank_documents,
generate_image): `order`, `only`, `ignore`, `allow_fallbacks`,
`require_parameters`, `zdr`, `sort`.

```json
{ "provider": { "order": ["anthropic", "google-vertex"], "allow_fallbacks": false } }
```

**Passthrough** (generate_image, generate_audio, transcribe_audio,
generate_video): `options`, keyed by provider slug, holding that provider's own
parameters. Only the provider that serves the request receives them.

```json
{ "provider": { "options": { "deepgram": { "diarize": true } } } }
```

That example turns on speaker labels for a Deepgram transcription. Other common
uses: `{"openai": {"instructions": "speak like a calm narrator"}}` on speech,
`{"black-forest-labs": {"steps": 28, "guidance": 3.5}}` on FLUX images,
`{"google-vertex": {"negativePrompt": "people, text"}}` on Veo video. Run
`describe_model` to see which keys a provider accepts.

Chat-family tools take routing only. OpenRouter's chat schema has no
passthrough field.

## Command line

The same binary is a CLI. Subcommands: `models`, `image`, `video`, `audio`,
`music`, `transcribe`, `describe`, `chat`, `embed`, `rerank`, `generation`,
`key`, `mcp`. Every network subcommand takes `--provider '<json>'` with the same
object the MCP tool takes. `--help` lists every flag.

```bash
# Ask a model something
openrouter-mcp chat -m anthropic/claude-sonnet-4.6 -p "Why Rust?" --temperature 0.3

# Read a PDF and answer as JSON matching a schema
openrouter-mcp chat -m openai/gpt-5.4 -p "Extract the invoice total." \
  --file ./invoice.pdf --pdf-engine mistral-ocr --json-schema ./invoice.schema.json

# Browse models
openrouter-mcp models --output-modalities image --sort newest --table
openrouter-mcp models --category programming --sort intelligence-high-to-low --table

# Make an image, then four seed-stepped variants
openrouter-mcp image -m google/gemini-3.1-flash-image-preview \
  -p "a photorealistic owl with one cybernetic eye" --aspect-ratio 1:1 --image-size 1K -o ./out/owl.png
openrouter-mcp image -m bytedance-seed/seedream-4.5 -p "a cute baby dragon" \
  --aspect-ratio 1:1 --image-size 1K --seed 1490 --variants 4 -o ./out/dragon.png

# Make a video from a still frame
openrouter-mcp video -m bytedance/seedance-2.0 --first-frame ./out/owl.png --duration 5 -o ./out/owl.mp4

# Speech, cloned voice, music
openrouter-mcp audio -m hexgrad/kokoro-82m --voice af_heart --input "Hello." -o ./out/hello.mp3
openrouter-mcp audio -m fish-audio/s1 --voice-reference ./sample.wav --input "Cloned hello." -o ./out/clone.mp3
openrouter-mcp music -m google/lyria-3-clip-preview -p "warm lo-fi loop, soft piano, 80 bpm" -o ./out/loop.mp3

# Transcribe with speaker labels
openrouter-mcp transcribe -m deepgram/nova-3 --file ./meeting.mp3 --response-format verbose_json \
  --provider '{"options":{"deepgram":{"diarize":true}}}'

# Embeddings, rerank, and the real cost of a request
openrouter-mcp embed -m openai/text-embedding-3-small -i "first text" -i "second text"
openrouter-mcp rerank -m cohere/rerank-v3.5 -q "rust async runtime" -d "Tokio is..." -d "Sourdough..." --top-n 1
openrouter-mcp generation --id gen-1234567890abcdef
```

## Configuration

`OPENROUTER_API_KEY` is the only required setting. The rest have sensible
defaults.

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Your OpenRouter key. | required |
| `OPENROUTER_MCP_OUTPUT_DIR` | Where generated files land when you give no `output` path. | `$HOME/Downloads/openrouter-mcp` |
| `OPENROUTER_MCP_IMAGE_PREVIEWS` | Embed previews of generated media in tool results: `auto`, `always`, `never`. `auto` embeds for every client except Claude Code, which can open the saved file itself. | `auto` |
| `OPENROUTER_IMAGE_MAX_DIMENSION` | Longest side, in pixels, that input images are scaled down to before upload. Hard ceiling 4096. | `1536` |
| `OPENROUTER_VIDEO_POLL_INTERVAL` | Seconds between video status checks. | `5` |
| `OPENROUTER_VIDEO_POLL_TIMEOUT` | Seconds to wait for a video job. | `600` |
| `OPENROUTER_VIDEO_DELIVERY_TIMEOUT` | Seconds allowed for downloading finished clips, retries included. | off |
| `OPENROUTER_MCP_SHUTDOWN_TIMEOUT` | Seconds to let running jobs finish after the client disconnects. | `30` |
| `OPENROUTER_HTTP_REFERER`, `OPENROUTER_X_TITLE` | App attribution headers shown in OpenRouter rankings. | this repo |

**Limits worth knowing.** Input images: 16 per request, 20 MiB each, 64 MiB per
batch. Audio for transcription or cloning: 25 MiB. Files, audio and video
attached to chat: 20 MiB each. At most 32 image or video jobs wait at once, and
8 synchronous calls run at once. Job status lives in memory and is gone when
the server exits.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

CI runs the same three checks plus a build at the minimum supported Rust
version on every push and pull request. A version tag triggers the release
workflow, which builds the `.mcpb` bundles for macOS, Windows and Linux.

Release notes and the list of OpenRouter features left out on purpose live in
[CHANGELOG.md](CHANGELOG.md).

## Privacy

`openrouter-mcp` runs on your machine and sends nothing anywhere except to
OpenRouter, plus a direct fetch of any image URL you pass as input. No
telemetry. Details in [PRIVACY.md](PRIVACY.md).

## License

Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE)) or MIT
([LICENSE-MIT](LICENSE-MIT)), at your option.
