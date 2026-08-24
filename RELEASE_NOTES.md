# v0.3.0

## What's New

### Speech-to-text: `transcribe_audio`

A new MCP tool and `openrouter-mcp transcribe` CLI subcommand, wrapping
OpenRouter's dedicated `POST /api/v1/audio/transcriptions` endpoint - the
counterpart to `generate_audio`.

- Audio by local **`path`** (container format inferred from the extension) or
  inline **`base64`** (a `data:` URL is accepted, and its subtype supplies the
  format)
- Formats: `wav`, `mp3`, `flac`, `m4a`, `ogg`, `webm`, `aac`, up to **25 MB** -
  the cap is enforced locally, so an oversized file fails immediately instead of
  after a long upload
- Optional ISO-639-1 `language` hint improves accuracy
- Synchronous - returns the transcript directly, no `task_id` polling
- Cost flows into `get_usage_stats`

STT models are absent from the default catalog; find them with `list_models` and
`output_modalities="transcription"`.

```bash
openrouter-mcp transcribe --model openai/gpt-4o-mini-transcribe --file ./hello.mp3 --language en
```

## Changed

- **MCP SDK: rmcp 1.7 -> 3.0.** No wire-protocol change - the server still
  negotiates **`2025-11-25`**, which is rmcp's own `LATEST`. rmcp 3.0 knows the
  `2026-07-28` stateless spec but does not default to it, and over stdio it buys
  nothing. A test pins this so a future SDK bump cannot shift it silently.
- **MSRV 1.96 -> 1.88.** The old value was 8 releases too high and refused
  `cargo install` for anyone on 1.88-1.95 for no reason. 1.88 is verified against
  a real toolchain and is what rmcp 3.0 and our let-chains actually require. CI
  now builds at the declared version so it cannot drift.
- **Upstream errors are no longer swallowed.** `/models`,
  `/models/{id}/endpoints`, `/videos/models`, `/key`, and `/credits` previously
  discarded the provider's error body - the only useful diagnostic. All nine
  endpoints now surface it, truncated to 500 chars so a proxy's HTML error page
  cannot flood an agent's context.
- `openrouter-mcp models` (JSON mode) now always prints its `N models` count to
  **stderr**, matching `--table`. stdout stays pipe-clean.

## Fixed

- **The server announced itself as `rmcp`.** rmcp's
  `Implementation::from_build_env()` expands `env!("CARGO_CRATE_NAME")` *inside
  the SDK*, so every client displayed the SDK's name and version instead of ours.
  Now correctly `openrouter-mcp v0.3.0`.

## Internal

~400 lines of duplication removed following a repo-wide audit: one shared
`send_checked`/`send_json` path for all nine endpoints (replacing three competing
styles), `describe_image` now delegates to `chat_gen::complete` instead of
near-copying it, `manifest::write` is generic and **async** (it was blocking a
Tokio worker on job completion), the `uuid` dependency is gone (task ids never
leave the process - an `AtomicU64` suffices), and nine fields kept alive only by
`#[allow(dead_code)]` were deleted.

**141 tests** (up from 135), 87% line coverage, clippy `-D warnings` and rustfmt
clean, verified end-to-end with a live JSON-RPC handshake over stdio.

## Upgrading

Drop-in - no config, no environment, and no protocol changes. The MSRV move is
strictly more permissive.

**Full changelog:** https://github.com/thesimj/rust-openrouter-mcp/compare/v0.2.17...v0.3.0
