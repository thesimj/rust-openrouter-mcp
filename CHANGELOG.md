# Changelog

## 0.11.0

- **Breaking:** the binary now serves MCP only. `openrouter-mcp mcp` starts the
  server as before, and the bare `openrouter-mcp` with no arguments does the same.
  Existing client configurations with `"args": ["mcp"]` keep working unchanged.
- Removed the operational CLI commands (`models`, `image`, `video`, `audio`, `music`,
  `transcribe`, `describe`, `chat`, `embed`, `rerank`, `generation`, `key`) and the
  help flags. Use the corresponding MCP tools.
  `--version` and `-V` remain available without credentials.
  `mcp` and the version flags must appear alone. Other arguments exit with status 2
  and a diagnostic on stderr.
- Preserved all 16 MCP tools, their schemas, generation behavior, saved files, and usage accounting.
  Clients must supply prompt text and schema objects directly, and poll `get_result` for unfinished jobs.
  See [Launch and version](README.md#launch-and-version) for input and output migration details.
- Removed Clap and unused CLI pricing and video catalog helpers.
  The desktop bundle and installation links still launch with `mcp`.

## 0.10.1

- Regenerated `Cargo.lock` to the latest Rust 1.88 compatible versions. rmcp
  moves to 3.4.0, reqwest to 0.13.5, plus transitive patch bumps.
- rmcp 3.4 deprecates the `ServerInfo` alias for the identical `ServerConfig`
  alias; renamed.
- README rewritten as a short guide. The detailed notes below moved here.

## 0.10.0

0.10.0 closes the gap between the tools and the OpenRouter request schemas:
every tool gained a `provider` object, several gained the documented top-level
fields they were missing, and three tools are new (`embed_text`,
`rerank_documents`, `get_generation`).

### Provider routing and passthrough

Every generation tool takes a `provider` object (CLI: `--provider '<json>'`).
Chat, `describe_image`, `generate_music`, `embed_text` and `rerank_documents`
take routing only (`order`, `only`, `ignore`, `allow_fallbacks`,
`require_parameters`, `zdr`, `sort` + `sort_partition`). `generate_image` takes
routing plus `provider.options`; `generate_audio`, `transcribe_audio` and
`generate_video` take `provider.options` only. `options` is keyed by provider
slug and holds that provider's own parameters. Only the slug that serves the
request is forwarded. Values are validated locally (each `options` value must
be an object; `sort` vocabulary) and recorded in the manifest.

### Contract changes

- `generate_audio`: `voice` changed from required to optional in the tool
  schema. It is provider-dependent: most TTS models still have no default voice
  and fail upstream without one; voice-cloning models (e.g. fish-audio, driven
  by `voice_reference`) take none.
- CLI `audio`: `--voice` is now optional.
- `AudioManifest`: `voice` is omitted when none was sent; new optional keys
  `provider` and `voice_reference`.
- Wire types: `SpeechBody.voice` is `Option<String>`; `SpeechGenRequest` gained
  `voice_reference` and `provider`.
- `generate_image` schema: `aspect_ratio` and `image_size` are no longer in the
  unconditional `required` array. They are required at runtime only when `size`
  is absent. New params `size` and `provider` (CLI `--size`, `--provider`).
- `generate_video.prompt` is now conditional: required at runtime only when no
  `first_frame`/`last_frame` and no reference (`reference_images`,
  `reference_audio`, `reference_videos`) is given. `VideoManifest.prompt` is
  optional and `prompt_source` can be `"none"`.
- Internal: `openrouter::InputReference` is now an enum
  (`image_url`/`audio_url`/`video_url`).
- `list_models` output starts with a `// server total_count: N ...` header line
  whenever OpenRouter returns `total_count`. JSON parsers of the tool text must
  skip leading `//` lines.
- `OpenRouterClient::list_models_page` is new and returns `ModelsResponse`.
  `Model` derives `Default`.
- Three new tools and CLI subcommands (`embed`, `rerank`, `generation`).
- `UsageStats::record_lookup`; `get_usage_stats` counts `get_generation` calls
  in `requests_total` without touching cost.
- `chat_completion` result: unchanged for the common case (`content[0]` is the
  reply text). When the response carries reasoning, annotations, or a
  `finish_reason` other than `"stop"`, a second text block with a JSON object
  `{reasoning?, annotations?, finish_reason?, usage?}` is appended.
- `ChatCompletionArgs` / `DescribeImageArgs` / `GenerateMusicArgs` gain new
  optional properties (`provider`, `web_search`, `json_schema`, `files`,
  `audio`, `videos`, ...). None added to `required`.
- `describe_image` accepts `system`, `temperature`, `max_tokens`.
- `MusicManifest` gains an optional `provider` field.

### New per-tool fields

- `generate_image`: `size`, `provider`; `quality` accepts `xhigh` and `max`;
  `aspect_ratio` accepts `2.35:1`, `5:2`, `9:19.5`, `19.5:9`, `9:20`, `20:9`;
  `image_size` accepts `512`; at most 16 input images.
- `generate_video`: `provider.options` (sent opaque; OpenRouter documents both
  `{"google-vertex": {"negativePrompt": ...}}` and the nested
  `{"google-vertex": {"parameters": {...}}}` shape), `reference_audio`,
  `reference_videos`, `creativity`, `upscale_factor`, `resolution` `768p`.
- `generate_audio`: `provider.options`, `voice_reference`
  (`{path|base64, format?}`), `voice_reference_text` (max 10000 chars, sample
  15 MiB decoded).
- `transcribe_audio`: `provider.options`; verbose_json segments and words carry
  a `speaker` index when the provider diarizes.
- `chat_completion`: `seed`, `top_p`, `top_k`, `stop`, `frequency_penalty`,
  `presence_penalty`, `verbosity`, `json_mode`, `json_schema`, `web_search`
  (`engine`, `mode`, `max_results`, `search_prompt`, `include_domains`,
  `exclude_domains`, `search_context_size`), `pdf_engine`,
  `reasoning_max_tokens`, `reasoning_exclude`, `files`, `audio`, `videos`.
  Reply carries `reasoning`, `annotations`, `finish_reason`, token usage and
  `generation_id`.
- `list_models`: `category`, `providers`, `model_authors`, `arch`,
  `min_price`/`max_price`, `min_output_price`/`max_output_price`, `zdr`,
  `region`, `distillable`, `min_age_days`/`max_age_days`, `limit`/`offset`,
  the intelligence/coding/agentic index and `tool_success_rate` min/max pairs,
  new sorts `intelligence-high-to-low`, `coding-high-to-low`,
  `agentic-high-to-low`, `design-arena-elo-high-to-low`. Rows carry
  `reasoning.supported_efforts`/`mandatory`, `supported_voices`,
  `knowledge_cutoff`, `expiration_date`.

### Not implemented on purpose

- Chat: `tools`/`tool_choice` (needs an agent loop the tool cannot run),
  `logit_bias`, `logprobs`, `top_logprobs`, `prediction`, `response_format`
  types `text`/`grammar`/`python`, `reasoning.context`/`mode` (GPT-5.6+ only),
  web plugin `max_uses`/`user_location`, `fallback_models`, `min_p`, `top_a`,
  `repetition_penalty`, `max_completion_tokens`, `provider.options` on chat
  (OpenRouter's chat `provider` schema is routing-only and rejects `options`).
- Provider routing: `quantizations`, `max_price`, `preferred_max_latency`,
  `preferred_min_throughput`, `data_collection`, `enforce_distillable_text`.
- Embeddings: `encoding_format` (float vectors only), token-array and
  multimodal `input`. Rerank: `{text, image}` documents (plain strings only).
- `describe_model`: no `/embeddings/models` lookup.
- `generate_image`: `n` stays unsent; `variants` fans out as N parallel calls.
- Everywhere: `stream`, `callback_url`, `session_id`, `user`, `trace`, and the
  Files/Containers/Workspaces/Guardrails/BYOK/Analytics management APIs, the
  Responses and Anthropic Messages endpoints.
- CLI: `list_models` exposes `--category`, `--providers`, `--limit`, `--offset`,
  `--zdr`, `--region` only; `chat` keeps `--temperature`/`--max-tokens`/
  `--reasoning-effort` and `--web-search` with defaults only; `--file`/`--audio`
  take local paths, `--video` a URL or path. The MCP tools expose everything.
- Manifests: none for `embed_text`/`rerank_documents`/`get_generation`; they
  save no files.
- Live end-to-end runs against OpenRouter were not part of the release gates;
  wire shapes are locked by wiremock body assertions at the client and tool
  layers.

## 0.9.0

- Music generation via streamed audio-output chat completions.

## 0.8.0

- Bounded generation resources and job shutdown tracking.
