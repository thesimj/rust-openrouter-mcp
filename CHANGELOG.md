# Changelog

## 0.13.0

Fixes (each covered by a test):

- `generate_music`: a stream that fails after reporting its cost (a bad audio
  fragment) keeps that cost and the chunk's generation id in the billing
  receipt, so `get_usage_stats` no longer counts it as an unknown cost.
- `chat_completion`: whitespace stop sequences such as `"\n"` reach the wire
  (they were dropped as blank). All inputs are checked locally before the
  model-capability gate, and the catalog is asked once per call.
- `describe_image`: a blank `prompt`, `system` or `reasoning_effort` counts as
  unset (a blank prompt used to be sent as `" "`).
- `generate_video`: blank `aspect_ratio`, `size`, `resolution`,
  `first_frame` and `last_frame` count as absent; `reference_audio` and
  `reference_videos` take at most 16 entries, like every other input list.
  References are resolved before the job starts, so a missing file is an
  invalid-params error naming the argument (it was a failed task), even when
  a frame means the references would be ignored.
- `transcribe_audio`: bad inline base64 or an unknown format is rejected as
  invalid params before any call (it was an internal error counted as a failed
  request). The result carries a `{"generation_id"}` block when OpenRouter
  sends one.
- `generate_audio`: the saved file and the recorded mime come from the audio
  bytes first, then a known content type, then the requested format
  (`audio/aac` replies were saved as `.mp3`). Music, speech and audio inputs
  share one container rule.
- `generate_image`: raster bytes beat a declared `image/svg+xml`; an SVG whose
  size cannot be read says so in a warning. A `"512"` request no longer warns
  that a 512 px image is `~0.5K`, and the `768` tier is recognized.
- `embed_text`: a reply with fewer vectors than inputs, or indices that skip or
  repeat, is a billed failure instead of a mismatched success.
- Raw base64 inputs decode as leniently as data URLs (line breaks and missing
  padding are accepted).
- Tool errors keep their cause (`list_models`, `describe_model`, `get_account`,
  manifest-write warnings); a missing output directory reports why it could not
  be created.
- `get_result`: over the retention bound, the result that finished longest ago
  is evicted, not the task created first (a long video job lost its result
  seconds after finishing).
- A `.env` that fails to parse is named on stderr.
- Model ids and video job ids are percent-encoded in request paths; a model
  id with an empty, `.` or `..` segment is refused.
- Blank-means-absent now also trims the kept value everywhere, e.g.
  `generate_image` sends `quality: " high "` as `"high"`.
- An output path with no file stem is named `output...` instead of `image...`.

OpenRouter API alignment (checked against `openapi.json` and the docs,
2026-09-25):

- The app title header is `X-OpenRouter-Title` (`X-Title` is legacy); the
  `OPENROUTER_X_TITLE` variable still sets it.
- `describe_model` renders the live video SKU families
  (`cents_per_video_output_second_*`, `cents_per_image_input`,
  `minimum_cents_per_generation`, `text_to_video_` / `image_to_video_`
  duration keys) in dollars with their unit.
- `json_schema` titles become valid schema names (`[A-Za-z0-9_-]`, at most 64).
- A chat reply whose `content` is an array of parts keeps its text.
- Audio inputs also accept `aiff`, `pcm16` and `pcm24`.
- Docs: `video_url.processing` is `agentic` / `static`, verbosity adds `xhigh`
  and `max`, web-search engines, the `360p` video tier, `/credits` needing a
  management key, and `is_provisioning_key` being deprecated.

### Internal refactor (no behavior change)

- One `request(method, path)` builder for every endpoint; `ChatWire` owns the
  `stream` flag; one error-text rule; one numbered output-path rule; one wait
  clamp; one "blank means absent" helper at the tool boundary; one
  text-call result tail (`finish_text_call`); the task snapshot carries a
  `Status` enum; the video job takes already-resolved references, so no domain
  module imports the server layer; the base64 / data-URL codec moved from
  `image_io` to `base64_codec`; shared test fixtures; unused fields and flags removed (`ImagesRequest.n`,
  `fetch_urls`, never-set manifest fields, a string-pricing fallback on image
  endpoints that never matched their array-shaped pricing).

## 0.12.1

- `get_usage_stats` rounds `actual_cost_usd` to six decimals instead of four,
  so a single decisions, embeddings or rerank call (about $0.00002) no longer
  displays as `0.0`.

## 0.12.0

- New tool `make_decisions` for OpenRouter's decisions models (TypeSafe Jev:
  `typesafe/jev-1.13`, alias `~typesafe/jev-latest`). It calls
  `POST /api/alpha/decisions`, the alpha endpoint outside `/api/v1`, with a
  `state` (string, object or array) and named `noul` / `choice` / `score`
  questions, and returns the typed answers (probabilities, confidence, legend),
  `usage {input_tokens, output_tokens, cost}` and a `generation_id`. `provider`
  takes the same routing subset as `embed_text` and `rerank_documents`.
- Local validation before any HTTP call: a non-blank `state`, at least one
  question, non-blank instructions and labels, `choice` criteria as a
  label -> description object, `score` criteria as an array of at least two
  levels (TypeSafe's documented minimum), `noul` criteria as `{"true", "false"}`.
  Errors name the question (`questions["team"]: ...`).
- `chat_completion` now points at `make_decisions` when OpenRouter rejects a
  decisions model on `/chat/completions`.
- `list_models` documents the `decisions` output modality; `get_usage_stats`
  counts `make_decisions` under `text_generations`.
- `list_models.search` follows the blank-means-absent rule: a padded needle is
  trimmed and a blank one applies no filter (before, `" gpt "` matched nothing
  and `""` matched everything).
- Client: `OpenRouterClient::api_root()` derives the server root above
  `/api/v1` for endpoints served outside it.
- Skipped on purpose: the `session_id`, `trace` and `user` request fields, and
  the rest of `ProviderPreferences` (quantizations, max_price, preferred_*).

### Internal refactor (no behavior change)

A design pass over the whole crate, one rule per place:

- `send_json_receipted` is the single "a 2xx that fails to decode keeps its
  billing receipt" step; chat, images, embeddings, rerank, decisions, video
  submit and transcription all use it. Its five per-endpoint test copies are
  one test.
- The `EmbedRequest` / `RerankRequest` / `DecideRequest` mirrors of the wire
  bodies and the three `*Reply { body, generation_id }` wrappers are gone: the
  tools build `EmbeddingsBody` / `RerankBody` / `DecisionsBody`, validate once
  at the boundary, and the client methods return `(response, generation_id)`.
- `list_models` presentation (`apply_filters`, the search match, the pagination
  note) moved from the HTTP client and DTOs into `server/models.rs`.
- `record_text_failure` / `record_audio_failure` own the "failed request +
  receipt" pairing; `json_text_result` owns the pretty-JSON text block;
  `clean_list` owns the trim-and-drop-blanks rule for slug and domain lists;
  `media::check_count` owns the 16-input cap for every input kind (the image
  message now reads "at most 16 image inputs are supported");
  `audio_gen::normalize_response_format` / `normalize_voice` own the speech
  normalization the tool and the job both apply.
- `chat_completion` and `describe_image` share `finish_chat_call`; the
  test-only `run_chat_completion` wrapper is gone (tests call the tool).
- `record_text` / `record_audio` lost their `success` flag: success and
  failure are separate methods, and the audio failure path has a test.
- One `BoundedResponse::decode` carries the decode-failure message and one
  `unwrap_data` the `{"data": ...}` envelope rule.
- Deleted: the unused `ImageConfig` chat field, the never-set `text` /
  `provider` fields on generated images and their manifest entries, the
  `ModelCapsCache` newtype (a map behind a lock on the server struct now),
  the `list_models` wrapper over `list_models_page`, the `MAX_INPUTS_PER_KIND`
  alias, a duplicated 30-day timer cap and "any reference present" rule in
  `video_gen`, two poll-interval floors the only caller already applies plus
  a dead `.max(1)` on the clip count, and the
  `prompt_source` / `input_source` parameter every tool passed as `"inline"`
  (the manifest field stays, always `inline`).
- Docs brought back in line with the code: PRIVACY.md (17 tools, files fetched
  by URL, the decisions and generation endpoints), README limits (25 MiB chat
  audio, 15 MiB cloning sample), routing lists per tool, CI commands, the
  `.env` search, the output-dir fallback; CONNECT.md provider shapes; manifest
  and Cargo.toml descriptions and keywords; stale module comments.

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
