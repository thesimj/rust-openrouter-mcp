# OpenRouter API validation — 2026-09-07

The 13 implemented method/path pairs match current OpenRouter routes. The audit found pricing display defects and smaller schema and documentation gaps.

This report compares all production clients and DTOs under `src/openrouter` with current official documentation and OpenAPI schemas. It also examines relevant audio, video, and pricing callers. It does not certify every OpenRouter capability or every provider implementation.

The audit used public documentation and four unauthenticated catalog GET requests. It made no generation requests and used no API keys. Authentication success, provider behavior, and billable responses therefore remain untested against production.

## Confirmed findings

### API-1: Image pricing ignores the explicit billing unit

`attach_image_pricing_human` infers units from `billable` instead of reading `unit`. The affected code is `src/pricing.rs:184–202`.

The public GPT Image 2 endpoint returned these records during the audit:

| billable | unit | cost_usd | Current display | Correct display |
| --- | --- | --- | --- | --- |
| `input_image` | `token` | `0.000008` | `$0.000008/image` | `$8/M tokens` |
| `input_text` | `token` | `0.000005` | `$0.000005/unit` | `$5/M tokens` |
| `output_image` | `token` | `0.00003` | `$0.00003/image` | `$30/M tokens` |

These labels misrepresent the unit that determines cost. They affect the image endpoint details returned by model descriptions. The raw pricing records remain available.

OpenRouter defines `billable` and `unit` separately. Its current schema permits `request`, `image`, `megapixel`, and `token` units. Read the explicit unit and preserve any `variant` label. [Image endpoint reference](https://openrouter.ai/docs/api/api-reference/images/list-endpoints-for-an-image-model), [image pricing guide](https://openrouter.ai/docs/guides/overview/multimodal/image-generation), [public GPT Image 2 record](https://openrouter.ai/api/v1/images/models/openai/gpt-image-2/endpoints).

### API-2: Video pricing drops the megapixel dimension

`video_price` and `humanize_price` treat every remaining SKU containing `second` as cents per second. The affected code is `src/pricing.rs:59–62` and `src/pricing.rs:106`.

The public video catalog includes `black-forest-labs/flux-video-upscale`. Its `cents_per_megapixel_second_precise` and `cents_per_megapixel_second_creative` values are `7.5` and `10.5`.

The code displays `$0.075–0.105/s`. The actual unit is dollars per megapixel-second. Consequently, the displayed rate omits the output image area. Both the CLI table and model descriptions use these formatters. [Official model pricing](https://openrouter.ai/black-forest-labs/flux-video-upscale), [public video catalog](https://openrouter.ai/api/v1/videos/models).

### API-3: Model lists discard the one-hour cache-write rate

`Pricing` lacks `input_cache_write_1h`. Serde therefore discards that field before `models_to_json` can format it. The affected code is `src/openrouter/dto/models.rs:120–131`.

The public catalog contained this field on 31 entries during the audit. For example, `anthropic/claude-opus-4.7` reported `0.00001`, or `$10/M tokens`.

This affects CLI JSON and MCP model lists. Model descriptions retain the rate because their client returns raw JSON. Add the field or preserve unknown pricing fields. [Public catalog](https://openrouter.ai/api/v1/models?output_modalities=all), [official OpenAPI, `PublicPricing`](https://openrouter.ai/openapi.json).

### API-4: The local transcription limit is described as an upstream limit

`read_audio_file` rejects files above 25 MiB. Its error claims that the transcription endpoint accepts no larger files. See `src/audio_gen.rs:22–25` and `src/audio_gen.rs:86–90`.

The client sends JSON containing base64 audio. OpenRouter's multipart schema limits file uploads to 25 MB and explicitly directs larger files to base64 JSON.

The local cap may remain a product constraint. Its documentation and error should identify it as local. This audit did not establish the maximum JSON payload accepted by every provider. [Transcription reference](https://openrouter.ai/docs/api/api-reference/stt/create-transcription), [speech-to-text guide](https://openrouter.ai/docs/guides/overview/multimodal/stt).

## Schema risks and diagnostic gaps

`VideoModel.pricing_skus` accepts missing fields but rejects explicit JSON `null`. OpenRouter declares this field nullable. A nullable record can reject the entire typed video catalog response. This affects CLI table enrichment, which tolerates the request failure but loses video pricing. Raw video model descriptions remain unaffected. None of the 28 live catalog entries contained null during this audit. This is a confirmed schema incompatibility, not an observed production failure. See `src/openrouter/dto/video.rs:21–24`. [Video model reference](https://openrouter.ai/docs/api/api-reference/video-generation/list-all-video-generation-models), [official OpenAPI, `VideoModel`](https://openrouter.ai/openapi.json).

`VideoPollResponse` discards the documented `error` string. The orchestration reports a failed status and job identifier without the provider's explanation. All six documented statuses receive appropriate terminal or pending treatment. Preserve `error` to improve failed-job diagnostics. See `src/openrouter/dto/video.rs:96–104` and `src/video_gen/job.rs:180–183`. [Poll reference](https://openrouter.ai/docs/api/api-reference/video-generation/poll-video-generation-status).

The chat DTO discards `finish_reason`. A length-limited completion can therefore appear as ordinary completed text. This also hides content-filter termination metadata. Treat this as a diagnostic gap unless callers require complete answers. See `src/openrouter/dto/chat.rs:100–114`. [Chat reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion).

## Endpoint matrix

Every path below uses the base URL `https://openrouter.ai/api/v1`. Every implemented request sends `Authorization: Bearer ...`. JSON POST requests use reqwest's JSON encoder. Binary responses use their response content type.

| Method and path | Request contract checked | Success contract checked | Assessment |
| --- | --- | --- | --- |
| `GET /models` | `q`, `output_modalities`, `input_modalities`, `supported_parameters`, `sort`, `context` | `200`, JSON `data[]` | Query names match. `output_modalities=all` correctly broadens capability lookups. API-3 affects retained pricing. [Reference](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties). |
| `GET /models/{author}/{slug}/endpoints` | Author and slug in model identifier | `200`, JSON `data` object | Client unwraps `data` and retains endpoint details. [OpenAPI](https://openrouter.ai/openapi.json). |
| `GET /images/models/{author}/{slug}/endpoints` | Author and slug in model identifier | `200`, top-level `id` and `endpoints[]` | Client accepts this shape and also tolerates an envelope. `404` becomes absent enrichment. API-1 affects formatting. [Reference](https://openrouter.ai/docs/api/api-reference/images/list-endpoints-for-an-image-model). |
| `GET /videos/models` | No query parameters | `200`, JSON `data[]`, optional nullable `pricing_skus` | Both typed catalog and raw detail callers exist. Nullable pricing and API-2 require attention. [Reference](https://openrouter.ai/docs/api/api-reference/video-generation/list-all-video-generation-models). |
| `GET /key` | Current bearer key | `200`, JSON `data` | Nullable limits and signed legacy request count match. Optional omitted metadata is a capability choice. [Reference](https://openrouter.ai/docs/api/api-reference/api-keys/get-current-api-key). |
| `GET /credits` | Management key required | `200`, numeric `data.total_credits` and `data.total_usage` | Shapes and derived balance match. Parent audit confirmed account output tolerates a credits error. [Reference](https://openrouter.ai/docs/api/api-reference/credits/get-remaining-credits). |
| `POST /chat/completions` | Model, messages, temperature, token limit, seed, reasoning effort, `stream:false` | `200`, JSON `choices[].message.content`, optional usage | Implemented text and vision subset matches. `max_tokens` remains accepted but is deprecated. [Reference](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion). |
| `POST /images` | Model, prompt, resolution, aspect ratio, seed, reference images, quality, output format, background, compression | `200`, JSON `data[].b64_json`, optional `media_type`, usage | Request names and reference shapes match. Raster and SVG decoding remain separate downstream concerns. [Reference](https://openrouter.ai/docs/api/api-reference/images/generate-an-image). |
| `POST /audio/speech` | Model, input, voice, `response_format`, speed | `200`, raw audio, content type, optional generation header | Domain explicitly sends MP3 by default. This safely overrides OpenRouter's PCM default. [Reference](https://openrouter.ai/docs/api/api-reference/tts/create-speech). |
| `POST /audio/transcriptions` | JSON model and raw base64 `input_audio`; language, response format, timestamps, temperature | `200`, JSON text, optional verbose fields and usage | Field names match. JSON uses `timestamp_granularities`, without multipart's `[]` suffix. API-4 affects local files. [Reference](https://openrouter.ai/docs/api/api-reference/stt/create-transcription). |
| `POST /videos` | Model, prompt, duration, resolution, ratio, size, frame images, references, audio flag, seed | `202`, JSON job identifier and polling location | Client accepts `202` and polls by identifier. Frame and image reference shapes match. [Reference](https://openrouter.ai/docs/api/api-reference/video-generation/submit-a-video-generation-request). |
| `GET /videos/{jobId}` | Submitted job identifier | `200`, status, optional URLs, generation identifier and usage | Status handling matches. DTO loses provider error text. Parent audit covers transient poll failures and usage accounting. [Reference](https://openrouter.ai/docs/api/api-reference/video-generation/poll-video-generation-status). |
| `GET /videos/{jobId}/content` | Nonnegative `index` query parameter | `200`, raw video bytes | Query and binary handling match. MIME determines extension. [Reference](https://openrouter.ai/docs/api/api-reference/video-generation/download-generated-video-content). |

The model catalog returns the full list when both pagination parameters are absent. The client's omission of `offset` and `limit` therefore does not establish a pagination defect. The public response contained 581 models. The local default display cap remains 20. [Models reference](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties).

## Error statuses

The current OpenAPI advertises these non-success statuses. The shared client rejects all non-2xx responses and includes a bounded response body. Image endpoint enrichment handles `404` separately.

| Endpoint | Documented non-success statuses |
| --- | --- |
| Models | `400, 403, 500` |
| Model endpoints | `403, 404, 500` |
| Image model endpoints | `404, 500` |
| Video models | `400, 500` |
| Key | `401, 500` |
| Credits | `401, 403, 500` |
| Chat | `400, 401, 402, 403, 404, 408, 413, 422, 429, 500, 502, 503, 524, 529` |
| Images | `400, 401, 402, 403, 404, 413, 429, 500, 502, 524, 529` |
| Speech and transcription | `400, 401, 402, 403, 404, 413, 429, 500, 502, 503, 524, 529` |
| Video submission | `400, 401, 402, 403, 404, 413, 429, 500` |
| Video polling | `401, 403, 404, 500` |
| Video download | `400, 401, 403, 404, 500, 502` |

These lists document the schema snapshot. They do not exclude other infrastructure responses. [Official OpenAPI](https://openrouter.ai/openapi.json).

## Supported subset and documentation drift

The typed model catalog also omits `supported_parameters`, `top_provider`, `default_parameters`, and `per_request_limits`. Model descriptions retain their raw provider records. This reduces catalog detail without invalidating a request. Treat these omissions as capability gaps unless the tool promises the complete upstream record. [Models reference](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties).

The seven reasoning efforts match current documentation: `max`, `xhigh`, `high`, `medium`, `low`, `minimal`, and `none`. Per-model support still varies. Null `supported_efforts` means no allowlist. A mandatory reasoning model rejects `none`. The untyped model reasoning object preserves these distinctions. Additional controls such as token budgets and reasoning summaries remain unsupported capabilities. [Reasoning guide](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

Speech supports `mp3` and `pcm`. Transcription supports the seven commonly documented input formats. Provider support may be narrower. Requiring a voice excludes provider-default voice selection. Voice cloning and provider options also remain unsupported capabilities. [Speech guide](https://openrouter.ai/docs/guides/overview/multimodal/tts), [transcription guide](https://openrouter.ai/docs/guides/overview/multimodal/stt).

Image generation omits streaming, provider routing, explicit pixel `size`, and direct multi-image batching. Video generation omits callbacks, audio/video references, provider options, and upscaling controls. Those omissions do not invalidate the implemented request subset. Per-endpoint capability records remain necessary when selecting model-specific options. [Image guide](https://openrouter.ai/docs/guides/overview/multimodal/image-generation), [video guide](https://openrouter.ai/docs/guides/overview/multimodal/video-generation).

The image DTO comment says `media_type` appears only for vector output. Current documentation allows it whenever the returned format is identifiable, including PNG. Runtime deserialization already accepts raster MIME values, so this is comment drift. [Image response guide](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).

`X-Title` remains a supported alias for `X-OpenRouter-Title`. The existing attribution header does not require a compatibility fix. [Attribution guide](https://openrouter.ai/docs/app-attribution).

Generic `PublicPricing.image_output` prose says dollars per image. The dedicated GPT Image 2 record instead explicitly identifies its matching rate as dollars per token. The audit therefore does not classify the generic token formatter as incorrect for that model. Prefer the dedicated endpoint's explicit unit when available. [Public record](https://openrouter.ai/api/v1/images/models/openai/gpt-image-2/endpoints), [OpenAPI](https://openrouter.ai/openapi.json).

## Evidence and limits

The parent supplied graph discovery and call-chain evidence for the implemented clients. This audit also queried 112 audio/video symbols with complete pagination.

Both `index_status` and `check_index_coverage` returned an approval-policy rejection. The graph generation and coverage therefore remain unverified. Direct source reads replaced graph completeness assumptions for all relied-on client, DTO, audio, video-job, and pricing files. This report makes no exhaustive whole-repository graph claim.

Several search results pointed to obsolete API-reference slugs. The report uses current paths from OpenRouter's [documentation index](https://openrouter.ai/docs/llms.txt).

The following SHA-256 hashes identify the downloaded public snapshots. Temporary files are supporting evidence, not committed fixtures.

The OpenAPI download completed at `2026-09-07T10:00:05Z`. The four catalog downloads completed at `2026-09-07T10:01:54Z`. These timestamps come from the local snapshot modification times.

These excerpts preserve the fields behind the pricing findings:

```json
{
  "image_endpoint_price": {
    "billable": "output_image",
    "unit": "token",
    "cost_usd": 0.00003
  },
  "video_catalog_entry": {
    "id": "black-forest-labs/flux-video-upscale",
    "pricing_skus": {
      "cents_per_megapixel_second_precise": "7.5",
      "cents_per_megapixel_second_creative": "10.5"
    }
  },
  "model_catalog_entry": {
    "id": "anthropic/claude-opus-4.7",
    "pricing": { "input_cache_write_1h": "0.00001" }
  }
}
```

| Snapshot | Result | SHA-256 |
| --- | --- | --- |
| `openapi.json` | Version `1.0.0` | `475896c7404e02a237530b4c677a7c88edbe7c357bfeb67dac0d33f46db2f6b5` |
| `/models?output_modalities=all` | 581 models | `af4df47f788a7be0978128a721c4c63d4e98e3b69ae187da5e51953c9a1bfeaf` |
| `/videos/models` | 28 models | `2e80b9f4e4fb4800652b10d336c48f7fc5a15cc745b706a5c3c0bbd18fa7ac74` |
| `/images/models/openai/gpt-image-2/endpoints` | Top-level endpoint record | `23861fb9c18a36763e4eab32f81c568b3bf6b8b59fc62c873aeb68fb7f843326` |
| `/models/openai/gpt-image-2/endpoints` | `data` envelope | `922c8c70764a7fd3a324dbddbe27a1bf0635b2048388afcf4178067911ee56cb` |

The parent audit owns compilation, tests, security findings, retry behavior, and final issue prioritization. This report records contract validation and its evidence only.
