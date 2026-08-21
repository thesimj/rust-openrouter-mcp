# Privacy Policy

_Last updated: 2026-08-21_

`openrouter-mcp` is a local Model Context Protocol (MCP) server that runs
entirely on your own machine. It is a thin client for the
[OpenRouter](https://openrouter.ai) API.

## What data is collected

The author of `openrouter-mcp` **collects nothing**. There is no telemetry,
no analytics, and no remote logging of any kind. The software has no servers
of its own.

## What the software sends, and to whom

`openrouter-mcp` communicates with exactly one third party - **OpenRouter** -
and only to perform the tool call you (or your AI assistant) explicitly make.
The 12 tools break down into a few classes:

- **Model discovery** (`list_models`, `describe_model`): sends your query/
  filter parameters, or a model id, to OpenRouter's model-catalog and
  per-model/per-provider-endpoint routes.
- **Chat and vision** (`chat_completion`, `describe_image`): sends your
  prompt, and any images you attach, to the chat/vision model you select.
- **Image, video, and speech generation** (`generate_image`,
  `generate_video`, `generate_audio`): sends your prompt and any local input
  images to the model you select, via OpenRouter's dedicated `/images`,
  `/videos`, and `/audio/speech` endpoints.
- **Transcription** (`transcribe_audio`): sends the audio you provide to the
  speech-to-text model you select.
- **Account/key info** (`get_account`): reads your key's label, usage, and
  credit balance from OpenRouter's `/key` and `/credits` endpoints.
- **Local only, no network call**: `get_result`, `get_usage_stats`, and
  `reset_usage_stats` read or reset the server's own in-memory job/usage
  state and never contact OpenRouter (or anyone else).

Every request above also carries two small app-attribution headers,
`HTTP-Referer` and `X-Title` - these identify the project to OpenRouter for
its public model rankings and have no effect on the response; see the env
var table in [README.md](README.md#configuration) if you want to override
them.

Your **OpenRouter API key** is sent to OpenRouter, and only OpenRouter, to
authenticate the requests above.

Separately, and not to OpenRouter: when you pass an image by `url` (to
`generate_image`, `describe_image`, or `chat_completion`), the software
fetches that URL itself, directly from your machine to whatever host you
named. This deliberately uses a plain, unauthenticated HTTP client - a
different client than the one used for OpenRouter - which is *why* your API
key is never sent anywhere but OpenRouter: it is never attached to this
fetch. The practical effect is that the URL's host sees the fetch request
(and your IP address) independent of OpenRouter.

OpenRouter's handling of this data is governed by OpenRouter's own
[Privacy Policy](https://openrouter.ai/privacy) and
[Terms of Service](https://openrouter.ai/terms).

## Where data is stored

- **API key**: when installed as a Claude Desktop extension, your API key is
  stored by Claude Desktop in your operating system's secure keychain. When run
  from the CLI it is read from the `OPENROUTER_API_KEY` environment variable (or
  a local `.env` file you control).
- **Generated images, video clips, audio files, and their manifests**:
  written to the path you specify, or, when you don't specify one,
  auto-named under `OPENROUTER_MCP_OUTPUT_DIR` if set, else
  `$HOME/Downloads/openrouter-mcp`, else the system temp directory. Always
  your local disk; this software never uploads them anywhere.
- **Usage statistics**: kept in memory for the lifetime of the server process
  only, and lost when it stops. Nothing is persisted or transmitted.

## Data retention

The software retains nothing beyond the files it writes to disk at your request.
Uninstalling the extension and deleting any generated files removes all data.

## Contact

Questions or concerns: open an issue at
<https://github.com/thesimj/rust-openrouter-mcp/issues>.
