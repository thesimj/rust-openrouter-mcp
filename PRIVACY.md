# Privacy Policy

_Last updated: 2026-06-17_

`openrouter-mcp` is a local Model Context Protocol (MCP) server that runs
entirely on your own machine. It is a thin client for the
[OpenRouter](https://openrouter.ai) API.

## What data is collected

The author of `openrouter-mcp` **collects nothing**. There is no telemetry,
no analytics, and no remote logging of any kind. The software has no servers
of its own.

## What the software sends, and to whom

`openrouter-mcp` communicates with exactly one third party - **OpenRouter** -
and only to perform the actions you (or your AI assistant) explicitly request:

- **Model discovery** (`list_models`): sends your query/filter parameters to
  OpenRouter's `/models` endpoint.
- **Image generation/editing** (`generate_image`): sends your prompt and any
  local input images you provide to the image model you select.
- **Image description** (`describe_image`): sends the local images you provide
  and your prompt to the vision model you select.

Your **OpenRouter API key** is sent to OpenRouter to authenticate these
requests. It is never sent anywhere else.

OpenRouter's handling of this data is governed by OpenRouter's own
[Privacy Policy](https://openrouter.ai/privacy) and
[Terms of Service](https://openrouter.ai/terms).

## Where data is stored

- **API key**: when installed as a Claude Desktop extension, your API key is
  stored by Claude Desktop in your operating system's secure keychain. When run
  from the CLI it is read from the `OPENROUTER_API_KEY` environment variable (or
  a local `.env` file you control).
- **Generated images and manifests**: written only to the output paths you
  specify, on your local disk.
- **Usage statistics**: kept in memory for the lifetime of the server process
  only, and lost when it stops. Nothing is persisted or transmitted.

## Data retention

The software retains nothing beyond the files it writes to disk at your request.
Uninstalling the extension and deleting any generated files removes all data.

## Contact

Questions or concerns: open an issue at
<https://github.com/thesimj/rust-openrouter-mcp/issues>.
