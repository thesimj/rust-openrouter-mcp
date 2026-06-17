# Connecting `openrouter-mcp` to your client

`openrouter-mcp` is a **local stdio MCP server** — a single binary that speaks the
Model Context Protocol over stdin/stdout. Almost every MCP-capable CLI, agent, and
editor can use it. This guide gives the exact config for the popular ones.

> **Claude Desktop** users: don't use this file — install the one-click
> [`.mcpb` extension](README.md#add-to-claude-desktop-one-click) instead.

## Prerequisites

1. **Install the binary** so it's on your `PATH`:
   ```bash
   cargo install openrouter-mcp
   ```
   Check the location with `which openrouter-mcp` (e.g. `~/.cargo/bin/openrouter-mcp`).
   If a client can't find it on `PATH`, use that **absolute path** as the `command`.
2. **Get an API key** at <https://openrouter.ai/keys>.

## The launch contract (same everywhere)

| Field | Value |
| --- | --- |
| command | `openrouter-mcp` |
| args | `["mcp"]` |
| env | `OPENROUTER_API_KEY=sk-or-...` |

Every client below is just a different way of expressing those three things. The
canonical JSON shape — used by **Claude Code, Cursor, Windsurf, Cline, Roo Code,
Gemini CLI, Antigravity** — is:

```json
{
  "mcpServers": {
    "openrouter": {
      "command": "openrouter-mcp",
      "args": ["mcp"],
      "env": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```

> 🔐 **Keep your key out of committed files.** Prefer a client's "add" CLI or
> shell-variable expansion (noted per client) over hardcoding the key into a
> config you might check into git.

---

## CLI agents

### Claude Code (CLI)
One command (no file editing):
```bash
# this project only (default)
claude mcp add openrouter --env OPENROUTER_API_KEY=sk-or-... -- openrouter-mcp mcp
# all your projects
claude mcp add --scope user openrouter --env OPENROUTER_API_KEY=sk-or-... -- openrouter-mcp mcp
# shared with your team (writes a committable .mcp.json)
claude mcp add --scope project openrouter --env OPENROUTER_API_KEY=sk-or-... -- openrouter-mcp mcp
```
Project scope writes `.mcp.json` (canonical `mcpServers` shape) and prompts for
approval on first use. Docs: <https://code.claude.com/docs/en/mcp-quickstart>

### OpenAI Codex CLI
**TOML**, in `~/.codex/config.toml` (note the `mcp_servers` underscore + plural):
```toml
[mcp_servers.openrouter]
command = "openrouter-mcp"
args = ["mcp"]

[mcp_servers.openrouter.env]
OPENROUTER_API_KEY = "sk-or-..."
```
Or: `codex mcp add openrouter --env OPENROUTER_API_KEY=sk-or-... -- openrouter-mcp mcp`.
Docs: <https://developers.openai.com/codex/mcp>

### Google Gemini CLI
`~/.gemini/settings.json` (or `.gemini/settings.json` per project), key `mcpServers`.
Gemini CLI expands `$VAR` inside `env`, so you can avoid hardcoding:
```json
{
  "mcpServers": {
    "openrouter": {
      "command": "openrouter-mcp",
      "args": ["mcp"],
      "env": { "OPENROUTER_API_KEY": "$OPENROUTER_API_KEY" }
    }
  }
}
```
Or: `gemini mcp add openrouter openrouter-mcp mcp -e OPENROUTER_API_KEY=sk-or-...`.
Docs: <https://google-gemini.github.io/gemini-cli/docs/tools/mcp-server.html>

### Google Antigravity
Shared config at `~/.gemini/config/mcp_config.json`, key `mcpServers` (canonical
shape above). Add via Settings → Customizations → **Open MCP Config**, then hit
refresh in *Installed MCP Servers*.
Docs: <https://composio.dev/content/howto-mcp-antigravity>

### opencode
`opencode.json` (project) or `~/.config/opencode/opencode.json`. **Different shape**:
key is `mcp`, `type` is `local`, `command` is a **single array**, and env is
`environment`:
```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "openrouter": {
      "type": "local",
      "command": ["openrouter-mcp", "mcp"],
      "enabled": true,
      "environment": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```
Docs: <https://opencode.ai/docs/mcp-servers/>

### Crush (Charm)
`crush.json` / `.crush.json` (project) or `~/.config/crush/crush.json`, key `mcp`,
`type: "stdio"`:
```json
{
  "$schema": "https://charm.land/crush.json",
  "mcp": {
    "openrouter": {
      "type": "stdio",
      "command": "openrouter-mcp",
      "args": ["mcp"],
      "env": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```
Docs: <https://github.com/charmbracelet/crush>

### Goose (Block)
**YAML**, `~/.config/goose/config.yaml`, under `extensions` (note `cmd`, not
`command`, and `envs`, not `env`):
```yaml
extensions:
  openrouter:
    name: OpenRouter
    type: stdio
    cmd: openrouter-mcp
    args: [mcp]
    enabled: true
    timeout: 300
    envs:
      OPENROUTER_API_KEY: "sk-or-..."
```
Easiest: run `goose configure` → Add Extension → Command-line Extension (stdio) →
command `openrouter-mcp mcp` → add `OPENROUTER_API_KEY`.
Docs: <https://block.github.io/goose/docs/getting-started/using-extensions>

### Amp (Sourcegraph)
`~/.config/amp/settings.json`, key `amp.mcpServers`:
```json
{
  "amp.mcpServers": {
    "openrouter": {
      "command": "openrouter-mcp",
      "args": ["mcp"],
      "env": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```
Or: `amp mcp add openrouter --env OPENROUTER_API_KEY=... -- openrouter-mcp mcp`.
Docs: <https://ampcode.com/manual>

### OpenHands (All Hands AI)
`config.toml` (project or `~/.openhands/config.toml`), `[mcp]` → `stdio_servers`:
```toml
[mcp]
stdio_servers = [
  { name = "openrouter", command = "openrouter-mcp", args = ["mcp"], env = { OPENROUTER_API_KEY = "sk-or-..." } }
]
```
Docs: <https://docs.openhands.dev/openhands/usage/settings/mcp-settings>

---

## Editors & IDEs

### Cursor
`.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global) — canonical
`mcpServers` shape. Docs: <https://cursor.com/docs/mcp>

### Windsurf (Codeium)
`~/.codeium/windsurf/mcp_config.json` (Linux: `~/.config/windsurf/mcp_config.json`)
— canonical `mcpServers` shape. Edit via Cascade → Plugins → *View raw config*;
Windsurf hot-reloads on save. Docs: <https://docs.windsurf.com/windsurf/cascade/mcp>

### VS Code (GitHub Copilot, agent mode) ⚠️
`.vscode/mcp.json` — **the odd one out**: top-level key is `servers` (not
`mcpServers`) and `type: "stdio"` is **required**:
```json
{
  "servers": {
    "openrouter": {
      "type": "stdio",
      "command": "openrouter-mcp",
      "args": ["mcp"],
      "env": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```
Docs: <https://code.visualstudio.com/docs/copilot/customization/mcp-servers>

### Zed ⚠️
`settings.json` (`cmd-,`) — key is `context_servers` (not `mcpServers`):
```json
{
  "context_servers": {
    "openrouter": {
      "command": "openrouter-mcp",
      "args": ["mcp"],
      "env": { "OPENROUTER_API_KEY": "sk-or-..." }
    }
  }
}
```
Docs: <https://zed.dev/docs/ai/mcp>

---

## VS Code extensions

### Cline
Click the **MCP Servers** icon → *Configure MCP Servers* to open
`cline_mcp_settings.json` (key `mcpServers`, canonical shape; supports `disabled`
and `autoApprove` fields).

### Roo Code
`.roo/mcp.json` (project) or the global settings file via the MCP UI — key
`mcpServers`, canonical shape. Supports `${env:OPENROUTER_API_KEY}` expansion.
Docs: <https://docs.roocode.com/features/mcp/using-mcp-in-roo>

### Continue
**YAML** at `~/.continue/config.yaml` — `mcpServers` is a **list**:
```yaml
mcpServers:
  - name: openrouter
    type: stdio
    command: openrouter-mcp
    args: [mcp]
    env:
      OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```
Docs: <https://docs.continue.dev/customize/deep-dives/mcp>

---

## Format gotchas at a glance

Most clients use `command` + `args` + `env` under a `mcpServers` object. The
exceptions:

| Client | Key | Notable difference |
| --- | --- | --- |
| VS Code (Copilot) | `servers` | `type: "stdio"` required |
| Zed | `context_servers` | — |
| Goose | `extensions` | YAML; `cmd` not `command`; `envs` not `env` |
| opencode | `mcp` | `type: "local"`; `command` is one array; env is `environment` |
| Crush | `mcp` | `type: "stdio"` |
| Codex CLI | `[mcp_servers.*]` | TOML; `env` is a sub-table |
| OpenHands | `[mcp] stdio_servers` | TOML array of inline tables |
| Continue | `mcpServers` (list) | YAML list of `{name, ...}` |

## Inline image previews per client

`generate_image` always saves to disk **and**, by default (`auto`), returns the
image inline for clients that can't read your filesystem. Local CLIs that share
your filesystem (detected as `claude-code`) get paths only — they can open the
file directly. Force it either way with `OPENROUTER_MCP_IMAGE_PREVIEWS=always|never`
in the server's `env`. See [Configuration](README.md#configuration).

## Verify the connection

Most clients list discovered tools after connecting. You should see:
`list_models`, `generate_image`, `get_result`, `describe_image`,
`get_usage_stats`, `reset_usage_stats`. Ask the agent to *"list OpenRouter image
models"* to confirm `list_models` runs.

You can also sanity-check the binary by hand:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  | OPENROUTER_API_KEY=sk-or-... openrouter-mcp mcp
```
A JSON line naming the `rmcp` server back means stdio is healthy.

---

<sub>Not included: **Aider** has no native MCP client support as of 2026 (tracked in
Aider issue #4506). "openclaw" and "Hermes" were investigated and are **not**
real, popular MCP coding clients — avoid configs claiming otherwise.</sub>
