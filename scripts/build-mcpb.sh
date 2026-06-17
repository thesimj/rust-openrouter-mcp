#!/usr/bin/env bash
# Build a platform-specific .mcpb bundle for openrouter-mcp.
#
# Produces dist/openrouter-mcp.mcpb containing the manifest and the release
# binary under server/. The binary is native to the machine this runs on, so
# build on each target OS you want to ship (macOS / Linux / Windows).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN_NAME="openrouter-mcp"
STAGE="$ROOT/target/mcpb"
DIST="$ROOT/dist"

# Windows binaries carry the .exe suffix; the manifest entry_point stays
# extension-less (the host appends .exe automatically for binary servers).
EXT=""
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT) EXT=".exe" ;;
esac

echo ">> Building release binary"
cargo build --release

echo ">> Staging bundle in $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE/server"
cp "$ROOT/bundle/manifest.json" "$STAGE/manifest.json"
cp "$ROOT/target/release/${BIN_NAME}${EXT}" "$STAGE/server/${BIN_NAME}${EXT}"
chmod +x "$STAGE/server/${BIN_NAME}${EXT}" || true

mkdir -p "$DIST"
OUT="$DIST/${BIN_NAME}.mcpb"
rm -f "$OUT"

# Prefer the official mcpb CLI (validates the manifest); fall back to zip.
if command -v mcpb >/dev/null 2>&1; then
  echo ">> Packing with mcpb CLI"
  mcpb pack "$STAGE" "$OUT"
elif command -v npx >/dev/null 2>&1 && npx --no-install @anthropic-ai/mcpb --version >/dev/null 2>&1; then
  echo ">> Packing with npx @anthropic-ai/mcpb"
  npx --no-install @anthropic-ai/mcpb pack "$STAGE" "$OUT"
else
  echo ">> mcpb CLI not found; packing with zip"
  ( cd "$STAGE" && zip -r -q "$OUT" manifest.json server )
fi

echo ">> Built $OUT"
unzip -l "$OUT" 2>/dev/null || true
