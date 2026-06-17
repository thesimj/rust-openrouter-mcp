#!/usr/bin/env bash
# Build a macOS-only Claude Desktop extension (.mcpb) for openrouter-mcp.
#
# Produces a universal (arm64 + x86_64) binary, drops it into mcpb/bin/, and
# packs mcpb/ into dist/openrouter-mcp.mcpb via @anthropic-ai/mcpb.
#
# Usage: scripts/build-mcpb.sh
# Requires: rustup, cargo, lipo (Xcode CLT), node/npx.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN_NAME="openrouter-mcp"
MCPB_DIR="$ROOT/mcpb"
DIST_DIR="$ROOT/dist"

echo "==> Ensuring macOS Rust targets are installed"
rustup target add aarch64-apple-darwin x86_64-apple-darwin

echo "==> Building release binaries"
cargo build --release --locked --target aarch64-apple-darwin
cargo build --release --locked --target x86_64-apple-darwin

echo "==> Creating universal binary with lipo"
mkdir -p "$MCPB_DIR/bin"
lipo -create -output "$MCPB_DIR/bin/$BIN_NAME" \
  "target/aarch64-apple-darwin/release/$BIN_NAME" \
  "target/x86_64-apple-darwin/release/$BIN_NAME"
chmod +x "$MCPB_DIR/bin/$BIN_NAME"
lipo -info "$MCPB_DIR/bin/$BIN_NAME"

echo "==> Refreshing icon"
node "$MCPB_DIR/make-icon.mjs"

echo "==> Validating manifest"
npx -y @anthropic-ai/mcpb validate "$MCPB_DIR/manifest.json"

echo "==> Packing .mcpb"
mkdir -p "$DIST_DIR"
npx -y @anthropic-ai/mcpb pack "$MCPB_DIR" "$DIST_DIR/$BIN_NAME.mcpb"

echo "==> Done: $DIST_DIR/$BIN_NAME.mcpb"
