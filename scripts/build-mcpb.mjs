#!/usr/bin/env node
// Build a per-platform Claude Desktop extension (.mcpb) for openrouter-mcp.
//
// Cross-platform (runs on Linux, macOS, and Windows via Node). Builds the
// release binary for the host platform, stages it with a patched manifest and
// the committed icon, and packs:
//
//   dist/openrouter-mcp-<os>.mcpb      where <os> is linux | macos | windows
//
// The manifest's version is read from Cargo.toml (single source of truth, so it
// can never drift from the crate), and entry_point / command / platforms are
// set for the target. On macOS a universal (arm64 + x86_64) binary is produced
// via lipo when both targets are installed; otherwise the native arch is used.
//
// Usage:  node scripts/build-mcpb.mjs
// Requires: cargo, node/npx. macOS universal builds also need lipo (Xcode CLT).

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync, rmSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const BIN_NAME = "openrouter-mcp";

// Map Node's process.platform to a friendly OS slug + manifest values.
const PLATFORMS = {
  linux: { slug: "linux", nodePlatform: "linux", exe: "" },
  darwin: { slug: "macos", nodePlatform: "darwin", exe: "" },
  win32: { slug: "windows", nodePlatform: "win32", exe: ".exe" },
};

const platform = PLATFORMS[process.platform];
if (!platform) {
  console.error(`Unsupported platform: ${process.platform}`);
  process.exit(1);
}

const run = (cmd, args, opts = {}) =>
  execFileSync(cmd, args, { cwd: ROOT, stdio: "inherit", shell: false, ...opts });

// `npx` is the `npx.cmd` batch shim on Windows, which can only be launched via a
// shell. All args here are trusted static paths, so shell quoting is not a concern.
const npx = (args) => run("npx", args, { shell: process.platform === "win32" });

// Single source of truth for the version: Cargo.toml [package] version.
function crateVersion() {
  const toml = readFileSync(join(ROOT, "Cargo.toml"), "utf8");
  const m = toml.match(/^\s*version\s*=\s*"([^"]+)"/m);
  if (!m) throw new Error("could not read version from Cargo.toml");
  return m[1];
}

// Build the release binary for the host. On macOS, attempt a universal binary.
function buildBinary(stageBinDir) {
  const out = join(stageBinDir, `${BIN_NAME}${platform.exe}`);
  if (process.platform === "darwin") {
    const targets = ["aarch64-apple-darwin", "x86_64-apple-darwin"];
    console.log("==> Building universal macOS binary");
    for (const t of targets) {
      run("rustup", ["target", "add", t]);
      run("cargo", ["build", "--release", "--locked", "--target", t]);
    }
    run("lipo", [
      "-create", "-output", out,
      join(ROOT, "target", targets[0], "release", BIN_NAME),
      join(ROOT, "target", targets[1], "release", BIN_NAME),
    ]);
    run("lipo", ["-info", out]);
  } else {
    console.log("==> Building release binary");
    run("cargo", ["build", "--release", "--locked"]);
    copyFileSync(join(ROOT, "target", "release", `${BIN_NAME}${platform.exe}`), out);
  }
}

function main() {
  const version = crateVersion();
  const distDir = join(ROOT, "dist");
  const stageDir = join(distDir, `stage-${platform.slug}`);
  const stageBinDir = join(stageDir, "bin");
  const outFile = join(distDir, `${BIN_NAME}-${platform.slug}.mcpb`);

  console.log(`==> Packing ${BIN_NAME} v${version} for ${platform.slug}`);

  rmSync(stageDir, { recursive: true, force: true });
  mkdirSync(stageBinDir, { recursive: true });
  mkdirSync(distDir, { recursive: true });

  // Patch the committed base manifest for this target.
  const manifest = JSON.parse(readFileSync(join(ROOT, "mcpb", "manifest.json"), "utf8"));
  manifest.version = version;
  manifest.server.entry_point = `bin/${BIN_NAME}${platform.exe}`;
  manifest.server.mcp_config.command = `\${__dirname}/bin/${BIN_NAME}${platform.exe}`;
  manifest.compatibility = manifest.compatibility || {};
  manifest.compatibility.platforms = [platform.nodePlatform];

  writeFileSync(join(stageDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
  copyFileSync(join(ROOT, "mcpb", "icon.png"), join(stageDir, "icon.png"));

  buildBinary(stageBinDir);

  console.log("==> Validating manifest");
  npx(["-y", "@anthropic-ai/mcpb", "validate", join(stageDir, "manifest.json")]);

  console.log("==> Packing .mcpb");
  npx(["-y", "@anthropic-ai/mcpb", "pack", stageDir, outFile]);

  rmSync(stageDir, { recursive: true, force: true });

  signBundle(outFile);
  console.log(`==> Done: ${outFile}`);
}

// Sign the packed bundle with the mcpb toolchain (appends a PKCS#7 SignedData
// block). Uses an explicit cert/key when MCPB_SIGN_CERT + MCPB_SIGN_KEY point at
// PEM files (e.g. a stable identity injected from CI secrets), otherwise falls
// back to a self-signed certificate. Best-effort: a signing failure warns but
// does not fail the build, since the unsigned bundle is still installable.
//
// Note: `mcpb verify` is broken in v2.1.2 (reports "not signed" for a correctly
// signed bundle), so we do not gate on it here.
function signBundle(outFile) {
  const cert = process.env.MCPB_SIGN_CERT;
  const key = process.env.MCPB_SIGN_KEY;
  const args = ["-y", "@anthropic-ai/mcpb", "sign", outFile];
  if (cert && key) {
    console.log("==> Signing .mcpb (cert/key from env)");
    args.push("-c", cert, "-k", key);
    if (process.env.MCPB_SIGN_INTERMEDIATE) {
      args.push("-i", process.env.MCPB_SIGN_INTERMEDIATE);
    }
  } else {
    console.log("==> Signing .mcpb (self-signed)");
    args.push("--self-signed");
  }
  try {
    npx(args);
  } catch (e) {
    console.warn(`WARNING: signing failed, shipping unsigned bundle: ${e.message}`);
  }
}

main();
