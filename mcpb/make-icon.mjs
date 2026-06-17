#!/usr/bin/env node
// Generates mcpb/icon.png — a 512x512 connector icon with no external deps.
// Pure Node: builds an RGBA buffer and encodes a PNG via zlib. Run:
//   node mcpb/make-icon.mjs
import { deflateSync } from "node:zlib";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const S = 512;
const R = 96; // corner radius

// Diagonal brand gradient (indigo -> violet) with rounded corners + an
// aperture-style ring evoking "routing" to many models.
const c0 = [99, 102, 241]; // #6366f1
const c1 = [139, 92, 246]; // #8b5cf6
const lerp = (a, b, t) => Math.round(a + (b - a) * t);

function insideRounded(x, y) {
  const nx = Math.min(x, S - 1 - x);
  const ny = Math.min(y, S - 1 - y);
  if (nx >= R || ny >= R) return true;
  const dx = R - nx;
  const dy = R - ny;
  return dx * dx + dy * dy <= R * R;
}

const buf = Buffer.alloc(S * S * 4);
const cx = S / 2;
const cy = S / 2;
const ringR = 150; // ring radius
const ringW = 34; // ring thickness
const dotR = 46; // center dot
for (let y = 0; y < S; y++) {
  for (let x = 0; x < S; x++) {
    const i = (y * S + x) * 4;
    if (!insideRounded(x, y)) {
      buf[i + 3] = 0;
      continue;
    }
    const t = (x + y) / (2 * S);
    let r = lerp(c0[0], c1[0], t);
    let g = lerp(c0[1], c1[1], t);
    let b = lerp(c0[2], c1[2], t);

    const d = Math.hypot(x - cx, y - cy);
    // White ring
    if (Math.abs(d - ringR) <= ringW / 2) {
      r = g = b = 245;
    }
    // Center dot
    if (d <= dotR) {
      r = g = b = 245;
    }
    buf[i] = r;
    buf[i + 1] = g;
    buf[i + 2] = b;
    buf[i + 3] = 255;
  }
}

// --- minimal PNG encoder (RGBA, 8-bit) ---
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const td = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td) >>> 0, 0);
  return Buffer.concat([len, td, crc]);
}
function crc32(b) {
  let c = ~0;
  for (let i = 0; i < b.length; i++) {
    c ^= b[i];
    for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
  }
  return ~c;
}
const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(S, 0);
ihdr.writeUInt32BE(S, 4);
ihdr[8] = 8; // bit depth
ihdr[9] = 6; // color type RGBA
// rows with filter byte 0
const raw = Buffer.alloc(S * (S * 4 + 1));
for (let y = 0; y < S; y++) {
  raw[y * (S * 4 + 1)] = 0;
  buf.copy(raw, y * (S * 4 + 1) + 1, y * S * 4, (y + 1) * S * 4);
}
const png = Buffer.concat([
  sig,
  chunk("IHDR", ihdr),
  chunk("IDAT", deflateSync(raw, { level: 9 })),
  chunk("IEND", Buffer.alloc(0)),
]);
const out = join(dirname(fileURLToPath(import.meta.url)), "icon.png");
writeFileSync(out, png);
console.log(`wrote ${out} (${png.length} bytes)`);
