**The working tree implements fixes for all 12 audit findings and adds 23 regression tests.**

This report follows the [original audit](codebase-audit-2026-09-07.md).
The original report describes commit `7a0127e26b91c6f4ec2f6495f3b8e19cb7131fef` before these changes.
The [API report](openrouter-api-validation-2026-09-07.md) retains the official endpoint references and public-response evidence.
No paid or authenticated OpenRouter calls were needed for these fixes.

| Finding | Change | Regression coverage |
| --- | --- | --- |
| F1: SVG file access | Both SVG parsers reject external image references through the string resolver. | Absolute, relative, and nested file references render transparent. Embedded SVG data still renders. |
| F2: Generated names lose suffixes | Auto-name tokens contain no dots, preserving configuration and the nonce during extension replacement. | Image, video, audio, manifest, and variant paths retain the complete basename. |
| F3: TLS build mismatch | Reqwest uses Rustls and the platform verifier. The lockfile removes native-tls and openssl-sys. | Client construction passes. Native macOS and static Linux musl release builds pass. |
| F4: Wrong image price units | Dedicated image pricing follows each explicit `unit`. Generic `image_output` remains marked ambiguous. | Token, image, megapixel, missing-unit, variant, and CLI table cases. |
| F5: Wrong video price units | Video SKU handling preserves megapixel-second units and retains unknown SKU names. | Rates of 7.5–10.5 cents render as $0.075–0.105/MP-s. Mixed units remain separate. |
| F6: Incorrect billing | Receipts survive decoding and file-write failures. Video cost belongs to the request, independently of clip count. | Mixed image outcomes, video download/write failures, multiple clips, unknown costs, and failed text accounting. |
| F7: Abandoned video jobs | The manifest records `job_id` before polling. Polling retries 408, 429, 5xx, and transient transport failures. | Fake-clock tests cover Retry-After, stalled requests, permanent failures, and the overall deadline. Delivery tests retain recovery metadata. |
| F8: Missing tool error flag | An immediately failed generation returns `isError: true`. | The original generation fails while a subsequent status lookup succeeds. |
| F9: Invalid video links | Links use absolute, escaped file URIs and each clip's actual media type. | Relative paths, spaces, Unicode, fragment characters, and WebM MIME. |
| F10: Dropped cache price | `Pricing` retains `input_cache_write_1h`. | DTO-to-tool JSON round trip retains the raw rate and formatted price. |
| F11: Nullable video pricing | Null and omitted `pricing_skus` deserialize as empty maps. | One catalog contains null, missing, and populated pricing entries. |
| F12: Duplicate seed paths | Seeded filenames also contain the variant index. | Sixteen variants remain distinct at `u64::MAX`. |

Additional corrections apply the local 25 MiB transcription limit to files and inline data.
Inline validation rejects malformed base64 and normalizes MIME aliases, including `audio/mpeg` to `mp3`.
Video failures retain the provider explanation and any reported billing.
The task registry rejects new generation jobs once 32 pending jobs exist.

Remote image fetching includes DNS resolution within its deadline.
It bypasses proxies to preserve the validated destination address.
IPv6 literal handling removes URL brackets before address resolution.
Reserved and multicast IPv4 addresses also fail validation.

Tool descriptions now describe synchronous provider waits without promising fast completion.
README clarifies local audio limits, filename changes, billing, task admission, and file-link access requirements.
The release workflow requires CI formatting, lint, tests, and the Rust 1.88 check before packaging.
The packer uses version 2.1.2 from the [official MCPB package manifest](https://github.com/modelcontextprotocol/mcpb/blob/main/package.json).
That version pin does not lock the packer's transitive npm dependencies.

| Validation | Result |
| --- | --- |
| `cargo test --locked --offline audit_regression` | All 23 new tests passed. These tests need no sockets or network. |
| `cargo test --locked --offline` | All 206 tests passed, including all 70 HTTP tests. No tests were ignored or filtered. |
| HTTP fixture correction | The endpoint fixture now includes `unit: "token"` and verifies that 0.00003 USD renders as $30/M tokens. |
| `cargo clippy --all-targets --all-features --locked --offline -- -D warnings` | Passed. |
| `cargo fmt --check` and `git diff --check` | Passed. |
| `cargo build --release --locked --offline` | Passed for aarch64 macOS. |
| `cargo zigbuild --release --locked --offline --target x86_64-unknown-linux-musl` | Passed. `file` identifies a statically linked x86-64 ELF executable. |
| Linux dependency tree | `openssl-sys` is absent. Rustls uses aws-lc, which requires C build tools. |
| Native CLI `--help` | Passed. |
| `cargo audit --no-fetch --no-yanked` | No vulnerabilities found using the cached database of 1,216 advisories. |
| `actionlint` on both workflows | Passed. |
| `node --check scripts/build-mcpb.mjs` | Passed. |

Run `python3 docs/audit/reproduce.py` to execute the permanent regressions.
Run `cargo test --locked --offline` directly to execute the full suite.
Earlier invocations with shell output redirection failed at the mock-server socket bind.
Direct Cargo invocations subsequently ran all HTTP tests successfully in this session.
The full run exposed one stale fixture, which omitted the explicit image pricing unit.
After correcting that fixture, all 206 tests passed. No sandbox policy change or test exclusion was required.

The Linux binary was cross-compiled with Zig and was not executed here.
Windows builds, universal macOS packaging, and the exact GitHub release workflow were not executed here.
Rust 1.88 was unavailable locally, so its validation remains the CI gate.
The cached advisory scan does not prove the absence of newer advisories.

No live TLS handshake, corporate certificate deployment, proxy route, or IPv6 network connection was tested during implementation.
DNS timeouts bound the caller's wait, but a system resolver thread may continue after that wait expires.
The pending-job cap does not bound concurrent synchronous chat or audio calls.
Video links require shared filesystem access. The server does not provide remote resource reads.

Graph status and coverage tools were denied under the session's approval policy.
Source inspection supplied the fallback evidence. Graph freshness remains unverified.
