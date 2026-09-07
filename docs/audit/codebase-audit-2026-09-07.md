**The audit found three high-priority defects and several API, accounting, and protocol defects.**

Audited commit: `7a0127e26b91c6f4ec2f6495f3b8e19cb7131fef` (`0.7.2`), on 2026-09-07.
The working tree was clean when the audit started.
This report records the original defects before the authorized fixes.
The [fix report](fix-validation-2026-09-07.md) describes the current changes and verification limits.
`reproduce.py` now runs permanent regression tests for corrected behavior.

The review covered the Rust application, REST client, DTOs, MCP tools, CLI, media pipelines, task registry, pricing, manifests, and release configuration.
All 13 implemented OpenRouter method/path combinations were compared with current official sources.
The [API validation report](openrouter-api-validation-2026-09-07.md) contains the endpoint matrix, public GET results, and source references.
Unimplemented OpenRouter features are capability gaps, not failed implementations.

| ID | Priority | Finding | Evidence |
| --- | --- | --- | --- |
| F1 | P1 | Untrusted SVG content can include local SVG files | Reproduced with synthetic files |
| F2 | P1 | Dotted model names discard filename suffixes and cause output collisions | Reproduced |
| F3 | P1 | TLS dependencies conflict with the Linux release recipe | Dependency tree and source |
| F4 | P2 | Image pricing ignores the explicit billing unit | Live API response and reproduction |
| F5 | P2 | Video pricing drops the megapixel factor | Live API response and official model page |
| F6 | P2 | Usage counters lose paid failures and duplicate multi-clip costs | Source trace |
| F7 | P2 | A transient video poll failure abandons the accepted job | Source trace |
| F8 | P2 | Immediate generation failures return MCP success | Reproduced |
| F9 | P2 | Video resource links misidentify relative paths and media types | URI reproduction and source |
| F10 | P2 | Model listings discard the one-hour cache-write price | Live catalog and DTO |
| F11 | P3 | Nullable video pricing rejects the entire typed catalog | Schema-valid reproduction; not observed live |
| F12 | P3 | Saturated seeds produce identical variant paths | Reproduced |

P1 means a defect deserves attention before the next release.
P2 means a correctness defect needs a planned fix.
P3 identifies a narrower condition or compatibility gap.

1. **F1 — Disable local resource resolution when parsing untrusted SVG.**

   [image_io.rs:116](/Users/patron/projects/rust-openrouter-mcp/src/image_io.rs:116) uses `usvg::Options::default()`.
   The comment incorrectly treats `resources_dir: None` as disabling external resources.
   The default resolver opens local paths. Its behavior is explicit in the [usvg documentation](https://docs.rs/usvg/latest/usvg/struct.ImageHrefResolver.html).

   A remote or inline SVG can reference a readable local SVG through an absolute or relative `xlink:href`.
   `prepare_inputs` rasterizes that content before sending it to OpenRouter.
   The probe rendered a synthetic local red rectangle through an untrusted wrapper SVG.
   Removing the local file made the same output transparent.

   The reproduction proves local SVG inclusion. It does not prove arbitrary text-file disclosure.
   The deployed process must have read access to the referenced file.
   Configure `image_href_resolver.resolve_string` to reject external paths.
   Apply the same policy in `svg_dimensions`, which also constructs default options.
   Test absolute paths, relative paths, and nested SVG references.

2. **F2 — Preserve the entire generated base name when adding extensions.**

   [naming.rs:37](/Users/patron/projects/rust-openrouter-mcp/src/server/naming.rs:37) retains dots in model names.
   [job.rs:113](/Users/patron/projects/rust-openrouter-mcp/src/image_gen/job.rs:113) later calls `with_extension` for single images.
   Video, speech, variant stems, and manifest paths apply equivalent extension or stem handling.

   An extensionless base containing `gemini-3.1-flash-image-preview_1x1_1K_308a` becomes `..._gemini-3.png`.
   Everything after the final dot disappears, including configuration, seed, and collision suffix.
   The probe generated two distinct names within one second and obtained identical image and manifest paths.
   `write_bytes` overwrites an existing path, so concurrent jobs can lose paid outputs.

   Distinguish generated extensionless bases from caller-supplied filenames, or remove dots from generated tokens.
   Test final artifact paths across image, video, speech, variants, and manifests.
   Also test dotted configuration values such as `0.5K`.

3. **F3 — Restore a TLS backend that supports the promised release build.**

   [Cargo.toml:50](/Users/patron/projects/rust-openrouter-mcp/Cargo.toml:50) enables `native-tls`.
   Its comments and README describe rustls.
   The [release workflow](/Users/patron/projects/rust-openrouter-mcp/.github/workflows/release-mcpb.yml:55) installs musl tools without provisioning musl OpenSSL.
   The [bundle script](/Users/patron/projects/rust-openrouter-mcp/scripts/build-mcpb.mjs:87) then builds `x86_64-unknown-linux-musl`.

   `cargo tree --target x86_64-unknown-linux-musl -i openssl-sys` confirms an OpenSSL dependency through reqwest and native-tls.
   Native-tls uses OpenSSL on Linux. Its vendored feature supplies a separate static-build option. [Native-tls documentation](https://docs.rs/native-tls/latest/native_tls/)
   The current recipe therefore lacks the required cross-target OpenSSL setup.
   A Linux release build was not available locally, so the predicted build failure remains an inference.

   Rustls is the better fit for this project's portable bundle.
   Reqwest 0.13.4 enables the platform verifier with its `rustls` feature.
   Local dependency source confirms the default certificate path uses that verifier.
   Keep certificate verification enabled and validate corporate roots on supported platforms.
   Rustls avoids OpenSSL, but its crypto provider can still require C build tools. [Reqwest TLS documentation](https://docs.rs/reqwest/latest/reqwest/tls/index.html)

4. **F4 — Use each image price's `unit`, rather than guessing from `billable`.**

   [pricing.rs:189](/Users/patron/projects/rust-openrouter-mcp/src/pricing.rs:189) reads `billable` and `cost_usd`, then ignores `unit`.
   The live GPT Image 2 endpoint returned `output_image`, `unit: "token"`, and `cost_usd: 0.00003`.
   The formatter reports `$0.00003/image`; the correct token display is `$30/M tokens`.
   The probe reproduces this exact transformation. [OpenRouter image endpoint](https://openrouter.ai/api/v1/images/models/openai/gpt-image-2/endpoints)

   The CLI also places `image_output` under `$/IMG`, while the JSON formatter calls it an output-token price.
   See [table.rs:128](/Users/patron/projects/rust-openrouter-mcp/src/cli/table.rs:128).
   The generic OpenAPI description and live token pricing disagree for this field.
   Prefer explicit endpoint units and disclose ambiguity when only generic pricing exists.
   Test the same fixture through both JSON and table rendering.

5. **F5 — Preserve megapixel-second units in video pricing.**

   [pricing.rs:59](/Users/patron/projects/rust-openrouter-mcp/src/pricing.rs:59) treats every remaining key containing `second` as cents per second.
   `humanize_price` repeats this assumption.
   FLUX Video Upscale returns `cents_per_megapixel_second_precise: "7.5"` and `cents_per_megapixel_second_creative: "10.5"`.
   The formatter displays `$0.075–0.105/s`, dropping the megapixel factor.
   OpenRouter identifies these prices as megapixel-second rates. [Official model page](https://openrouter.ai/black-forest-labs/flux-video-upscale)

   Decode known SKU units explicitly.
   Preserve unknown SKU names instead of assigning a misleading unit.
   Discovery still exposes this model even though this server lacks its source-video upscaling inputs.

6. **F6 — Track generation cost independently from output delivery.**

   [job.rs:209](/Users/patron/projects/rust-openrouter-mcp/src/image_gen/job.rs:209) assigns image cost and generation ID only after the file write succeeds.
   A provider can complete and charge successfully while a local write fails.
   That branch discards known billing metadata.
   [image.rs:500](/Users/patron/projects/rust-openrouter-mcp/src/server/image.rs:500) then sums costs only from saved images.
   The all-failed branch records zero cost and zero unknown costs.

   Video accounting has the opposite error for multiple clips.
   [job.rs:203](/Users/patron/projects/rust-openrouter-mcp/src/video_gen/job.rs:203) copies the job-level `usage.cost` into every clip.
   [video.rs:255](/Users/patron/projects/rust-openrouter-mcp/src/server/video.rs:255) sums those copies.
   A two-clip response with job cost `$0.90` becomes `$1.80` in usage statistics.
   OpenRouter defines usage on the job response. [Video status API](https://openrouter.ai/docs/api/api-reference/video-generation/poll-video-generation-status)

   Keep request-level billing records separate from file outcomes.
   Record known cost even when saving, decoding, or extracting content fails.
   Count generated clips separately from generation requests.
   Add tests for write failure, mixed success, and multi-clip responses.
   These conclusions follow source branches; the sandbox blocked HTTP-backed reproductions.

7. **F7 — Preserve accepted video jobs across transient polling errors.**

   [job.rs:169](/Users/patron/projects/rust-openrouter-mcp/src/video_gen/job.rs:169) propagates any poll error immediately with `result?`.
   A temporary timeout, 429, or 5xx therefore ends local tracking after a successful submission.
   The function exits before writing its manifest.
   The returned HTTP error does not include the upstream job ID.
   `get_result` cannot resume polling because the task becomes terminal.

   Persist or return the upstream job ID immediately after submission.
   Retry transient GET failures within the existing deadline.
   Respect `Retry-After` where available.
   Keep authentication and permanent errors terminal.
   Avoid retrying the billable POST without an idempotency strategy.
   This failure path was inspected, but socket restrictions prevented its mock-server reproduction.

8. **F8 — Mark immediate generation failures as MCP tool errors.**

   [result.rs:220](/Users/patron/projects/rust-openrouter-mcp/src/server/result.rs:220) always returns `CallToolResult::success`.
   A failed background job can finish within the initial wait window.
   The generation call then returns `status: "failed"` inside text, with `isError: false` outside it.
   The probe confirms this contradictory result.

   Set `isError: true` when the generation tool itself fails.
   MCP describes execution failures as tool results carrying this flag. [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#error-handling)
   A successful `get_result` lookup may intentionally report a failed job without failing the lookup.
   Define that distinction explicitly instead of sharing one unconditional success constructor.

9. **F9 — Build valid video resource links and preserve the actual MIME type.**

   [result.rs:103](/Users/patron/projects/rust-openrouter-mcp/src/server/result.rs:103) concatenates `file://` with an arbitrary path.
   It also hardcodes `video/mp4`.
   A relative output `out/clip.mp4` becomes `file://out/clip.mp4`, where `out` is the URI host.
   The probe confirms that interpretation.
   WebM and MOV files also receive the wrong MIME type despite the download pipeline preserving their types.

   Resolve a file path to an absolute path before using `Url::from_file_path`.
   Carry each clip's actual MIME type into its resource link.
   A file link cannot itself grant a sandboxed client access to the server's filesystem.
   This server advertises no resource reader, so the claimed sandbox preview behavior needs client validation or another delivery mechanism.
   Desktop preview behavior was not tested here.

10. **F10 — Preserve `input_cache_write_1h` in model listings.**

    [models.rs:94](/Users/patron/projects/rust-openrouter-mcp/src/openrouter/dto/models.rs:94) defines a closed pricing structure without this field.
    The live catalog contained 31 records with the one-hour cache-write rate.
    Deserialization drops it before the CLI and MCP serialize model listings.
    The formatter already supports its name, but never receives the discarded value.
    The API validation report records the live examples and current schema.

    Add the field or preserve unknown pricing fields through a flattened map.
    Verify deserialize-to-output behavior with a captured catalog fixture.
    `describe_model` preserves raw JSON and does not share this omission.

11. **F11 — Accept explicitly null video pricing.**

    [video.rs:24](/Users/patron/projects/rust-openrouter-mcp/src/openrouter/dto/video.rs:24) uses `BTreeMap<String, String>` with `serde(default)`.
    A default handles an absent field, but it does not handle explicit `null`.
    The current OpenAPI schema permits null `pricing_skus`.
    A schema-valid record therefore rejects the entire typed video model response.
    The probe reports `invalid type: null, expected a map`.

    Current public results contained 28 video models without null pricing.
    This is a latent compatibility defect, not an observed catalog outage.
    Its application impact is missing video pricing in CLI table output.
    `describe_model` uses an untyped lookup and remains unaffected.
    Use an optional map or a null-to-empty deserializer.

12. **F12 — Reject seed ranges that overflow, or name variants independently from seeds.**

    [job.rs:40](/Users/patron/projects/rust-openrouter-mcp/src/image_gen/job.rs:40) uses `saturating_add` for successive seeds.
    [job.rs:115](/Users/patron/projects/rust-openrouter-mcp/src/image_gen/job.rs:115) names each variant using only that seed.
    Near `u64::MAX`, different variants receive the same seed and path.
    The second successful write overwrites the first.
    The probe confirms identical paths for two variants starting at `u64::MAX`.

    Reject overflow before sending requests, or include the variant index in every filename.
    This data-loss condition requires an extreme seed and a provider that accepts it.

**Additional observations**

The transcription path enforces a local 25 MiB file limit while calling it a universal OpenRouter limit.
Current docs distinguish multipart uploads from larger base64 JSON requests.
This client sends JSON. Keep the local policy if desired, but document its actual scope.
Inline transcription data also bypasses the file-size check and local format validation.
Standard data URL MIME aliases such as `audio/mpeg` become `mpeg`, rather than the accepted `mp3` format.

Video failure responses contain an upstream `error` field, but the DTO drops it.
Terminal failures consequently report only status and job ID.
The polling loop correctly handles cancelled and expired statuses.

The task registry bounds retained terminal results, but it never bounds pending jobs.
The four-request semaphore applies per image job, not across the process.
Resource exhaustion under concurrent calls remains a load-testing gap.
Remote image DNS lookup also occurs before the configured HTTP deadline.
Proxy routing and IPv6 literal behavior need separate network tests.

Speech and chat tools describe their calls as fast despite sharing a 300-second read timeout.
Whether clients time out first depends on their settings and the chosen provider.
No client-specific timeout guarantee was validated.

Release packaging fetches unversioned `@anthropic-ai/mcpb` through `npx -y`.
Pinning the packer would improve release reproducibility.
The release workflow also lacks an explicit dependency on the test workflow.
No release was built or published during this audit.

Several module comments still describe image generation through chat completions.
The implementation correctly uses `/images`.
README calls `chat_completion` a write tool, while its actual annotation declares it read-only.
The connection guide's third-party client setup claims were read but not revalidated against every client vendor.

**Validation and limits**

| Check | Result |
| --- | --- |
| `cargo fmt --check` | Passed |
| `cargo clippy --all-targets --all-features --locked --offline -- -D warnings` | Passed |
| `cargo test --locked --offline` | 113 passed; 70 blocked by mock-server socket binding |
| Audit probes | Seven passed, confirming the described current defects |
| `node --check scripts/build-mcpb.mjs` | Passed |
| `cargo audit --no-fetch --no-yanked --json` | No findings in cached advisory database |
| Advisory freshness | Unknown; database reports no update timestamp or commit |
| Dependency scope | Cargo audit scanned 249 locked dependencies against 1,216 cached advisories |
| Rust 1.88 minimum | Not rerun; that toolchain is not installed |
| Platform releases | Linux, Windows, and universal macOS release builds not run |
| Authenticated OpenRouter calls | Not run |
| Billable generation calls | Not run |
| Public API probes | Performed without an API key; details in the API report |

All 70 test failures report `Operation not permitted` when wiremock binds a local port.
They do not establish application failures or passing integration behavior.
The seven audit probes run without sockets and use synthetic local files.
They assert observed defects, so passing confirms reproduction rather than correctness.
Run them with `python3 docs/audit/reproduce.py`.
The script copies source into a temporary directory and uses a separate Cargo target directory.

Graph discovery used project `Users-patron-projects-rust-openrouter-mcp`.
`list_projects` reported 1,406 nodes and 4,853 edges.
The audit attempted Tier 3 verification, but approval policy rejected `index_status` and every `check_index_coverage` call.
Generation, freshness, and missed coverage ranges therefore remain unknown.
The final coverage request included all reviewed tracked source and configuration paths plus the repository scope.

The endpoint symbol query returned 66 rows, with no remaining page.
The finding symbol query returned 29 rows, with no remaining page.
Relevant symbols received source snippets and bidirectional traces without truncation.
An initial broad discovery page was narrowed rather than treated as exhaustive evidence.
Graph traces included incorrect name-based edges, so conclusions rely on direct source reads and probes.
No graph completeness claim supports this report.

**Suggested fix order**

1. Block external SVG references and repair final output naming.
2. Restore the TLS build contract and validate release targets.
3. Correct pricing units, billing records, and video recovery.
4. Repair MCP failure flags, resource links, and catalog compatibility.
5. Run the full HTTP test suite and targeted live checks in an environment that permits them.
