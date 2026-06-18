//! The `generate_audio` text-to-speech tool and its argument struct.

use base64::Engine;
use rmcp::{
    ErrorData, RoleServer,
    handler::server::wrapper::Parameters,
    model::{AnnotateAble, CallToolResult, Content, RawAudioContent, RawContent},
    service::RequestContext,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::audio_gen::{self, SpeechGenRequest};
use crate::server::naming;
use crate::server::result::{MAX_INLINE_AUDIO_BYTES, client_wants_inline_previews};
use crate::server::schema::{de_opt_f64, require_all, scalarize_nullable};

use super::OpenRouterServer;

/// Arguments for the `generate_audio` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
pub(crate) struct GenerateAudioArgs {
    /// TTS model id, e.g. "openai/gpt-4o-mini-tts" or "hexgrad/kokoro-82m".
    pub model: String,
    /// REQUIRED (no default): the text to synthesize.
    #[serde(default)]
    pub input: Option<String>,
    /// REQUIRED (no default): voice id (varies by model, e.g. "alloy").
    #[serde(default)]
    pub voice: Option<String>,
    /// Output audio format: "mp3" (default) or "pcm".
    #[serde(default)]
    pub response_format: Option<String>,
    /// Playback speed (select models only).
    #[serde(default, deserialize_with = "de_opt_f64")]
    pub speed: Option<f64>,
    /// Output file path (extension corrected to the returned format, e.g. .mp3).
    /// Optional: when omitted, an auto-named file is written under
    /// OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp).
    #[serde(default)]
    pub output: Option<String>,
}

#[tool_router(router = audio_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Generate speech (text-to-speech) with an OpenRouter TTS model (e.g. \
        openai/gpt-4o-mini-tts or hexgrad/kokoro-82m) and save the audio to `output`. This is a \
        synchronous, fast call (not a background task). This tool has NO defaults: model, input \
        (the text), and voice must all be specified, or the call fails naming what is \
        missing. `output` is optional - omit it for an auto-named file under \
        OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp). Returns the saved file path in JSON; for sandboxed clients it also returns a \
        native inline audio content block when the file is small enough. response_format defaults \
        to mp3 so the extension is deterministic.",
        annotations(
            title = "Generate Speech",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn generate_audio(
        &self,
        Parameters(args): Parameters<GenerateAudioArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let inline = client_wants_inline_previews(&context);
        self.run_generate_audio(args, inline).await
    }

    /// Core of `generate_audio` (synchronous, mirrors `describe_image`),
    /// parameterized on inline media so tests can drive it directly.
    pub(crate) async fn run_generate_audio(
        &self,
        args: GenerateAudioArgs,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        // No defaults: input and voice are the things agents forget.
        let mut missing: Vec<&str> = Vec::new();
        if args
            .input
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            missing.push("input (the text to synthesize)");
        }
        if args
            .voice
            .as_deref()
            .map(str::trim)
            .unwrap_or("")
            .is_empty()
        {
            missing.push("voice (voice id, varies by model e.g. \"alloy\")");
        }
        require_all("generate_audio", "speech", &missing)?;

        let model = args.model.clone();
        let req = SpeechGenRequest {
            model: args.model,
            input: args.input.unwrap_or_default(),
            voice: args.voice.unwrap_or_default(),
            response_format: args.response_format,
            speed: args.speed,
        };
        let fmt = req.response_format.as_deref().unwrap_or("mp3");
        let output = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Audio,
            &model,
            &[req.voice.as_str(), fmt],
            None,
        );

        match audio_gen::run_job(&self.client, &req, &output, "inline").await {
            Ok(result) => {
                self.stats.record_audio(&model, true, None).await;
                let mut env = json!({
                    "ok": true,
                    "kind": "audio",
                    "model": result.model,
                    "audio": {
                        "path": result.audio.path.to_string_lossy(),
                        "mime": result.audio.mime,
                        "voice": result.audio.voice,
                        "response_format": result.audio.response_format,
                    },
                    "manifest": result.manifest_path.to_string_lossy(),
                });
                if !result.warnings.is_empty() {
                    env["warnings"] = json!(result.warnings);
                }
                let body = serde_json::to_string_pretty(&env)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                let mut blocks = vec![Content::text(body)];

                // Inline native AudioContent for sandboxed clients, under the cap.
                if inline_previews {
                    let path = result.audio.path.clone();
                    let mime = result.audio.mime.clone();
                    let small = std::fs::metadata(&path)
                        .map(|m| m.len() <= MAX_INLINE_AUDIO_BYTES)
                        .unwrap_or(false);
                    if small {
                        let read = tokio::task::spawn_blocking(move || std::fs::read(&path))
                            .await
                            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                        if let Ok(bytes) = read {
                            let data = base64::engine::general_purpose::STANDARD.encode(bytes);
                            blocks.push(
                                RawContent::Audio(RawAudioContent {
                                    data,
                                    mime_type: mime,
                                })
                                .no_annotation(),
                            );
                        }
                    }
                }
                Ok(CallToolResult::success(blocks))
            }
            Err(e) => {
                self.stats.record_audio(&model, false, None).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn generate_audio_synthesizes_and_returns_path_json() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-tool/voice.mp3");
        let args = GenerateAudioArgs {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: Some("hello".to_string()),
            voice: Some("alloy".to_string()),
            response_format: None,
            speed: None,
            output: Some(out.to_string_lossy().into_owned()),
        };
        // inline_previews=false -> JSON only, no embedded audio block.
        let res = server.run_generate_audio(args, false).await.unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["ok"], true);
        assert_eq!(v["kind"], "audio");
        assert_eq!(v["audio"]["voice"], "alloy");
        assert_eq!(v["audio"]["mime"], "audio/mpeg");
        assert!(v["audio"]["path"].as_str().unwrap().ends_with(".mp3"));

        // The stats counter recorded the audio generation.
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["audio_files"], 1);
    }

    #[tokio::test]
    async fn generate_audio_embeds_inline_audio_block_for_sandboxed_clients() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3-FAKE".to_vec()),
            )
            .mount(&mock)
            .await;

        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-audio-inline/voice.mp3");
        let args = GenerateAudioArgs {
            model: "openai/gpt-4o-mini-tts".to_string(),
            input: Some("hello".to_string()),
            voice: Some("alloy".to_string()),
            response_format: None,
            speed: None,
            output: Some(out.to_string_lossy().into_owned()),
        };
        // inline_previews=true (a sandboxed client like Claude Desktop): the
        // small file is embedded as a native audio content block alongside JSON.
        let res = server.run_generate_audio(args, true).await.unwrap();
        let full = serde_json::to_value(&res).unwrap();
        let audio_block = full["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["type"] == "audio")
            .expect("an audio content block is present");
        assert_eq!(audio_block["mimeType"], "audio/mpeg");
        assert!(!audio_block["data"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn generate_audio_requires_input_and_voice() {
        // Validation runs before any HTTP call.
        let server = server_for("http://127.0.0.1:9".to_string());
        let args = GenerateAudioArgs {
            model: "m".to_string(),
            input: None,
            voice: Some("  ".to_string()), // blank-after-trim counts as missing
            response_format: None,
            speed: None,
            output: Some("out.mp3".to_string()),
        };
        let err = server.run_generate_audio(args, false).await.unwrap_err();
        assert!(err.message.contains("input"));
        assert!(err.message.contains("voice"));
        assert!(err.message.contains("no defaults"));
    }
}
