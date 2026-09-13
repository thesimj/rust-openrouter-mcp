//! The `generate_music` tool and its argument struct.

use rmcp::{
    ErrorData, RoleServer,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, ContentBlock},
    service::RequestContext,
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::music_gen::{self, MusicGenRequest};
use crate::server::naming;
use crate::server::result::{
    attach_warnings_errors, client_wants_inline_previews, inline_audio_block,
};
use crate::server::schema::{RequireFields, de_opt_uint, require_all, scalarize_nullable};

use super::OpenRouterServer;

/// Arguments for the `generate_music` tool.
#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(transform = scalarize_nullable)]
#[schemars(transform = RequireFields(&["prompt"]))]
pub(crate) struct GenerateMusicArgs {
    /// Music model id: "google/lyria-3-clip-preview" (30-second clip) or
    /// "google/lyria-3-pro-preview" (full song). Music models are chat models
    /// whose output_modalities include "audio" - discover them with list_models
    /// using output_modalities="audio".
    pub model: String,
    /// REQUIRED (no default): the musical description - genre, mood, tempo,
    /// instruments, structure, and lyrics if the track should have any.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Requested container, passed through as `audio.format` (wav, mp3, flac,
    /// opus, pcm16). Model-specific: Lyria ignores it and returns MP3 (verified
    /// live). The saved file's extension always follows the bytes returned.
    #[serde(default)]
    pub format: Option<String>,
    /// Seed for reproducible-ish generation (model support varies; Lyria lists
    /// `seed` in its supported_parameters).
    #[serde(default, deserialize_with = "de_opt_uint")]
    pub seed: Option<u64>,
    /// Output file path (extension corrected to the returned container, e.g.
    /// .mp3). Optional: when omitted, an auto-named file is written under
    /// OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp).
    #[serde(default)]
    pub output: Option<String>,
}

#[tool_router(router = music_router, vis = "pub(crate)")]
impl OpenRouterServer {
    #[tool(
        description = "Generate music from a text prompt with an OpenRouter music model (e.g. \
        google/lyria-3-clip-preview for a 30-second clip, google/lyria-3-pro-preview for a \
        full song) and save the track to `output`. OpenRouter has no dedicated music \
        endpoint: music models are chat models whose output_modalities include \"audio\", so \
        this streams /chat/completions with modalities [\"text\",\"audio\"] and saves the \
        streamed audio. Synchronous: it waits for the whole track (a Lyria clip takes about \
        10-25 s). `prompt` has no default and must be given. `format` is optional and passed \
        through as audio.format, and model-specific - Lyria ignores it and returns MP3 (verified \
        live); the saved extension follows the bytes returned. Pricing: list_models shows 0 token pricing for these models because they bill a \
        flat fee per track, stated only in the model description ($0.04 per Lyria clip, \
        $0.08 per song); the actual charge comes back as cost_usd. Returns JSON with the saved \
        path, mime, the model's text (lyrics, or \"<instrumental>\"), cost_usd, and the \
        manifest path; for sandboxed clients also an inline audio block when the file is \
        small enough. If the stream ends before its [DONE] sentinel the track is still saved but \
        the result carries a `warnings` entry saying it may be cut short. `output` is optional - \
        omit it for an auto-named file under \
        OPENROUTER_MCP_OUTPUT_DIR (default $HOME/Downloads/openrouter-mcp).",
        annotations(
            title = "Generate Music",
            read_only_hint = false,
            destructive_hint = false,
            open_world_hint = true
        )
    )]
    async fn generate_music(
        &self,
        Parameters(args): Parameters<GenerateMusicArgs>,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let inline = client_wants_inline_previews(&context);
        self.run_generate_music(args, inline).await
    }

    /// Core of `generate_music` (synchronous, mirrors `generate_audio`),
    /// parameterized on inline media so tests can drive it directly.
    pub(crate) async fn run_generate_music(
        &self,
        args: GenerateMusicArgs,
        inline_previews: bool,
    ) -> Result<CallToolResult, ErrorData> {
        let _work = self.admit_work()?;
        let prompt = args
            .prompt
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty());
        let mut missing: Vec<&str> = Vec::new();
        if prompt.is_none() {
            missing
                .push("prompt (the musical description: genre, mood, tempo, instruments, lyrics)");
        }
        require_all("generate_music", "audio", &missing)?;

        let model = args.model.clone();
        let req = MusicGenRequest {
            model: args.model,
            prompt: prompt.unwrap_or_default().to_string(),
            format: args.format,
            seed: args.seed,
        };
        // The filename token is the requested format when there is one, run
        // through the same normalization run_job applies to the wire value so
        // the two never diverge; the extension itself is decided later by the
        // returned bytes.
        let fmt = music_gen::normalize_format(req.format.as_deref()).unwrap_or_default();
        let output = naming::resolve_output_base(
            args.output,
            naming::MediaKind::Music,
            &model,
            &[fmt.as_str()],
            req.seed,
        );

        match music_gen::run_job(&self.client, &req, &output, "inline").await {
            Ok(result) => {
                self.stats.record_audio(&model, true, result.cost).await;
                let mut env = json!({
                    "ok": true,
                    "kind": "music",
                    "model": result.model,
                    "music": {
                        "path": result.music.path.to_string_lossy(),
                        "mime": result.music.mime,
                    },
                    "manifest": result.manifest_path.to_string_lossy(),
                });
                if let Some(text) = &result.music.text {
                    env["music"]["text"] = json!(text);
                }
                if let Some(transcript) = &result.music.transcript {
                    env["music"]["transcript"] = json!(transcript);
                }
                if let Some(cost) = result.cost {
                    env["cost_usd"] = json!(cost);
                }
                if let Some(id) = &result.generation_id {
                    env["generation_id"] = json!(id);
                }
                attach_warnings_errors(&mut env, &result.warnings, &[]);
                let body = serde_json::to_string_pretty(&env)
                    .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                let mut blocks = vec![ContentBlock::text(body)];
                if inline_previews {
                    blocks.extend(
                        inline_audio_block(result.music.path.clone(), result.music.mime.clone())
                            .await?,
                    );
                }
                Ok(CallToolResult::success(blocks))
            }
            Err(e) => {
                self.stats.record_audio(&model, false, None).await;
                self.stats.record_failed_receipt(&model, &e).await;
                Err(ErrorData::internal_error(format!("{e:#}"), None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::test_support::{server_for, tool_result_json};
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// The live Lyria stream shape, with a real ID3 header as the audio.
    const LYRIA_STREAM: &str = concat!(
        ": OPENROUTER PROCESSING\n\n",
        "data: {\"id\":\"gen-m\",\"choices\":[{\"delta\":{\"content\":\"<instrumental>\"}}]}\n\n",
        "data: {\"id\":\"gen-m\",\"choices\":[{\"delta\":{\"content\":\"\",\"audio\":{\"data\":\"SUQzAwAAAAAvMg==\"}}}]}\n\n",
        "data: {\"id\":\"gen-m\",\"choices\":[{\"delta\":{\"content\":\"\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"gen-m\",\"choices\":[],\"usage\":{\"cost\":0.04}}\n\n",
        "data: [DONE]\n\n",
    );

    fn args(prompt: Option<&str>, output: &std::path::Path) -> GenerateMusicArgs {
        GenerateMusicArgs {
            model: "google/lyria-3-clip-preview".to_string(),
            prompt: prompt.map(str::to_string),
            format: None,
            seed: Some(3),
            output: Some(output.to_string_lossy().into_owned()),
        }
    }

    async fn lyria_mock() -> MockServer {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "model": "google/lyria-3-clip-preview",
                "messages": [{"role": "user", "content": "warm lo-fi loop"}],
                "modalities": ["text", "audio"],
                "seed": 3,
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("x-generation-id", "gen-m")
                    .set_body_string(LYRIA_STREAM),
            )
            .mount(&mock)
            .await;
        mock
    }

    #[tokio::test]
    async fn generate_music_streams_saves_and_returns_path_json_with_cost() {
        let mock = lyria_mock().await;
        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-music-tool/track.wav");
        let res = server
            .run_generate_music(args(Some("  warm lo-fi loop "), &out), false)
            .await
            .unwrap();
        let v = tool_result_json(&res);
        assert_eq!(v["ok"], true);
        assert_eq!(v["kind"], "music");
        assert_eq!(v["model"], "google/lyria-3-clip-preview");
        assert_eq!(v["music"]["mime"], "audio/mpeg");
        // The bytes are MP3, so the .wav the caller guessed is corrected.
        assert!(v["music"]["path"].as_str().unwrap().ends_with(".mp3"));
        assert_eq!(v["music"]["text"], "<instrumental>");
        assert!(v["music"].get("transcript").is_none());
        assert_eq!(v["cost_usd"], 0.04);
        assert_eq!(v["generation_id"], "gen-m");
        assert!(v["manifest"].as_str().unwrap().ends_with(".json"));
        // JSON only: no inline audio block for a filesystem-sharing client.
        assert_eq!(
            serde_json::to_value(&res).unwrap()["content"]
                .as_array()
                .unwrap()
                .len(),
            1
        );

        // Counted as an audio generation, with the reported cost.
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["audio_generations"], 1);
        assert_eq!(stats["audio_files"], 1);
        assert_eq!(stats["actual_cost_usd"], 0.04);
    }

    #[tokio::test]
    async fn generate_music_embeds_inline_audio_block_for_sandboxed_clients() {
        let mock = lyria_mock().await;
        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-music-inline/track.mp3");
        let res = server
            .run_generate_music(args(Some("warm lo-fi loop"), &out), true)
            .await
            .unwrap();
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
    async fn generate_music_requires_a_prompt_before_any_call() {
        let server = server_for("http://127.0.0.1:9".to_string());
        for prompt in [None, Some("   ")] {
            let err = server
                .run_generate_music(args(prompt, std::path::Path::new("track.mp3")), false)
                .await
                .unwrap_err();
            assert!(err.message.contains("prompt"), "{}", err.message);
            assert!(err.message.contains("no defaults"), "{}", err.message);
            assert!(
                err.message.contains("output_modalities=\"audio\""),
                "{}",
                err.message
            );
        }
    }

    #[tokio::test]
    async fn generate_music_surfaces_provider_errors_and_counts_the_failure() {
        let mock = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(400)
                    .set_body_string("{\"error\":{\"message\":\"audio output not supported\"}}"),
            )
            .mount(&mock)
            .await;
        let server = server_for(mock.uri());
        let out = std::env::temp_dir().join("openrouter-mcp-music-fail/track.mp3");
        let err = server
            .run_generate_music(args(Some("anything"), &out), false)
            .await
            .unwrap_err();
        assert!(
            err.message.contains("audio output not supported"),
            "{}",
            err.message
        );
        let stats = tool_result_json(&server.get_usage_stats().await.unwrap());
        assert_eq!(stats["requests_failed"], 1);
        assert_eq!(stats["audio_files"], 0);
    }
}
