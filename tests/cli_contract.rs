use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tempfile::TempDir;

fn isolated_command() -> (TempDir, Command) {
    let dir = tempfile::tempdir().unwrap();
    // Stop dotenv from searching parent directories for a developer's key.
    std::fs::write(dir.path().join(".env"), "").unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_openrouter-mcp"));
    command
        .env_clear()
        .current_dir(dir.path())
        .stdin(Stdio::null());
    // Windows may need SystemRoot to locate system components.
    if let Some(root) = std::env::var_os("SystemRoot") {
        command.env("SystemRoot", root);
    }
    (dir, command)
}

#[test]
fn version_flags_work_without_credentials_or_stdin() {
    for flag in ["--version", "-V"] {
        let (_dir, mut command) = isolated_command();
        let output = command.args([flag]).output().unwrap();
        assert!(output.status.success(), "{flag}: {output:?}");
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("openrouter-mcp {}\n", env!("CARGO_PKG_VERSION"))
        );
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn unsupported_arguments_reject_before_loading_credentials() {
    for args in [
        vec!["mcp", "extra"],
        vec!["mcp", "--version"],
        vec!["models"],
        vec!["image"],
        vec!["video"],
        vec!["audio"],
        vec!["music"],
        vec!["transcribe"],
        vec!["describe"],
        vec!["chat"],
        vec!["embed"],
        vec!["rerank"],
        vec!["generation"],
        vec!["key"],
        vec!["--help"],
        vec!["-h"],
        vec!["help"],
        vec!["--unknown"],
        vec!["--version", "extra"],
        vec!["-V", "extra"],
        vec!["--version", "-V"],
        vec!["--", "--version"],
    ] {
        let (_dir, mut command) = isolated_command();
        let output = command.args(&args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{args:?}: {output:?}");
        assert!(!output.stderr.is_empty(), "{args:?}");
    }
}

#[test]
fn mcp_requires_credentials_and_keeps_errors_off_stdout() {
    // Both launch forms start the server: bare, and the historical `mcp` subcommand.
    for args in [vec![], vec!["mcp"]] {
        let (_dir, mut command) = isolated_command();
        let output = command.args(&args).output().unwrap();
        assert!(!output.status.success(), "{args:?}: {output:?}");
        assert_ne!(output.status.code(), Some(2), "{args:?}: {output:?}");
        assert!(output.stdout.is_empty(), "{args:?}: {output:?}");
        assert!(
            String::from_utf8(output.stderr)
                .unwrap()
                .contains("OPENROUTER_API_KEY"),
            "{args:?}"
        );
    }
}

/// A `.env` that exists but cannot be parsed is named on stderr: otherwise
/// the only symptom is a misleading "OPENROUTER_API_KEY is not set".
#[test]
fn a_malformed_env_file_is_reported_on_stderr() {
    let (dir, mut command) = isolated_command();
    std::fs::write(
        dir.path().join(".env"),
        "BROKEN=\"unterminated\nOPENROUTER_API_KEY=local-test-key\n",
    )
    .unwrap();
    let output = command.output().unwrap();
    assert!(output.stdout.is_empty(), "{output:?}");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains(".env"), "{stderr}");
}

// Reap the process even when a protocol assertion fails or times out.
struct ServerProcess(Child);

impl Drop for ServerProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn exchange(input: &mut ChildStdin, output: &Receiver<String>, request: Value) -> Value {
    writeln!(input, "{request}").unwrap();
    input.flush().unwrap();
    let line = output
        .recv_timeout(Duration::from_secs(10))
        .expect("MCP reply timed out");
    let response: Value = serde_json::from_str(&line).expect("stdout must contain only MCP JSON");
    assert_eq!(response["jsonrpc"], "2.0");
    assert_eq!(response["id"], request["id"]);
    response
}

#[test]
fn bare_binary_serves_all_tools_and_local_calls_then_exits_on_eof() {
    let (dir, mut command) = isolated_command();
    // Exercise .env loading with a dummy key. These calls never contact OpenRouter.
    std::fs::write(
        dir.path().join(".env"),
        "OPENROUTER_API_KEY=local-test-key\n",
    )
    .unwrap();
    let mut process = ServerProcess(
        command
            .env("OPENROUTER_MCP_SHUTDOWN_TIMEOUT", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let mut input = process.0.stdin.take().unwrap();
    let stdout = process.0.stdout.take().unwrap();
    let (sender, output) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if sender.send(line.unwrap()).is_err() {
                break;
            }
        }
    });

    let initialized = exchange(
        &mut input,
        &output,
        json!({
            "jsonrpc":"2.0", "id":1, "method":"initialize", "params":{
                "protocolVersion":"2025-11-25", "capabilities":{},
                "clientInfo":{"name":"contract-test","version":"1"}
            }
        }),
    );
    assert_eq!(
        initialized["result"]["serverInfo"]["name"],
        "openrouter-mcp"
    );
    assert_eq!(
        initialized["result"]["serverInfo"]["version"],
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(initialized["result"]["protocolVersion"], "2025-11-25");
    writeln!(
        input,
        "{}",
        json!({"jsonrpc":"2.0","method":"notifications/initialized"})
    )
    .unwrap();

    let listed = exchange(
        &mut input,
        &output,
        json!({
            "jsonrpc":"2.0","id":2,"method":"tools/list","params":{}
        }),
    );
    let tools = listed["result"]["tools"].as_array().unwrap();
    let mut names: Vec<_> = tools
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "chat_completion",
            "describe_image",
            "describe_model",
            "embed_text",
            "generate_audio",
            "generate_image",
            "generate_music",
            "generate_video",
            "get_account",
            "get_generation",
            "get_result",
            "get_usage_stats",
            "list_models",
            "make_decisions",
            "rerank_documents",
            "reset_usage_stats",
            "transcribe_audio",
        ]
    );
    for tool in tools {
        assert_eq!(tool["inputSchema"]["type"], "object");
    }
    for (id, name, args) in [
        (3, "get_usage_stats", json!({})),
        (4, "reset_usage_stats", json!({"confirm":true})),
    ] {
        let result = exchange(
            &mut input,
            &output,
            json!({
                "jsonrpc":"2.0","id":id,"method":"tools/call",
                "params":{"name":name,"arguments":args}
            }),
        );
        assert!(result.get("error").is_none(), "{result}");
        assert_ne!(result["result"]["isError"], true, "{result}");
        assert!(!result["result"]["content"].as_array().unwrap().is_empty());
    }
    for (id, name, args) in [
        (5, "reset_usage_stats", json!({"confirm":false})),
        (6, "get_result", json!({"task_id":"unknown-task"})),
    ] {
        let result = exchange(
            &mut input,
            &output,
            json!({
                "jsonrpc":"2.0","id":id,"method":"tools/call",
                "params":{"name":name,"arguments":args}
            }),
        );
        assert!(
            result.get("error").is_some() || result["result"]["isError"] == true,
            "{result}"
        );
    }

    drop(input);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = process.0.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        assert!(
            Instant::now() < deadline,
            "server did not exit after stdin closed"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    reader.join().unwrap();
    assert!(
        output.try_iter().next().is_none(),
        "unexpected stdout after the last reply"
    );
}
