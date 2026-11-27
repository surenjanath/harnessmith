//! The `codex` provider: a chat-with-tools call served by the Codex CLI (`codex exec`).
//!
//! It lets every role run on the models of a ChatGPT / Codex sign-in (`codex login`), with no
//! API key in harnessmith. Each call is one stateless, ephemeral `codex exec` in an empty,
//! read-only scratch directory: the harness's system prompt, its tools and the whole
//! conversation so far are written into the prompt, and Codex answers with a JSON object
//! (enforced by `--output-schema`) holding the reply text and any tool calls. The harness runs
//! those tools itself, exactly as it does for an API model; Codex's own shell is not used.
//!
//! Model ids: `codex/<model>` with an optional reasoning effort, `codex/gpt-6.1-terra@high`.
//! A connection's base URL, when set, is the path to the `codex` binary (default: `codex` on
//! PATH, or `HARNESS2_CODEX_BIN`).

use super::client::{ChatReq, ChatResp, Msg, ToolCall, CANCELLED};
use super::connections::Connection;
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Reasoning effort used when the model id names none.
pub const DEFAULT_EFFORT: &str = "medium";
const EFFORTS: &[&str] = &["minimal", "low", "medium", "high", "xhigh"];

/// The `codex` binary a connection runs.
pub fn binary(conn: &Connection) -> String {
    let b = conn.base_url.trim();
    if !b.is_empty() && !b.contains("://") {
        return crate::config::expand_home(b).to_string_lossy().into_owned();
    }
    std::env::var("HARNESS2_CODEX_BIN").ok().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "codex".into())
}

/// `gpt-6.1-terra@high` -> ("gpt-6.1-terra", "high"). An unknown suffix is left in the name.
pub fn split_effort(model: &str) -> (String, String) {
    if let Some((m, e)) = model.rsplit_once('@') {
        if EFFORTS.contains(&e) {
            return (m.to_string(), e.to_string());
        }
    }
    let e = std::env::var("HARNESS2_CODEX_EFFORT").ok().filter(|e| EFFORTS.contains(&e.as_str())).unwrap_or_else(|| DEFAULT_EFFORT.into());
    (model.to_string(), e)
}

/// The models this Codex sign-in offers, read from Codex's own cache (`~/.codex/models_cache.json`).
pub fn list_models() -> Result<Vec<String>> {
    let home = std::env::var("CODEX_HOME").map(std::path::PathBuf::from).unwrap_or_else(|_| crate::config::expand_home("~/.codex"));
    let p = home.join("models_cache.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| anyhow!("cannot read {}: {e} (run `codex` once, signed in, so it caches its model list)", p.display()))?)?;
    let arr = v.get("models").and_then(|m| m.as_array()).cloned().or_else(|| v.as_array().cloned()).unwrap_or_default();
    let out: Vec<String> = arr
        .iter()
        .filter(|m| m.get("visibility").and_then(|x| x.as_str()).map(|x| x != "hide").unwrap_or(true))
        .filter_map(|m| m.get("slug").or_else(|| m.get("id")).and_then(|s| s.as_str()).map(|s| s.to_string()))
        .filter(|s| !s.contains("auto-review"))
        .collect();
    if out.is_empty() {
        bail!("no models listed in {}", p.display());
    }
    Ok(out)
}

/// Whether the Codex CLI is installed and signed in: Ok(one-line status) or the reason not.
pub fn status(conn: &Connection) -> Result<String> {
    let bin = binary(conn);
    let out = Command::new(&bin).args(["login", "status"]).stdin(Stdio::null()).output().map_err(|e| anyhow!("cannot run {bin:?}: {e}. Install the Codex CLI (npm i -g @openai/codex) or set the connection's base URL to its path"))?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)).trim().to_string();
    if !out.status.success() || text.to_lowercase().contains("not logged in") {
        bail!("Codex is not signed in ({}). Run `codex login` in a terminal", crate::util::take_chars(&text, 200));
    }
    Ok(text)
}

pub fn schema(with_tools: bool) -> Value {
    let mut props = json!({"text": {"type": "string", "description": "The assistant's reply text (may be empty when calling tools)."}});
    let mut req = vec!["text"];
    if with_tools {
        props["tool_calls"] = json!({
            "type": "array",
            "description": "Tools to call now, in order. Empty when this is the final reply.",
            "items": {
                "type": "object",
                "additionalProperties": false,
                "required": ["name", "arguments_json"],
                "properties": {
                    "name": {"type": "string"},
                    "arguments_json": {"type": "string", "description": "The arguments as one JSON object, encoded as a string."}
                }
            }
        });
        req.push("tool_calls");
    }
    json!({"type": "object", "additionalProperties": false, "required": req, "properties": props})
}

/// Replaces Codex's own ~10k-token coding-agent instructions (`model_instructions_file`).
pub const INSTRUCTIONS: &str = "You are the language model behind a chat API for another program, an agent harness. \
You have no workspace and no tools of your own: never run commands, read or write files, or browse. \
The only tools that exist are the ones listed under TOOLS in the user message; to use one, put it in `tool_calls` of your JSON reply, \
and the harness runs it and shows you the result in the next turn. \
Write only the next assistant message of the conversation, following its SYSTEM instructions, as JSON matching the output schema.\n";

/// Codex features that add tool schemas or context this use never needs. Set through `-c`, so a
/// name a Codex version does not know is ignored instead of failing the call.
pub const OFF_FEATURES: &[&str] = &["apps", "browser_use", "computer_use", "goals", "shell_tool", "unified_exec", "image_generation", "multi_agent", "plugins", "remote_plugin", "skill_search", "sleep_tool", "tool_suggest", "view_image", "shell_snapshot", "memories", "code_mode", "code_mode_host"];
/// Context blocks Codex adds to every prompt (the skills list alone is ~2.8k tokens). Measured
/// with Codex 0.165: these and OFF_FEATURES take a call's fixed cost from ~17.5k to ~4.3k tokens.
pub const OFF_SETTINGS: &[&str] = &["include_skills_usage_instructions", "skills.include_instructions", "skills.bundled.enabled", "include_collaboration_mode_instructions", "include_permissions_instructions", "include_environment_context", "include_apps_instructions"];

/// A fixed, empty working folder holding the instructions file. It must not change between
/// calls: Codex puts the cwd near the top of its prompt, and a new folder per call would defeat
/// the provider's prefix cache on every turn.
pub fn home() -> Result<std::path::PathBuf> {
    let d = std::env::temp_dir().join("harnessmith-codex");
    let ws = d.join("ws");
    std::fs::create_dir_all(&ws)?;
    let ins = d.join("instructions.md");
    if std::fs::read_to_string(&ins).ok().as_deref() != Some(INSTRUCTIONS) {
        let tmp = d.join(format!("instructions.{}.tmp", std::process::id()));
        std::fs::write(&tmp, INSTRUCTIONS)?;
        std::fs::rename(&tmp, &ins)?;
    }
    Ok(d)
}

/// The whole request as one prompt for a stateless Codex turn. Earlier turns render to the same
/// text every time, so each prompt extends the previous one and the cached prefix is reused.
pub fn render(req: &ChatReq) -> String {
    let mut p = String::new();
    if !req.system.trim().is_empty() {
        p.push_str("=== SYSTEM ===\n");
        p.push_str(req.system.trim());
        p.push_str("\n\n");
    }
    if !req.tools.is_empty() {
        p.push_str("=== TOOLS (call them via tool_calls; arguments_json is a JSON object string) ===\n");
        for t in req.tools {
            p.push_str(&format!("- {}: {}\n  parameters: {}\n", t.name, t.description.trim(), t.parameters));
        }
        p.push('\n');
    } else {
        p.push_str("(No tools are available for this turn: reply with text only.)\n\n");
    }
    p.push_str("=== CONVERSATION ===\n");
    for m in req.messages {
        match m {
            Msg::User(t) => p.push_str(&format!("[user]\n{}\n\n", t.trim_end())),
            Msg::Assistant { text, calls } => {
                if !text.trim().is_empty() {
                    p.push_str(&format!("[assistant]\n{}\n\n", text.trim_end()));
                }
                for c in calls {
                    p.push_str(&format!("[assistant called tool {} (id {})]\n{}\n\n", c.name, c.id, c.args));
                }
            }
            Msg::Tools(rs) => {
                for r in rs {
                    let tag = if r.is_error { " — ERROR" } else { "" };
                    p.push_str(&format!("[tool result for {} (id {}){tag}]\n{}\n\n", r.name, r.id, r.content.trim_end()));
                }
            }
        }
    }
    p.push_str(&format!("=== YOUR TURN ===\nReply as the assistant, with JSON matching the schema{}.\n", if req.tools.is_empty() { " (text only)" } else { ": the reply text, and the tool calls to make now (an empty list when you are done)" }));
    p
}

/// Parse `codex exec --json` events: (last agent message, usage, error).
pub fn parse_events(stdout: &str) -> (Option<String>, (u64, u64, u64), Option<String>) {
    let (mut msg, mut usage, mut err) = (None, (0, 0, 0), None);
    for line in stdout.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else { continue };
        match v["type"].as_str().unwrap_or("") {
            "item.completed" if v["item"]["type"] == "agent_message" => msg = v["item"]["text"].as_str().map(|s| s.to_string()),
            "turn.completed" => {
                let u = &v["usage"];
                let cached = u["cached_input_tokens"].as_u64().unwrap_or(0);
                usage = (u["input_tokens"].as_u64().unwrap_or(0).saturating_sub(cached), u["output_tokens"].as_u64().unwrap_or(0) + u["reasoning_output_tokens"].as_u64().unwrap_or(0), cached);
            }
            "turn.failed" => err = v["error"]["message"].as_str().map(|s| s.to_string()).or(Some(v.to_string())),
            "error" => err = v["message"].as_str().map(|s| s.to_string()).or(Some(v.to_string())),
            _ => {}
        }
    }
    (msg, usage, err)
}

/// Tool arguments from Codex's `arguments_json`: strict, then the first complete object with any
/// trailing junk ignored (seen live: a valid object followed by `"}]}=} CX1;…`), then lenient.
pub fn parse_args(s: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return Some(v);
    }
    if let Some(Ok(v)) = serde_json::Deserializer::from_str(s.trim_start()).into_iter::<Value>().next() {
        if v.is_object() {
            return Some(v);
        }
    }
    super::client::lenient_json(s)
}

/// Turn Codex's final message into a reply (tool calls parsed leniently).
pub fn to_resp(final_msg: &str, with_tools: bool) -> ChatResp {
    let mut out = ChatResp { stop: "stop".into(), ..Default::default() };
    let v = serde_json::from_str::<Value>(final_msg.trim()).ok().or_else(|| super::client::lenient_json(final_msg));
    let Some(v) = v.filter(|v| v.is_object()) else {
        out.text = final_msg.to_string();
        return out;
    };
    out.text = v["text"].as_str().unwrap_or("").to_string();
    if with_tools {
        for (i, c) in v["tool_calls"].as_array().cloned().unwrap_or_default().iter().enumerate() {
            let name = c["name"].as_str().unwrap_or("").to_string();
            if name.is_empty() {
                continue;
            }
            let args = match &c["arguments_json"] {
                Value::String(s) => parse_args(s).unwrap_or_else(|| json!({"_raw": s})),
                Value::Null => c.get("arguments").cloned().unwrap_or(json!({})),
                o => o.clone(),
            };
            out.calls.push(ToolCall { id: format!("cx_{}_{i}", crate::util::sha256_hex(format!("{name}{args}{}", out.text.len()).as_bytes()).get(..8).unwrap_or("0")), name, args, extra: None });
        }
        if !out.calls.is_empty() {
            out.stop = "tool_calls".into();
        }
    }
    out
}

fn chrono_stamp() -> String {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    format!("{}{:03}", d.as_secs(), d.subsec_millis())
}

/// A TOML basic string.
fn toml_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The per-call cap (HARNESS2_CODEX_CALL_TIMEOUT seconds, default 600), at least two minutes.
fn call_limit(timeout: Duration) -> Duration {
    let cap = Duration::from_secs(crate::config::env_int("HARNESS2_CODEX_CALL_TIMEOUT", 600).ok().filter(|s| *s >= 30).unwrap_or(600) as u64);
    timeout.max(Duration::from_secs(120)).min(cap)
}

/// One model call: through the shared `codex app-server` (no process start per call), or with
/// `codex exec` when the server is off or cannot serve.
pub fn chat(conn: &Connection, model: &str, req: &ChatReq, timeout: Duration) -> Result<ChatResp> {
    if super::codex_server::enabled() {
        let (m, effort) = split_effort(model);
        match super::codex_server::chat(&binary(conn), &m, &effort, req, call_limit(timeout)) {
            Err(e) if format!("{e}").starts_with(super::codex_server::SERVER_ERR) => {
                crate::say!("    ! {e}; using codex exec for this call");
            }
            r => return r,
        }
    }
    chat_exec(conn, model, req, timeout)
}

fn chat_exec(conn: &Connection, model: &str, req: &ChatReq, timeout: Duration) -> Result<ChatResp> {
    let (model, effort) = split_effort(model);
    let dir = tempfile::Builder::new().prefix("harnessmith-codex-").tempdir()?;
    let with_tools = !req.tools.is_empty();
    let schema_path = dir.path().join("schema.json");
    std::fs::write(&schema_path, schema(with_tools).to_string())?;
    let home = home()?;
    let ws = home.join("ws");
    let last = dir.path().join("last.txt");
    let bin = binary(conn);
    let mut cmd = Command::new(&bin);
    cmd.args(["exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-rules", "--ignore-user-config", "-s", "read-only", "--color", "never"])
        .arg("-C")
        .arg(&ws)
        .arg("-c")
        .arg(format!("model_reasoning_effort=\"{effort}\""))
        .arg("-c")
        .arg(format!("model_instructions_file={}", toml_str(&home.join("instructions.md").to_string_lossy())))
        .arg("-c")
        .arg("web_search=\"disabled\"");
    for f in OFF_FEATURES {
        cmd.arg("-c").arg(format!("features.{f}=false"));
    }
    for k in OFF_SETTINGS {
        cmd.arg("-c").arg(format!("{k}=false"));
    }
    cmd
        .arg("--output-schema")
        .arg(&schema_path)
        .arg("-o")
        .arg(&last);
    if !model.is_empty() && model != "default" {
        cmd.arg("-m").arg(&model);
    }
    cmd.arg("-").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| anyhow!("cannot run {bin:?}: {e}. Install the Codex CLI and run `codex login`"))?;
    let prompt = render(req);
    // HARNESS2_CODEX_DUMP=<dir>: keep every prompt and event stream, to see what a call cost.
    let dump = std::env::var("HARNESS2_CODEX_DUMP").ok().filter(|d| !d.trim().is_empty()).map(std::path::PathBuf::from);
    let stamp = format!("{}-{}", chrono_stamp(), std::process::id());
    if let Some(d) = &dump {
        let _ = std::fs::create_dir_all(d);
        let _ = std::fs::write(d.join(format!("{stamp}.prompt.txt")), &prompt);
    }
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(prompt.as_bytes());
    });
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let ro = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = so.read_to_string(&mut s);
        s
    });
    let re = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = se.read_to_string(&mut s);
        s
    });
    // Reasoning models think for a while, so a turn gets at least two minutes; but the caller's
    // timeout is often the whole run's remaining time, and a `codex exec` that stalls (seen
    // live: one call hung 13 minutes on an open connection) must not eat it. Cap each call
    // (HARNESS2_CODEX_CALL_TIMEOUT seconds, default 600); a killed call is retried or fails over.
    let limit = call_limit(timeout);
    let t0 = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if crate::log::cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            bail!(CANCELLED);
        }
        if t0.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            bail!("HTTP 504 from codex: no reply within {}s (call killed; HARNESS2_CODEX_CALL_TIMEOUT raises the cap)", limit.as_secs());
        }
        std::thread::sleep(Duration::from_millis(150));
    };
    let _ = writer.join();
    let stdout = ro.join().unwrap_or_default();
    let stderr = re.join().unwrap_or_default();
    if let Some(d) = &dump {
        let _ = std::fs::write(d.join(format!("{stamp}.events.jsonl")), &stdout);
    }
    let (msg, (inp, outp, cached), err) = parse_events(&stdout);
    let final_msg = std::fs::read_to_string(&last).ok().filter(|s| !s.trim().is_empty()).or(msg);
    let Some(final_msg) = final_msg else {
        let why = err.unwrap_or_else(|| crate::util::take_chars(stderr.trim(), 600));
        // Usage limits and sign-in problems read like HTTP errors so the chain fails over.
        let low = why.to_lowercase();
        let code = if low.contains("usage limit") || low.contains("rate limit") || low.contains("429") { "HTTP 429" } else if low.contains("log in") || low.contains("login") || low.contains("401") { "HTTP 401" } else { "codex exec failed" };
        bail!("{code} from codex ({}): {}", status, crate::util::take_chars(&why, 600));
    };
    let mut r = to_resp(&final_msg, with_tools);
    r.input_tokens = inp;
    r.output_tokens = outp;
    r.cached_tokens = cached;
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::client::{ToolDef, ToolResult};

    #[test]
    fn effort_suffix() {
        assert_eq!(split_effort("gpt-6.1-terra@high"), ("gpt-6.1-terra".into(), "high".into()));
        assert_eq!(split_effort("gpt-6.1-terra").0, "gpt-6.1-terra");
        assert_eq!(split_effort("a@weird").0, "a@weird");
    }

    #[test]
    fn events_and_reply_parse() {
        let ev = r#"{"type":"thread.started","thread_id":"x"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"{\"text\":\"\",\"tool_calls\":[{\"name\":\"ping\",\"arguments_json\":\"{\\\"word\\\":\\\"ok\\\"}\"}]}"}}
{"type":"turn.completed","usage":{"input_tokens":17443,"cached_input_tokens":443,"output_tokens":32,"reasoning_output_tokens":8}}"#;
        let (m, u, e) = parse_events(ev);
        assert!(e.is_none());
        assert_eq!(u, (17000, 40, 443));
        let r = to_resp(&m.unwrap(), true);
        assert_eq!(r.calls.len(), 1);
        assert_eq!(r.calls[0].name, "ping");
        assert_eq!(r.calls[0].args["word"], "ok");
        let junk = parse_args(r#"{"path":"a.md","content":"x\n"}]}=} CX1;`]](A=[{)"#).unwrap();
        assert_eq!(junk["path"], "a.md");
        let plain = to_resp("not json at all", true);
        assert_eq!(plain.text, "not json at all");
        assert!(plain.calls.is_empty());
    }

    #[test]
    fn prompt_carries_tools_and_transcript() {
        let tools = [ToolDef { name: "write_file".into(), description: "Write a file.".into(), parameters: json!({"type":"object"}) }];
        let msgs = [
            Msg::User("Make answer.md".into()),
            Msg::Assistant { text: String::new(), calls: vec![ToolCall { id: "c1".into(), name: "write_file".into(), args: json!({"path":"answer.md"}), extra: None }] },
            Msg::Tools(vec![ToolResult { id: "c1".into(), name: "write_file".into(), content: "wrote 3 bytes".into(), is_error: false }]),
        ];
        let p = render(&ChatReq { system: "Be brief.", messages: &msgs, tools: &tools, max_tokens: 100, temperature: None });
        for needle in ["Be brief.", "- write_file: Write a file.", "[user]\nMake answer.md", "[assistant called tool write_file (id c1)]", "[tool result for write_file (id c1)]\nwrote 3 bytes"] {
            assert!(p.contains(needle), "missing {needle:?}");
        }
    }
}
