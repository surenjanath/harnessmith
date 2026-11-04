//! codex-cli as a solver substrate.
//!
//! Carries the c1..c8 surface (CODEX_SURFACE), the one install mode (codex_root: AGENTS.md,
//! .agents, scripts and .codex copied to the workspace root, where `codex exec -C <ws>` finds
//! them), the `codex exec` flag contract, and the codex thread-item trajectory renderer.

use super::{cap_rendered, clip, pystr, Install};
use crate::config;
use crate::core::runmark;
use crate::core::tree::{self, SurfaceSpec};
use crate::util;
use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub fn codex_surface() -> SurfaceSpec {
    SurfaceSpec {
        name: "codex",
        always_on: vec!["AGENTS.md"],
        always_on_max: 8_000,
        always_on_cap_reason: "AGENTS.md is injected whole into EVERY turn, and codex additionally truncates it silently at 32,768 bytes",
        skill_dir: ".agents/skills",
        skill_name_rule: "overrides_dir",
        trigger_hint: "Fix it by adding a trigger line naming `{name}` to AGENTS.md, e.g. \"When <the situation this skill handles>, read .agents/skills/{name}/SKILL.md and follow it.\"",
        autotrigger_file: "AGENTS.md",
        trigger_line: "- When {desc}, read .agents/skills/{name}/SKILL.md and follow it.\n",
        subagent_requires_mode: false,
        plugin_dir: None,
        hooks_file: Some(".codex/hooks.json"),
        hook_events: vec![
            "PreToolUse", "PostToolUse", "PermissionRequest", "PreCompact", "PostCompact", "SessionStart",
            "SubagentStart", "SubagentStop", "UserPromptSubmit", "Stop",
        ],
        compile_skill_scripts: true,
        reject_empty_always_on: true,
        stubs: vec![
            ("scripts/README.md", "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. Add one by emitting scripts/<name>.py. -->\n"),
            ("sub_agents/README.md", "<!-- c5: delegated codex agent roles, each spawned in its OWN context. Add one as sub_agents/<name>/AGENT.md with frontmatter `description:`; the runner installs it as a [agents.<name>] role. -->\n"),
            (".codex/README.md", "<!-- c8: codex lifecycle hooks -- the only lever that acts INSIDE the tool loop, so the model cannot skip it. Add ONE file, .codex/hooks.json, Claude-Code hook schema: {\"hooks\": {\"PostToolUse\": [{\"matcher\": \"Bash\", \"hooks\": [{\"type\": \"command\", \"command\": \"...\"}]}]}}. A PostToolUse command that prints {\"decision\":\"block\",\"reason\":\"...\"} feeds the reason back to the model. -->\n"),
            (".agents/skills/README.md", "<!-- c6: reusable workflows. Add one as .agents/skills/<name>/SKILL.md with YAML frontmatter: `description:` is REQUIRED (no frontmatter = skill silently dropped); `name:` is optional and must match the directory. Only name+description are advertised to the agent; the body is read on demand. -->\n"),
        ],
    }
}

pub const CODEX_ROOT_ENTRIES: &[&str] = &["AGENTS.md", ".agents", "scripts", ".codex"];

pub struct CodexSubstrate;

impl CodexSubstrate {
    pub const NAME: &'static str = "cx";

    pub fn surface(&self) -> SurfaceSpec {
        codex_surface()
    }

    /// Stage exactly the codex_root entries for a copy to the workspace root.
    pub fn materialise(&self, tree_dir: &Path, install_mode: &str, ctx: &BTreeMap<String, String>) -> Result<Install> {
        if install_mode != "codex_root" {
            bail!("install_mode {install_mode:?} is not the codex mode; this substrate has exactly one: 'codex_root'");
        }
        if !tree::is_tree(&codex_surface(), tree_dir) {
            bail!("{} is not a codex harness tree (no AGENTS.md)", tree_dir.display());
        }
        let staged = tempfile::Builder::new().prefix("adapt-cx-root-").tempdir()?.keep();
        for entry in CODEX_ROOT_ENTRIES {
            let src = tree_dir.join(entry);
            if src.is_dir() {
                util::copy_tree(&src, &staged.join(entry))?;
            } else if src.is_file() {
                std::fs::copy(&src, staged.join(entry))?;
            }
        }
        let mut extras: BTreeMap<String, String> = ctx.iter().filter(|(k, _)| !k.starts_with("file:")).map(|(k, v)| (k.clone(), v.clone())).collect();
        extras.insert("install_mode".into(), "codex_root".into());
        Ok(Install { staging_dir: staged, fingerprint: runmark::fingerprint(Some(tree_dir)), extras })
    }

    /// The reference `codex exec` argv.
    pub fn exec_argv(&self, prompt: &str, workspace: &Path, model: &str, reasoning_effort: &str) -> Vec<String> {
        let w = workspace.to_string_lossy().to_string();
        vec![
            config::get().codex_dist.join("codex").to_string_lossy().to_string(),
            "exec".into(),
            prompt.into(),
            "--model".into(),
            model.into(),
            "-c".into(),
            format!("model_reasoning_effort=\"{reasoning_effort}\""),
            "--skip-git-repo-check".into(),
            "--dangerously-bypass-approvals-and-sandbox".into(),
            "--dangerously-bypass-hook-trust".into(),
            "--ephemeral".into(),
            "--json".into(),
            "-C".into(),
            w.clone(),
            "--add-dir".into(),
            w,
        ]
    }

    /// The reference `codex exec` environment.
    pub fn exec_env(&self, codex_home: &Path) -> BTreeMap<String, String> {
        let mut env = config::scrub_env(None, &[]);
        env.remove("OPENAI_BASE_URL");
        env.insert("CODEX_HOME".into(), codex_home.to_string_lossy().to_string());
        let path = env.get("PATH").cloned().unwrap_or_default();
        env.insert("PATH".into(), format!("{}:{path}", config::get().codex_dist.join("codex-path").display()));
        env
    }

    pub fn read_trajectory(&self, stream_path: &Path, obs_trunc: usize, max_chars: usize) -> String {
        render_events(stream_path, obs_trunc, max_chars, 2000)
    }

    /// Check the CLI and Responses endpoint, including container configuration.
    pub fn preflight(&self, require_bridge: bool) -> Result<()> {
        let cfg = config::get();
        config::require("HARNESS2_CODEX_BASE_URL", "Codex Responses endpoint")?;
        config::require("HARNESS2_CODEX_API_KEY", "Codex endpoint authentication")?;
        let url = &cfg.codex_base_url;
        let (scheme, rest) = url.split_once("://").unwrap_or(("", ""));
        let host_port = rest.split('/').next().unwrap_or("");
        if !(scheme == "http" || scheme == "https") || host_port.is_empty() {
            bail!("HARNESS2_CODEX_BASE_URL must be an HTTP(S) URL");
        }
        if !cfg.codex_dist.join("codex").is_file() {
            bail!("Codex is not installed; run `harness2 setup <bench> --substrate cx`");
        }
        if require_bridge && cfg.codex_container_url.is_empty() {
            bail!("Set HARNESS2_CODEX_CONTAINER_URL to a container-accessible Responses endpoint");
        }
        let addr = if host_port.contains(':') { host_port.to_string() } else { format!("{host_port}:{}", if scheme == "https" { 443 } else { 80 }) };
        use std::net::ToSocketAddrs;
        let sa = addr.to_socket_addrs()?.next().ok_or_else(|| anyhow::anyhow!("cannot resolve {addr}"))?;
        std::net::TcpStream::connect_timeout(&sa, std::time::Duration::from_secs(5))?;
        Ok(())
    }
}

const AGENTS_MD_TITLE: &str = "# AGENTS.md instructions";

/// Best-effort text extraction from a message-ish payload (str, .text, content list).
pub fn text_of(obj: Option<&Value>) -> String {
    match obj {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(o)) => {
            for k in ["text", "message", "content", "summary"] {
                match o.get(k) {
                    Some(Value::String(s)) if !s.trim().is_empty() => return s.clone(),
                    Some(v @ (Value::Array(_) | Value::Object(_))) => {
                        let t = text_of(Some(v));
                        if !t.trim().is_empty() {
                            return t;
                        }
                    }
                    _ => {}
                }
            }
            String::new()
        }
        Some(Value::Array(a)) => a.iter().map(|x| text_of(Some(x))).filter(|t| !t.trim().is_empty()).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// Read the exec --json stream (junk tolerated), in stream order, thread-items deduped by
/// .item.id: last state wins, first-appearance position kept.
pub fn parse_items(path: &Path) -> Vec<Value> {
    let raw = match std::fs::read(path) {
        Ok(b) => String::from_utf8_lossy(&b).replace('\0', ""),
        Err(e) => return vec![serde_json::json!({"type": "_read_error", "message": e.to_string()})],
    };
    let mut events: Vec<Value> = vec![];
    let mut by_id: BTreeMap<String, usize> = BTreeMap::new();
    for ln in raw.lines() {
        let ln = ln.trim();
        if !(ln.starts_with("{\"type\":") || ln.starts_with("{\"timestamp\":")) {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<Value>(ln) else { continue };
        if !ev.is_object() {
            continue;
        }
        if let Some(iid) = ev.get("item").and_then(|i| i.get("id")).filter(|v| !v.is_null()) {
            let key = iid.to_string();
            if let Some(i) = by_id.get(&key) {
                events[*i] = ev;
                continue;
            }
            by_id.insert(key, events.len());
        }
        events.push(ev);
    }
    events
}

fn emit(lines: &mut Vec<String>, line: String) {
    if lines.last() != Some(&line) {
        lines.push(line);
    }
}

fn emit_if(lines: &mut Vec<String>, tag: &str, txt: &str, trunc: usize) {
    if !txt.trim().is_empty() {
        emit(lines, format!("[{tag}] {}", clip(txt, trunc)));
    }
}

fn emit_user(lines: &mut Vec<String>, txt: &str, obs: usize) {
    if txt.trim_start().starts_with(AGENTS_MD_TITLE) {
        emit(lines, format!("[HARNESS AGENTS.md] {}", clip(txt, obs)));
    } else {
        emit_if(lines, "USER", txt, obs);
    }
}

fn render_item(item: &Value, lines: &mut Vec<String>, obs: usize, args: usize) {
    let itype = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match itype {
        "command_execution" => {
            lines.push(format!("[ACTION] shell: {}", clip(&pystr(item.get("command")), args)));
            let status = item.get("status").and_then(|s| s.as_str());
            let exit = item.get("exit_code").filter(|v| !v.is_null());
            let nonzero = exit.map(|e| e.as_i64() != Some(0)).unwrap_or(false);
            let failed = status == Some("failed") || nonzero;
            let mut o = clip(&pystr(item.get("aggregated_output")), obs);
            let suffix = if nonzero { format!(" | exit_code={}", exit.unwrap()) } else { String::new() };
            if status == Some("in_progress") && o.is_empty() {
                o = "(no output captured — command still in_progress at end of stream)".into();
            }
            lines.push(format!("[OBSERVATION{}] {o}{suffix}", if failed { " ERROR" } else { "" }));
        }
        "agent_message" => {
            if let Some(t) = item.get("text").and_then(|t| t.as_str()).filter(|t| !t.trim().is_empty()) {
                lines.push(format!("[ASSISTANT] {}", clip(t, obs)));
            }
        }
        "reasoning" => {
            let t = text_of(Some(item));
            if !t.trim().is_empty() {
                lines.push(format!("[REASONING] {}", clip(&t, obs)));
            }
        }
        "todo_list" => {
            if let Some(rows) = item.get("items").and_then(|r| r.as_array()).filter(|r| !r.is_empty()) {
                let plan = rows
                    .iter()
                    .filter(|r| r.is_object())
                    .map(|r| {
                        format!(
                            "[{}] {}",
                            if r.get("completed").and_then(|c| c.as_bool()).unwrap_or(false) { "x" } else { " " },
                            clip(&pystr(r.get("text")), 200)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" | ");
                lines.push(format!("[ASSISTANT] plan: {plan}"));
            }
        }
        "error" => lines.push(format!("[HARNESS] {}", clip(&pystr(item.get("message")), 1000))),
        "user_message" | "message" => {
            let t = text_of(Some(item));
            if t.trim_start().starts_with(AGENTS_MD_TITLE) {
                lines.push(format!("[HARNESS AGENTS.md] {}", clip(&t, obs)));
            } else if !t.trim().is_empty() {
                lines.push(format!("[USER] {}", clip(&t, obs)));
            }
        }
        _ => {
            let t = text_of(Some(item));
            if !t.trim().is_empty() {
                lines.push(format!("[HARNESS] {}: {}", if itype.is_empty() { "item" } else { itype }, clip(&t, obs)));
            }
        }
    }
}

fn render_rollout_payload(payload: &Value, lines: &mut Vec<String>, obs: usize, args_trunc: usize) {
    let ptype = payload.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match ptype {
        "function_call" => {
            let mut args = payload.get("arguments").cloned().unwrap_or(Value::Null);
            if let Value::String(s) = &args {
                if let Ok(v) = serde_json::from_str::<Value>(s) {
                    args = v;
                }
            }
            let cmd = if args.is_object() { args.get("cmd").cloned().unwrap_or(Value::Null) } else { args.clone() };
            let name = payload.get("name").and_then(|n| n.as_str()).filter(|n| !n.is_empty()).unwrap_or("shell");
            let shown = if cmd.is_null() { pystr(Some(&args)) } else { pystr(Some(&cmd)) };
            emit(lines, format!("[ACTION] {name}: {}", clip(&shown, args_trunc)));
        }
        "function_call_output" => {
            let out = payload.get("output");
            let t = text_of(out);
            let t = if t.is_empty() { pystr(out) } else { t };
            emit(lines, format!("[OBSERVATION] {}", clip(&t, obs)));
        }
        "message" => {
            let role = payload.get("role").and_then(|r| r.as_str()).unwrap_or("");
            let t = text_of(Some(payload));
            match role {
                "user" => emit_user(lines, &t, obs),
                "assistant" => emit_if(lines, "ASSISTANT", &t, obs),
                r => emit_if(lines, &format!("HARNESS {}", if r.is_empty() { "message" } else { r }), &t, obs),
            }
        }
        "user_message" => emit_user(lines, &text_of(Some(payload)), obs),
        "agent_message" => emit_if(lines, "ASSISTANT", &text_of(Some(payload)), obs),
        "reasoning" => emit_if(lines, "REASONING", &text_of(Some(payload)), obs),
        _ => {}
    }
}

/// Render a codex exec/rollout stream into the shared grammar.
pub fn render_events(path: &Path, obs_trunc: usize, max_chars: usize, args_trunc: usize) -> String {
    let mut lines = vec![];
    for ev in parse_items(path) {
        let et = ev.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if et == "_read_error" {
            return format!("(failed to read trajectory: {})", pystr(ev.get("message")));
        }
        if let Some(item) = ev.get("item").filter(|i| i.is_object()) {
            render_item(item, &mut lines, obs_trunc, args_trunc);
            continue;
        }
        if let Some(p) = ev.get("payload").filter(|p| p.is_object()) {
            render_rollout_payload(p, &mut lines, obs_trunc, args_trunc);
            continue;
        }
        if et == "turn.completed" {
            if let Some(u) = ev.get("usage").filter(|u| u.is_object()) {
                let f = |k: &str| u.get(k).map(|v| v.to_string()).unwrap_or_else(|| "None".into());
                lines.push(format!(
                    "[HARNESS] turn completed | tokens in={} cached={} out={} reasoning={}",
                    f("input_tokens"), f("cached_input_tokens"), f("output_tokens"), f("reasoning_output_tokens")
                ));
            }
            continue;
        }
        if et == "turn.failed" || et == "error" || ev.get("error").map(|e| e.is_object() || e.is_string()).unwrap_or(false) {
            let mut msg = text_of(ev.get("error"));
            if msg.is_empty() {
                msg = text_of(Some(&ev));
            }
            if msg.is_empty() {
                msg = util::take_chars(&ev.to_string(), 300);
            }
            lines.push(format!("[ERROR] {}", clip(&msg, 1000)));
        }
    }
    cap_rendered(&lines, max_chars)
}

/// Last non-empty agent_message text: the deliverable summary before turn.completed.
pub fn final_message(path: &Path) -> String {
    let mut last = String::new();
    for ev in parse_items(path) {
        if let Some(item) = ev.get("item").filter(|i| i.is_object()) {
            if item.get("type").and_then(|t| t.as_str()) == Some("agent_message") {
                if let Some(t) = item.get("text").and_then(|t| t.as_str()).filter(|t| !t.trim().is_empty()) {
                    last = t.to_string();
                }
            }
            continue;
        }
        if let Some(p) = ev.get("payload").filter(|p| p.is_object()) {
            match p.get("type").and_then(|t| t.as_str()) {
                Some("agent_message") => {
                    let t = text_of(Some(p));
                    if !t.trim().is_empty() {
                        last = t;
                    }
                }
                Some("task_complete") => {
                    let t = text_of(p.get("last_agent_message"));
                    if !t.trim().is_empty() {
                        last = t;
                    }
                }
                _ => {}
            }
        }
    }
    last
}

pub fn codex_home() -> PathBuf {
    config::get().codex_home_dir.clone()
}
