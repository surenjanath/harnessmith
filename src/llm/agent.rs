//! A tool-using agent loop over any connected model.
//!
//! Tools work inside one workspace directory: `list_files`, `read_file`, `write_file`,
//! `edit_file`, `delete_file`, optionally `bash`, and -- when a harness is installed -- `skill`
//! (load a skill body) and `task` (run a sub-agent in its own context). `hooks.json` lifecycle
//! hooks (`PreToolUse`, `PostToolUse`, `Stop`) run inside the loop, so a harness can enforce
//! what prose cannot. Every event is appended to a JSONL trajectory.

use super::client::{self, ChatReq, Msg, ToolCall, ToolDef, ToolResult};
use super::connections;
use crate::{config, say, util};
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// `<connection>/<model>`.
    pub model_id: String,
    pub system: String,
    pub workspace: PathBuf,
    pub max_steps: u32,
    pub wall: Duration,
    pub allow_shell: bool,
    /// A harness installed at `<workspace>/.harness` enables `skill`, `task` and hooks.
    pub harness: Option<PathBuf>,
    pub trajectory: Option<PathBuf>,
    pub max_tokens: u32,
    pub depth: u32,
    /// Log a one-line summary of each tool call.
    pub verbose: bool,
    pub log_prefix: String,
}

impl AgentConfig {
    pub fn new(model_id: &str, system: &str, workspace: &Path) -> AgentConfig {
        AgentConfig {
            model_id: model_id.into(),
            system: system.into(),
            workspace: workspace.into(),
            max_steps: 60,
            wall: Duration::from_secs(1800),
            allow_shell: config::env("HARNESS2_ALLOW_SHELL").map(|v| v != "0").unwrap_or(true),
            harness: None,
            trajectory: None,
            max_tokens: 8192,
            depth: 0,
            verbose: true,
            log_prefix: "    ".into(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AgentOutcome {
    pub final_text: String,
    pub cost_usd: f64,
    pub steps: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub error: String,
    pub finished: bool,
}

fn obj(props: Value, required: &[&str]) -> Value {
    json!({"type": "object", "properties": props, "required": required})
}

pub fn tool_defs(cfg: &AgentConfig) -> Vec<ToolDef> {
    let mut t = vec![
        ToolDef {
            name: "list_files".into(),
            description: "List files under a directory of the workspace (recursive, up to 500 entries).".into(),
            parameters: obj(json!({"path": {"type": "string", "description": "directory relative to the workspace; default '.'"}}), &[]),
        },
        ToolDef {
            name: "read_file".into(),
            description: "Read a text file from the workspace. Use offset/limit (in lines) to page through large files.".into(),
            parameters: obj(
                json!({"path": {"type": "string"}, "offset": {"type": "integer", "description": "first line, 1-based"}, "limit": {"type": "integer", "description": "number of lines"}}),
                &["path"],
            ),
        },
        ToolDef {
            name: "write_file".into(),
            description: "Create or overwrite a text file in the workspace (parent directories are created).".into(),
            parameters: obj(json!({"path": {"type": "string"}, "content": {"type": "string"}}), &["path", "content"]),
        },
        ToolDef {
            name: "edit_file".into(),
            description: "Replace an exact substring in a file. old_string must occur exactly once unless replace_all is true.".into(),
            parameters: obj(
                json!({"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}),
                &["path", "old_string", "new_string"],
            ),
        },
        ToolDef {
            name: "delete_file".into(),
            description: "Delete a file from the workspace.".into(),
            parameters: obj(json!({"path": {"type": "string"}}), &["path"]),
        },
        ToolDef {
            name: "finish".into(),
            description: "End the session. Pass your final answer or summary.".into(),
            parameters: obj(json!({"answer": {"type": "string"}}), &["answer"]),
        },
    ];
    if cfg.allow_shell {
        t.push(ToolDef {
            name: "bash".into(),
            description: "Run a shell command in the workspace directory (e.g. `python3 scripts/x.py`). Output is truncated to 30k chars.".into(),
            parameters: obj(json!({"command": {"type": "string"}, "timeout_s": {"type": "integer", "description": "default 120, max 600"}}), &["command"]),
        });
    }
    if cfg.harness.is_some() {
        t.push(ToolDef {
            name: "skill".into(),
            description: "Load the full instructions of a named skill from the harness.".into(),
            parameters: obj(json!({"name": {"type": "string"}}), &["name"]),
        });
        if cfg.depth == 0 {
            t.push(ToolDef {
                name: "task".into(),
                description: "Delegate a bounded sub-task to a named sub-agent. It runs in its OWN context (it cannot see this conversation) and returns text.".into(),
                parameters: obj(json!({"agent": {"type": "string"}, "prompt": {"type": "string"}}), &["agent", "prompt"]),
            });
        }
    }
    t
}

/// Resolve a model-supplied path inside the workspace, refusing escapes.
pub fn safe_path(ws: &Path, p: &str) -> Result<PathBuf> {
    let p = p.trim();
    let pb = Path::new(p);
    let rel = if pb.is_absolute() {
        match pb.strip_prefix(ws) {
            Ok(r) => r.to_path_buf(),
            Err(_) => bail!("path {p:?} is outside the workspace; use a path relative to it"),
        }
    } else {
        pb.to_path_buf()
    };
    if rel.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        bail!("path {p:?} escapes the workspace");
    }
    Ok(ws.join(rel))
}

fn strip_frontmatter(body: &str) -> String {
    if let Some(rest) = body.strip_prefix("---") {
        if let Some(i) = rest.find("\n---") {
            return rest[i + 4..].trim_start_matches(['\r', '\n']).to_string();
        }
    }
    body.to_string()
}

struct Traj(Option<std::fs::File>);
impl Traj {
    fn ev(&mut self, v: Value) {
        if let Some(f) = &mut self.0 {
            let _ = writeln!(f, "{v}");
        }
    }
}

/// Run one hook event; `Some(reason)` when a hook blocked.
fn run_hooks(cfg: &AgentConfig, event: &str, tool: &str, input: &Value, response: Option<&str>) -> Option<String> {
    let h = cfg.harness.as_ref()?.join("hooks.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(h).ok()?).ok()?;
    let entries = doc.get("hooks")?.get(event)?.as_array()?.clone();
    let mut reasons = vec![];
    for e in entries {
        let matcher = e.get("matcher").and_then(|m| m.as_str()).unwrap_or("");
        if !matcher.is_empty() && matcher != "*" {
            let re = Regex::new(&format!("^(?:{matcher})$")).ok()?;
            if !re.is_match(tool) {
                continue;
            }
        }
        for hk in e.get("hooks").and_then(|x| x.as_array()).cloned().unwrap_or_default() {
            let Some(cmd) = hk.get("command").and_then(|c| c.as_str()) else { continue };
            let mut payload = json!({"hook_event_name": event, "tool_name": tool, "tool_input": input, "cwd": cfg.workspace});
            if let Some(r) = response {
                payload["tool_response"] = json!(r);
            }
            let mut c = std::process::Command::new("bash");
            c.arg("-c").arg(cmd).current_dir(&cfg.workspace);
            let Ok(out) = util::run_with_timeout(c, Some(&payload.to_string()), Duration::from_secs(60)) else { continue };
            let s = out.stdout.trim();
            if let Ok(v) = serde_json::from_str::<Value>(s) {
                if v.get("decision").and_then(|d| d.as_str()) == Some("block") {
                    reasons.push(v.get("reason").and_then(|r| r.as_str()).unwrap_or("blocked by hook").to_string());
                }
            } else if out.code == Some(2) {
                reasons.push(util::take_chars(out.stderr.trim(), 2000));
            }
        }
    }
    (!reasons.is_empty()).then(|| reasons.join("\n"))
}

fn list_files(ws: &Path, dir: &Path) -> String {
    let mut out = vec![];
    for p in util::files_sorted(dir) {
        let rel = util::rel(ws, &p);
        if rel.starts_with(".harness/") || util::in_pycache(&p) {
            continue;
        }
        let size = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        out.push(format!("{rel}  ({size} bytes)"));
        if out.len() >= 500 {
            out.push("… (truncated at 500 entries)".into());
            break;
        }
    }
    if out.is_empty() { "(no files)".into() } else { out.join("\n") }
}

/// Execute one tool call. Ok(text) is the observation; Err is reported to the model as an error.
fn exec_tool(cfg: &AgentConfig, call: &ToolCall) -> Result<String> {
    let a = &call.args;
    let s = |k: &str| a.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let ws = &cfg.workspace;
    match call.name.as_str() {
        "list_files" => {
            let p = s("path");
            let dir = safe_path(ws, if p.is_empty() { "." } else { &p })?;
            Ok(list_files(ws, &dir))
        }
        "read_file" => {
            let p = safe_path(ws, &s("path"))?;
            let body = util::read_lossy(&p)?;
            let off = a.get("offset").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;
            let lim = a.get("limit").and_then(|v| v.as_u64()).map(|v| v as usize);
            let lines: Vec<&str> = body.lines().collect();
            let total = lines.len();
            let end = lim.map(|l| (off - 1 + l).min(total)).unwrap_or(total);
            let mut text = lines.get(off - 1..end).map(|s| s.join("\n")).unwrap_or_default();
            if util::char_len(&text) > 100_000 {
                text = format!("{}\n…(truncated; {total} lines total -- page with offset/limit)", util::take_chars(&text, 100_000));
            } else if lim.is_some() || off > 1 {
                text = format!("{text}\n(lines {off}-{end} of {total})");
            }
            Ok(text)
        }
        "write_file" => {
            let p = safe_path(ws, &s("path"))?;
            let c = s("content");
            util::write(&p, &c)?;
            Ok(format!("wrote {} ({} chars)", util::rel(ws, &p), util::char_len(&c)))
        }
        "edit_file" => {
            let p = safe_path(ws, &s("path"))?;
            let body = util::read_lossy(&p)?;
            let (old, new) = (s("old_string"), s("new_string"));
            if old.is_empty() {
                bail!("old_string must not be empty");
            }
            let n = body.matches(&old).count();
            let all = a.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);
            if n == 0 {
                bail!("old_string not found in {}", util::rel(ws, &p));
            }
            if n > 1 && !all {
                bail!("old_string occurs {n} times; make it unique or set replace_all");
            }
            let out = if all { body.replace(&old, &new) } else { body.replacen(&old, &new, 1) };
            util::write(&p, out)?;
            Ok(format!("edited {} ({n} replacement{})", util::rel(ws, &p), if n == 1 { "" } else { "s" }))
        }
        "delete_file" => {
            let p = safe_path(ws, &s("path"))?;
            if !p.is_file() {
                bail!("no such file: {}", util::rel(ws, &p));
            }
            std::fs::remove_file(&p)?;
            Ok(format!("deleted {}", util::rel(ws, &p)))
        }
        "bash" if cfg.allow_shell => {
            let t = a.get("timeout_s").and_then(|v| v.as_u64()).unwrap_or(120).clamp(1, 600);
            let mut c = std::process::Command::new("bash");
            c.arg("-c").arg(s("command")).current_dir(ws);
            config::apply_env(&mut c, &config::scrub_env(None, &[]));
            let out = util::run_with_timeout(c, None, Duration::from_secs(t))?;
            let mut text = out.stdout;
            if !out.stderr.trim().is_empty() {
                text = format!("{text}\n[stderr]\n{}", out.stderr);
            }
            if out.timed_out {
                text = format!("{text}\n(killed after {t}s timeout)");
            }
            if util::char_len(&text) > 30_000 {
                text = format!("{}\n…(output truncated)", util::take_chars(&text, 30_000));
            }
            let code = out.code.map(|c| c.to_string()).unwrap_or_else(|| "killed".into());
            if out.code != Some(0) {
                bail!("exit_code={code}\n{text}");
            }
            Ok(if text.trim().is_empty() { "(no output)".into() } else { text })
        }
        "skill" => {
            let h = cfg.harness.as_ref().ok_or_else(|| anyhow::anyhow!("no harness installed"))?;
            let name = s("name");
            let p = h.join("skills").join(&name).join("SKILL.md");
            if name.contains('/') || !p.is_file() {
                let avail: Vec<String> = std::fs::read_dir(h.join("skills"))
                    .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| e.path().join("SKILL.md").is_file()).map(|e| e.file_name().to_string_lossy().to_string()).collect())
                    .unwrap_or_default();
                bail!("no skill named {name:?}; available: {}", if avail.is_empty() { "(none)".into() } else { avail.join(", ") });
            }
            Ok(strip_frontmatter(&util::read_lossy(&p)?))
        }
        "task" if cfg.depth == 0 => {
            let h = cfg.harness.as_ref().ok_or_else(|| anyhow::anyhow!("no harness installed"))?;
            let name = s("agent");
            let p = h.join("sub_agents").join(&name).join("AGENT.md");
            if name.contains('/') || !p.is_file() {
                bail!("no sub-agent named {name:?}");
            }
            let mut sub = cfg.clone();
            sub.system = strip_frontmatter(&util::read_lossy(&p)?);
            sub.depth = 1;
            sub.max_steps = cfg.max_steps.min(40);
            sub.log_prefix = format!("{}  [{name}] ", cfg.log_prefix);
            let r = run(&sub, &s("prompt"))?;
            Ok(if r.final_text.trim().is_empty() { format!("(sub-agent returned nothing{})", if r.error.is_empty() { String::new() } else { format!(": {}", r.error) }) } else { r.final_text })
        }
        "finish" => Ok(s("answer")),
        other => bail!("unknown tool {other:?}"),
    }
}

fn summarize_args(v: &Value) -> String {
    let s = match v.get("path").or_else(|| v.get("command")).or_else(|| v.get("name")).or_else(|| v.get("agent")) {
        Some(Value::String(s)) => s.clone(),
        _ => v.to_string(),
    };
    util::take_chars(&s.replace('\n', " "), 100)
}

/// Shrink old tool observations when the conversation grows past `budget` chars.
fn compact(messages: &mut [Msg], budget: usize) {
    let size = |m: &Msg| match m {
        Msg::User(t) => t.len(),
        Msg::Assistant { text, calls } => text.len() + calls.iter().map(|c| c.args.to_string().len()).sum::<usize>(),
        Msg::Tools(rs) => rs.iter().map(|r| r.content.len()).sum(),
    };
    let mut total: usize = messages.iter().map(size).sum();
    let n = messages.len();
    for m in messages.iter_mut().take(n.saturating_sub(6)) {
        if total <= budget {
            break;
        }
        if let Msg::Tools(rs) = m {
            for r in rs.iter_mut() {
                if r.content.len() > 2000 {
                    let before = r.content.len();
                    r.content = format!("{}\n…(older observation elided to save context; re-run the tool if you need it)", util::take_chars(&r.content, 1500));
                    total -= before - r.content.len();
                }
            }
        }
    }
}

/// Run the loop until the model finishes, the step budget runs out, or the wall clock does.
pub fn run(cfg: &AgentConfig, task: &str) -> Result<AgentOutcome> {
    let (conn, model) = connections::resolve(&cfg.model_id)?;
    let tools = tool_defs(cfg);
    let mut traj = Traj(match &cfg.trajectory {
        Some(p) => {
            if let Some(d) = p.parent() {
                std::fs::create_dir_all(d)?;
            }
            Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?)
        }
        None => None,
    });
    traj.ev(json!({"type": "session", "model": cfg.model_id, "depth": cfg.depth, "at": util::now_iso()}));
    traj.ev(json!({"type": "user", "text": task}));
    let mut messages = vec![Msg::User(task.to_string())];
    let mut out = AgentOutcome::default();
    let t0 = Instant::now();
    let max_tokens = conn.max_tokens.unwrap_or(cfg.max_tokens);
    let mut stop_blocks = 0;
    loop {
        if out.steps >= cfg.max_steps {
            out.error = format!("step budget of {} exhausted", cfg.max_steps);
            traj.ev(json!({"type": "error", "message": out.error}));
            break;
        }
        let left = cfg.wall.checked_sub(t0.elapsed()).unwrap_or_default();
        if left.is_zero() {
            out.error = format!("wall-clock budget of {}s exhausted", cfg.wall.as_secs());
            traj.ev(json!({"type": "error", "message": out.error}));
            break;
        }
        out.steps += 1;
        compact(&mut messages, 600_000);
        let resp = match client::chat(&conn, &model, &ChatReq { system: &cfg.system, messages: &messages, tools: &tools, max_tokens }, left.max(Duration::from_secs(30))) {
            Ok(r) => r,
            Err(e) => {
                out.error = format!("{e}");
                traj.ev(json!({"type": "error", "message": out.error}));
                break;
            }
        };
        out.input_tokens += resp.input_tokens;
        out.output_tokens += resp.output_tokens;
        out.cost_usd += (resp.input_tokens as f64 * conn.input_price + resp.output_tokens as f64 * conn.output_price) / 1e6;
        if !resp.text.trim().is_empty() {
            traj.ev(json!({"type": "assistant", "text": resp.text}));
        }
        traj.ev(json!({"type": "usage", "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}));
        messages.push(Msg::Assistant { text: resp.text.clone(), calls: resp.calls.clone() });
        if resp.calls.is_empty() {
            if let Some(reason) = run_hooks(cfg, "Stop", "", &json!({}), None).filter(|_| stop_blocks < 3) {
                stop_blocks += 1;
                traj.ev(json!({"type": "hook", "event": "Stop", "reason": reason}));
                messages.push(Msg::User(format!("A Stop hook refused to end the session:\n{reason}")));
                continue;
            }
            out.final_text = resp.text;
            out.finished = true;
            break;
        }
        let mut results = vec![];
        let mut finish: Option<String> = None;
        for call in &resp.calls {
            if cfg.verbose {
                say!("{}· {}({})", cfg.log_prefix, call.name, summarize_args(&call.args));
            }
            if let Some(reason) = run_hooks(cfg, "PreToolUse", &call.name, &call.args, None) {
                traj.ev(json!({"type": "tool", "name": call.name, "input": call.args, "status": "error", "output": format!("blocked by PreToolUse hook: {reason}")}));
                results.push(ToolResult { id: call.id.clone(), name: call.name.clone(), content: format!("Blocked by a PreToolUse hook: {reason}"), is_error: true });
                continue;
            }
            let (mut content, is_error) = match exec_tool(cfg, call) {
                Ok(t) => (t, false),
                Err(e) => (format!("ERROR: {e}"), true),
            };
            if let Some(reason) = run_hooks(cfg, "PostToolUse", &call.name, &call.args, Some(&content)) {
                content = format!("{content}\n\n[PostToolUse hook] {reason}");
            }
            traj.ev(json!({"type": "tool", "name": call.name, "input": call.args, "status": if is_error { "error" } else { "ok" }, "output": content}));
            if call.name == "finish" && !is_error {
                finish = Some(content.clone());
            }
            results.push(ToolResult { id: call.id.clone(), name: call.name.clone(), content, is_error });
        }
        messages.push(Msg::Tools(results));
        if let Some(ans) = finish {
            if let Some(reason) = run_hooks(cfg, "Stop", "finish", &json!({}), None).filter(|_| stop_blocks < 3) {
                stop_blocks += 1;
                traj.ev(json!({"type": "hook", "event": "Stop", "reason": reason}));
                messages.push(Msg::User(format!("A Stop hook refused to end the session:\n{reason}")));
                continue;
            }
            out.final_text = ans;
            out.finished = true;
            break;
        }
    }
    traj.ev(json!({"type": "finish", "text": out.final_text, "error": out.error, "steps": out.steps, "cost_usd": out.cost_usd,
        "input_tokens": out.input_tokens, "output_tokens": out.output_tokens}));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn path_sandbox() {
        let ws = Path::new("/tmp/ws");
        assert!(safe_path(ws, "a/b.txt").is_ok());
        assert!(safe_path(ws, "../x").is_err());
        assert!(safe_path(ws, "/etc/passwd").is_err());
        assert!(safe_path(ws, "/tmp/ws/ok.md").is_ok());
    }
    #[test]
    fn frontmatter() {
        assert_eq!(strip_frontmatter("---\nname: x\n---\nbody"), "body");
        assert_eq!(strip_frontmatter("plain"), "plain");
    }
}
