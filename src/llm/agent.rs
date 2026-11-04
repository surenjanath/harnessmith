//! A tool-using agent loop over any connected model.
//!
//! Tools work inside one workspace directory: `list_files`, `read_file`, `write_file`,
//! `edit_file`, `delete_file`, optionally `bash`, and -- when a harness is installed -- `skill`
//! (load a skill body) and `task` (run a sub-agent in its own context). `hooks.json` lifecycle
//! hooks (`PreToolUse`, `PostToolUse`, `Stop`) run inside the loop, so a harness can enforce
//! what prose cannot. Every event is appended to a JSONL trajectory.

use super::client::{self, ChatReq, Msg, ToolCall, ToolDef, ToolResult};
use super::{connections, health};
use std::collections::{HashMap, HashSet};
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
    /// Self-healing deadline rescue: when the step or time budget is nearly gone and this
    /// directory is still empty (None: always), append `rescue_hint` to the next observation.
    pub rescue_dir: Option<PathBuf>,
    pub rescue_hint: String,
    /// The operator's guardrail gate, checked before every tool call (None: no gate).
    pub gate: Option<super::gate::Gate>,
    /// Project memory: enables `remember` / `recall`, with where new notes come from.
    pub memory: Option<(super::memory::Store, super::memory::Source)>,
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
            rescue_dir: None,
            rescue_hint: String::new(),
            gate: None,
            memory: None,
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
    /// Self-healing actions taken (failovers, loop breaks, rescues...).
    pub heals: u32,
    /// The model id that produced the last reply.
    pub served_by: String,
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
    if cfg.memory.is_some() {
        t.push(ToolDef {
            name: "remember".into(),
            description: "Save one durable, reusable note for later goals in this project: a fact about the inputs, a technique that worked, a pitfall to avoid. Not the task's answer, never a secret. At most 600 chars.".into(),
            parameters: obj(json!({"note": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}, "description": "a few lowercase keywords"}}), &["note"]),
        });
        t.push(ToolDef {
            name: "recall".into(),
            description: "Search notes saved by earlier goals in this project. Pass a few keywords about the task or a step; returns the best matches.".into(),
            parameters: obj(json!({"query": {"type": "string"}, "k": {"type": "integer", "description": "how many notes, default 5"}}), &[]),
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
        // Wired tools run commands, so they follow the shell setting.
        if cfg.allow_shell {
            t.extend(wired_tools(cfg.harness.as_ref().unwrap()).into_iter().map(|w| w.def));
        }
    }
    t
}

/// A tool declared by the harness in `tools/<name>.json`.
#[derive(Debug, Clone)]
pub struct WiredTool {
    pub def: ToolDef,
    pub command: String,
    pub timeout_s: u64,
}

/// Load the harness's wired tools; malformed files and built-in names are skipped (the
/// validator reports them when the harness is proposed).
pub fn wired_tools(harness: &Path) -> Vec<WiredTool> {
    let mut out = vec![];
    let Ok(rd) = std::fs::read_dir(harness.join("tools")) else { return out };
    let mut paths: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map(|x| x == "json").unwrap_or(false)).collect();
    paths.sort();
    for p in paths {
        let Ok(v) = serde_json::from_str::<Value>(&std::fs::read_to_string(&p).unwrap_or_default()) else { continue };
        let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let command = v.get("command").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        if name.is_empty() || name != stem || command.trim().is_empty() || crate::core::tree::BUILTIN_TOOLS.contains(&name.as_str()) {
            continue;
        }
        let params = v.get("parameters").cloned().filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("object")).unwrap_or_else(|| obj(json!({}), &[]));
        out.push(WiredTool {
            def: ToolDef { name, description: v.get("description").and_then(|x| x.as_str()).unwrap_or("").to_string(), parameters: params },
            command,
            timeout_s: v.get("timeout_s").and_then(|x| x.as_u64()).unwrap_or(120).clamp(1, 600),
        });
    }
    out
}

fn find_wired(cfg: &AgentConfig, name: &str) -> Option<WiredTool> {
    if !cfg.allow_shell {
        return None;
    }
    wired_tools(cfg.harness.as_ref()?).into_iter().find(|t| t.def.name == name)
}

/// Run a wired tool: bash in the workspace, arguments as JSON on stdin and as `ARG_<NAME>`.
fn run_wired(cfg: &AgentConfig, t: &WiredTool, args: &Value) -> Result<String> {
    if let Some(req) = t.def.parameters.get("required").and_then(|r| r.as_array()) {
        let missing: Vec<String> = req.iter().filter_map(|k| k.as_str()).filter(|k| args.get(*k).map(|v| v.is_null()).unwrap_or(true)).map(String::from).collect();
        if !missing.is_empty() {
            bail!("{} needs: {}", t.def.name, missing.join(", "));
        }
    }
    let ws = &cfg.workspace;
    let mut c = std::process::Command::new("bash");
    c.arg("-c").arg(&t.command).current_dir(ws);
    let mut extra: Vec<(String, String)> = vec![("TOOL_ARGS".into(), args.to_string())];
    if let Some(h) = &cfg.harness {
        extra.push(("HARNESS_DIR".into(), h.display().to_string()));
    }
    if let Some(m) = args.as_object() {
        for (k, v) in m {
            let key: String = k.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch.to_ascii_uppercase() } else { '_' }).collect();
            extra.push((format!("ARG_{key}"), v.as_str().map(String::from).unwrap_or_else(|| v.to_string())));
        }
    }
    let mut env = config::scrub_env(None, &[]);
    env.extend(extra);
    config::apply_env(&mut c, &env);
    let out = util::run_with_timeout(c, Some(&args.to_string()), Duration::from_secs(t.timeout_s))?;
    let mut text = out.stdout;
    if !out.stderr.trim().is_empty() {
        text = format!("{text}\n[stderr]\n{}", out.stderr);
    }
    if out.timed_out {
        text = format!("{text}\n(killed after {}s timeout)", t.timeout_s);
    }
    if util::char_len(&text) > 30_000 {
        text = format!("{}\n…(output truncated)", util::take_chars(&text, 30_000));
    }
    if out.code != Some(0) {
        bail!("{} failed (exit {})\n{text}", t.def.name, out.code.map(|c| c.to_string()).unwrap_or_else(|| "killed".into()));
    }
    Ok(if text.trim().is_empty() { "(no output)".into() } else { text })
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

/// Required arguments per tool, so a call with missing ones gets a precise error.
fn required(tool: &str) -> &'static [&'static str] {
    match tool {
        "read_file" | "delete_file" => &["path"],
        "write_file" => &["path", "content"],
        "edit_file" => &["path", "old_string", "new_string"],
        "bash" => &["command"],
        "skill" => &["name"],
        "task" => &["agent", "prompt"],
        "remember" => &["note"],
        _ => &[],
    }
}

/// When a path does not exist: the closest file by name, and what is actually there.
fn missing_path_hint(ws: &Path, wanted: &str) -> String {
    let files: Vec<String> = util::files_sorted(ws).iter().map(|p| util::rel(ws, p)).filter(|r| !r.starts_with(".harness/")).collect();
    let base = wanted.rsplit('/').next().unwrap_or(wanted).to_lowercase();
    let close = files.iter().find(|f| f.rsplit('/').next().unwrap_or(f).to_lowercase() == base).or_else(|| {
        files.iter().find(|f| !base.is_empty() && f.to_lowercase().contains(base.trim_end_matches(|c: char| !c.is_alphanumeric())))
    });
    let shown: Vec<&String> = files.iter().take(30).collect();
    format!(
        "{}Files in the workspace: {}{}",
        close.map(|c| format!("Did you mean `{c}`? ")).unwrap_or_default(),
        if shown.is_empty() { "(none)".to_string() } else { shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ") },
        if files.len() > 30 { format!(" (+{} more)", files.len() - 30) } else { String::new() }
    )
}

/// Execute one tool call, turning low-level failures into messages a model can act on.
fn exec_tool(cfg: &AgentConfig, call: &ToolCall) -> Result<String> {
    let missing: Vec<&str> = required(&call.name).iter().filter(|k| call.args.get(**k).map(|v| v.is_null()).unwrap_or(true)).cloned().collect();
    if !missing.is_empty() {
        bail!("{} needs {} (missing: {}). Arguments must be a JSON object.", call.name, required(&call.name).iter().map(|k| format!("`{k}`")).collect::<Vec<_>>().join(", "), missing.join(", "));
    }
    match exec_tool_raw(cfg, call) {
        Ok(t) => Ok(t),
        Err(e) => {
            let ws = &cfg.workspace;
            let msg = format!("{e:#}").replace(&format!("{}/", ws.display()), "").replace(&ws.display().to_string(), ".");
            let p = call.args.get("path").and_then(|v| v.as_str()).unwrap_or("");
            let not_found = msg.contains("No such file") || msg.contains("not found") || msg.contains("no such file");
            if not_found && matches!(call.name.as_str(), "read_file" | "edit_file" | "delete_file" | "list_files") {
                bail!("{p:?} does not exist. {}", missing_path_hint(ws, p));
            }
            if not_found && call.name == "bash" {
                bail!("{msg}\n{}", missing_path_hint(ws, ""));
            }
            bail!("{msg}")
        }
    }
}

fn exec_tool_raw(cfg: &AgentConfig, call: &ToolCall) -> Result<String> {
    let a = &call.args;
    let s = |k: &str| a.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let ws = &cfg.workspace;
    match call.name.as_str() {
        "list_files" => {
            let p = s("path");
            let dir = safe_path(ws, if p.is_empty() { "." } else { &p })?;
            if !dir.exists() {
                bail!("No such file or directory");
            }
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
        "remember" if cfg.memory.is_some() => {
            let (store, src) = cfg.memory.as_ref().unwrap();
            let mut src = src.clone();
            src.node = crate::log::node();
            src.mode = crate::log::ctx().mode;
            let tags: Vec<String> = a.get("tags").and_then(|t| t.as_array()).map(|v| v.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default();
            let (n, dup) = store.remember(&s("note"), &tags, src, "candidate")?;
            crate::log::event("memory", "remember", json!({"id": n.id, "text": util::take_chars(&n.text, 160), "dup": dup}));
            Ok(if dup { format!("Already in memory as {} (tags merged).", n.id) } else { format!("Saved as {}. It becomes active for later goals if this attempt is adopted.", n.id) })
        }
        "recall" if cfg.memory.is_some() => {
            let (store, src) = cfg.memory.as_ref().unwrap();
            let q = s("query");
            let k = a.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
            let hits = store.recall(&q, k, &src.goal)?;
            crate::log::event("memory", "recall", json!({"query": util::take_chars(&q, 120), "hits": hits.len(), "ids": hits.iter().map(|n| n.id.clone()).collect::<Vec<_>>()}));
            if hits.is_empty() {
                return Ok("No matching notes yet.".into());
            }
            Ok(hits.iter().map(|n| format!("- [{}{}] {}{}", n.id, if n.status == "candidate" { ", unconfirmed" } else { "" }, n.text, if n.tags.is_empty() { String::new() } else { format!("  (tags: {})", n.tags.join(", ")) })).collect::<Vec<_>>().join("\n"))
        }
        other => match find_wired(cfg, other) {
            Some(t) => run_wired(cfg, &t, a),
            None => bail!("unknown tool {other:?}"),
        },
    }
}

fn summarize_args(v: &Value) -> String {
    let s = match v.get("path").or_else(|| v.get("command")).or_else(|| v.get("name")).or_else(|| v.get("agent")) {
        Some(Value::String(s)) => s.clone(),
        _ => v.to_string(),
    };
    util::take_chars(&s.replace('\n', " "), 100)
}

/// A reply that tried to call a tool in text but could not be parsed.
fn looks_like_broken_call(text: &str) -> bool {
    let t = text.trim();
    t.contains("<tool_call>") || t.starts_with("<|python_tag|>") || ((t.starts_with('{') || t.starts_with("```json")) && t.contains("\"name\"") && (t.contains("\"arguments\"") || t.contains("\"parameters\"")))
}

/// Keep the conversation inside `budget` chars: first shrink older observations, then, if one
/// recent observation alone is too big for the window (a large file read on a small local
/// model), cap that one too. The task message and the model's own turns are never cut.
fn compact(messages: &mut [Msg], budget: usize) {
    fn size(m: &Msg) -> usize {
        match m {
            Msg::User(t) => t.len(),
            Msg::Assistant { text, calls } => text.len() + calls.iter().map(|c| c.args.to_string().len()).sum::<usize>(),
            Msg::Tools(rs) => rs.iter().map(|r| r.content.len()).sum(),
        }
    }
    let shrink = |r: &mut ToolResult, keep: usize, note: &str| -> usize {
        if r.content.len() <= keep + 200 {
            return 0;
        }
        let before = r.content.len();
        r.content = format!("{}\n…({note})", util::take_chars(&r.content, keep));
        before.saturating_sub(r.content.len())
    };
    let mut total: usize = messages.iter().map(size).sum();
    let n = messages.len();
    for m in messages.iter_mut().take(n.saturating_sub(6)) {
        if total <= budget {
            return;
        }
        if let Msg::Tools(rs) = m {
            for r in rs.iter_mut() {
                total -= shrink(r, 1500, "older observation elided to save context; re-run the tool if you need it");
            }
        }
    }
    if total <= budget {
        return;
    }
    let cap = (budget / 3).max(2000);
    for m in messages.iter_mut().rev() {
        if total <= budget {
            break;
        }
        if let Msg::Tools(rs) = m {
            for r in rs.iter_mut() {
                total -= shrink(r, cap, "observation truncated to fit the model's context window; page through it with read_file offset/limit");
            }
        }
    }
}

/// Run the loop until the model finishes, the step budget runs out, or the wall clock does.
/// A role's model chain: `"a/x, b/y"` -> the resolvable entries, in order.
pub fn resolve_chain(model_id: &str) -> Result<Vec<(connections::Connection, String, String)>> {
    let mut chain = vec![];
    let mut errs = vec![];
    for id in model_id.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        match connections::resolve(id) {
            Ok((c, m)) => chain.push((c, m, id.to_string())),
            Err(e) => errs.push(format!("{e}")),
        }
    }
    if chain.is_empty() {
        bail!("no usable model in {model_id:?}: {}", errs.join("; "));
    }
    Ok(chain)
}

/// The first model in the chain that is usable now: its breaker is not open and this session
/// has not just seen it fail. If none is, the one usable soonest, and how long until then.
fn pick(chain: &[(connections::Connection, String, String)], avoid: &[Option<Instant>]) -> (usize, Duration) {
    let now = Instant::now();
    let wait_of = |i: usize| -> Duration {
        let b = health::reopens_in(&chain[i].0.name);
        let a = avoid[i].map(|t| t.saturating_duration_since(now)).unwrap_or_default();
        b.max(a)
    };
    if let Some(i) = (0..chain.len()).find(|i| wait_of(*i).is_zero()) {
        return (i, Duration::ZERO);
    }
    (0..chain.len()).map(|i| (i, wait_of(i))).min_by_key(|(_, d)| *d).unwrap_or((0, Duration::ZERO))
}

/// How long a session avoids a model after one of its calls failed.
const AVOID_AFTER_FAIL: Duration = Duration::from_secs(20);

/// Short stable fingerprint of a tool call's observation, for loop detection.
fn obs_key(s: &str) -> String {
    use sha2::Digest as _;
    let h = sha2::Sha256::digest(s.as_bytes());
    hex::encode(&h[..8])
}

/// Run the loop until the model finishes, the step budget runs out, or the wall clock does.
/// Self-healing: model failover along the chain, waiting out a full outage, loop breaking,
/// error-streak and deadline rescue.
pub fn run(cfg: &AgentConfig, task: &str) -> Result<AgentOutcome> {
    let chain = resolve_chain(&cfg.model_id)?;
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
    let mut stop_blocks = 0;
    let mut nudges = 0;
    let mut outage_retries = 0u32;
    let mut last_idx: Option<usize> = None;
    let mut avoid: Vec<Option<Instant>> = vec![None; chain.len()];
    let mut last_err = String::new();
    let mut seen: HashMap<String, (u32, String)> = HashMap::new();
    let mut warned: HashSet<String> = HashSet::new();
    let mut err_streak = 0u32;
    let mut rescued = false;
    let event = |status: &str, extra: Value| crate::log::event("step", status, extra);
    let heal = |out: &mut AgentOutcome, traj: &mut Traj, kind: &str, detail: Value| {
        out.heals += 1;
        traj.ev(json!({"type": "heal", "kind": kind, "detail": detail}));
        crate::log::event("heal", kind, detail);
    };
    loop {
        if crate::log::cancelled() {
            out.error = "cancelled by user".into();
            traj.ev(json!({"type": "error", "message": out.error}));
            break;
        }
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
        let (ci, wait) = pick(&chain, &avoid);
        if !wait.is_zero() {
            // Every model in the chain is down: wait for the first to half-open, rather than fail.
            let w = wait.min(left).min(Duration::from_secs(300));
            heal(&mut out, &mut traj, "wait_recovery", json!({"seconds": w.as_secs(), "model": chain[ci].2, "error": util::take_chars(&last_err, 200)}));
            say!("{}! every model is unavailable; waiting {}s for {} to recover", cfg.log_prefix, w.as_secs(), chain[ci].2);
            let until = Instant::now() + w;
            while Instant::now() < until && !crate::log::cancelled() {
                std::thread::sleep(Duration::from_millis(500));
            }
            continue;
        }
        let (conn, model, model_id) = (&chain[ci].0, &chain[ci].1, &chain[ci].2);
        if let Some(prev) = last_idx.filter(|p| *p != ci) {
            // Reported once per switch: down the chain is a failover, back up is a recovery.
            let kind = if ci < prev { "recovered" } else { "failover" };
            let why = if kind == "failover" { util::take_chars(&last_err, 200) } else { String::new() };
            heal(&mut out, &mut traj, kind, json!({"from": chain[prev].2, "to": model_id, "error": why}));
            say!("{}! {} from {} to {model_id}", cfg.log_prefix, if kind == "recovered" { "recovered: back" } else { "failing over" }, chain[prev].2);
        }
        last_idx = Some(ci);
        let max_tokens = conn.max_tokens.unwrap_or(cfg.max_tokens).min(conn.window() / 2).max(256);
        // Chars of conversation the window can hold (~3 chars/token), minus system and reply room.
        let budget = ((conn.window().saturating_sub(max_tokens) as usize) * 3).saturating_sub(cfg.system.len()).clamp(8_000, 600_000);
        out.steps += 1;
        compact(&mut messages, budget);
        let resp = match client::chat(conn, model, &ChatReq { system: &cfg.system, messages: &messages, tools: &tools, max_tokens }, left.max(Duration::from_secs(30))) {
            Ok(r) => {
                out.served_by = model_id.clone();
                avoid[ci] = None;
                last_err.clear();
                r
            }
            Err(e) => {
                out.steps -= 1;
                outage_retries += 1;
                last_err = format!("{e}");
                traj.ev(json!({"type": "error", "message": format!("{model_id}: {e}")}));
                // Avoid this model briefly; the next pass picks another, or waits for one.
                avoid[ci] = Some(Instant::now() + AVOID_AFTER_FAIL);
                if outage_retries <= 12 && !crate::log::cancelled() {
                    continue;
                }
                out.error = last_err.clone();
                break;
            }
        };
        out.input_tokens += resp.input_tokens;
        out.output_tokens += resp.output_tokens;
        out.cost_usd += (resp.input_tokens as f64 * conn.input_price + resp.output_tokens as f64 * conn.output_price) / 1e6;
        let _ = &model;
        if !resp.text.trim().is_empty() {
            traj.ev(json!({"type": "assistant", "text": resp.text}));
        }
        traj.ev(json!({"type": "usage", "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens}));
        event("turn", json!({
            "step": out.steps, "depth": cfg.depth, "in": out.input_tokens, "out": out.output_tokens,
            "tools": resp.calls.iter().map(|c| format!("{}({})", c.name, summarize_args(&c.args))).collect::<Vec<_>>(),
            "say": util::take_chars(resp.text.trim(), 160),
        }));
        if resp.calls.is_empty() && resp.text.trim().is_empty() && nudges < 2 && out.steps < cfg.max_steps {
            nudges += 1;
            messages.push(Msg::Assistant { text: String::new(), calls: vec![] });
            messages.push(Msg::User("Your last reply was empty. Continue: call a tool, or give your final answer.".into()));
            continue;
        }
        if resp.calls.is_empty() && looks_like_broken_call(&resp.text) && nudges < 3 && out.steps < cfg.max_steps {
            nudges += 1;
            traj.ev(json!({"type": "error", "message": "unparseable tool call in reply; asked the model to retry"}));
            messages.push(Msg::Assistant { text: resp.text.clone(), calls: vec![] });
            messages.push(Msg::User(format!(
                "Your last reply looks like a tool call, but it is not valid JSON{}, so nothing was executed. Call the tool again with a valid JSON object of arguments. For long file contents, write the file in smaller parts (write_file, then edit_file).",
                if resp.stop == "length" || resp.stop == "max_tokens" { " (it was cut off at the output-token limit)" } else { "" }
            )));
            continue;
        }
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
            if let Some(g) = &cfg.gate {
                let wired = if crate::core::tree::BUILTIN_TOOLS.contains(&call.name.as_str()) { None } else { find_wired(cfg, &call.name).map(|w| w.command) };
                if let Some(v) = g.check(&call.name, &call.args, wired.as_deref()) {
                    let enforce = g.mode == super::gate::Mode::Enforce;
                    let detail = json!({"tool": call.name, "call": format!("{}({})", call.name, summarize_args(&call.args)), "rule": v.rule, "reason": v.reason, "mode": g.mode.as_str()});
                    traj.ev(json!({"type": "gate", "status": if enforce { "blocked" } else { "would_block" }, "detail": detail}));
                    crate::log::event("gate", if enforce { "blocked" } else { "would_block" }, detail);
                    say!("{}⛔ gate {} {}: {}", cfg.log_prefix, if enforce { "blocked" } else { "would block" }, v.rule, v.reason);
                    if enforce {
                        results.push(ToolResult {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            content: format!("[gate] Blocked by the operator's guardrail gate (rule `{}`): {}. This rule cannot be changed from inside the run; reach the goal another way.", v.rule, v.reason),
                            is_error: true,
                        });
                        continue;
                    }
                }
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
            // Self-check 1: the same call returning the same thing again and again.
            let sig = format!("{}:{}", call.name, call.args);
            let key = obs_key(&content);
            let entry = seen.entry(sig.clone()).or_insert((0, key.clone()));
            if entry.1 == key {
                entry.0 += 1;
            } else {
                *entry = (1, key);
            }
            if entry.0 >= 3 && call.name != "finish" && warned.insert(sig.clone()) {
                content.push_str("\n\n[self-check] You have now made this exact call 3 times and got the same result each time. Repeating it will not change the outcome. Change approach: look at what actually exists (list_files, read_file), fix the command or path, or write the deliverable directly.");
                heal(&mut out, &mut traj, "loop_break", json!({"call": util::take_chars(&sig, 160)}));
            }
            // Self-check 2: a streak of failing calls.
            err_streak = if is_error { err_streak + 1 } else { 0 };
            if err_streak == 4 {
                content.push_str("\n\n[self-check] Your last 4 tool calls all failed. Stop and re-read the error messages above. Check which files exist with list_files, use the exact paths shown, and pass every required argument as valid JSON.");
                heal(&mut out, &mut traj, "error_streak", json!({"streak": err_streak}));
            }
            traj.ev(json!({"type": "tool", "name": call.name, "input": call.args, "status": if is_error { "error" } else { "ok" }, "output": content}));
            if call.name == "finish" && !is_error {
                finish = Some(content.clone());
            }
            results.push(ToolResult { id: call.id.clone(), name: call.name.clone(), content, is_error });
        }
        // Self-check 3: deadline rescue -- budget nearly gone and nothing delivered yet.
        let steps_left = cfg.max_steps.saturating_sub(out.steps);
        let time_left = cfg.wall.checked_sub(t0.elapsed()).unwrap_or_default();
        let near = steps_left <= (cfg.max_steps / 6).max(2) || time_left.as_secs_f64() < cfg.wall.as_secs_f64() * 0.15;
        if near && !rescued && finish.is_none() && !cfg.rescue_hint.is_empty() {
            let empty = cfg.rescue_dir.as_ref().map(|d| util::files_sorted(d).iter().all(|p| std::fs::metadata(p).map(|m| m.len() == 0).unwrap_or(true))).unwrap_or(true);
            if empty {
                rescued = true;
                if let Some(last) = results.last_mut() {
                    last.content.push_str(&format!("\n\n[self-check] Only {steps_left} step(s) and {}s remain. {}", time_left.as_secs(), cfg.rescue_hint));
                }
                heal(&mut out, &mut traj, "deadline_rescue", json!({"steps_left": steps_left, "seconds_left": time_left.as_secs()}));
            }
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
    event("end", json!({"steps": out.steps, "in": out.input_tokens, "out": out.output_tokens, "usd": out.cost_usd, "error": out.error, "depth": cfg.depth, "heals": out.heals, "model": out.served_by}));
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
    fn compact_fits_window() {
        let big = "x".repeat(50_000);
        let mut m = vec![Msg::User("task".into())];
        for i in 0..8 {
            m.push(Msg::Assistant { text: String::new(), calls: vec![] });
            m.push(Msg::Tools(vec![ToolResult { id: format!("{i}"), name: "read_file".into(), content: big.clone(), is_error: false }]));
        }
        compact(&mut m, 30_000);
        let total: usize = m.iter().map(|x| match x { Msg::Tools(r) => r[0].content.len(), _ => 4 }).sum();
        assert!(total < 40_000, "{total}");
        assert!(matches!(&m[0], Msg::User(t) if t == "task"));
    }

    #[test]
    fn helpful_errors() {
        let d = tempfile::tempdir().unwrap();
        util::write(&d.path().join("regions.csv"), "a").unwrap();
        let cfg = AgentConfig::new("mock/m", "", d.path());
        let call = |name: &str, args: Value| exec_tool(&cfg, &ToolCall { id: "1".into(), name: name.into(), args, extra: None });
        let e = call("read_file", json!({"path": "input/regions.csv"})).unwrap_err().to_string();
        assert!(e.contains("Did you mean `regions.csv`?"), "{e}");
        assert!(!e.contains(&d.path().display().to_string()), "{e}");
        let e = call("write_file", json!({})).unwrap_err().to_string();
        assert!(e.contains("write_file needs `path`, `content`"), "{e}");
    }

    #[test]
    fn broken_calls_detected() {
        assert!(looks_like_broken_call("<tool_call> {\"name\": \"write_file\", \"arguments\": \"{bad"));
        assert!(!looks_like_broken_call("The answer is 25%."));
    }

    #[test]
    fn frontmatter() {
        assert_eq!(strip_frontmatter("---\nname: x\n---\nbody"), "body");
        assert_eq!(strip_frontmatter("plain"), "plain");
    }
}
