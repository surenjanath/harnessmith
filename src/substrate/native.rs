//! The native substrate ("nx"): the harness runs on this crate's own agent loop, against any
//! connected model, with no external agent CLI.
//!
//! The surface mirrors OpenCode's four always-on files plus skills, sub-agents and scripts;
//! the in-loop lever is `hooks.json` (PreToolUse / PostToolUse / Stop), executed by the loop.
//!
//! ```text
//!     <run_dir>/workspace/            the agent's working directory
//!     <run_dir>/workspace/<inputs>    the task's source material, copied in
//!     <run_dir>/workspace/scripts/    the harness's scripts, runnable as python3 scripts/x.py
//!     <run_dir>/workspace/.harness/   the full harness (skills, sub-agents, hooks)
//!     <run_dir>/workspace/output/     deliverables
//!     <run_dir>/trajectory.jsonl      every event
//! ```

use super::{cap_rendered, clip, pystr};
use crate::core::runmark;
use crate::core::tree::{self, SurfaceSpec};
use crate::llm::agent::{self, AgentConfig};
use crate::util;
use anyhow::Result;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub fn native_surface() -> SurfaceSpec {
    SurfaceSpec {
        name: "native",
        always_on: vec!["systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md"],
        always_on_max: 24_000,
        always_on_cap_reason: "every token here competes with the agent's own output budget",
        skill_dir: "skills",
        skill_name_rule: "must_match",
        trigger_hint: "Fix it by adding a trigger line naming `{name}` to systemprompt.md or guardrails.md, e.g. \"When <the situation this skill handles>, load the `{name}` skill.\"",
        autotrigger_file: "guardrails.md",
        trigger_line: "- When {desc}, load the `{name}` skill.\n",
        subagent_requires_mode: true,
        plugin_dir: None,
        hooks_file: Some("hooks.json"),
        hook_events: vec!["PreToolUse", "PostToolUse", "Stop"],
        compile_skill_scripts: false,
        reject_empty_always_on: false,
        stubs: vec![
            ("scripts/README.md", "<!-- c7: deterministic helpers, run with the `bash` tool as `python3 scripts/<name>.py`. Add one by emitting scripts/<name>.py. -->\n"),
            ("sub_agents/README.md", "<!-- c5: delegated sub-agents, invoked with the `task` tool in their OWN context. Add one as sub_agents/<name>/AGENT.md with frontmatter `mode: subagent` and `description:`. -->\n"),
            ("skills/README.md", "<!-- c6: reusable workflows loaded on demand via the `skill` tool. Add one as skills/<name>/SKILL.md with frontmatter `name:` matching the directory and a `description:`. -->\n"),
        ],
    }
}

pub const NAME: &str = "nx";

/// The default base harness for a new goal project: short, general, and editable.
pub fn default_base_tree() -> tree::Tree {
    let mut t = tree::Tree::new();
    t.insert(
        "systemprompt.md".into(),
        "You are a careful, capable autonomous agent. You are given a task and a workspace. Read the task and any input files before acting, plan briefly, then do the work with your tools.\n\nWrite every deliverable the task asks for as a file under `output/` (use `output/answer.md` for a written answer). Check what you wrote by reading it back before you finish. When you are done, call `finish` with a short summary.\n".into(),
    );
    t.insert(
        "guardrails.md".into(),
        "- Never invent facts, figures, citations or file contents; if the inputs do not establish something, say so.\n- Deliver exactly what was asked, in the requested format and shape.\n- Do not stop before every requested deliverable exists under `output/`.\n".into(),
    );
    t.insert("LongTermMEMORY.md".into(), String::new());
    t.insert(
        "tool_descriptions/tool_guidance.md".into(),
        "- `list_files` / `read_file` to inspect inputs (page large files with offset/limit).\n- `write_file` / `edit_file` to produce deliverables under `output/`.\n- `bash` runs shell commands in the workspace, e.g. `python3 scripts/<name>.py`.\n- `skill` loads a named skill's full instructions; `task` delegates to a named sub-agent.\n".into(),
    );
    t
}

const RUNTIME_RULES: &str = "# RUNTIME\nYour working directory is the workspace. Input files for the task, if any, are in it. Write deliverables under `output/`. Use relative paths.";

/// The system prompt the solver sees: the always-on harness plus an index of on-demand parts.
pub fn compose_system(harness: &Path) -> String {
    let surface = native_surface();
    let mut parts: Vec<String> = surface
        .always_on
        .iter()
        .filter_map(|f| util::read_lossy(&harness.join(f)).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty()))
        .collect();
    let idx = super::opencode::component_index(harness, "skills", "SKILL.md");
    if !idx.is_empty() {
        parts.push(format!("# AVAILABLE SKILLS\nLoad with the `skill` tool when the situation matches:\n{idx}"));
    }
    let idx = super::opencode::component_index(harness, "sub_agents", "AGENT.md");
    if !idx.is_empty() {
        parts.push(format!("# AVAILABLE SUBAGENTS\nDelegate with the `task` tool when the situation matches:\n{idx}"));
    }
    let mut scripts: Vec<String> = std::fs::read_dir(harness.join("scripts"))
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".py")).collect())
        .unwrap_or_default();
    scripts.sort();
    if !scripts.is_empty() {
        parts.push(format!("# HELPER SCRIPTS\nAvailable in `scripts/`: {}\nRun with `python3 scripts/<name>.py` via the `bash` tool.", scripts.join(", ")));
    }
    parts.push(RUNTIME_RULES.into());
    parts.join("\n\n---\n\n")
}

#[derive(Debug, Clone)]
pub struct SolveOpts {
    pub model_id: String,
    pub max_steps: u32,
    pub timeout: u64,
    pub allow_shell: bool,
    pub inputs_fingerprint: Option<String>,
}

/// Run one rollout of `harness` (None = `base`) on `task` into `run_dir`, idempotently:
/// a completed run with the same harness and inputs is reused.
pub fn solve(task: &str, source: Option<&Path>, harness: &Path, harness_is_base: bool, run_dir: &Path, o: &SolveOpts) -> Result<PathBuf> {
    let fp_dir = if harness_is_base { None } else { Some(harness) };
    if runmark::is_complete(run_dir, fp_dir, true, o.inputs_fingerprint.as_deref()) {
        crate::say!("    (reusing completed rollout {})", run_dir.display());
        return Ok(run_dir.to_path_buf());
    }
    util::rmtree(run_dir);
    let ws = run_dir.join("workspace");
    std::fs::create_dir_all(ws.join("output"))?;
    if let Some(src) = source.filter(|s| s.is_dir()) {
        util::copy_tree(src, &ws)?;
    }
    let hdir = ws.join(".harness");
    util::copy_tree_filtered(harness, &hdir, |r| !r.ends_with("README.md"))?;
    if harness.join("scripts").is_dir() {
        util::copy_tree_filtered(&harness.join("scripts"), &ws.join("scripts"), |r| r.ends_with(".py"))?;
    }
    let mut cfg = AgentConfig::new(&o.model_id, &compose_system(harness), &ws);
    cfg.max_steps = o.max_steps;
    cfg.wall = Duration::from_secs(o.timeout);
    cfg.allow_shell = o.allow_shell;
    cfg.harness = Some(hdir);
    cfg.trajectory = Some(run_dir.join("trajectory.jsonl"));
    cfg.log_prefix = "      ".into();
    let out = agent::run(&cfg, task)?;
    util::write(&run_dir.join("final.md"), &out.final_text)?;
    // A session that died (transport error, cancellation) before producing anything is a dead
    // rollout: report it and leave it UNMARKED, so a rerun rolls it again instead of serving an
    // empty attempt as evidence.
    let died = !out.error.is_empty() && !out.error.contains("budget") && out.final_text.trim().is_empty() && deliverables(run_dir).is_empty();
    if died || out.error.contains("cancelled") {
        anyhow::bail!("solver session failed: {}", out.error);
    }
    if deliverables(run_dir).is_empty() && !out.final_text.trim().is_empty() {
        util::write(&ws.join("output/answer.md"), &out.final_text)?;
    }
    let mut extra = serde_json::Map::new();
    extra.insert("model".into(), Value::String(o.model_id.clone()));
    extra.insert("steps".into(), Value::from(out.steps));
    extra.insert("cost_usd".into(), Value::from(out.cost_usd));
    extra.insert("error".into(), Value::String(out.error.clone()));
    util::write(
        &run_dir.join("usage.json"),
        serde_json::to_string_pretty(&serde_json::json!({"steps": out.steps, "input_tokens": out.input_tokens, "output_tokens": out.output_tokens, "cost_usd": out.cost_usd, "error": out.error}))?,
    )?;
    runmark::mark_complete(run_dir, &run_dir.to_string_lossy(), fp_dir, o.inputs_fingerprint.as_deref(), Some(extra))?;
    Ok(run_dir.to_path_buf())
}

/// The deliverable files of a native run (non-empty files under workspace/output).
pub fn deliverables(run_dir: &Path) -> Vec<PathBuf> {
    let out = run_dir.join("workspace/output");
    util::files_sorted(&out).into_iter().filter(|p| std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false)).collect()
}

/// Render the native JSONL trajectory into the shared grammar.
pub fn render_events(path: &Path, obs_trunc: usize, max_chars: usize, args_trunc: usize) -> String {
    let raw = match std::fs::read(path) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => return format!("(no trajectory: {e})"),
    };
    let mut lines = vec![];
    for ln in raw.lines() {
        let Ok(ev) = serde_json::from_str::<Value>(ln.trim()) else { continue };
        match ev.get("type").and_then(|t| t.as_str()).unwrap_or("") {
            "assistant" => lines.push(format!("[ASSISTANT] {}", clip(&pystr(ev.get("text")), obs_trunc))),
            "tool" => {
                let input = ev.get("input").map(|v| v.to_string()).unwrap_or_default();
                lines.push(format!("[ACTION] {}({})", pystr(ev.get("name")), clip(&input, args_trunc)));
                let tag = if ev.get("status").and_then(|s| s.as_str()) == Some("error") { "OBSERVATION ERROR" } else { "OBSERVATION" };
                lines.push(format!("[{tag}] {}", clip(&pystr(ev.get("output")), obs_trunc)));
            }
            "hook" => lines.push(format!("[HARNESS] {} hook: {}", pystr(ev.get("event")), clip(&pystr(ev.get("reason")), obs_trunc))),
            "error" => lines.push(format!("[ERROR] {}", clip(&pystr(ev.get("message")), 1000))),
            "finish" => {
                let steps = ev.get("steps").and_then(|s| s.as_u64()).unwrap_or(0);
                let sub = ev.get("error").and_then(|e| e.as_str()).filter(|e| !e.is_empty()).map(|e| format!(" ({e})")).unwrap_or_default();
                lines.push(format!("[HARNESS] session ended after {steps} step(s){sub}"));
            }
            _ => {}
        }
    }
    cap_rendered(&lines, max_chars)
}
