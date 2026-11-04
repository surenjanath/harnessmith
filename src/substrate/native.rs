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
        tool_dir: Some("tools"),
        compile_skill_scripts: false,
        reject_empty_always_on: false,
        stubs: vec![
            ("scripts/README.md", "<!-- c7: deterministic helpers, run with the `bash` tool as `python3 scripts/<name>.py`. Add one by emitting scripts/<name>.py. -->\n"),
            ("sub_agents/README.md", "<!-- c5: delegated sub-agents, invoked with the `task` tool in their OWN context. Add one as sub_agents/<name>/AGENT.md with frontmatter `mode: subagent` and `description:`. -->\n"),
            ("tools/README.md", "<!-- c9: wired tools the agent calls directly, like read_file. Add one as tools/<name>.json: {\"name\": \"<name>\", \"description\": \"when to call it\", \"parameters\": {\"type\": \"object\", \"properties\": {...}, \"required\": [...]}, \"command\": \"python3 scripts/<name>.py\", \"timeout_s\": 120}. The command runs with bash in the workspace; the call's arguments arrive as JSON on stdin and as $ARG_<NAME> variables; stdout is the result and a non-zero exit is an error. Wire a tool when the agent keeps re-deriving the same step. -->\n"),
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
        "tools/csv_profile.json".into(),
        "{\n  \"name\": \"csv_profile\",\n  \"description\": \"Profile a CSV file before analysing it: column names, inferred types, row count, nulls, and min/max/mean/sum for numeric columns. Call it first on any CSV input instead of reading the raw rows.\",\n  \"parameters\": {\n    \"type\": \"object\",\n    \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"CSV path relative to the workspace\"}},\n    \"required\": [\"path\"]\n  },\n  \"command\": \"python3 scripts/csv_profile.py\",\n  \"timeout_s\": 60\n}\n".into(),
    );
    t.insert("scripts/csv_profile.py".into(), CSV_PROFILE.into());
    t.insert(
        "tool_descriptions/tool_guidance.md".into(),
        "- `list_files` / `read_file` to inspect inputs (page large files with offset/limit).\n- `write_file` / `edit_file` to produce deliverables under `output/`.\n- `bash` runs shell commands in the workspace, e.g. `python3 scripts/<name>.py`.\n- `skill` loads a named skill's full instructions; `task` delegates to a named sub-agent.\n- Wired tools (listed under WIRED TOOLS) are called directly; prefer them over hand-rolled shell steps.\n".into(),
    );
    t
}

/// The starter wired tool's script: arguments as JSON on stdin, a readable profile on stdout.
const CSV_PROFILE: &str = r#"import csv, json, sys

args = json.load(sys.stdin)
path = args["path"]
with open(path, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print(f"{path}: no data rows")
    sys.exit(0)
cols = list(rows[0].keys())
print(f"{path}: {len(rows)} rows x {len(cols)} columns")
for c in cols:
    vals = [(r.get(c) or "").strip() for r in rows]
    present = [v for v in vals if v != ""]
    nulls = len(vals) - len(present)
    nums = []
    for v in present:
        try:
            nums.append(float(v.replace(",", "")))
        except ValueError:
            break
    if present and len(nums) == len(present):
        mean = sum(nums) / len(nums)
        print(f"- {c}: number · min {min(nums):g} · max {max(nums):g} · mean {mean:.4g} · sum {sum(nums):g} · nulls {nulls}")
    else:
        uniq = sorted(set(present))
        sample = ", ".join(uniq[:5]) + (" ..." if len(uniq) > 5 else "")
        print(f"- {c}: text · {len(uniq)} distinct · nulls {nulls} · e.g. {sample}")
"#;

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
    let tools = super::super::llm::agent::wired_tools(harness);
    if !tools.is_empty() {
        let lines: Vec<String> = tools.iter().map(|t| format!("- `{}`: {}", t.def.name, t.def.description)).collect();
        parts.push(format!("# WIRED TOOLS\nCall these directly, like any other tool, whenever they fit:\n{}", lines.join("\n")));
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
    pub gate_mode: String,
    pub gate_rules: Vec<String>,
    pub gate_protect_inputs: bool,
    /// Project root holding memory.json, and the goal id notes are filed under (None: no memory).
    pub memory: Option<(PathBuf, String)>,
    pub memory_embed: String,
    pub memory_shared: bool,
    /// "auto" | "bwrap" | "unshare" | "off", and whether sandboxed commands get network access.
    pub sandbox: String,
    pub sandbox_network: bool,
    /// MCP servers whose tools the solver may call.
    pub mcp_servers: Vec<crate::llm::mcp::ServerCfg>,
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
    cfg.rescue_dir = Some(ws.join("output"));
    cfg.rescue_hint = "output/ is still empty. Write your best complete deliverable there now with write_file (a partial answer beats none), then call finish.".into();
    let mut gate = crate::llm::gate::Gate::new(&o.gate_mode, &o.gate_rules);
    if o.gate_protect_inputs {
        gate.protect_inputs(&ws);
    }
    let ro = if o.gate_protect_inputs { gate.protected_list().into_iter().filter(|p| !p.starts_with("scripts/")).collect() } else { vec![] };
    let sb = crate::llm::sandbox::Sandbox::new(&o.sandbox, &ws, &ro, o.sandbox_network);
    if sb.kind != crate::llm::sandbox::Kind::Off {
        crate::say!("      sandbox: {} · network {}", sb.kind.as_str(), if o.sandbox_network { "on" } else { "off" });
        cfg.sandbox = Some(sb);
    }
    cfg.gate = Some(gate);
    if o.mcp_servers.iter().any(|s| s.enabled) {
        let hub = crate::llm::mcp::Hub::start(&o.mcp_servers);
        for (n, e) in &hub.failed {
            crate::say!("      ! MCP server {n} did not start: {}", util::take_chars(e, 200));
        }
        if !hub.clients.is_empty() {
            crate::say!("      MCP: {} server(s), {} tool(s)", hub.clients.len(), hub.tools().len());
            cfg.mcp = Some(std::sync::Arc::new(hub));
        }
    }
    if let Some((root, goal)) = &o.memory {
        let store = crate::llm::memory::Store::open(root).with(Some(o.memory_embed.clone()), o.memory_shared);
        cfg.system = format!("{}\n\n---\n\n{}", cfg.system, crate::llm::memory::system_section(&store.pinned()));
        cfg.memory = Some((store, crate::llm::memory::Source { goal: goal.clone(), by: "agent".into(), ..Default::default() }));
    }
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
        serde_json::to_string_pretty(&serde_json::json!({"steps": out.steps, "input_tokens": out.input_tokens, "output_tokens": out.output_tokens, "cost_usd": out.cost_usd, "error": out.error, "heals": out.heals, "model": out.served_by}))?,
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

#[cfg(test)]
mod wired_tool_tests {
    use super::*;

    #[test]
    fn default_base_validates_with_its_wired_tool() {
        let t = default_base_tree();
        let census = tree::validate(&native_surface(), &t, true, None, &tree::reserved_default()).unwrap();
        assert_eq!(census["tools"], 1);
        assert!(tree::is_allowed(&native_surface(), "tools/csv_profile.json"));
        assert!(!tree::is_allowed(&super::super::opencode::opencode_surface(), "tools/csv_profile.json"));
    }

    #[test]
    fn bad_wired_tools_are_rejected_with_reasons() {
        let mut t = default_base_tree();
        t.insert("tools/bash.json".into(), r#"{"name":"bash","description":"x","command":"true"}"#.into());
        t.insert("tools/nodesc.json".into(), r#"{"name":"other","command":"python3 scripts/missing.py"}"#.into());
        t.insert("tools/broken.json".into(), "{not json".into());
        let e = tree::validate(&native_surface(), &t, true, None, &tree::reserved_default()).unwrap_err().to_string();
        assert!(e.contains("built-in tool name"), "{e}");
        assert!(e.contains("`name` must be \"nodesc\""), "{e}");
        assert!(e.contains("non-empty `description`"), "{e}");
        assert!(e.contains("scripts/missing.py"), "{e}");
        assert!(e.contains("not valid JSON"), "{e}");
    }
}
