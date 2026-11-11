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
use crate::llm::experience;
use crate::llm::precise;
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
        "tool_descriptions/tool_guidance.md".into(),
        "- `list_files` / `read_file` to inspect inputs (page large files with offset/limit).\n- `write_file` / `edit_file` to produce deliverables under `output/`.\n- `bash` runs shell commands in the workspace.\n- `skill` loads a named skill's full instructions; `task` delegates to a named sub-agent.\n- Wired tools (listed under WIRED TOOLS) are called directly; prefer them over hand-rolled shell steps.\n".into(),
    );
    t
}

/// What the machine can run, probed once per process, so the solver does not spend its steps
/// discovering that pandoc or reportlab is missing.
pub fn environment_note() -> String {
    static NOTE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NOTE.get_or_init(|| {
        let bins = ["python3", "node", "pandoc", "pdflatex", "xelatex", "wkhtmltopdf", "soffice", "chromium", "google-chrome"];
        let have = |b: &str| std::process::Command::new("bash").arg("-c").arg(format!("command -v {b}")).output().map(|o| o.status.success()).unwrap_or(false);
        let found: Vec<&str> = bins.iter().copied().filter(|b| have(b)).collect();
        let missing: Vec<&str> = bins.iter().copied().filter(|b| !found.contains(b)).collect();
        let mods = ["reportlab", "fpdf", "weasyprint", "markdown", "docx", "openpyxl", "pandas", "numpy", "matplotlib"];
        let py = std::process::Command::new("python3")
            .arg("-c")
            .arg(format!("import importlib.util as u;print(' '.join(m for m in {mods:?} if u.find_spec(m)))"))
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default();
        format!(
            "\n\n---\n\n# ENVIRONMENT\nCommands available: {}. Not installed: {}.\nPython packages available: {}. Do not try to install packages: assume only what is listed here.\nTo make a PDF, write the content as Markdown or HTML, then call export_pdf on it. If the goal asks for another format nothing here can produce, write the content in the closest format you can produce properly (HTML or Markdown) and say so plainly in output/answer.md. Never write a binary format such as .pdf or .docx as plain text.",
            if found.is_empty() { "(none)".into() } else { found.join(", ") },
            if missing.is_empty() { "(nothing checked is missing)".into() } else { missing.join(", ") },
            if py.is_empty() { "standard library only".into() } else { py },
        )
    })
    .clone()
}

/// Optional starter wired tools, offered in the UI. Not in the default base: a data-only
/// tool in every harness pulls small models toward CSV work on unrelated goals.
pub fn starter_tools_tree() -> tree::Tree {
    let mut t = tree::Tree::new();
    t.insert(
        "tools/csv_profile.json".into(),
        "{\n  \"name\": \"csv_profile\",\n  \"description\": \"Profile a CSV file before analysing it: column names, inferred types, row count, nulls, and min/max/mean/sum for numeric columns. Call it first on any CSV input instead of reading the raw rows.\",\n  \"parameters\": {\n    \"type\": \"object\",\n    \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"CSV path relative to the workspace\"}},\n    \"required\": [\"path\"]\n  },\n  \"command\": \"python3 scripts/csv_profile.py\",\n  \"timeout_s\": 60\n}\n".into(),
    );
    t.insert("scripts/csv_profile.py".into(), CSV_PROFILE.into());
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

const RUNTIME_RULES: &str = "# RUNTIME\nYour working directory is the workspace. Input files for the task, if any, are in it. Write deliverables under `output/`. Use relative paths.\n\n# HOW TO WORK\n1. Read the goal and list, for yourself, every requirement in it (format, file type, length, style, sections).\n2. Create each deliverable with `write_file`, in full, in one go (not with shell echo or heredocs, which truncate). Use the file type the goal asks for: an HTML page is a complete `output/index.html` with <!DOCTYPE html>, <head>, inline <style> and </html>; a document is Markdown with headings and real depth.\n3. Read every file back and check it against each requirement; fix what is missing.\n4. Only then call `finish` with a short summary. Describing work in a reply does not do it: only tool calls change files.";

/// Added to the solver's instructions when the accuracy boost is on.
const BOOST_RULES: &str = "\n\n---\n\n# NUMBERS\nEvery figure you report must come from the input files or from a tool result in this session. For any calculation, count, sort or date difference, call the `python` tool (or `calc` for plain arithmetic) and copy its result; never work a number out in your head and never guess one. Before you finish, the figures in your deliverable are checked against what you actually read and computed.";

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
    pub web_access: bool,
    /// Accuracy boost: "auto" (on for local and small models) | "on" | "off".
    pub accuracy: String,
    /// Independent attempts the answer engine votes over.
    pub accuracy_samples: usize,
    /// A stronger model asked once when the attempts do not agree (empty: none).
    pub escalate_model: String,
    /// Project root whose experience (verified worked examples) is read and added to, with the
    /// goal id new examples are filed under. None: no learning from experience.
    pub experience: Option<(PathBuf, String)>,
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
    let boost = precise::enabled(&o.accuracy, &o.model_id);
    let mut cfg = AgentConfig::new(&o.model_id, &compose_system(harness), &ws);
    cfg.boost = boost;
    if boost {
        cfg.system.push_str(BOOST_RULES);
    }
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
    // Accuracy boost: a goal that asks for one exact answer goes to the answer engine, which has
    // the model write a program instead of guessing and votes over independent attempts.
    let mut engine_usage = (0u32, 0u64, 0u64, 0.0f64);
    if o.allow_shell && precise::enabled(&o.accuracy, &o.model_id) && precise::is_exact_question(task) && crate::llm::pycode::available() {
        let files: Vec<String> = source.filter(|s| s.is_dir()).map(|s| util::files_sorted(s).iter().map(|p| util::rel(s, p)).collect()).unwrap_or_default();
        let mut po = precise::Opts::new(&o.model_id);
        po.samples = o.accuracy_samples.clamp(1, 9);
        po.notes = precise::notes_from_harness(harness);
        po.sandbox = (sb.kind != crate::llm::sandbox::Kind::Off).then(|| sb.clone());
        po.wall = Duration::from_secs(o.timeout);
        po.trajectory = Some(run_dir.join("trajectory.jsonl"));
        po.escalate_model = o.escalate_model.clone();
        po.gate = Some(gate.clone());
        // Experience: the most similar verified solutions of this project, as worked examples.
        let store = o.experience.as_ref().map(|(root, _)| experience::Store::open(root));
        let sigs: Vec<experience::FileSig> = files.iter().map(|f| experience::sig_of(&ws, f)).collect();
        if let Some(st) = &store {
            let mut ex = st.similar(task, &sigs, 2, 0.5);
            // A second example only when it is nearly as close as the first.
            if ex.len() == 2 && ex[1].0 < ex[0].0 - 0.15 {
                ex.truncate(1);
            }
            if !ex.is_empty() {
                crate::say!("      experience: {} worked example(s) from earlier goals ({})", ex.len(), ex.iter().map(|(s, e)| format!("{} {:.0}%", e.id, s * 100.0)).collect::<Vec<_>>().join(", "));
                crate::log::event("accuracy", "examples", serde_json::json!({"ids": ex.iter().map(|(_, e)| e.id.clone()).collect::<Vec<_>>(), "scores": ex.iter().map(|(s, _)| (s * 100.0).round() / 100.0).collect::<Vec<_>>()}));
                st.mark_used(&ex.iter().map(|(_, e)| e.id.clone()).collect::<Vec<_>>());
            }
            po.examples = ex;
        }
        crate::say!("      accuracy boost: exact question → program-aided answer, up to {} attempt(s){}", po.samples, if po.sandbox.is_some() { " (sandboxed)" } else { "" });
        match precise::answer(task, &ws, &files, &po) {
            Ok(v) if v.answer.is_some() => {
                // A verified solution becomes experience: two or more computed attempts agree
                // and none of them disagrees.
                let answer = v.answer.clone().unwrap_or_default();
                let computed: Vec<&precise::Sample> = v.samples.iter().filter(|s| s.answer.is_some() && s.how != "reasoned").collect();
                let unanimous = computed.len() >= 2 && computed.iter().all(|s| precise::same_answer(s.answer.as_deref().unwrap_or(""), &answer, task));
                if let (Some(st), Some((_, goal)), true, false) = (&store, &o.experience, unanimous, v.program.trim().is_empty()) {
                    let rec = st.record(experience::Exemplar {
                        goal: goal.clone(),
                        question: task.to_string(),
                        files: sigs.clone(),
                        program: v.program.clone(),
                        output: v.output.clone(),
                        answer: answer.clone(),
                        agree: v.agree,
                        answered: v.answered,
                        model: v.served_by.clone(),
                        status: "active".into(),
                        ..Default::default()
                    });
                    if let Ok((e, replaced)) = rec {
                        crate::say!("      experience: kept this solution as a worked example ({}{})", e.id, if replaced { ", replacing an older one of the same kind" } else { "" });
                        crate::log::event("accuracy", "learned", serde_json::json!({"id": e.id, "replaced": replaced}));
                    }
                }
                return finish_precise(run_dir, &ws, fp_dir, o, &v);
            }
            Ok(v) => {
                if crate::log::cancelled() {
                    anyhow::bail!("solver session failed: cancelled by user");
                }
                crate::say!("      accuracy boost: no attempt produced an answer ({}); continuing with the agent loop", util::take_chars(&v.error.replace('\n', " "), 160));
                engine_usage = (v.calls, v.input_tokens, v.output_tokens, v.cost_usd);
            }
            Err(e) => {
                if format!("{e}").contains("cancelled") {
                    anyhow::bail!("solver session failed: cancelled by user");
                }
                crate::say!("      accuracy boost unavailable ({}); continuing with the agent loop", util::take_chars(&format!("{e}"), 160));
            }
        }
    }
    // Accuracy boost: a single page or a Python coding task is built by the maker, which asks for
    // the file in one fenced block, checks it by running it (tests) or reading it (design
    // checklist), repairs it, and keeps the best of a few candidates.
    if boost {
        if let Some(k) = crate::llm::maker::kind(task) {
            let mut mo = crate::llm::maker::Opts::new(&o.model_id);
            mo.candidates = o.accuracy_samples.clamp(1, 4);
            mo.notes = precise::notes_from_harness(harness);
            mo.sandbox = (sb.kind != crate::llm::sandbox::Kind::Off).then(|| sb.clone());
            mo.wall = Duration::from_secs(o.timeout);
            mo.trajectory = Some(run_dir.join("trajectory.jsonl"));
            mo.gate = Some(gate.clone());
            crate::say!("      accuracy boost: {} → built from fenced blocks and checked, up to {} candidate(s)", if k == crate::llm::maker::Kind::Page { "a page" } else { "a coding task" }, mo.candidates);
            match crate::llm::maker::make(task, &ws, k, &mo) {
                Ok(m) if !m.file.is_empty() => return finish_made(run_dir, &ws, fp_dir, o, &m),
                Ok(m) => engine_usage = (m.calls, m.input_tokens, m.output_tokens, m.cost_usd),
                Err(e) => {
                    if format!("{e}").contains("cancelled") {
                        anyhow::bail!("solver session failed: cancelled by user");
                    }
                    crate::say!("      accuracy boost: the maker gave nothing ({}); continuing with the agent loop", util::take_chars(&format!("{e}"), 160));
                }
            }
        }
    }
    if sb.kind != crate::llm::sandbox::Kind::Off {
        crate::say!("      sandbox: {} · network {}", sb.kind.as_str(), if o.sandbox_network { "on" } else { "off" });
        if !o.sandbox_network {
            // With web access on, the web tools are the way out; say so instead of "no internet".
            cfg.system.push_str(if o.web_access {
                "\n\n---\n\n# NETWORK\nShell commands have no internet access: do not use wget, curl, pip install or git clone. To read the web, call the web_fetch tool (and web_search when it is listed)."
            } else {
                "\n\n---\n\n# NETWORK\nThis workspace has NO internet access. Do not try to download or install anything (wget, curl, pip install, git clone): it cannot work. Use the input files and your own knowledge, and say which facts a reader should verify."
            });
        }
        cfg.sandbox = Some(sb);
    }
    if cfg.allow_shell {
        cfg.system.push_str(&environment_note());
    }
    cfg.web = o.web_access;
    if o.web_access {
        cfg.system.push_str(&format!("\n\nWEB: you can read pages with web_fetch{}. Cite every outside fact with the URL you read it at; never cite a source you did not open.", if crate::llm::web::search_backend().is_some() { " and search with web_search" } else { "" }));
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
    // Accuracy boost: small inputs are shown with the task, so the first step is work, not a read.
    let shown_task = if boost {
        let files: Vec<String> = source.filter(|s| s.is_dir()).map(|s| util::files_sorted(s).iter().map(|p| util::rel(s, p)).collect()).unwrap_or_default();
        let block = precise::files_block(&ws, &files);
        if !block.is_empty() && util::char_len(&block) <= 6000 { format!("{task}{}", block.replace("Files in the current directory:", "The input files (already in your workspace):")) } else { task.to_string() }
    } else {
        task.to_string()
    };
    let mut out = agent::run(&cfg, &shown_task)?;
    out.steps += engine_usage.0;
    out.input_tokens += engine_usage.1;
    out.output_tokens += engine_usage.2;
    out.cost_usd += engine_usage.3;
    util::write(&run_dir.join("final.md"), &out.final_text)?;
    // A session that died (transport error, cancellation) before producing anything is a dead
    // rollout: report it and leave it UNMARKED, so a rerun rolls it again instead of serving an
    // empty attempt as evidence.
    let died = !out.error.is_empty() && !out.error.contains("budget") && out.final_text.trim().is_empty() && deliverables(run_dir).is_empty();
    if died || out.error.contains("cancelled") {
        anyhow::bail!("solver session failed: {}", out.error);
    }
    // The final reply becomes output/answer.md only when it is an answer, not an error report.
    if deliverables(run_dir).is_empty() && !out.final_text.trim().is_empty() && !crate::core::schedule::looks_failed(&out.final_text) {
        util::write(&ws.join("output/answer.md"), &out.final_text)?;
    }
    let mut extra = serde_json::Map::new();
    extra.insert("model".into(), Value::String(o.model_id.clone()));
    extra.insert("steps".into(), Value::from(out.steps));
    extra.insert("cost_usd".into(), Value::from(out.cost_usd));
    extra.insert("error".into(), Value::String(out.error.clone()));
    util::write(
        &run_dir.join("usage.json"),
        serde_json::to_string_pretty(&serde_json::json!({"steps": out.steps, "input_tokens": out.input_tokens, "output_tokens": out.output_tokens, "cost_usd": out.cost_usd, "error": out.error, "heals": out.heals, "model": out.served_by, "review_open": out.review_open}))?,
    )?;
    runmark::mark_complete(run_dir, &run_dir.to_string_lossy(), fp_dir, o.inputs_fingerprint.as_deref(), Some(extra))?;
    Ok(run_dir.to_path_buf())
}

/// Deliver what the maker built as the rollout's result.
fn finish_made(run_dir: &Path, ws: &Path, fp_dir: Option<&Path>, o: &SolveOpts, m: &crate::llm::maker::Made) -> Result<PathBuf> {
    let open = if m.open.is_empty() { String::new() } else { format!("\n\nStill open:\n{}", m.open.iter().map(|x| format!("- {x}")).collect::<Vec<_>>().join("\n")) };
    let note = format!("`{}`\n\n{}{open}", m.file, m.summary);
    if m.kind == "code" {
        util::write(&ws.join("output/answer.md"), &format!("{note}\n"))?;
    }
    util::write(&run_dir.join("final.md"), &note)?;
    util::write(&run_dir.join("accuracy.json"), serde_json::to_string_pretty(&m.to_json())?)?;
    crate::say!("      accuracy boost: wrote {} — {}/{} checks passed (confidence {})", m.file, m.passed, m.total, m.confidence());
    let mut extra = serde_json::Map::new();
    extra.insert("model".into(), Value::String(o.model_id.clone()));
    extra.insert("steps".into(), Value::from(m.calls));
    extra.insert("cost_usd".into(), Value::from(m.cost_usd));
    extra.insert("error".into(), Value::String(String::new()));
    let review_open = crate::llm::review::review(&m.summary, &ws.join("output")).into_iter().chain(m.open.iter().cloned()).collect::<Vec<_>>();
    util::write(
        &run_dir.join("usage.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "steps": m.calls, "input_tokens": m.input_tokens, "output_tokens": m.output_tokens, "cost_usd": m.cost_usd, "error": "", "heals": 0, "model": m.served_by, "review_open": review_open,
            "accuracy": {"kind": m.kind, "passed": m.passed, "total": m.total, "candidates": m.candidates, "confidence": m.confidence()},
        }))?,
    )?;
    crate::log::event("step", "end", serde_json::json!({"steps": m.calls, "in": m.input_tokens, "out": m.output_tokens, "usd": m.cost_usd, "error": "", "depth": 0, "heals": 0, "model": m.served_by}));
    runmark::mark_complete(run_dir, &run_dir.to_string_lossy(), fp_dir, o.inputs_fingerprint.as_deref(), Some(extra))?;
    Ok(run_dir.to_path_buf())
}

/// Deliver the answer engine's verdict as the rollout's result: the answer first, then how it
/// was reached and how far it can be trusted.
fn finish_precise(run_dir: &Path, ws: &Path, fp_dir: Option<&Path>, o: &SolveOpts, v: &precise::Verdict) -> Result<PathBuf> {
    let answer = v.answer.clone().unwrap_or_default();
    let others: Vec<String> = {
        let mut seen: Vec<String> = vec![];
        for s in &v.samples {
            if let Some(a) = &s.answer {
                if !precise::same_answer(a, &answer, "") && !seen.iter().any(|x| precise::same_answer(x, a, "")) {
                    seen.push(a.clone());
                }
            }
        }
        seen
    };
    let how = match v.confidence() {
        "high" if v.escalated => format!("The attempts did not agree, so a stronger model settled it; {} of {} answers match.", v.agree, v.answered),
        "high" => format!("Checked: {} of {} independent attempts gave this answer.", v.agree, v.answered),
        _ if v.how == "reasoned" || v.how == "direct" => "Low confidence: no program produced an answer, so this is the model's own step-by-step estimate. Check it before relying on it.".to_string(),
        "medium" if v.answered == 1 => "Computed once by a program; no second attempt confirmed it.".to_string(),
        "medium" => format!("{} of {} independent attempts gave this answer; the others differed ({}).", v.agree, v.answered, others.iter().map(|a| util::take_chars(a, 40)).collect::<Vec<_>>().join(", ")),
        _ => format!("Low confidence: the attempts did not agree (other answers: {}). Treat this as a best guess and check it.", others.iter().map(|a| util::take_chars(a, 40)).collect::<Vec<_>>().join(", ")),
    };
    let mut md = format!("{answer}\n\n---\n\n*{how}*\n");
    if !v.program.trim().is_empty() {
        md.push_str(&format!("\n<details><summary>The program that computed it</summary>\n\n```python\n{}\n```\n\nIt printed:\n\n```\n{}\n```\n\n</details>\n", v.program.trim(), util::take_chars(v.output.trim(), 1500)));
    }
    util::write(&ws.join("output/answer.md"), &md)?;
    util::write(&run_dir.join("final.md"), &answer)?;
    util::write(&run_dir.join("accuracy.json"), serde_json::to_string_pretty(&v.to_json())?)?;
    crate::say!("      accuracy boost: {} — {} of {} attempts agree (confidence {})", util::take_chars(&answer, 100), v.agree, v.answered, v.confidence());
    let mut extra = serde_json::Map::new();
    extra.insert("model".into(), Value::String(o.model_id.clone()));
    extra.insert("steps".into(), Value::from(v.calls));
    extra.insert("cost_usd".into(), Value::from(v.cost_usd));
    extra.insert("error".into(), Value::String(String::new()));
    util::write(
        &run_dir.join("usage.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "steps": v.calls, "input_tokens": v.input_tokens, "output_tokens": v.output_tokens, "cost_usd": v.cost_usd, "error": "", "heals": 0, "model": v.served_by, "review_open": Vec::<String>::new(),
            "accuracy": {"agree": v.agree, "answered": v.answered, "tried": v.tried, "confidence": v.confidence(), "escalated": v.escalated},
        }))?,
    )?;
    crate::log::event("step", "end", serde_json::json!({"steps": v.calls, "in": v.input_tokens, "out": v.output_tokens, "usd": v.cost_usd, "error": "", "depth": 0, "heals": 0, "model": v.served_by}));
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
    fn starter_tool_validates_and_is_not_in_the_default_base() {
        let mut t = default_base_tree();
        assert!(!t.contains_key("tools/csv_profile.json"), "the default base ships no task-specific tools");
        t.extend(starter_tools_tree());
        let census = tree::validate(&native_surface(), &t, true, None, &tree::reserved_default()).unwrap();
        assert_eq!(census["tools"], 1);
        assert!(tree::is_allowed(&native_surface(), "tools/csv_profile.json"));
        assert!(!tree::is_allowed(&super::super::opencode::opencode_surface(), "tools/csv_profile.json"));
    }

    #[test]
    fn bad_wired_tools_are_rejected_with_reasons() {
        let mut t = default_base_tree();
        t.extend(starter_tools_tree());
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
