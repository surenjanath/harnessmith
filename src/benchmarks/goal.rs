//! Open-world goals: a task you type (plus optional input files) becomes a Harness² task.
//!
//! A *project* groups goals into one domain library, so later goals in the same project learn
//! from the harnesses adopted for earlier ones. Execution runs on the native substrate, so any
//! connected model can be the solver; the improver and selector run on the native `api`
//! channel unless a CLI channel is configured.
//!
//! ```text
//!     <root>/project.json                   settings (models, budgets)
//!     <root>/base/                          the project's base harness (editable)
//!     <root>/tasks/<id>/task.json           prompt, domain, optional rubric (never shown to models)
//!     <root>/tasks/<id>/files/              input files
//!     <root>/executions/<run_id>/           rollouts
//!     <root>/runs/adapt/...                 scheduler state (summaries, harnesses, selections)
//!     <root>/runs/library/...               the experience library
//! ```

use crate::core::spec::{channel_map, BenchPrompts, BenchSpec, BenchmarkAdapter, ChannelSpec};
use crate::core::tree::{self, SurfaceSpec, Tree};
use crate::llm::{client, connections};
use crate::substrate::native::{self, SolveOpts};
use crate::{core::runmark, util};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalSettings {
    /// `<connection>/<model>` that executes the task.
    pub solver_model: String,
    /// Model that edits the harness (Propose / Compose).
    pub improver_model: String,
    /// Model that picks among final outputs and writes the library lesson.
    pub selector_model: String,
    /// "api" (native) | "opencode" | "claude_cli" for the improver/selector channel.
    #[serde(default = "api")]
    pub improver_channel: String,
    #[serde(default = "steps60")]
    pub solver_steps: u32,
    #[serde(default = "steps80")]
    pub improver_steps: u32,
    #[serde(default = "t1800")]
    pub solver_timeout: u64,
    #[serde(default = "t1800")]
    pub improver_timeout: u64,
    #[serde(default = "yes")]
    pub allow_shell: bool,
    #[serde(default)]
    pub allow_subagents: bool,
    /// Model used to grade against a rubric when one is supplied (optional).
    #[serde(default)]
    pub judge_model: String,
    #[serde(default = "high")]
    pub effort: String,
    /// Stages and rollouts run at once. 1 suits a single local model (its prompt cache is
    /// reused between an agent's turns); 2 is the reference setting.
    #[serde(default = "two")]
    pub parallelism: usize,
    /// Guardrail gate on solver tool calls: "off" | "observe" | "enforce".
    #[serde(default = "enforce")]
    pub gate_mode: String,
    /// Extra operator rules (regexes over shell commands), on top of the built-in ones.
    #[serde(default)]
    pub gate_rules: Vec<String>,
    /// Task inputs are read-only to the solver.
    #[serde(default = "yes")]
    pub gate_protect_inputs: bool,
    /// Project memory (`remember` / `recall` for the solver).
    #[serde(default = "yes")]
    pub memory_enabled: bool,
    /// Append notes from adopted attempts to the base harness's LongTermMEMORY.md.
    #[serde(default = "yes")]
    pub memory_promote: bool,
    /// Sandbox for solver commands: "auto" | "bwrap" | "unshare" | "off".
    #[serde(default = "auto")]
    pub sandbox: String,
    #[serde(default)]
    pub sandbox_network: bool,
    /// Regression guard: the base attempt competes in selection; failure-looking finals drop out.
    #[serde(default = "yes")]
    pub select_guard: bool,
    /// MCP servers (stdio) whose tools the solver can call.
    #[serde(default)]
    pub mcp_servers: Vec<crate::llm::mcp::ServerCfg>,
    /// Goals that run again on a timer.
    #[serde(default)]
    pub schedules: Vec<Schedule>,
    /// Spending cap per run in USD (0: none). Agents stop once the run has spent this much.
    #[serde(default)]
    pub max_usd_per_run: f64,
    /// Grade the base attempt and the adopted output against the rubric after each run.
    #[serde(default = "yes")]
    pub auto_judge: bool,
    /// Embedding model for semantic memory recall (`connection/model`; empty: keywords only).
    #[serde(default)]
    pub memory_embed_model: String,
    /// Also recall notes shared from other projects.
    #[serde(default)]
    pub memory_shared: bool,
    /// Give improvers a code-computed digest of the rollouts (context/brief.md).
    #[serde(default = "yes")]
    pub improver_brief: bool,
    /// Grade this many times and take each criterion's majority vote (judge noise is large).
    #[serde(default = "three")]
    pub judge_repeats: usize,
    /// Let the solver fetch web pages (and search, when a search API key is set).
    #[serde(default)]
    pub web_access: bool,
    /// Base-harness files the improver may not change or delete (upstream #94).
    #[serde(default)]
    pub pinned_paths: Vec<String>,
}

/// Characters that hide text from a human reviewer but not from a model: zero-width, bidi
/// overrides and Unicode tag characters (upstream #96).
pub fn hidden_chars(s: &str) -> Vec<char> {
    let mut found: Vec<char> = s
        .chars()
        .filter(|c| matches!(*c as u32, 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2064 | 0x2066..=0x2069 | 0xFEFF | 0xE0000..=0xE007F))
        .collect();
    // A BOM at the very start of a file is harmless.
    if s.starts_with('\u{FEFF}') {
        found.remove(0);
    }
    found.dedup();
    found
}
fn three() -> usize {
    3
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Schedule {
    pub id: String,
    pub goal: String,
    #[serde(default)]
    pub domain: String,
    /// Minutes between runs (at least 5).
    pub every_minutes: u64,
    #[serde(default = "one")]
    pub k: usize,
    #[serde(default = "parallel")]
    pub mode: String,
    #[serde(default = "yes")]
    pub enabled: bool,
    /// Optional webhook that receives each result.
    #[serde(default)]
    pub reply_url: String,
}
fn one() -> usize {
    1
}
fn parallel() -> String {
    "parallel".into()
}
fn auto() -> String {
    "auto".into()
}
fn enforce() -> String {
    "enforce".into()
}
fn two() -> usize {
    2
}
fn api() -> String {
    "api".into()
}
fn high() -> String {
    "high".into()
}
fn steps60() -> u32 {
    60
}
fn steps80() -> u32 {
    80
}
fn t1800() -> u64 {
    1800
}
fn yes() -> bool {
    true
}

impl Default for GoalSettings {
    fn default() -> Self {
        let reg = connections::load();
        let role = |r: &str| reg.roles.get(r).cloned().unwrap_or_default();
        let solver = role("solver");
        GoalSettings {
            improver_model: if role("improver").is_empty() { solver.clone() } else { role("improver") },
            selector_model: if role("selector").is_empty() { solver.clone() } else { role("selector") },
            solver_model: solver,
            improver_channel: api(),
            solver_steps: 60,
            improver_steps: 80,
            solver_timeout: 1800,
            improver_timeout: 1800,
            allow_shell: true,
            allow_subagents: true,
            judge_model: role("judge"),
            effort: high(),
            parallelism: 2,
            gate_mode: enforce(),
            gate_rules: vec![],
            gate_protect_inputs: true,
            memory_enabled: true,
            memory_promote: true,
            sandbox: auto(),
            sandbox_network: false,
            select_guard: true,
            mcp_servers: vec![],
            schedules: vec![],
            max_usd_per_run: 0.0,
            auto_judge: true,
            memory_embed_model: String::new(),
            memory_shared: false,
            improver_brief: true,
            judge_repeats: 3,
            web_access: false,
            pinned_paths: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalTask {
    pub id: String,
    pub prompt: String,
    #[serde(default = "general")]
    pub domain: String,
    /// Optional grading criteria, used only by `score` after selection; never shown to models.
    #[serde(default)]
    pub rubric: String,
    #[serde(default)]
    pub created_at: String,
}
fn general() -> String {
    "general".into()
}

const WORK: &str = "open-ended work a person hands to an autonomous agent: research, analysis, writing, data wrangling, coding and file production, judged by whether the requested deliverables exist, are correct against the inputs, and are complete";

pub fn goal_prompts() -> BenchPrompts {
    BenchPrompts {
        work: WORK.into(),
        domain_word: "project".into(),
        deliverable_word: "deliverable".into(),
        inv_levers: "Levers that usually matter for this agent: reading every input before acting, writing deliverables under `output/` early and refining them, checking a written file by reading it back, and bounding retries after a tool error.".into(),
        dom_levers: "Domain knowledge belongs in skills named for the situation they handle (a document type, a data format, an analysis pattern), grounded in what the inputs and trajectories show.".into(),
        ..Default::default()
    }
}

pub struct GoalAdapter {
    pub root: PathBuf,
    pub settings: GoalSettings,
    spec: BenchSpec,
}

pub fn slugify(s: &str) -> String {
    let re = regex::Regex::new(r"[^a-z0-9]+").unwrap();
    let s = re.replace_all(&s.to_lowercase(), "-").trim_matches('-').to_string();
    let s = util::take_chars(&s, 40).trim_matches('-').to_string();
    if s.is_empty() { "goal".into() } else { s }
}

impl GoalAdapter {
    pub fn open(root: &Path) -> Result<GoalAdapter> {
        std::fs::create_dir_all(root)?;
        let settings: GoalSettings = std::fs::read_to_string(root.join("project.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| serde_json::from_value(v.get("settings").cloned().unwrap_or(v)).ok())
            .unwrap_or_default();
        GoalAdapter::with_settings(root, settings)
    }

    pub fn with_settings(root: &Path, settings: GoalSettings) -> Result<GoalAdapter> {
        let ch = |m: &str| -> Result<ChannelSpec> {
            let kind = settings.improver_channel.as_str();
            let model = if kind == "claude_cli" { m.split_once('/').map(|x| x.1).unwrap_or(m) } else { m };
            ChannelSpec::with(kind, model, &settings.effort, settings.improver_steps, settings.improver_timeout)
        };
        let mut spec = BenchSpec::new("goal", root.join("runs"), "native", goal_prompts().checked()?, channel_map(ch(&settings.improver_model)?, ch(&settings.selector_model)?));
        spec.domain_word = "project".into();
        spec.allow_subagents = settings.allow_subagents;
        spec.improver_brief = settings.improver_brief;
        let a = GoalAdapter { root: root.to_path_buf(), settings, spec: spec.checked()? };
        let base = a.root.join("base");
        if !tree::is_tree(&native::native_surface(), &base) {
            tree::write_tree(&native::native_surface(), &native::default_base_tree(), &base, false)?;
        }
        Ok(a)
    }

    pub fn save_settings(&self) -> Result<()> {
        let mut v: Value = std::fs::read_to_string(self.root.join("project.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
        v["settings"] = serde_json::to_value(&self.settings)?;
        util::write(&self.root.join("project.json"), serde_json::to_string_pretty(&v)?)
    }

    pub fn task_dir(&self, id: &str) -> PathBuf {
        self.root.join("tasks").join(id)
    }

    pub fn load_task(&self, id: &str) -> Result<GoalTask> {
        let p = self.task_dir(id).join("task.json");
        Ok(serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| anyhow::anyhow!("unknown goal {id:?}: {e}"))?)?)
    }

    /// Register a new goal; returns its id. Files are copied into the goal's input folder.
    pub fn add_task(&self, prompt: &str, domain: &str, rubric: &str, files: &[(String, Vec<u8>)]) -> Result<String> {
        if prompt.trim().is_empty() {
            bail!("the goal must not be empty");
        }
        let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        let mut id = format!("{stamp}-{}", slugify(prompt));
        let mut n = 2;
        while self.task_dir(&id).exists() {
            id = format!("{stamp}-{}-{n}", slugify(prompt));
            n += 1;
        }
        let t = GoalTask {
            id: id.clone(),
            prompt: prompt.trim().to_string(),
            domain: if domain.trim().is_empty() { "general".into() } else { domain.trim().to_string() },
            rubric: rubric.trim().to_string(),
            created_at: util::now_iso(),
        };
        let d = self.task_dir(&id);
        std::fs::create_dir_all(d.join("files"))?;
        for (name, bytes) in files {
            let rel = tree::normalize_rel(name)?;
            if rel.is_empty() {
                continue;
            }
            util::write(&d.join("files").join(rel), bytes)?;
        }
        util::write(&d.join("task.json"), serde_json::to_string_pretty(&t)?)?;
        Ok(id)
    }

    pub fn all_tasks(&self) -> Vec<GoalTask> {
        let mut v: Vec<GoalTask> = std::fs::read_dir(self.root.join("tasks"))
            .map(|rd| rd.filter_map(|e| e.ok()).filter_map(|e| self.load_task(&e.file_name().to_string_lossy()).ok()).collect())
            .unwrap_or_default();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    fn run_dir(&self, run_id: &str) -> PathBuf {
        self.root.join("executions").join(run_id)
    }
}

/// A file's text for the evidence tree; binaries become a one-line stub.
pub fn file_text(p: &Path, cap: usize) -> String {
    let Ok(b) = std::fs::read(p) else { return "(unreadable)".into() };
    let sniff = &b[..b.len().min(8192)];
    if sniff.contains(&0) || std::str::from_utf8(sniff).is_err() && String::from_utf8_lossy(sniff).matches('\u{FFFD}').count() > 8 {
        return format!("(binary file, {} bytes)", b.len());
    }
    let s = String::from_utf8_lossy(&b).into_owned();
    if util::char_len(&s) > cap {
        return format!("{}\n…(+{} chars — render truncation)", util::take_chars(&s, cap), util::char_len(&s) - cap);
    }
    s
}

impl BenchmarkAdapter for GoalAdapter {
    fn spec(&self) -> &BenchSpec {
        &self.spec
    }
    fn substrate_name(&self) -> &str {
        native::NAME
    }
    fn surface(&self) -> SurfaceSpec {
        native::native_surface()
    }
    fn tasks(&self, _split: &str) -> Result<Vec<String>> {
        Ok(self.all_tasks().into_iter().map(|t| t.id).collect())
    }
    fn domain_of(&self, task_id: &str) -> String {
        self.load_task(task_id).map(|t| t.domain).unwrap_or_else(|_| "general".into())
    }
    fn task_prompt(&self, task_id: &str) -> Result<String> {
        let t = self.load_task(task_id)?;
        let files: Vec<String> = util::files_sorted(&self.task_dir(task_id).join("files")).iter().map(|p| util::rel(&self.task_dir(task_id).join("files"), p)).collect();
        let mut s = t.prompt;
        if !files.is_empty() {
            s += &format!("\n\nInput files (in your workspace): {}", files.join(", "));
        }
        Ok(s)
    }
    fn task_files(&self, task_id: &str) -> Option<PathBuf> {
        let d = self.task_dir(task_id).join("files");
        d.is_dir().then_some(d)
    }
    fn execute(&self, task_id: &str, harness_dir: Option<&Path>, run_id: &str, timeout: Option<u64>) -> Result<PathBuf> {
        if self.settings.solver_model.trim().is_empty() {
            bail!("no solver model is set; connect a model and choose it as the solver");
        }
        let base = self.base_harness()?;
        let prompt = self.task_prompt(task_id)?;
        let files = self.task_files(task_id);
        let inputs = runmark::inputs_fingerprint(&prompt, files.as_deref());
        native::solve(
            &prompt,
            files.as_deref(),
            harness_dir.unwrap_or(&base),
            harness_dir.is_none(),
            &self.run_dir(run_id),
            &SolveOpts {
                model_id: self.settings.solver_model.clone(),
                max_steps: self.settings.solver_steps,
                timeout: timeout.unwrap_or(self.settings.solver_timeout),
                allow_shell: self.settings.allow_shell,
                inputs_fingerprint: Some(inputs),
                gate_mode: self.settings.gate_mode.clone(),
                gate_rules: self.settings.gate_rules.clone(),
                gate_protect_inputs: self.settings.gate_protect_inputs,
                memory: self.settings.memory_enabled.then(|| (self.root.clone(), task_id.to_string())),
                memory_embed: self.settings.memory_embed_model.clone(),
                memory_shared: self.settings.memory_shared,
                sandbox: self.settings.sandbox.clone(),
                sandbox_network: self.settings.sandbox_network,
                mcp_servers: self.settings.mcp_servers.clone(),
                web_access: self.settings.web_access,
            },
        )
    }
    fn delivered(&self, run_dir: &Path) -> bool {
        !native::deliverables(run_dir).is_empty()
    }
    fn guard_selection(&self) -> bool {
        self.settings.select_guard
    }
    fn trajectory(&self, run_dir: &Path, obs_trunc: usize) -> Result<String> {
        Ok(native::render_events(&run_dir.join("trajectory.jsonl"), obs_trunc, 2_000_000, 400))
    }
    fn deliverable_files(&self, run_dir: &Path) -> Option<BTreeMap<String, String>> {
        let out = run_dir.join("workspace/output");
        let m: BTreeMap<String, String> = native::deliverables(run_dir).iter().map(|p| (util::rel(&out, p), file_text(p, 60_000))).collect();
        (!m.is_empty()).then_some(m)
    }
    fn deliverable_text(&self, run_dir: &Path) -> Result<String> {
        let files = self.deliverable_files(run_dir).unwrap_or_default();
        if files.len() == 1 && files.contains_key("answer.md") {
            return Ok(files["answer.md"].clone());
        }
        Ok(files.iter().map(|(n, b)| format!("## {n}\n\n{b}")).collect::<Vec<_>>().join("\n\n"))
    }
    fn base_harness(&self) -> Result<PathBuf> {
        Ok(self.root.join("base"))
    }
    fn validate_tree(&self, t: &Tree) -> Result<()> {
        let mut errs = vec![];
        for (rel, body) in t {
            let h = hidden_chars(body);
            if !h.is_empty() {
                errs.push(format!("{rel} contains hidden characters ({}); remove them, every character in the harness must be visible", h.iter().take(4).map(|c| format!("U+{:04X}", *c as u32)).collect::<Vec<_>>().join(", ")));
            }
        }
        if !self.settings.pinned_paths.is_empty() {
            let base = tree::read_tree(&native::native_surface(), &self.root.join("base"), None);
            for p in &self.settings.pinned_paths {
                match (base.get(p), t.get(p)) {
                    (Some(b), Some(n)) if b != n => errs.push(format!("{p} is pinned by the project owner and must stay exactly as it is; undo your changes to it")),
                    (Some(_), None) => errs.push(format!("{p} is pinned by the project owner and cannot be removed")),
                    _ => {}
                }
            }
        }
        if errs.is_empty() { Ok(()) } else { bail!("{}", errs.join(" | ")) }
    }
    fn preflight(&self) -> Result<()> {
        for (role, m) in [("solver", &self.settings.solver_model), ("improver", &self.settings.improver_model), ("selector", &self.settings.selector_model)] {
            if m.trim().is_empty() {
                bail!("no {role} model is set");
            }
            if self.settings.improver_channel == "api" || role == "solver" {
                connections::resolve(m)?;
            }
        }
        Ok(())
    }
    /// Grade against the goal's rubric with the judge model, when both exist.
    fn score(&self, run_dir: &Path, task_id: &str) -> Result<Option<Value>> {
        let t = self.load_task(task_id)?;
        if t.rubric.trim().is_empty() || self.settings.judge_model.trim().is_empty() {
            return Ok(None);
        }
        let (conn, model) = connections::resolve(&self.settings.judge_model)?;
        let deliverable = self.deliverable_text(run_dir)?;
        let msg = format!(
            "Grade a deliverable against criteria.\n\n# TASK\n{}\n\n# CRITERIA (one per line)\n{}\n\n# DELIVERABLE\n{}\n\nFor each criterion decide pass or fail strictly from the deliverable. Reply with JSON only: {{\"criteria\": [{{\"criterion\": \"...\", \"pass\": true, \"why\": \"...\"}}]}}",
            t.prompt, t.rubric, deliverable
        );
        let msgs = vec![client::Msg::User(msg)];
        // Grade several times and take each criterion's majority vote: one grading of the same
        // output can swing by a criterion or two, which is larger than most real differences.
        let n_criteria = t.rubric.lines().filter(|l| !l.trim().is_empty()).count();
        let repeats = self.settings.judge_repeats.clamp(1, 7);
        let mut votes: Vec<Vec<Value>> = vec![];
        let mut last_err = None;
        for _ in 0..repeats {
            match client::chat(&conn, &model, &client::ChatReq { system: "You are a strict, fair grader. Output JSON only.", messages: &msgs, tools: &[], max_tokens: 4096 }, Duration::from_secs(300)) {
                Ok(r) => {
                    let txt = r.text.trim();
                    let body = match (txt.find('{'), txt.rfind('}')) {
                        (Some(a), Some(b)) if b > a => &txt[a..=b],
                        _ => continue,
                    };
                    if let Ok(v) = serde_json::from_str::<Value>(body) {
                        let crit = v["criteria"].as_array().cloned().unwrap_or_default();
                        if !crit.is_empty() && (n_criteria == 0 || crit.len() == n_criteria) {
                            votes.push(crit);
                        }
                    }
                }
                Err(e) => last_err = Some(e),
            }
        }
        if votes.is_empty() {
            return match last_err {
                Some(e) => Err(e),
                None => Ok(None),
            };
        }
        let n = votes.iter().map(|v| v.len()).min().unwrap_or(0);
        let crit: Vec<Value> = (0..n)
            .map(|i| {
                let yes = votes.iter().filter(|v| v[i]["pass"].as_bool().unwrap_or(false)).count();
                let pass = yes * 2 > votes.len();
                let why = votes.iter().find(|v| v[i]["pass"].as_bool().unwrap_or(false) == pass).map(|v| v[i]["why"].clone()).unwrap_or(Value::Null);
                json!({"criterion": votes[0][i]["criterion"], "pass": pass, "why": why, "votes": format!("{yes}/{}", votes.len())})
            })
            .collect();
        let passed = crit.iter().filter(|c| c["pass"].as_bool().unwrap_or(false)).count();
        let per_run: Vec<usize> = votes.iter().map(|v| v.iter().take(n).filter(|c| c["pass"].as_bool().unwrap_or(false)).count()).collect();
        Ok(Some(json!({"n_passed": passed, "n_criteria": n, "all_pass": passed == n, "criteria": crit, "metric": "criteria",
            "gradings": votes.len(), "spread": [per_run.iter().min(), per_run.iter().max()]})))
    }
}

/// Where goal projects live.
pub fn projects_root() -> PathBuf {
    crate::config::get().runs.join("goals")
}

pub fn project_dir(name: &str) -> Result<PathBuf> {
    if !regex::Regex::new(r"^[A-Za-z0-9_.\-]+$").unwrap().is_match(name) {
        bail!("project name {name:?} must be letters, digits, '.', '_' or '-'");
    }
    Ok(projects_root().join(name))
}

#[cfg(test)]
mod pin_tests {
    use super::*;

    #[test]
    fn pinned_files_and_hidden_characters_are_refused() {
        let d = tempfile::tempdir().unwrap();
        let mut a = GoalAdapter::open(d.path()).unwrap();
        let base = tree::read_tree(&native::native_surface(), &d.path().join("base"), None);
        assert!(a.validate_tree(&base).is_ok());
        // upstream #96: zero-width / bidi / tag characters hide instructions from a reviewer
        let mut t = base.clone();
        t.insert("guardrails.md".into(), "- Be careful.\u{200B}\u{202E}hidden\n".into());
        let e = a.validate_tree(&t).unwrap_err().to_string();
        assert!(e.contains("hidden characters") && e.contains("U+200B"), "{e}");
        // upstream #94: pinned files cannot change or disappear
        a.settings.pinned_paths = vec!["guardrails.md".into()];
        let mut t = base.clone();
        t.insert("guardrails.md".into(), "- anything goes\n".into());
        assert!(a.validate_tree(&t).unwrap_err().to_string().contains("pinned"));
        let mut t = base.clone();
        t.remove("guardrails.md");
        assert!(a.validate_tree(&t).unwrap_err().to_string().contains("cannot be removed"));
        let mut t = base.clone();
        t.insert("systemprompt.md".into(), "Edited prompt.\n".into());
        assert!(a.validate_tree(&t).is_ok(), "unpinned files stay editable");
    }
}
