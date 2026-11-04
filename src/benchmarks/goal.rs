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
            },
        )
    }
    fn delivered(&self, run_dir: &Path) -> bool {
        !native::deliverables(run_dir).is_empty()
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
    fn validate_tree(&self, _t: &Tree) -> Result<()> {
        Ok(())
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
        let r = client::chat(&conn, &model, &client::ChatReq { system: "You are a strict, fair grader. Output JSON only.", messages: &msgs, tools: &[], max_tokens: 4096 }, Duration::from_secs(300))?;
        let txt = r.text.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim().to_string();
        let v: Value = match serde_json::from_str(&txt) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let crit = v["criteria"].as_array().cloned().unwrap_or_default();
        if crit.is_empty() {
            return Ok(None);
        }
        let passed = crit.iter().filter(|c| c["pass"].as_bool().unwrap_or(false)).count();
        Ok(Some(json!({"n_passed": passed, "n_criteria": crit.len(), "all_pass": passed == crit.len(), "criteria": crit, "metric": "criteria"})))
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
