//! LAB (harvey-labs) adapter: one type for both substrate cells.
//!
//! Upstream tools run in subprocesses under the LAB interpreter. `task_json` strips grading
//! criteria before exposing task data to the improver.

use super::{bp_from, lookup, run_detached, tasks};
use crate::config;
use crate::core::runmark as RM;
use crate::core::spec::{channel_map, resolve_channel_model, BenchPrompts, BenchSpec, BenchmarkAdapter};
use crate::core::tree::{self, SurfaceSpec, Tree};
use crate::substrate::codex::{codex_surface, CodexSubstrate};
use crate::substrate::opencode::{opencode_surface, OpencodeSubstrate};
use crate::substrate::{Install, RunRequest};
use crate::{say, util};
use anyhow::{bail, Result};
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

static DATA: Lazy<Value> = Lazy::new(|| serde_json::from_str(include_str!("data/lab.json")).unwrap());
static BASE_DONE: Lazy<Mutex<HashSet<PathBuf>>> = Lazy::new(|| Mutex::new(HashSet::new()));
static BASE_BODY: Lazy<Mutex<HashMap<String, String>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn d(name: &str) -> String {
    DATA[name].as_str().unwrap_or("").to_string()
}

fn table(name: &str) -> BTreeMap<String, String> {
    DATA[name].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()
}

/// LAB-wide profile plus optional practice-area and work-type overlays.
pub fn lab_prompts(practice_area: &str, work_type: &str) -> BenchPrompts {
    let base = bp_from(&DATA["LAB_BASE"]);
    let parts = [base.domain_guidance.clone(), lookup(&table("LAB_PRACTICE_AREAS"), practice_area), lookup(&table("LAB_WORK_TYPES"), work_type)];
    let g = parts.iter().map(|p| p.trim()).filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n\n");
    BenchPrompts { domain_guidance: g, ..base }
}

pub fn codex_base_body() -> String {
    d("CODEX_BASE_BODY")
}

/// The scrubbed credentials the LAB judge needs for `model`; () when it needs none.
fn judge_keys(model: &str) -> &'static [&'static str] {
    let n = model.to_lowercase();
    if n.starts_with("vertex/") {
        &[]
    } else if n.starts_with("claude") {
        &["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"]
    } else if n.starts_with("gemini") {
        &["GOOGLE_API_KEY", "GEMINI_API_KEY"]
    } else if ["gpt", "o1", "o3", "o4", "o5"].iter().any(|p| n.starts_with(p)) {
        &["OPENAI_API_KEY"]
    } else {
        &[]
    }
}

/// LAB subprocess env; `keep` lists the scrubbed credentials the subprocess still needs.
fn lab_env(keep: &[&str]) -> BTreeMap<String, String> {
    let cfg = config::get();
    let mut e = config::scrub_env(None, keep);
    e.insert("PYTHONPATH".into(), cfg.lab_repo().to_string_lossy().to_string());
    e.insert(
        "PATH".into(),
        format!(
            "{}:{}:{}/codex-path:{}",
            cfg.lab_python().parent().unwrap().display(),
            cfg.bun.parent().unwrap().display(),
            cfg.codex_dist.display(),
            std::env::var("PATH").unwrap_or_default()
        ),
    );
    e
}

fn lab_py(args: &[&str], timeout: u64) -> Result<util::Output> {
    let cfg = config::get();
    let mut c = std::process::Command::new(cfg.lab_python());
    c.args(args).current_dir(cfg.lab_repo());
    config::apply_env(&mut c, &lab_env(&[]));
    util::run_with_timeout(c, None, Duration::from_secs(timeout))
}

fn require_runner(p: &Path) -> Result<()> {
    if !p.is_file() {
        bail!(
            "upstream LAB runner missing: {}. The upstream clone does not track it at the pinned commit; it ships as the tracked overlay third_party/lab_runner/ — run `harness2 setup lab` to copy it into place.",
            p.display()
        );
    }
    Ok(())
}

/// Composed once per process per mode: composing reads every SKILL.md.
fn opencode_base_body(skill_mode: &str) -> Result<String> {
    if let Some(b) = BASE_BODY.lock().unwrap().get(skill_mode) {
        return Ok(b.clone());
    }
    require_runner(&config::get().lab_repo().join("opencode/render.py"))?;
    let src = d("_BASE_BODY_SRC");
    let r = lab_py(&["-c", &src, skill_mode], 300).map_err(|e| anyhow::anyhow!("base-body compose failed to run: {e}"))?;
    if r.code != Some(0) || r.stdout.trim().is_empty() {
        bail!("base-body compose failed under {} (rc={:?}): {}", config::get().lab_python().display(), r.code, util::tail_chars(r.stderr.trim(), 800));
    }
    BASE_BODY.lock().unwrap().insert(skill_mode.into(), r.stdout.clone());
    Ok(r.stdout)
}

/// Script basenames the LAB opencode runtime puts into every workspace.
fn runtime_scripts() -> Vec<String> {
    let skills = config::get().lab_repo().join("harness/skills");
    let mut v: Vec<String> = util::files_sorted(&skills)
        .into_iter()
        .filter(|p| p.extension().map(|e| e == "py").unwrap_or(false) && p.parent().map(|q| q.ends_with("scripts")).unwrap_or(false))
        .filter(|p| util::rel(&skills, p).split('/').count() == 3)
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    v.sort();
    v.dedup();
    v
}

fn require_runtime_scripts() -> Result<Vec<String>> {
    let n = runtime_scripts();
    if n.is_empty() {
        bail!(
            "no skill scripts under {} — an empty exemption set is precisely the condition this derivation exists to prevent, and it fails silently as mass fell_back rather than as an error; materialise the LAB checkout with `harness2 setup lab`",
            config::get().lab_repo().join("harness/skills").display()
        );
    }
    Ok(n)
}

/// The LAB BenchSpec for one substrate cell.
pub fn lab_spec(substrate: &str) -> Result<BenchSpec> {
    if substrate != "oc" && substrate != "cx" {
        bail!("substrate must be 'oc' or 'cx', got {substrate:?}");
    }
    let ch = resolve_channel_model(None, None, "high", 120, 3600)?;
    let mut s = BenchSpec::new(
        "lab",
        config::get().runs.join("lab"),
        if substrate == "oc" { "workspace_config" } else { "codex_root" },
        bp_from(&DATA["LAB_BASE"]),
        channel_map(ch.clone(), ch),
    );
    s.domain_word = "practice area".into();
    s.leak_globs = vec!["task.json".into()];
    s.allow_subagents = false;
    s.reserved_scripts = if substrate == "oc" { runtime_scripts() } else { vec![] };
    s.obs_trunc = 50_000;
    s.checked()
}

pub struct LabAdapter {
    spec: BenchSpec,
    substrate: String,
    model: String,
    results_root: PathBuf,
    timeout: u64,
    skill_mode: String,
    solver_effort: String,
    judge_model: String,
    split_file: Option<PathBuf>,
    deliverables: Mutex<HashMap<String, BTreeMap<String, String>>>,
}

impl LabAdapter {
    pub fn new(substrate: &str, split_file: Option<&Path>) -> Result<LabAdapter> {
        let cfg = config::get();
        Ok(LabAdapter {
            spec: lab_spec(substrate)?,
            substrate: substrate.into(),
            model: if substrate == "oc" { cfg.solver_model_oc.clone() } else { cfg.solver_model_cx.clone() },
            results_root: cfg.runs.join("lab/results").join(substrate),
            timeout: 3600,
            skill_mode: "native".into(),
            solver_effort: "high".into(),
            judge_model: cfg.lab_judge_model(),
            split_file: split_file.map(|p| p.to_path_buf()),
            deliverables: Mutex::new(HashMap::new()),
        })
    }

    /// task.json WITHOUT `criteria` — the ONLY place this adapter reads a task.
    fn task_json(&self, task_id: &str) -> Result<Value> {
        let p = config::get().lab_repo().join("tasks").join(task_id).join("task.json");
        let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        if let Some(o) = v.as_object_mut() {
            o.remove("criteria");
        }
        Ok(v)
    }

    fn run_dir(&self, run_id: &str) -> PathBuf {
        self.results_root.join(run_id)
    }

    fn install(&self, harness_dir: Option<&Path>) -> Result<Install> {
        let Some(h) = harness_dir else {
            let mut extras = BTreeMap::new();
            extras.insert("harness".into(), "base".into());
            return Ok(Install { staging_dir: self.base_harness()?, fingerprint: "base".into(), extras });
        };
        let mut inst = if self.substrate == "oc" {
            OpencodeSubstrate.materialise(h, &self.spec.install_mode, &BTreeMap::new())?
        } else {
            CodexSubstrate.materialise(h, &self.spec.install_mode, &BTreeMap::new())?
        };
        inst.extras.insert("harness".into(), "staged".into());
        Ok(inst)
    }

    fn launch(&self, req: &RunRequest, install: &Install) -> Result<Vec<PathBuf>> {
        let cfg = config::get();
        let mut dirs = vec![];
        for run_id in &req.run_ids {
            let (argv, env, wall) = if self.substrate == "oc" {
                let runner = cfg.lab_repo().join("opencode/run_opencode.py");
                require_runner(&runner)?;
                let mut argv: Vec<String> = vec![
                    cfg.lab_python().to_string_lossy().into(), "-u".into(), runner.to_string_lossy().into(),
                    "--task".into(), req.task_ref.clone(), "--run-id".into(), run_id.clone(),
                    "--results-root".into(), self.results_root.to_string_lossy().into(), "--model".into(), self.model.clone(),
                    "--tool-mode".into(), "base".into(), "--reasoning-effort".into(), self.solver_effort.clone(),
                    "--skill-mode".into(), self.skill_mode.clone(), "--timeout".into(), req.timeout.to_string(),
                    "--opencode-dir".into(), cfg.opencode_dir.to_string_lossy().into(), "--bun".into(), cfg.bun.to_string_lossy().into(),
                ];
                if install.extras.get("harness").map(|s| s.as_str()) != Some("base") {
                    argv.extend(["--harness-dir".into(), install.staging_dir.to_string_lossy().into()]);
                }
                (argv, lab_env(config::provider_keys(&self.model)), req.timeout * 2 + 600)
            } else {
                let runner = cfg.lab_repo().join("external/adapt_runs_cx/run_codex_lab.py");
                require_runner(&runner)?;
                let argv: Vec<String> = vec![
                    cfg.lab_python().to_string_lossy().into(), "-u".into(), runner.to_string_lossy().into(),
                    "--task".into(), req.task_ref.clone(), "--run-id".into(), run_id.clone(), "--model".into(), self.model.clone(),
                    "--reasoning-effort".into(), self.solver_effort.clone(), "--timeout".into(), req.timeout.to_string(),
                    "--harness-dir".into(), install.staging_dir.to_string_lossy().into(),
                ];
                let mut env = lab_env(&[]);
                env.insert("ADAPT_LABCX_CODEX_BIN".into(), cfg.codex_dist.join("codex").to_string_lossy().into());
                env.insert("ADAPT_LABCX_RESULTS".into(), self.results_root.to_string_lossy().into());
                env.insert("ADAPT_LABCX_SCRATCH".into(), config::env("ADAPT_LABCX_SCRATCH").unwrap_or_else(|| cfg.runs.join("lab/scratch").to_string_lossy().into()));
                env.insert("ADAPT_LABCX_BASE_URL".into(), config::env("ADAPT_LABCX_BASE_URL").unwrap_or_else(|| cfg.codex_base_url.clone()));
                if !cfg.vertex_proxy_key.is_empty() {
                    env.insert("VERTEX_PROXY_KEY".into(), cfg.vertex_proxy_key.clone());
                }
                (argv, env, req.timeout + 900)
            };
            match run_detached(&argv, &env, Some(&cfg.lab_repo()), wall, Some(Path::new("/dev/null")))? {
                None => say!("  [lab-{}] runner wall-timeout for {} @ {run_id}", self.substrate, req.task_ref),
                Some(0) => {}
                Some(c) => bail!("Benchmark runner exited with status {c}"),
            }
            dirs.push(self.run_dir(run_id));
        }
        Ok(dirs)
    }

    /// rm -rf restricted to this cell's results root — a guard, not a convenience.
    fn safe_rmtree(&self, d: &Path) -> Result<()> {
        let d = std::fs::canonicalize(d).unwrap_or_else(|_| d.to_path_buf());
        let root = std::fs::canonicalize(&self.results_root).unwrap_or_else(|_| self.results_root.clone());
        if root != d && !d.starts_with(&root) {
            bail!("refusing to remove {}: outside the LAB results root {}", d.display(), root.display());
        }
        util::rmtree(&d);
        Ok(())
    }

    fn output_fingerprint(&self, run_dir: &Path) -> Option<String> {
        self.delivered(run_dir).then(|| RM::fingerprint(Some(&run_dir.join("output"))))
    }

    fn validate_oc(&self, t: &Tree) -> Result<()> {
        let td = tempfile::tempdir()?;
        let h = tree::write_tree(&opencode_surface(), t, &td.path().join("h"), false)?;
        let src = d("_RENDER_CHECK_SRC");
        let r = lab_py(&["-c", &src, &h.to_string_lossy(), &self.skill_mode], 300).map_err(|e| anyhow::anyhow!("harness does not render: {e}"))?;
        let got: Value = serde_json::from_str(r.stdout.trim()).map_err(|e| anyhow::anyhow!("harness does not render: {e}"))?;
        if let Some(e) = got["error"].as_str().filter(|e| !e.is_empty()) {
            bail!("harness does not render: {e}");
        }
        let missing: Vec<&str> = got["missing"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect()).unwrap_or_default();
        let mut errs = vec![];
        if missing.contains(&"output/") {
            errs.push("the rendered prompt must say deliverables go to `output/<filename>` explicitly -- relative write/edit paths are NOT auto-routed and anything outside output/ is invisible to grading");
        }
        if missing.contains(&"read_doc.py") {
            errs.push("the rendered prompt must say .docx/.pdf/.pptx/.xlsx are readable only via `python3 scripts/read_doc.py <path>` -- the `read` tool cannot parse them and the agent will silently 'read' binary noise");
        }
        if !errs.is_empty() {
            bail!("{}", errs.join(" | "));
        }
        Ok(())
    }

    fn validate_cx(&self, t: &Tree) -> Result<()> {
        let body = tree::always_on_text(&codex_surface(), t);
        let mut errs = vec![];
        if !body.contains("output/") {
            errs.push("AGENTS.md must say deliverables go to `output/<filename>` explicitly -- under codex nothing routes relative paths, and anything left outside output/ is invisible to grading. Add the output/ rule back to AGENTS.md.");
        }
        if !body.contains("read_doc.py") {
            errs.push("AGENTS.md must say .docx/.pdf/.pptx/.xlsx are readable only via `python3 scripts/read_doc.py <path>` -- shell reads of binary documents return noise, silently. Add the read_doc.py rule back to AGENTS.md.");
        }
        if !errs.is_empty() {
            bail!("{}", errs.join(" | "));
        }
        Ok(())
    }

    fn tool_gate_problems(&self, run_dir: &Path) -> Vec<String> {
        let Some(c) = std::fs::read_to_string(run_dir.join("workspace/opencode.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) else {
            return vec![];
        };
        let perm = c.get("permission").cloned().unwrap_or(json!({}));
        let mut out = vec![];
        for (name, enabled) in c.get("tools").and_then(|t| t.as_object()).cloned().unwrap_or_default() {
            let want = if enabled.as_bool().unwrap_or(false) { "allow" } else { "deny" };
            if perm.get(&name).and_then(|p| p.as_str()) != Some(want) {
                out.push(format!("tool gate conflict for {name:?}: tools={enabled}, permission={}; expected {want:?}", perm.get(&name).cloned().unwrap_or(Value::Null)));
            }
        }
        out
    }

    fn check_tool_gate(&self, run_dir: &Path) -> Result<()> {
        let p = self.tool_gate_problems(run_dir);
        if !p.is_empty() {
            bail!("{}", p.join(" | "));
        }
        Ok(())
    }
}

impl BenchmarkAdapter for LabAdapter {
    fn spec(&self) -> &BenchSpec {
        &self.spec
    }
    fn substrate_name(&self) -> &str {
        &self.substrate
    }
    fn surface(&self) -> SurfaceSpec {
        if self.substrate == "oc" { opencode_surface() } else { codex_surface() }
    }
    fn tasks(&self, split: &str) -> Result<Vec<String>> {
        tasks::select(&tasks::discover("lab", None)?, split, self.split_file.as_deref(), "lab")
    }
    fn domain_of(&self, task_id: &str) -> String {
        task_id.split('/').next().unwrap_or("").to_string()
    }
    fn reference_domain_order(&self) -> Option<Vec<String>> {
        let mut v: Vec<String> = tasks::discover("lab", None).ok()?.into_values().collect();
        v.sort();
        v.dedup();
        Some(v)
    }
    fn prompts_for(&self, task_id: &str) -> BenchPrompts {
        let wt = self.task_json(task_id).ok().and_then(|t| t["work_type"].as_str().map(|s| s.to_string())).unwrap_or_default();
        lab_prompts(&self.domain_of(task_id), &wt)
    }
    /// Answer-free markdown rendering of a task.
    fn task_prompt(&self, task_id: &str) -> Result<String> {
        let t = self.task_json(task_id)?;
        let s = |k: &str| match &t[k] {
            Value::String(x) => x.clone(),
            Value::Null => String::new(),
            o => o.to_string(),
        };
        let mut parts = vec![format!("# TASK {task_id}"), format!("**Title:** {}", s("title")), format!("**Work type:** {}", s("work_type"))];
        let tags: Vec<String> = t["tags"].as_array().map(|a| a.iter().map(|x| x.as_str().map(|s| s.to_string()).unwrap_or_else(|| x.to_string())).collect()).unwrap_or_default();
        if !tags.is_empty() {
            parts.push(format!("**Tags:** {}", tags.join(", ")));
        }
        parts.push(format!("\n## Instructions\n{}", s("instructions").trim()));
        if let Some(dv) = t["deliverables"].as_object().filter(|o| !o.is_empty()) {
            let lines: Vec<String> = dv
                .iter()
                .map(|(name, desc)| {
                    let desc = desc.as_str().unwrap_or("");
                    format!("- `{name}`{}", if !desc.is_empty() && desc != name { format!(" — {desc}") } else { String::new() })
                })
                .collect();
            parts.push(format!("\n## Deliverables\n{}", lines.join("\n")));
        }
        Ok(util::take_chars(&parts.join("\n\n"), 12_000))
    }
    fn task_files(&self, task_id: &str) -> Option<PathBuf> {
        let d = config::get().lab_repo().join("tasks").join(task_id).join("documents");
        d.is_dir().then_some(d)
    }
    fn execute(&self, task_id: &str, harness_dir: Option<&Path>, run_id: &str, timeout: Option<u64>) -> Result<PathBuf> {
        let out = self.run_dir(run_id);
        let inputs = RM::inputs_fingerprint(&self.task_prompt(task_id)?, self.task_files(task_id).as_deref());
        if RM::is_complete(&out, harness_dir, true, Some(&inputs)) && self.artifacts_intact(&out) == Some(true) {
            self.check_tool_gate(&out)?;
            return Ok(out);
        }
        if out.is_dir() && std::fs::read_dir(&out).map(|mut d| d.next().is_some()).unwrap_or(false) {
            let mut reason = RM::why_not(&out, harness_dir, Some(&inputs));
            if reason == "reusable" {
                reason = "output fingerprint missing or changed".into();
            }
            say!("  [lab-{}] re-rolling {run_id}: {reason}", self.substrate);
            self.safe_rmtree(&out)?;
        }
        self.deliverables.lock().unwrap().remove(&out.to_string_lossy().to_string());
        let install = self.install(harness_dir)?;
        let res = self.launch(&RunRequest { task_ref: task_id.into(), run_ids: vec![run_id.into()], timeout: timeout.unwrap_or(self.timeout) }, &install);
        if install.extras.get("harness").map(|s| s.as_str()) == Some("staged") {
            install.drop_staging();
        }
        res?;
        if !out.join("metrics.json").is_file() {
            bail!(
                "LAB rollout did not produce metrics.json: {} — the runner refuses to write it when the session stream had no parseable events (dead chain / missing binary), so this failure must not be cached",
                out.display()
            );
        }
        self.check_tool_gate(&out)?;
        let mut extra = serde_json::Map::new();
        extra.insert("output_fingerprint".into(), json!(self.output_fingerprint(&out)));
        RM::mark_complete(&out, run_id, harness_dir, Some(&inputs), Some(extra))?;
        Ok(out)
    }
    fn artifacts_intact(&self, run_dir: &Path) -> Option<bool> {
        let m = RM::read_mark(run_dir).unwrap_or(json!({}));
        Some(run_dir.join("metrics.json").is_file() && m.get("output_fingerprint").is_some() && m["output_fingerprint"] == json!(self.output_fingerprint(run_dir)))
    }
    fn delivered(&self, run_dir: &Path) -> bool {
        !util::files_sorted(&run_dir.join("output")).is_empty()
    }
    fn trajectory(&self, run_dir: &Path, obs_trunc: usize) -> Result<String> {
        let name = if self.substrate == "oc" { "opencode_session.jsonl" } else { "codex_session.jsonl" };
        let stream = run_dir.join(name);
        if !stream.is_file() {
            return Ok(format!("(no trajectory recorded for {})", run_dir.file_name().unwrap_or_default().to_string_lossy()));
        }
        Ok(if self.substrate == "oc" { OpencodeSubstrate.read_trajectory(&stream, obs_trunc, 1_000_000_000) } else { CodexSubstrate.read_trajectory(&stream, obs_trunc, 1_000_000_000) })
    }
    /// filename -> extracted text via the JUDGE's own reader (cached). A failure is an error.
    fn deliverable_files(&self, run_dir: &Path) -> Option<BTreeMap<String, String>> {
        self.deliverable_files_checked(run_dir).ok()
    }
    fn deliverable_text(&self, run_dir: &Path) -> Result<String> {
        let files = self.deliverable_files_checked(run_dir)?;
        if files.is_empty() {
            return Ok("(this run produced NO deliverables)".into());
        }
        Ok(files
            .iter()
            .map(|(n, t)| format!("----- FILE: {n} -----\n{}", if t.trim().is_empty() { "(empty file)" } else { t.trim() }))
            .collect::<Vec<_>>()
            .join("\n\n"))
    }
    fn deliverable_images(&self, run_dir: &Path) -> Vec<PathBuf> {
        for sub in ["output", "workspace/output"] {
            let d = run_dir.join(sub);
            if d.is_dir() {
                return util::files_sorted(&d)
                    .into_iter()
                    .filter(|p| p.extension().map(|e| ["png", "jpg", "jpeg", "gif", "webp"].contains(&e.to_string_lossy().to_lowercase().as_str())).unwrap_or(false))
                    .take(8)
                    .collect();
            }
        }
        vec![]
    }
    fn base_harness(&self) -> Result<PathBuf> {
        let d = if self.substrate == "oc" {
            self.spec.runs_root.join(format!("_base_harness_{}_{}", self.skill_mode, std::process::id()))
        } else {
            self.spec.runs_root.join(format!("_base_harness_cx_{}", std::process::id()))
        };
        let mut done = BASE_DONE.lock().unwrap();
        if !done.contains(&d) {
            let mut t = Tree::new();
            if self.substrate == "oc" {
                t.insert("systemprompt.md".into(), opencode_base_body(&self.skill_mode)?);
                for f in ["guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md"] {
                    t.insert(f.into(), String::new());
                }
                tree::write_tree(&opencode_surface(), &t, &d, true)?;
            } else {
                t.insert("AGENTS.md".into(), codex_base_body());
                tree::write_tree(&codex_surface(), &t, &d, true)?;
            }
            done.insert(d.clone());
        }
        Ok(d)
    }
    fn validate_tree(&self, t: &Tree) -> Result<()> {
        if self.substrate == "oc" { self.validate_oc(t) } else { self.validate_cx(t) }
    }
    fn preflight(&self) -> Result<()> {
        let cfg = config::get();
        if self.substrate == "oc" {
            OpencodeSubstrate.preflight()?;
        } else {
            CodexSubstrate.preflight(false)?;
        }
        if !cfg.lab_python().is_file() {
            bail!("LAB interpreter missing at {}; run `harness2 setup lab`", cfg.lab_python().display());
        }
        if self.substrate == "oc" {
            require_runner(&cfg.lab_repo().join("opencode/run_opencode.py"))?;
            require_runtime_scripts()?;
        } else {
            require_runner(&cfg.lab_repo().join("external/adapt_runs_cx/run_codex_lab.py"))?;
        }
        let r = lab_py(&["-c", "import evaluation.scoring"], 120).map_err(|e| anyhow::anyhow!("could not probe the LAB judge reader: {e}"))?;
        if r.code != Some(0) {
            let last = r.stderr.trim().lines().last().unwrap_or("no output").to_string();
            bail!("`import evaluation.scoring` fails under {} ({last}) — run `harness2 setup lab` to install its dependency environment", cfg.lab_python().display());
        }
        Ok(())
    }
    /// LAB's own judge (evaluation.run_eval). Judge failure => None, NEVER 0.
    fn score(&self, run_dir: &Path, task_id: &str) -> Result<Option<Value>> {
        let p = run_dir.join("scores.json");
        if !p.is_file() {
            let cfg = config::get();
            let abs = std::fs::canonicalize(run_dir).unwrap_or_else(|_| run_dir.to_path_buf());
            let log = run_dir.join("judge.log");
            let argv: Vec<String> = vec![
                cfg.lab_python().to_string_lossy().into(), "-u".into(), "-m".into(), "evaluation.run_eval".into(),
                "--run-id".into(), abs.to_string_lossy().into(), "--task".into(), task_id.into(), "--judge-model".into(), self.judge_model.clone(),
            ];
            match run_detached(&argv, &lab_env(judge_keys(&self.judge_model)), Some(&cfg.lab_repo()), 3600, Some(&log)) {
                Err(_) | Ok(None) => return Ok(None),
                Ok(Some(0)) => {}
                Ok(Some(c)) => bail!("Benchmark runner exited with status {c}; see {}", log.display()),
            }
        }
        let Some(s) = std::fs::read_to_string(&p).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) else { return Ok(None) };
        Ok(Some(json!({"n_passed": s["n_passed"], "n_criteria": s["n_criteria"], "all_pass": s["all_pass"]})))
    }
}

impl LabAdapter {
    fn deliverable_files_checked(&self, run_dir: &Path) -> Result<BTreeMap<String, String>> {
        let key = std::fs::canonicalize(run_dir).unwrap_or_else(|_| run_dir.to_path_buf()).to_string_lossy().to_string();
        if let Some(h) = self.deliverables.lock().unwrap().get(&key) {
            return Ok(h.clone());
        }
        let src = d("_EXTRACT_SRC");
        let out = run_dir.join("output");
        let r = lab_py(&["-c", &src, &out.to_string_lossy(), &config::LAB_DELIV_CAP.to_string()], 600)
            .map_err(|e| anyhow::anyhow!("deliverable extraction failed for {}: {e}", run_dir.display()))?;
        if r.code != Some(0) {
            bail!("deliverable extraction failed for {} (rc={:?}): {}", run_dir.display(), r.code, util::tail_chars(r.stderr.trim(), 500));
        }
        let v: Value = serde_json::from_str(&r.stdout)?;
        let files: BTreeMap<String, String> = v.as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()).unwrap_or_default();
        self.deliverables.lock().unwrap().insert(key, files.clone());
        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompts() {
        let bp = lab_prompts("antitrust-competition", "");
        bp.clone().checked().unwrap();
        assert!(!bp.work.is_empty());
        assert!(codex_base_body().contains("read_doc.py"));
        assert_eq!(judge_keys("claude-x"), &["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"]);
    }
}
