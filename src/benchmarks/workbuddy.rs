//! WorkBuddy-Bench adapter: one (subset, substrate) cell per `WbAdapter` instance.
//!
//! Execution is a Harbor batch, not a single run: one label is one rollout-store namespace and
//! one batch is one generated job YAML with n_attempts=k. Everything a model sees of a
//! deliverable comes through `wb_render`.

use super::{bp_from, join, lookup, tasks, wb_render as RENDER};
use crate::config;
use crate::core::runmark as RM;
use crate::core::spec::{channel_map, resolve_channel_model, BenchPrompts, BenchSpec, BenchmarkAdapter};
use crate::core::tree::{self, SurfaceSpec, Tree};
use crate::substrate::codex::{codex_surface, CodexSubstrate};
use crate::substrate::opencode::{self as OC, opencode_surface, OpencodeSubstrate};
use crate::substrate::Install;
use crate::util;
use anyhow::{bail, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use sha1::{Digest, Sha1};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

static DATA: Lazy<Value> = Lazy::new(|| serde_json::from_str(include_str!("data/workbuddy.json")).unwrap());
static BASE_DONE: Lazy<Mutex<HashSet<PathBuf>>> = Lazy::new(|| Mutex::new(HashSet::new()));
static LABEL_LOCKS: Lazy<Mutex<HashMap<String, Arc<Mutex<()>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static SLOTS: Lazy<HashMap<&'static str, crate::core::proposer::Semaphore>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("oc", crate::core::proposer::Semaphore::new(MAX_JOBS));
    m.insert("cx", crate::core::proposer::Semaphore::new(MAX_JOBS));
    m
});

pub const TRAJ_MAX_CHARS: usize = 2_000_000;
pub const EPHEMERAL_PREFIX: &str = "_ab-";
const MAX_JOBS: usize = 1;
const MAX_TRIALS: usize = 2;
const HARBOR_NAME_MAX: usize = 32;

fn harness_oc(fork: bool) -> String {
    format!("{}/{}", if fork { "oc-harness" } else { "opencode" }, config::WB_OC_HARNESS_VER)
}
fn harness_cx() -> String {
    format!("codex-harness/{}", config::CODEX_VERSION)
}

/// One WB subset: its dataset, domain unit, deliverable and scoring.
#[derive(Debug, Clone)]
pub struct Subset {
    pub name: &'static str,
    pub dataset_id: &'static str,
    pub domain_field: &'static str,
    pub deliverable: &'static str,
    pub scoring: &'static str,
    pub leak_globs: Vec<&'static str>,
    pub llm_judge_model: &'static str,
}

impl Subset {
    pub fn tasks_dir(&self) -> PathBuf {
        config::get().wb_repo().join("datasets").join(self.dataset_id).join("tasks")
    }
}

pub fn subset(name: &str) -> Result<Subset> {
    Ok(match name {
        "code" => Subset { name: "code", dataset_id: "wb-bench-code-v1.0", domain_field: "role", deliverable: "a diff against a real repository", scoring: "rule", leak_globs: vec!["tests/**", "solution/**", "environment/**"], llm_judge_model: "" },
        "web" => Subset { name: "web", dataset_id: "wb-bench-web-v1.0", domain_field: "category", deliverable: "a working web page or app", scoring: "llm_judge", leak_globs: vec!["tests/**", "environment/**"], llm_judge_model: config::WB_WEB_JUDGE },
        "office" => Subset { name: "office", dataset_id: "wb-bench-office-v1.0", domain_field: "category", deliverable: "office documents (spreadsheets, docs) written to disk", scoring: "rubric", leak_globs: vec!["tests/**", "environment/**"], llm_judge_model: "" },
        s => bail!("unknown subset {s:?}; one of [\"code\", \"office\", \"web\"]"),
    })
}

pub fn label_sanitize(s: &str) -> String {
    Regex::new(r"[^A-Za-z0-9_.-]").unwrap().replace_all(s, "-").into_owned()
}

/// Map a Harbor trial name (possibly truncated) back to a task id.
pub fn resolve_task(name: &str, subset: &str) -> Result<Option<String>> {
    let universe: Vec<String> = tasks::discover("wb", Some(subset))?.into_keys().collect();
    for t in &universe {
        if t == name || util::take_chars(t, HARBOR_NAME_MAX) == name {
            return Ok(Some(t.clone()));
        }
    }
    let c: Vec<&String> = universe.iter().filter(|t| t.starts_with(name)).collect();
    if c.len() == 1 {
        return Ok(Some(c[0].clone()));
    }
    if c.len() > 1 {
        bail!("trial name {name:?} is ambiguous across {} tasks: {:?}", c.len(), &c[..3.min(c.len())]);
    }
    Ok(None)
}

/// WorkBuddy subset profile plus its role/category overlay.
pub fn workbuddy_prompts(subset: &str, domain: &str) -> Result<BenchPrompts> {
    let key = super::norm(subset);
    let Some(base) = DATA["WORKBUDDY_BASES"].get(&key) else { bail!("unknown WorkBuddy subset {subset:?}; expected code, web or office") };
    let base = bp_from(base);
    let table: BTreeMap<String, String> = DATA["WORKBUDDY_DOMAINS"][&key].as_object().map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()).unwrap_or_default();
    let g = lookup(&table, domain);
    Ok(BenchPrompts { domain_guidance: join(&[&base.domain_guidance, &g]), ..base })
}

fn unit_float(v: &Value) -> Option<f64> {
    let n = match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }?;
    (0.0..=1.0).contains(&n).then_some(n)
}

fn int_or_none(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// Mirror of the bench scorer's precedence: reward > overall > test_pass_rate >
/// tests_passed/tests_total. Values outside [0,1] are rejected, not clamped.
pub fn score_from_payload(p: &Value) -> Option<f64> {
    for k in ["reward", "overall", "test_pass_rate"] {
        if let Some(s) = p.get(k).and_then(unit_float) {
            return Some(s);
        }
    }
    let passed = p.get("tests_passed").and_then(int_or_none)?;
    let total = p.get("tests_total").and_then(int_or_none)?;
    (total > 0).then(|| (passed as f64 / total as f64).clamp(0.0, 1.0))
}

fn json_obj(p: &Path) -> Option<Value> {
    serde_json::from_str::<Value>(&std::fs::read_to_string(p).ok()?).ok().filter(|v| v.is_object())
}

fn trial_finished(t: &Path) -> bool {
    t.join("verifier/score.json").is_file()
}

fn trial_graded(t: &Path) -> bool {
    json_obj(&t.join("verifier/score.json")).map(|p| score_from_payload(&p).is_some()).unwrap_or(false)
}

fn mtime(p: &Path) -> SystemTime {
    std::fs::metadata(p).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH)
}

fn log(msg: &str) {
    crate::say!("[{}] wb | {msg}", chrono::Local::now().format("%H:%M:%S"));
}

fn label_lock(label: &str) -> Arc<Mutex<()>> {
    LABEL_LOCKS.lock().unwrap().entry(label.into()).or_default().clone()
}

fn conc() -> usize {
    (MAX_TRIALS / MAX_JOBS).max(1)
}

/// Delete ephemeral job YAMLs left by a killed run; scoped to our own prefix.
pub fn sweep_orphans(max_age_s: u64) -> usize {
    let dir = config::get().wb_repo().join("configs/jobs");
    let mut n = 0;
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with(EPHEMERAL_PREFIX) && name.ends_with(".yaml") && mtime(&e.path()).elapsed().map(|d| d.as_secs() > max_age_s).unwrap_or(false) {
            let _ = std::fs::remove_file(e.path());
            n += 1;
        }
    }
    n
}

fn job_yaml(s: &Subset, substrate: &str, label: &str, todo: &[String], ws: Option<&Path>, k: usize, jobs_dir: &Path) -> String {
    let cfg = config::get();
    let sel = todo.iter().map(|t| format!("    - {t}")).collect::<Vec<_>>().join("\n");
    let abs = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().to_string();
    let (harness, extra) = if substrate == "oc" {
        let mut hp: BTreeMap<String, String> = BTreeMap::new();
        hp.insert("OPENCODE_VARIANT".into(), "high".into());
        if let Some(ws) = ws {
            hp.insert("WB_HARNESS_WS".into(), abs(ws));
            let composed = ws.join("_composed/prompt.j2");
            if composed.is_file() {
                hp.insert("prompt_template_path".into(), abs(&composed));
            }
        }
        let lines = hp.iter().map(|(k, v)| format!("  {k}: {}", json!(v))).collect::<Vec<_>>().join("\n");
        (harness_oc(ws.is_some()), format!("harness_params_override:\n{lines}\n"))
    } else {
        let mut hp: BTreeMap<String, String> = BTreeMap::new();
        hp.insert("CODEX_MODEL".into(), cfg.solver_model_cx.clone());
        if let Some(ws) = ws {
            hp.insert("WB_HARNESS_WS".into(), abs(ws));
        }
        let lines = hp.iter().map(|(k, v)| format!("  {k}: {}", json!(v))).collect::<Vec<_>>().join("\n");
        (
            harness_cx(),
            format!(
                "extra_allowed_hosts:\n  - host.docker.internal\nenvironment_override:\n  mounts:\n    - type: bind\n      source: {}\n      target: /opt/codex\n      read_only: true\nenv_override:\n  VERTEX_PROXY_KEY: {}\n  VERTEXAI_LOCATION: {}\n  CODEX_BRIDGE_URL: {}\nharness_params_override:\n{lines}\n",
                cfg.codex_dist.display(),
                json!(cfg.vertex_proxy_key),
                cfg.vertex_region,
                json!(cfg.codex_container_url)
            ),
        )
    };
    let judge = if s.llm_judge_model.is_empty() { String::new() } else { format!("llm_judge_override:\n  model: {}\n", s.llm_judge_model) };
    format!(
        "# GENERATED by harness2 (benchmarks/workbuddy.rs) -- ephemeral, safe to delete. label={label}\nmodel: {}\nharness: {harness}\ndataset: datasets/{}/tasks\nharness_backend: local\nmodel_connection: {}\nn_attempts: {k}\njobs_dir: {}\ntask_selection:\n  mode: name\n  names:\n{sel}\norchestrator_override:\n  n_concurrent_trials: {}\n{extra}{judge}",
        config::WB_SOLVER_MODEL,
        s.dataset_id,
        config::WB_MODEL_CONNECTION,
        jobs_dir.display(),
        conc()
    )
}

fn run_dirs(label_dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(label_dir).map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect()).unwrap_or_default();
    v.sort();
    v
}

fn trials_in(dirs: &[PathBuf], task_id: &str, subset: &str) -> Vec<PathBuf> {
    let re = Regex::new(r"^(?P<task>.+)__[A-Za-z0-9]+$").unwrap();
    let mut out = vec![];
    for run in dirs {
        let mut ds: Vec<PathBuf> = std::fs::read_dir(run).map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).collect()).unwrap_or_default();
        ds.sort();
        for d in ds {
            let name = d.file_name().unwrap().to_string_lossy().to_string();
            if let Some(m) = re.captures(&name) {
                if d.is_dir() && resolve_task(&m["task"], subset).ok().flatten().as_deref() == Some(task_id) {
                    out.push(d);
                }
            }
        }
    }
    out
}

/// Roll `todo` k times each under harness workspace `ws`, returning the run dirs THIS
/// invocation created.
#[allow(clippy::too_many_arguments)]
pub fn run_batch(s: &Subset, substrate: &str, label_dir: &Path, label: &str, tasks: &[String], ws: Option<&Path>, mut k: usize, done: &dyn Fn(&str) -> bool, need: Option<&dyn Fn(&str) -> usize>, timeout_s: Option<u64>) -> Result<Vec<PathBuf>> {
    let lock = label_lock(label);
    let _g = lock.lock().unwrap();
    let todo: Vec<String> = tasks.iter().filter(|t| !done(t)).cloned().collect();
    if todo.is_empty() {
        log(&format!("{label}: all {} task(s) already complete -- skipping", tasks.len()));
        return Ok(vec![]);
    }
    if todo.len() < tasks.len() {
        log(&format!("{label}: resuming, {} of {} already done", tasks.len() - todo.len(), tasks.len()));
    }
    if let Some(need) = need {
        let short = todo.iter().map(|t| need(t)).max().unwrap_or(1).max(1);
        if short != k {
            log(&format!("{label}: partial resume, asking for {short} attempt(s) not {k}"));
        }
        k = short;
    }
    let before: HashSet<PathBuf> = run_dirs(label_dir).into_iter().collect();
    let mut sorted = todo.clone();
    sorted.sort();
    let digest = hex::encode(Sha1::digest(format!("{label}|{}|{}", s.name, crate::core::tree::py_list(&sorted)).as_bytes()));
    let slug = format!("{EPHEMERAL_PREFIX}{}-{}", util::take_chars(&label_sanitize(label), 40), &digest[..8]);
    let cfg = config::get();
    let yaml = cfg.wb_repo().join("configs/jobs").join(format!("{slug}.yaml"));
    std::fs::create_dir_all(label_dir)?;
    util::write(&yaml, job_yaml(s, substrate, label, &todo, ws, k, label_dir))?;
    let timeout_s = timeout_s.filter(|t| *t > 0).unwrap_or(((todo.len() * k) as f64 / conc() as f64 * 3600.0 * 1.2 + 900.0) as u64);
    let mut env = config::scrub_env(None, &[]);
    env.insert("NO_FORCE_BUILD".into(), "1".into());
    env.insert("HARNESS2_WB_VERTEX_URL".into(), format!("{}/chat/completions", cfg.vertex_openai_url()));
    let res: Result<()> = (|| {
        let _slot = SLOTS[if substrate == "oc" { "oc" } else { "cx" }].acquire();
        log(&format!("{label}: {} task(s) x k={k} [{substrate}]", todo.len()));
        let argv: Vec<String> = ["uv", "run", "./scripts/run.sh", "--job", &slug].iter().map(|s| s.to_string()).collect();
        match super::run_detached(&argv, &env, Some(&cfg.wb_repo()), timeout_s, Some(&label_dir.join(format!("{slug}.log"))))? {
            None => log(&format!("{label}: TIMEOUT after {timeout_s}s -- stopped the job group")),
            Some(0) => {}
            Some(c) => log(&format!("{label}: run.sh exit {c} (partial results are still usable)")),
        }
        Ok(())
    })();
    let _ = std::fs::remove_file(&yaml);
    res?;
    let mut fresh: Vec<PathBuf> = run_dirs(label_dir).into_iter().filter(|d| !before.contains(d)).collect();
    fresh.sort();
    Ok(fresh)
}

pub fn spec_for(subset_name: &str, substrate: &str) -> Result<BenchSpec> {
    let s = subset(subset_name)?;
    let ch = resolve_channel_model(None, None, "high", 120, 3600)?;
    let mut spec = BenchSpec::new(
        &format!("wb-{subset_name}"),
        config::get().runs.join("wb").join(subset_name),
        if substrate == "oc" { "home_dir_jinja" } else { "codex_root" },
        workbuddy_prompts(subset_name, "")?,
        channel_map(ch.clone(), ch),
    );
    spec.domain_word = "category".into();
    spec.leak_globs = s.leak_globs.iter().map(|g| g.to_string()).collect();
    spec.allow_subagents = substrate == "oc";
    spec.obs_trunc = 50_000;
    spec.proposer_deliverables = if subset_name == "code" { "names" } else { "contents" }.into();
    spec.state_deliverable = true;
    spec.checked()
}

pub struct WbAdapter {
    pub subset: String,
    info: Subset,
    spec: BenchSpec,
    substrate: String,
    split_file: Option<PathBuf>,
}

impl WbAdapter {
    pub fn new(subset_name: &str, substrate: &str, split_file: Option<&Path>) -> Result<WbAdapter> {
        if !["oc", "cx"].contains(&substrate) {
            bail!("unknown substrate {substrate:?}; one of (\"oc\", \"cx\")");
        }
        Ok(WbAdapter { subset: subset_name.into(), info: subset(subset_name)?, spec: spec_for(subset_name, substrate)?, substrate: substrate.into(), split_file: split_file.map(|p| p.to_path_buf()) })
    }

    fn label(&self, run_id: &str, n: usize) -> Result<String> {
        let label = format!("{}{}", run_id.replace('/', "__"), if n > 1 { format!("__x{n}") } else { String::new() });
        let prefix = format!("{}-{}-", self.spec.name, self.substrate);
        if !label.starts_with(&prefix) {
            bail!("run id {run_id:?} does not belong to this cell (expected a namespace starting {prefix:?}); labels are constructed by core::schedule::ns only");
        }
        Ok(label)
    }
    fn label_dir(&self, label: &str) -> PathBuf {
        self.spec.runs_root.join("rollouts").join(label_sanitize(label))
    }
    fn trials(&self, label: &str, task_id: &str) -> Vec<PathBuf> {
        trials_in(&run_dirs(&self.label_dir(label)), task_id, &self.subset)
    }
    /// Newest-first, graded preferred but never filtered out.
    fn rank(&self, mut trials: Vec<PathBuf>) -> Vec<PathBuf> {
        trials.sort_by_key(|q| std::cmp::Reverse(mtime(q)));
        let (g, u): (Vec<PathBuf>, Vec<PathBuf>) = trials.into_iter().partition(|t| trial_graded(t));
        g.into_iter().chain(u).collect()
    }
    fn cell_identity(&self) -> serde_json::Map<String, Value> {
        let cfg = config::get();
        let mut m = serde_json::Map::new();
        m.insert("solver".into(), json!(if self.substrate == "oc" { cfg.solver_model_oc.clone() } else { cfg.solver_model_cx.clone() }));
        m.insert("oc_ver".into(), json!(config::WB_OC_HARNESS_VER));
        m.insert("connection".into(), json!(config::WB_MODEL_CONNECTION));
        m
    }
    fn identity_matches(&self, t: &Path) -> bool {
        let m = RM::read_mark(t).unwrap_or(json!({}));
        self.cell_identity().iter().all(|(k, v)| m.get(k) == Some(v))
    }
    fn reusable(&self, t: &Path, h: Option<&Path>) -> bool {
        trial_finished(t) && RM::is_complete(t, h, true, None) && self.identity_matches(t)
    }
    fn install(&self, h: Option<&Path>) -> Result<Install> {
        let Some(h) = h else {
            let mut extras = BTreeMap::new();
            extras.insert("install_mode".into(), "none".into());
            return Ok(Install { staging_dir: self.spec.runs_root.clone(), fingerprint: "base".into(), extras });
        };
        if self.substrate == "oc" {
            OpencodeSubstrate.materialise(h, &self.spec.install_mode, &BTreeMap::new())
        } else {
            CodexSubstrate.materialise(h, &self.spec.install_mode, &BTreeMap::new())
        }
    }
    fn drop_staging(i: &Install) {
        if i.fingerprint != "base" {
            i.drop_staging();
        }
    }
    /// The launcher body: the finished trials among the run dirs this invocation created, ranked.
    fn launch(&self, task_ref: &str, run_ids: &[String], timeout: u64, install: &Install) -> Result<Vec<PathBuf>> {
        let ws = (install.fingerprint != "base").then(|| install.staging_dir.clone());
        let k = run_ids.len();
        let label = self.label(&run_ids[0], k)?;
        if self.substrate == "cx" {
            CodexSubstrate.preflight(true)?;
        }
        let have = |tid: &str| -> usize {
            self.trials(&label, tid).iter().filter(|t| trial_finished(t) && mark_matches(t, &install.fingerprint) && self.identity_matches(t)).count()
        };
        let done = |tid: &str| have(tid) >= k;
        let need = |tid: &str| k.saturating_sub(have(tid));
        let new_dirs = run_batch(&self.info, &self.substrate, &self.label_dir(&label), &label, &[task_ref.to_string()], ws.as_deref(), k, &done, Some(&need), Some(timeout))?;
        let fresh: Vec<PathBuf> = trials_in(&new_dirs, task_ref, &self.subset).into_iter().filter(|t| trial_finished(t)).collect();
        Ok(self.rank(fresh))
    }
    fn trajectory_path(&self, run_dir: &Path) -> PathBuf {
        run_dir.join("agent").join(if self.substrate == "oc" { "oc-output.txt" } else { "codex-output.jsonl" })
    }
    /// True when the capture stream ends on a provider error object.
    fn killed(&self, run_dir: &Path) -> bool {
        let Ok(raw) = util::read_lossy(&self.trajectory_path(run_dir)) else { return false };
        for ln in raw.lines().rev() {
            let ln = ln.trim();
            if !ln.starts_with('{') {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<Value>(ln) else { return false };
            if !ev.is_object() {
                return false;
            }
            return ev.get("type").and_then(|t| t.as_str()) == Some("error") || ev.get("error").map(|e| e.is_object()).unwrap_or(false);
        }
        false
    }
}

fn mark_matches(t: &Path, fp: &str) -> bool {
    RM::read_mark(t).map(|m| m["version"].as_u64() == Some(RM::VERSION) && m["harness"].as_str() == Some(fp)).unwrap_or(false)
}

impl BenchmarkAdapter for WbAdapter {
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
        tasks::select(&tasks::discover("wb", Some(&self.subset))?, split, self.split_file.as_deref(), &format!("wb-{}", self.subset))
    }
    fn domain_of(&self, task_id: &str) -> String {
        tasks::discover("wb", Some(&self.subset)).ok().and_then(|m| m.get(task_id).cloned()).unwrap_or_default()
    }
    fn reference_domain_order(&self) -> Option<Vec<String>> {
        let mut v: Vec<String> = tasks::discover("wb", Some(&self.subset)).ok()?.into_values().collect();
        v.sort();
        v.dedup();
        Some(v)
    }
    fn prompts_for(&self, task_id: &str) -> BenchPrompts {
        workbuddy_prompts(&self.subset, &self.domain_of(task_id)).unwrap_or_else(|_| self.spec.prompts.clone())
    }
    fn task_prompt(&self, task_id: &str) -> Result<String> {
        util::read_lossy(&self.info.tasks_dir().join(task_id).join("instruction.md"))
    }
    fn task_files(&self, task_id: &str) -> Option<PathBuf> {
        let d = self.info.tasks_dir().join(task_id);
        d.is_dir().then_some(d)
    }
    /// One rollout = a one-task Harbor batch, idempotent by run id and harness id.
    fn execute(&self, task_id: &str, harness_dir: Option<&Path>, run_id: &str, timeout: Option<u64>) -> Result<PathBuf> {
        let label = self.label(run_id, 1)?;
        let usable: Vec<PathBuf> = self.trials(&label, task_id).into_iter().filter(|t| self.reusable(t, harness_dir)).collect();
        if let Some(u) = usable.into_iter().max_by_key(|q| mtime(q)) {
            return Ok(u);
        }
        let install = self.install(harness_dir)?;
        let fresh = self.launch(task_id, &[run_id.to_string()], timeout.unwrap_or(0), &install);
        Self::drop_staging(&install);
        let fresh = fresh?;
        let Some(chosen) = fresh.first().cloned() else { bail!("Harbor produced no finished trial for {label} / {task_id}") };
        RM::mark_complete(&chosen, run_id, harness_dir, None, Some(self.cell_identity()))?;
        Ok(chosen)
    }
    /// k replicates in ONE batch. Only the trials this invocation produced are marked.
    fn execute_many(&self, task_id: &str, harness_dir: Option<&Path>, run_ids: &[String]) -> Option<Result<Vec<PathBuf>>> {
        if run_ids.is_empty() {
            return Some(Ok(vec![]));
        }
        Some((|| {
            let k = run_ids.len();
            let label = self.label(&run_ids[0], k)?;
            let mine = || -> Vec<PathBuf> { self.trials(&label, task_id).into_iter().filter(|t| self.reusable(t, harness_dir)).collect() };
            let mut m = mine();
            if m.len() < k {
                let install = self.install(harness_dir)?;
                let fresh = self.launch(task_id, run_ids, 0, &install);
                Self::drop_staging(&install);
                for t in fresh? {
                    if RM::read_mark(&t).is_none() {
                        RM::mark_complete(&t, &run_ids[0], harness_dir, None, Some(self.cell_identity()))?;
                    }
                }
                m = mine();
            }
            let mut r = self.rank(m);
            r.truncate(k);
            Ok(r)
        })())
    }
    fn delivered(&self, run_dir: &Path) -> bool {
        RENDER::body_chars(run_dir, &self.subset) >= 200
    }
    fn trajectory(&self, run_dir: &Path, obs_trunc: usize) -> Result<String> {
        let f = self.trajectory_path(run_dir);
        if !f.is_file() {
            return Ok("(no trajectory)".into());
        }
        Ok(if self.substrate == "oc" { OpencodeSubstrate.read_trajectory(&f, obs_trunc, TRAJ_MAX_CHARS) } else { CodexSubstrate.read_trajectory(&f, obs_trunc, TRAJ_MAX_CHARS) })
    }
    fn deliverable_files(&self, run_dir: &Path) -> Option<BTreeMap<String, String>> {
        RENDER::render(run_dir, &self.subset).ok()
    }
    fn deliverable_listing(&self, run_dir: &Path) -> Option<Result<Vec<String>>> {
        Some(RENDER::listing(run_dir, &self.subset))
    }
    fn deliverable_text(&self, run_dir: &Path) -> Result<String> {
        RENDER::flat_text(run_dir, &self.subset)
    }
    fn deliverable_assets(&self, run_dir: &Path) -> Option<Vec<PathBuf>> {
        if self.subset == "office" {
            return Some(RENDER::raw_files(run_dir, &self.subset).map(|v| v.into_iter().map(|(_, p)| p).collect()).unwrap_or_default());
        }
        Some(vec![])
    }
    fn base_harness(&self) -> Result<PathBuf> {
        let d = self.spec.runs_root.join(format!("_base_harness_{}_{}", self.substrate, std::process::id()));
        let mut done = BASE_DONE.lock().unwrap();
        if !done.contains(&d) {
            let mut t = Tree::new();
            if self.substrate == "oc" {
                for f in opencode_surface().always_on {
                    t.insert(f.into(), String::new());
                }
                tree::write_tree(&opencode_surface(), &t, &d, true)?;
            } else {
                t.insert("AGENTS.md".into(), "Follow the task instructions and complete the requested work in the workspace, writing every deliverable where the instructions say.".into());
                tree::write_tree(&codex_surface(), &t, &d, true)?;
            }
            done.insert(d.clone());
        }
        Ok(d)
    }
    /// oc: check the Jinja invariant on the COMPOSED template. cx: nothing.
    fn validate_tree(&self, t: &Tree) -> Result<()> {
        if self.substrate != "oc" {
            return Ok(());
        }
        let td = tempfile::tempdir()?;
        let staged = tree::write_tree(&opencode_surface(), t, &td.path().join("h"), true)?;
        OC::assert_jinja_safe(&OC::compose_prompt(&staged))
    }
    fn preflight(&self) -> Result<()> {
        if util::which("uv").is_none() {
            bail!("`uv` is not on PATH; WB batches run as `uv run ./scripts/run.sh` inside the benchmark repo (see docs/BENCHMARKS.md)");
        }
        let repo = config::get().wb_repo();
        if !repo.join("scripts/run.sh").is_file() {
            bail!("the WorkBuddy-Bench checkout at {} has no scripts/run.sh — run `harness2 setup wb`", repo.display());
        }
        let slugs = if self.substrate == "oc" { vec![harness_oc(false), harness_oc(true)] } else { vec![harness_cx()] };
        for slug in slugs {
            let (family, ver) = slug.split_once('/').unwrap();
            let entry = repo.join("configs/harnesses").join(family).join("versions").join(format!("{ver}.yaml"));
            if !entry.is_file() {
                bail!("harness entry {slug:?} is missing at {} — the harness configs ship as third_party/wb_runner/configs/harnesses/; run `harness2 setup wb`", entry.display());
            }
        }
        if self.substrate == "cx" {
            CodexSubstrate.preflight(true)
        } else {
            OpencodeSubstrate.preflight()
        }
    }
    /// Read the verifier's own score.json; missing or unparseable is a JUDGE failure.
    fn score(&self, run_dir: &Path, _task_id: &str) -> Result<Option<Value>> {
        if !self.delivered(run_dir) && self.killed(run_dir) {
            return Ok(None);
        }
        let Some(p) = json_obj(&run_dir.join("verifier/score.json")) else { return Ok(None) };
        let Some(sc) = score_from_payload(&p) else { return Ok(None) };
        Ok(Some(json!({"n_passed": (sc * 100.0).round(), "n_criteria": 100, "all_pass": sc >= 1.0, "raw_score": sc})))
    }
}

#[allow(dead_code)]
fn _unused(_: Duration) {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn score_precedence() {
        assert_eq!(score_from_payload(&json!({"reward": 0.5, "overall": 0.9})), Some(0.5));
        assert_eq!(score_from_payload(&json!({"reward": 2.0, "overall": 0.9})), Some(0.9));
        assert_eq!(score_from_payload(&json!({"tests_passed": 3, "tests_total": 4})), Some(0.75));
        assert_eq!(score_from_payload(&json!({})), None);
    }
    #[test]
    fn prompts() {
        for s in ["code", "web", "office"] {
            workbuddy_prompts(s, "").unwrap().checked().unwrap();
        }
        assert!(workbuddy_prompts("web", "").unwrap().selection_lead.starts_with("DOES IT WORK"));
    }
}
