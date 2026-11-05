//! Namespacing, the seeded stream order, and the drivers: one task, one domain, a sweep.
//!
//! `ns()` is the only function that builds a run namespace; `Namespace` is the only thing
//! that globs over one. Scheme: `<bench>-<substrate>-<mode><k>[-s<seed>][-<tag>]`.

use super::library::{self, Library};
use super::proposer::Propose;
use super::pyrandom::PyRandom;
use super::recursion::{self, ModeArgs};
use super::runmark;
use super::selection::{self, Selection};
use super::spec::{BenchmarkAdapter, RunView};
use super::tree::{self, Tree};
use crate::{say, util};
use anyhow::{bail, Result};
use fs2::FileExt;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const MODES: &[&str] = &["agg", "seq"];
pub const ROLLOUT_PARALLEL: usize = recursion::PHASE_WIDTH;

pub fn mode_name(token: &str) -> &str {
    match token {
        "agg" => "parallel",
        "seq" => "sequential",
        t => t,
    }
}

/// The on-disk token for a mode name: parallel/agg -> "agg", sequential/seq -> "seq".
pub fn mode_token(name: &str) -> Result<&'static str> {
    match name.trim().to_lowercase().as_str() {
        "agg" | "parallel" => Ok("agg"),
        "seq" | "sequential" => Ok("seq"),
        _ => bail!("unknown mode {name:?}; expected parallel | sequential (or the on-disk tokens {MODES:?})"),
    }
}

/// Check a tag's shape: it becomes a path segment and must not spell a seed segment.
pub fn validate_tag(tag: &str) -> Result<()> {
    if tag.is_empty() {
        return Ok(());
    }
    if !regex::Regex::new(r"^[A-Za-z0-9_.][A-Za-z0-9_.\-]*$").unwrap().is_match(tag) {
        bail!("tag {tag:?} must be [A-Za-z0-9_.-] with no leading '-' and no '/': it becomes a path segment and a run-id segment");
    }
    if regex::Regex::new(r"^s\d+(-|$)").unwrap().is_match(tag) {
        bail!("tag {tag:?} spells a seed segment; pass --seed instead so seeded and tagged namespaces can never collide");
    }
    Ok(())
}

/// The per-mode records of a summary (the legacy "arms" key is still read).
fn modes_of(record: &Value) -> Map<String, Value> {
    record
        .get("modes")
        .or_else(|| record.get("arms"))
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default()
}

pub fn ns(benchmark: &str, substrate: &str, mode: &str, k: usize, seed: Option<i64>, tag: &str) -> Result<String> {
    if !MODES.contains(&mode) {
        bail!("unknown mode {mode:?}; the method has exactly two: {MODES:?}");
    }
    if substrate.is_empty() {
        bail!("namespace needs a substrate name ('oc' | 'cx' | 'nx'); an empty one would let the substrates' caches collide on every label");
    }
    validate_tag(tag)?;
    let mut out = format!("{benchmark}-{substrate}-{mode}{k}");
    if let Some(s) = seed {
        out += &format!("-s{s}");
    }
    if !tag.is_empty() {
        out += &format!("-{tag}");
    }
    Ok(out)
}

/// Every name for one (benchmark, substrate, mode, k, seed, tag) setting.
#[derive(Debug, Clone)]
pub struct Namespace {
    pub benchmark: String,
    pub substrate: String,
    pub mode: String,
    pub k: usize,
    pub seed: Option<i64>,
    pub tag: String,
}

impl Namespace {
    pub fn new(benchmark: &str, substrate: &str, mode: &str, k: usize, seed: Option<i64>, tag: &str) -> Namespace {
        Namespace { benchmark: benchmark.into(), substrate: substrate.into(), mode: mode.into(), k, seed, tag: tag.into() }
    }
    pub fn name(&self) -> String {
        ns(&self.benchmark, &self.substrate, &self.mode, self.k, self.seed, &self.tag).expect("valid namespace")
    }
    pub fn run_id(&self, task_id: &str, label: &str) -> String {
        format!("{}/{task_id}/{label}", self.name())
    }
    pub fn task_dir(&self, root: &Path, task_id: &str) -> PathBuf {
        root.join(self.name()).join(task_id.replace('/', "__"))
    }
    /// The only glob over namespaces; both patterns match seeded, tagged and bare forms.
    pub fn glob(benchmark: &str, substrate: &str, k: usize, mode: &str) -> (String, String) {
        let base = format!("{benchmark}-{substrate}-{mode}{k}");
        (base.clone(), format!("{base}-*"))
    }
}

/// The full (domain, task_id) stream, the one constructor of continual-learning order.
///
/// seed=None is the reference order: `reference_domain_order`, tasks sorted. An int seed
/// shuffles under Random(f"{seed}:__domains__") and Random(f"{seed}:{domain}") -- bit-exact
/// with CPython.
pub fn stream_order(tasks_by_domain: &BTreeMap<String, Vec<String>>, seed: Option<i64>, reference_domain_order: &[String]) -> Result<Vec<(String, String)>> {
    let domains: Vec<String> = tasks_by_domain.keys().cloned().collect();
    let ordered: Vec<String> = match seed {
        None => {
            let refset: HashSet<&String> = reference_domain_order.iter().collect();
            let missing: Vec<&String> = domains.iter().filter(|d| !refset.contains(d)).collect();
            if !missing.is_empty() {
                bail!(
                    "reference_domain_order does not cover domain(s) {missing:?}; the frozen order list is stale — regenerate it rather than letting an unlisted domain land in an arbitrary position"
                );
            }
            reference_domain_order.iter().filter(|d| tasks_by_domain.contains_key(*d)).cloned().collect()
        }
        Some(s) => {
            let mut d = domains.clone();
            PyRandom::from_str(&format!("{s}:__domains__")).shuffle(&mut d);
            d
        }
    };
    let mut out = vec![];
    for d in ordered {
        let mut tasks = tasks_by_domain[&d].clone();
        tasks.sort();
        if let Some(s) = seed {
            PyRandom::from_str(&format!("{s}:{d}")).shuffle(&mut tasks);
        }
        out.extend(tasks.into_iter().map(|t| (d.clone(), t)));
    }
    Ok(out)
}

/// Descending task count, then name.
pub fn default_reference_domain_order(by_domain: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    let mut v: Vec<String> = by_domain.keys().cloned().collect();
    v.sort_by(|a, b| by_domain[b].len().cmp(&by_domain[a].len()).then(a.cmp(b)));
    v
}

fn reference_domain_order(adapter: &dyn BenchmarkAdapter, by_domain: &BTreeMap<String, Vec<String>>) -> Vec<String> {
    adapter.reference_domain_order().unwrap_or_else(|| default_reference_domain_order(by_domain))
}

/// Honor the selector channel while preserving fake/custom proposers when specs match.
pub fn selector_for(adapter: &dyn BenchmarkAdapter, prop: &Arc<dyn Propose>, selector: Option<Arc<dyn Propose>>) -> Result<Arc<dyn Propose>> {
    if let Some(s) = selector {
        return Ok(s);
    }
    let spec = adapter.spec().channels["selector"].clone();
    if prop.spec() == &spec {
        return Ok(prop.clone());
    }
    let audit = prop.audit_dir().map(|d| d.join("selector"));
    prop.fork(spec, audit)
}

fn tree_is_base(adapter: &dyn BenchmarkAdapter, base: &Tree, t: &Tree) -> bool {
    tree::diff_summary(&adapter.surface(), base, t).unchanged()
}

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub domain: String,
    pub k: usize,
    pub seed: Option<i64>,
    pub status: String,
    pub inputs: Option<String>,
    pub modes: Map<String, Value>,
    pub selections: Map<String, Value>,
    pub base: Vec<Value>,
    pub errors: Vec<String>,
}

impl TaskResult {
    pub fn to_json(&self) -> Value {
        json!({
            "task_id": self.task_id, "domain": self.domain, "k": self.k, "seed": self.seed,
            "status": self.status, "inputs": self.inputs, "base": self.base,
            "modes": self.modes, "selections": self.selections, "errors": self.errors,
        })
    }
}

fn task_inputs(adapter: &dyn BenchmarkAdapter, task_id: &str) -> Option<String> {
    let prompt = adapter.task_prompt(task_id).ok()?;
    let files = adapter.task_files(task_id);
    Some(runmark::inputs_fingerprint(&prompt, files.as_deref()))
}

fn sel_done(sel: Option<&Value>) -> bool {
    let Some(s) = sel else { return false };
    s.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)
        && s.get("chosen_label").and_then(|v| v.as_str()).map(|l| !l.is_empty()).unwrap_or(false)
}

/// ok iff every mode record on the merged summary succeeded and no error is recorded.
fn merged_status(merged: &Value) -> String {
    let modes = merged.get("modes").and_then(|m| m.as_object()).cloned().unwrap_or_default();
    let sels = merged.get("selections").and_then(|m| m.as_object()).cloned().unwrap_or_default();
    let no_errors = merged.get("errors").and_then(|e| e.as_array()).map(|a| a.is_empty()).unwrap_or(true);
    let ok = !modes.is_empty()
        && no_errors
        && modes.iter().all(|(m, v)| v.get("ok").and_then(|o| o.as_bool()).unwrap_or(false) && sel_done(sels.get(m)));
    if ok { "ok" } else { "partial" }.into()
}

/// Merge `res` into the summary under an flock, atomically, deriving the merged status.
fn write_summary(summary_path: &Path, res: &TaskResult) -> Result<()> {
    let lock = summary_path.with_extension("lock");
    let lf = std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(&lock)?;
    lf.lock_exclusive()?;
    let out: Result<()> = (|| {
        let mut merged = res.to_json();
        if summary_path.is_file() {
            let old: Value = std::fs::read_to_string(summary_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
            let old_inputs = old.get("inputs").and_then(|v| v.as_str());
            let inputs_changed = res.inputs.is_some() && old_inputs.is_some() && old_inputs != res.inputs.as_deref();
            if !inputs_changed {
                let mut prev = modes_of(&old);
                for (k, v) in merged["modes"].as_object().cloned().unwrap_or_default() {
                    prev.insert(k, v);
                }
                merged["modes"] = Value::Object(prev);
                let mut prev = old.get("selections").and_then(|s| s.as_object()).cloned().unwrap_or_default();
                for (k, v) in merged["selections"].as_object().cloned().unwrap_or_default() {
                    prev.insert(k, v);
                }
                merged["selections"] = Value::Object(prev);
                let mut errors: Vec<Value> = merged["errors"].as_array().cloned().unwrap_or_default();
                let mut seen: HashSet<String> = errors.iter().map(|e| e.to_string()).collect();
                for e in old.get("errors").and_then(|e| e.as_array()).cloned().unwrap_or_default() {
                    let es = e.as_str().map(|s| s.to_string()).unwrap_or_else(|| e.to_string());
                    if es == "no base rollout produced anything" && (res.status == "ok" || res.status == "partial") {
                        continue;
                    }
                    let mode = es.split(':').next().unwrap_or("");
                    if res.modes.get(mode).and_then(|m| m.get("ok")).and_then(|o| o.as_bool()).unwrap_or(false) && sel_done(res.selections.get(mode)) {
                        continue;
                    }
                    if seen.insert(e.to_string()) {
                        errors.push(e);
                    }
                }
                merged["errors"] = Value::Array(errors);
                let st = merged["status"].as_str().unwrap_or("").to_string();
                if st == "ok" || st == "partial" {
                    merged["status"] = json!(merged_status(&merged));
                }
            }
        }
        let tmp = summary_path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&merged)?)?;
        std::fs::rename(&tmp, summary_path)?;
        Ok(())
    })();
    let _ = fs2::FileExt::unlock(&lf);
    out
}

/// Replay a skipped task into its libraries from the saved chosen harness.
#[allow(clippy::too_many_arguments)]
fn restore_libraries(task_id: &str, k: usize, adapter: &dyn BenchmarkAdapter, runs_root: &Path, modes: &[&str], seed: Option<i64>, tag: &str, libs: &BTreeMap<String, Library>, selections: &Map<String, Value>) -> Result<()> {
    let surface = adapter.surface();
    for mode in modes {
        let (Some(lib), Some(sel)) = (libs.get(*mode), selections.get(*mode)) else { continue };
        let label = sel.get("chosen_label").and_then(|v| v.as_str()).unwrap_or("");
        if label.is_empty() || !sel.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) || !library::has_experience(sel.get("experience")) {
            continue;
        }
        let n = Namespace::new(&adapter.spec().name, adapter.substrate_name(), mode, k, seed, tag);
        let t = tree::read_tree(&surface, &n.task_dir(runs_root, task_id).join("harnesses").join(label), None);
        if t.is_empty() {
            continue;
        }
        lib.add(
            task_id,
            &t,
            sel.get("why").and_then(|v| v.as_str()).unwrap_or(""),
            label,
            sel.get("chosen_run_id").and_then(|v| v.as_str()).unwrap_or(""),
            sel.get("experience"),
        )?;
    }
    Ok(())
}

/// Fill `res` from a cached summary covering every requested mode; true if it did.
#[allow(clippy::too_many_arguments)]
fn resume(summary_path: &Path, task_id: &str, k: usize, adapter: &dyn BenchmarkAdapter, runs_root: &Path, modes: &[&str], seed: Option<i64>, tag: &str, libs: &BTreeMap<String, Library>, res: &mut TaskResult) -> Result<bool> {
    let got: Value = match std::fs::read_to_string(summary_path).map_err(anyhow::Error::from).and_then(|s| Ok(serde_json::from_str(&s)?)) {
        Ok(v) => v,
        Err(e) => {
            say!("  [{task_id}] unreadable summary.json ({e}); re-running");
            return Ok(false);
        }
    };
    let st = got.get("status").and_then(|s| s.as_str()).unwrap_or("");
    if st != "ok" && st != "partial" {
        return Ok(false);
    }
    if let (Some(rec), Some(cur)) = (got.get("inputs").and_then(|v| v.as_str()), res.inputs.as_deref()) {
        if rec != cur {
            say!("  [{task_id}] task inputs {rec} != {cur} -- the task prompt or its source files changed since this summary was written; re-running");
            return Ok(false);
        }
    }
    let cached_modes = modes_of(&got);
    let cached_sels = got.get("selections").and_then(|s| s.as_object()).cloned().unwrap_or_default();
    let mut missing: Vec<&str> = modes
        .iter()
        .filter(|m| !(cached_modes.get(**m).and_then(|v| v.get("ok")).and_then(|o| o.as_bool()).unwrap_or(false) && sel_done(cached_sels.get(**m))))
        .cloned()
        .collect();
    missing.sort();
    if !missing.is_empty() {
        let mut have: Vec<&String> = cached_modes.keys().collect();
        have.sort();
        say!("  [{task_id}] cached summary covers {have:?}, re-running for {missing:?}");
        return Ok(false);
    }
    for mode in modes {
        let sel = &cached_sels[*mode];
        let n = Namespace::new(&adapter.spec().name, adapter.substrate_name(), mode, k, seed, tag);
        let label = sel["chosen_label"].as_str().unwrap_or("");
        let harness_dir = n.task_dir(runs_root, task_id).join("harnesses").join(label);
        let rds = sel.get("chosen_run_dir").and_then(|v| v.as_str()).unwrap_or("");
        let run_dir = PathBuf::from(rds);
        let check: Result<()> = (|| {
            if rds.is_empty() || !run_dir.is_dir() {
                bail!("adopted run directory is missing");
            }
            let artifact = if sel.get("evidence").and_then(|v| v.as_str()) == Some("trajectory") && adapter.spec().state_deliverable {
                adapter.trajectory(&run_dir, adapter.spec().obs_trunc)?
            } else if adapter.delivered(&run_dir) {
                adapter.deliverable_text(&run_dir)?
            } else {
                String::new()
            };
            if crate::core::context::is_absent_evidence(&artifact) {
                bail!("adopted output is missing");
            }
            if adapter.artifacts_intact(&run_dir) == Some(false) {
                bail!("adopted execution artifacts have changed");
            }
            if !tree::is_tree(&adapter.surface(), &harness_dir) {
                bail!("adopted harness is missing");
            }
            if let Some(fp) = sel.get("harness_fingerprint").and_then(|v| v.as_str()).filter(|s| !s.is_empty()) {
                if runmark::fingerprint(Some(&harness_dir)) != fp {
                    bail!("adopted harness has changed");
                }
            }
            Ok(())
        })();
        if let Err(e) = check {
            say!("  [{task_id}] cannot resume {mode}: {e}; re-running");
            return Ok(false);
        }
    }
    res.status = "skipped".into();
    res.base = got.get("base").and_then(|b| b.as_array()).cloned().unwrap_or_default();
    res.modes = cached_modes;
    res.selections = cached_sels;
    res.errors = got
        .get("errors")
        .and_then(|e| e.as_array())
        .map(|a| a.iter().map(|x| x.as_str().map(|s| s.to_string()).unwrap_or_else(|| x.to_string())).collect())
        .unwrap_or_default();
    restore_libraries(task_id, k, adapter, runs_root, modes, seed, tag, libs, &res.selections)?;
    Ok(true)
}

fn dead(label: &str, rid: &str, dead_dir: &Path, error: &str) -> RunView {
    RunView::new(label, rid, dead_dir.to_path_buf(), false, false, error)
}

/// Roll the pending (label, run_id) reference executions, natively batched when possible.
fn roll_bases(task_id: &str, adapter: &dyn BenchmarkAdapter, pending: &[(String, String)], dead_dir: &Path, width: usize) -> BTreeMap<String, RunView> {
    let mut rolled = BTreeMap::new();
    if pending.is_empty() {
        return rolled;
    }
    if pending.len() > 1 {
        let rids: Vec<String> = pending.iter().map(|(_, r)| r.clone()).collect();
        if let Some(batch) = adapter.execute_many(task_id, None, &rids) {
            match batch {
                Ok(dirs) => {
                    for (i, (label, rid)) in pending.iter().enumerate() {
                        let rv = match dirs.get(i) {
                            None => dead(label, rid, dead_dir, "batch adapter returned too few run directories"),
                            Some(rd) => {
                                let exists = rd.exists();
                                RunView::new(label, rid, rd.clone(), exists, exists && adapter.delivered(rd), if exists { "" } else { "adapter returned a missing run directory" })
                            }
                        };
                        rolled.insert(label.clone(), rv);
                    }
                }
                Err(e) => {
                    for (label, rid) in pending {
                        rolled.insert(label.clone(), dead(label, rid, dead_dir, &format!("{e}")));
                    }
                }
            }
            return rolled;
        }
    }
    let width = pending.len().min(width.max(1));
    let got = util::par_map(pending.to_vec(), width, |(label, rid)| {
        crate::log::at(&label, "run");
        crate::log::event("rollout", "start", json!({"run_id": rid}));
        say!("  [{label}] base rollout {rid}");
        match recursion::execute_healing(adapter, task_id, None, &rid) {
            Ok(rd) => {
                let d = adapter.delivered(&rd);
                say!("  [{label}] base rollout done (delivered={d})");
                crate::log::event("rollout", "done", recursion::rollout_info(&rd, d));
                RunView::new(&label, &rid, rd, true, d, "")
            }
            Err(e) => {
                say!("  [{label}] base rollout failed: {e}");
                crate::log::event("rollout", "failed", json!({"error": util::take_chars(&format!("{e}"), 600)}));
                dead(&label, &rid, dead_dir, &format!("{e}"))
            }
        }
    });
    for rv in got {
        rolled.insert(rv.label.clone(), rv);
    }
    rolled
}

#[allow(clippy::too_many_arguments)]
fn run_one_mode(mode: &str, task_id: &str, k: usize, base: &[RunView], adapter: &dyn BenchmarkAdapter, prop: &dyn Propose, selector: &dyn Propose, n: &Namespace, runs_root: &Path, mode_lib: Option<&Library>, base_tree: &Tree, pool_policy: &str, width: usize, res: &mut TaskResult) -> Result<()> {
    let work_dir = n.task_dir(runs_root, task_id);
    let lib_dir = mode_lib.and_then(|l| l.context_dir());
    let nn = n.clone();
    let tid = task_id.to_string();
    let run_id_of = move |lbl: &str| nn.run_id(&tid, lbl);
    let r = recursion::run_mode(
        mode,
        adapter,
        prop,
        ModeArgs {
            task_id,
            k,
            base,
            work_dir: work_dir.clone(),
            run_id_of: &run_id_of,
            base_tree: base_tree.clone(),
            library_dir: lib_dir,
            propose_workers: width.max(1),
            rollout_workers: width.max(1),
        },
    )?;
    res.modes.insert(mode.into(), r.to_json());
    for s in r.stages.iter().filter(|s| s.execution_failed) {
        res.errors.push(format!("{mode}: {}: proposer execution failed: {}", s.label, s.error));
    }
    let mut candidates = selection::selection_pool(mode, k, &r.rollouts, base);
    let guarded = adapter.guard_selection() && guard_pool(&mut candidates, base, adapter, task_id);
    let mut sel: Selection = if pool_policy == "finals" && k == 1 && !(guarded && candidates.len() > 1) {
        selection::trivial_selection(task_id, k, mode, &candidates, adapter, &work_dir)?
    } else {
        selection::run_selection(task_id, k, mode, &candidates, adapter, selector, &work_dir, "finals")?
    };
    res.selections.insert(mode.into(), sel.to_json());
    if mode_lib.is_some() && sel.ok {
        let decisions = r.decisions();
        let out = selection::reflect_adoption(task_id, &mut sel, &r.rollouts, adapter, selector, &work_dir, Some(&decisions));
        res.selections.insert(mode.into(), sel.to_json());
        out?;
    }
    let mut d = sel.to_json();
    d["selected_at"] = json!(util::now_iso());
    d["chosen_run_dir"] = json!(candidates.get(&sel.chosen_label).map(|c| c.run_dir.to_string_lossy().to_string()).unwrap_or_default());
    let chosen_tree = r.trees.get(&sel.chosen_label).cloned().unwrap_or_else(|| base_tree.clone());
    d["chosen_tree_is_base"] = json!(tree_is_base(adapter, base_tree, &chosen_tree));
    if !sel.chosen_label.is_empty() {
        d["harness_fingerprint"] = json!(runmark::fingerprint(Some(&work_dir.join("harnesses").join(&sel.chosen_label))));
    }
    res.selections.insert(mode.into(), d);
    if !sel.ok {
        res.errors.push(format!("{mode}: selection failed: {}", sel.error));
    }
    if let Some(chosen) = candidates.get(&sel.chosen_label) {
        if sel.ok && !chosen.delivered {
            res.errors.push(format!("{mode}: selected run {} produced no deliverable", chosen.run_id));
        }
    }
    if let Some(lib) = mode_lib {
        let ev = sel.experience_value();
        if sel.ok && library::has_experience(Some(&ev)) {
            lib.add(task_id, &chosen_tree, &sel.why, &sel.chosen_label, &sel.chosen_run_id, Some(&ev))?;
        }
    }
    Ok(())
}

/// Phrases that mark a deliverable as a failure report rather than an answer.
static FAILURE_TEXT: once_cell::sync::Lazy<regex::Regex> = once_cell::sync::Lazy::new(|| {
    regex::Regex::new(r#"(?i)(\bno\s+[`'\w./-]+\s+(file\s+)?(was\s+)?found\b|\bnot\s+found\b|does\s+not\s+exist|unable\s+to\s+(find|locate|access|read|open)|could\s+not\s+(find|locate|read|open)|please\s+(ensure|provide|upload|make\s+sure)|"status"\s*:\s*"error"|^\W*error\b|no\s+sub-agent\s+named|no\s+skill\s+named|unknown\s+tool|traceback\s+\(most\s+recent|\bi\s+(can't|cannot|am\s+unable\s+to)\b)"#).unwrap()
});

/// Whether a delivered output reads like "I could not do it": short, and saying so.
pub fn looks_failed(text: &str) -> bool {
    let t = text.trim();
    t.chars().count() < 12 || (t.chars().count() < 600 && FAILURE_TEXT.is_match(t))
}

/// The regression guard: add the base attempt as a candidate, drop failure-looking finals while
/// a healthy candidate remains. Returns whether the pool changed.
/// File types the goal asks for ("an HTML page", "a CSV"), as lowercase extensions.
pub fn expected_exts(prompt: &str) -> Vec<&'static str> {
    let p = prompt.to_lowercase();
    let mut v = vec![];
    for (ext, words) in [
        ("html", &["html", "web page", "webpage", "landing page", "website"][..]),
        ("csv", &["csv"][..]),
        ("json", &["json"][..]),
        ("py", &["python script", ".py"][..]),
        ("svg", &["svg"][..]),
        ("js", &["javascript", ".js"][..]),
        ("css", &["stylesheet", ".css"][..]),
        ("xlsx", &["xlsx", "excel"][..]),
        ("docx", &["docx", "word document"][..]),
        ("pdf", &["pdf"][..]),
    ] {
        if words.iter().any(|w| p.contains(w)) {
            v.push(ext);
        }
    }
    v
}

fn guard_pool(cands: &mut BTreeMap<String, RunView>, base: &[RunView], adapter: &dyn BenchmarkAdapter, task_id: &str) -> bool {
    let text = |rv: &RunView| adapter.deliverable_text(&rv.run_dir).unwrap_or_default();
    let want = expected_exts(&adapter.task_prompt(task_id).unwrap_or_default());
    let on_format = |rv: &RunView| {
        want.is_empty()
            || adapter.deliverable_files(&rv.run_dir).map(|m| m.keys().any(|f| want.iter().any(|e| f.to_lowercase().ends_with(&format!(".{e}"))))).unwrap_or(false)
    };
    let mut added = vec![];
    if let Some(b) = base.iter().find(|b| b.delivered && b.ok && !cands.contains_key(&b.label) && !looks_failed(&text(b))) {
        cands.insert(b.label.clone(), b.clone());
        added.push(b.label.clone());
    }
    let mut suspect: Vec<String> = cands.iter().filter(|(_, rv)| !rv.delivered || looks_failed(&text(rv))).map(|(l, _)| l.clone()).collect();
    // Of the healthy ones, those without the requested file type lose to those with it.
    let healthy: Vec<&String> = cands.keys().filter(|l| !suspect.contains(l)).collect();
    let off: Vec<String> = healthy.iter().filter(|l| !on_format(&cands[**l])).map(|l| (*l).clone()).collect();
    if !off.is_empty() && off.len() < healthy.len() {
        suspect.extend(off);
    }
    let dropped: Vec<String> = if suspect.len() < cands.len() { suspect } else { vec![] };
    for l in &dropped {
        cands.remove(l);
    }
    if !added.is_empty() || !dropped.is_empty() {
        crate::log::at("select", "select");
        crate::log::event("select", "guard", json!({"added": added, "dropped": dropped}));
        say!("  [guard] selection pool: added {:?}, dropped failure-looking {:?}", added, dropped);
        return true;
    }
    false
}

/// Options shared by the task, domain and sweep drivers.
#[derive(Clone)]
pub struct RunOpts {
    pub modes: Vec<&'static str>,
    pub seed: Option<i64>,
    pub tag: String,
    pub selector: Option<Arc<dyn Propose>>,
    pub pool_policy: String,
    /// How many stages / rollouts run at once (the reference uses 2). Use 1 with a single
    /// local model so its prompt cache is reused across an agent's turns.
    pub width: usize,
}

impl Default for RunOpts {
    fn default() -> Self {
        RunOpts { modes: MODES.to_vec(), seed: None, tag: String::new(), selector: None, pool_policy: "finals".into(), width: ROLLOUT_PARALLEL }
    }
}

/// k shared reference executions -> every mode -> blind selection per mode -> summary.json.
pub fn run_task(task_id: &str, k: usize, adapter: &dyn BenchmarkAdapter, prop: &Arc<dyn Propose>, runs_root: &Path, libs: &BTreeMap<String, Library>, o: &RunOpts) -> Result<TaskResult> {
    if k < 1 {
        bail!("k must be positive, got {k}");
    }
    let uniq: HashSet<&&str> = o.modes.iter().collect();
    if o.modes.is_empty() || uniq.len() != o.modes.len() {
        bail!("modes must contain at least one distinct recursion mode");
    }
    if o.modes.iter().any(|m| !MODES.contains(m)) {
        bail!("unknown modes {:?}; expected a subset of {MODES:?}", o.modes);
    }
    if o.pool_policy != "finals" {
        bail!("unknown pool_policy {:?}; this release builds exactly one selection pool, 'finals' (the k composed finals)", o.pool_policy);
    }
    let selector = selector_for(adapter, prop, o.selector.clone())?;
    let (bench, sub) = (adapter.spec().name.clone(), adapter.substrate_name().to_string());
    let domain = adapter.domain_of(task_id);
    let mut res = TaskResult {
        task_id: task_id.into(),
        domain,
        k,
        seed: o.seed,
        status: "running".into(),
        inputs: None,
        modes: Map::new(),
        selections: Map::new(),
        base: vec![],
        errors: vec![],
    };
    let base_ns = Namespace::new(&bench, &sub, "agg", k, o.seed, &o.tag);
    let work = base_ns.task_dir(runs_root, task_id);
    std::fs::create_dir_all(&work)?;
    let summary_path = work.join("summary.json");
    res.inputs = task_inputs(adapter, task_id);
    crate::log::set_mode("");
    crate::log::at("", "");
    crate::log::event("task", "start", json!({"task_id": task_id, "k": k, "modes": o.modes}));
    if summary_path.is_file() && resume(&summary_path, task_id, k, adapter, runs_root, &o.modes, o.seed, &o.tag, libs, &mut res)? {
        say!("  [{task_id}] resumed from summary.json");
        crate::log::set_mode("");
        crate::log::at("", "");
        crate::log::event("task", "done", json!({"status": "skipped"}));
        return Ok(res);
    }
    let surface = adapter.surface();
    let base_tree = tree::read_tree(&surface, &adapter.base_harness()?, None);
    let labels: Vec<String> = (1..=k).map(|i| format!("base{i}")).collect();
    let pending: Vec<(String, String)> = labels.iter().map(|l| (l.clone(), base_ns.run_id(task_id, l))).collect();
    say!("[{task_id}] rolling {k} base execution(s)");
    let rolled = roll_bases(task_id, adapter, &pending, &work.join("missing"), o.width);
    let base: Vec<RunView> = labels.iter().map(|l| rolled[l].clone()).collect();
    res.base = base.iter().map(|b| json!({"label": b.label, "run_id": b.run_id, "run_dir": b.run_dir.to_string_lossy()})).collect();
    if !base.iter().any(|b| b.ok) {
        let err = "no base rollout produced anything";
        res.status = "base-failed".into();
        res.errors.push(err.into());
        for m in &o.modes {
            res.modes.insert((*m).into(), json!({"ok": false, "error": err}));
            let mut s = Selection::new(task_id, m, k);
            s.error = err.into();
            res.selections.insert((*m).into(), s.to_json());
        }
        write_summary(&summary_path, &res)?;
        crate::log::event("task", "done", json!({"status": res.status, "errors": res.errors}));
        return Ok(res);
    }
    for mode in &o.modes {
        let n = Namespace::new(&bench, &sub, mode, k, o.seed, &o.tag);
        if let Err(e) = run_one_mode(mode, task_id, k, &base, adapter, prop.as_ref(), selector.as_ref(), &n, runs_root, libs.get(*mode), &base_tree, &o.pool_policy, o.width, &mut res) {
            let err = format!("{e}");
            say!("  [{task_id}] {mode} failed: {err}");
            // Add the error to the record run_mode already wrote, rather than replacing it, so the
            // stages and their spend stay visible to the report (upstream #98).
            let mut rec = res.modes.get(*mode).cloned().filter(|v| v.is_object()).unwrap_or_else(|| json!({}));
            rec["ok"] = json!(false);
            rec["error"] = json!(err);
            rec["traceback"] = json!(util::tail_chars(&format!("{e:?}"), 1500));
            res.modes.insert((*mode).into(), rec);
            let usd = res.selections.get(*mode).and_then(|s| s.get("usd")).and_then(|u| u.as_f64()).unwrap_or(0.0);
            let mut s = Selection::new(task_id, mode, k);
            s.error = err.clone();
            s.usd = usd;
            res.selections.insert((*mode).into(), s.to_json());
            res.errors.push(format!("{mode}: {err}"));
        }
        res.status = "partial".into();
        write_summary(&summary_path, &res)?;
    }
    res.status = merged_status(&res.to_json());
    write_summary(&summary_path, &res)?;
    crate::log::set_mode("");
    crate::log::at("", "");
    crate::log::event("task", "done", json!({"status": res.status, "errors": res.errors}));
    Ok(res)
}

/// Every task in one domain, in `stream_order()` order, threading the library forward.
pub fn run_domain(domain: &str, task_ids: &[String], k: usize, adapter: &dyn BenchmarkAdapter, prop: &Arc<dyn Propose>, runs_root: &Path, library_root: &Path, o: &RunOpts) -> Value {
    let surface = adapter.surface();
    let libs: BTreeMap<String, Library> =
        o.modes.iter().map(|m| (m.to_string(), Library::new(library_root.join(m), domain, surface.clone()))).collect();
    let (mut done, mut failed) = (0, 0);
    let mut rows = vec![];
    for (i, tid) in task_ids.iter().enumerate() {
        let i = i + 1;
        let row: Value = (|| -> Result<Value> {
            for lib in libs.values() {
                lib.retain(&task_ids[..i - 1])?;
            }
            let before: Map<String, Value> = libs.iter().map(|(m, l)| (m.clone(), json!(l.entries().len()))).collect();
            let r = run_task(tid, k, adapter, prop, runs_root, &libs, o)?;
            Ok(if r.status == "ok" || r.status == "skipped" {
                json!({"task": tid, "status": r.status, "library_entries_before": before})
            } else {
                json!({"task": tid, "status": r.status, "error": if r.errors.is_empty() { r.status.clone() } else { r.errors.join("; ") }, "library_entries_before": before})
            })
        })()
        .unwrap_or_else(|e| json!({"task": tid, "status": "exception", "error": format!("{e}")}));
        let st = row["status"].as_str().unwrap_or("");
        if st == "ok" || st == "skipped" {
            done += 1;
        } else {
            failed += 1;
        }
        let err = row.get("error").and_then(|e| e.as_str()).map(|e| format!("  :: {e}")).unwrap_or_default();
        say!("  [{domain}] {i}/{} {st:<12} {tid}{err}", task_ids.len());
        rows.push(row);
    }
    json!({"domain": domain, "n_tasks": task_ids.len(), "seed": o.seed, "done": done, "failed": failed, "tasks": rows})
}

/// Resolve the ordered task stream for execution, previews, and resume identity.
/// Domains come back in stream order.
pub fn task_stream(adapter: &dyn BenchmarkAdapter, split: &str, seed: Option<i64>, only_domains: &[String], only_tasks: &[String], max_tasks: Option<usize>) -> Result<Vec<(String, Vec<String>)>> {
    let mut by_domain: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for t in adapter.tasks(split)? {
        by_domain.entry(adapter.domain_of(&t)).or_default().push(t);
    }
    if !only_domains.is_empty() {
        let mut missing: Vec<&String> = only_domains.iter().filter(|d| !by_domain.contains_key(*d)).collect();
        missing.sort();
        if !missing.is_empty() {
            bail!("only_domains names unknown domain(s) {missing:?}; split {split:?} has {:?}", by_domain.keys().collect::<Vec<_>>());
        }
        by_domain.retain(|d, _| only_domains.contains(d));
    }
    if !only_tasks.is_empty() {
        let all: HashSet<&String> = by_domain.values().flatten().collect();
        let mut missing: Vec<&String> = only_tasks.iter().filter(|t| !all.contains(t)).collect();
        missing.sort();
        if !missing.is_empty() {
            bail!("only_tasks names unknown task id(s) {missing:?} for split {split:?}");
        }
    }
    if max_tasks == Some(0) {
        bail!("max_tasks must be positive, got 0");
    }
    let order = reference_domain_order(adapter, &by_domain);
    let stream = stream_order(&by_domain, seed, &order)?;
    let mut ordered: Vec<(String, Vec<String>)> = vec![];
    for (d, t) in stream {
        if !only_tasks.is_empty() && !only_tasks.contains(&t) {
            continue;
        }
        let idx = match ordered.iter().position(|(od, _)| *od == d) {
            Some(i) => i,
            None => {
                ordered.push((d.clone(), vec![]));
                ordered.len() - 1
            }
        };
        if max_tasks.map(|m| ordered[idx].1.len() >= m).unwrap_or(false) {
            continue;
        }
        ordered[idx].1.push(t);
    }
    Ok(ordered)
}

pub struct SweepOpts {
    pub split: String,
    pub domain_parallel: usize,
    pub only_domains: Vec<String>,
    pub max_tasks: Option<usize>,
    pub only_tasks: Vec<String>,
}

/// Run domains concurrently, preserving task order after stream filters are applied.
pub fn run_sweep(k: usize, adapter: &dyn BenchmarkAdapter, prop: &Arc<dyn Propose>, runs_root: &Path, library_root: &Path, o: &RunOpts, s: &SweepOpts) -> Result<Value> {
    let ordered = task_stream(adapter, &s.split, o.seed, &s.only_domains, &s.only_tasks, s.max_tasks)?;
    let order: Vec<String> = ordered.iter().map(|(d, _)| d.clone()).collect();
    let ordered: BTreeMap<String, Vec<String>> = ordered.into_iter().collect();
    let started = util::now_iso();
    let selector = selector_for(adapter, prop, o.selector.clone())?;
    let mut o2 = o.clone();
    o2.selector = Some(selector);
    let rows = util::par_map(order.clone(), s.domain_parallel.max(1), |d| {
        run_domain(&d, &ordered[&d], k, adapter, prop, runs_root, library_root, &o2)
    });
    Ok(json!({
        "benchmark": adapter.spec().name, "substrate": adapter.substrate_name(), "k": k,
        "modes": o.modes, "seed": o.seed, "tag": o.tag, "domain_order": order,
        "max_tasks": s.max_tasks, "only_tasks": s.only_tasks, "started_at": started,
        "finished_at": util::now_iso(), "n_domains": order.len(), "domains": rows,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn namespaces() {
        assert_eq!(ns("jb", "oc", "agg", 2, Some(7), "").unwrap(), "jb-oc-agg2-s7");
        assert!(ns("jb", "oc", "agg", 2, None, "s3").is_err());
        assert!(ns("jb", "", "agg", 2, None, "").is_err());
    }
    #[test]
    fn stream_order_matches_cpython() {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), vec!["a1".to_string(), "a2".into(), "a3".into()]);
        m.insert("b".to_string(), vec!["b1".to_string(), "b2".into()]);
        m.insert("c".to_string(), vec!["c1".to_string()]);
        let none = stream_order(&m, None, &default_reference_domain_order(&m)).unwrap();
        assert_eq!(none[0], ("a".to_string(), "a1".to_string()));
        let seeded = stream_order(&m, Some(7), &[]).unwrap();
        let got: Vec<String> = seeded.into_iter().map(|(_, t)| t).collect();
        let want: Vec<String> = serde_json::from_str(include_str!("../../tests/golden/stream7.json")).unwrap();
        assert_eq!(got, want);
    }
}

#[cfg(test)]
mod guard_tests {
    use super::looks_failed;

    #[test]
    fn failure_reports_are_spotted_and_answers_are_not() {
        assert!(looks_failed("No `regions.csv` file found in the input directory. Please ensure the file is present and try again."));
        assert!(looks_failed("Error: the file does not exist."));
        assert!(looks_failed("  "));
        assert!(!looks_failed("| region | q1 | q2 |\n|---|---|---|\n| D | 48 | 42 |\n\nRegion D fell the most: -12.5%."));
        assert!(!looks_failed(&format!("{} not found in the 2019 data, so we used 2020.", "x".repeat(700))), "long answers that mention 'not found' are kept");
        assert!(looks_failed("```\n{\"status\":\"ERROR\",\"data\":{\"error\": \"No sub-agent named 'client'.\"}}\n```"));
        assert!(looks_failed("Error: unknown tool `client`"));
        assert_eq!(super::expected_exts("create a hello world sleek html page"), vec!["html"]);
        assert!(super::expected_exts("write a research report on dogs").is_empty());
    }
}
