//! Parallel and sequential task-level recursion.
//!
//! Each mode schedules 4k rollouts and 3k improvement stages, including shared base
//! executions. Invalid proposals receive one retry before inheriting the current harness.

use super::context::{self, PoolEntry};
use super::ledger::Ledger;
use super::prompts;
use super::proposer::{CallOpts, Propose};
use super::spec::{resolve_prompts, BenchmarkAdapter, RunView};
use super::tree::{self, Tree};
use crate::{config, say, util};
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

pub const STAGE_ATTEMPTS: u32 = 2;
pub const TRANSPORT_ATTEMPTS: u32 = config::TRANSPORT_ATTEMPTS;
pub const PHASE_WIDTH: usize = config::PHASE_WIDTH;

/// An improvement stage, including retries, its harness, and decision notes.
#[derive(Debug, Clone)]
pub struct Stage {
    pub label: String,
    pub dim: String,
    pub tree: Tree,
    pub kept_base: bool,
    pub fell_back: bool,
    pub error: String,
    pub usd: f64,
    pub attempts: u32,
    pub cached: bool,
    pub notes: String,
    pub execution_failed: bool,
}

impl Stage {
    fn fallback(label: &str, dim: &str, tree: &Tree) -> Stage {
        Stage {
            label: label.into(),
            dim: dim.into(),
            tree: tree.clone(),
            kept_base: false,
            fell_back: true,
            error: String::new(),
            usd: 0.0,
            attempts: 0,
            cached: false,
            notes: String::new(),
            execution_failed: false,
        }
    }

    pub fn decision(&self) -> String {
        if self.notes.is_empty() { self.error.clone() } else { self.notes.clone() }
    }
}

#[derive(Debug, Clone)]
pub struct ModeResult {
    pub mode: String,
    pub task_id: String,
    pub k: usize,
    pub stages: Vec<Stage>,
    pub trees: BTreeMap<String, Tree>,
    pub rollouts: BTreeMap<String, RunView>,
    pub finals: Vec<String>,
    pub ledger: Option<Ledger>,
    pub similarity: Option<Value>,
    pub errors: Vec<String>,
}

impl ModeResult {
    fn new(mode: &str, task_id: &str, k: usize, ledger: Ledger) -> ModeResult {
        ModeResult {
            mode: mode.into(),
            task_id: task_id.into(),
            k,
            stages: vec![],
            trees: BTreeMap::new(),
            rollouts: BTreeMap::new(),
            finals: vec![],
            ledger: Some(ledger),
            similarity: None,
            errors: vec![],
        }
    }

    pub fn ok(&self) -> bool {
        !self.finals.is_empty() && self.errors.is_empty() && !self.stages.iter().any(|s| s.execution_failed)
    }

    pub fn fell_back(&self) -> Vec<String> {
        self.stages.iter().filter(|s| s.fell_back).map(|s| s.label.clone()).collect()
    }

    pub fn decisions(&self) -> BTreeMap<String, String> {
        self.stages.iter().map(|s| (s.label.clone(), s.decision())).collect()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "mode": self.mode,
            "task_id": self.task_id,
            "k": self.k,
            "finals": self.finals,
            "ok": self.ok(),
            "fell_back": self.fell_back(),
            "kept_base": self.stages.iter().filter(|s| s.kept_base).map(|s| s.label.clone()).collect::<Vec<_>>(),
            "stages": self.stages.iter().map(|s| json!({
                "label": s.label, "dim": s.dim, "fell_back": s.fell_back, "kept_base": s.kept_base,
                "execution_failed": s.execution_failed, "attempts": s.attempts, "usd": s.usd,
                "cached": s.cached, "notes": s.notes, "error": util::take_chars(&s.error, 400),
            })).collect::<Vec<_>>(),
            "similarity": self.similarity,
            "ledger": self.ledger.as_ref().map(|l| l.to_json()),
            "errors": self.errors,
        })
    }
}

fn reserved(adapter: &dyn BenchmarkAdapter) -> HashSet<String> {
    let mut r = tree::reserved_default();
    r.extend(adapter.spec().reserved_scripts.iter().cloned());
    r
}

/// Repair then validate a proposed tree: (kept_base, "") if accepted, else the error.
fn accept_tree(adapter: &dyn BenchmarkAdapter, new: &mut Tree, fallback: &Tree, label: &str) -> (bool, String) {
    let surface = adapter.surface();
    let triggered = tree::autotrigger(&surface, new, None);
    if !triggered.is_empty() {
        say!("  [{label}] auto-triggered skill(s): {}", triggered.join(", "));
    }
    let kept = tree::diff_summary(&surface, fallback, new).unchanged();
    let res = tree::validate(&surface, new, adapter.spec().allow_subagents, None, &reserved(adapter))
        .and_then(|_| adapter.validate_tree(new))
        .and_then(|_| context::assert_tree_has_no_grader_verdicts(new));
    match res {
        Ok(()) => (kept, String::new()),
        Err(e) => (kept, e.to_string()),
    }
}

pub struct StageArgs<'a> {
    pub label: String,
    pub dim: String,
    pub job: String,
    pub fallback: Tree,
    pub ctx_dir: PathBuf,
    pub task_id: &'a str,
    pub rollouts: &'a [RunView],
    pub harness_dir: Option<PathBuf>,
    pub library_dir: Option<PathBuf>,
    pub pool: Vec<PoolEntry>,
    pub decisions: Option<BTreeMap<String, String>>,
}

/// Build evidence, propose an edit, and retry an invalid proposal once.
pub fn run_stage(adapter: &dyn BenchmarkAdapter, prop: &dyn Propose, a: StageArgs) -> Stage {
    let surface = adapter.surface();
    crate::log::at(&a.label, "edit");
    crate::log::event("stage", "start", json!({"dim": a.dim}));
    if crate::log::cancelled() {
        let mut s = Stage::fallback(&a.label, &a.dim, &a.fallback);
        s.error = "cancelled by user".into();
        crate::log::event("stage", "fell_back", json!({"dim": a.dim, "error": s.error}));
        return s;
    }
    say!("  [{}] {} stage: building evidence", a.label, a.dim);
    let built = context::build(
        &a.ctx_dir.join(&a.label),
        adapter,
        a.task_id,
        a.rollouts,
        a.harness_dir.as_deref(),
        a.library_dir.as_deref(),
        &a.pool,
        a.decisions.as_ref(),
        None,
    );
    let ctree = match built {
        Ok(t) => t,
        Err(e) => {
            let mut s = Stage::fallback(&a.label, &a.dim, &a.fallback);
            s.error = util::take_chars(&format!("could not build evidence: {e}"), 400);
            say!("  [{}] fell back: {}", a.label, s.error);
            crate::log::event("stage", "fell_back", json!({"dim": a.dim, "error": s.error}));
            return s;
        }
    };
    let mission = context::mission(&ctree, &a.job);
    let mut err = String::new();
    let (mut spent_usd, mut spent_calls, mut all_cached, mut execution_failed) = (0.0, 0u32, true, false);
    for attempt in 1..=STAGE_ATTEMPTS {
        let text = if err.is_empty() {
            mission.clone()
        } else {
            format!("{mission}\n\n# YOUR PREVIOUS ATTEMPT WAS REJECTED\n\n{err}\n\nFix it in `context/harness/` and reply again.")
        };
        say!("  [{}] proposing (attempt {attempt})", a.label);
        let reply = match prop.call(
            &text,
            Some(&ctree.root),
            &format!("{}.a{attempt}", a.label),
            CallOpts { harvest: Some("context/harness".into()), max_attempts: Some(TRANSPORT_ATTEMPTS), ..Default::default() },
        ) {
            Ok(r) => r,
            Err(e) => {
                err = e.to_string();
                execution_failed = true;
                spent_calls += if e.attempts > 0 { e.attempts } else { TRANSPORT_ATTEMPTS };
                spent_usd += e.cost_usd;
                all_cached = false;
                break;
            }
        };
        spent_usd += reply.cost_usd;
        spent_calls += if reply.cached { 0 } else { reply.attempts.max(1) };
        all_cached = all_cached && reply.cached;
        let Some(ws) = &reply.workspace else {
            err = "the harness directory did not come back from the session; edit the files in `context/harness/` in place".into();
            continue;
        };
        let mut rejected = vec![];
        let mut new = tree::read_tree(&surface, ws, Some(&mut rejected));
        if new.is_empty() {
            err = "`context/harness/` came back empty -- edit the files, do not delete them all".into();
            continue;
        }
        if !rejected.is_empty() {
            rejected.sort();
            err = format!(
                "these files are off the editable surface and were not applied: {}. Component names must be lowercase letters, digits, `_` or `-` (e.g. `{}/doc_review/SKILL.md`, `scripts/check_totals.py`).",
                rejected.iter().take(6).cloned().collect::<Vec<_>>().join(", "),
                surface.skill_dir
            );
            // The retry reads a fresh copy of `context/`, so it must hold this attempt's edits.
            let _ = tree::write_tree(&surface, &new, &ctree.root.join("harness"), true);
            continue;
        }
        let (kept, e) = accept_tree(adapter, &mut new, &a.fallback, &a.label);
        if !e.is_empty() {
            err = e;
            say!("  [{}] proposal rejected: {}", a.label, util::take_chars(&err, 240));
            crate::log::event("stage", "retry", json!({"dim": a.dim, "attempt": attempt, "error": util::take_chars(&err, 600)}));
            let _ = tree::write_tree(&surface, &new, &ctree.root.join("harness"), true);
            continue;
        }
        let notes_re = Regex::new(r"(?s)<notes>(.*?)</notes>").unwrap();
        let notes = notes_re
            .captures(&reply.text)
            .map(|c| c[1].trim().to_string())
            .unwrap_or_else(|| reply.text.trim().to_string());
        say!("  [{}] accepted{}", a.label, if kept { " (KEEP_BASE)" } else { "" });
        let changed = tree::diff_summary(&surface, &a.fallback, &new);
        crate::log::event("stage", "done", json!({
            "dim": a.dim, "kept_base": kept, "attempts": spent_calls, "usd": spent_usd, "cached": all_cached,
            "notes": util::take_chars(&notes, 1500), "changed": changed,
        }));
        return Stage {
            label: a.label,
            dim: a.dim,
            tree: new,
            kept_base: kept,
            fell_back: false,
            error: String::new(),
            usd: spent_usd,
            attempts: spent_calls,
            cached: all_cached,
            notes,
            execution_failed: false,
        };
    }
    say!("  [{}] fell back to the inherited harness: {}", a.label, util::take_chars(&err, 240));
    crate::log::event("stage", "fell_back", json!({"dim": a.dim, "error": util::take_chars(&err, 600), "execution_failed": execution_failed}));
    let mut s = Stage::fallback(&a.label, &a.dim, &a.fallback);
    s.error = util::take_chars(&err, 400);
    s.usd = spent_usd;
    s.attempts = spent_calls;
    s.cached = all_cached;
    s.execution_failed = execution_failed;
    s
}

fn record_sessions(ledger: &mut Ledger, stages: &[Stage]) {
    for s in stages {
        ledger.session(&s.label, s.usd, s.cached, s.attempts);
    }
}

pub type RunIdOf<'a> = &'a (dyn Fn(&str) -> String + Sync);

/// Materialise a harness and roll it once. A dead rollout is recorded, never raised.
fn roll(adapter: &dyn BenchmarkAdapter, task_id: &str, label: &str, t: Option<&Tree>, run_id_of: RunIdOf, work_dir: &Path) -> RunView {
    let mut hdir = None;
    if let Some(t) = t {
        match tree::write_tree(&adapter.surface(), t, &work_dir.join("harnesses").join(label), true) {
            Ok(d) => hdir = Some(d),
            Err(e) => return RunView::new(label, &run_id_of(label), work_dir.join("missing"), false, false, &format!("{e}")),
        }
    }
    let run_id = run_id_of(label);
    crate::log::at(label, "run");
    if crate::log::cancelled() {
        crate::log::event("rollout", "failed", json!({"error": "cancelled by user"}));
        return RunView::new(label, &run_id, work_dir.join("missing"), false, false, "cancelled by user");
    }
    crate::log::event("rollout", "start", json!({"run_id": run_id}));
    say!("  [{label}] rollout {run_id}");
    match execute_healing(adapter, task_id, hdir.as_deref(), &run_id) {
        Ok(rd) => {
            let exists = rd.exists();
            let delivered = exists && adapter.delivered(&rd);
            say!("  [{label}] rollout done (delivered={delivered})");
            crate::log::event("rollout", "done", rollout_info(&rd, delivered));
            RunView::new(label, &run_id, rd, exists, delivered, if exists { "" } else { "adapter returned a missing run directory" })
        }
        Err(e) => {
            say!("  [{label}] rollout failed: {e}");
            crate::log::event("rollout", "failed", json!({"error": util::take_chars(&format!("{e}"), 600)}));
            RunView::new(label, &run_id, work_dir.join("missing"), false, false, &format!("{e}"))
        }
    }
}

/// Run one rollout, healing a transient failure: an error that is not a cancellation is
/// retried once after a pause, with a `heal` event, before the rollout counts as dead.
pub fn execute_healing(adapter: &dyn BenchmarkAdapter, task_id: &str, harness_dir: Option<&Path>, run_id: &str) -> Result<PathBuf> {
    match adapter.execute(task_id, harness_dir, run_id, None) {
        Ok(rd) => Ok(rd),
        Err(e) if crate::log::cancelled() || format!("{e}").contains("cancelled") => Err(e),
        Err(e) => {
            let wait = config::env_int("HARNESS2_ROLLOUT_RETRY_WAIT", 10).unwrap_or(10).max(0) as u64;
            crate::log::event("heal", "retry_rollout", json!({"run_id": run_id, "error": util::take_chars(&format!("{e}"), 300), "wait_s": wait}));
            say!("  ! rollout {run_id} failed ({}); retrying once in {wait}s", util::take_chars(&format!("{e}"), 160));
            std::thread::sleep(std::time::Duration::from_secs(wait));
            adapter.execute(task_id, harness_dir, run_id, None)
        }
    }
}

/// What the UI shows about a finished rollout: delivery, and usage when the substrate records it.
pub fn rollout_info(rd: &Path, delivered: bool) -> Value {
    let usage: Value = std::fs::read_to_string(rd.join("usage.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
    json!({"delivered": delivered, "run_dir": rd, "usage": usage})
}

fn check_cancel() -> Result<()> {
    if crate::log::cancelled() {
        bail!("cancelled by user");
    }
    Ok(())
}

/// Roll independent harnesses concurrently, then record them in the caller's order.
fn roll_many(
    adapter: &dyn BenchmarkAdapter,
    ledger: &mut Ledger,
    task_id: &str,
    items: Vec<(String, Tree)>,
    run_id_of: RunIdOf,
    work_dir: &Path,
    width: usize,
) -> BTreeMap<String, RunView> {
    let labels: Vec<String> = items.iter().map(|(l, _)| l.clone()).collect();
    let got = util::par_map(items, width, |(label, t)| roll(adapter, task_id, &label, Some(&t), run_id_of, work_dir));
    let mut out = BTreeMap::new();
    for (label, rv) in labels.into_iter().zip(got) {
        ledger.rollout(&label, 0.0, false);
        out.insert(label, rv);
    }
    out
}

pub struct ModeArgs<'a> {
    pub task_id: &'a str,
    pub k: usize,
    pub base: &'a [RunView],
    pub work_dir: PathBuf,
    pub run_id_of: RunIdOf<'a>,
    pub base_tree: Tree,
    pub library_dir: Option<PathBuf>,
    pub propose_workers: usize,
    pub rollout_workers: usize,
}

fn surface_prompt_name(adapter: &dyn BenchmarkAdapter) -> &'static str {
    adapter.surface().name
}

/// WIDTH: 2k iid candidates from shared evidence, then k INDEPENDENT compositions.
pub fn run_parallel(adapter: &dyn BenchmarkAdapter, prop: &dyn Propose, a: ModeArgs) -> Result<ModeResult> {
    let surface = adapter.surface();
    let sname = surface_prompt_name(adapter);
    let (task_id, k) = (a.task_id, a.k);
    let mut ledger = Ledger::new("agg", task_id, k);
    for r in a.base {
        ledger.rollout(&r.label, 0.0, false);
    }
    let mut result = ModeResult::new("agg", task_id, k, ledger.clone());
    for r in a.base {
        result.rollouts.insert(r.label.clone(), r.clone());
    }
    let ctx_dir = a.work_dir.join("context");
    let bp = resolve_prompts(adapter, task_id);
    let base_dir = tree::write_tree(&surface, &a.base_tree, &a.work_dir.join("harnesses/base"), true)?;
    crate::log::set_mode("agg");
    crate::log::at("", "");
    crate::log::event("mode", "start", json!({"k": k}));
    say!("[parallel k={k}] {task_id}: proposing {} candidates", 2 * k);

    let mut specs = vec![];
    for i in 1..=k {
        for d in ["INV", "DOM"] {
            specs.push((format!("{}{i}", d.to_lowercase()), d, i));
        }
    }
    let jobs: Vec<(String, &str, String)> = specs
        .into_iter()
        .map(|(l, d, i)| prompts::propose(&bp, d, sname, k, i, None).map(|j| (l, d, j)))
        .collect::<Result<_>>()?;
    let stages: Vec<Stage> = util::par_map(jobs, a.propose_workers, |(label, dim, job)| {
        run_stage(
            adapter,
            prop,
            StageArgs {
                label,
                dim: dim.into(),
                job,
                fallback: a.base_tree.clone(),
                ctx_dir: ctx_dir.clone(),
                task_id,
                rollouts: a.base,
                harness_dir: Some(base_dir.clone()),
                library_dir: a.library_dir.clone(),
                pool: vec![],
                decisions: None,
            },
        )
    });
    record_sessions(&mut ledger, &stages);
    for s in &stages {
        result.stages.push(s.clone());
        result.trees.insert(s.label.clone(), s.tree.clone());
    }
    check_cancel()?;
    say!("[parallel k={k}] {task_id}: probing {} candidates", stages.len());
    let rolled = roll_many(
        adapter,
        &mut ledger,
        task_id,
        stages.iter().map(|s| (s.label.clone(), s.tree.clone())).collect(),
        a.run_id_of,
        &a.work_dir,
        a.rollout_workers,
    );
    result.rollouts.extend(rolled);

    let pool_ev: Vec<PoolEntry> = stages
        .iter()
        .map(|s| (s.label.clone(), a.work_dir.join("harnesses").join(&s.label), result.rollouts.get(&s.label).cloned()))
        .collect();
    let decisions: BTreeMap<String, String> = stages.iter().map(|s| (s.label.clone(), s.decision())).collect();
    check_cancel()?;
    say!("[parallel k={k}] {task_id}: composing {k} finals");
    let agg_jobs: Vec<(String, String)> = (1..=k)
        .map(|i| prompts::aggregate(&bp, sname, k, i).map(|j| (format!("agg{i}"), j)))
        .collect::<Result<_>>()?;
    let aggs: Vec<Stage> = util::par_map(agg_jobs, a.propose_workers, |(label, job)| {
        run_stage(
            adapter,
            prop,
            StageArgs {
                label,
                dim: "AGG".into(),
                job,
                fallback: a.base_tree.clone(),
                ctx_dir: ctx_dir.clone(),
                task_id,
                rollouts: a.base,
                harness_dir: Some(base_dir.clone()),
                library_dir: a.library_dir.clone(),
                pool: pool_ev.clone(),
                decisions: Some(decisions.clone()),
            },
        )
    });
    record_sessions(&mut ledger, &aggs);
    for s in &aggs {
        result.stages.push(s.clone());
        result.trees.insert(s.label.clone(), s.tree.clone());
        result.finals.push(s.label.clone());
    }
    let rolled = roll_many(
        adapter,
        &mut ledger,
        task_id,
        aggs.iter().map(|s| (s.label.clone(), s.tree.clone())).collect(),
        a.run_id_of,
        &a.work_dir,
        a.rollout_workers,
    );
    result.rollouts.extend(rolled);
    result.similarity = Some(tree::diversity(&aggs.iter().map(|s| (s.label.clone(), s.tree.clone())).collect()));
    check_cancel()?;
    ledger.assert_identity()?;
    result.ledger = Some(ledger);
    Ok(result)
}

/// DEPTH: k nested passes over the same shared base evidence.
pub fn run_sequential(adapter: &dyn BenchmarkAdapter, prop: &dyn Propose, a: ModeArgs) -> Result<ModeResult> {
    let surface = adapter.surface();
    let sname = surface_prompt_name(adapter);
    let (task_id, k) = (a.task_id, a.k);
    let mut ledger = Ledger::new("seq", task_id, k);
    for r in a.base {
        ledger.rollout(&r.label, 0.0, false);
    }
    let mut result = ModeResult::new("seq", task_id, k, ledger.clone());
    for r in a.base {
        result.rollouts.insert(r.label.clone(), r.clone());
    }
    let ctx_dir = a.work_dir.join("context");
    let bp = resolve_prompts(adapter, task_id);
    let mut current = a.base_tree.clone();
    let mut history: Vec<PoolEntry> = vec![];
    crate::log::set_mode("seq");
    crate::log::at("", "");
    crate::log::event("mode", "start", json!({"k": k}));
    for t in 1..=k {
        check_cancel()?;
        let current_label = if t > 1 { format!("c{}", t - 1) } else { "base".into() };
        let current_dir = tree::write_tree(&surface, &current, &a.work_dir.join("harnesses").join(&current_label), true)?;
        say!("[sequential k={k}] {task_id}: round {t} proposing");
        let decisions = result.decisions();
        let jobs: Vec<(String, &str, String)> = [("inv", "INV"), ("dom", "DOM")]
            .iter()
            .map(|(p, d)| prompts::propose(&bp, d, sname, k, t, if k > 1 { Some(t) } else { None }).map(|j| (format!("{p}{t}"), *d, j)))
            .collect::<Result<_>>()?;
        let stages: Vec<Stage> = util::par_map(jobs, a.propose_workers.min(2), |(label, dim, job)| {
            run_stage(
                adapter,
                prop,
                StageArgs {
                    label,
                    dim: dim.into(),
                    job,
                    fallback: current.clone(),
                    ctx_dir: ctx_dir.clone(),
                    task_id,
                    rollouts: a.base,
                    harness_dir: Some(current_dir.clone()),
                    library_dir: a.library_dir.clone(),
                    pool: history.clone(),
                    decisions: Some(decisions.clone()),
                },
            )
        });
        record_sessions(&mut ledger, &stages);
        for s in &stages {
            result.stages.push(s.clone());
            result.trees.insert(s.label.clone(), s.tree.clone());
        }
        let rolled = roll_many(
            adapter,
            &mut ledger,
            task_id,
            stages.iter().map(|s| (s.label.clone(), s.tree.clone())).collect(),
            a.run_id_of,
            &a.work_dir,
            a.rollout_workers,
        );
        result.rollouts.extend(rolled);
        let mut round_pool = history.clone();
        for s in &stages {
            round_pool.push((s.label.clone(), a.work_dir.join("harnesses").join(&s.label), result.rollouts.get(&s.label).cloned()));
        }
        check_cancel()?;
        let ck = format!("c{t}");
        let comp_job = if k == 1 { prompts::aggregate(&bp, sname, 1, 1)? } else { prompts::compose_round(&bp, sname, t, k)? };
        say!("[sequential k={k}] {task_id}: round {t} composing {ck}");
        let comp = run_stage(
            adapter,
            prop,
            StageArgs {
                label: ck.clone(),
                dim: "COMPOSE".into(),
                job: comp_job,
                fallback: current.clone(),
                ctx_dir: ctx_dir.clone(),
                task_id,
                rollouts: a.base,
                harness_dir: Some(current_dir.clone()),
                library_dir: a.library_dir.clone(),
                pool: round_pool.clone(),
                decisions: Some(result.decisions()),
            },
        );
        record_sessions(&mut ledger, std::slice::from_ref(&comp));
        result.stages.push(comp.clone());
        result.trees.insert(ck.clone(), comp.tree.clone());
        let rv = roll(adapter, task_id, &ck, Some(&comp.tree), a.run_id_of, &a.work_dir);
        ledger.rollout(&ck, 0.0, false);
        result.rollouts.insert(ck.clone(), rv.clone());
        history = round_pool;
        history.push((ck.clone(), a.work_dir.join("harnesses").join(&ck), Some(rv)));
        current = comp.tree;
    }
    result.finals = (1..=k).map(|i| format!("c{i}")).collect();
    check_cancel()?;
    ledger.assert_identity()?;
    result.ledger = Some(ledger);
    Ok(result)
}

/// Dispatch on the on-disk mode token: "agg" (parallel) or "seq" (sequential).
pub fn run_mode(mode: &str, adapter: &dyn BenchmarkAdapter, prop: &dyn Propose, a: ModeArgs) -> Result<ModeResult> {
    match mode {
        "agg" => run_parallel(adapter, prop, a),
        "seq" => run_sequential(adapter, prop, a),
        _ => bail!("unknown mode {mode:?}; the method has exactly two: 'agg' (parallel) and 'seq' (sequential)"),
    }
}
