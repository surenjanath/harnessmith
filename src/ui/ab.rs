//! Experiments on one goal, each a set of solver attempts that are graded and compared:
//!
//! - **A/B test**: the base harness vs the adopted harness, n times each. One run against one
//!   run mostly measures noise; this measures whether adaptation helped.
//! - **Model shootout**: the base harness on several solver models, n times each, to see which
//!   model does this goal best (e.g. a hosted NVIDIA model vs a local one).
//!
//! Results land in the goal's `ab.json` / `shootout.json` as they arrive. Each experiment has
//! its own log sink, so it can be cancelled from the UI.

use crate::benchmarks::goal::{self, GoalAdapter};
use crate::core::spec::BenchmarkAdapter;
use crate::{log as hlog, util};
use anyhow::{bail, Result};
use axum::extract::Path as AxPath;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Running experiments: "<project>/<goal>/<kind>" -> log sink name.
static ACTIVE: once_cell::sync::Lazy<Mutex<std::collections::HashMap<String, String>>> = once_cell::sync::Lazy::new(Default::default);

#[derive(Deserialize, Default)]
pub(crate) struct ExpReq {
    pub n: Option<usize>,
    pub mode: Option<String>,
    /// Shootout: solver model ids to compare.
    pub models: Option<Vec<String>>,
}

fn path_of(a: &GoalAdapter, id: &str, kind: &str) -> PathBuf {
    a.task_dir(id).join(format!("{kind}.json"))
}

/// The adopted harness of a mode, or why there is nothing to compare.
fn adopted_harness(a: &GoalAdapter, id: &str, mode: &str) -> Result<(String, PathBuf)> {
    for d in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
        let Ok(t) = std::fs::read_to_string(d.path().join(id).join("summary.json")) else { continue };
        let v: Value = serde_json::from_str(&t)?;
        let sel = &v["selections"][mode];
        let label = sel["chosen_label"].as_str().unwrap_or("").to_string();
        if label.is_empty() {
            continue;
        }
        if sel["chosen_is_base"].as_bool().unwrap_or(false) || label.starts_with("base") {
            bail!("the adopted output came from the base harness, so both sides would be the same harness. Run the goal again (or with k=2) to get an adapted harness to test");
        }
        let ns = d.file_name().to_string_lossy().replacen("-agg", &format!("-{mode}"), 1);
        let dir = a.root.join("runs/adapt").join(ns).join(id).join("harnesses").join(&label);
        if dir.is_dir() {
            return Ok((label, dir));
        }
    }
    bail!("no adopted harness found for this goal yet; finish a run first")
}

/// One graded attempt: delivered, review problems, rubric score (when there is a rubric), time, tokens.
fn score_row(a: &GoalAdapter, id: &str, i: usize, run_dir: &Path, secs: f64, err: Option<String>) -> Value {
    if let Some(e) = err {
        return json!({"i": i, "error": e, "seconds": secs});
    }
    let out = run_dir.join("workspace/output");
    let delivered = util::files_sorted(&out).iter().any(|p| std::fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false));
    let issues = if out.is_dir() { crate::llm::review::review(&a.load_task(id).map(|t| t.prompt).unwrap_or_default(), &out) } else { vec![] };
    let usage: Value = std::fs::read_to_string(run_dir.join("usage.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
    let mut row = json!({"i": i, "delivered": delivered, "issues": issues.len(), "issue_list": issues.iter().take(3).collect::<Vec<_>>(), "seconds": secs.round(),
        "tokens": usage["input_tokens"].as_u64().unwrap_or(0) + usage["output_tokens"].as_u64().unwrap_or(0), "usd": usage["cost_usd"], "run_dir": run_dir});
    let graded = a.load_task(id).map(|t| !t.rubric.trim().is_empty()).unwrap_or(false) && !a.settings.judge_model.trim().is_empty();
    if graded {
        match a.score(run_dir, id) {
            Ok(Some(v)) => {
                let (p, n) = (v["n_passed"].as_f64().unwrap_or(0.0), v["n_criteria"].as_f64().unwrap_or(0.0));
                row["passed"] = json!(p);
                row["criteria"] = json!(n);
                row["frac"] = json!(if n > 0.0 { p / n } else { 0.0 });
            }
            Ok(None) => row["error"] = json!("judge returned nothing"),
            Err(e) => row["error"] = json!(format!("judge: {e}")),
        }
    }
    row
}

fn summarize(rows: &[Value]) -> Value {
    let f: Vec<f64> = rows.iter().filter_map(|r| r["frac"].as_f64()).collect();
    let n = rows.len().max(1) as f64;
    let delivered = rows.iter().filter(|r| r["delivered"] == true).count();
    let secs = rows.iter().filter_map(|r| r["seconds"].as_f64()).sum::<f64>() / n;
    let toks = rows.iter().filter_map(|r| r["tokens"].as_f64()).sum::<f64>() / n;
    let issues = rows.iter().filter_map(|r| r["issues"].as_f64()).sum::<f64>() / n;
    let mut s = json!({"runs": rows.len(), "delivered": delivered, "avg_seconds": secs.round(), "avg_tokens": toks.round(), "avg_issues": (issues * 10.0).round() / 10.0});
    if !f.is_empty() {
        let mean = f.iter().sum::<f64>() / f.len() as f64;
        let sd = (f.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / f.len() as f64).sqrt();
        s["n"] = json!(f.len());
        s["mean"] = json!(mean);
        s["sd"] = json!(sd);
    }
    s
}

/// Run `jobs` (group label, attempt index, solver model, harness) in the background, writing
/// rows into `state[group]` as they finish; `finish` adds the summary at the end.
fn launch(a: GoalAdapter, id: String, kind: &'static str, key: String, sink_name: String, state: Arc<Mutex<Value>>, jobs: Vec<(String, usize, String, Option<PathBuf>)>, finish: fn(&mut Value)) {
    let sink = hlog::named(&sink_name);
    let a = Arc::new(a);
    std::thread::spawn(move || {
        let width = a.settings.parallelism.max(1);
        let stamp = sink_name.clone();
        util::par_map(jobs, width, |(group, i, model, harness)| {
            hlog::set_current(Some(sink.clone()));
            if sink.is_cancelled() {
                return;
            }
            hlog::line(format!("[{kind}] {group} #{i} on {model}: starting"));
            let mut ad = GoalAdapter::open(&a.root).expect("project reopens");
            ad.settings.solver_model = model.clone();
            let run_id = format!("{stamp}/{id}/{}{i}", util::slug_safe(&group));
            let t0 = Instant::now();
            let res = ad.execute(&id, harness.as_deref(), &run_id, None);
            let secs = t0.elapsed().as_secs_f64();
            let row = match res {
                Ok(rd) => score_row(&ad, &id, i, &rd, secs, None),
                Err(e) => score_row(&ad, &id, i, Path::new(""), secs, Some(format!("{e}"))),
            };
            hlog::line(format!("[{kind}] {group} #{i}: {}", row["frac"].as_f64().map(|f| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| if row["delivered"] == true { "delivered".into() } else { "no output".into() })));
            let mut st = state.lock().unwrap();
            if let Some(arr) = st["groups"][group.as_str()].as_array_mut() {
                arr.push(row);
            }
            let _ = util::write(&path_of(&a, &id, kind), serde_json::to_string_pretty(&*st).unwrap_or_default());
        });
        let mut st = state.lock().unwrap();
        let groups: Vec<String> = st["groups"].as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default();
        for g in groups {
            let s = summarize(st["groups"][&g].as_array().map(|v| v.as_slice()).unwrap_or(&[]));
            st["summary"][&g] = s;
        }
        finish(&mut st);
        st["status"] = json!(if sink.is_cancelled() { "cancelled" } else { "done" });
        st["finished"] = json!(util::now_iso());
        let _ = util::write(&path_of(&a, &id, kind), serde_json::to_string_pretty(&*st).unwrap_or_default());
        ACTIVE.lock().unwrap().remove(&key);
    });
}

fn begin(p: &str, id: &str, kind: &str) -> Result<(String, String)> {
    let key = format!("{p}/{id}/{kind}");
    let mut act = ACTIVE.lock().unwrap();
    if act.contains_key(&key) {
        bail!("this experiment is already running for this goal");
    }
    let sink = format!("{kind}-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
    act.insert(key.clone(), sink.clone());
    Ok((key, sink))
}

fn ab_verdict(st: &mut Value) {
    let (b, ad) = (&st["summary"]["base"], &st["summary"]["adopted"]);
    let v = match (b["mean"].as_f64(), ad["mean"].as_f64(), b["sd"].as_f64(), ad["sd"].as_f64()) {
        (Some(bm), Some(am), Some(bs), Some(as_)) => {
            let noise = (bs + as_).max(0.05);
            if am - bm > noise { "adapted harness is better" } else if bm - am > noise { "adapted harness is worse" } else { "no clear difference (within run-to-run spread)" }
        }
        _ => "not enough graded runs",
    };
    st["summary"]["verdict"] = json!(v);
}

fn shootout_verdict(st: &mut Value) {
    let s = st["summary"].as_object().cloned().unwrap_or_default();
    // Best by mean score when graded, else by delivered runs then fewest review problems.
    let best = s.iter().max_by(|(_, x), (_, y)| {
        let key = |v: &Value| (v["mean"].as_f64().unwrap_or(-1.0), v["delivered"].as_f64().unwrap_or(0.0), -v["avg_issues"].as_f64().unwrap_or(99.0));
        key(x).partial_cmp(&key(y)).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some((m, _)) = best {
        st["best"] = json!(m);
    }
}

pub(crate) async fn start(AxPath((p, id)): AxPath<(String, String)>, body: Option<Json<ExpReq>>) -> super::ApiResult {
    let req = body.map(|b| b.0).unwrap_or_default();
    super::blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let t = a.load_task(&id)?;
        if t.rubric.trim().is_empty() || a.settings.judge_model.trim().is_empty() {
            bail!("an A/B test needs grading criteria on the goal and a judge model (Models → Roles)");
        }
        let mode = req.mode.unwrap_or_else(|| "agg".into());
        let n = req.n.unwrap_or(3).clamp(1, 10);
        let (label, harness) = adopted_harness(&a, &id, &mode)?;
        let (key, sink) = begin(&p, &id, "ab")?;
        let solver = a.settings.solver_model.clone();
        let state = Arc::new(Mutex::new(json!({"status": "running", "n": n, "mode": mode, "adopted_label": label, "started": util::now_iso(), "log": sink, "groups": {"base": [], "adopted": []}, "summary": {}})));
        util::write(&path_of(&a, &id, "ab"), serde_json::to_string_pretty(&*state.lock().unwrap())?)?;
        let jobs = (1..=n).flat_map(|i| [("base".to_string(), i, solver.clone(), None), ("adopted".to_string(), i, solver.clone(), Some(harness.clone()))]).collect();
        launch(a, id, "ab", key, sink, state, jobs, ab_verdict);
        Ok(json!({"started": true, "n": n}))
    })
    .await
}

pub(crate) async fn start_shootout(AxPath((p, id)): AxPath<(String, String)>, Json(req): Json<ExpReq>) -> super::ApiResult {
    super::blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        a.load_task(&id)?;
        let mut models: Vec<String> = req.models.unwrap_or_default().into_iter().map(|m| m.trim().to_string()).filter(|m| !m.is_empty()).collect();
        models.dedup();
        if models.len() < 2 {
            bail!("pick at least two solver models to compare");
        }
        for m in &models {
            let (c, _) = crate::llm::connections::resolve(m.split(',').next().unwrap_or(""))?;
            if let Some(why) = c.missing_key() {
                bail!("{m}: {why}");
            }
        }
        let n = req.n.unwrap_or(2).clamp(1, 5);
        let (key, sink) = begin(&p, &id, "shootout")?;
        let groups: serde_json::Map<String, Value> = models.iter().map(|m| (m.clone(), json!([]))).collect();
        let graded = !a.settings.judge_model.trim().is_empty() && a.load_task(&id).map(|t| !t.rubric.trim().is_empty()).unwrap_or(false);
        let state = Arc::new(Mutex::new(json!({"status": "running", "n": n, "models": models, "graded": graded, "started": util::now_iso(), "log": sink, "groups": groups, "summary": {}})));
        util::write(&path_of(&a, &id, "shootout"), serde_json::to_string_pretty(&*state.lock().unwrap())?)?;
        let jobs = (1..=n).flat_map(|i| models.iter().map(move |m| (m.clone(), i, m.clone(), None))).collect();
        launch(a, id, "shootout", key, sink, state, jobs, shootout_verdict);
        Ok(json!({"started": true, "n": n}))
    })
    .await
}

fn read(p: &str, id: &str, kind: &str) -> Result<Value> {
    let a = GoalAdapter::open(&goal::project_dir(p)?)?;
    let mut v: Value = std::fs::read_to_string(path_of(&a, id, kind)).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({"status": "none"}));
    // A "running" file whose experiment is not in this process was cut short by a restart.
    if v["status"] == "running" && !ACTIVE.lock().unwrap().contains_key(&format!("{p}/{id}/{kind}")) {
        v["status"] = json!("interrupted");
    }
    if let Some(log) = v["log"].as_str() {
        let lines = hlog::named(log).lines_from(0);
        v["log_tail"] = json!(lines.iter().rev().take(8).rev().cloned().collect::<Vec<_>>());
    }
    Ok(v)
}

pub(crate) async fn get(AxPath((p, id)): AxPath<(String, String)>) -> super::ApiResult {
    super::blocking(move || read(&p, &id, "ab")).await
}

pub(crate) async fn get_shootout(AxPath((p, id)): AxPath<(String, String)>) -> super::ApiResult {
    super::blocking(move || read(&p, &id, "shootout")).await
}

#[derive(Deserialize)]
pub(crate) struct CancelReq {
    kind: String,
}

/// Stop a running experiment: running attempts end at their next step; queued ones never start.
pub(crate) async fn cancel(AxPath((p, id)): AxPath<(String, String)>, Json(b): Json<CancelReq>) -> super::ApiResult {
    super::blocking(move || {
        let sink = ACTIVE.lock().unwrap().get(&format!("{p}/{id}/{}", b.kind)).cloned();
        match sink {
            Some(s) => {
                hlog::named(&s).cancel();
                Ok(json!({"cancelled": true}))
            }
            None => bail!("nothing is running"),
        }
    })
    .await
}
