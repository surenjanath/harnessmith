//! `judge`: grade the SELECTED run of each finished task in one namespace, and cache the rows.
//! `report`: render the score report for one cell from cached judged.json rows — read-only.
//!
//! A judge failure is recorded as ungraded (score=None), never as a zero. Re-running re-judges
//! every selection; a failed re-judge never replaces the existing graded row of the same
//! selection.

use super::common::{self, CellArgs, IdentityArgs};
use crate::config;
use crate::core::judge::{self as J, Scored};
use crate::core::report as R;
use crate::core::schedule;
use anyhow::{bail, Result};
use clap::Args;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const PARALLEL: usize = 8;

#[derive(Args)]
pub struct JudgeArgs {
    #[command(flatten)]
    pub cell: CellArgs,
    #[command(flatten)]
    pub id: IdentityArgs,
    /// score this namespace's reference executions into base_judged.json instead
    #[arg(long)]
    pub bases: bool,
}

#[derive(Args)]
pub struct ReportArgs {
    #[command(flatten)]
    pub cell: CellArgs,
    /// comma list of scales to report, e.g. 1,2,3
    #[arg(long, default_value = "1")]
    pub k: String,
    #[arg(long)]
    pub seed: Option<i64>,
    #[arg(long = "mode", alias = "modes", default_value = "parallel", help = common::MODES_HELP)]
    pub modes: String,
}

/// Rebuild Scored rows from a judged.json written by `core::judge::write`.
pub fn load_rows(path: &Path) -> Result<Vec<Scored>> {
    if !path.is_file() {
        return Ok(vec![]);
    }
    let raw: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path)?)
        .map_err(|e| anyhow::anyhow!("unreadable judged cache {}: {e}; move it aside to rebuild", path.display()))?;
    let mut rows = vec![];
    for mut r in raw {
        if let Some(o) = r.as_object_mut() {
            o.remove("frac");
            o.remove("graded");
            if !o.contains_key("mode") {
                if let Some(a) = o.remove("arm") {
                    o.insert("mode".into(), a);
                }
            }
        }
        rows.push(serde_json::from_value(r)?);
    }
    Ok(rows)
}

/// Per (task, mode, k): a changed chosen run or a re-selected task replaces; for the same
/// selection the graded row wins.
pub fn merge(old: Vec<Scored>, new: Vec<Scored>) -> Vec<Scored> {
    let mut by: BTreeMap<(String, String, usize), Scored> = old.into_iter().map(|r| ((r.task_id.clone(), r.mode.clone(), r.k), r)).collect();
    for r in new {
        let key = (r.task_id.clone(), r.mode.clone(), r.k);
        if let Some(prev) = by.get(&key) {
            if prev.run_id == r.run_id && prev.selected_at == r.selected_at && prev.graded() && !r.graded() && !r.error.starts_with("selection") {
                continue;
            }
        }
        by.insert(key, r);
    }
    by.into_values().collect()
}

fn summaries(ns_dir: &Path) -> Vec<(PathBuf, Value)> {
    let mut out = vec![];
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(ns_dir).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path().join("summary.json")).filter(|p| p.is_file()).collect()).unwrap_or_default();
    dirs.sort();
    for p in dirs {
        match std::fs::read_to_string(&p).map_err(anyhow::Error::from).and_then(|s| Ok(serde_json::from_str::<Value>(&s)?)) {
            Ok(v) => out.push((p, v)),
            Err(e) => println!("  ! skipping unreadable {}: {e}", p.display()),
        }
    }
    out
}

/// Every requested mode's selection record from every finished task summary.
pub fn collect_selections(ns_dir: &Path, modes: &[&str]) -> Vec<Value> {
    let mut out = vec![];
    for (_, s) in summaries(ns_dir) {
        for m in modes {
            if let Some(sel) = s.get("selections").and_then(|x| x.get(*m)).filter(|v| v.as_object().map(|o| !o.is_empty()).unwrap_or(false)) {
                let mut sel = sel.clone();
                if sel.get("mode").is_none() {
                    if let Some(a) = sel.get("arm").cloned() {
                        sel["mode"] = a;
                    }
                }
                out.push(sel);
            }
        }
    }
    out
}

fn judge_bases(adapter: &dyn crate::core::spec::BenchmarkAdapter, ns_dir: &Path) -> Result<i32> {
    let (mut rows, mut legacy) = (vec![], 0);
    for (_, s) in summaries(ns_dir) {
        let task_id = s["task_id"].as_str().unwrap_or("").to_string();
        for e in s["base"].as_array().cloned().unwrap_or_default() {
            let Some(rd) = e.get("run_dir").and_then(|v| v.as_str()).filter(|v| !v.is_empty()) else {
                legacy += 1;
                continue;
            };
            let label = e["label"].as_str().unwrap_or("").to_string();
            let run_dir = PathBuf::from(rd);
            let got = if run_dir.is_dir() { adapter.score(&run_dir, &task_id) } else { Ok(None) };
            let got = match got {
                Err(err) => {
                    println!("  {:<52} {label} ERROR {err}", crate::util::take_chars(&task_id, 52));
                    None
                }
                Ok(g) => {
                    let g = g.filter(|g| J::has_grade(g["n_passed"].as_f64(), g["n_criteria"].as_f64()));
                    let shown = g.as_ref().map(|g| format!("{}/{}", g["n_passed"], g["n_criteria"])).unwrap_or_else(|| "-".into());
                    println!("  {:<52} {label} {shown}", crate::util::take_chars(&task_id, 52));
                    g
                }
            };
            rows.push(json!({"task": task_id, "label": label, "run_id": e["run_id"], "score": got}));
        }
    }
    if legacy > 0 {
        println!("  {legacy} reference execution(s) predate run-directory recording and cannot be scored; re-run those tasks to include them.");
    }
    if rows.is_empty() {
        println!("no reference executions recorded under {}; wrote nothing", ns_dir.display());
        return Ok(2);
    }
    let out = ns_dir.join("base_judged.json");
    std::fs::write(&out, serde_json::to_string_pretty(&rows)?)?;
    let graded = rows.iter().filter(|r| !r["score"].is_null()).count();
    println!("wrote {}: {} row(s), {graded} graded", out.display(), rows.len());
    Ok(if graded == rows.len() { 0 } else { 2 })
}

pub fn judge_cmd(a: JudgeArgs) -> Result<i32> {
    let modes = common::parse_modes(&a.id.modes)?;
    if a.cell.bench != "goal" {
        config::get().validate_models(true)?;
    }
    let adapter = common::make_adapter(&a.cell, None)?;
    let ns = common::summary_namespace(adapter.as_ref(), a.id.k, a.id.seed, "")?;
    let ns_dir = common::runs_root_for(adapter.as_ref()).join(&ns);
    let identity = ns_dir.join("config.json");
    if identity.is_file() {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&identity)?)?;
        if v["judge"].as_str().unwrap_or("") != config::get().judge_model {
            bail!("Judge model changed for this experiment; choose a new HARNESS2_RUNS_DIR");
        }
    }
    if !ns_dir.is_dir() {
        bail!("nothing to judge: {} does not exist — has the sweep `{ns}` run on this machine?", ns_dir.display());
    }
    if a.bases {
        return judge_bases(adapter.as_ref(), &ns_dir);
    }
    let sels = collect_selections(&ns_dir, &modes);
    if sels.is_empty() {
        bail!(
            "nothing to judge: no selections for modes {:?} under {} (tasks may still be running, or the wrong --k/--seed was given)",
            modes.iter().map(|m| schedule::mode_name(m)).collect::<Vec<_>>(),
            ns_dir.display()
        );
    }
    println!("judging {} selection(s) under {ns} ...", sels.len());
    let rows = J::judge_selections(adapter.as_ref(), &sels, PARALLEL);
    let out = ns_dir.join("judged.json");
    let all_graded = !rows.is_empty() && rows.iter().all(|r| r.graded());
    let merged = merge(load_rows(&out)?, rows);
    J::write(&merged, &out)?;
    let graded = merged.iter().filter(|r| r.graded()).count();
    println!("judged.json: {} row(s), {graded} graded, {} ungraded (judge failure or no pick — NOT zeros)", merged.len(), merged.len() - graded);
    println!("wrote {}", out.display());
    Ok(if all_graded { 0 } else { 2 })
}

/// (fell_back, usd) keyed (mode token, k), from the summaries the sweep checkpointed.
fn spend_and_fallbacks(ns_dir: &Path, modes: &[&str], k: usize) -> (BTreeMap<(String, usize), usize>, BTreeMap<(String, usize), f64>) {
    let (mut fb, mut usd) = (BTreeMap::new(), BTreeMap::new());
    for (_, s) in summaries(ns_dir) {
        let recs = s.get("modes").or_else(|| s.get("arms")).cloned().unwrap_or(json!({}));
        for m in modes {
            let key = (m.to_string(), k);
            let rec = recs.get(*m).cloned().unwrap_or(json!({}));
            *fb.entry(key.clone()).or_insert(0) += rec["fell_back"].as_array().map(|a| a.len()).unwrap_or(0);
            let spend: f64 = rec["stages"].as_array().map(|a| a.iter().map(|s| s["usd"].as_f64().unwrap_or(0.0)).sum()).unwrap_or(0.0);
            let sel_usd = s["selections"][*m]["usd"].as_f64().unwrap_or(0.0);
            *usd.entry(key).or_insert(0.0) += spend + sel_usd;
        }
    }
    (fb, usd)
}

pub fn report_cmd(a: ReportArgs) -> Result<i32> {
    let modes = common::parse_modes(&a.modes)?;
    let mut ks: Vec<usize> = a.k.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse()).collect::<Result<_, _>>().map_err(|_| anyhow::anyhow!("--k takes a comma list of ints, got {:?}", a.k))?;
    ks.sort();
    ks.dedup();
    if ks.is_empty() {
        bail!("--k named no scale");
    }
    let adapter = common::make_adapter(&a.cell, None)?;
    let (mut rows, mut fell_back, mut usd) = (vec![], BTreeMap::new(), BTreeMap::new());
    for k in ks {
        let ns_dir = common::runs_root_for(adapter.as_ref()).join(common::summary_namespace(adapter.as_ref(), k, a.seed, "")?);
        let judged = ns_dir.join("judged.json");
        let got = load_rows(&judged)?;
        if got.is_empty() {
            println!("  ! no judged rows at {} — run `harness2 judge` for k={k} first", judged.display());
            continue;
        }
        rows.extend(got.into_iter().filter(|r| modes.contains(&r.mode.as_str())));
        let (fb, sp) = spend_and_fallbacks(&ns_dir, &modes, k);
        fell_back.extend(fb);
        usd.extend(sp);
    }
    if rows.is_empty() {
        bail!("nothing to report");
    }
    let title = format!(
        "{}-{}{} modes={}",
        adapter.spec().name,
        adapter.substrate_name(),
        a.seed.map(|s| format!(" seed={s}")).unwrap_or_default(),
        modes.iter().map(|m| schedule::mode_name(m)).collect::<Vec<_>>().join(",")
    );
    let sums = R::summarise(&rows, &fell_back, &usd);
    println!("{}", R::render(&sums, &title, &adapter.spec().name));
    let spend: f64 = sums.iter().map(|s| s.usd).sum();
    if spend > 0.0 {
        println!("\n  improver (editors, composer, selection, reflection) spend across reported modes: ${spend:.2}");
    }
    Ok(0)
}
