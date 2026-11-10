//! The accuracy side of the UI: a project's experience (worked examples), its sleep (consolidate
//! and dream), and the accuracy check that measures what the harness adds to a model.
//!
//! Sleep and the accuracy check run in the background with their own log sink, like the
//! experiments in `ab`, so they can be watched and cancelled.

use crate::benchmarks::accuracy;
use crate::benchmarks::goal::{self, GoalAdapter};
use crate::llm::{experience, sleep};
use crate::{log as hlog, util};
use anyhow::{bail, Result};
use axum::extract::Path as AxPath;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Mutex;

/// Running jobs: "sleep/<project>" or "accuracy" -> log sink name.
static ACTIVE: once_cell::sync::Lazy<Mutex<std::collections::HashMap<String, String>>> = once_cell::sync::Lazy::new(Default::default);
/// What the running accuracy check was asked to do.
static CHECK: once_cell::sync::Lazy<Mutex<Value>> = once_cell::sync::Lazy::new(|| Mutex::new(Value::Null));

fn tail(sink: &str, n: usize) -> Vec<String> {
    let lines = hlog::named(sink).lines_from(0);
    lines.iter().rev().take(n).rev().cloned().collect()
}

fn accuracy_root() -> std::path::PathBuf {
    crate::config::get().runs.join("accuracy")
}

// ---- experience -----------------------------------------------------------------------------

pub(crate) async fn experience_get(AxPath(p): AxPath<String>) -> super::ApiResult {
    super::blocking(move || {
        let root = goal::project_dir(&p)?;
        let a = GoalAdapter::open(&root)?;
        let mut all = experience::Store::open(&root).load();
        all.sort_by(|x, y| (x.status != "active").cmp(&(y.status != "active")).then(y.pinned.cmp(&x.pinned)).then(y.at.cmp(&x.at)));
        let active: Vec<&experience::Exemplar> = all.iter().filter(|e| e.status == "active").collect();
        let stats = json!({
            "active": active.len(), "archived": all.len() - active.len(), "dreamed": active.iter().filter(|e| e.origin == "dream").count(),
            "uses": active.iter().map(|e| e.uses as u64).sum::<u64>(), "rehearsed": active.iter().filter(|e| e.dreams > 0).count(),
            "helped": active.iter().map(|e| e.helped as u64).sum::<u64>(), "unrehearsed": active.iter().filter(|e| e.dreams == 0).count(),
        });
        let examples: Vec<Value> = all
            .iter()
            .map(|e| {
                let mut v = serde_json::to_value(e).unwrap_or(Value::Null);
                v["value"] = json!((e.value() * 100.0).round() / 100.0);
                v
            })
            .collect();
        let sink = ACTIVE.lock().unwrap().get(&format!("sleep/{p}")).cloned();
        let last_log = root.join("sleep").join("last_log");
        let log_name = sink.clone().or_else(|| std::fs::read_to_string(&last_log).ok());
        Ok(json!({
            "examples": examples, "stats": stats,
            "sleep": {"running": sink.is_some(), "log_tail": log_name.map(|s| tail(s.trim(), 14)).unwrap_or_default(), "journals": sleep::journals(&root, 5)},
            "settings": {"experience": a.settings.experience, "sleep_auto": a.settings.sleep_auto, "sleep_dreams": a.settings.sleep_dreams, "accuracy": a.settings.accuracy, "accuracy_samples": a.settings.accuracy_samples, "escalate_model": a.settings.escalate_model},
            "boost_on": crate::llm::precise::enabled(&a.settings.accuracy, &a.settings.solver_model),
        }))
    })
    .await
}

#[derive(Deserialize)]
pub(crate) struct ExEdit {
    pinned: Option<bool>,
    status: Option<String>,
}

pub(crate) async fn experience_edit(AxPath((p, id)): AxPath<(String, String)>, Json(e): Json<ExEdit>) -> super::ApiResult {
    super::blocking(move || {
        let st = experience::Store::open(&goal::project_dir(&p)?);
        st.update(|all| {
            let Some(x) = all.iter_mut().find(|x| x.id == id) else { bail!("no example {id}") };
            if let Some(pin) = e.pinned {
                x.pinned = pin;
            }
            if let Some(s) = &e.status {
                if !["active", "archived"].contains(&s.as_str()) {
                    bail!("status is active or archived");
                }
                x.status = s.clone();
                x.note = if s == "archived" { "archived by hand".into() } else { String::new() };
            }
            Ok(json!({"ok": true}))
        })
    })
    .await
}

pub(crate) async fn experience_delete(AxPath((p, id)): AxPath<(String, String)>) -> super::ApiResult {
    super::blocking(move || {
        let st = experience::Store::open(&goal::project_dir(&p)?);
        st.update(|all| {
            let n = all.len();
            all.retain(|x| x.id != id);
            Ok(json!({"deleted": n - all.len()}))
        })
    })
    .await
}

// ---- sleep ----------------------------------------------------------------------------------

#[derive(Deserialize, Default)]
pub(crate) struct SleepReq {
    dreams: Option<usize>,
    explore: Option<usize>,
}

/// Start a sleep for `project` in the background; an error when one is already running.
pub(crate) fn start_sleep(project: &str, dreams: Option<usize>, explore: Option<usize>, why: &str) -> Result<()> {
    let root = goal::project_dir(project)?;
    let a = GoalAdapter::open(&root)?;
    if a.settings.solver_model.trim().is_empty() {
        bail!("no solver model is set for this project");
    }
    let key = format!("sleep/{project}");
    let sink_name = format!("sleep-{project}-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
    {
        let mut act = ACTIVE.lock().unwrap();
        if act.contains_key(&key) {
            bail!("this project is already asleep");
        }
        act.insert(key.clone(), sink_name.clone());
    }
    let _ = util::write(&root.join("sleep").join("last_log"), &sink_name);
    let mut o = crate::cli::sleep_opts(&a, None);
    if let Some(d) = dreams {
        o.dreams = d.clamp(0, 30);
    }
    o.explore = explore.unwrap_or(2).min(6);
    let sink = hlog::named(&sink_name);
    let why = why.to_string();
    std::thread::spawn(move || {
        hlog::set_current(Some(sink.clone()));
        hlog::line(format!("sleep started ({why})"));
        if let Err(e) = sleep::sleep(&root, &o) {
            hlog::line(format!("sleep ended early: {e}"));
        }
        ACTIVE.lock().unwrap().remove(&key);
    });
    Ok(())
}

pub(crate) async fn sleep_start(AxPath(p): AxPath<String>, body: Option<Json<SleepReq>>) -> super::ApiResult {
    let r = body.map(|b| b.0).unwrap_or_default();
    super::blocking(move || {
        start_sleep(&p, r.dreams, r.explore, "asked from the UI")?;
        Ok(json!({"started": true}))
    })
    .await
}

pub(crate) async fn sleep_cancel(AxPath(p): AxPath<String>) -> super::ApiResult {
    super::blocking(move || match ACTIVE.lock().unwrap().get(&format!("sleep/{p}")).cloned() {
        Some(s) => {
            hlog::named(&s).cancel();
            Ok(json!({"cancelled": true}))
        }
        None => bail!("this project is not asleep"),
    })
    .await
}

/// Called by the scheduler: a project with `sleep_auto` on sleeps when nothing is running,
/// there is experience it has not rehearsed yet, and its last sleep was over six hours ago.
pub(crate) fn auto_sleep_tick() {
    if super::RUNS.lock().unwrap().values().any(|r| r.status == "running" || r.status == "queued") || !ACTIVE.lock().unwrap().is_empty() {
        return;
    }
    let Ok(rd) = std::fs::read_dir(goal::projects_root()) else { return };
    for dir in rd.flatten() {
        let name = dir.file_name().to_string_lossy().to_string();
        let Ok(a) = GoalAdapter::open(&dir.path()) else { continue };
        if !a.settings.sleep_auto || !a.settings.experience {
            continue;
        }
        let fresh = experience::Store::open(&dir.path()).load().iter().filter(|e| e.status == "active" && e.dreams == 0).count();
        if fresh < 2 {
            continue;
        }
        let recent = sleep::journals(&dir.path(), 1).first().and_then(|j| j["at"].as_str().and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())).map(|t| chrono::Utc::now() - t.with_timezone(&chrono::Utc) < chrono::Duration::hours(6)).unwrap_or(false);
        if recent {
            continue;
        }
        if start_sleep(&name, None, None, "idle, with unrehearsed experience").is_ok() {
            crate::say!("sleep: project {name} is idle with {fresh} unrehearsed example(s); sleeping");
            return; // one at a time: a local model serves one job well
        }
    }
}

// ---- accuracy check -------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct CheckReq {
    model: String,
    #[serde(default)]
    modes: Vec<String>,
    per_category: Option<usize>,
    seed: Option<u64>,
    samples: Option<usize>,
    #[serde(default)]
    escalate: String,
}

pub(crate) async fn accuracy_get() -> super::ApiResult {
    super::blocking(move || {
        let sink = ACTIVE.lock().unwrap().get("accuracy").cloned();
        let last = std::fs::read_to_string(accuracy_root().join("last_log")).ok();
        Ok(json!({
            "running": if sink.is_some() { CHECK.lock().unwrap().clone() } else { Value::Null },
            "log_tail": sink.or(last).map(|s| tail(s.trim(), 16)).unwrap_or_default(),
            "reports": accuracy::saved_reports(&accuracy_root()),
            "kinds": accuracy::CATEGORIES,
        }))
    })
    .await
}

pub(crate) async fn accuracy_start(Json(r): Json<CheckReq>) -> super::ApiResult {
    super::blocking(move || {
        if r.model.trim().is_empty() {
            bail!("choose a model to measure");
        }
        let (c, _) = crate::llm::connections::resolve(r.model.split(',').next().unwrap_or("").trim())?;
        if let Some(why) = c.missing_key() {
            bail!("{why}");
        }
        let mut modes: Vec<String> = r.modes.iter().map(|m| m.trim().to_string()).filter(|m| ["plain", "agent", "boost", "learned"].contains(&m.as_str())).collect();
        if modes.is_empty() {
            modes = vec!["plain".into(), "boost".into()];
        }
        let slug: String = r.model.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '_' }).collect();
        let seed = r.seed.unwrap_or(1);
        let o = accuracy::BenchOpts {
            model: r.model.clone(),
            modes: modes.clone(),
            seed,
            per_category: r.per_category.unwrap_or(1).clamp(1, 5),
            only: vec![],
            out: accuracy_root().join(format!("{slug}-seed{seed}")),
            samples: r.samples.unwrap_or(3).clamp(1, 7),
            agent_steps: 14,
            timeout: 600,
            escalate_model: r.escalate.clone(),
            dreams: 6,
        };
        let sink_name = format!("accuracy-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
        {
            let mut act = ACTIVE.lock().unwrap();
            if act.contains_key("accuracy") {
                bail!("an accuracy check is already running");
            }
            act.insert("accuracy".into(), sink_name.clone());
        }
        *CHECK.lock().unwrap() = json!({"model": r.model, "modes": modes, "questions": o.per_category * accuracy::CATEGORIES.len(), "seed": seed, "started": util::now_iso()});
        let _ = util::write(&accuracy_root().join("last_log"), &sink_name);
        let sink = hlog::named(&sink_name);
        std::thread::spawn(move || {
            hlog::set_current(Some(sink.clone()));
            if let Err(e) = accuracy::run(&o) {
                hlog::line(format!("accuracy check ended early: {e}"));
            }
            ACTIVE.lock().unwrap().remove("accuracy");
        });
        Ok(json!({"started": true}))
    })
    .await
}

pub(crate) async fn accuracy_cancel() -> super::ApiResult {
    super::blocking(move || match ACTIVE.lock().unwrap().get("accuracy").cloned() {
        Some(s) => {
            hlog::named(&s).cancel();
            Ok(json!({"cancelled": true}))
        }
        None => bail!("no accuracy check is running"),
    })
    .await
}
