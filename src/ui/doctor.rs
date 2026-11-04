//! The Doctor: diagnose what can go wrong around a run, and fix what can be fixed safely.
//!
//! Each check returns a status (ok / warn / fail / info), a sentence of detail, and, when a
//! fix exists, a fix id that `fix()` knows how to apply. `heal_all()` applies every fix that
//! is safe to apply unattended; the server runs it on start and then periodically.

use super::{save_run, RUNS};
use crate::benchmarks::goal::{self, GoalAdapter};
use crate::core::{runmark, spec::BenchmarkAdapter, tree};
use crate::llm::{client, connections, health};
use crate::substrate::native;
use crate::util;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

fn check(id: &str, name: &str, status: &str, detail: String, fix: Option<(&str, &str)>) -> Value {
    json!({"id": id, "name": name, "status": status, "detail": detail,
        "fix": fix.map(|(f, label)| json!({"id": f, "label": label}))})
}

fn projects() -> Vec<(String, PathBuf)> {
    std::fs::read_dir(goal::projects_root())
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().join("project.json").is_file() || e.path().join("base").is_dir())
        .map(|e| (e.file_name().to_string_lossy().to_string(), e.path()))
        .collect()
}

/// Native rollouts marked complete although their session died before producing anything
/// (written by builds before the dead-rollout fix): rerunning would reuse them.
fn dead_rollouts() -> Vec<PathBuf> {
    let mut out = vec![];
    for (_, root) in projects() {
        for mark in util::files_sorted(&root.join("executions")) {
            if mark.file_name().map(|n| n != runmark::MARK).unwrap_or(true) {
                continue;
            }
            let rd = mark.parent().unwrap().to_path_buf();
            let usage: Value = std::fs::read_to_string(rd.join("usage.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(json!({}));
            let err = usage["error"].as_str().unwrap_or("");
            if native::deliverables(&rd).is_empty() && !err.is_empty() && !err.contains("budget") {
                out.push(rd);
            }
        }
    }
    out
}

fn stale_temp() -> Vec<PathBuf> {
    let cutoff = SystemTime::now() - Duration::from_secs(6 * 3600);
    std::fs::read_dir(std::env::temp_dir())
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            ["harness2-prop-", "harness2-harvest-", "adapt-oc-", "adapt-cx-", "hrender_"].iter().any(|p| n.starts_with(p))
        })
        .filter(|e| e.metadata().and_then(|m| m.modified()).map(|t| t < cutoff).unwrap_or(false))
        .map(|e| e.path())
        .collect()
}

/// Every model id any project's roles name (chains split), deduplicated.
fn role_models() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = vec![];
    for (name, root) in projects() {
        let Ok(a) = GoalAdapter::open(&root) else { continue };
        let s = &a.settings;
        for (role, m) in [("solver", &s.solver_model), ("improver", &s.improver_model), ("selector", &s.selector_model), ("judge", &s.judge_model)] {
            for id in m.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
                if !v.iter().any(|(x, _)| x == id) {
                    v.push((id.to_string(), format!("{name}:{role}")));
                }
            }
        }
    }
    v
}

pub fn checks() -> Vec<Value> {
    let mut out = vec![];
    let cfg = crate::config::get();

    // 1. Storage.
    let runs = &cfg.runs;
    let _ = std::fs::create_dir_all(runs);
    let probe = runs.join(".doctor-write-test");
    let writable = std::fs::write(&probe, "ok").is_ok();
    let _ = std::fs::remove_file(&probe);
    let free = fs2::available_space(runs).unwrap_or(0);
    let gb = free as f64 / 1e9;
    out.push(check(
        "storage",
        "Run storage",
        if !writable || gb < 0.2 { "fail" } else if gb < 2.0 { "warn" } else { "ok" },
        format!("{} — {} · {:.1} GB free", runs.display(), if writable { "writable" } else { "NOT writable" }, gb),
        (gb < 2.0 && !stale_temp().is_empty()).then_some(("clean_temp", "Clean temp files")),
    ));

    // 2. Python (harness scripts and the demo run with it).
    let py = util::which("python3");
    out.push(check(
        "python",
        "Python for harness scripts",
        if py.is_some() { "ok" } else { "warn" },
        py.map(|p| format!("{}", p.display())).unwrap_or_else(|| "python3 not found: harness scripts (scripts/*.py) cannot run, and proposals' scripts are not syntax-checked".into()),
        None,
    ));

    // 3. Every model the roles use: resolvable, reachable, breaker state.
    let models = role_models();
    if models.is_empty() {
        out.push(check("models", "Role models", "warn", "No project has models set. Open Models and choose a solver, improver and selector.".into(), None));
    }
    let probes = util::par_map(models, 8, |(id, used_by)| {
        let r = connections::resolve(&id);
        let (status, detail, fix) = match r {
            Err(e) => ("fail".to_string(), format!("{e}"), None),
            Ok((c, m)) => {
                let h = health::get(&c.name);
                let st = health::state(&h);
                if st == "open" {
                    ("warn".into(), format!("circuit breaker OPEN after {} failures (last: {}); calls are failing over, retry in {}s", h.consecutive_fail, h.last_error, health::reopens_in(&c.name).as_secs()), Some(format!("reset_breaker:{}", c.name)))
                } else {
                    match client::list_models(&c) {
                        Ok(list) if !list.is_empty() && !list.iter().any(|x| x == &m || x.starts_with(&format!("{m}:")) || m.starts_with(x.as_str())) => (
                            "warn".into(),
                            format!("reachable, but {m:?} is not among its {} models (e.g. {})", list.len(), list.iter().take(4).cloned().collect::<Vec<_>>().join(", ")),
                            None,
                        ),
                        Ok(list) => ("ok".into(), format!("reachable · {} model(s) · {}{}", list.len(), st, if h.latency_ms > 0.0 { format!(" · ~{:.1}s per call", h.latency_ms / 1000.0) } else { String::new() }), None),
                        // A local server busy generating may not answer the probe; recent
                        // successful calls mean it is up, not down.
                        Err(_) if h.last_ok.map(|t| t.elapsed().as_secs() < 300).unwrap_or(false) => (
                            "ok".into(),
                            format!("busy (did not answer the probe) · last successful call {}s ago · ~{:.1}s per call", h.last_ok.unwrap().elapsed().as_secs(), h.latency_ms / 1000.0),
                            None,
                        ),
                        Err(e) => ("fail".into(), format!("unreachable: {}", util::take_chars(&format!("{e}"), 220)), None),
                    }
                }
            }
        };
        json!({"id": format!("model:{id}"), "name": format!("Model {id}"), "status": status, "detail": format!("{detail} (used by {used_by})"), "fix": fix.map(|f| json!({"id": f, "label": "Reset breaker"}))})
    });
    out.extend(probes);

    // 4. Secrets file permissions.
    let cp = connections::path();
    #[cfg(unix)]
    if cp.is_file() {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&cp).map(|m| m.permissions().mode() & 0o777).unwrap_or(0);
        let has_literal = connections::load().connections.iter().any(|c| !c.api_key.is_empty());
        out.push(check(
            "conn_perms",
            "Connections file",
            if has_literal && mode & 0o077 != 0 { "warn" } else { "ok" },
            format!("{} (mode {:o}){}", cp.display(), mode, if has_literal { " · contains literal API keys" } else { "" }),
            (has_literal && mode & 0o077 != 0).then_some(("chmod_conns", "Make private (0600)")),
        ));
    }

    // 5. Dead rollouts cached as complete.
    let dead = dead_rollouts();
    out.push(check(
        "dead_rollouts",
        "Cached failed rollouts",
        if dead.is_empty() { "ok" } else { "warn" },
        if dead.is_empty() { "none".into() } else { format!("{} rollout(s) are marked complete but produced nothing; reruns would reuse them", dead.len()) },
        (!dead.is_empty()).then_some(("clear_dead", "Clear them")),
    ));

    // 6. Interrupted and stuck runs.
    let runs_now: Vec<super::RunInfo> = RUNS.lock().unwrap().values().cloned().collect();
    let interrupted: Vec<&super::RunInfo> = runs_now.iter().filter(|r| r.status == "interrupted").collect();
    out.push(check(
        "interrupted",
        "Interrupted runs",
        if interrupted.is_empty() { "ok" } else { "warn" },
        if interrupted.is_empty() { "none".into() } else { format!("{} run(s) stopped mid-way", interrupted.len()) },
        (!interrupted.is_empty()).then_some(("resume_all", "Resume them")),
    ));
    for r in runs_now.iter().filter(|r| r.status == "running") {
        let last = crate::log::named(&r.id).events_from(0).last().and_then(|e| e["t"].as_i64()).unwrap_or(0);
        let idle = (chrono::Utc::now().timestamp_millis() - last) / 1000;
        if last > 0 && idle > 20 * 60 {
            out.push(check(&format!("stuck:{}", r.id), &format!("Run {}", r.id), "warn", format!("no progress for {} min — a model call may be hanging", idle / 60), Some((&format!("cancel:{}", r.id), "Stop it"))));
        }
    }

    // 7. Old temp directories.
    let tmp = stale_temp();
    out.push(check(
        "temp",
        "Temporary files",
        if tmp.is_empty() { "ok" } else { "info" },
        if tmp.is_empty() { "clean".into() } else { format!("{} leftover session folder(s) older than 6 h in {}", tmp.len(), std::env::temp_dir().display()) },
        (!tmp.is_empty()).then_some(("clean_temp", "Clean up")),
    ));

    // 8. Every project's base harness passes validation.
    for (name, root) in projects() {
        let Ok(a) = GoalAdapter::open(&root) else { continue };
        let surface = a.surface();
        let base = a.base_harness().unwrap_or_default();
        let t = tree::read_tree(&surface, &base, None);
        let res = if t.is_empty() { Err(anyhow::anyhow!("empty")) } else { tree::validate(&surface, &t, a.settings.allow_subagents, None, &tree::reserved_default()).map(|_| ()) };
        out.push(match res {
            Ok(()) => check(&format!("base:{name}"), &format!("Base harness · {name}"), "ok", format!("{} file(s), valid", t.len()), None),
            Err(e) => check(&format!("base:{name}"), &format!("Base harness · {name}"), "fail", util::take_chars(&format!("{e}"), 300), Some((&format!("restore_base:{name}"), "Restore default (backs up the current one)"))),
        });
    }

    // 9. Local servers worth knowing about.
    let found = client::discover_local();
    let known: Vec<String> = connections::load().connections.iter().map(|c| c.base()).collect();
    let new: Vec<String> = found.iter().filter(|f| !known.iter().any(|k| k == f["base_url"].as_str().unwrap_or(""))).map(|f| format!("{} ({})", f["base_url"].as_str().unwrap_or(""), f["models"].as_array().map(|m| m.len()).unwrap_or(0))).collect();
    out.push(check(
        "local",
        "Local model servers",
        "info",
        if found.is_empty() { "none running on the usual ports".into() } else if new.is_empty() { format!("{} found, all connected", found.len()) } else { format!("not connected yet: {} — add them on the Models tab", new.join(", ")) },
        None,
    ));
    out
}

/// Apply one fix.
pub fn fix(id: &str) -> Result<String> {
    if let Some(conn) = id.strip_prefix("reset_breaker:") {
        health::reset(conn);
        return Ok(format!("breaker for {conn} reset"));
    }
    if let Some(run) = id.strip_prefix("cancel:") {
        crate::log::named(run).cancel();
        return Ok(format!("stopping {run}"));
    }
    if let Some(p) = id.strip_prefix("restore_base:") {
        let root = goal::project_dir(p)?;
        let base = root.join("base");
        if base.exists() {
            let bak = root.join(format!("base.bak-{}", chrono::Utc::now().format("%Y%m%d%H%M%S")));
            std::fs::rename(&base, &bak)?;
        }
        tree::write_tree(&native::native_surface(), &native::default_base_tree(), &base, false)?;
        return Ok(format!("restored the default base harness for {p} (old one kept as base.bak-*)"));
    }
    match id {
        "clean_temp" => {
            let t = stale_temp();
            for p in &t {
                util::rmtree(p);
            }
            Ok(format!("removed {} folder(s)", t.len()))
        }
        "clear_dead" => {
            let d = dead_rollouts();
            for p in &d {
                // Only remove inside a project's executions folder.
                if p.components().any(|c| c.as_os_str() == "executions") {
                    util::rmtree(p);
                }
            }
            Ok(format!("cleared {} rollout(s); they will be re-run", d.len()))
        }
        "chmod_conns" => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(connections::path(), std::fs::Permissions::from_mode(0o600))?;
            }
            Ok("connections file is now private".into())
        }
        "resume_all" => {
            let list: Vec<super::RunInfo> = RUNS.lock().unwrap().values().filter(|r| r.status == "interrupted").cloned().collect();
            let mut n = 0;
            for r in list {
                let modes: Vec<&'static str> = r.modes.iter().filter_map(|m| crate::core::schedule::mode_token(m).ok()).collect();
                if let Ok(new) = super::start_run_from(&r.project, &r.goal_id, r.k, if modes.is_empty() { vec!["agg"] } else { modes }, &r.id) {
                    if let Some(x) = RUNS.lock().unwrap().get_mut(&r.id) {
                        x.status = "resumed".into();
                        x.error = format!("resumed as {new}");
                        save_run(x);
                    }
                    n += 1;
                }
            }
            Ok(format!("resumed {n} run(s)"))
        }
        _ => bail!("unknown fix {id:?}"),
    }
}

/// Fixes that are safe to apply unattended.
pub fn heal_all() -> Vec<String> {
    let mut done = vec![];
    for (id, ok) in [("clear_dead", !dead_rollouts().is_empty()), ("clean_temp", !stale_temp().is_empty())] {
        if ok {
            if let Ok(m) = fix(id) {
                done.push(m);
            }
        }
    }
    done
}

#[allow(dead_code)]
fn _p(_: &Path) {}
