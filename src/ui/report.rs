//! A goal's results as one Markdown report (to share or archive), and moving a goal to the trash.

use crate::benchmarks::goal::{self, GoalAdapter};
use crate::core::spec::BenchmarkAdapter;
use crate::util;
use anyhow::{bail, Result};
use axum::extract::Path as AxPath;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use std::path::PathBuf;

fn read_json(p: PathBuf) -> Value {
    std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null)
}

fn pct(v: &Value) -> String {
    v.as_f64().map(|f| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| "—".into())
}

pub fn build(a: &GoalAdapter, id: &str) -> Result<String> {
    let t = a.load_task(id)?;
    let mut s = format!("# Harnessmith report\n\n- Project: `{}`\n- Goal id: `{id}`\n- Domain: {}\n- Created: {}\n- Report written: {}\n\n## Goal\n\n{}\n\n", a.root.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), t.domain, t.created_at, util::now_iso(), t.prompt.trim());
    if !t.rubric.trim().is_empty() {
        s += &format!("## Grading criteria\n\n{}\n\n", t.rubric.lines().filter(|l| !l.trim().is_empty()).map(|l| format!("- {}", l.trim())).collect::<Vec<_>>().join("\n"));
    }
    let runs: Vec<_> = super::RUNS.lock().unwrap().values().filter(|r| r.goal_id == id).cloned().collect();
    if let Some(r) = runs.iter().max_by(|x, y| x.started.cmp(&y.started)) {
        s += &format!("## Last run\n\n- Status: **{}**\n- Started {} · finished {}\n", if r.status == "review" { "needs review" } else { r.status.as_str() }, r.started, if r.finished.is_empty() { "—" } else { &r.finished });
        if !r.error.is_empty() {
            s += &format!("- Notes: {}\n", r.error);
        }
        s += "\n";
    }
    for d in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
        let v = read_json(d.path().join(id).join("summary.json"));
        let Some(sels) = v["selections"].as_object() else { continue };
        for (mode, sel) in sels {
            s += &format!("## Adopted output ({})\n\n- Chosen: `{}`{}\n", if mode == "agg" { "parallel" } else { "sequential" }, sel["chosen_label"].as_str().unwrap_or("—"), if sel["chosen_is_base"] == true { " (the unmodified base attempt)" } else { "" });
            if let Some(why) = sel["why"].as_str().or(sel["reason"].as_str()) {
                s += &format!("- Why: {}\n", why.replace('\n', " "));
            }
            if let Some(rd) = sel["chosen_run_dir"].as_str().filter(|x| !x.is_empty()) {
                let rd = PathBuf::from(rd);
                let out = rd.join("workspace/output");
                let files: Vec<String> = util::files_sorted(&out).iter().map(|p| format!("`{}`", util::rel(&out, p))).collect();
                s += &format!("- Files: {}\n- On disk: `{}`\n", if files.is_empty() { "(none)".into() } else { files.join(", ") }, out.display());
                let issues = crate::llm::review::review(&t.prompt, &out);
                if !issues.is_empty() {
                    s += &format!("- Open problems:\n{}\n", issues.iter().map(|i| format!("  - {i}")).collect::<Vec<_>>().join("\n"));
                }
                let text = a.deliverable_text(&rd).unwrap_or_default();
                if !text.trim().is_empty() {
                    s += &format!("\n<details><summary>Output</summary>\n\n{}\n\n</details>\n", util::take_chars(&text, 40_000));
                }
            }
            s += "\n";
        }
    }
    let judged = read_json(a.task_dir(id).join("judged.json"));
    if let Some(rows) = judged.as_array().filter(|r| !r.is_empty()) {
        s += "## Rubric scores\n\n| output | score | passed |\n|---|---|---|\n";
        for r in rows {
            s += &format!("| {} `{}` | {} | {}/{} |\n", if r["is_base_attempt"] == true { "base attempt" } else { "adopted" }, r["label"].as_str().unwrap_or(""), pct(&r["frac"]), r["n_passed"], r["n_criteria"]);
        }
        s += "\n";
    }
    let ab = read_json(a.task_dir(id).join("ab.json"));
    if ab["summary"]["verdict"].is_string() {
        s += &format!("## A/B test ({}× each)\n\n- Base: {} ± {}\n- Adopted `{}`: {} ± {}\n- Verdict: **{}**\n\n", ab["n"], pct(&ab["summary"]["base"]["mean"]), pct(&ab["summary"]["base"]["sd"]), ab["adopted_label"].as_str().unwrap_or(""), pct(&ab["summary"]["adopted"]["mean"]), pct(&ab["summary"]["adopted"]["sd"]), ab["summary"]["verdict"].as_str().unwrap_or(""));
    }
    let so = read_json(a.task_dir(id).join("shootout.json"));
    if let Some(sm) = so["summary"].as_object().filter(|m| !m.is_empty()) {
        s += &format!("## Model shootout ({}× each)\n\n| solver model | score | delivered | avg problems | avg time | avg tokens |\n|---|---|---|---|---|---|\n", so["n"]);
        for (m, v) in sm {
            s += &format!("| `{m}`{} | {} | {}/{} | {} | {}s | {} |\n", if so["best"].as_str() == Some(m.as_str()) { " ★" } else { "" }, pct(&v["mean"]), v["delivered"], v["runs"], v["avg_issues"], v["avg_seconds"], v["avg_tokens"]);
        }
        s += "\n";
    }
    Ok(s)
}

pub(crate) async fn download(AxPath((p, id)): AxPath<(String, String)>) -> Response {
    let res = tokio::task::spawn_blocking(move || -> Result<String> { build(&GoalAdapter::open(&goal::project_dir(&p)?)?, &id) }).await;
    match res {
        Ok(Ok(md)) => ([(header::CONTENT_TYPE, "text/markdown; charset=utf-8".to_string()), (header::CONTENT_DISPOSITION, "attachment; filename=\"harnessmith-report.md\"".to_string())], md).into_response(),
        Ok(Err(e)) => (StatusCode::NOT_FOUND, Json(json!({"error": format!("{e}")}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("{e}")}))).into_response(),
    }
}

/// Move a goal to `<project>/trash/` (recoverable by moving it back). Its executions stay on disk.
pub(crate) async fn trash(AxPath((p, id)): AxPath<(String, String)>) -> super::ApiResult {
    super::blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        if id.contains('/') || id.contains("..") {
            bail!("bad goal id");
        }
        let src = a.task_dir(&id);
        if !src.is_dir() {
            bail!("no such goal");
        }
        if super::RUNS.lock().unwrap().values().any(|r| r.goal_id == id && (r.status == "running" || r.status == "queued")) {
            bail!("this goal is running; stop it first");
        }
        let dest = a.root.join("trash").join(format!("{id}-{}", chrono::Utc::now().format("%Y%m%d%H%M%S")));
        std::fs::create_dir_all(dest.parent().unwrap())?;
        std::fs::rename(&src, &dest)?;
        Ok(json!({"trashed": true, "moved_to": dest}))
    })
    .await
}
