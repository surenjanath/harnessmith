//! Where a goal's files are: every attempt's output folder (adopted first), the inputs, and the
//! project folder; a safe preview; and "Open folder" in the OS file manager when the server runs
//! on the person's own machine.

use super::{blocking, ApiResult};
use crate::benchmarks::goal::{self, GoalAdapter};
use crate::util;
use anyhow::{bail, Result};
use axum::extract::{Path as AxPath, Query};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn inside(root: &Path, p: &Path) -> Result<PathBuf> {
    let root = root.canonicalize()?;
    let p = p.canonicalize().map_err(|_| anyhow::anyhow!("no such file: {}", p.display()))?;
    if !p.starts_with(&root) {
        bail!("that path is outside the project folder");
    }
    Ok(p)
}

fn list(dir: &Path) -> Vec<Value> {
    util::files_sorted(dir)
        .into_iter()
        .filter(|p| !util::in_pycache(p))
        .take(400)
        .map(|p| json!({"name": util::rel(dir, &p), "path": p.display().to_string(), "size": std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)}))
        .collect()
}

/// GET /api/projects/:p/goals/:id/files
pub(crate) async fn goal_files(AxPath((p, id)): AxPath<(String, String)>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let mut adopted: Vec<(String, String)> = vec![];
        for dir in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
            if let Ok(s) = std::fs::read_to_string(dir.path().join(&id).join("summary.json")) {
                if let Ok(v) = serde_json::from_str::<Value>(&s) {
                    for (m, sel) in v["selections"].as_object().cloned().unwrap_or_default() {
                        if let Some(d) = sel["chosen_run_dir"].as_str().filter(|d| !d.is_empty()) {
                            adopted.push((m, d.to_string()));
                        }
                    }
                }
            }
        }
        let mut groups = vec![];
        for ns in std::fs::read_dir(a.root.join("executions")).into_iter().flatten().flatten() {
            let gdir = ns.path().join(&id);
            for run in std::fs::read_dir(&gdir).into_iter().flatten().flatten() {
                let out = run.path().join("workspace/output");
                if !out.is_dir() {
                    continue;
                }
                let label = run.file_name().to_string_lossy().to_string();
                let rd = run.path().display().to_string();
                let modes: Vec<String> = adopted.iter().filter(|(_, d)| *d == rd).map(|(m, _)| m.clone()).collect();
                groups.push(json!({"label": label, "namespace": ns.file_name().to_string_lossy(), "adopted": !modes.is_empty(), "modes": modes, "dir": out.display().to_string(), "workspace": run.path().join("workspace").display().to_string(), "files": list(&out)}));
            }
        }
        groups.sort_by_key(|g| (!g["adopted"].as_bool().unwrap_or(false), g["label"].as_str().unwrap_or("").to_string()));
        let inputs = a.task_dir(&id).join("files");
        Ok(json!({
            "project_dir": a.root.display().to_string(),
            "goal_dir": a.task_dir(&id).display().to_string(),
            "inputs": if inputs.is_dir() { json!({"dir": inputs.display().to_string(), "files": list(&inputs)}) } else { Value::Null },
            "groups": groups,
            "can_open": opener().is_some(),
        }))
    })
    .await
}

#[derive(Deserialize)]
pub struct FileQ {
    path: String,
}

/// GET /api/projects/:p/file?path=… — a file from the project, previewed safely: HTML is served
/// in a CSP sandbox (its own opaque origin, so it cannot reach this UI's API).
pub async fn project_file(AxPath(p): AxPath<String>, Query(q): Query<FileQ>) -> Response {
    let r: Result<(Vec<u8>, String)> = (|| {
        let root = goal::project_dir(&p)?;
        let f = inside(&root, Path::new(&q.path))?;
        if !f.is_file() {
            bail!("not a file");
        }
        if std::fs::metadata(&f)?.len() > 20 << 20 {
            bail!("file is over 20 MB; open the folder instead");
        }
        let ext = f.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let ct = match ext.as_str() {
            "html" | "htm" => "text/html; charset=utf-8",
            "svg" => "image/svg+xml",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "pdf" => "application/pdf",
            "json" => "application/json",
            "csv" => "text/csv; charset=utf-8",
            _ => "text/plain; charset=utf-8",
        };
        Ok((std::fs::read(&f)?, ct.to_string()))
    })();
    match r {
        Ok((b, ct)) => (
            [(header::CONTENT_TYPE, ct), (header::CONTENT_SECURITY_POLICY, "sandbox allow-scripts allow-popups".to_string()), (header::X_CONTENT_TYPE_OPTIONS, "nosniff".to_string()), (header::CACHE_CONTROL, "no-store".to_string())],
            b,
        )
            .into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{e}")}))).into_response(),
    }
}

fn opener() -> Option<(&'static str, Vec<&'static str>)> {
    if cfg!(target_os = "macos") {
        Some(("open", vec![]))
    } else if cfg!(target_os = "windows") {
        Some(("explorer", vec![]))
    } else if util::which("xdg-open").is_some() && (std::env::var_os("DISPLAY").is_some() || std::env::var_os("WAYLAND_DISPLAY").is_some()) {
        Some(("xdg-open", vec![]))
    } else {
        None
    }
}

#[derive(Deserialize)]
pub struct OpenReq {
    project: String,
    path: String,
}

/// POST /api/open — show a folder (or reveal a file) in the OS file manager.
pub(crate) async fn open_in_os(Json(o): Json<OpenReq>) -> ApiResult {
    blocking(move || {
        let root = goal::project_dir(&o.project)?;
        let p = inside(&root, Path::new(&o.path))?;
        let Some((cmd, _)) = opener() else {
            bail!("this server has no desktop to open folders on (it may be remote or headless); copy the path instead");
        };
        let mut c = std::process::Command::new(cmd);
        if cmd == "open" && p.is_file() {
            c.arg("-R");
        }
        let target = if cmd != "open" && p.is_file() { p.parent().unwrap_or(&p).to_path_buf() } else { p.clone() };
        c.arg(&target).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        c.spawn()?;
        Ok(json!({"opened": target.display().to_string()}))
    })
    .await
}
