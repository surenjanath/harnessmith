//! The web UI: connect models, give the system a goal, watch it refine a harness, read the
//! result. A local axum server with one embedded page; runs execute on background threads.
//!
//! JSON API (all under /api):
//! ```text
//!   GET    status                         version, paths
//!   GET    connections                    registry (keys redacted)
//!   POST   connections                    add/replace one connection
//!   DELETE connections/{name}
//!   POST   connections/test               {model} -> one-line round trip
//!   GET    connections/{name}/models      models the endpoint serves
//!   PUT    roles                          default models per role
//!   GET    projects                       projects + their settings
//!   GET    projects/{p}                   settings, goals, base harness
//!   PUT    projects/{p}/settings
//!   PUT    projects/{p}/base              replace the base harness tree
//!   POST   projects/{p}/goals             multipart: goal, domain, rubric, k, mode, files
//!   GET    projects/{p}/goals/{id}        summary, deliverables, stages, harnesses
//!   POST   projects/{p}/goals/{id}/rerun
//!   POST   projects/{p}/goals/{id}/judge
//!   GET    projects/{p}/library
//!   GET    runs                           background runs
//!   GET    runs/{id}/log?from=N           log lines from N
//!   GET    runs/{id}/events?from=N        structured pipeline events from N
//!   POST   runs/{id}/cancel               stop after the current model call
//! ```

use crate::benchmarks::goal::{self, GoalAdapter, GoalSettings};
use crate::core::spec::BenchmarkAdapter;
use crate::core::{judge, schedule, tree};
use crate::llm::{client, connections};
use crate::{log as hlog, util};
use anyhow::Result;
use axum::extract::{Multipart, Path as AxPath, Query};
use axum::http::{header, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use once_cell::sync::Lazy;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const INDEX: &str = include_str!("index.html");

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct RunInfo {
    id: String,
    project: String,
    goal_id: String,
    goal: String,
    status: String,
    started: String,
    finished: String,
    error: String,
    k: usize,
    modes: Vec<String>,
}

static RUNS: Lazy<Mutex<BTreeMap<String, RunInfo>>> = Lazy::new(|| Mutex::new(BTreeMap::new()));

/// Where each UI run keeps run.json, log.txt and events.jsonl.
fn runs_dir() -> PathBuf {
    crate::config::get().runs.join("ui-runs")
}

fn save_run(r: &RunInfo) {
    let _ = util::write(&runs_dir().join(&r.id).join("run.json"), serde_json::to_string_pretty(r).unwrap_or_default());
}

/// Reload earlier runs so their logs and pipelines survive a restart.
fn load_runs() {
    let mut runs = RUNS.lock().unwrap();
    for e in std::fs::read_dir(runs_dir()).into_iter().flatten().flatten() {
        let Ok(t) = std::fs::read_to_string(e.path().join("run.json")) else { continue };
        let Ok(mut r) = serde_json::from_str::<RunInfo>(&t) else { continue };
        if r.status == "running" {
            r.status = "interrupted".into();
            r.error = "the server stopped while this run was in progress; run it again to resume from what finished".into();
            save_run(&r);
        }
        hlog::register(&r.id, hlog::Sink::load(e.path()));
        runs.insert(r.id.clone(), r);
    }
}

struct ApiError(anyhow::Error);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(json!({"error": format!("{}", self.0)}))).into_response()
    }
}
impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(e.into())
    }
}
type ApiResult = std::result::Result<Json<Value>, ApiError>;

async fn blocking<F: FnOnce() -> Result<Value> + Send + 'static>(f: F) -> ApiResult {
    Ok(Json(tokio::task::spawn_blocking(f).await.map_err(|e| anyhow::anyhow!("{e}"))??))
}

pub fn serve(host: &str, port: u16) -> Result<()> {
    load_runs();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        let app = Router::new()
            .route("/", get(|| async { Html(INDEX) }))
            .route("/api/status", get(status))
            .route("/api/connections", get(list_connections).post(save_connection))
            .route("/api/connections/test", post(test_connection))
            .route("/api/connections/:name", delete(delete_connection))
            .route("/api/connections/:name/models", get(list_models))
            .route("/api/roles", put(save_roles))
            .route("/api/discover", get(discover))
            .route("/api/projects", get(list_projects))
            .route("/api/projects/:p", get(get_project))
            .route("/api/projects/:p/settings", put(save_settings))
            .route("/api/projects/:p/base", put(save_base))
            .route("/api/projects/:p/goals", post(new_goal))
            .route("/api/projects/:p/goals/:id", get(get_goal))
            .route("/api/projects/:p/goals/:id/rerun", post(rerun_goal))
            .route("/api/projects/:p/goals/:id/judge", post(judge_goal))
            .route("/api/projects/:p/library", get(get_library))
            .route("/api/runs", get(list_runs))
            .route("/api/runs/:id/log", get(run_log))
            .route("/api/runs/:id/events", get(run_events))
            .route("/api/runs/:id/cancel", post(cancel_run))
            .layer(axum::extract::DefaultBodyLimit::max(200 * 1024 * 1024));
        let listener = tokio::net::TcpListener::bind((host, port)).await?;
        let addr = listener.local_addr()?;
        println!("Harness² UI on http://{addr}  (Ctrl-C to stop)");
        if host != "127.0.0.1" && host != "localhost" {
            println!("WARNING: the UI has no authentication and can run shell commands through the agent; bind it to 127.0.0.1 unless you trust the network.");
        }
        axum::serve(listener, app).await?;
        anyhow::Ok(())
    })
}

async fn status() -> ApiResult {
    let cfg = crate::config::get();
    Ok(Json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "runs_dir": cfg.runs,
        "connections_file": connections::path(),
        "projects_root": goal::projects_root(),
        "providers": connections::PROVIDERS,
        "python": util::which("python3").is_some(),
    })))
}

async fn list_connections() -> ApiResult {
    let reg = connections::load();
    Ok(Json(json!({
        "connections": reg.connections.iter().map(|c| c.redacted()).collect::<Vec<_>>(),
        "roles": reg.roles,
        "builtins": ["openai", "anthropic", "gemini", "ollama", "openrouter", "groq", "deepseek", "together", "mistral", "mock"],
    })))
}

async fn save_connection(Json(mut c): Json<connections::Connection>) -> ApiResult {
    let mut reg = connections::load();
    if c.api_key == "••••••" {
        c.api_key = reg.connections.iter().find(|x| x.name == c.name).map(|x| x.api_key.clone()).unwrap_or_default();
    }
    connections::validate(&c)?;
    reg.connections.retain(|x| x.name != c.name);
    reg.connections.push(c);
    reg.connections.sort_by(|a, b| a.name.cmp(&b.name));
    connections::save(&reg)?;
    restrict_perms(&connections::path());
    Ok(Json(json!({"ok": true})))
}

fn restrict_perms(p: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o600));
    }
}

async fn delete_connection(AxPath(name): AxPath<String>) -> ApiResult {
    let mut reg = connections::load();
    reg.connections.retain(|x| x.name != name);
    connections::save(&reg)?;
    Ok(Json(json!({"ok": true})))
}

#[derive(Deserialize)]
struct TestReq {
    model: String,
}

async fn test_connection(Json(r): Json<TestReq>) -> ApiResult {
    blocking(move || {
        let (c, m) = connections::resolve(&r.model)?;
        let t0 = std::time::Instant::now();
        let reply = client::ping(&c, &m)?;
        Ok(json!({"ok": true, "reply": reply, "ms": t0.elapsed().as_millis()}))
    })
    .await
}

async fn list_models(AxPath(name): AxPath<String>) -> ApiResult {
    blocking(move || {
        let (c, _) = connections::resolve(&format!("{name}/_")).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(json!({"models": client::list_models(&c)?}))
    })
    .await
}

async fn discover() -> ApiResult {
    blocking(|| Ok(json!({"found": client::discover_local()}))).await
}

async fn save_roles(Json(roles): Json<BTreeMap<String, String>>) -> ApiResult {
    let mut reg = connections::load();
    for (k, v) in roles {
        if !["solver", "improver", "selector", "judge"].contains(&k.as_str()) {
            continue;
        }
        if v.trim().is_empty() {
            reg.roles.remove(&k);
        } else {
            reg.roles.insert(k, v.trim().to_string());
        }
    }
    connections::save(&reg)?;
    Ok(Json(json!({"ok": true})))
}

fn project_names() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(goal::projects_root())
        .map(|r| r.filter_map(|e| e.ok()).filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().to_string()).collect())
        .unwrap_or_default();
    if !v.contains(&"default".to_string()) {
        v.push("default".into());
    }
    v.sort();
    v
}

async fn list_projects() -> ApiResult {
    Ok(Json(json!({"projects": project_names()})))
}

fn goal_status(a: &GoalAdapter, id: &str) -> Value {
    let running = RUNS.lock().unwrap().values().filter(|r| r.goal_id == id && r.status == "running").map(|r| r.id.clone()).next();
    let mut best: Option<Value> = None;
    for dir in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
        let p = dir.path().join(id).join("summary.json");
        if let Ok(s) = std::fs::read_to_string(&p) {
            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                best = Some(json!({"namespace": dir.file_name().to_string_lossy(), "status": v["status"], "k": v["k"], "modes": v["modes"].as_object().map(|m| m.keys().cloned().collect::<Vec<_>>()).unwrap_or_default()}));
            }
        }
    }
    json!({"running": running, "summary": best})
}

async fn get_project(AxPath(p): AxPath<String>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let goals: Vec<Value> = a
            .all_tasks()
            .into_iter()
            .rev()
            .map(|t| {
                let st = goal_status(&a, &t.id);
                json!({"id": t.id, "prompt": t.prompt, "domain": t.domain, "has_rubric": !t.rubric.is_empty(), "created_at": t.created_at, "state": st})
            })
            .collect();
        let base = tree::read_tree(&a.surface(), &a.base_harness()?, None);
        Ok(json!({"name": p, "settings": a.settings, "goals": goals, "base": base, "surface": a.surface().always_on}))
    })
    .await
}

async fn save_settings(AxPath(p): AxPath<String>, Json(s): Json<GoalSettings>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::with_settings(&goal::project_dir(&p)?, s)?;
        a.save_settings()?;
        Ok(json!({"ok": true}))
    })
    .await
}

async fn save_base(AxPath(p): AxPath<String>, Json(t): Json<tree::Tree>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let surface = a.surface();
        let mut t = t;
        t.retain(|k, _| tree::is_allowed(&surface, k));
        tree::autotrigger(&surface, &mut t, None);
        tree::validate(&surface, &t, a.settings.allow_subagents, None, &tree::reserved_default())?;
        tree::write_tree(&surface, &t, &a.base_harness()?, false)?;
        Ok(json!({"ok": true, "files": t.len()}))
    })
    .await
}

fn start_run(project: &str, goal_id: &str, k: usize, modes: Vec<&'static str>) -> Result<String> {
    let root = goal::project_dir(project)?;
    let a = GoalAdapter::open(&root)?;
    a.preflight()?;
    let t = a.load_task(goal_id)?;
    if RUNS.lock().unwrap().values().any(|r| r.goal_id == goal_id && r.status == "running") {
        anyhow::bail!("this goal is already running");
    }
    let id = format!("run-{}-{}", chrono::Utc::now().format("%Y%m%d%H%M%S%3f"), &goal_id[goal_id.len().saturating_sub(6)..]);
    let info = RunInfo {
        id: id.clone(),
        project: project.into(),
        goal_id: goal_id.into(),
        goal: util::take_chars(&t.prompt, 200),
        status: "running".into(),
        started: util::now_iso(),
        finished: String::new(),
        error: String::new(),
        k,
        modes: modes.iter().map(|m| schedule::mode_name(m).to_string()).collect(),
    };
    save_run(&info);
    RUNS.lock().unwrap().insert(id.clone(), info);
    let sink = hlog::Sink::persistent(runs_dir().join(&id));
    hlog::register(&id, sink.clone());
    let (rid, gid) = (id.clone(), goal_id.to_string());
    std::thread::spawn(move || {
        hlog::set_current(Some(sink));
        crate::say!("Starting {} k={k} for goal {gid}", modes.iter().map(|m| schedule::mode_name(m)).collect::<Vec<_>>().join(" + "));
        let res = crate::cli::run_goal(&a, &gid, k, modes);
        let mut runs = RUNS.lock().unwrap();
        let r = runs.get_mut(&rid).unwrap();
        r.finished = util::now_iso();
        match res {
            Ok(v) => {
                r.status = if v["status"] == "ok" || v["status"] == "skipped" { "done".into() } else { "partial".into() };
                let errs: Vec<String> = v["errors"].as_array().map(|a| a.iter().filter_map(|e| e.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
                r.error = errs.join("; ");
                if hlog::named(&rid).is_cancelled() {
                    r.status = "cancelled".into();
                }
                save_run(r);
                drop(runs);
                crate::say!("Finished: {}", v["status"]);
            }
            Err(e) => {
                r.status = if hlog::named(&rid).is_cancelled() { "cancelled".into() } else { "failed".into() };
                r.error = format!("{e}");
                save_run(r);
                drop(runs);
                crate::say!("FAILED: {e}");
            }
        }
    });
    Ok(id)
}

fn parse_modes(m: &str) -> Result<Vec<&'static str>> {
    if m == "both" {
        return Ok(schedule::MODES.to_vec());
    }
    Ok(vec![schedule::mode_token(m)?])
}

async fn new_goal(AxPath(p): AxPath<String>, mut mp: Multipart) -> ApiResult {
    let (mut text, mut domain, mut rubric, mut k, mut mode) = (String::new(), String::new(), String::new(), 1usize, "parallel".to_string());
    let mut files: Vec<(String, Vec<u8>)> = vec![];
    while let Some(field) = mp.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "files" => {
                let fname = field.file_name().unwrap_or("file").to_string();
                let data = field.bytes().await?;
                if !fname.is_empty() && !data.is_empty() {
                    files.push((fname, data.to_vec()));
                }
            }
            _ => {
                let v = field.text().await?;
                match name.as_str() {
                    "goal" => text = v,
                    "domain" => domain = v,
                    "rubric" => rubric = v,
                    "k" => k = v.trim().parse().unwrap_or(1).clamp(1, 8),
                    "mode" => mode = v,
                    _ => {}
                }
            }
        }
    }
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        a.preflight()?;
        let id = a.add_task(&text, &domain, &rubric, &files)?;
        let run = start_run(&p, &id, k, parse_modes(&mode)?)?;
        Ok(json!({"goal_id": id, "run_id": run}))
    })
    .await
}

#[derive(Deserialize)]
struct RerunReq {
    k: Option<usize>,
    mode: Option<String>,
}

async fn rerun_goal(AxPath((p, id)): AxPath<(String, String)>, Json(r): Json<RerunReq>) -> ApiResult {
    blocking(move || {
        let run = start_run(&p, &id, r.k.unwrap_or(1).clamp(1, 8), parse_modes(r.mode.as_deref().unwrap_or("parallel"))?)?;
        Ok(json!({"run_id": run}))
    })
    .await
}

fn read_tree_json(a: &GoalAdapter, dir: &Path) -> Value {
    json!(tree::read_tree(&a.surface(), dir, None))
}

async fn get_goal(AxPath((p, id)): AxPath<(String, String)>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let t = a.load_task(&id)?;
        let base = tree::read_tree(&a.surface(), &a.base_harness()?, None);
        let mut namespaces = vec![];
        for dir in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
            let ns = dir.file_name().to_string_lossy().to_string();
            let td = dir.path().join(&id);
            let Ok(s) = std::fs::read_to_string(td.join("summary.json")) else { continue };
            let summary: Value = serde_json::from_str(&s)?;
            let mut out = json!({"namespace": ns, "summary": summary, "results": {}});
            for (mode, sel) in summary["selections"].as_object().cloned().unwrap_or_default() {
                let rd = sel["chosen_run_dir"].as_str().unwrap_or("").to_string();
                let label = sel["chosen_label"].as_str().unwrap_or("").to_string();
                let mode_ns = ns.replacen("-agg", &format!("-{mode}"), 1);
                let hdir = a.root.join("runs/adapt").join(&mode_ns).join(&id).join("harnesses").join(&label);
                let harness = read_tree_json(&a, &hdir);
                let ht: tree::Tree = serde_json::from_value(harness.clone()).unwrap_or_default();
                let diff = tree::diff_summary(&a.surface(), &base, &ht);
                let files = if rd.is_empty() { json!({}) } else { json!(a.deliverable_files(Path::new(&rd)).unwrap_or_default()) };
                let traj = if rd.is_empty() { String::new() } else { a.trajectory(Path::new(&rd), 4000).unwrap_or_default() };
                let usage: Value = std::fs::read_to_string(Path::new(&rd).join("usage.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
                let edits = std::fs::read_to_string(a.root.join("runs/adapt").join(&mode_ns).join(&id).join("reflection").join(format!("{mode}{}", summary["k"])).join("edits.md")).unwrap_or_default();
                out["results"][&mode] = json!({
                    "label": label, "run_dir": rd, "files": files, "trajectory": util::take_chars(&traj, 200_000),
                    "harness": harness, "diff": diff, "usage": usage, "why": sel["why"], "experience": sel["experience"], "edits": edits,
                });
            }
            namespaces.push(out);
        }
        let mut judged = json!(null);
        if let Ok(s) = std::fs::read_to_string(a.task_dir(&id).join("judged.json")) {
            judged = serde_json::from_str(&s).unwrap_or(json!(null));
        }
        let runs: Vec<Value> = RUNS.lock().unwrap().values().filter(|r| r.goal_id == id).rev().map(|r| json!({"id": r.id, "status": r.status, "started": r.started, "k": r.k, "modes": r.modes})).collect();
        Ok(json!({"task": t, "base": base, "namespaces": namespaces, "state": goal_status(&a, &id), "judged": judged, "runs": runs}))
    })
    .await
}

async fn judge_goal(AxPath((p, id)): AxPath<(String, String)>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let mut sels = vec![];
        for dir in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
            if let Ok(s) = std::fs::read_to_string(dir.path().join(&id).join("summary.json")) {
                let v: Value = serde_json::from_str(&s)?;
                for (_m, sel) in v["selections"].as_object().cloned().unwrap_or_default() {
                    sels.push(sel);
                }
            }
        }
        let rows = judge::judge_selections(&a, &sels, 4);
        let out: Vec<Value> = rows.iter().map(|r| r.to_json()).collect();
        util::write(&a.task_dir(&id).join("judged.json"), serde_json::to_string_pretty(&out)?)?;
        Ok(json!({"rows": out}))
    })
    .await
}

async fn get_library(AxPath(p): AxPath<String>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let root = a.root.join("runs/library");
        let mut entries = vec![];
        for f in util::files_sorted(&root) {
            if f.file_name().map(|n| n == "EXPERIENCE.md").unwrap_or(false) {
                entries.push(json!({"path": util::rel(&root, &f), "body": util::read_lossy(&f).unwrap_or_default()}));
            }
        }
        Ok(json!({"entries": entries}))
    })
    .await
}

async fn list_runs() -> ApiResult {
    let runs: Vec<Value> = RUNS
        .lock()
        .unwrap()
        .values()
        .rev()
        .map(|r| json!({"id": r.id, "project": r.project, "goal_id": r.goal_id, "goal": r.goal, "status": r.status, "started": r.started, "finished": r.finished, "error": r.error, "k": r.k, "modes": r.modes, "lines": hlog::named(&r.id).len(), "events": hlog::named(&r.id).n_events()}))
        .collect();
    Ok(Json(json!({"runs": runs})))
}

#[derive(Deserialize)]
struct LogQ {
    from: Option<usize>,
}

async fn run_log(AxPath(id): AxPath<String>, Query(q): Query<LogQ>) -> Response {
    let Some(info) = RUNS.lock().unwrap().get(&id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown run"}))).into_response();
    };
    let from = q.from.unwrap_or(0);
    let sink = hlog::named(&id);
    let lines = sink.lines_from(from);
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({"from": from, "next": from + lines.len(), "lines": lines, "status": info.status, "error": info.error, "project": info.project, "goal_id": info.goal_id})),
    )
        .into_response()
}

async fn run_events(AxPath(id): AxPath<String>, Query(q): Query<LogQ>) -> Response {
    let Some(info) = RUNS.lock().unwrap().get(&id).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "unknown run"}))).into_response();
    };
    let from = q.from.unwrap_or(0);
    let evs = hlog::named(&id).events_from(from);
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(json!({"from": from, "next": from + evs.len(), "events": evs, "run": info})),
    )
        .into_response()
}

async fn cancel_run(AxPath(id): AxPath<String>) -> ApiResult {
    let running = RUNS.lock().unwrap().get(&id).map(|r| r.status == "running").unwrap_or(false);
    if !running {
        return Err(anyhow::anyhow!("run {id} is not running").into());
    }
    hlog::named(&id).cancel();
    Ok(Json(json!({"ok": true, "note": "stopping after the current model call"})))
}
