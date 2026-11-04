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

pub mod advisor;
pub mod assist;
pub mod auth;
pub mod files;
pub mod doctor;
pub mod inbound;

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
pub(crate) struct RunInfo {
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
    /// The interrupted run this one resumes, if any.
    #[serde(default)]
    resumed_from: String,
    /// Healing attempts the run made (whole-run retries).
    #[serde(default)]
    attempts: u32,
    /// Where the result goes when the run ends (webhook, GitHub issue), if anywhere.
    #[serde(default)]
    reply: Option<Reply>,
    /// What started it: "ui", "webhook", "github", "schedule".
    #[serde(default)]
    source: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub(crate) struct Reply {
    /// "webhook" | "github".
    kind: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    repo: String,
    #[serde(default)]
    issue: u64,
}

/// At most this many goals run at once; the rest wait as "queued" (HARNESS2_MAX_RUNS).
static RUN_SLOTS: Lazy<crate::core::proposer::Semaphore> = Lazy::new(|| {
    crate::core::proposer::Semaphore::new(crate::config::env_int("HARNESS2_MAX_RUNS", 1).unwrap_or(1).max(1) as usize)
});

fn set_status(id: &str, f: impl FnOnce(&mut RunInfo)) {
    let mut runs = RUNS.lock().unwrap();
    if let Some(r) = runs.get_mut(id) {
        f(r);
        save_run(r);
    }
}

pub(crate) static RUNS: Lazy<Mutex<BTreeMap<String, RunInfo>>> = Lazy::new(|| Mutex::new(BTreeMap::new()));

/// Where each UI run keeps run.json, log.txt and events.jsonl.
fn runs_dir() -> PathBuf {
    crate::config::get().runs.join("ui-runs")
}

pub(crate) fn save_run(r: &RunInfo) {
    let _ = util::write(&runs_dir().join(&r.id).join("run.json"), serde_json::to_string_pretty(r).unwrap_or_default());
}

/// Reload earlier runs so their logs and pipelines survive a restart.
fn load_runs() {
    let mut interrupted = vec![];
    {
        let mut runs = RUNS.lock().unwrap();
        for e in std::fs::read_dir(runs_dir()).into_iter().flatten().flatten() {
            let Ok(t) = std::fs::read_to_string(e.path().join("run.json")) else { continue };
            let Ok(mut r) = serde_json::from_str::<RunInfo>(&t) else { continue };
            if r.status == "running" || r.status == "queued" {
                r.status = "interrupted".into();
                r.error = "the server stopped while this run was in progress".into();
                save_run(&r);
                interrupted.push(r.clone());
            }
            hlog::register(&r.id, hlog::Sink::load(e.path()));
            runs.insert(r.id.clone(), r);
        }
    }
    // Self-healing: pick interrupted runs back up. Completed rollouts and cached improver
    // replies are reused, so a resumed run only redoes what was in flight.
    if crate::config::env("HARNESS2_AUTO_RESUME").map(|v| v != "0").unwrap_or(true) {
        for r in interrupted {
            let modes: Vec<&'static str> = r.modes.iter().filter_map(|m| schedule::mode_token(m).ok()).collect();
            match start_run_from(&r.project, &r.goal_id, r.k, if modes.is_empty() { vec!["agg"] } else { modes }, &r.id) {
                Ok(new) => {
                    println!("auto-resuming interrupted run {} as {new}", r.id);
                    set_status(&r.id, |x| {
                        x.status = "resumed".into();
                        x.error = format!("resumed automatically as {new}");
                    });
                }
                Err(e) => println!("could not auto-resume {}: {e}", r.id),
            }
        }
    }
}

/// Model failures that a fresh attempt can plausibly fix.
fn transient(errors: &[String]) -> bool {
    errors.iter().any(|e| e.contains("execution failed") || e.contains("transport") || e.contains("HTTP 5") || e.contains("timed out") || e.contains("timeout") || e.contains("connection"))
}

pub(crate) struct ApiError(anyhow::Error);
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
pub(crate) type ApiResult = std::result::Result<Json<Value>, ApiError>;

pub(crate) async fn blocking<F: FnOnce() -> Result<Value> + Send + 'static>(f: F) -> ApiResult {
    Ok(Json(tokio::task::spawn_blocking(f).await.map_err(|e| anyhow::anyhow!("{e}"))??))
}

pub fn serve(host: &str, port: u16) -> Result<()> {
    let healed = doctor::heal_all();
    if !healed.is_empty() {
        println!("doctor: {}", healed.join("; "));
    }
    load_runs();
    inbound::start_scheduler();
    // Background healer: every 10 minutes, apply the safe fixes.
    std::thread::spawn(|| loop {
        std::thread::sleep(std::time::Duration::from_secs(600));
        let done = doctor::heal_all();
        if !done.is_empty() {
            println!("doctor: {}", done.join("; "));
        }
    });
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
            .route("/api/health", get(health_snapshot))
            .route("/api/doctor", get(doctor_checks))
            .route("/api/doctor/fix", post(doctor_fix))
            .route("/api/doctor/heal", post(doctor_heal))
            .route("/api/projects", get(list_projects))
            .route("/api/projects/:p", get(get_project))
            .route("/api/projects/:p/settings", put(save_settings))
            .route("/api/projects/:p/base", put(save_base))
            .route("/api/projects/:p/goals", post(new_goal))
            .route("/api/projects/:p/goals/:id", get(get_goal))
            .route("/api/projects/:p/goals/:id/rerun", post(rerun_goal))
            .route("/api/projects/:p/goals/:id/judge", post(judge_goal))
            .route("/api/projects/:p/library", get(get_library))
            .route("/api/projects/:p/estimate", get(estimate))
            .route("/api/projects/:p/gate", get(gate_summary))
            .route("/api/projects/:p/results", get(results))
            .route("/api/projects/:p/assist", post(assist_goal))
            .route("/api/projects/:p/goals/:id/files", get(files::goal_files))
            .route("/api/projects/:p/file", get(files::project_file))
            .route("/api/open", post(files::open_in_os))
            .route("/api/advisor", get(advisor::advisor))
            .route("/api/advisor/bench", post(advisor::bench))
            .route("/api/advisor/pull", post(advisor::pull))
            .route("/api/advisor/pulls", get(advisor::pull_status))
            .route("/api/projects/:p/schedules", get(schedules_get))
            .route("/api/hooks/goal", post(inbound::hook_goal))
            .route("/api/hooks/github", post(inbound::hook_github))
            .route("/api/approvals", get(approvals_list))
            .route("/api/approvals/:id", post(approvals_decide))
            .route("/api/mcp/test", post(mcp_test))
            .route("/api/projects/:p/memory", get(memory_list).post(memory_add))
            .route("/api/projects/:p/memory/:id", put(memory_edit).delete(memory_delete))
            .route("/api/projects/:p/memory/:id/promote", post(memory_promote))
            .route("/api/projects/:p/memory/:id/share", post(memory_share))
            .route("/api/memory/shared", get(memory_shared_list))
            .route("/api/projects/:p/goals/:id/download", get(download))
            .route("/api/runs", get(list_runs))
            .route("/api/runs/:id/log", get(run_log))
            .route("/api/runs/:id/events", get(run_events))
            .route("/api/runs/:id/cancel", post(cancel_run))
            .route("/api/runs/:id/stream", get(run_stream))
            .route("/api/live", get(live_stream))
            .route("/login", get(auth::login_page).post(auth::login_post))
            .route("/logout", get(auth::logout))
            .layer(axum::extract::DefaultBodyLimit::max(200 * 1024 * 1024))
            .layer(axum::middleware::from_fn(auth::guard));
        let listener = tokio::net::TcpListener::bind((host, port)).await?;
        let addr = listener.local_addr()?;
        println!("Harnessmith UI on http://{addr}  (Ctrl-C to stop){}", if auth::password().is_some() { "  · login required" } else { "" });
        if host != "127.0.0.1" && host != "localhost" && auth::password().is_none() {
            println!("WARNING: the UI is open to the network with no login, and it can run shell commands through the agent. Set HARNESS2_UI_PASSWORD (8+ characters), or bind to 127.0.0.1.");
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
        "auth": auth::password().is_some(),
        "sandbox": {"bwrap": crate::llm::sandbox::availability().0, "unshare": crate::llm::sandbox::availability().1},
        "policy": {
            "max_runs": crate::config::env_int("HARNESS2_MAX_RUNS", 1).unwrap_or(1).max(1),
            "run_retries": crate::config::env_int("HARNESS2_RUN_RETRIES", 1).unwrap_or(1).max(0),
            "run_retry_wait_s": crate::config::env_int("HARNESS2_RUN_RETRY_WAIT", 30).unwrap_or(30).max(0),
            "rollout_retry_wait_s": crate::config::env_int("HARNESS2_ROLLOUT_RETRY_WAIT", 10).unwrap_or(10).max(0),
            "auto_resume": crate::config::env("HARNESS2_AUTO_RESUME").map(|v| v != "0").unwrap_or(true),
        },
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

async fn health_snapshot() -> ApiResult {
    Ok(Json(json!({"connections": crate::llm::health::snapshot()})))
}

async fn doctor_checks() -> ApiResult {
    blocking(|| Ok(json!({"checks": doctor::checks()}))).await
}

#[derive(Deserialize)]
struct FixReq {
    id: String,
}

async fn doctor_fix(Json(r): Json<FixReq>) -> ApiResult {
    blocking(move || Ok(json!({"ok": true, "message": doctor::fix(&r.id)?}))).await
}

async fn doctor_heal() -> ApiResult {
    blocking(|| Ok(json!({"ok": true, "applied": doctor::heal_all()}))).await
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
    let running = RUNS.lock().unwrap().values().filter(|r| r.goal_id == id && (r.status == "running" || r.status == "queued")).map(|r| r.id.clone()).next();
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
        // Starter wired tools from the default harness, so older projects can add them.
        let starter: std::collections::BTreeMap<String, String> = crate::substrate::native::default_base_tree()
            .into_iter()
            .filter(|(k, _)| k.starts_with("tools/") || k == "scripts/csv_profile.py")
            .collect();
        Ok(json!({"name": p, "settings": a.settings, "goals": goals, "base": base, "surface": a.surface().always_on, "starter_tools": starter}))
    })
    .await
}

async fn save_settings(AxPath(p): AxPath<String>, Json(s): Json<GoalSettings>) -> ApiResult {
    blocking(move || {
        let bad = crate::llm::gate::check_rules(&s.gate_rules);
        if !bad.is_empty() {
            anyhow::bail!("these gate rules are not valid regexes: {}", bad.join(", "));
        }
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
    start_run_from(project, goal_id, k, modes, "")
}

pub(crate) fn start_run_from(project: &str, goal_id: &str, k: usize, modes: Vec<&'static str>, resumed_from: &str) -> Result<String> {
    let (reply, source) = RUNS.lock().unwrap().get(resumed_from).map(|r| (r.reply.clone(), r.source.clone())).unwrap_or((None, "ui".into()));
    start_run_with(project, goal_id, k, modes, resumed_from, reply, &source)
}

pub(crate) fn start_run_with(project: &str, goal_id: &str, k: usize, modes: Vec<&'static str>, resumed_from: &str, reply: Option<Reply>, source: &str) -> Result<String> {
    let root = goal::project_dir(project)?;
    let a = GoalAdapter::open(&root)?;
    a.preflight()?;
    let t = a.load_task(goal_id)?;
    if RUNS.lock().unwrap().values().any(|r| r.goal_id == goal_id && (r.status == "running" || r.status == "queued")) {
        anyhow::bail!("this goal is already running or queued");
    }
    let id = format!("run-{}-{}", chrono::Utc::now().format("%Y%m%d%H%M%S%3f"), &goal_id[goal_id.len().saturating_sub(6)..]);
    let info = RunInfo {
        id: id.clone(),
        project: project.into(),
        goal_id: goal_id.into(),
        goal: util::take_chars(&t.prompt, 200),
        status: "queued".into(),
        started: util::now_iso(),
        finished: String::new(),
        error: String::new(),
        k,
        modes: modes.iter().map(|m| schedule::mode_name(m).to_string()).collect(),
        resumed_from: resumed_from.into(),
        attempts: 0,
        reply,
        source: source.into(),
    };
    save_run(&info);
    RUNS.lock().unwrap().insert(id.clone(), info);
    let sink = hlog::Sink::persistent(runs_dir().join(&id));
    hlog::register(&id, sink.clone());
    let (rid, gid) = (id.clone(), goal_id.to_string());
    let root_dir = root.clone();
    std::thread::spawn(move || {
        hlog::set_current(Some(sink.clone()));
        let cap = GoalAdapter::open(&root_dir).map(|a| a.settings.max_usd_per_run).unwrap_or(0.0);
        hlog::set_budget(Some(std::sync::Arc::new(hlog::Budget { cap_usd: cap, ..Default::default() })));
        crate::say!("Queued (at most {} run(s) at once; HARNESS2_MAX_RUNS)", crate::config::env_int("HARNESS2_MAX_RUNS", 1).unwrap_or(1).max(1));
        let _slot = RUN_SLOTS.acquire();
        if sink.is_cancelled() {
            set_status(&rid, |r| {
                r.status = "cancelled".into();
                r.finished = util::now_iso();
            });
            crate::say!("Cancelled before it started.");
            return;
        }
        set_status(&rid, |r| r.status = "running".into());
        let retries = crate::config::env_int("HARNESS2_RUN_RETRIES", 1).unwrap_or(1).max(0) as u32;
        let mut attempt = 0u32;
        loop {
            crate::say!("Starting {} k={k} for goal {gid}{}", modes.iter().map(|m| schedule::mode_name(m)).collect::<Vec<_>>().join(" + "), if attempt > 0 { format!(" (healing attempt {attempt})") } else { String::new() });
            let res = crate::cli::run_goal(&a, &gid, k, modes.clone());
            let cancelled = hlog::named(&rid).is_cancelled();
            let (status, errs): (String, Vec<String>) = match &res {
                Ok(v) => (
                    if v["status"] == "ok" || v["status"] == "skipped" { "done".into() } else { "partial".into() },
                    v["errors"].as_array().map(|a| a.iter().filter_map(|e| e.as_str().map(|s| s.to_string())).collect()).unwrap_or_default(),
                ),
                Err(e) => ("failed".into(), vec![format!("{e}")]),
            };
            // Self-healing: a run that failed, or finished partial because a model session or
            // rollout failed, is retried; finished work is reused, so only the broken parts redo.
            let capped = hlog::budget().map(|b| b.exceeded()).unwrap_or(false);
            let status = if capped { "capped".to_string() } else { status };
            let heal = !cancelled && !capped && attempt < retries && (status == "failed" || (status == "partial" && transient(&errs)));
            if heal {
                attempt += 1;
                let wait = crate::config::env_int("HARNESS2_RUN_RETRY_WAIT", 30).unwrap_or(30).max(0) as u64;
                hlog::set_mode("");
                hlog::at("", "");
                hlog::event("heal", "retry_run", json!({"attempt": attempt, "errors": errs, "wait_s": wait}));
                crate::say!("! run ended {status} ({}); healing: retrying in {wait}s, reusing finished work", util::take_chars(&errs.join("; "), 200));
                set_status(&rid, |r| r.attempts = attempt);
                if util::sleep_cancellable(std::time::Duration::from_secs(wait)) {
                    set_status(&rid, |r| {
                        r.finished = util::now_iso();
                        r.status = "cancelled".into();
                        r.error = "stopped while waiting to retry".into();
                    });
                    break;
                }
                continue;
            }
            set_status(&rid, |r| {
                r.finished = util::now_iso();
                r.status = if cancelled { "cancelled".into() } else { status.clone() };
                r.error = errs.join("; ");
                r.attempts = attempt;
            });
            if !cancelled && status != "failed" {
                settle_memory(&a, &gid, &rid);
            }
            if !cancelled && status != "failed" && a.settings.auto_judge && !a.settings.judge_model.trim().is_empty() && a.load_task(&gid).map(|t| !t.rubric.trim().is_empty()).unwrap_or(false) {
                match judge_goal_sync(&a, &gid) {
                    Ok(rows) => crate::say!("judged against the rubric: {}", rows.iter().map(|r| format!("{} {}", r["label"].as_str().unwrap_or(""), r["frac"].as_f64().map(|f| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| "—".into()))).collect::<Vec<_>>().join(" · ")),
                    Err(e) => crate::say!("! auto-judge failed: {e}"),
                }
            }
            deliver_result(&rid, &status, &errs, res.as_ref().ok());
            match res {
                Ok(v) => crate::say!("Finished: {}", v["status"]),
                Err(e) => crate::say!("FAILED: {e}"),
            }
            break;
        }
    });
    Ok(id)
}

/// Send the result where the run's reply points, and to HARNESS2_NOTIFY_URL.
fn deliver_result(rid: &str, status: &str, errs: &[String], v: Option<&Value>) {
    let Some(info) = RUNS.lock().unwrap().get(rid).cloned() else { return };
    let outputs: BTreeMap<String, String> = v
        .and_then(|v| v["deliverables"].as_object())
        .map(|m| m.iter().map(|(k, d)| (k.clone(), d["text"].as_str().unwrap_or("").to_string())).collect())
        .unwrap_or_default();
    crate::llm::approval::notify(
        &format!("harnessmith: run {} on \"{}\" finished: {status}{}", info.id, util::take_chars(&info.goal, 80), if errs.is_empty() { String::new() } else { format!(" ({})", util::take_chars(&errs.join("; "), 200)) }),
        json!({"event": "run_finished", "run_id": info.id, "project": info.project, "status": status}),
    );
    let Some(r) = info.reply.clone() else { return };
    let body = json!({"run_id": info.id, "project": info.project, "goal_id": info.goal_id, "goal": info.goal, "status": status, "errors": errs, "outputs": outputs});
    let res: Result<()> = (|| {
        match r.kind.as_str() {
            "webhook" => {
                ureq::post(&r.url).timeout(std::time::Duration::from_secs(20)).send_json(body.clone())?;
            }
            "github" => {
                let token = crate::config::env("HARNESS2_GITHUB_TOKEN").or_else(|| crate::config::env("GITHUB_TOKEN")).ok_or_else(|| anyhow::anyhow!("set HARNESS2_GITHUB_TOKEN to reply on GitHub"))?;
                let out = outputs.values().next().cloned().unwrap_or_default();
                let md = format!(
                    "**harnessmith** finished this goal: `{status}`{}\n\n{}\n\n<sub>run `{}` · project `{}`</sub>",
                    if errs.is_empty() { String::new() } else { format!("\n\n> {}", util::take_chars(&errs.join("; "), 500)) },
                    if out.trim().is_empty() { "_No output was produced._".to_string() } else { util::take_chars(&out, 60_000) },
                    info.id,
                    info.project
                );
                ureq::post(&format!("https://api.github.com/repos/{}/issues/{}/comments", r.repo, r.issue))
                    .set("Authorization", &format!("Bearer {token}"))
                    .set("Accept", "application/vnd.github+json")
                    .set("User-Agent", "harnessmith")
                    .timeout(std::time::Duration::from_secs(20))
                    .send_json(json!({"body": md}))?;
            }
            _ => {}
        }
        Ok(())
    })();
    match res {
        Ok(()) => crate::say!("result sent to {}", if r.kind == "github" { format!("{}#{}", r.repo, r.issue) } else { r.url.clone() }),
        Err(e) => crate::say!("! could not send the result to {}: {e}", r.kind),
    }
}

/// Memory after a run: keep notes from the adopted attempts (and base runs), archive the rest,
/// and promote the adopted ones into the base harness's LongTermMEMORY.md.
fn settle_memory(a: &GoalAdapter, gid: &str, rid: &str) {
    if !a.settings.memory_enabled {
        return;
    }
    let adopted: Vec<(String, String)> = hlog::named(rid)
        .events_from(0)
        .into_iter()
        .filter(|e| e["kind"] == "select" && e["status"] == "done")
        .filter_map(|e| e["chosen"].as_str().map(|c| (e["mode"].as_str().unwrap_or("").to_string(), c.to_string())))
        .collect();
    let store = crate::llm::memory::Store::open(&a.root);
    let Ok((on, off, won)) = store.settle(gid, &adopted) else { return };
    let faded = store.fade().unwrap_or(0);
    if faded > 0 {
        crate::say!("memory: {faded} unused note(s) faded out");
    }
    let mut promoted = 0;
    if a.settings.memory_promote && !won.is_empty() {
        if let Ok(base) = a.base_harness() {
            promoted = store.promote(&base, &won.iter().map(|n| n.id.clone()).collect::<Vec<_>>()).unwrap_or(0);
        }
    }
    if on + off > 0 {
        hlog::set_mode("");
        hlog::at("", "");
        hlog::event("memory", "settle", json!({"kept": on, "archived": off, "promoted": promoted}));
        crate::say!("memory: kept {on} note(s), archived {off}, promoted {promoted} into LongTermMEMORY.md");
    }
}

pub(crate) fn parse_modes(m: &str) -> Result<Vec<&'static str>> {
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
        intake_gate(&p, &a.settings, &text, &files, k)?;
        a.preflight()?;
        let id = a.add_task(&text, &domain, &rubric, &files)?;
        let run = start_run(&p, &id, k, parse_modes(&mode)?)?;
        Ok(json!({"goal_id": id, "run_id": run}))
    })
    .await
}

/// The intake gate: deterministic checks before a goal is accepted. Every decision is logged
/// to `<runs>/gate_log.jsonl`; a stop returns every reason at once.
fn intake_checks(s: &GoalSettings, text: &str, files: &[(String, Vec<u8>)], k: usize) -> Vec<(String, bool, String)> {
    let mut c: Vec<(String, bool, String)> = vec![];
    let n = text.trim().chars().count();
    c.push(("goal".into(), n > 0 && n <= 20_000, if n == 0 { "the goal is empty".into() } else if n > 20_000 { format!("the goal is {n} chars; the limit is 20000 (attach long material as a file)") } else { format!("{n} chars") }));
    let bytes: usize = files.iter().map(|(_, d)| d.len()).sum();
    let bad_name = files.iter().find(|(f, _)| f.contains("..") || f.starts_with('/'));
    c.push((
        "inputs".into(),
        files.len() <= 200 && bytes <= 200 << 20 && bad_name.is_none(),
        if let Some((f, _)) = bad_name { format!("unsafe file name {f:?}") } else if files.len() > 200 { format!("{} files; the limit is 200", files.len()) } else if bytes > 200 << 20 { format!("{} MB of inputs; the limit is 200 MB", bytes >> 20) } else { format!("{} file(s), {} KB", files.len(), bytes / 1024) },
    ));
    let kmax = crate::config::env_int("HARNESS2_MAX_K", 4).unwrap_or(4).max(1) as usize;
    c.push(("budget".into(), k <= kmax, if k <= kmax { format!("k={k}") } else { format!("k={k} is over the limit of {kmax} (HARNESS2_MAX_K)") }));
    for (role, id) in [("solver", &s.solver_model), ("improver", &s.improver_model), ("selector", &s.selector_model)] {
        let ok = match crate::llm::agent::resolve_chain(id) {
            Ok(ch) if !ch.is_empty() => (true, format!("{} model(s) in chain", ch.len())),
            Ok(_) => (false, "no model set".into()),
            Err(e) => (false, util::take_chars(&format!("{e}"), 200)),
        };
        c.push((format!("{role} model"), ok.0, if id.trim().is_empty() { "not set (Models tab)".into() } else { ok.1 }));
    }
    let bad = crate::llm::gate::check_rules(&s.gate_rules);
    c.push(("gate rules".into(), bad.is_empty(), if bad.is_empty() { format!("mode {} · {} custom rule(s)", crate::llm::gate::Mode::parse(&s.gate_mode).as_str(), s.gate_rules.iter().filter(|r| !r.trim().is_empty()).count()) } else { format!("invalid regex: {}", bad.join(", ")) }));
    c
}

fn gate_log(entry: Value) {
    let p = crate::config::get().runs.join("gate_log.jsonl");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
        use std::io::Write;
        let _ = writeln!(f, "{entry}");
    }
}

pub(crate) fn intake_gate(project: &str, s: &GoalSettings, text: &str, files: &[(String, Vec<u8>)], k: usize) -> Result<()> {
    let checks = intake_checks(s, text, files, k);
    let failed: Vec<String> = checks.iter().filter(|(_, ok, _)| !ok).map(|(n, _, d)| format!("{n}: {d}")).collect();
    let decision = if failed.is_empty() { "accept" } else { "stop" };
    gate_log(json!({"at": util::now_iso(), "project": project, "decision": decision, "goal": util::take_chars(text.trim(), 120), "reasons": failed}));
    if !failed.is_empty() {
        anyhow::bail!("Gate stopped this goal — {}", failed.join("; "));
    }
    Ok(())
}

/// Gate summary for the Overview: modes, rules, intake checks for the current settings, history.
async fn gate_summary(AxPath(p): AxPath<String>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let s = &a.settings;
        let checks: Vec<Value> = intake_checks(s, "preview", &[], 1).into_iter().filter(|(n, _, _)| n != "goal").map(|(n, ok, d)| json!({"name": n, "ok": ok, "detail": d})).collect();
        let log = std::fs::read_to_string(crate::config::get().runs.join("gate_log.jsonl")).unwrap_or_default();
        let entries: Vec<Value> = log.lines().filter_map(|l| serde_json::from_str::<Value>(l).ok()).filter(|e| e["project"] == p.as_str()).collect();
        let accepted = entries.iter().filter(|e| e["decision"] == "accept").count();
        // Tool-gate decisions across this project's runs.
        let (mut blocked, mut would) = (0usize, 0usize);
        let ids: Vec<String> = RUNS.lock().map(|rs| rs.values().filter(|r| r.project == p).map(|r| r.id.clone()).collect()).unwrap_or_default();
        for id in ids {
            let body = std::fs::read_to_string(runs_dir().join(&id).join("events.jsonl")).unwrap_or_default();
            for e in body.lines().filter(|l| l.contains("\"gate\"")).filter_map(|l| serde_json::from_str::<Value>(l).ok()) {
                if e["kind"] == "gate" {
                    if e["status"] == "blocked" { blocked += 1 } else { would += 1 }
                }
            }
        }
        Ok(json!({
            "mode": crate::llm::gate::Mode::parse(&s.gate_mode).as_str(),
            "protect_inputs": s.gate_protect_inputs,
            "rules": crate::llm::gate::DEFAULT_RULES.iter().map(|(id, re, why)| json!({"id": id, "regex": re, "reason": why})).collect::<Vec<_>>(),
            "custom_rules": s.gate_rules,
            "checks": checks,
            "intake": {"accepted": accepted, "stopped": entries.len() - accepted, "recent": entries.iter().rev().take(6).cloned().collect::<Vec<_>>()},
            "tool": {"blocked": blocked, "would_block": would},
        }))
    })
    .await
}

// ---------- goal assistant ----------
#[derive(Deserialize)]
struct AssistReq {
    text: String,
    #[serde(default)]
    files: Vec<String>,
}

async fn assist_goal(AxPath(p): AxPath<String>, Json(r): Json<AssistReq>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let chain = if a.settings.improver_model.trim().is_empty() { a.settings.solver_model.clone() } else { a.settings.improver_model.clone() };
        assist::expand(&chain, &r.text, &r.files)
    })
    .await
}

// ---------- approvals, MCP, schedules ----------
async fn approvals_list() -> ApiResult {
    Ok(Json(json!({"pending": crate::llm::approval::list(), "timeout_s": crate::llm::approval::timeout().as_secs()})))
}

#[derive(Deserialize)]
struct Decide {
    allow: bool,
}

async fn approvals_decide(AxPath(id): AxPath<String>, Json(d): Json<Decide>) -> ApiResult {
    if !crate::llm::approval::decide(&id, d.allow) {
        return Err(anyhow::anyhow!("no pending approval {id} (answered already, or timed out)").into());
    }
    Ok(Json(json!({"ok": true})))
}

async fn mcp_test(Json(c): Json<crate::llm::mcp::ServerCfg>) -> ApiResult {
    blocking(move || {
        let tools = crate::llm::mcp::probe(&c)?;
        Ok(json!({"tools": tools.into_iter().map(|(n, d)| json!({"name": n, "description": d})).collect::<Vec<_>>()}))
    })
    .await
}

async fn schedules_get(AxPath(p): AxPath<String>) -> ApiResult {
    Ok(Json(json!({"state": inbound::schedule_state(&p), "hooks": {
        "goal": crate::config::env("HARNESS2_HOOK_TOKEN").map(|t| t.len() >= 16).unwrap_or(false),
        "github": crate::config::env("HARNESS2_GITHUB_SECRET").is_some(),
        "github_reply": crate::config::env("HARNESS2_GITHUB_TOKEN").or_else(|| crate::config::env("GITHUB_TOKEN")).is_some(),
        "notify": crate::config::env("HARNESS2_NOTIFY_URL").is_some(),
    }})))
}

// ---------- memory API ----------
fn mem_store(p: &str) -> Result<(GoalAdapter, crate::llm::memory::Store)> {
    let a = GoalAdapter::open(&goal::project_dir(p)?)?;
    let st = crate::llm::memory::Store::open(&a.root);
    Ok((a, st))
}

async fn memory_list(AxPath(p): AxPath<String>) -> ApiResult {
    blocking(move || {
        let (_, st) = mem_store(&p)?;
        let mut notes = st.load();
        notes.iter_mut().for_each(|n| {
            n.vec_model = if n.vec.is_empty() { String::new() } else { n.vec_model.clone() };
            n.vec.clear();
        });
        notes.sort_by(|a, b| b.pinned.cmp(&a.pinned).then(b.created.cmp(&a.created)));
        let count = |s: &str| notes.iter().filter(|n| n.status == s).count();
        Ok(json!({
            "stats": {"active": count("active"), "candidate": count("candidate"), "archived": count("archived"), "pinned": notes.iter().filter(|n| n.pinned && n.status != "archived").count(), "promoted": notes.iter().filter(|n| n.promoted).count(), "uses": notes.iter().map(|n| n.uses as u64).sum::<u64>()},
            "notes": notes,
        }))
    })
    .await
}

#[derive(Deserialize)]
struct MemAdd {
    text: String,
    #[serde(default)]
    tags: Vec<String>,
}

async fn memory_add(AxPath(p): AxPath<String>, Json(m): Json<MemAdd>) -> ApiResult {
    blocking(move || {
        let (_, st) = mem_store(&p)?;
        let (n, dup) = st.remember(&m.text, &m.tags, crate::llm::memory::Source { by: "user".into(), ..Default::default() }, "active")?;
        Ok(json!({"note": n, "dup": dup}))
    })
    .await
}

#[derive(Deserialize)]
struct MemEdit {
    text: Option<String>,
    tags: Option<Vec<String>>,
    pinned: Option<bool>,
    status: Option<String>,
}

async fn memory_edit(AxPath((p, id)): AxPath<(String, String)>, Json(m): Json<MemEdit>) -> ApiResult {
    blocking(move || {
        let (_, st) = mem_store(&p)?;
        st.update(|notes| {
            let n = notes.iter_mut().find(|n| n.id == id).ok_or_else(|| anyhow::anyhow!("no note {id}"))?;
            if let Some(t) = m.text {
                let t = t.trim().to_string();
                if t.chars().count() < 8 || t.chars().count() > crate::llm::memory::MAX_NOTE_CHARS {
                    anyhow::bail!("a note is 8 to {} characters", crate::llm::memory::MAX_NOTE_CHARS);
                }
                n.text = t;
            }
            if let Some(t) = m.tags {
                n.tags = t.into_iter().map(|x| x.trim().to_lowercase()).filter(|x| !x.is_empty()).collect();
            }
            if let Some(pn) = m.pinned {
                n.pinned = pn;
            }
            if let Some(s) = m.status {
                if !["active", "candidate", "archived"].contains(&s.as_str()) {
                    anyhow::bail!("status is active, candidate or archived");
                }
                n.note = if s == "archived" { "archived by hand".into() } else { String::new() };
                n.status = s;
            }
            Ok(json!({"note": n.clone()}))
        })
    })
    .await
}

async fn memory_delete(AxPath((p, id)): AxPath<(String, String)>) -> ApiResult {
    blocking(move || {
        let (_, st) = mem_store(&p)?;
        st.update(|notes| {
            let before = notes.len();
            notes.retain(|n| n.id != id);
            Ok(json!({"deleted": before - notes.len()}))
        })
    })
    .await
}

async fn memory_share(AxPath((p, id)): AxPath<(String, String)>) -> ApiResult {
    blocking(move || {
        let (_, st) = mem_store(&p)?;
        let n = st.share(&id, &p)?;
        Ok(json!({"shared": n.id}))
    })
    .await
}

async fn memory_shared_list() -> ApiResult {
    blocking(move || {
        let st = crate::llm::memory::Store { path: crate::llm::memory::shared_path(), embed: None, shared: None };
        let mut notes = st.load();
        notes.iter_mut().for_each(|n| n.vec.clear());
        Ok(json!({"notes": notes}))
    })
    .await
}

async fn memory_promote(AxPath((p, id)): AxPath<(String, String)>) -> ApiResult {
    blocking(move || {
        let (a, st) = mem_store(&p)?;
        let n = st.promote(&a.base_harness()?, &[id])?;
        Ok(json!({"promoted": n}))
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
                let base_rd = summary["base"][0]["run_dir"].as_str().unwrap_or("").to_string();
                let base_files = if base_rd.is_empty() { json!({}) } else { json!(a.deliverable_files(Path::new(&base_rd)).unwrap_or_default()) };
                out["results"][&mode] = json!({
                    "base_files": base_files,
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
        Ok(json!({"rows": judge_goal_sync(&a, &id)?}))
    })
    .await
}

/// Grade the adopted output of every mode, and the unmodified base attempt, against the
/// goal's rubric with the judge model; saved to the goal's judged.json.
pub(crate) fn judge_goal_sync(a: &GoalAdapter, id: &str) -> Result<Vec<Value>> {
    let mut sels = vec![];
    let mut base: Option<(String, String, PathBuf)> = None;
    for dir in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
        if let Ok(s) = std::fs::read_to_string(dir.path().join(id).join("summary.json")) {
            let v: Value = serde_json::from_str(&s)?;
            for (_m, sel) in v["selections"].as_object().cloned().unwrap_or_default() {
                sels.push(sel);
            }
            if base.is_none() {
                if let Some(b) = v["base"].get(0) {
                    base = Some((b["label"].as_str().unwrap_or("base1").into(), b["run_id"].as_str().unwrap_or("").into(), PathBuf::from(b["run_dir"].as_str().unwrap_or(""))));
                }
            }
        }
    }
    let mut rows: Vec<Value> = judge::judge_selections(a, &sels, 4).iter().map(|r| r.to_json()).collect();
    if let Some((label, run_id, dir)) = base.filter(|b| b.2.is_dir()) {
        let mut r = judge::judge_one(a, id, "base", 1, &label, &run_id, &dir, true, "").to_json();
        r["is_base_attempt"] = json!(true);
        rows.push(r);
    }
    util::write(&a.task_dir(id).join("judged.json"), serde_json::to_string_pretty(&rows)?)?;
    Ok(rows)
}

/// Per-goal results for the dashboard: adopted vs base scores, cost, time, heals and gates.
async fn results(AxPath(p): AxPath<String>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let runs: Vec<RunInfo> = RUNS.lock().unwrap().values().filter(|r| r.project == p).cloned().collect();
        let mut out = vec![];
        for t in a.all_tasks() {
            let judged: Vec<Value> = std::fs::read_to_string(a.task_dir(&t.id).join("judged.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            let base_frac = judged.iter().find(|r| r["is_base_attempt"] == true).and_then(|r| r["frac"].as_f64());
            let adopted: Vec<&Value> = judged.iter().filter(|r| r["is_base_attempt"] != true).collect();
            let adopted_frac = adopted.iter().filter_map(|r| r["frac"].as_f64()).fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))));
            let mut chosen = vec![];
            let mut chosen_is_base = false;
            let mut status = String::new();
            for dir in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
                if let Ok(s) = std::fs::read_to_string(dir.path().join(&t.id).join("summary.json")) {
                    if let Ok(v) = serde_json::from_str::<Value>(&s) {
                        status = v["status"].as_str().unwrap_or("").into();
                        for (m, sel) in v["selections"].as_object().cloned().unwrap_or_default() {
                            chosen.push(format!("{m}:{}", sel["chosen_label"].as_str().unwrap_or("")));
                            chosen_is_base |= sel["chosen_is_base"].as_bool().unwrap_or(false) || sel["chosen_label"].as_str().map(|l| l.starts_with("base")).unwrap_or(false);
                        }
                    }
                }
            }
            let last = runs.iter().filter(|r| r.goal_id == t.id && !r.finished.is_empty()).max_by(|x, y| x.started.cmp(&y.started));
            let (mut usd, mut heals, mut gates, mut minutes) = (0.0, 0, 0, None);
            if let Some(r) = last {
                for e in hlog::named(&r.id).events_from(0) {
                    match (e["kind"].as_str(), e["status"].as_str()) {
                        (Some("step"), Some("end")) => usd += e["usd"].as_f64().unwrap_or(0.0),
                        (Some("heal"), _) => heals += 1,
                        (Some("gate"), _) => gates += 1,
                        _ => {}
                    }
                }
                if let (Ok(s), Ok(f)) = (chrono::DateTime::parse_from_rfc3339(&r.started), chrono::DateTime::parse_from_rfc3339(&r.finished)) {
                    minutes = Some((f - s).num_seconds() as f64 / 60.0);
                }
            }
            let verdict = match (base_frac, adopted_frac) {
                (Some(b), Some(x)) if x > b + 1e-9 => "win",
                (Some(b), Some(x)) if x < b - 1e-9 => "loss",
                (Some(_), Some(_)) => "tie",
                _ => "",
            };
            out.push(json!({"id": t.id, "prompt": util::take_chars(&t.prompt, 160), "domain": t.domain, "created_at": t.created_at, "has_rubric": !t.rubric.is_empty(),
                "status": status, "chosen": chosen, "chosen_is_base": chosen_is_base, "base_score": base_frac, "adopted_score": adopted_frac, "verdict": verdict,
                "usd": (usd * 10000.0).round() / 10000.0, "minutes": minutes, "heals": heals, "gates": gates, "run_status": last.map(|r| r.status.clone())}));
        }
        Ok(json!({"goals": out}))
    })
    .await
}

#[derive(Deserialize)]
struct EstQ {
    k: Option<usize>,
    mode: Option<String>,
}

/// Time and cost before Start: from this project's finished runs when there are any (mean
/// solver-rollout and improver-session durations and tokens), else from measured model speed.
async fn estimate(AxPath(p): AxPath<String>, Query(q): Query<EstQ>) -> ApiResult {
    blocking(move || {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let k = q.k.unwrap_or(1).max(1);
        let modes = if q.mode.as_deref() == Some("both") { 2 } else { 1 };
        let n_runs = k + modes * 3 * k;
        let n_sessions = modes * (3 * k + if k > 1 { 1 } else { 0 } + 1);
        let (mut roll, mut sess, mut roll_tok, mut sess_tok, mut runs_seen) = (vec![], vec![], vec![], vec![], 0);
        let ids: Vec<String> = RUNS.lock().unwrap().values().filter(|r| r.project == p && r.status == "done").map(|r| r.id.clone()).collect();
        for id in ids {
            runs_seen += 1;
            let evs = hlog::named(&id).events_from(0);
            let mut starts: BTreeMap<String, i64> = BTreeMap::new();
            let mut toks: BTreeMap<String, u64> = BTreeMap::new();
            for e in &evs {
                let key = format!("{}|{}|{}", e["mode"].as_str().unwrap_or(""), e["node"].as_str().unwrap_or(""), e["kind"].as_str().unwrap_or(""));
                let t = e["t"].as_i64().unwrap_or(0);
                match (e["kind"].as_str(), e["status"].as_str()) {
                    (Some("stage" | "rollout"), Some("start")) => {
                        starts.insert(key, t);
                    }
                    (Some("step"), Some("end")) if e["depth"].as_u64().unwrap_or(0) == 0 => {
                        let ph = if e["phase"] == "run" { "rollout" } else { "stage" };
                        let k2 = format!("{}|{}|{ph}", e["mode"].as_str().unwrap_or(""), e["node"].as_str().unwrap_or(""));
                        toks.insert(k2, e["in"].as_u64().unwrap_or(0) + e["out"].as_u64().unwrap_or(0));
                    }
                    (Some("rollout"), Some("done")) | (Some("stage"), Some("done" | "fell_back")) => {
                        if let Some(t0) = starts.get(&key) {
                            let d = (t - t0) as f64 / 1000.0;
                            let tk = *toks.get(&key).unwrap_or(&0) as f64;
                            if e["kind"] == "rollout" {
                                roll.push(d);
                                roll_tok.push(tk);
                            } else {
                                sess.push(d);
                                sess_tok.push(tk);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        let mean = |v: &Vec<f64>, d: f64| if v.is_empty() { d } else { v.iter().sum::<f64>() / v.len() as f64 };
        // Seconds per model call for a role: the measured average latency of its first model.
        let latency = |m: &str| -> f64 {
            let id = m.split(',').next().unwrap_or("").trim();
            connections::resolve(id).map(|(c, _)| crate::llm::health::get(&c.name).latency_ms / 1000.0).unwrap_or(0.0)
        };
        let (sl, il) = (latency(&a.settings.solver_model), latency(&a.settings.improver_model));
        let basis = if !roll.is_empty() { format!("{runs_seen} earlier run(s) in this project") } else if sl > 0.0 { "measured model latency".into() } else { "rough defaults (no history yet)".into() };
        // Without history: about 70% of the step budget (capped at 15 steps) at the measured
        // per-call latency, or 15 s per call when the model has not been called yet.
        let steps = |n: u32| (n.min(15) as f64 * 0.7).max(3.0);
        let roll_s = mean(&roll, steps(a.settings.solver_steps) * if sl > 0.0 { sl } else { 15.0 });
        let sess_s = mean(&sess, steps(a.settings.improver_steps) * if il > 0.0 { il } else { 15.0 });
        let width = a.settings.parallelism.max(1) as f64;
        let minutes = (k as f64 * roll_s + (n_runs - k) as f64 * roll_s / width.min(2.0) + n_sessions as f64 * sess_s / width.min(2.0)) / 60.0;
        let price = |m: &str| -> (f64, f64) {
            let id = m.split(',').next().unwrap_or("").trim();
            connections::resolve(id).map(|(c, _)| (c.input_price, c.output_price)).unwrap_or((0.0, 0.0))
        };
        let (si, so) = price(&a.settings.solver_model);
        let (ii, io) = price(&a.settings.improver_model);
        let rt = mean(&roll_tok, 20_000.0);
        let st = mean(&sess_tok, 30_000.0);
        // ~92% of tokens are prompt tokens in these sessions.
        let usd = n_runs as f64 * rt * (0.92 * si + 0.08 * so) / 1e6 + n_sessions as f64 * st * (0.92 * ii + 0.08 * io) / 1e6;
        Ok(json!({"runs": n_runs, "sessions": n_sessions, "minutes": (minutes * 10.0).round() / 10.0, "usd": (usd * 1000.0).round() / 1000.0, "basis": basis, "tokens": (n_runs as f64 * rt + n_sessions as f64 * st).round()}))
    })
    .await
}

#[derive(Deserialize)]
struct DlQ {
    mode: Option<String>,
    what: Option<String>,
}

/// The adopted output files, or the adopted harness, as a .tar.gz.
async fn download(AxPath((p, id)): AxPath<(String, String)>, Query(q): Query<DlQ>) -> Response {
    let res = tokio::task::spawn_blocking(move || -> Result<(String, Vec<u8>)> {
        let a = GoalAdapter::open(&goal::project_dir(&p)?)?;
        let mode = q.mode.unwrap_or_else(|| "agg".into());
        let what = q.what.unwrap_or_else(|| "output".into());
        let mut dir = None;
        for d in std::fs::read_dir(a.root.join("runs/adapt")).into_iter().flatten().flatten() {
            let Ok(t) = std::fs::read_to_string(d.path().join(&id).join("summary.json")) else { continue };
            let v: Value = serde_json::from_str(&t)?;
            let sel = &v["selections"][&mode];
            let ns = d.file_name().to_string_lossy().replacen("-agg", &format!("-{mode}"), 1);
            dir = Some(if what == "harness" {
                a.root.join("runs/adapt").join(ns).join(&id).join("harnesses").join(sel["chosen_label"].as_str().unwrap_or(""))
            } else {
                PathBuf::from(sel["chosen_run_dir"].as_str().unwrap_or("")).join("workspace/output")
            });
        }
        let dir = dir.filter(|d| d.is_dir()).ok_or_else(|| anyhow::anyhow!("nothing to download yet"))?;
        let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        {
            let mut tar = tar::Builder::new(&mut gz);
            tar.append_dir_all(&what, &dir)?;
            tar.finish()?;
        }
        Ok((format!("{id}-{mode}-{what}.tar.gz"), gz.finish()?))
    })
    .await;
    match res {
        Ok(Ok((name, bytes))) => ([(header::CONTENT_TYPE, "application/gzip".to_string()), (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{name}\""))], bytes).into_response(),
        Ok(Err(e)) => (StatusCode::NOT_FOUND, Json(json!({"error": format!("{e}")}))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": format!("{e}")}))).into_response(),
    }
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
        .map(|r| json!({"id": r.id, "project": r.project, "goal_id": r.goal_id, "goal": r.goal, "status": r.status, "started": r.started, "finished": r.finished, "error": r.error, "k": r.k, "modes": r.modes, "lines": hlog::named(&r.id).len(), "events": hlog::named(&r.id).n_events(), "resumed_from": r.resumed_from, "attempts": r.attempts}))
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

// ---------- push updates (server-sent events) ----------
type SseStream = std::pin::Pin<Box<dyn futures_util::Stream<Item = std::result::Result<axum::response::sse::Event, std::convert::Infallible>> + Send>>;

/// One run's new log lines and events as they happen; ends when the run does.
async fn run_stream(AxPath(id): AxPath<String>, Query(q): Query<StreamQ>) -> axum::response::Sse<SseStream> {
    use axum::response::sse::{Event, KeepAlive, Sse};
    let st = (id, q.log.unwrap_or(0), q.ev.unwrap_or(0), false);
    let s = futures_util::stream::unfold(st, |(id, mut lf, mut ef, done)| async move {
        if done {
            return None;
        }
        loop {
            let info = RUNS.lock().unwrap().get(&id).cloned();
            let Some(info) = info else {
                return Some((Ok(Event::default().event("gone").data("{}")), (id, lf, ef, true)));
            };
            let sink = hlog::named(&id);
            let lines = sink.lines_from(lf);
            let evs = sink.events_from(ef);
            let finished = info.status != "running" && info.status != "queued";
            if !lines.is_empty() || !evs.is_empty() || finished {
                lf += lines.len();
                ef += evs.len();
                let data = json!({"lines": lines, "events": evs, "status": info.status, "error": info.error, "project": info.project, "goal_id": info.goal_id, "log_next": lf, "ev_next": ef});
                return Some((Ok(Event::default().event("tick").data(data.to_string())), (id, lf, ef, finished)));
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
    });
    Sse::new(Box::pin(s) as SseStream).keep_alive(KeepAlive::default())
}

#[derive(Deserialize)]
struct StreamQ {
    log: Option<usize>,
    ev: Option<usize>,
}

/// Where each running run is right now: active stages with their model and last tool call,
/// plus the latest notable events (heals, gate decisions, memory, guard).
fn live_snapshot() -> Value {
    let runs: Vec<RunInfo> = RUNS.lock().unwrap().values().filter(|r| r.status == "running" || r.status == "queued").cloned().collect();
    let mut out = vec![];
    for r in runs {
        let evs = hlog::named(&r.id).events_from(0);
        let mut active: BTreeMap<String, Value> = BTreeMap::new();
        let mut notable = vec![];
        let (mut runs_done, mut sessions_done) = (0, 0);
        for e in &evs {
            let node = e["node"].as_str().unwrap_or("").to_string();
            let key = format!("{}|{}", e["mode"].as_str().unwrap_or(""), node);
            let kind = e["kind"].as_str().unwrap_or("");
            let status = e["status"].as_str().unwrap_or("");
            match (kind, status) {
                ("stage" | "rollout" | "select" | "learn", "start") => {
                    active.insert(key, json!({"node": node, "mode": e["mode"], "phase": e["phase"], "kind": kind, "since": e["t"], "model": "", "tool": ""}));
                }
                ("stage" | "rollout" | "select" | "learn", "done" | "failed" | "fell_back" | "retry") => {
                    if status != "retry" {
                        active.remove(&key);
                        if kind == "rollout" { runs_done += 1 } else { sessions_done += 1 }
                    }
                }
                ("step", "call") => {
                    if let Some(a) = active.get_mut(&key) {
                        a["model"] = e["model"].clone();
                        a["calling"] = json!(true);
                        a["call_t"] = e["t"].clone();
                        a["step"] = e["step"].clone();
                    }
                }
                ("step", "turn") => {
                    if let Some(a) = active.get_mut(&key) {
                        a["calling"] = json!(false);
                        a["model"] = e["model"].clone();
                        a["tool"] = e["tools"].get(0).cloned().unwrap_or(Value::Null);
                        a["step"] = e["step"].clone();
                        a["last"] = e["t"].clone();
                    }
                }
                ("heal" | "gate" | "memory", _) | ("select", "guard") => notable.push(e.clone()),
                _ => {}
            }
        }
        let n = notable.len();
        out.push(json!({"id": r.id, "project": r.project, "goal": r.goal, "status": r.status, "source": r.source,
            "active": active.into_values().collect::<Vec<_>>(), "recent": notable.into_iter().skip(n.saturating_sub(8)).collect::<Vec<_>>(),
            "rollouts_done": runs_done, "sessions_done": sessions_done, "k": r.k, "modes": r.modes}));
    }
    json!({"runs": out, "health": crate::llm::health::snapshot(), "approvals": crate::llm::approval::list().len(), "at": util::now_iso()})
}

/// The whole system, pushed every second for the live Overview.
async fn live_stream() -> axum::response::Sse<SseStream> {
    use axum::response::sse::{Event, KeepAlive, Sse};
    let s = futures_util::stream::unfold(0u64, |n| async move {
        if n > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
        }
        let snap = tokio::task::spawn_blocking(live_snapshot).await.unwrap_or(Value::Null);
        Some((Ok(Event::default().event("live").data(snap.to_string())), n + 1))
    });
    Sse::new(Box::pin(s) as SseStream).keep_alive(KeepAlive::default())
}

async fn cancel_run(AxPath(id): AxPath<String>) -> ApiResult {
    let running = RUNS.lock().unwrap().get(&id).map(|r| r.status == "running" || r.status == "queued").unwrap_or(false);
    if !running {
        return Err(anyhow::anyhow!("run {id} is not running").into());
    }
    hlog::named(&id).cancel();
    Ok(Json(json!({"ok": true, "note": "stopping now: in-flight model calls and commands are being interrupted"})))
}
