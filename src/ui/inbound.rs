//! Goals from outside the UI: a token-protected webhook, GitHub issues, and schedules.
//! Each one goes through the same intake gate as a goal typed into the UI, and its result is
//! sent back where it came from.

use super::{intake_gate, parse_modes, start_run_with, ApiResult, Reply, RUNS};
use crate::benchmarks::goal::{self, GoalAdapter};
use crate::core::spec::BenchmarkAdapter;
use crate::{config, util};
use anyhow::{bail, Result};
use axum::body::Bytes;
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

/// Add a goal to a project and start it.
pub(crate) fn submit(project: &str, text: &str, domain: &str, rubric: &str, k: usize, mode: &str, reply: Option<Reply>, source: &str) -> Result<(String, String)> {
    let a = GoalAdapter::open(&goal::project_dir(project)?)?;
    intake_gate(project, &a.settings, text, &[], k)?;
    a.preflight()?;
    let id = a.add_task(text, domain, rubric, &[])?;
    let run = start_run_with(project, &id, k, parse_modes(mode)?, "", reply, source)?;
    Ok((id, run))
}

fn bearer_ok(h: &HeaderMap, env: &str) -> Result<()> {
    let Some(want) = config::env(env).filter(|t| t.len() >= 16) else {
        bail!("this endpoint is off: set {env} (16+ characters) to enable it");
    };
    let got = h.get("authorization").and_then(|v| v.to_str().ok()).and_then(|v| v.strip_prefix("Bearer ")).unwrap_or("");
    if !ct_eq(got.as_bytes(), want.as_bytes()) {
        bail!("bad or missing bearer token");
    }
    Ok(())
}

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// HMAC-SHA256, hex.
pub fn hmac_sha256_hex(key: &[u8], msg: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut k = if key.len() > 64 { Sha256::digest(key).to_vec() } else { key.to_vec() };
    k.resize(64, 0);
    let ipad: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = k.iter().map(|b| b ^ 0x5c).collect();
    let inner = Sha256::new().chain_update(&ipad).chain_update(msg).finalize();
    let outer = Sha256::new().chain_update(&opad).chain_update(inner).finalize();
    outer.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Deserialize)]
pub struct HookGoal {
    #[serde(default = "default_project")]
    project: String,
    goal: String,
    #[serde(default)]
    domain: String,
    #[serde(default)]
    rubric: String,
    #[serde(default = "one")]
    k: usize,
    #[serde(default = "parallel")]
    mode: String,
    /// POSTed the result as JSON when the run ends.
    #[serde(default)]
    reply_url: String,
}
fn default_project() -> String {
    "default".into()
}
fn one() -> usize {
    1
}
fn parallel() -> String {
    "parallel".into()
}

/// POST /api/hooks/goal  (Authorization: Bearer $HARNESS2_HOOK_TOKEN)
pub(crate) async fn hook_goal(h: HeaderMap, Json(g): Json<HookGoal>) -> ApiResult {
    super::blocking(move || {
        bearer_ok(&h, "HARNESS2_HOOK_TOKEN")?;
        let reply = (!g.reply_url.is_empty()).then(|| Reply { kind: "webhook".into(), url: g.reply_url.clone(), repo: String::new(), issue: 0 });
        if !g.reply_url.is_empty() && !g.reply_url.starts_with("http") {
            bail!("reply_url must be an http(s) URL");
        }
        let (goal_id, run_id) = submit(&g.project, &g.goal, &g.domain, &g.rubric, g.k.clamp(1, 8), &g.mode, reply, "webhook")?;
        Ok(json!({"goal_id": goal_id, "run_id": run_id}))
    })
    .await
}

/// POST /api/hooks/github — a GitHub webhook (issues, issue_comment), signed with
/// HARNESS2_GITHUB_SECRET. An issue labelled HARNESS2_GITHUB_LABEL (default "harnessmith"), or a
/// comment starting with "/harnessmith ", becomes a goal; the result is posted as a comment.
pub(crate) async fn hook_github(h: HeaderMap, body: Bytes) -> ApiResult {
    super::blocking(move || {
        let secret = config::env("HARNESS2_GITHUB_SECRET").filter(|s| !s.is_empty()).ok_or_else(|| anyhow::anyhow!("this endpoint is off: set HARNESS2_GITHUB_SECRET to the webhook's secret"))?;
        let sig = h.get("x-hub-signature-256").and_then(|v| v.to_str().ok()).unwrap_or("");
        let want = format!("sha256={}", hmac_sha256_hex(secret.as_bytes(), &body));
        if !ct_eq(sig.as_bytes(), want.as_bytes()) {
            bail!("bad webhook signature");
        }
        let event = h.get("x-github-event").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
        let v: Value = serde_json::from_slice(&body)?;
        let label = config::env("HARNESS2_GITHUB_LABEL").unwrap_or_else(|| "harnessmith".into());
        let repo = v["repository"]["full_name"].as_str().unwrap_or("").to_string();
        let issue = v["issue"]["number"].as_u64().unwrap_or(0);
        let project = config::env("HARNESS2_GITHUB_PROJECT").unwrap_or_else(|| repo.replace('/', "-"));
        let goal = match (event.as_str(), v["action"].as_str().unwrap_or("")) {
            ("issues", "opened") | ("issues", "labeled") => {
                let has = v["issue"]["labels"].as_array().map(|ls| ls.iter().any(|l| l["name"] == label.as_str())).unwrap_or(false);
                let just = event == "issues" && v["action"] == "labeled" && v["label"]["name"] != label.as_str();
                if !has || just {
                    return Ok(json!({"ignored": format!("issue is not labelled {label:?}")}));
                }
                format!("{}\n\n{}", v["issue"]["title"].as_str().unwrap_or(""), v["issue"]["body"].as_str().unwrap_or(""))
            }
            ("issue_comment", "created") => match v["comment"]["body"].as_str().unwrap_or("").trim().strip_prefix("/harnessmith ") {
                Some(rest) => rest.to_string(),
                None => return Ok(json!({"ignored": "comment is not a /harnessmith command"})),
            },
            ("ping", _) => return Ok(json!({"ok": "pong"})),
            _ => return Ok(json!({"ignored": format!("event {event}")})),
        };
        if repo.is_empty() || issue == 0 {
            bail!("payload has no repository or issue");
        }
        let reply = Some(Reply { kind: "github".into(), url: String::new(), repo: repo.clone(), issue });
        let (goal_id, run_id) = submit(&project, goal.trim(), "github", "", 1, "parallel", reply, "github")?;
        crate::say!("github: {repo}#{issue} → goal {goal_id} (run {run_id})");
        Ok(json!({"goal_id": goal_id, "run_id": run_id}))
    })
    .await
}

/// Every 30 s: start each enabled schedule that is due (one run per schedule at a time).
pub fn start_scheduler() {
    std::thread::spawn(|| {
        let mut n = 0u64;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(30));
            let _ = tick();
            // Every five minutes: let an idle project with unrehearsed experience sleep.
            n += 1;
            if n % 10 == 0 {
                super::boost::auto_sleep_tick();
            }
        }
    });
}

fn tick() -> Result<()> {
    let now = chrono::Utc::now();
    for dir in std::fs::read_dir(goal::projects_root())?.flatten() {
        let name = dir.file_name().to_string_lossy().to_string();
        let Ok(a) = GoalAdapter::open(&dir.path()) else { continue };
        if a.settings.schedules.iter().all(|s| !s.enabled) {
            continue;
        }
        let state_p = dir.path().join("schedules_state.json");
        let mut state: serde_json::Map<String, Value> = std::fs::read_to_string(&state_p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        for s in a.settings.schedules.iter().filter(|s| s.enabled && !s.goal.trim().is_empty()) {
            let every = chrono::Duration::minutes(s.every_minutes.max(5) as i64);
            let last = state.get(&s.id).and_then(|v| v["last"].as_str()).and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok()).map(|t| t.with_timezone(&chrono::Utc));
            if last.map(|l| now - l < every).unwrap_or(false) {
                continue;
            }
            let busy = state.get(&s.id).and_then(|v| v["run"].as_str()).map(|r| RUNS.lock().unwrap().get(r).map(|x| x.status == "running" || x.status == "queued").unwrap_or(false)).unwrap_or(false);
            if busy {
                continue;
            }
            let reply = (!s.reply_url.is_empty()).then(|| Reply { kind: "webhook".into(), url: s.reply_url.clone(), repo: String::new(), issue: 0 });
            let r = submit(&name, &s.goal, &s.domain, "", s.k.clamp(1, 4), &s.mode, reply, "schedule");
            let entry = match &r {
                Ok((g, run)) => json!({"last": util::now_iso(), "goal_id": g, "run": run}),
                Err(e) => json!({"last": util::now_iso(), "error": util::take_chars(&format!("{e}"), 300)}),
            };
            crate::say!("schedule {}/{}: {}", name, s.id, match &r { Ok((_, run)) => format!("started {run}"), Err(e) => format!("not started: {e}") });
            state.insert(s.id.clone(), entry);
        }
        let _ = util::write(&state_p, serde_json::to_string_pretty(&state)?);
    }
    Ok(())
}

/// Schedule state for the UI: last run and next due time per schedule.
pub fn schedule_state(project: &str) -> Value {
    let p = goal::project_dir(project).map(|d| d.join("schedules_state.json"));
    p.ok().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}))
}

#[cfg(test)]
mod tests {
    #[test]
    fn hmac_matches_known_vector() {
        // RFC 4231 test case 2.
        assert_eq!(super::hmac_sha256_hex(b"Jefe", b"what do ya want for nothing?"), "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");
    }
}
