//! Improver sessions with file access in an isolated context workspace.
//!
//! OpenCode reads output/answer.md or the final message; the Claude CLI reads the final
//! message; the native `api` channel runs this crate's own agent loop against a model endpoint
//! (`crate::llm`). Prompts use stdin, and empty replies raise an error.

use super::spec::ChannelSpec;
use crate::{config, util};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub const SYSTEM_ARG_MAX: usize = 120_000;

/// A call did not produce a usable reply. Never swallowed into an empty string.
#[derive(Debug, Clone)]
pub struct ProposerError {
    pub message: String,
    pub attempts: u32,
    pub cost_usd: f64,
}
impl std::fmt::Display for ProposerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for ProposerError {}

/// One session: what came back, what it cost, and where the audit trail is.
#[derive(Debug, Clone, Default)]
pub struct Reply {
    pub label: String,
    pub text: String,
    pub ok: bool,
    pub cost_usd: f64,
    pub wall_s: f64,
    pub attempts: u32,
    pub cached: bool,
    pub channel: String,
    pub error: String,
    pub audit: Option<PathBuf>,
    pub workspace: Option<PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct CallOpts {
    pub system: String,
    pub timeout: Option<u64>,
    pub max_attempts: Option<u32>,
    /// A subpath of the session's working directory to copy out before it is destroyed --
    /// e.g. "context/harness", which is how an edited tree comes back.
    pub harvest: Option<String>,
}

/// What the method needs of a proposer. `Proposer` is the real one; the demo supplies a
/// scripted one.
pub trait Propose: Send + Sync {
    fn spec(&self) -> &ChannelSpec;
    fn audit_dir(&self) -> Option<PathBuf>;
    fn call(&self, mission: &str, context_dir: Option<&Path>, label: &str, opts: CallOpts) -> Result<Reply, ProposerError>;
    fn calls(&self) -> Vec<Reply>;
    /// A proposer for a different channel (the selector), sharing this one's settings.
    fn fork(&self, spec: ChannelSpec, audit_dir: Option<PathBuf>) -> anyhow::Result<Arc<dyn Propose>> {
        Ok(Arc::new(Proposer::new(spec, audit_dir, 2)?))
    }
}

/// A counting semaphore (Python's BoundedSemaphore).
pub struct Semaphore {
    n: Mutex<usize>,
    cv: Condvar,
}
impl Semaphore {
    pub const fn new(n: usize) -> Semaphore {
        Semaphore { n: Mutex::new(n), cv: Condvar::new() }
    }
    pub fn acquire(&self) -> SemGuard<'_> {
        let mut g = self.n.lock().unwrap();
        while *g == 0 {
            g = self.cv.wait(g).unwrap();
        }
        *g -= 1;
        SemGuard(self)
    }
}
pub struct SemGuard<'a>(&'a Semaphore);
impl Drop for SemGuard<'_> {
    fn drop(&mut self) {
        *self.0.n.lock().unwrap() += 1;
        self.0.cv.notify_one();
    }
}

static SEM: Lazy<Semaphore> = Lazy::new(|| {
    let n = config::env_int("HARNESS2_PROPOSER_CONCURRENCY", config::PROPOSER_CONCURRENCY as i64).unwrap_or(2);
    Semaphore::new(if n > 0 { n as usize } else { usize::MAX / 2 })
});

/// Runs agentic sessions on one channel, with an audit trail and a cache.
pub struct Proposer {
    pub spec: ChannelSpec,
    pub audit_dir: Option<PathBuf>,
    pub attempts: u32,
    pub project: String,
    pub region: String,
    calls: Mutex<Vec<Reply>>,
}

impl Proposer {
    pub fn new(spec: ChannelSpec, audit_dir: Option<PathBuf>, attempts: u32) -> anyhow::Result<Proposer> {
        if spec.model.trim().is_empty() {
            anyhow::bail!("Set HARNESS2_IMPROVER_MODEL before starting model sessions");
        }
        let cfg = config::get();
        Ok(Proposer {
            spec,
            audit_dir,
            attempts,
            project: cfg.vertex_project.clone(),
            region: cfg.vertex_region.clone(),
            calls: Mutex::new(vec![]),
        })
    }

    pub fn env(&self) -> std::collections::BTreeMap<String, String> {
        let keep: Vec<&str> = if self.spec.kind == "claude_cli" { vec![] } else { config::provider_keys(&self.spec.model).to_vec() };
        let mut e = config::scrub_env(None, &keep);
        if self.spec.kind == "claude_cli" {
            e.insert("CLAUDE_CODE_USE_VERTEX".into(), "1".into());
            e.insert("ANTHROPIC_VERTEX_PROJECT_ID".into(), self.project.clone());
            e.insert("CLOUD_ML_REGION".into(), self.region.clone());
            e.insert("ANTHROPIC_MODEL".into(), self.spec.model.clone());
        } else {
            e.insert("GOOGLE_VERTEX_PROJECT".into(), self.project.clone());
            e.insert("GOOGLE_VERTEX_LOCATION".into(), self.region.clone());
            e.insert("VERTEXAI_LOCATION".into(), self.region.clone());
        }
        e
    }

    fn claude_cli(&self, cwd: &Path, mission: &str, system: &str, wall: u64) -> anyhow::Result<(String, f64, String)> {
        let cli = util::which("claude").unwrap_or_else(|| PathBuf::from("claude"));
        let mut cmd = Command::new(cli);
        cmd.args(["--dangerously-skip-permissions", "-p", "--output-format", "json", "--effort", &self.spec.effort]);
        let mut stdin = mission.to_string();
        if !system.is_empty() {
            if system.len() > SYSTEM_ARG_MAX {
                stdin = format!("# SYSTEM INSTRUCTIONS\n\n{system}\n\n# END\n\n{mission}");
                cmd.args(["--append-system-prompt", "Follow the SYSTEM INSTRUCTIONS block at the top of the message."]);
            } else {
                cmd.args(["--append-system-prompt", system]);
            }
        }
        cmd.current_dir(cwd);
        config::apply_env(&mut cmd, &self.env());
        let p = util::run_with_timeout(cmd, Some(&stdin), Duration::from_secs(wall))?;
        if p.timed_out {
            anyhow::bail!("timeout after {wall}s");
        }
        let payload = parse_json(&p.stdout);
        let mut text = vstr(payload.get("result").or_else(|| payload.get("text")));
        let mut err = String::new();
        let subtype = payload.get("subtype").and_then(|v| v.as_str());
        if payload.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false) || !matches!(subtype, None | Some("success")) {
            err = format!("CLI error: {} {}", subtype.unwrap_or("None"), util::take_chars(&text, 200));
            text.clear();
        } else if p.code != Some(0) && text.trim().is_empty() {
            err = format!("rc={:?}: {}", p.code, util::tail_chars(&p.stderr, 300));
        }
        Ok((text, payload.get("total_cost_usd").and_then(|v| v.as_f64()).unwrap_or(0.0), err))
    }

    fn opencode(&self, cwd: &Path, mission: &str, system: &str, wall: u64) -> anyhow::Result<(String, f64, String)> {
        util::write(&cwd.join("MISSION.md"), format!("{}{mission}", if system.is_empty() { String::new() } else { format!("{system}\n\n---\n\n") }))?;
        let out = cwd.join("output");
        std::fs::create_dir_all(&out)?;
        util::write(&cwd.join("opencode.json"), json!({"agent": {"build": {"steps": self.spec.steps}}}).to_string())?;
        let prompt = "Read MISSION.md first. Do what it asks. Reply with the requested output as your final message AND save it verbatim to output/answer.md.";
        let oc = opencode_dir()?;
        let mut env = self.env();
        let c = cwd.to_string_lossy().to_string();
        env.insert(
            "OPENCODE_PERMISSION".into(),
            json!({"external_directory": {"*": "deny", c.clone(): "allow", format!("{c}/**"): "allow"}}).to_string(),
        );
        env.insert("XDG_DATA_HOME".into(), cwd.join(".ocdata").to_string_lossy().into());
        env.insert("XDG_STATE_HOME".into(), cwd.join(".ocstate").to_string_lossy().into());
        let mut cmd = Command::new(&config::get().bun);
        cmd.arg("run").arg("--cwd").arg(oc.join("packages/opencode")).arg("--conditions=browser").arg("src/index.ts");
        cmd.args(["run", prompt, "--dir", &c, "--model", &self.spec.model, "--format", "json"]);
        if !self.spec.effort.is_empty() {
            cmd.args(["--variant", &self.spec.effort]);
        }
        cmd.current_dir(cwd);
        config::apply_env(&mut cmd, &env);
        let p = util::run_with_timeout(cmd, None, Duration::from_secs(wall))?;
        if p.timed_out {
            anyhow::bail!("timeout after {wall}s");
        }
        let cost = opencode_cost(&p.stdout);
        let (text, err) = opencode_stdout(&p.stdout);
        if !err.is_empty() {
            return Ok((String::new(), cost, err));
        }
        let ans = out.join("answer.md");
        if let Ok(a) = util::read_lossy(&ans) {
            if !a.trim().is_empty() {
                return Ok((a, cost, String::new()));
            }
        }
        if !text.trim().is_empty() {
            return Ok((text, cost, String::new()));
        }
        let tail = util::tail_chars(strip_oc_banner(&p.stderr).trim(), 300);
        Ok((String::new(), cost, format!("rc={:?}: {}", p.code, if tail.is_empty() { "empty reply".into() } else { tail })))
    }

    fn api(&self, cwd: &Path, mission: &str, system: &str, wall: u64) -> anyhow::Result<(String, f64, String)> {
        let r = crate::llm::session::run_improver_session(&self.spec, cwd, mission, system, wall)?;
        Ok((r.text, r.cost_usd, r.error))
    }

    fn harvest(&self, tmp: &Path, sub: &str, label: &str, cache_key: &str) -> Option<PathBuf> {
        let src = tmp.join(sub);
        if !src.is_dir() {
            return None;
        }
        let base = match &self.audit_dir {
            Some(d) => d.clone(),
            None => tempfile::Builder::new().prefix("harness2-harvest-").tempdir().ok()?.keep(),
        };
        let mut safe = util::slug_safe(label);
        if !cache_key.is_empty() {
            safe = format!("{safe}.{cache_key}");
        }
        let dest = base.join(format!("{safe}.workspace"));
        util::rmtree(&dest);
        util::copy_tree(&src, &dest).ok()?;
        Some(dest)
    }

    fn paths(&self, label: &str, cache_key: &str) -> Option<AuditPaths> {
        let d = self.audit_dir.as_ref()?;
        std::fs::create_dir_all(d).ok()?;
        let mut safe = util::slug_safe(label);
        if !cache_key.is_empty() {
            safe = format!("{safe}.{cache_key}");
        }
        Some(AuditPaths {
            mission: d.join(format!("{safe}.mission.md")),
            system: d.join(format!("{safe}.system.md")),
            response: d.join(format!("{safe}.response.md")),
            meta: d.join(format!("{safe}.meta.json")),
            workspace: d.join(format!("{safe}.workspace")),
        })
    }

    fn cached(&self, label: &str, cache_key: &str, require_workspace: bool) -> Option<Reply> {
        let p = self.paths(label, cache_key)?;
        if !p.response.is_file() || (require_workspace && !p.workspace.is_dir()) {
            return None;
        }
        let txt = util::read_lossy(&p.response).ok()?;
        if txt.trim().is_empty() {
            return None;
        }
        Some(Reply {
            label: label.into(),
            text: txt,
            ok: true,
            cached: true,
            channel: self.spec.kind.clone(),
            audit: Some(p.response.clone()),
            workspace: p.workspace.is_dir().then(|| p.workspace.clone()),
            ..Default::default()
        })
    }

    fn write_audit(&self, r: &mut Reply, mission: &str, system: &str, cache_key: &str) {
        let Some(p) = self.paths(&r.label, cache_key) else { return };
        let _ = std::fs::write(&p.mission, mission);
        if !system.is_empty() {
            let _ = std::fs::write(&p.system, system);
        }
        let _ = std::fs::write(&p.response, &r.text);
        let meta = json!({
            "label": r.label, "ok": r.ok, "channel": r.channel, "model": self.spec.model,
            "effort": self.spec.effort, "attempts": r.attempts, "wall_s": r.wall_s,
            "cost_usd": r.cost_usd, "error": r.error,
        });
        let _ = std::fs::write(&p.meta, serde_json::to_string_pretty(&meta).unwrap());
        r.audit = Some(p.response);
    }
}

struct AuditPaths {
    mission: PathBuf,
    system: PathBuf,
    response: PathBuf,
    meta: PathBuf,
    workspace: PathBuf,
}

impl Propose for Proposer {
    fn spec(&self) -> &ChannelSpec {
        &self.spec
    }
    fn audit_dir(&self) -> Option<PathBuf> {
        self.audit_dir.clone()
    }
    fn calls(&self) -> Vec<Reply> {
        self.calls.lock().unwrap().clone()
    }
    fn fork(&self, spec: ChannelSpec, audit_dir: Option<PathBuf>) -> anyhow::Result<Arc<dyn Propose>> {
        let mut p = Proposer::new(spec, audit_dir, self.attempts)?;
        p.project = self.project.clone();
        p.region = self.region.clone();
        Ok(Arc::new(p))
    }

    /// Run one session. Fails with ProposerError if no usable reply after `attempts`.
    fn call(&self, mission: &str, context_dir: Option<&Path>, label: &str, opts: CallOpts) -> Result<Reply, ProposerError> {
        let cache_key = call_fingerprint(label, mission, &opts.system, context_dir);
        if let Some(c) = self.cached(label, &cache_key, opts.harvest.is_some()) {
            self.calls.lock().unwrap().push(c.clone());
            return Ok(c);
        }
        let wall = opts.timeout.unwrap_or(self.spec.timeout);
        let tries = opts.max_attempts.unwrap_or(self.attempts).max(1);
        let mut last = String::new();
        let mut spent = 0.0;
        let t0 = Instant::now();
        for attempt in 1..=tries {
            if attempt > 1 {
                std::thread::sleep(Duration::from_secs(10 * (attempt as u64 - 1)));
            }
            let tmp = match tempfile::Builder::new().prefix("harness2-prop-").tempdir() {
                Ok(t) => t,
                Err(e) => {
                    last = format!("IOError: {e}");
                    continue;
                }
            };
            let res: anyhow::Result<Option<Reply>> = (|| {
                if let Some(cd) = context_dir.filter(|c| c.is_dir()) {
                    util::copy_tree(cd, &tmp.path().join("context"))?;
                }
                let (text, cost, err) = {
                    let _g = SEM.acquire();
                    match self.spec.kind.as_str() {
                        "claude_cli" => self.claude_cli(tmp.path(), mission, &opts.system, wall)?,
                        "api" => self.api(tmp.path(), mission, &opts.system, wall)?,
                        _ => self.opencode(tmp.path(), mission, &opts.system, wall)?,
                    }
                };
                spent += cost;
                if text.trim().is_empty() {
                    last = if err.is_empty() { "empty reply".into() } else { err };
                    return Ok(None);
                }
                let mut r = Reply {
                    label: label.into(),
                    text,
                    ok: true,
                    cost_usd: spent,
                    wall_s: (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
                    attempts: attempt,
                    channel: self.spec.kind.clone(),
                    ..Default::default()
                };
                if let Some(h) = &opts.harvest {
                    r.workspace = self.harvest(tmp.path(), h, label, &cache_key);
                    if r.workspace.is_none() {
                        if let Some(p) = self.paths(label, &cache_key) {
                            util::rmtree(&p.workspace);
                        }
                    }
                }
                self.write_audit(&mut r, mission, &opts.system, &cache_key);
                Ok(Some(r))
            })();
            match res {
                Ok(Some(r)) => {
                    self.calls.lock().unwrap().push(r.clone());
                    return Ok(r);
                }
                Ok(None) => {}
                Err(e) => last = format!("{e}"),
            }
        }
        let mut r = Reply {
            label: label.into(),
            ok: false,
            wall_s: (t0.elapsed().as_secs_f64() * 10.0).round() / 10.0,
            attempts: tries,
            channel: self.spec.kind.clone(),
            error: last.clone(),
            cost_usd: spent,
            ..Default::default()
        };
        self.write_audit(&mut r, mission, &opts.system, &cache_key);
        self.calls.lock().unwrap().push(r);
        Err(ProposerError {
            message: format!("[{label}] {}/{}: {last}", self.spec.kind, self.spec.model),
            attempts: tries,
            cost_usd: spent,
        })
    }
}

/// The upstream OpenCode checkout to run through bun.
pub fn opencode_dir() -> anyhow::Result<PathBuf> {
    let oc = config::get().opencode_dir.clone();
    if !oc.join("packages/opencode/src/index.ts").is_file() {
        anyhow::bail!(
            "no OpenCode checkout at {} (config.OPENCODE_DIR); run `harness2 setup <benchmark>` to clone the pinned commit there",
            oc.display()
        );
    }
    Ok(oc)
}

fn vstr(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => String::new(),
        Some(o) => o.to_string(),
    }
}

/// The claude CLI's one JSON object, tolerating banners and trailing noise around it.
pub fn parse_json(stdout: &str) -> Value {
    let s = stdout.trim();
    if s.is_empty() {
        return json!({});
    }
    if let Ok(v) = serde_json::from_str::<Value>(s) {
        return v;
    }
    for line in s.lines().rev() {
        let line = line.trim();
        if line.starts_with('{') && line.ends_with('}') {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                return v;
            }
        }
    }
    if let (Some(i), Some(j)) = (s.find('{'), s.rfind('}')) {
        if i < j {
            if let Ok(v) = serde_json::from_str::<Value>(&s[i..=j]) {
                return v;
            }
        }
    }
    json!({"result": s})
}

static OC_BANNER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?m)^(?:Performing one time database migration[^\n]*|sqlite-migration:[^\n]*|Database migration complete\.)[ \t]*$").unwrap()
});

pub fn strip_oc_banner(s: &str) -> String {
    OC_BANNER.replace_all(s, "").into_owned()
}

fn obj<'a>(d: &'a Value, key: &str) -> Option<&'a serde_json::Map<String, Value>> {
    d.get(key).and_then(|v| v.as_object())
}

/// Fold `opencode run --format json` into (final assistant text, error text).
pub fn opencode_stdout(stdout: &str) -> (String, String) {
    let mut texts = vec![];
    let mut errors = vec![];
    let mut saw_json = false;
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<Value>(line) else { continue };
        if !ev.is_object() {
            continue;
        }
        saw_json = true;
        let part = obj(&ev, "part");
        let ptype = part.and_then(|p| p.get("type")).and_then(|t| t.as_str()).or_else(|| ev.get("type").and_then(|t| t.as_str()));
        if ptype == Some("text") {
            if let Some(t) = part.and_then(|p| p.get("text")).and_then(|t| t.as_str()) {
                if !t.trim().is_empty() {
                    texts.push(t.to_string());
                }
            }
        } else if ptype == Some("error") {
            let msg = match ev.get("error") {
                Some(Value::Object(e)) => e
                    .get("data")
                    .and_then(|d| d.get("message"))
                    .map(|m| vstr(Some(m)))
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| vstr(e.get("name"))),
                other => vstr(other),
            };
            let msg = msg.trim().to_string();
            if !msg.is_empty() {
                errors.push(msg);
            }
        }
    }
    if !saw_json {
        return (strip_oc_banner(stdout).trim().to_string(), String::new());
    }
    (texts.join("\n\n"), util::tail_chars(&errors.join("; "), 300))
}

/// Sum `part.cost` over the step_finish events of an `opencode run --format json` stream.
pub fn opencode_cost(stdout: &str) -> f64 {
    let mut total = 0.0;
    for line in stdout.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<Value>(line) else { continue };
        let part = obj(&ev, "part");
        let t = part.and_then(|p| p.get("type")).and_then(|t| t.as_str()).or_else(|| ev.get("type").and_then(|t| t.as_str()));
        if matches!(t, Some("step-finish") | Some("step_finish")) {
            if let Some(c) = part.and_then(|p| p.get("cost")).and_then(|c| c.as_f64()) {
                total += c;
            }
        }
    }
    (total * 1e6).round() / 1e6
}

/// Content-address one call, including its evidence and its mode/task workspace path.
/// The resolved path is part of the key: the two modes can have byte-identical k=1 evidence
/// and must still be independent draws.
pub fn call_fingerprint(label: &str, mission: &str, system: &str, context_dir: Option<&Path>) -> String {
    let mut h = Sha256::new();
    for v in [label, mission, system] {
        h.update(v.as_bytes());
        h.update(b"\0");
    }
    if let Some(root) = context_dir {
        let resolved = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        h.update(resolved.to_string_lossy().as_bytes());
        h.update(b"\0");
        for p in util::files_sorted(root) {
            if util::in_pycache(&p) {
                continue;
            }
            h.update(util::rel(root, &p).as_bytes());
            h.update(b"\0");
            match std::fs::read(&p) {
                Ok(b) => h.update(&b),
                Err(e) => h.update(format!("<unreadable:OSError:{e}>").as_bytes()),
            }
            h.update(b"\0");
        }
    }
    hex::encode(h.finalize())[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oc_stream_parsing() {
        let s = "Performing one time database migration\n{\"type\":\"text\",\"part\":{\"type\":\"text\",\"text\":\"hi\"}}\n{\"part\":{\"type\":\"step-finish\",\"cost\":0.25}}\n";
        assert_eq!(opencode_stdout(s), ("hi".to_string(), String::new()));
        assert_eq!(opencode_cost(s), 0.25);
        assert_eq!(opencode_stdout("plain answer\n").0, "plain answer");
    }
    #[test]
    fn claude_json_tolerates_noise() {
        let v = parse_json("banner\n{\"result\":\"ok\",\"total_cost_usd\":1.5}\n");
        assert_eq!(v["result"], "ok");
    }
}
