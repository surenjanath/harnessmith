//! Deployment settings and dependency versions. Shell values override `.env`.

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub const METHOD_VERSION: u32 = 2;

/// A required deployment value is missing; the message carries the remediation.
#[derive(Debug)]
pub struct ConfigError(pub String);
impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ConfigError {}

/// Set `key=value` in the project's `.env` (replacing an existing line, or appending), keep the
/// file private (0600), and set it in this process so it works at once.
pub fn set_dotenv(key: &str, value: &str) -> std::io::Result<PathBuf> {
    let path = get().root.join(".env");
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let quoted = if value.contains(char::is_whitespace) || value.contains('#') || value.contains('"') { format!("'{}'", value.replace('\'', "")) } else { value.to_string() };
    let mut found = false;
    let mut lines: Vec<String> = text
        .lines()
        .map(|l| {
            let bare = l.trim_start().strip_prefix("export ").unwrap_or(l.trim_start());
            if bare.split_once('=').map(|(k, _)| k.trim() == key).unwrap_or(false) {
                found = true;
                format!("{key}={quoted}")
            } else {
                l.to_string()
            }
        })
        .collect();
    if !found {
        if lines.is_empty() {
            lines.push("# Local secrets for harnessmith (git-ignored). Read at startup and again when a key is missing.".into());
        }
        lines.push(format!("{key}={quoted}"));
    }
    std::fs::write(&path, lines.join("\n") + "\n")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    std::env::set_var(key, value);
    Ok(path)
}

/// Load KEY=VALUE lines from `path` into the environment without overriding (the shell wins).
pub fn load_dotenv(path: &Path) {
    let Ok(text) = std::fs::read_to_string(path) else { return };
    for raw in text.lines() {
        let mut line = raw.trim();
        if line.is_empty() || line.starts_with('#') || !line.contains('=') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("export ") {
            line = rest.trim_start();
        }
        let (key, value) = line.split_once('=').unwrap();
        let key = key.trim();
        if key.is_empty() || !key.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let mut value = value.trim().to_string();
        let b = value.as_bytes();
        if b.len() >= 2 && b[0] == b[b.len() - 1] && (b[0] == b'\'' || b[0] == b'"') {
            value = value[1..value.len() - 1].to_string();
        } else if let Some(i) = value.find(" #") {
            value = value[..i].trim_end().to_string();
        }
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, value);
        }
    }
}

/// Env override where a set-but-empty variable is the same as an unset one.
pub fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

pub fn env_or(name: &str, default: &str) -> String {
    env(name).unwrap_or_else(|| default.to_string())
}

pub fn env_int(name: &str, default: i64) -> Result<i64> {
    match env(name) {
        None => Ok(default),
        Some(v) => v
            .parse()
            .map_err(|_| anyhow!(ConfigError(format!("{name} must be an integer, got {v:?}")))),
    }
}

/// The value of `name`, or ConfigError with the remediation line when it is unset/empty.
pub fn require(name: &str, needed_by: &str) -> Result<String> {
    env(name).ok_or_else(|| {
        anyhow!(ConfigError(format!(
            "{name} is required for {needed_by}; set it in .env or the shell (see .env.example)"
        )))
    })
}

/// Every resolved setting, read once.
#[derive(Debug, Clone)]
pub struct Config {
    pub root: PathBuf,
    pub dependencies: PathBuf,
    pub third_party: PathBuf,
    pub runs: PathBuf,
    pub vertex_project: String,
    pub vertex_region: String,
    pub solver_model_oc: String,
    pub solver_model_cx: String,
    pub improver_model: String,
    pub judge_model: String,
    pub opencode_dir: PathBuf,
    pub bun: PathBuf,
    pub codex_dist: PathBuf,
    pub codex_base_url: String,
    pub codex_container_url: String,
    pub vertex_proxy_key: String,
    pub codex_home_dir: PathBuf,
    pub jb_judge_api_base: String,
    pub jb_judge_api_key: String,
    pub default_proposer_channel: String,
    pub connections_file: PathBuf,
}

pub const CODEX_VERSION: &str = "0.144.6";
pub const PHASE_WIDTH: usize = 2;
pub const TRANSPORT_ATTEMPTS: u32 = 2;
pub const PROPOSER_CONCURRENCY: usize = 2;
pub const LAB_DELIV_CAP: usize = 60000;
pub const JB_OBS_TRUNC: usize = 50000;
pub const JB_JUDGE_REASONING_EFFORT: &str = "low";
pub const WB_SOLVER_MODEL: &str = "harness2/solver";
pub const WB_WEB_JUDGE: &str = "harness2/judge";
pub const WB_OC_HARNESS_VER: &str = "1.14.18";
pub const WB_MODEL_CONNECTION: &str = "local_proxy";

pub static DEPENDENCY_PINS: &[(&str, &str)] = &[
    ("job-bench-eval", "fa03a17f9ff662f136798a274e10011c3aa00fd0"),
    ("workbuddy-bench", "b516950be5b56eb3be406c2f76ee1c5111dcb57f"),
    ("harvey-labs", "b58a28ae9fde5ee4b810e53b4fd0a1915325a35d"),
    ("opencode", "23fb5e0516c99ac04a1aa46c193efda2e1b9bb24"),
];
pub static DEPENDENCY_ORIGINS: &[(&str, &str)] = &[
    ("job-bench-eval", "https://github.com/Job-Bench/job-bench-eval.git"),
    ("workbuddy-bench", "https://github.com/Tencent/workbuddy-bench.git"),
    ("harvey-labs", "https://github.com/harveyai/harvey-labs.git"),
    ("opencode", "https://github.com/anomalyco/opencode.git"),
];

pub fn pin(name: &str) -> &'static str {
    DEPENDENCY_PINS.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).unwrap_or("")
}
pub fn origin(name: &str) -> &'static str {
    DEPENDENCY_ORIGINS.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).unwrap_or("")
}

fn find_root() -> PathBuf {
    if let Some(r) = env("HARNESS2_ROOT") {
        return PathBuf::from(r);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for d in cwd.ancestors() {
        if d.join("Cargo.toml").is_file() && d.join("src/core").is_dir() {
            return d.to_path_buf();
        }
    }
    // A release tarball or install: `third_party/` sits next to the binary (or one level up),
    // so benchmark mode works from any directory (upstream #146 / #144, our version).
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|e| e.canonicalize().ok()).and_then(|e| e.parent().map(|p| p.to_path_buf())) {
        for d in [exe_dir.clone(), exe_dir.join(".."), exe_dir.join("../share/harnessmith")] {
            if d.join("third_party").is_dir() {
                return d.canonicalize().unwrap_or(d);
            }
        }
    }
    cwd
}

impl Config {
    pub fn load() -> Config {
        let root = find_root();
        load_dotenv(&root.join(".env"));
        let deps = root.join("dependencies");
        let runs = env("HARNESS2_RUNS_DIR")
            .map(|p| expand_home(&p))
            .unwrap_or_else(|| root.join("runs"));
        let runs = std::fs::canonicalize(&runs).unwrap_or(runs);
        let vertex_project =
            env("GOOGLE_CLOUD_PROJECT").or_else(|| env("HARNESS2_VERTEX_PROJECT")).unwrap_or_default();
        let vertex_region = env_or("GOOGLE_CLOUD_LOCATION", "global");
        if !vertex_project.is_empty() && std::env::var_os("GOOGLE_CLOUD_PROJECT").is_none() {
            std::env::set_var("GOOGLE_CLOUD_PROJECT", &vertex_project);
        }
        if std::env::var_os("GOOGLE_CLOUD_LOCATION").is_none() {
            std::env::set_var("GOOGLE_CLOUD_LOCATION", &vertex_region);
        }
        let solver_oc = env_or("HARNESS2_SOLVER_MODEL", "");
        let solver_cx = env("HARNESS2_CODEX_MODEL")
            .unwrap_or_else(|| solver_oc.split_once('/').map(|x| x.1).unwrap_or(&solver_oc).to_string());
        let improver = env("HARNESS2_IMPROVER_MODEL").unwrap_or_else(|| solver_oc.clone());
        let judge = env_or("HARNESS2_JUDGE_MODEL", "");
        let vertex_host = if vertex_region == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{vertex_region}-aiplatform.googleapis.com")
        };
        let vertex_openai = format!(
            "https://{vertex_host}/v1/projects/{vertex_project}/locations/{vertex_region}/endpoints/openapi"
        );
        let jb_base = env("HARNESS2_JUDGE_BASE_URL").unwrap_or_else(|| {
            if vertex_project.is_empty() { String::new() } else { vertex_openai.clone() }
        });
        let codex_base = env_or("HARNESS2_CODEX_BASE_URL", "");
        Config {
            third_party: root.join("third_party"),
            opencode_dir: deps.join("opencode"),
            bun: deps.join("tools/node_modules/.bin/bun"),
            codex_dist: deps.join("codex/dist"),
            codex_home_dir: deps.join("codex/home"),
            codex_container_url: env("HARNESS2_CODEX_CONTAINER_URL").unwrap_or_else(|| codex_base.clone()),
            codex_base_url: codex_base,
            vertex_proxy_key: env_or("HARNESS2_CODEX_API_KEY", ""),
            jb_judge_api_base: jb_base,
            jb_judge_api_key: env_or("HARNESS2_JUDGE_API_KEY", ""),
            default_proposer_channel: env_or("HARNESS2_IMPROVER_CHANNEL", "opencode"),
            connections_file: env("HARNESS2_CONNECTIONS")
                .map(|p| expand_home(&p))
                .unwrap_or_else(|| root.join("connections.json")),
            dependencies: deps,
            root,
            runs,
            vertex_project,
            vertex_region,
            solver_model_oc: solver_oc,
            solver_model_cx: solver_cx,
            improver_model: improver,
            judge_model: judge,
        }
    }

    /// Vertex AI's OpenAI-compatible endpoint for this project/region.
    pub fn vertex_openai_url(&self) -> String {
        let host = if self.vertex_region == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{}-aiplatform.googleapis.com", self.vertex_region)
        };
        format!("https://{host}/v1/projects/{}/locations/{}/endpoints/openapi", self.vertex_project, self.vertex_region)
    }

    pub fn jb_repo(&self) -> PathBuf {
        self.dependencies.join("job-bench-eval")
    }
    pub fn wb_repo(&self) -> PathBuf {
        self.dependencies.join("workbuddy-bench")
    }
    pub fn lab_repo(&self) -> PathBuf {
        self.dependencies.join("harvey-labs")
    }
    pub fn lab_python(&self) -> PathBuf {
        self.lab_repo().join(".venv/bin/python")
    }
    pub fn jb_judge_python(&self) -> PathBuf {
        self.jb_repo().join(".venv/bin/python")
    }
    pub fn lab_judge_model(&self) -> String {
        if self.judge_model.starts_with("claude") || self.judge_model.starts_with("gemini") {
            format!("vertex/{}", self.judge_model)
        } else {
            self.judge_model.clone()
        }
    }
    pub fn jb_judge_model(&self) -> String {
        if self.judge_model.starts_with("gemini") && env("HARNESS2_JUDGE_BASE_URL").is_none() {
            format!("google/{}", self.judge_model)
        } else {
            self.judge_model.clone()
        }
    }

    /// The proposer model for a channel (both CLI channels use the improver model).
    pub fn proposer_model_by_channel(&self, _kind: &str) -> String {
        self.improver_model.clone()
    }

    pub fn validate_models(&self, judge: bool) -> Result<()> {
        require("HARNESS2_SOLVER_MODEL", "executor rollouts")?;
        if !self.solver_model_oc.contains('/') || !self.improver_model.contains('/') {
            return Err(anyhow!(ConfigError(
                "Executor and improver models must use OpenCode's provider/model syntax".into()
            )));
        }
        if [&self.solver_model_oc, &self.improver_model].iter().any(|m| m.starts_with("google-vertex"))
            && self.vertex_project.is_empty()
        {
            return Err(anyhow!(ConfigError("Set GOOGLE_CLOUD_PROJECT for Vertex AI".into())));
        }
        if judge {
            require("HARNESS2_JUDGE_MODEL", "benchmark judging")?;
        }
        Ok(())
    }
}

pub fn expand_home(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(h) = std::env::var_os("HOME") {
            return PathBuf::from(h).join(rest);
        }
    }
    PathBuf::from(p)
}

static CONFIG: Lazy<Config> = Lazy::new(Config::load);

/// The process-wide configuration.
pub fn get() -> &'static Config {
    &CONFIG
}

pub static SCRUB_ENV: &[&str] = &[
    "HARNESS2_JUDGE_API_KEY",
    "HARNESS2_BRIDGE_KEY",
    "HARNESS2_CODEX_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENERATIVE_AI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "JIRA_API_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_BEARER_TOKEN_BEDROCK",
];

/// A copy of `base_env` (default: the process environment) with runner and host secrets removed.
pub fn scrub_env(base_env: Option<&HashMap<String, String>>, keep: &[&str]) -> BTreeMap<String, String> {
    let src: Vec<(String, String)> = match base_env {
        Some(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        None => std::env::vars().collect(),
    };
    src.into_iter()
        .filter(|(k, _)| !SCRUB_ENV.contains(&k.as_str()) || keep.contains(&k.as_str()))
        .collect()
}

/// The scrubbed credentials an OpenCode `provider/model` needs; () for Vertex and the rest.
pub fn provider_keys(model: &str) -> &'static [&'static str] {
    match model.split('/').next().unwrap_or("") {
        "openai" => &["OPENAI_API_KEY"],
        "anthropic" => &["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"],
        "google" => &["GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"],
        "amazon-bedrock" => &[
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_BEARER_TOKEN_BEDROCK",
        ],
        _ => &[],
    }
}

/// Apply a scrubbed env to a `Command` (clearing the inherited environment first).
pub fn apply_env(cmd: &mut std::process::Command, env: &BTreeMap<String, String>) {
    cmd.env_clear();
    for (k, v) in env {
        cmd.env(k, v);
    }
}
