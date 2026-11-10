//! Model connections: named endpoints a model id `<connection>/<model>` resolves through.
//!
//! Stored as JSON (default `./connections.json`, override with `HARNESS2_CONNECTIONS`) and
//! edited from the web UI. When a name is not registered, the well-known provider names work
//! out of the box from the usual environment variables: `openai/…` (OPENAI_API_KEY),
//! `anthropic/…` (ANTHROPIC_API_KEY), `gemini/…` or `google/…` (GEMINI_API_KEY /
//! GOOGLE_API_KEY), `ollama/…` (local, no key), `openrouter/…` (OPENROUTER_API_KEY).

use crate::config;
use anyhow::{bail, Result};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::RwLock;

/// The wire protocol a connection speaks.
pub const PROVIDERS: &[&str] = &["openai", "anthropic", "gemini", "ollama", "mock"];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Connection {
    /// The prefix of a model id, e.g. `work` in `work/gpt-4.1`.
    pub name: String,
    /// One of PROVIDERS. `openai` covers every OpenAI-compatible server (vLLM, LM Studio,
    /// OpenRouter, Together, Groq, DeepSeek, Azure-compatible gateways, ...).
    pub provider: String,
    #[serde(default)]
    pub base_url: String,
    /// A literal key. Prefer `api_key_env` so the secret stays out of the file.
    #[serde(default)]
    pub api_key: String,
    /// Name of the environment variable holding the key.
    #[serde(default)]
    pub api_key_env: String,
    /// Model used when a role names the connection without a model.
    #[serde(default)]
    pub default_model: String,
    /// USD per million input / output tokens, for cost accounting (optional).
    #[serde(default)]
    pub input_price: f64,
    #[serde(default)]
    pub output_price: f64,
    /// Max output tokens per call.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    /// The model's context window in tokens. Ollama is asked for this window (`num_ctx`);
    /// the agent loop trims old observations to stay inside it.
    #[serde(default)]
    pub context_window: Option<u32>,
    /// Max simultaneous requests to this endpoint (default: 1 for Ollama, unlimited otherwise).
    #[serde(default)]
    pub max_concurrency: Option<usize>,
    /// USD per million cached input tokens (default: 10% of input for Anthropic, 50% otherwise).
    #[serde(default)]
    pub cached_input_price: Option<f64>,
    /// Daily spending cap in USD (UTC day); over it, calls fail over to the next model in the chain.
    #[serde(default)]
    pub daily_usd: Option<f64>,
    /// Prompt caching where the provider supports it (default on).
    #[serde(default)]
    pub prompt_cache: Option<bool>,
}

impl Connection {
    pub fn key(&self) -> String {
        if !self.api_key.is_empty() {
            return self.api_key.clone();
        }
        if !self.api_key_env.is_empty() {
            let v = std::env::var(&self.api_key_env).unwrap_or_default();
            if !v.is_empty() {
                return v;
            }
            // A key added to .env after the server started: read the file again.
            crate::config::load_dotenv(&crate::config::get().root.join(".env"));
            return std::env::var(&self.api_key_env).unwrap_or_default();
        }
        String::new()
    }

    /// Why calls would be refused for lack of a key, if they would (hosted endpoints only).
    pub fn missing_key(&self) -> Option<String> {
        let base = self.base();
        let local = base.is_empty() || base.contains("localhost") || base.contains("127.0.0.1") || base.contains("[::1]") || self.provider == "ollama" || self.provider == "mock";
        if local || !self.key().is_empty() {
            return None;
        }
        Some(if self.api_key_env.is_empty() {
            format!("connection {:?} has no API key: set \"API key env var\" (e.g. NVIDIA_API_KEY) or paste a key", self.name)
        } else {
            format!("{} is not set. Add a line `{}=...` to the .env file in the harnessmith folder ({}), or export it in the shell that starts harness2", self.api_key_env, self.api_key_env, crate::config::get().root.join(".env").display())
        })
    }

    /// USD for one reply: fresh input, cache writes (1.25x) and reads, and output.
    pub fn cost(&self, r: &super::client::ChatResp) -> f64 {
        let cached = self.cached_input_price.unwrap_or(self.input_price * if self.provider == "anthropic" { 0.1 } else { 0.5 });
        (r.input_tokens as f64 * self.input_price + r.cache_write_tokens as f64 * self.input_price * 1.25 + r.cached_tokens as f64 * cached + r.output_tokens as f64 * self.output_price) / 1e6
    }

    /// Effective concurrency limit.
    pub fn concurrency(&self) -> usize {
        self.max_concurrency.filter(|n| *n > 0).unwrap_or(if self.provider == "ollama" { 1 } else { 64 })
    }

    /// Effective context window in tokens (a conservative default when unknown).
    pub fn window(&self) -> u32 {
        self.context_window.filter(|n| *n > 0).unwrap_or(if self.provider == "ollama" { 32_768 } else { 200_000 })
    }

    pub fn base(&self) -> String {
        let b = if self.base_url.trim().is_empty() { default_base(&self.provider).to_string() } else { self.base_url.trim().to_string() };
        b.trim_end_matches('/').to_string()
    }

    /// The JSON shown to the UI: never carries a literal key.
    pub fn redacted(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self).unwrap();
        v["api_key"] = serde_json::json!(if self.api_key.is_empty() { "" } else { "••••••" });
        let k = self.key();
        v["has_key"] = serde_json::json!(!k.is_empty());
        // Where the key comes from; no part of the key itself leaves the server.
        let tail = String::new();
        v["key_source"] = serde_json::json!(if !self.api_key.is_empty() { "file" } else if !self.api_key_env.is_empty() { if k.is_empty() { "env_missing" } else { "env" } } else { "none" });
        v["key_tail"] = serde_json::json!(tail);
        v["needs_key"] = serde_json::json!(self.missing_key().is_some() || (!k.is_empty()));
        v
    }

    /// The environment variable a key for this connection is stored under by default.
    pub fn default_key_env(&self) -> String {
        if !self.api_key_env.is_empty() {
            return self.api_key_env.clone();
        }
        let b = self.base();
        for (host, var) in [("nvidia.com", "NVIDIA_API_KEY"), ("openrouter.ai", "OPENROUTER_API_KEY"), ("groq.com", "GROQ_API_KEY"), ("deepseek.com", "DEEPSEEK_API_KEY"), ("together.xyz", "TOGETHER_API_KEY"), ("mistral.ai", "MISTRAL_API_KEY"), ("anthropic.com", "ANTHROPIC_API_KEY"), ("googleapis.com", "GEMINI_API_KEY"), ("api.openai.com", "OPENAI_API_KEY")] {
            if b.contains(host) {
                return var.into();
            }
        }
        let n: String = self.name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' }).collect();
        format!("{n}_API_KEY")
    }
}

/// Billions of parameters, when the model's name states it: `qwen2.5:3b`, `llama-3.2-1b-instruct`,
/// `gpt-oss-20b`, `smollm2-360m`.
pub fn params_b(model: &str) -> Option<f64> {
    let re = regex::Regex::new(r"(?i)(?:^|[^a-z0-9.])(\d+(?:\.\d+)?)\s*([bm])(?:[^a-z0-9]|$)").unwrap();
    // The last match: "llama-3.2-1b" has the size after the version.
    re.captures_iter(model).last().and_then(|c| {
        let n: f64 = c[1].parse().ok()?;
        Some(if c[2].eq_ignore_ascii_case("m") { n / 1000.0 } else { n })
    })
}

/// A model the accuracy boost is meant for: served locally, or small by its name.
pub fn is_small(c: &Connection, model: &str) -> bool {
    let base = c.base();
    let host = base.split("://").nth(1).unwrap_or(&base).split(['/', ':']).next().unwrap_or("").to_string();
    let private = host == "localhost" || host.starts_with("127.") || host == "[" || host == "host.docker.internal" || host.starts_with("192.168.") || host.starts_with("10.")
        || host.strip_prefix("172.").and_then(|r| r.split('.').next()).and_then(|o| o.parse::<u8>().ok()).map(|o| (16..=31).contains(&o)).unwrap_or(false);
    let local = c.provider == "ollama" && (c.base_url.trim().is_empty() || private) || (c.provider == "openai" && private);
    let by_size = params_b(model).map(|b| b <= 40.0).unwrap_or(false);
    let by_name = regex::Regex::new(r"(?i)(^|[^a-z])(mini|tiny|small|nano|smol|lite)([^a-z]|$)").unwrap().is_match(model);
    local || by_size || by_name
}

pub fn default_base(provider: &str) -> &'static str {
    match provider {
        "openai" => "https://api.openai.com/v1",
        "anthropic" => "https://api.anthropic.com",
        "gemini" => "https://generativelanguage.googleapis.com",
        "ollama" => "http://localhost:11434",
        _ => "",
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default)]
    pub connections: Vec<Connection>,
    /// Default model ids for each role (solver, improver, selector).
    #[serde(default)]
    pub roles: std::collections::BTreeMap<String, String>,
}

static CACHE: Lazy<RwLock<Option<(PathBuf, Registry)>>> = Lazy::new(|| RwLock::new(None));

pub fn path() -> PathBuf {
    config::get().connections_file.clone()
}

pub fn load() -> Registry {
    let p = path();
    if let Some((cp, r)) = CACHE.read().unwrap().as_ref() {
        if *cp == p {
            return r.clone();
        }
    }
    let r: Registry = std::fs::read_to_string(&p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    *CACHE.write().unwrap() = Some((p, r.clone()));
    r
}

pub fn save(r: &Registry) -> Result<()> {
    for c in &r.connections {
        validate(c)?;
    }
    let p = path();
    crate::util::write(&p, serde_json::to_string_pretty(r)?)?;
    *CACHE.write().unwrap() = Some((p, r.clone()));
    Ok(())
}

pub fn validate(c: &Connection) -> Result<()> {
    if !regex::Regex::new(r"^[A-Za-z0-9_.\-]+$").unwrap().is_match(&c.name) {
        bail!("connection name {:?} must be letters, digits, '.', '_' or '-' (it is the prefix of a model id)", c.name);
    }
    if !PROVIDERS.contains(&c.provider.as_str()) {
        bail!("unknown provider {:?}; expected one of {PROVIDERS:?}", c.provider);
    }
    Ok(())
}

/// The built-in fallback for an unregistered prefix.
fn builtin(name: &str) -> Option<Connection> {
    let mk = |provider: &str, base: &str, env: &str| Connection {
        name: name.into(),
        provider: provider.into(),
        base_url: base.into(),
        api_key: String::new(),
        api_key_env: env.into(),
        default_model: String::new(),
        input_price: 0.0,
        output_price: 0.0,
        max_tokens: None,
        temperature: None,
        context_window: None,
        max_concurrency: None,
        cached_input_price: None,
        daily_usd: None,
        prompt_cache: None,
    };
    Some(match name {
        "openai" => mk("openai", "", "OPENAI_API_KEY"),
        "anthropic" => mk("anthropic", "", "ANTHROPIC_API_KEY"),
        "gemini" | "google" => {
            let env = if std::env::var("GEMINI_API_KEY").is_ok() { "GEMINI_API_KEY" } else { "GOOGLE_API_KEY" };
            mk("gemini", "", env)
        }
        "ollama" => mk("ollama", "", ""),
        "openrouter" => mk("openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "groq" => mk("openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
        "deepseek" => mk("openai", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        "together" => mk("openai", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
        "mistral" => mk("openai", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
        // NVIDIA's hosted models; their ids carry an org, e.g. nvidia/mistralai/mistral-large-4-1t-a52b.
        "nvidia" => mk("openai", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
        "mock" => mk("mock", "", ""),
        _ => return None,
    })
}

/// Resolve `<connection>/<model>` (or a bare connection name) to (connection, model).
pub fn resolve(model_id: &str) -> Result<(Connection, String)> {
    let (prefix, model) = match model_id.split_once('/') {
        Some((p, m)) => (p.to_string(), m.to_string()),
        None => (model_id.to_string(), String::new()),
    };
    let reg = load();
    let conn = reg
        .connections
        .iter()
        .find(|c| c.name == prefix)
        .cloned()
        .or_else(|| builtin(&prefix))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no model connection named {prefix:?}; add one in the UI (Models tab) or in {}, or use a built-in prefix (openai, anthropic, gemini, ollama, openrouter, groq, deepseek, together, mistral, nvidia)",
                path().display()
            )
        })?;
    let model = if model.is_empty() { conn.default_model.clone() } else { model };
    if model.is_empty() {
        bail!("model id {model_id:?} names no model and connection {prefix:?} has no default_model");
    }
    Ok((conn, model))
}

#[cfg(test)]
mod size_tests {
    use super::*;

    #[test]
    fn model_sizes_are_read_from_names() {
        assert_eq!(params_b("qwen2.5:3b"), Some(3.0));
        assert_eq!(params_b("llama-3.2-1b-instruct"), Some(1.0));
        assert_eq!(params_b("gpt-oss-20b"), Some(20.0));
        assert_eq!(params_b("phi-4-mini-3.8b"), Some(3.8));
        assert_eq!(params_b("smollm2-360m"), Some(0.36));
        assert_eq!(params_b("gpt-4.1"), None);
        assert_eq!(params_b("claude-opus-5-5"), None);
        let api = builtin("openai").unwrap();
        assert!(!is_small(&api, "gpt-4.1"));
        assert!(is_small(&api, "gpt-4.1-mini"));
        assert!(is_small(&builtin("ollama").unwrap(), "anything"));
        assert!(is_small(&builtin("openrouter").unwrap(), "meta-llama/llama-3.2-3b-instruct:free"));
        assert!(!is_small(&builtin("openrouter").unwrap(), "meta-llama/llama-3.3-70b-instruct"));
        let mut local = builtin("openai").unwrap();
        local.base_url = "http://127.0.0.1:8080/v1".into();
        assert!(is_small(&local, "some-model"));
    }
}
