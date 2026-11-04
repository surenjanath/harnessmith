//! The solver substrates behind one interface.
//!
//! `opencode` carries OPENCODE_SURFACE and three install modes; `codex` carries CODEX_SURFACE
//! and one; `native` is this port's own agent runtime (see `crate::llm`), which runs a harness
//! against any connected model with no external CLI. Benchmark adapters supply launcher bodies.

pub mod codex;
pub mod native;
pub mod opencode;

use std::collections::BTreeMap;
use std::path::PathBuf;

/// A materialised harness ready for a launcher; `fingerprint` digests the SOURCE tree.
#[derive(Debug, Clone)]
pub struct Install {
    pub staging_dir: PathBuf,
    pub fingerprint: String,
    pub extras: BTreeMap<String, String>,
}

impl Install {
    pub fn drop_staging(&self) {
        crate::util::rmtree(&self.staging_dir);
    }
}

/// One execution request handed to a benchmark-supplied launcher body.
#[derive(Debug, Clone)]
pub struct RunRequest {
    pub task_ref: String,
    pub run_ids: Vec<String>,
    pub timeout: u64,
}

pub const TRUNC_FMT: &str = "…(+{n} chars — render truncation)";

/// Clip `s` to `n` chars with the shared renderer-truncation marker.
pub fn clip(s: &str, n: usize) -> String {
    let s = s.replace('\0', "");
    let len = s.chars().count();
    if n == 0 || len <= n {
        return s;
    }
    format!("{}{}", s.chars().take(n).collect::<String>(), TRUNC_FMT.replace("{n}", &(len - n).to_string()))
}

/// Cap a rendered trajectory at `max_chars`.
pub fn cap_rendered(lines: &[String], max_chars: usize) -> String {
    let rendered = lines.join("\n").trim().to_string();
    if rendered.is_empty() {
        return "(no parseable events in trajectory)".into();
    }
    let len = rendered.chars().count();
    if len > max_chars {
        let head: String = rendered.chars().take(max_chars).collect();
        return format!("{head}\n{}", TRUNC_FMT.replace("{n}", &(len - max_chars).to_string()));
    }
    rendered
}

/// Python's `str(x)` for a JSON value (strings unquoted, None -> "None").
pub fn pystr(v: Option<&serde_json::Value>) -> String {
    match v {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(o) => o.to_string(),
    }
}
