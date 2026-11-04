//! Trial identity: was this rollout finished, and by this configuration?
//!
//! Completion is recorded explicitly by the code that observed the run return; presence of
//! output is not enough. The marker also fingerprints the harness that produced the rollout,
//! and optionally the task inputs the rollout answered, so a rollout from a different
//! configuration -- or one answering a task that has since been corrected -- is never reused.

use crate::util;
use anyhow::Result;
use serde_json::{json, Map, Value};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};

pub const MARK: &str = ".adapt_complete";
pub const VERSION: u64 = 1;

fn ignored(p: &Path) -> bool {
    util::in_pycache(p)
        || p.extension().map(|e| e == "pyc").unwrap_or(false)
        || p.file_name().map(|n| n == ".DS_Store").unwrap_or(false)
}

fn digest_dir(h: &mut Sha1, root: &Path, skip_mark: bool) {
    for p in util::files_sorted(root) {
        if (skip_mark && p.file_name().map(|n| n == MARK).unwrap_or(false)) || ignored(&p) {
            continue;
        }
        h.update(util::rel(root, &p).as_bytes());
        h.update(b"\0");
        match std::fs::read(&p) {
            Ok(b) => h.update(&b),
            Err(_) => h.update(b"<unreadable>"),
        }
        h.update(b"\0");
    }
}

/// A stable digest of the harness tree that produced a rollout; `None` means a base rollout
/// on the inherited harness and digests to the shared value "base".
pub fn fingerprint(harness_dir: Option<&Path>) -> String {
    let Some(root) = harness_dir else { return "base".into() };
    if !root.is_dir() {
        return "absent".into();
    }
    let mut h = Sha1::new();
    digest_dir(&mut h, root, true);
    hex::encode(h.finalize())[..16].to_string()
}

/// A stable digest of what the agent was asked and what it was given.
pub fn inputs_fingerprint(task_prompt: &str, files_dir: Option<&Path>) -> String {
    let mut h = Sha1::new();
    h.update(b"prompt\0");
    h.update(task_prompt.as_bytes());
    h.update(b"\0");
    if let Some(root) = files_dir.filter(|r| r.is_dir()) {
        digest_dir(&mut h, root, false);
    }
    hex::encode(h.finalize())[..16].to_string()
}

/// Record that `run_id` finished. Call only after the runner returned normally.
pub fn mark_complete(where_: &Path, run_id: &str, harness_dir: Option<&Path>, inputs: Option<&str>, extra: Option<Map<String, Value>>) -> Result<PathBuf> {
    std::fs::create_dir_all(where_)?;
    let mut payload = Map::new();
    payload.insert("version".into(), json!(VERSION));
    payload.insert("run_id".into(), json!(run_id));
    payload.insert(
        "at".into(),
        json!(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()),
    );
    payload.insert("harness".into(), json!(fingerprint(harness_dir)));
    for (k, v) in extra.unwrap_or_default() {
        payload.insert(k, v);
    }
    if let Some(i) = inputs {
        payload.insert("inputs".into(), json!(i));
    }
    let mut sorted: Vec<(String, Value)> = payload.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let obj: Map<String, Value> = sorted.into_iter().collect();
    let tmp = where_.join(format!("{MARK}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, serde_json::to_string_pretty(&Value::Object(obj))?)?;
    let fin = where_.join(MARK);
    std::fs::rename(&tmp, &fin)?;
    Ok(fin)
}

pub fn read_mark(where_: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(where_.join(MARK)).ok()?).ok()
}

fn inputs_match(m: &Value, inputs: Option<&str>) -> bool {
    match (m.get("inputs").and_then(|v| v.as_str()), inputs) {
        (None, _) | (_, None) => true,
        (Some(a), Some(b)) => a == b,
    }
}

/// Whether the artifacts at `where_` may be reused.
pub fn is_complete(where_: &Path, harness_dir: Option<&Path>, match_harness: bool, inputs: Option<&str>) -> bool {
    let Some(m) = read_mark(where_) else { return false };
    if m.get("version").and_then(|v| v.as_u64()) != Some(VERSION) {
        return false;
    }
    if inputs.is_some() && !inputs_match(&m, inputs) {
        return false;
    }
    if !match_harness {
        return true;
    }
    m.get("harness").and_then(|v| v.as_str()) == Some(fingerprint(harness_dir).as_str())
}

/// A sentence for a log line explaining why `where_` is or is not reusable.
pub fn why_not(where_: &Path, harness_dir: Option<&Path>, inputs: Option<&str>) -> String {
    let Some(m) = read_mark(where_) else {
        return format!("no {MARK} -- never finished, or finished before completion was recorded");
    };
    if m.get("version").and_then(|v| v.as_u64()) != Some(VERSION) {
        return format!("marker version {} != {VERSION}", m.get("version").cloned().unwrap_or(Value::Null));
    }
    if !inputs_match(&m, inputs) {
        return format!(
            "task inputs {} != {} -- the task prompt or its source files changed since this rollout finished",
            m.get("inputs").and_then(|v| v.as_str()).unwrap_or("None"),
            inputs.unwrap_or("None")
        );
    }
    let want = fingerprint(harness_dir);
    let got = m.get("harness").and_then(|v| v.as_str()).unwrap_or("").to_string();
    if want != got {
        return format!("harness fingerprint {got} != {want} -- produced by a different configuration");
    }
    "reusable".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_matches_python_algorithm() {
        let d = tempfile::tempdir().unwrap();
        util::write(&d.path().join("a.md"), "hello").unwrap();
        util::write(&d.path().join("s/b.py"), "x=1\n").unwrap();
        // python3: h=sha1(); for rel,b in [("a.md",b"hello"),("s/b.py",b"x=1\n")]:
        //   h.update(rel.encode()+b"\0"+b+b"\0")
        let mut h = Sha1::new();
        for (rel, b) in [("a.md", &b"hello"[..]), ("s/b.py", &b"x=1\n"[..])] {
            h.update(rel.as_bytes());
            h.update(b"\0");
            h.update(b);
            h.update(b"\0");
        }
        assert_eq!(fingerprint(Some(d.path())), hex::encode(h.finalize())[..16]);
        assert_eq!(fingerprint(None), "base");
    }

    #[test]
    fn mark_roundtrip() {
        let d = tempfile::tempdir().unwrap();
        mark_complete(d.path(), "r1", None, Some("abc"), None).unwrap();
        assert!(is_complete(d.path(), None, true, Some("abc")));
        assert!(!is_complete(d.path(), None, true, Some("zzz")));
        assert_eq!(why_not(d.path(), None, Some("abc")), "reusable");
    }
}
