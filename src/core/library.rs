//! The per-domain experience library R: what earlier tasks in this domain produced.
//!
//! ```text
//!     <root>/<run-tag>/<mode>/<domain>/INDEX.md
//!     <root>/<run-tag>/<mode>/<domain>/harnesses/<task>/harness/<rel>   the full adopted tree
//!     <root>/<run-tag>/<mode>/<domain>/harnesses/<task>/EXPERIENCE.md   why picked + lesson
//! ```
//!
//! Retrieval is by construction: a domain's directory only ever holds same-domain entries.
//! The whole tree is carried, never a summary. Roots are scoped per run and mode.

use super::context;
use super::tree::{self, SurfaceSpec, Tree};
use crate::util;
use anyhow::Result;
use regex::Regex;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

pub fn slug(task_id: &str) -> String {
    task_id.replace('/', "__")
}

/// The lesson fields a library entry needs.
pub fn has_experience(experience: Option<&Value>) -> bool {
    match experience.and_then(|e| e.as_object()) {
        Some(o) => ["cue", "lesson", "evidence"]
            .iter()
            .all(|k| o.get(*k).and_then(|v| v.as_str()).map(|s| !s.trim().is_empty()).unwrap_or(false)),
        None => false,
    }
}

/// One directory name for a domain, never a path.
pub fn domain_dir(domain: &str) -> String {
    let re = Regex::new(r"\s*[/\\]\s*").unwrap();
    let s = re.replace_all(domain, "__").trim().to_string();
    if s.is_empty() { "unknown".into() } else { s }
}

/// One (run, domain) library: read before each task, appended after.
#[derive(Debug, Clone)]
pub struct Library {
    pub root: PathBuf,
    pub domain: String,
    pub surface: SurfaceSpec,
}

impl Library {
    pub fn new(root: PathBuf, domain: &str, surface: SurfaceSpec) -> Library {
        Library { root, domain: domain.into(), surface }
    }
    pub fn dir(&self) -> PathBuf {
        self.root.join(domain_dir(&self.domain))
    }
    pub fn index(&self) -> PathBuf {
        self.dir().join("INDEX.md")
    }
    pub fn entry_dir(&self, task_id: &str) -> PathBuf {
        self.dir().join("harnesses").join(slug(task_id))
    }
    /// Hold this domain's lock while `f` runs. Workers and concurrent runs share a library, and
    /// writing an entry then rebuilding INDEX.md from disk must not interleave (upstream #54).
    fn locked<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        use fs2::FileExt;
        std::fs::create_dir_all(&self.root)?;
        let lf = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(self.root.join(format!(".{}.lock", domain_dir(&self.domain))))?;
        lf.lock_exclusive()?;
        let r = f();
        let _ = FileExt::unlock(&lf);
        r
    }

    pub fn exists(&self) -> bool {
        self.index().is_file()
    }

    /// The directory handed to an improver session, or None when this domain is empty.
    pub fn context_dir(&self) -> Option<PathBuf> {
        let has = util::files_sorted(&self.dir()).iter().any(|p| p.file_name().map(|n| n == "EXPERIENCE.md").unwrap_or(false));
        (self.exists() && has).then(|| self.dir())
    }

    pub fn entries(&self) -> Vec<String> {
        let d = self.dir().join("harnesses");
        let mut v: Vec<String> = std::fs::read_dir(&d)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.file_name().to_string_lossy().to_string())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    pub fn tree_of(&self, task_id: &str) -> Tree {
        tree::read_tree(&self.surface, &self.entry_dir(task_id).join("harness"), None)
    }

    /// Drop the entries for these tasks and rebuild INDEX.md; unknown tasks are ignored.
    pub fn forget(&self, task_ids: &[String]) -> Result<()> {
        self.locked(|| self.forget_unlocked(task_ids))
    }

    fn forget_unlocked(&self, task_ids: &[String]) -> Result<()> {
        let mut removed = false;
        for tid in task_ids {
            let d = self.entry_dir(tid);
            if d.is_dir() {
                util::rmtree(&d);
                removed = true;
            }
        }
        if removed {
            self.rebuild_index_unlocked()?;
        }
        Ok(())
    }

    /// Keep only the entries preceding the current task in this stream.
    pub fn retain(&self, task_ids: &[String]) -> Result<()> {
        let keep: HashSet<String> = task_ids.iter().map(|t| slug(t)).collect();
        self.locked(|| {
            let drop: Vec<String> = self.entries().into_iter().filter(|e| !keep.contains(e)).collect();
            self.forget_unlocked(&drop)
        })
    }

    /// Store one adopted harness plus why it was adopted, and rebuild INDEX.md from disk.
    pub fn add(&self, task_id: &str, harness: &Tree, why: &str, label: &str, run_id: &str, experience: Option<&Value>) -> Result<Option<PathBuf>> {
        if !has_experience(experience) {
            return Ok(None);
        }
        self.locked(|| self.add_unlocked(task_id, harness, why, label, run_id, experience.unwrap()))
    }

    fn add_unlocked(&self, task_id: &str, harness: &Tree, why: &str, label: &str, run_id: &str, exp: &Value) -> Result<Option<PathBuf>> {
        context::assert_tree_has_no_grader_verdicts(harness)?;
        let mut probe = BTreeMap::new();
        probe.insert("EXPERIENCE.md".to_string(), format!("{exp}{why}"));
        context::assert_tree_has_no_grader_verdicts(&probe)?;
        let d = self.entry_dir(task_id);
        util::rmtree(&d.join("harness"));
        std::fs::create_dir_all(d.join("harness"))?;
        tree::write_tree(&self.surface, harness, &d.join("harness"), false)?;
        let get = |k: &str| exp.get(k).map(value_text).unwrap_or_default();
        let mut lines = vec![
            format!("# {task_id}"),
            String::new(),
            format!("- adopted: `{label}`{}", if run_id.is_empty() { String::new() } else { format!("  (run `{run_id}`)") }),
            format!("- components: {}", inventory(&self.surface, harness)),
            String::new(),
        ];
        if !why.trim().is_empty() {
            lines.extend(["## Why this was chosen".into(), why.trim().to_string(), String::new()]);
        }
        if !get("edits").is_empty() {
            lines.extend(["## What each edit did".into(), get("edits").trim().to_string(), String::new()]);
        }
        if !get("lesson").is_empty() {
            lines.push("## Transferable lesson".into());
            for f in ["cue", "lesson", "evidence", "confidence"] {
                if !get(f).is_empty() {
                    lines.push(format!("- **{f}**: {}", get(f).trim()));
                }
            }
            lines.push(String::new());
        }
        util::write(&d.join("EXPERIENCE.md"), lines.join("\n"))?;
        self.rebuild_index_unlocked()?;
        Ok(Some(d))
    }

    /// Regenerate INDEX.md from what is on disk.
    pub fn rebuild_index(&self) -> Result<PathBuf> {
        self.locked(|| self.rebuild_index_unlocked())
    }

    fn rebuild_index_unlocked(&self) -> Result<PathBuf> {
        let mut rows = vec![
            format!("# Library — {}", self.domain),
            String::new(),
            "Each entry is an earlier task in THIS domain whose harness was chosen, stored with its FULL component tree and why it was picked. Browse this index, then open the entries that look relevant to the matter in front of you.".into(),
            String::new(),
        ];
        let mut names = self.entries();
        // Lessons the operator pinned as reliable come first.
        let pinned = |n: &String| self.dir().join("harnesses").join(n).join("PINNED").is_file();
        names.sort_by_key(|n| !pinned(n));
        if names.is_empty() {
            rows.push("_(empty — this is the first task in this domain)_".into());
        }
        for name in names {
            let ed = self.dir().join("harnesses").join(&name);
            let cue = field(&ed.join("EXPERIENCE.md"), "cue");
            let t = tree::read_tree(&self.surface, &ed.join("harness"), None);
            rows.push(format!("## `harnesses/{name}/`"));
            if ed.join("PINNED").is_file() {
                rows.push("- pinned:  the operator marked this lesson as reliable".into());
            }
            rows.push(format!("- harness: `harnesses/{name}/harness/` — {}", inventory(&self.surface, &t)));
            rows.push(format!("- why:     `harnesses/{name}/EXPERIENCE.md`"));
            if !cue.is_empty() {
                rows.push(format!("- cue:     {cue}"));
            }
            rows.push(String::new());
        }
        std::fs::create_dir_all(self.dir())?;
        util::write(&self.index(), rows.join("\n"))?;
        Ok(self.index())
    }
}

fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        o => o.to_string(),
    }
}

/// A one-line census of the components actually in this harness.
pub fn inventory(surface: &SurfaceSpec, t: &Tree) -> String {
    let mut st: Vec<(&str, usize)> = vec![("skills", 0), ("sub_agents", 0), ("scripts", 0)];
    let skill_prefix = format!("{}/", surface.skill_dir);
    let plugin_prefix = surface.plugin_dir.map(|p| format!("{p}/"));
    let mut bump = |k: &'static str| {
        if let Some(e) = st.iter_mut().find(|(n, _)| *n == k) {
            e.1 += 1;
        } else {
            st.push((k, 1));
        }
    };
    for rel in t.keys() {
        if rel.starts_with(&skill_prefix) && rel.ends_with("SKILL.md") {
            bump("skills");
        } else if rel.starts_with("sub_agents/") {
            bump("sub_agents");
        } else if rel.starts_with("scripts/") {
            bump("scripts");
        } else if plugin_prefix.as_ref().map(|p| rel.starts_with(p.as_str())).unwrap_or(false) {
            bump("plugins");
        } else if Some(rel.as_str()) == surface.hooks_file {
            bump("hooks");
        }
    }
    let on = util::char_len(&tree::always_on_text(surface, t));
    let bits: Vec<String> = st.iter().filter(|(_, v)| *v > 0).map(|(k, v)| format!("{v} {k}")).collect();
    let bits = if bits.is_empty() { "always-on edits only".to_string() } else { bits.join(", ") };
    format!("{bits}; {} always-on chars", context::thousands(on))
}

fn field(path: &Path, name: &str) -> String {
    let Ok(body) = util::read_lossy(path) else { return String::new() };
    let prefix = format!("- **{name}**:");
    body.lines()
        .find(|l| l.starts_with(&prefix))
        .map(|l| l.split_once(':').unwrap().1.trim().to_string())
        .unwrap_or_default()
}

/// Per-run library root `<base>/<benchmark>/k<k>-<variant>`; an explicit `tag` opts into
/// sharing one library across runs.
pub fn root_for(base: &Path, benchmark: &str, k: usize, variant: &str, tag: &str) -> PathBuf {
    if !tag.is_empty() {
        return base.join(benchmark).join(tag);
    }
    let mut parts = vec![format!("k{k}")];
    if !variant.is_empty() {
        parts.push(variant.into());
    }
    base.join(benchmark).join(parts.join("-"))
}
