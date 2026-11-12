//! The per-domain experience library R: what earlier tasks in this domain produced.
//!
//! ```text
//!     <root>/<run-tag>/<mode>/<domain>/INDEX.md
//!     <root>/<run-tag>/<mode>/<domain>/harnesses/<task>/harness/<rel>   the full adopted tree
//!     <root>/<run-tag>/<mode>/<domain>/harnesses/<task>/EXPERIENCE.md   why picked + lesson
//!     <root>/<run-tag>/<mode>/<domain>/harnesses/<task>/NO_ENTRY.md     reflected on, no lesson kept, and why
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

/// The four verdicts an edit row may carry.
pub const VERDICTS: [&str; 4] = ["helped", "hurt", "no observed effect", "confounded"];

/// Edit-row verdicts that are not one of `VERDICTS` (upstream #143). A verdict may be
/// followed by a short reason after punctuation ("confounded -- two edits at once").
pub fn bad_verdicts(edits: &str) -> Vec<String> {
    edits
        .lines()
        .filter_map(|l| l.trim().strip_prefix("verdict:"))
        .map(|v| v.trim().trim_matches(|c: char| c == '`' || c == '*').trim().to_string())
        .filter(|v| {
            let low = v.to_lowercase();
            !VERDICTS.iter().any(|ok| low == *ok || low.strip_prefix(ok).map(|rest| rest.starts_with(|c: char| !c.is_alphanumeric())).unwrap_or(false))
        })
        .collect()
}

/// The lesson fields a library entry needs: cue, lesson and evidence; and, for records with a
/// provenance (written since upstream #143), confidence too and a provenance that is not
/// `unverified`. Older records keep the older rule so resuming them keeps their lessons.
pub fn has_experience(experience: Option<&Value>) -> bool {
    let Some(o) = experience.and_then(|e| e.as_object()) else { return false };
    let filled = |k: &str| o.get(k).and_then(|v| v.as_str()).map(|s| !s.trim().is_empty()).unwrap_or(false);
    if !["cue", "lesson", "evidence"].iter().all(|k| filled(k)) {
        return false;
    }
    match o.get("provenance").and_then(|v| v.as_str()) {
        None => true,
        Some(p) => p != "unverified" && filled("confidence"),
    }
}

/// Why a reflection left no library entry, in words for NO_ENTRY.md and INDEX.md.
pub fn outcome_text(outcome: &str) -> &'static str {
    match outcome {
        "declined" => "declined: the reflection found no transferable lesson (NO_ENTRY)",
        "malformed" => "malformed: the reflection's lesson was incomplete or used a verdict outside helped / hurt / no observed effect / confounded",
        "session_failed" => "session failed: the reflection session did not complete",
        "unverified" => "unverified: the lesson's evidence cites no run of this task and no earlier library entry",
        _ => "no lesson recorded",
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

    /// Tasks with a lesson (EXPERIENCE.md).
    pub fn entries(&self) -> Vec<String> {
        self.all_dirs().into_iter().filter(|n| self.dir().join("harnesses").join(n).join("EXPERIENCE.md").is_file()).collect()
    }

    /// Tasks that were reflected on and left no lesson (NO_ENTRY.md).
    pub fn declined(&self) -> Vec<String> {
        self.all_dirs().into_iter().filter(|n| self.dir().join("harnesses").join(n).join("NO_ENTRY.md").is_file()).collect()
    }

    fn all_dirs(&self) -> Vec<String> {
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
            let drop: Vec<String> = self.all_dirs().into_iter().filter(|e| !keep.contains(e)).collect();
            self.forget_unlocked(&drop)
        })
    }

    /// Store one adopted harness plus why it was adopted, and rebuild INDEX.md from disk.
    /// A reflection that left no lesson is stored as NO_ENTRY.md with its reason and edit
    /// rows instead (upstream #143, ask 2), and the result is None.
    pub fn add(&self, task_id: &str, harness: &Tree, why: &str, label: &str, run_id: &str, experience: Option<&Value>) -> Result<Option<PathBuf>> {
        if !has_experience(experience) {
            // Record why only when a reflection ran and said; an empty record stays a gap.
            if let Some(exp) = experience.filter(|e| e.get("outcome").is_some()) {
                self.locked(|| self.decline_unlocked(task_id, label, run_id, exp))?;
            }
            return Ok(None);
        }
        self.locked(|| self.add_unlocked(task_id, harness, why, label, run_id, experience.unwrap()))
    }

    fn decline_unlocked(&self, task_id: &str, label: &str, run_id: &str, exp: &Value) -> Result<()> {
        let get = |k: &str| exp.get(k).map(value_text).unwrap_or_default();
        let outcome = get("outcome");
        let mut lines = vec![
            format!("# {task_id}"),
            String::new(),
            format!("- adopted: `{label}`{}", if run_id.is_empty() { String::new() } else { format!("  (run `{run_id}`)") }),
            format!("- **outcome**: {outcome}"),
            format!("- {}", outcome_text(&outcome)),
        ];
        if !get("error").trim().is_empty() {
            lines.push(format!("- error: {}", util::take_chars(get("error").trim(), 400)));
        }
        lines.push(String::new());
        if !get("edits").trim().is_empty() {
            lines.extend(["## What each edit did".into(), get("edits").trim().to_string(), String::new()]);
        }
        let body = lines.join("\n");
        let mut probe = BTreeMap::new();
        probe.insert("NO_ENTRY.md".to_string(), body.clone());
        context::assert_tree_has_no_grader_verdicts(&probe)?;
        let d = self.entry_dir(task_id);
        util::rmtree(&d);
        std::fs::create_dir_all(&d)?;
        util::write(&d.join("NO_ENTRY.md"), body)?;
        self.rebuild_index_unlocked()?;
        Ok(())
    }

    fn add_unlocked(&self, task_id: &str, harness: &Tree, why: &str, label: &str, run_id: &str, exp: &Value) -> Result<Option<PathBuf>> {
        context::assert_tree_has_no_grader_verdicts(harness)?;
        let mut probe = BTreeMap::new();
        probe.insert("EXPERIENCE.md".to_string(), format!("{exp}{why}"));
        context::assert_tree_has_no_grader_verdicts(&probe)?;
        let d = self.entry_dir(task_id);
        util::rmtree(&d.join("harness"));
        let _ = std::fs::remove_file(d.join("NO_ENTRY.md"));
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
            for f in ["cue", "lesson", "evidence", "confidence", "provenance"] {
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
            rows.push(if self.declined().is_empty() { "_(empty — this is the first task in this domain)_" } else { "_(no lessons yet)_" }.into());
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
        // Tried and left nothing transferable is not the same as never tried (upstream #143).
        let declined = self.declined();
        if !declined.is_empty() {
            rows.push("## Tried, no lesson kept".into());
            rows.push(String::new());
            rows.push("These earlier tasks were reflected on and left no entry. Their edit rows are still in each NO_ENTRY.md.".into());
            rows.push(String::new());
            for name in declined {
                let outcome = field(&self.dir().join("harnesses").join(&name).join("NO_ENTRY.md"), "outcome");
                rows.push(format!("- `harnesses/{name}/NO_ENTRY.md` — {}", outcome_text(&outcome)));
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_reflection_without_a_lesson_leaves_a_record() {
        let d = tempfile::tempdir().unwrap();
        let lib = Library::new(d.path().to_path_buf(), "accounting", crate::substrate::codex::codex_surface());
        let mut t = Tree::new();
        t.insert("AGENTS.md".into(), "base".into());
        let lesson = json!({"cue": "c", "lesson": "l", "evidence": "inv1 showed it", "confidence": "high", "provenance": "measured"});
        assert!(lib.add("t1", &t, "why", "agg1", "r1", Some(&lesson)).unwrap().is_some());
        let declined = json!({"outcome": "declined", "edits": "component: AGENTS.md\ncandidates: inv1\nverdict: confounded"});
        assert!(lib.add("t2", &t, "why", "agg1", "r2", Some(&declined)).unwrap().is_none());
        // A record with no outcome (no reflection ran) stays a gap.
        assert!(lib.add("t3", &t, "", "agg1", "r3", Some(&json!({}))).unwrap().is_none());
        assert_eq!(lib.entries(), vec!["t1"]);
        assert_eq!(lib.declined(), vec!["t2"]);
        let tomb = std::fs::read_to_string(lib.entry_dir("t2").join("NO_ENTRY.md")).unwrap();
        assert!(tomb.contains("- **outcome**: declined") && tomb.contains("verdict: confounded"), "{tomb}");
        let index = std::fs::read_to_string(lib.index()).unwrap();
        assert!(index.contains("## Tried, no lesson kept") && index.contains("`harnesses/t2/NO_ENTRY.md` — declined"), "{index}");
        assert!(!lib.entry_dir("t3").exists());
        // A later lesson for the same task replaces its no-lesson record.
        lib.add("t2", &t, "why", "agg1", "r2", Some(&lesson)).unwrap();
        assert!(lib.declined().is_empty() && lib.entries().len() == 2);
        // New-format records need confidence and a verified provenance; old ones keep the old rule.
        assert!(!has_experience(Some(&json!({"cue": "c", "lesson": "l", "evidence": "e", "confidence": "high", "provenance": "unverified"}))));
        assert!(!has_experience(Some(&json!({"cue": "c", "lesson": "l", "evidence": "e", "provenance": "measured"}))));
        assert!(has_experience(Some(&json!({"cue": "c", "lesson": "l", "evidence": "e"}))));
    }
}
