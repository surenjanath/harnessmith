//! Project memory: notes the solver saves with `remember` and finds again with `recall`.
//!
//! Notes live in `<project>/memory.json`, outside the harness (which is read-only during a run).
//! A note starts as a *candidate*. When a run ends, notes from the adopted attempt and the base
//! runs become *active*, the rest are *archived*, so experiments that lost do not pollute memory.
//! Pinned notes are put into every solver's instructions; adopted notes can be promoted into the
//! base harness's LongTermMEMORY.md.

use anyhow::{bail, Result};
use fs2::FileExt;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const MAX_NOTES: usize = 300;
pub const MAX_NOTE_CHARS: usize = 600;
/// Chars of promoted notes allowed in LongTermMEMORY.md (the always-on cap is shared).
pub const PROMOTE_BUDGET: usize = 4_000;
const PROMOTE_HEADER: &str = "## Learned notes (promoted from adopted runs)";

static SECRET: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(sk-[A-Za-z0-9_\-]{16,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{30,})").unwrap()
});
static WORD: Lazy<Regex> = Lazy::new(|| Regex::new(r"[a-z0-9][a-z0-9_\-]{2,}").unwrap());
const STOP: &[&str] = &["the", "and", "for", "with", "that", "this", "from", "are", "was", "use", "into", "when", "then", "than", "not", "but", "you", "your", "its", "all", "any", "can", "has", "have", "how", "what", "which", "will"];

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Source {
    #[serde(default)]
    pub goal: String,
    #[serde(default)]
    pub mode: String,
    #[serde(default)]
    pub node: String,
    /// "agent" or "user".
    #[serde(default)]
    pub by: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub tags: Vec<String>,
    /// "candidate" | "active" | "archived".
    pub status: String,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub promoted: bool,
    #[serde(default)]
    pub source: Source,
    pub created: String,
    #[serde(default)]
    pub uses: u32,
    #[serde(default)]
    pub last_used: String,
    /// Why a note was archived, when it was.
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct Store {
    pub path: PathBuf,
}

fn words(s: &str) -> HashSet<String> {
    WORD.find_iter(&s.to_lowercase()).map(|m| m.as_str().to_string()).filter(|w| !STOP.contains(&w.as_str())).collect()
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

impl Store {
    pub fn open(project_root: &Path) -> Store {
        Store { path: project_root.join("memory.json") }
    }

    pub fn load(&self) -> Vec<Note> {
        std::fs::read_to_string(&self.path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }

    /// Run `f` on the notes under an exclusive lock, then save them.
    pub fn update<R>(&self, f: impl FnOnce(&mut Vec<Note>) -> Result<R>) -> Result<R> {
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(self.path.with_extension("lock"))?;
        lock.lock_exclusive()?;
        let mut notes = self.load();
        let r = f(&mut notes);
        if r.is_ok() {
            let tmp = self.path.with_extension("json.tmp");
            std::fs::write(&tmp, serde_json::to_string_pretty(&notes)?)?;
            std::fs::rename(&tmp, &self.path)?;
        }
        let _ = FileExt::unlock(&lock);
        r
    }

    /// Save a note. Returns the note and whether it was already known (then it is refreshed).
    pub fn remember(&self, text: &str, tags: &[String], source: Source, status: &str) -> Result<(Note, bool)> {
        let text = text.trim();
        let n = text.chars().count();
        if n < 8 {
            bail!("a note needs at least 8 characters: write the fact or technique itself");
        }
        if n > MAX_NOTE_CHARS {
            bail!("a note is at most {MAX_NOTE_CHARS} characters (this one is {n}); keep one fact per note");
        }
        if SECRET.is_match(text) {
            bail!("this note looks like it contains a key or token; secrets are never stored in memory");
        }
        let tags: Vec<String> = tags.iter().map(|t| t.trim().to_lowercase()).filter(|t| !t.is_empty()).take(8).collect();
        self.update(|notes| {
            if let Some(ex) = notes.iter_mut().find(|x| norm(&x.text) == norm(text) && x.status != "archived") {
                for t in &tags {
                    if !ex.tags.contains(t) {
                        ex.tags.push(t.clone());
                    }
                }
                return Ok((ex.clone(), true));
            }
            let id = format!("m{}", &crate::util::sha256_hex(format!("{}{}", text, crate::util::now_iso()).as_bytes())[..10]);
            let note = Note { id, text: text.into(), tags, status: status.into(), pinned: false, promoted: false, source, created: crate::util::now_iso(), uses: 0, last_used: String::new(), note: String::new() };
            notes.push(note.clone());
            // Over the cap: archive the least useful unpinned notes.
            let live = notes.iter().filter(|x| x.status != "archived").count();
            if live > MAX_NOTES {
                let mut idx: Vec<usize> = (0..notes.len()).filter(|&i| notes[i].status != "archived" && !notes[i].pinned).collect();
                idx.sort_by_key(|&i| (notes[i].uses, notes[i].created.clone()));
                for &i in idx.iter().take(live - MAX_NOTES) {
                    notes[i].status = "archived".into();
                    notes[i].note = "archived to stay under the note cap".into();
                }
            }
            Ok((note, false))
        })
    }

    /// The best notes for `query`: active ones, plus this goal's own candidates.
    pub fn recall(&self, query: &str, k: usize, goal: &str) -> Result<Vec<Note>> {
        let q = words(query);
        self.update(|notes| {
            let mut scored: Vec<(f64, usize)> = notes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.status == "active" || (n.status == "candidate" && n.source.goal == goal))
                .map(|(i, n)| {
                    let w = words(&n.text);
                    let overlap = q.intersection(&w).count() as f64;
                    let tag = n.tags.iter().filter(|t| q.contains(*t)).count() as f64;
                    let s = if q.is_empty() { 1.0 } else { overlap + 2.0 * tag };
                    (s + 0.1 * (n.uses.min(20) as f64) + if n.pinned { 0.5 } else { 0.0 }, i)
                })
                .filter(|(s, _)| *s >= 1.0)
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(notes[b.1].created.cmp(&notes[a.1].created)));
            let now = crate::util::now_iso();
            let out: Vec<Note> = scored.iter().take(k.clamp(1, 10)).map(|(_, i)| *i).collect::<Vec<_>>().into_iter().map(|i| {
                notes[i].uses += 1;
                notes[i].last_used = now.clone();
                notes[i].clone()
            }).collect();
            Ok(out)
        })
    }

    pub fn pinned(&self) -> Vec<Note> {
        self.load().into_iter().filter(|n| n.pinned && n.status != "archived").collect()
    }

    /// After a run: notes from adopted nodes and base runs become active, other candidates of
    /// this goal are archived. Returns (activated, archived, the activated notes from adopted nodes).
    pub fn settle(&self, goal: &str, adopted: &[(String, String)]) -> Result<(usize, usize, Vec<Note>)> {
        self.update(|notes| {
            let (mut on, mut off, mut won) = (0, 0, vec![]);
            for n in notes.iter_mut().filter(|n| n.status == "candidate" && n.source.goal == goal) {
                let from_adopted = adopted.iter().any(|(m, l)| *l == n.source.node && (m.is_empty() || *m == n.source.mode));
                if from_adopted || n.source.node.starts_with("base") {
                    n.status = "active".into();
                    on += 1;
                    if from_adopted {
                        won.push(n.clone());
                    }
                } else {
                    n.status = "archived".into();
                    n.note = "from an attempt that was not adopted".into();
                    off += 1;
                }
            }
            Ok((on, off, won))
        })
    }

    /// Append notes to `<base>/LongTermMEMORY.md` under the learned-notes section, within the
    /// promotion budget, and mark them promoted. Returns how many were written.
    pub fn promote(&self, base: &Path, ids: &[String]) -> Result<usize> {
        let f = base.join("LongTermMEMORY.md");
        let mut body = std::fs::read_to_string(&f).unwrap_or_default();
        let used = body.split(PROMOTE_HEADER).nth(1).map(|s| s.chars().count()).unwrap_or(0);
        let mut budget = PROMOTE_BUDGET.saturating_sub(used);
        let mut written = vec![];
        self.update(|notes| {
            for n in notes.iter_mut().filter(|n| ids.contains(&n.id) && !n.promoted && n.status != "archived") {
                let line = format!("- {}\n", n.text.replace('\n', " "));
                if line.chars().count() > budget || body.contains(line.trim()) {
                    continue;
                }
                budget -= line.chars().count();
                n.promoted = true;
                written.push(line);
            }
            Ok(())
        })?;
        if written.is_empty() {
            return Ok(0);
        }
        if !body.contains(PROMOTE_HEADER) {
            if !body.trim().is_empty() && !body.ends_with('\n') {
                body.push('\n');
            }
            body.push_str(&format!("{}{PROMOTE_HEADER}\n", if body.trim().is_empty() { "" } else { "\n" }));
        }
        for l in &written {
            body.push_str(l);
        }
        crate::util::write(&f, &body)?;
        Ok(written.len())
    }
}

/// What the solver sees about memory in its instructions.
pub fn system_section(pinned: &[Note]) -> String {
    let mut s = String::from("# PROJECT MEMORY\nNotes saved by earlier goals in this project. Call `recall` with a few keywords before planning, and `remember` a durable, reusable fact or technique you discover (one per note; never the task's answer, never a secret).");
    if !pinned.is_empty() {
        s.push_str("\nPinned notes:");
        for n in pinned {
            s.push_str(&format!("\n- {}", n.text.replace('\n', " ")));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(goal: &str, node: &str) -> Source {
        Source { goal: goal.into(), mode: "agg".into(), node: node.into(), by: "agent".into() }
    }

    #[test]
    fn remember_recall_settle_promote() {
        let d = tempfile::tempdir().unwrap();
        let st = Store::open(d.path());
        let (a, dup) = st.remember("regions.csv uses a semicolon delimiter", &["csv".into()], src("g1", "base1"), "candidate").unwrap();
        assert!(!dup);
        assert!(st.remember("Regions.csv uses a  semicolon delimiter", &[], src("g1", "inv1"), "candidate").unwrap().1, "duplicates merge");
        st.remember("percent change is (q2 - q1) / q1, round at the end", &[], src("g1", "agg1"), "candidate").unwrap();
        st.remember("a dead end: parsing with split(',') breaks quoted cells", &[], src("g1", "dom1"), "candidate").unwrap();
        assert!(st.remember("key sk-abcdefghijklmnopqrstuv", &[], src("g1", "base1"), "candidate").is_err());
        assert!(st.remember("short", &[], src("g1", "base1"), "candidate").is_err());

        // another goal cannot see g1's candidates yet
        assert!(st.recall("csv delimiter", 5, "g2").unwrap().is_empty());
        assert_eq!(st.recall("csv delimiter", 5, "g1").unwrap()[0].id, a.id);

        let (on, off, won) = st.settle("g1", &[("agg".into(), "agg1".into())]).unwrap();
        assert_eq!((on, off, won.len()), (2, 1, 1));
        let hits = st.recall("percent change q1 q2", 5, "g2").unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.starts_with("percent change"));
        assert!(st.recall("quoted cells split", 5, "g2").unwrap().is_empty(), "archived notes are not recalled");

        let base = d.path().join("base");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("LongTermMEMORY.md"), "Existing memory.\n").unwrap();
        assert_eq!(st.promote(&base, &[won[0].id.clone()]).unwrap(), 1);
        assert_eq!(st.promote(&base, &[won[0].id.clone()]).unwrap(), 0, "promoted once");
        let m = std::fs::read_to_string(base.join("LongTermMEMORY.md")).unwrap();
        assert!(m.starts_with("Existing memory.\n\n## Learned notes"));
        assert!(m.contains("- percent change is (q2 - q1) / q1"));
    }
}
