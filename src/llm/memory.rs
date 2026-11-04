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
    /// Embedding for semantic recall, and the model that made it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vec: Vec<f32>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub vec_model: String,
    /// For notes in the shared store: the project that shared it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub project: String,
}

/// Unused, unpinned notes fade out of recall over time (half-life), and are archived after this.
pub const HALF_LIFE_DAYS: f64 = 30.0;
pub const FADE_AFTER_DAYS: i64 = 90;

#[derive(Debug, Clone)]
pub struct Store {
    pub path: PathBuf,
    /// Embedding model id for semantic recall (None: keywords only).
    pub embed: Option<String>,
    /// A store shared across projects, also searched by `recall`.
    pub shared: Option<PathBuf>,
}

/// Where notes shared across projects live.
pub fn shared_path() -> PathBuf {
    crate::config::get().runs.join("shared_memory").join("memory.json")
}

fn age_days(n: &Note) -> f64 {
    let t = if n.last_used.is_empty() { &n.created } else { &n.last_used };
    chrono::DateTime::parse_from_rfc3339(t).map(|t| (chrono::Utc::now() - t.with_timezone(&chrono::Utc)).num_minutes() as f64 / 1440.0).unwrap_or(0.0).max(0.0)
}

fn words(s: &str) -> HashSet<String> {
    WORD.find_iter(&s.to_lowercase()).map(|m| m.as_str().to_string()).filter(|w| !STOP.contains(&w.as_str())).collect()
}

fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

impl Store {
    pub fn open(project_root: &Path) -> Store {
        Store { path: project_root.join("memory.json"), embed: None, shared: None }
    }

    pub fn with(mut self, embed: Option<String>, shared: bool) -> Store {
        self.embed = embed.filter(|e| !e.trim().is_empty());
        self.shared = shared.then(shared_path);
        self
    }

    fn embed_one(&self, text: &str) -> Option<Vec<f32>> {
        let m = self.embed.as_ref()?;
        super::client::embed(m, &[text.to_string()]).ok().and_then(|mut v| v.pop())
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
        let vec = self.embed_one(text).unwrap_or_default();
        let vec_model = if vec.is_empty() { String::new() } else { self.embed.clone().unwrap_or_default() };
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
            let note = Note { id, text: text.into(), tags, status: status.into(), pinned: false, promoted: false, source, created: crate::util::now_iso(), uses: 0, last_used: String::new(), note: String::new(), vec, vec_model, project: String::new() };
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

    /// The best notes for `query`: active ones, plus this goal's own candidates, plus the shared
    /// store when enabled. Keyword overlap and tags, plus embedding similarity when an embedding
    /// model is set; unused notes fade with a 30-day half-life (pinned ones do not).
    pub fn recall(&self, query: &str, k: usize, goal: &str) -> Result<Vec<Note>> {
        let mut hits = self.recall_scored(query, k, goal)?;
        if let Some(sp) = &self.shared {
            let sh = Store { path: sp.clone(), embed: self.embed.clone(), shared: None };
            hits.extend(sh.recall_scored(query, k, "")?);
        }
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(hits.into_iter().take(k.clamp(1, 10)).map(|(_, n)| n).collect())
    }

    fn recall_scored(&self, query: &str, k: usize, goal: &str) -> Result<Vec<(f64, Note)>> {
        let q = words(query);
        let qv = if q.is_empty() { None } else { self.embed_one(query) };
        // Embed notes that have no vector yet (or one from another model), a batch at a time.
        if let (Some(m), Some(_)) = (&self.embed, &qv) {
            let missing: Vec<(String, String)> = self.load().into_iter().filter(|n| n.status != "archived" && (n.vec.is_empty() || &n.vec_model != m)).take(64).map(|n| (n.id, n.text)).collect();
            if !missing.is_empty() {
                if let Ok(vs) = super::client::embed(m, &missing.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>()) {
                    let _ = self.update(|notes| {
                        for ((id, _), v) in missing.iter().zip(vs) {
                            if let Some(n) = notes.iter_mut().find(|n| &n.id == id) {
                                n.vec = v;
                                n.vec_model = m.clone();
                            }
                        }
                        Ok(())
                    });
                }
            }
        }
        self.update(|notes| {
            let mut scored: Vec<(f64, usize)> = notes
                .iter()
                .enumerate()
                .filter(|(_, n)| n.status == "active" || (n.status == "candidate" && n.source.goal == goal))
                .map(|(i, n)| {
                    let w = words(&n.text);
                    let overlap = q.intersection(&w).count() as f64;
                    let tag = n.tags.iter().filter(|t| q.contains(*t)).count() as f64;
                    let sim = qv.as_ref().map(|v| super::client::cosine(v, &n.vec) as f64).unwrap_or(0.0);
                    let s = if q.is_empty() { 1.0 } else { overlap + 2.0 * tag + if sim > 0.25 { 4.0 * sim } else { 0.0 } };
                    let fade = if n.pinned { 1.0 } else { 0.5f64.powf(age_days(n) / HALF_LIFE_DAYS).max(0.25) };
                    (s, (s + 0.1 * (n.uses.min(20) as f64) + if n.pinned { 0.5 } else { 0.0 }) * fade, i)
                })
                .filter(|(raw, _, _)| *raw >= 1.0)
                .map(|(_, ranked, i)| (ranked, i))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(notes[b.1].created.cmp(&notes[a.1].created)));
            let now = crate::util::now_iso();
            let out: Vec<(f64, Note)> = scored.iter().take(k.clamp(1, 10)).cloned().collect::<Vec<_>>().into_iter().map(|(sc, i)| {
                notes[i].uses += 1;
                notes[i].last_used = now.clone();
                let mut n = notes[i].clone();
                n.vec.clear();
                (sc, n)
            }).collect();
            Ok(out)
        })
    }

    /// Archive unpinned notes nobody recalled for FADE_AFTER_DAYS. Returns how many faded.
    pub fn fade(&self) -> Result<usize> {
        self.update(|notes| {
            let mut n = 0;
            for x in notes.iter_mut().filter(|x| x.status == "active" && !x.pinned && x.uses == 0 && age_days(x) > FADE_AFTER_DAYS as f64) {
                x.status = "archived".into();
                x.note = format!("faded: not recalled for {FADE_AFTER_DAYS} days");
                n += 1;
            }
            Ok(n)
        })
    }

    /// Copy a note into the shared store so other projects can recall it.
    pub fn share(&self, id: &str, project: &str) -> Result<Note> {
        let n = self.load().into_iter().find(|n| n.id == id).ok_or_else(|| anyhow::anyhow!("no note {id}"))?;
        if n.status == "archived" {
            bail!("restore the note before sharing it");
        }
        let sh = Store { path: shared_path(), embed: None, shared: None };
        sh.update(|notes| {
            if let Some(ex) = notes.iter().find(|x| norm(&x.text) == norm(&n.text)) {
                return Ok(ex.clone());
            }
            let mut c = n.clone();
            c.status = "active".into();
            c.project = project.into();
            c.uses = 0;
            notes.push(c.clone());
            Ok(c)
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

    #[test]
    fn semantic_recall_and_fading() {
        let d = tempfile::tempdir().unwrap();
        let st = Store::open(d.path()).with(Some("mock/embed".into()), false);
        st.remember("the export uses semicolons as the delimiter", &[], src("g1", "base1"), "active").unwrap();
        // no shared keyword ("semicolon" vs "semicolons", "delimiters" vs "delimiter"): only the embedding matches
        let kw_only = Store::open(d.path());
        assert!(kw_only.recall("semicolon delimiters", 5, "g9").unwrap().is_empty());
        let hits = st.recall("semicolon delimiters", 5, "g9").unwrap();
        assert_eq!(hits.len(), 1, "semantic recall finds it");
        assert!(hits[0].vec.is_empty(), "vectors stay out of tool results");
        // an unused note from long ago fades out
        st.remember("an old fact nobody needs any more", &[], src("g1", "base1"), "active").unwrap();
        st.update(|notes| {
            for n in notes.iter_mut().filter(|n| n.text.starts_with("an old")) {
                n.created = "2020-01-01T00:00:00+00:00".into();
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(st.fade().unwrap(), 1);
        assert_eq!(st.load().iter().filter(|n| n.status == "archived").count(), 1);
    }

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
