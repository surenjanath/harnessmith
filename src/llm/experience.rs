//! Experience: verified solutions a project keeps, and reuses as worked examples.
//!
//! When the answer engine reaches an answer that independent attempts agree on, the question,
//! the shape of its input files and the program that computed it are kept here. The next
//! question of the same kind gets that program as a worked example, which is the cheapest way
//! there is to make a small model right: it adapts a solution instead of inventing one.
//!
//! Each example carries a record of how it did in rehearsal (see `sleep`): whether the model
//! needed it, and whether it helped. Examples that never help are forgotten.

use anyhow::Result;
use fs2::FileExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const MAX_ACTIVE: usize = 80;

/// The shape of an input file: enough to tell "the same kind of file" without its contents.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FileSig {
    pub name: String,
    /// csv | tsv | json | log | text | other
    pub kind: String,
    /// CSV: the header line. JSON: its top-level keys. Otherwise empty.
    #[serde(default)]
    pub header: String,
    #[serde(default)]
    pub lines: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Exemplar {
    pub id: String,
    pub at: String,
    /// The goal it was learned from (its input files are the material for rehearsal).
    #[serde(default)]
    pub goal: String,
    pub question: String,
    #[serde(default)]
    pub files: Vec<FileSig>,
    pub program: String,
    #[serde(default)]
    pub output: String,
    pub answer: String,
    #[serde(default)]
    pub agree: usize,
    #[serde(default)]
    pub answered: usize,
    #[serde(default)]
    pub model: String,
    /// "active" | "archived"
    #[serde(default = "active")]
    pub status: String,
    /// Why it was archived, when it was.
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub pinned: bool,
    /// Times it was shown as a worked example.
    #[serde(default)]
    pub uses: u32,
    /// Rehearsal record: practice runs, how many the model got right without it, how many it
    /// got right only with it, and how many it got wrong even with it.
    #[serde(default)]
    pub dreams: u32,
    #[serde(default)]
    pub cold_ok: u32,
    #[serde(default)]
    pub helped: u32,
    #[serde(default)]
    pub failed: u32,
    #[serde(default)]
    pub last_dream: String,
    /// "goal" (learned from a real goal) or "dream" (learned by exploring during sleep).
    #[serde(default)]
    pub origin: String,
}

fn active() -> String {
    "active".into()
}

impl Exemplar {
    /// How much this example is worth showing, from rehearsal: +1 when it turned failures into
    /// right answers, about 0 when the model did not need it, negative when it did not help.
    pub fn value(&self) -> f64 {
        if self.dreams == 0 {
            return 0.0;
        }
        ((self.helped as f64 - 0.5 * self.failed as f64) / self.dreams as f64).clamp(-1.0, 1.0)
    }
}

/// The shape of the file at `workspace/rel`.
pub fn sig_of(workspace: &Path, rel: &str) -> FileSig {
    let ext = rel.rsplit_once('.').map(|(_, e)| e.to_lowercase()).unwrap_or_default();
    let kind = match ext.as_str() {
        "csv" => "csv",
        "tsv" => "tsv",
        "json" | "jsonl" => "json",
        "log" => "log",
        "txt" | "md" | "text" => "text",
        _ => "other",
    };
    let mut s = FileSig { name: rel.to_string(), kind: kind.into(), ..Default::default() };
    let Ok(bytes) = std::fs::read(workspace.join(rel)) else { return s };
    if bytes[..bytes.len().min(4096)].contains(&0) {
        s.kind = "other".into();
        return s;
    }
    let text = String::from_utf8_lossy(&bytes);
    s.lines = text.lines().count();
    match kind {
        "csv" | "tsv" => s.header = crate::util::take_chars(text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim(), 240),
        "json" => {
            if let Ok(serde_json::Value::Object(m)) = serde_json::from_str::<serde_json::Value>(&text) {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort();
                s.header = crate::util::take_chars(&keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(","), 240);
            }
        }
        _ => {}
    }
    s
}

const STOP: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "and", "or", "for", "with", "by", "it", "its", "this", "that", "as", "be", "has", "have", "had", "from", "what", "which", "who", "when",
    "where", "how", "many", "much", "does", "do", "did", "all", "each", "per", "using", "use", "answer", "give", "me", "please", "there", "these", "those", "their", "than", "then", "across", "over", "whose",
];

/// The words of a question that say what kind of question it is (numbers and codes dropped).
fn content_words(q: &str) -> BTreeSet<String> {
    let re = Regex::new(r"[A-Za-z][A-Za-z_]{1,}").unwrap();
    re.find_iter(&q.to_lowercase()).map(|m| m.as_str().to_string()).filter(|w| !STOP.contains(&w.as_str())).collect()
}

/// A question with its particulars masked: two questions with the same key are the same kind.
pub fn kind_key(q: &str) -> String {
    let goal = q.split("\n\nInput files (in your workspace):").next().unwrap_or(q).to_lowercase();
    let quoted = Regex::new(r#""[^"]*"|'[^']*'"#).unwrap();
    let coded = Regex::new(r"\b[a-z]*[-_]?\d[\w.,:/\-]*").unwrap();
    let s = quoted.replace_all(&goal, "§");
    let s = coded.replace_all(&s, "#");
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn schema_key(files: &[FileSig]) -> String {
    let mut v: Vec<String> = files.iter().map(|f| format!("{}|{}", f.kind, f.header)).collect();
    v.sort();
    v.join(";")
}

/// How alike a new question (and its files) is to a stored example, 0..~1.1.
pub fn similarity(question: &str, files: &[FileSig], e: &Exemplar) -> f64 {
    let (a, b) = (content_words(question), content_words(&e.question));
    let union = a.union(&b).count();
    let words = if union == 0 { 0.0 } else { a.intersection(&b).count() as f64 / union as f64 };
    let base = match (files.is_empty(), e.files.is_empty()) {
        (true, true) => words,
        (false, false) => {
            let same = files.iter().filter(|f| e.files.iter().any(|g| g.kind == f.kind && g.header == f.header && (f.kind != "other"))).count();
            0.6 * words + 0.4 * (same as f64 / files.len().max(e.files.len()) as f64)
        }
        // One has files and the other none: different kinds of question.
        _ => words - 0.3,
    };
    // Rehearsal value nudges the ranking; an example the project dreamed up ranks just below
    // one learned from a real goal.
    base + 0.1 * e.value() - if e.origin == "dream" { 0.03 } else { 0.0 }
}

#[derive(Debug, Clone)]
pub struct Store {
    pub path: PathBuf,
}

impl Store {
    pub fn open(project_root: &Path) -> Store {
        Store { path: project_root.join("experience.json") }
    }

    pub fn load(&self) -> Vec<Exemplar> {
        std::fs::read_to_string(&self.path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    }

    /// Run `f` on the examples under an exclusive lock, then save them.
    pub fn update<R>(&self, f: impl FnOnce(&mut Vec<Exemplar>) -> Result<R>) -> Result<R> {
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(self.path.with_extension("lock"))?;
        lock.lock_exclusive()?;
        let mut all = self.load();
        let r = f(&mut all);
        if r.is_ok() {
            let tmp = self.path.with_extension("json.tmp");
            std::fs::write(&tmp, serde_json::to_string_pretty(&all)?)?;
            std::fs::rename(&tmp, &self.path)?;
        }
        let _ = FileExt::unlock(&lock);
        r
    }

    /// Keep a verified solution. A question of the same kind on the same file shapes replaces
    /// the older example (its rehearsal record carries over); returns the stored example and
    /// whether it replaced one.
    pub fn record(&self, mut ex: Exemplar) -> Result<(Exemplar, bool)> {
        use sha2::Digest as _;
        ex.program = crate::util::take_chars(ex.program.trim(), 4000);
        ex.output = crate::util::take_chars(ex.output.trim(), 600);
        ex.question = crate::util::take_chars(ex.question.split("\n\nInput files (in your workspace):").next().unwrap_or(&ex.question).trim(), 1500);
        if ex.at.is_empty() {
            ex.at = crate::util::now_iso();
        }
        let key = format!("{}\n{}", kind_key(&ex.question), schema_key(&ex.files));
        ex.id = format!("x-{}", hex::encode(&sha2::Sha256::digest(format!("{key}\n{}", ex.at).as_bytes())[..4]));
        self.update(|all| {
            let dup = all.iter().position(|e| e.status == "active" && format!("{}\n{}", kind_key(&e.question), schema_key(&e.files)) == key);
            let replaced = dup.is_some();
            if let Some(i) = dup {
                let old = all.remove(i);
                ex.id = old.id;
                ex.pinned = old.pinned;
                ex.uses = old.uses;
                (ex.dreams, ex.cold_ok, ex.helped, ex.failed, ex.last_dream) = (old.dreams, old.cold_ok, old.helped, old.failed, old.last_dream);
            }
            all.push(ex.clone());
            // Over the cap: the least valuable unpinned examples go first.
            let mut active: Vec<usize> = all.iter().enumerate().filter(|(_, e)| e.status == "active" && !e.pinned).map(|(i, _)| i).collect();
            if all.iter().filter(|e| e.status == "active").count() > MAX_ACTIVE {
                active.sort_by(|a, b| all[*a].value().partial_cmp(&all[*b].value()).unwrap().then(all[*a].at.cmp(&all[*b].at)));
                let over = all.iter().filter(|e| e.status == "active").count() - MAX_ACTIVE;
                for i in active.into_iter().take(over) {
                    all[i].status = "archived".into();
                    all[i].note = "made room for newer examples".into();
                }
            }
            Ok((ex, replaced))
        })
    }

    /// The examples most like this question, best first: at most `k`, each at least `min` alike.
    pub fn similar(&self, question: &str, files: &[FileSig], k: usize, min: f64) -> Vec<(f64, Exemplar)> {
        let goal = question.split("\n\nInput files (in your workspace):").next().unwrap_or(question);
        let mut v: Vec<(f64, Exemplar)> = self.load().into_iter().filter(|e| e.status == "active" && !e.program.trim().is_empty()).map(|e| (similarity(goal, files, &e), e)).filter(|(s, _)| *s >= min).collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        v.truncate(k);
        v
    }

    pub fn mark_used(&self, ids: &[String]) {
        if ids.is_empty() {
            return;
        }
        let _ = self.update(|all| {
            for e in all.iter_mut().filter(|e| ids.contains(&e.id)) {
                e.uses += 1;
            }
            Ok(())
        });
    }
}

/// Worked examples as the engine's prompt shows them.
pub fn examples_block(examples: &[(f64, Exemplar)]) -> String {
    if examples.is_empty() {
        return String::new();
    }
    let mut s = String::from("\n\nSolved earlier in this project. Reuse how it reads the files and its approach; compute what THIS question asks, with this question's values:");
    for (_, e) in examples {
        let files = if e.files.is_empty() { String::new() } else { format!("\nFiles: {}", e.files.iter().map(|f| if f.header.is_empty() { f.name.clone() } else { format!("{} ({})", f.name, crate::util::take_chars(&f.header, 120)) }).collect::<Vec<_>>().join("; ")) };
        s.push_str(&format!(
            "\n\nQuestion: {}{files}\nProgram:\n```python\n{}\n```\nIt printed: {}",
            crate::util::take_chars(e.question.trim(), 400),
            crate::util::take_chars(e.program.trim(), 1400),
            crate::util::take_chars(e.output.trim().lines().last().unwrap_or(""), 160)
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(q: &str, files: Vec<FileSig>) -> Exemplar {
        Exemplar { question: q.into(), files, program: "print(1)".into(), answer: "1".into(), status: "active".into(), ..Default::default() }
    }
    fn csv(header: &str) -> FileSig {
        FileSig { name: "sales.csv".into(), kind: "csv".into(), header: header.into(), lines: 10 }
    }

    #[test]
    fn the_same_kind_of_question_is_recognised() {
        let h = "region,product,units,unit_price";
        let e = ex("Using sales.csv, what is the total revenue (units x unit_price) across all rows?", vec![csv(h)]);
        // same question, new data
        assert!(similarity("Using sales.csv, what is the total revenue (units x unit_price) across all rows?", &[csv(h)], &e) > 0.95);
        // a different question about the same file is related, but less
        let other = similarity("Using sales.csv, how many units of Gadget were sold in total across all regions?", &[csv(h)], &e);
        assert!(other > 0.4 && other < 0.8, "{other}");
        // an unrelated question is not
        assert!(similarity("How many days are there from 2024-02-17 to 2024-11-05?", &[], &e) < 0.2);
        assert!(similarity("What is the total revenue?", &[], &e) < 0.5, "files on one side only");
        assert_eq!(kind_key("How many days are there from 2024-02-17 to 2024-11-05?"), kind_key("How many days are there from 2023-01-09 to 2023-08-30?"));
        assert_ne!(kind_key("How many days are there from 2024-02-17 to 2024-11-05?"), kind_key("What is the date 90 days after 2024-02-17?"));
        assert_eq!(kind_key("How many times does the letter 'r' appear in \"a b\"?"), kind_key("How many times does the letter 'e' appear in \"c d e\"?"));
    }

    #[test]
    fn records_replace_the_same_kind_and_keep_their_rehearsal_record() {
        let d = tempfile::tempdir().unwrap();
        let s = Store::open(d.path());
        let (a, dup) = s.record(ex("How many days are there from 2024-02-17 to 2024-11-05?", vec![])).unwrap();
        assert!(!dup);
        s.update(|all| {
            all[0].dreams = 2;
            all[0].helped = 2;
            Ok(())
        })
        .unwrap();
        let (b, dup) = s.record(ex("How many days are there from 2023-01-09 to 2023-08-30?", vec![])).unwrap();
        assert!(dup, "same kind of question");
        assert_eq!(a.id, b.id);
        let all = s.load();
        assert_eq!(all.len(), 1);
        assert_eq!((all[0].dreams, all[0].helped), (2, 2));
        assert!((all[0].value() - 1.0).abs() < 1e-9);
        s.record(ex("What is the median of these numbers: 1, 2, 3?", vec![])).unwrap();
        assert_eq!(s.load().len(), 2);
        let hits = s.similar("How many days are there from 2025-03-01 to 2025-04-01?", &[], 2, 0.5);
        assert_eq!(hits.len(), 1);
        assert!(examples_block(&hits).contains("```python"));
        s.mark_used(&[hits[0].1.id.clone()]);
        assert_eq!(s.load().iter().find(|e| e.id == hits[0].1.id).unwrap().uses, 1);
    }

    #[test]
    fn file_shapes() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.csv"), "x,y\n1,2\n").unwrap();
        std::fs::write(d.path().join("c.json"), "{\"b\": 1, \"a\": [1]}").unwrap();
        std::fs::write(d.path().join("app.log"), "one\ntwo\n").unwrap();
        assert_eq!(sig_of(d.path(), "a.csv"), FileSig { name: "a.csv".into(), kind: "csv".into(), header: "x,y".into(), lines: 2 });
        assert_eq!(sig_of(d.path(), "c.json").header, "a,b");
        assert_eq!(sig_of(d.path(), "app.log").kind, "log");
    }
}
