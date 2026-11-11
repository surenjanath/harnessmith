//! The answer engine: exact answers out of small models.
//!
//! A small model asked a factual or numeric question directly will usually guess. Asked to
//! *write a short program* that computes the answer, the same model is right far more often,
//! because the arithmetic, the counting and the file reading are done by Python, not by the
//! model. This module drives that as a fixed sequence the harness controls, one narrow request
//! at a time (small models follow one instruction well and a protocol badly):
//!
//! ```text
//!   1. write a program        -> run it (notebook-style, sandboxed)
//!   2. it failed / printed nothing -> show the short error, ask for the fixed program
//!   3. read the answer off the program's output (never from the model's memory of it)
//!   4. repeat 1-3 independently and take the answer most attempts agree on
//!   5. still no agreement -> one step-by-step attempt as a tie-break, then a stronger
//!      model if the project names one
//! ```
//!
//! The agreement count is reported as confidence, so a caller can tell a checked answer from a
//! lone guess.

use super::client::{self, ChatReq, Msg};
use super::sandbox::Sandbox;
use super::{agent, connections, health, pycode};
use crate::util;
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Opts {
    /// `<connection>/<model>`, or a failover chain `a/x, b/y`.
    pub model_id: String,
    /// Independent attempts to vote over (1 turns voting off).
    pub samples: usize,
    /// Fix-it rounds per attempt after a failing program.
    pub repairs: usize,
    /// Guidance from the harness (guardrails, lessons) added to every attempt.
    pub notes: String,
    pub sandbox: Option<Sandbox>,
    pub wall: Duration,
    pub trajectory: Option<PathBuf>,
    /// A stronger model asked once when the attempts do not agree (empty: none).
    pub escalate_model: String,
    pub max_tokens: u32,
    pub py_timeout: Duration,
    pub log_prefix: String,
    /// The operator's guardrail gate, applied to every program before it runs.
    pub gate: Option<super::gate::Gate>,
    /// Worked examples from the project's experience (similarity, example), best first.
    pub examples: Vec<(f64, super::experience::Exemplar)>,
}

impl Opts {
    pub fn new(model_id: &str) -> Opts {
        Opts {
            model_id: model_id.into(),
            samples: 3,
            repairs: 1,
            notes: String::new(),
            sandbox: None,
            wall: Duration::from_secs(900),
            trajectory: None,
            escalate_model: String::new(),
            max_tokens: 900,
            py_timeout: Duration::from_secs(30),
            log_prefix: "      ".into(),
            gate: None,
            examples: vec![],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Sample {
    pub model: String,
    pub answer: Option<String>,
    pub program: String,
    pub output: String,
    pub error: String,
    /// "program" (computed), "direct" (the model answered without a program), "reasoned"
    /// (step-by-step tie-break), "escalated" (the stronger model).
    pub how: &'static str,
    pub repairs: u32,
    pub seconds: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Verdict {
    pub answer: Option<String>,
    /// Attempts that gave the winning answer / attempts that gave any answer / attempts made.
    pub agree: usize,
    pub answered: usize,
    pub tried: usize,
    pub program: String,
    pub output: String,
    pub samples: Vec<Sample>,
    pub escalated: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub calls: u32,
    pub served_by: String,
    pub error: String,
    /// How the winning answer was reached (a `Sample::how`).
    pub how: String,
}

impl Verdict {
    /// "high": at least two attempts agree and they are the majority. "medium": one computed
    /// answer with nothing against it, or a plurality. "low": attempts disagree. "none".
    pub fn confidence(&self) -> &'static str {
        if self.answer.is_none() {
            "none"
        } else if self.escalated || (self.agree >= 2 && self.agree * 2 > self.answered) {
            "high"
        } else if self.how == "reasoned" || self.how == "direct" {
            // Nothing was computed: this is the model's own estimate.
            "low"
        } else if self.answered == 1 || self.agree >= 2 {
            "medium"
        } else {
            "low"
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "answer": self.answer, "agree": self.agree, "answered": self.answered, "tried": self.tried, "confidence": self.confidence(), "how": self.how,
            "program": self.program, "output": util::take_chars(&self.output, 4000), "escalated": self.escalated,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens, "cost_usd": self.cost_usd, "calls": self.calls,
            "model": self.served_by, "error": self.error,
            "samples": self.samples.iter().map(|s| json!({
                "model": s.model, "answer": s.answer, "how": s.how, "repairs": s.repairs, "seconds": (s.seconds * 10.0).round() / 10.0,
                "program": s.program, "output": util::take_chars(&s.output, 1500), "error": util::take_chars(&s.error, 600),
            })).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Which goals the engine takes
// ---------------------------------------------------------------------------------------------

/// Is the accuracy boost on for this solver under the project's setting? "on" / "off" are
/// explicit; "auto" turns it on for local and small models, which are the ones that guess.
pub fn enabled(setting: &str, model_id: &str) -> bool {
    match setting.trim().to_lowercase().as_str() {
        "off" | "false" | "0" | "no" => false,
        "on" | "true" | "1" | "yes" | "always" => true,
        _ => {
            let first = model_id.split(',').next().unwrap_or("").trim();
            match connections::resolve(first) {
                Ok((c, m)) => c.provider != "mock" && connections::is_small(&c, &m),
                Err(_) => false,
            }
        }
    }
}

/// Does this goal ask for one exact answer (a number, a name, a date, a count) rather than for
/// something to be written or built? Only those go to the engine; everything else runs on the
/// agent loop.
pub fn is_exact_question(task: &str) -> bool {
    // Only the goal itself counts, not the file list the harness appends.
    let goal = task.split("\n\nInput files (in your workspace):").next().unwrap_or(task);
    if util::char_len(goal) > 1500 {
        return false;
    }
    let g = goal.to_lowercase();
    // A named output type other than a plain answer means an artifact is wanted.
    if !crate::core::schedule::expected_exts(task).is_empty() {
        return false;
    }
    let makes = Regex::new(r"\b(write|create|make|build|generate|draft|design|develop|implement|produce|compose|prepare|give me|i want|i need)\b[^.?!\n]{0,60}\b(report|essay|article|document|page|website|site|html|app|application|script|program|story|poem|presentation|slides|deck|email|letter|plan|guide|tutorial|blog|post|readme|code|function|class|chart|plot|diagram|dashboard|spreadsheet|proposal|outline|brief|memo|newsletter|landing)\b").unwrap();
    let open_ended = Regex::new(r"\b(explain|describe|summari[sz]e|summary|discuss|elaborate|compare|contrast|analy[sz]e|review|critique|outline|brainstorm|research|recommend|suggest|advise|translate|rewrite|rephrase|proofread|refactor|why\b|pros and cons|in detail|step[- ]by[- ]step guide)").unwrap();
    if makes.is_match(&g) || open_ended.is_match(&g) {
        return false;
    }
    let asks = Regex::new(r"\b(how (many|much|long|far|old|often|tall|big|large|fast|heavy|high|deep|wide)|what(['’]s| is| was| are| were| will| would| does| do| did| percentage| percent| fraction| number| date| day| time| year| month)|which\b[^.?!\n]{0,80}\b(is|has|have|had|was|were|are|did|does)\b|who\b|whose\b|when (is|was|did|does|will)|where (is|was|are|did|does)|calculate|compute|count\b|find the|find out|determine|evaluate|solve|convert|add up|sum (of|up)|the total|the average|the mean|the median|the difference|the number of|is it true|true or false|yes or no)").unwrap();
    asks.is_match(&g)
}

// ---------------------------------------------------------------------------------------------
// Reading answers: numbers, names, agreement
// ---------------------------------------------------------------------------------------------

/// A number as it was written, with how many decimals it showed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Num {
    pub v: f64,
    pub decimals: u32,
}

/// Same value, allowing for the coarser of the two having been rounded.
pub fn same_num(a: &Num, b: &Num) -> bool {
    let d = a.decimals.min(b.decimals).min(9);
    let m = 10f64.powi(d as i32);
    let (ra, rb) = ((a.v * m).round(), (b.v * m).round());
    if (ra - rb).abs() < 0.5 {
        return true;
    }
    // Float noise (0.30000000000000004) and half-way cases rounded the other way.
    (a.v - b.v).abs() <= 0.5000001 / m && a.decimals != b.decimals || (a.v - b.v).abs() <= 1e-9 * a.v.abs().max(1.0)
}

/// Numbers and other content tokens of a text. Dates, times and codes such as `REQ-7731` stay
/// whole (as tokens), so `2024-11-05` is not read as three numbers.
pub fn atoms(s: &str) -> (Vec<Num>, Vec<String>) {
    let plain = Regex::new(r"^-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?$").unwrap();
    let sci = Regex::new(r"^-?\d+(\.\d+)?[eE][-+]?\d+$").unwrap();
    let unit = Regex::new(r"^(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)([A-Za-z°/%µ][A-Za-z°/%µ0-9²³]*)$").unwrap();
    let (mut nums, mut words) = (vec![], vec![]);
    let push_num = |nums: &mut Vec<Num>, t: &str| {
        let clean = t.replace(',', "");
        if let Ok(v) = clean.parse::<f64>() {
            let decimals = clean.split_once('.').map(|(_, d)| d.trim_end_matches(|c: char| !c.is_ascii_digit()).len() as u32).unwrap_or(0);
            nums.push(Num { v, decimals });
        }
    };
    for raw in s.split(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\'' | '`' | ';' | '!' | '?' | '*' | '|' | '<' | '>' | '=' | '~' | '≈')) {
        let t = raw.trim_start_matches(['$', '€', '£', '¥', '+', '#']).trim_end_matches(['.', ',', ':', '%', '$', '€', '£']);
        if t.is_empty() {
            continue;
        }
        if plain.is_match(t) {
            push_num(&mut nums, t);
        } else if sci.is_match(t) {
            if let Ok(v) = t.parse::<f64>() {
                nums.push(Num { v, decimals: 6 });
            }
        } else if let Some(c) = unit.captures(t) {
            push_num(&mut nums, &c[1]);
            words.push(c[2].to_lowercase());
        } else if t.chars().any(|c| c.is_alphanumeric()) {
            // A word or a code. "key:value" / "a/b" pieces are split once more.
            for piece in t.split([':', '/', ',']) {
                let piece = piece.trim_matches(|c: char| !c.is_alphanumeric());
                if piece.is_empty() {
                    continue;
                }
                if plain.is_match(piece) && !t.contains('-') && !t.contains(':') {
                    push_num(&mut nums, piece);
                } else {
                    words.push(piece.to_lowercase());
                }
            }
        }
    }
    (nums, words)
}

const STOP: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at", "to", "and", "or", "for", "with", "by", "it", "its", "this", "that", "as", "be", "has", "have", "had", "from", "total", "answer", "final",
    "result", "value", "approximately", "about", "equals", "equal", "there", "will", "would", "than", "so", "then", "which", "what", "who", "when", "where", "how", "many", "much", "we", "you", "i", "they", "he",
    "she", "their", "our", "your", "not", "no", "yes", "do", "does", "did", "can", "all", "each", "per", "only", "also", "therefore", "thus", "hence", "after", "before", "now", "just", "exactly", "roughly", "around",
];

/// The words of an answer that carry it: not filler, and not merely echoed from the question.
fn key_words(answer_words: &[String], question_words: &BTreeSet<String>) -> BTreeSet<String> {
    answer_words.iter().filter(|w| !STOP.contains(&w.as_str()) && !question_words.contains(*w)).cloned().collect()
}

/// Do two answers say the same thing? Numbers must match (rounding allowed); names must
/// overlap when both answers carry any.
pub fn same_answer(a: &str, b: &str, question: &str) -> bool {
    let qw: BTreeSet<String> = atoms(question).1.into_iter().collect();
    let ((na, wa), (nb, wb)) = (atoms(a), atoms(b));
    let (ka, kb) = (key_words(&wa, &qw), key_words(&wb, &qw));
    if !na.is_empty() || !nb.is_empty() {
        if na.len() != nb.len() {
            return false;
        }
        let mut rest: Vec<Num> = nb.clone();
        for x in &na {
            match rest.iter().position(|y| same_num(x, y)) {
                Some(i) => {
                    rest.remove(i);
                }
                None => return false,
            }
        }
        if ka.is_empty() || kb.is_empty() {
            return true;
        }
    } else if ka.is_empty() && kb.is_empty() {
        // Nothing but filler or question words: compare what is there.
        let norm = |w: &[String]| w.iter().filter(|x| !STOP.contains(&x.as_str())).cloned().collect::<Vec<_>>();
        return norm(&wa) == norm(&wb);
    } else if ka.is_empty() || kb.is_empty() {
        return false;
    }
    let inter = ka.intersection(&kb).count();
    let union = ka.union(&kb).count();
    inter == ka.len().min(kb.len()) || inter * 2 >= union
}

/// The value after an `ANSWER:` label (the last one in the text), without markdown dressing.
pub fn labelled_answer(text: &str) -> Option<String> {
    let re = Regex::new(r"(?im)^[\s>*_#`-]*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$").unwrap();
    let a = re.captures_iter(text).last().map(|c| c[1].to_string())?;
    let mut a = a.trim_matches(|c: char| c.is_whitespace() || matches!(c, '*' | '`' | '_')).to_string();
    // "ANSWER: ANSWER: 42": the label repeated.
    let again = Regex::new(r"(?i)^(?:final\s+)?answer\s*[:=]\s*").unwrap();
    while let Some(m) = again.find(&a) {
        a = a[m.end()..].trim_matches(|c: char| c.is_whitespace() || matches!(c, '*' | '`' | '_')).to_string();
    }
    // A placeholder copied from the instruction is not an answer.
    if a.is_empty() || a.starts_with('<') && a.ends_with('>') {
        return None;
    }
    Some(a)
}

/// A program that prints "not found", "None" or an error has not answered. Counting that as an
/// answer would let two failed lookups "agree".
pub fn is_non_answer(a: &str) -> bool {
    let low = a.trim().to_lowercase();
    let bare = low.trim_matches(|c: char| !c.is_alphanumeric());
    if bare.is_empty() || matches!(bare, "none" | "null" | "nan" | "n/a" | "na" | "unknown" | "undefined" | "error") {
        return true;
    }
    Regex::new(r"\b(not found|no match(es)?( found)?|not available|does not exist|doesn't exist|no such|no data|not present|cannot be determined|can't be determined|could not|couldn't|unable to|no results?|no answer|not in the (file|log|data|list)|traceback|exception)\b").unwrap().is_match(&low)
}

/// Is every figure (or, without figures, some name) of `answer` present in `evidence`?
pub fn grounded_in(answer: &str, evidence: &str) -> bool {
    let (na, wa) = atoms(answer);
    let (ne, we) = atoms(evidence);
    if !na.is_empty() {
        return na.iter().all(|x| ne.iter().any(|y| same_num(x, y)));
    }
    let we: BTreeSet<String> = we.into_iter().collect();
    let keys: Vec<&String> = wa.iter().filter(|w| !STOP.contains(&w.as_str())).collect();
    keys.is_empty() || keys.iter().any(|w| we.contains(*w))
}

// ---------------------------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------------------------

const SYSTEM: &str = "You answer questions by writing a short Python program. The program is run for you, and what it prints is the answer.
Rules:
- Never calculate, count or look things up in your head: the program must compute every value.
- Put the numbers and text from the question directly into the program as variables. Never call input().
- Use only the Python standard library (csv, json, math, datetime, statistics, re, collections).
- Keep it simple: several short lines with plain for-loops and if-statements, building up a total or a dict step by step. Do not squeeze the logic into one long line.
- Mind the units: first convert every quantity to the units the answer needs (for example hours and minutes to seconds, km to metres), one conversion per line.
- The last line must print() only the final answer: the value or name asked for, in the form the question asks for (rounding, units).";

/// How to load each input file, written for the files actually there: the exact line to copy,
/// with the real file name, its columns or keys, and what one line of it looks like.
pub fn cookbook(workspace: &Path, files: &[String]) -> String {
    let mut lines = vec![];
    for rel in files.iter().take(6) {
        let sig = super::experience::sig_of(workspace, rel);
        let text = std::fs::read(workspace.join(rel)).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
        match sig.kind.as_str() {
            "csv" | "tsv" => {
                let delim = if sig.kind == "tsv" { ", delimiter=\"\\t\"" } else { "" };
                let sep = if sig.kind == "tsv" { '\t' } else { ',' };
                // Which columns hold numbers and which hold text, from the first data row.
                let cols: Vec<&str> = sig.header.split(sep).map(|c| c.trim()).collect();
                let row: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).nth(1).map(|l| l.split(sep).map(|c| c.trim()).collect()).unwrap_or_default();
                let is_num = |i: usize| row.get(i).map(|v| v.replace(',', "").parse::<f64>().is_ok()).unwrap_or(false);
                let nums: Vec<&str> = cols.iter().enumerate().filter(|(i, _)| is_num(*i)).map(|(_, c)| *c).collect();
                let texts: Vec<&str> = cols.iter().enumerate().filter(|(i, _)| !is_num(*i)).map(|(_, c)| *c).collect();
                let mut note = format!("columns: {}", util::take_chars(&cols.join(", "), 200));
                if let Some(n) = nums.first() {
                    note.push_str(&format!(". Every value is read as a string. Number columns ({}): convert with float(row[\"{n}\"])", nums.join(", ")));
                }
                if !texts.is_empty() {
                    note.push_str(&format!(". Text columns ({}): use as they are, never convert them", texts.join(", ")));
                }
                lines.push(format!("  rows = list(csv.DictReader(open(\"{rel}\"){delim}))   # {note}"));
            }
            "json" => {
                let mut note = if sig.header.is_empty() { String::new() } else { format!("   # top-level keys: {}", sig.header) };
                if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&text) {
                    if let Some((k, Value::Array(a))) = m.iter().find(|(_, v)| v.as_array().map(|a| a.first().map(|x| x.is_object()).unwrap_or(false)).unwrap_or(false)) {
                        if let Some(Value::Object(o)) = a.first() {
                            note.push_str(&format!("; data[\"{k}\"] is a list of dicts with keys: {}", o.keys().take(10).cloned().collect::<Vec<_>>().join(", ")));
                        }
                    }
                }
                lines.push(format!("  data = json.load(open(\"{rel}\")){note}"));
            }
            "log" | "text" => {
                let sample = util::take_chars(text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim(), 160);
                let mut l = format!("  lines = open(\"{rel}\").read().splitlines()   # plain text, one record per line, e.g. {sample:?}");
                // key=value fields: show how to pick a line and read one field from it.
                if let Some(c) = Regex::new(r"\b([A-Za-z_][\w]*)=(\d+)\b").unwrap().captures_iter(&sample).last() {
                    l.push_str(&format!("\n  # pick lines with: if \"some text\" in line      read a field with: int(re.search(r\"{}=(\\d+)\", line).group(1))", &c[1]));
                }
                lines.push(l);
            }
            _ => {}
        }
    }
    // Two tables that share a column are meant to be combined through it.
    let tables: Vec<(String, Vec<String>)> = files
        .iter()
        .take(6)
        .map(|rel| (rel.clone(), super::experience::sig_of(workspace, rel)))
        .filter(|(_, s)| s.kind == "csv" || s.kind == "tsv")
        .map(|(rel, s)| (rel, s.header.split(if s.kind == "tsv" { '\t' } else { ',' }).map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect()))
        .collect();
    for i in 0..tables.len() {
        for j in i + 1..tables.len() {
            if let Some(col) = tables[i].1.iter().find(|c| tables[j].1.contains(c)) {
                lines.push(format!(
                    "  # {} and {} share the column \"{col}\". To combine them: first build a lookup from one file, e.g. info = {{}} then for row in <its rows>: info[row[\"{col}\"]] = row ; then loop over the other file's rows and use info[row[\"{col}\"]].",
                    tables[i].0, tables[j].0
                ));
            }
        }
    }
    if lines.is_empty() {
        return "\n- Read the files by their exact names from the current directory.".into();
    }
    format!("\nHow to load the files (they are in the current directory). Nothing is loaded for you: copy the line you need into your program.\n{}", lines.join("\n"))
}

const NO_FILES: &str = "\n- There are no files: do not open any file.";

const ASK: &str = "Write the Python program that computes the answer. Reply with one ```python block and nothing else.";

/// The input files as the model sees them: small text files whole, large ones by their first
/// lines, binaries by name.
pub fn files_block(workspace: &Path, files: &[String]) -> String {
    if files.is_empty() {
        return String::new();
    }
    let per_file = if files.len() <= 2 { 2600 } else { 1400 };
    let mut parts = vec![];
    for rel in files.iter().take(12) {
        let p = workspace.join(rel);
        let Ok(bytes) = std::fs::read(&p) else { continue };
        let sniff = &bytes[..bytes.len().min(4096)];
        if sniff.contains(&0) {
            parts.push(format!("--- {rel} (binary file, {} bytes) ---", bytes.len()));
            continue;
        }
        let text = String::from_utf8_lossy(&bytes);
        let n_lines = text.lines().count();
        if util::char_len(&text) <= per_file {
            parts.push(format!("--- {rel} ({n_lines} lines, shown in full) ---\n{}", text.trim_end()));
        } else {
            let mut head: Vec<String> = vec![];
            let mut used = 0;
            for l in text.lines().take(14) {
                let l = util::take_chars(l, 300);
                used += l.len();
                head.push(l);
                if used > per_file {
                    break;
                }
            }
            parts.push(format!("--- {rel} ({n_lines} lines in total; only a sample of the first {} is shown here, so the program must open the file and go through ALL of its lines) ---\n{}\n...", head.len(), head.join("\n")));
        }
    }
    if files.len() > 12 {
        parts.push(format!("(+{} more files)", files.len() - 12));
    }
    format!("\n\nFiles in the current directory:\n{}", parts.join("\n"))
}

/// Harness guidance worth showing the engine: lessons and rules, minus the lines that are about
/// the agent loop's mechanics (deliverable folders, finishing).
pub fn notes_from_harness(harness: &Path) -> String {
    let mut keep = vec![];
    for f in ["guardrails.md", "LongTermMEMORY.md"] {
        let Ok(body) = util::read_lossy(&harness.join(f)) else { continue };
        for l in body.lines() {
            let low = l.to_lowercase();
            if l.trim().is_empty() || low.contains("output/") || low.contains("deliverable") || low.contains("finish") || low.contains("skill") || low.contains("sub-agent") {
                continue;
            }
            keep.push(l.trim_end().to_string());
        }
    }
    util::take_chars(&keep.join("\n"), 1800)
}

// ---------------------------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------------------------

struct Engine<'a> {
    o: &'a Opts,
    question: &'a str,
    workspace: &'a Path,
    chain: Vec<(connections::Connection, String, String)>,
    system: String,
    user: String,
    traj: Option<std::fs::File>,
    t0: Instant,
    v: Verdict,
    /// What made earlier attempts fail, told to later ones so they take another route.
    avoid: Vec<String>,
}

impl<'a> Engine<'a> {
    fn ev(&mut self, v: Value) {
        if let Some(f) = &mut self.traj {
            let _ = writeln!(f, "{v}");
        }
    }

    fn time_left(&self) -> Duration {
        self.o.wall.checked_sub(self.t0.elapsed()).unwrap_or_default()
    }

    /// One model call on the first usable model of `chain` (failing over down the chain).
    fn call_on(&mut self, chain: &[(connections::Connection, String, String)], system: &str, msgs: &[Msg], max_tokens: u32, temperature: Option<f64>) -> Result<String> {
        let mut last = String::from("no usable model");
        for (conn, model, id) in chain {
            if crate::log::cancelled() {
                bail!(client::CANCELLED);
            }
            if health::over_cap(conn) || !health::reopens_in(&conn.name).is_zero() {
                last = format!("{id} is unavailable (circuit open or over its spending cap)");
                continue;
            }
            if let Some(b) = crate::log::budget().filter(|b| b.exceeded()) {
                bail!("the run's spending cap of ${:.2} was reached", b.cap_usd);
            }
            let left = self.time_left();
            if left.is_zero() {
                bail!("time budget of {}s exhausted", self.o.wall.as_secs());
            }
            self.v.calls += 1;
            crate::log::event("step", "call", json!({"model": id, "step": self.v.calls, "depth": 0}));
            let max_tokens = conn.max_tokens.unwrap_or(max_tokens).min(max_tokens).min(conn.window() / 2).max(64);
            let t = temperature.or(conn.temperature);
            match client::chat(conn, model, &ChatReq { system, messages: msgs, tools: &[], max_tokens, temperature: t }, left.max(Duration::from_secs(20))) {
                Ok(r) => {
                    self.v.input_tokens += r.input_tokens;
                    self.v.output_tokens += r.output_tokens;
                    let usd = conn.cost(&r);
                    self.v.cost_usd += usd;
                    health::record_cost(&conn.name, usd);
                    if let Some(b) = crate::log::budget() {
                        b.add(usd);
                    }
                    self.v.served_by = id.clone();
                    self.ev(json!({"type": "assistant", "text": r.text}));
                    self.ev(json!({"type": "usage", "input_tokens": r.input_tokens, "output_tokens": r.output_tokens}));
                    crate::log::event("step", "turn", json!({"step": self.v.calls, "depth": 0, "in": self.v.input_tokens, "out": self.v.output_tokens, "model": id, "tools": Vec::<String>::new(), "say": util::take_chars(r.text.trim(), 160)}));
                    return Ok(r.text);
                }
                Err(e) => {
                    last = format!("{id}: {e}");
                    if last.contains(client::CANCELLED) {
                        bail!(client::CANCELLED);
                    }
                    self.ev(json!({"type": "error", "message": last}));
                }
            }
        }
        bail!(last)
    }

    fn call(&mut self, system: &str, msgs: &[Msg], max_tokens: u32, temperature: Option<f64>) -> Result<String> {
        let chain = self.chain.clone();
        self.call_on(&chain, system, msgs, max_tokens, temperature)
    }

    /// Read the answer off a program's output. One short line is the answer as it stands; more
    /// than that, the model reports which part answers the question, and its report must be
    /// made of values the program actually printed.
    fn read_answer(&mut self, chain: &[(connections::Connection, String, String)], output: &str) -> Option<String> {
        let lines: Vec<&str> = output.lines().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
        let last = lines.last().copied()?;
        if lines.len() == 1 && util::char_len(last) <= 200 {
            return Some(last.to_string());
        }
        let msg = format!(
            "Question: {}\n\nA program was run to answer it and printed:\n{}\n\nReply with one line and nothing else: `ANSWER: ` followed by the final answer (only the value(s) or name(s) asked for, copied exactly from the printed output).",
            self.question.trim(),
            util::take_chars(output.trim(), 1800)
        );
        let reported = self
            .call_on(chain, "You report results exactly, without changing any value.", &[Msg::User(msg)], 160, Some(0.0))
            .ok()
            .and_then(|r| labelled_answer(&r).or_else(|| r.lines().map(|l| l.trim()).find(|l| !l.is_empty()).map(String::from)))
            .filter(|a| util::char_len(a) <= 300 && grounded_in(a, output));
        reported.or_else(|| (util::char_len(last) <= 200).then(|| last.to_string()))
    }

    /// One independent attempt: program, run, repair, read the answer.
    fn attempt(&mut self, chain: &[(connections::Connection, String, String)], temperature: f64, how: &'static str) -> Sample {
        let t0 = Instant::now();
        let mut s = Sample { how, ..Default::default() };
        let mut first = self.user.clone();
        if !self.avoid.is_empty() {
            let at = first.rfind(ASK).unwrap_or(first.len());
            first.insert_str(at, &format!("Earlier attempts at this failed like this, so write the program differently:\n{}\n\n", self.avoid.iter().map(|e| format!("- {e}")).collect::<Vec<_>>().join("\n")));
        }
        let mut msgs = vec![Msg::User(first)];
        let system = self.system.clone();
        let mut asked_for_code = false;
        let mut asked_for_precision = false;
        for round in 0..=self.o.repairs as u32 {
            // A repair at near-zero temperature tends to return the same broken program.
            let temp = if round == 0 { temperature } else { temperature.max(0.6) };
            let reply = match self.call_on(chain, &system, &msgs, self.o.max_tokens, Some(temp)) {
                Ok(r) => r,
                Err(e) => {
                    s.error = format!("{e}");
                    break;
                }
            };
            s.model = self.v.served_by.clone();
            let Some(code) = pycode::extract_code(&reply) else {
                if let Some(a) = labelled_answer(&reply).filter(|a| !is_non_answer(a)) {
                    s.answer = Some(a);
                    s.how = "direct";
                    break;
                }
                if asked_for_code {
                    s.error = "the model did not write a program".into();
                    break;
                }
                asked_for_code = true;
                msgs.push(Msg::Assistant { text: reply, calls: vec![] });
                msgs.push(Msg::User("Reply with the program in one ```python block and nothing else.".into()));
                continue;
            };
            s.program = pycode::clean(&code);
            // The operator's gate reads the program first; in enforce / ask mode a hit is not run.
            let blocked = self.o.gate.as_ref().and_then(|g| g.check("python", &json!({"code": s.program}), None).map(|v| (g.mode, v)));
            let out = match blocked {
                Some((mode, v)) if mode != super::gate::Mode::Observe => {
                    crate::log::event("gate", "blocked", json!({"tool": "python", "call": util::take_chars(&s.program, 100), "rule": v.rule, "reason": v.reason, "mode": mode.as_str()}));
                    self.ev(json!({"type": "gate", "status": "blocked", "detail": {"tool": "python", "rule": v.rule, "reason": v.reason}}));
                    pycode::PyOut { error: format!("Blocked by the operator's guardrail gate (rule `{}`): {}. Compute the answer another way.", v.rule, v.reason), ..Default::default() }
                }
                other => {
                    if let Some((mode, v)) = other {
                        crate::log::event("gate", "would_block", json!({"tool": "python", "rule": v.rule, "reason": v.reason, "mode": mode.as_str()}));
                    }
                    match pycode::run(self.workspace, self.o.sandbox.as_ref(), &code, self.o.py_timeout) {
                        Ok(o) => o,
                        Err(e) => {
                            s.error = format!("could not run Python: {e}");
                            break;
                        }
                    }
                }
            };
            let shown = out.render();
            self.ev(json!({"type": "tool", "name": "python", "input": {"code": s.program}, "status": if out.ok() { "ok" } else { "error" }, "output": util::take_chars(&shown, 4000)}));
            if self.o.log_prefix.len() < 40 {
                crate::say!("{}· python → {}", self.o.log_prefix, util::take_chars(&shown.replace('\n', " ⏎ "), 110));
            }
            let n_lines = out.stdout.lines().filter(|l| !l.trim().is_empty()).count();
            let problem = if !out.ok() {
                format!("The program failed:\n{}", util::take_chars(&shown, 900))
            } else if n_lines == 0 {
                "The program printed nothing. Its last line must print() the final answer (if you wrote a function, call it and print what it returns).".to_string()
            } else if n_lines > 40 {
                format!("The program printed {n_lines} lines. Print only the final answer: one line with the value or name asked for.")
            } else {
                match self.read_answer(chain, &out.stdout) {
                    // The question asks for decimals ("to the cent") and the program printed a
                    // whole number: almost always an over-eager round(). Ask once for the
                    // precision the question wants; keep this answer in case the fix fails.
                    Some(a) if !is_non_answer(&a) && !asked_for_precision && round < self.o.repairs as u32 && wanted_decimals(self.question).map(|d| d > 0 && !atoms(&a).0.is_empty() && atoms(&a).0.iter().all(|n| n.decimals == 0)).unwrap_or(false) => {
                        asked_for_precision = true;
                        s.output = out.stdout.clone();
                        s.answer = Some(a.clone());
                        format!("The question asks for the answer to {} decimal place(s), but the program printed {a:?}, a whole number. Do not round to a whole number: print the value rounded to {} decimal place(s), e.g. print(round(value, {})).", wanted_decimals(self.question).unwrap_or(2), wanted_decimals(self.question).unwrap_or(2), wanted_decimals(self.question).unwrap_or(2))
                    }
                    Some(a) if !is_non_answer(&a) => {
                        s.output = out.stdout.clone();
                        s.answer = Some(a);
                        s.error.clear();
                        break;
                    }
                    got => format!(
                        "The program printed {:?}, which is not an answer. What the question asks for is in the data: print two raw lines (or rows) first to see their exact format, then go through EVERY line of the file, from the first one, to find it.",
                        util::take_chars(got.as_deref().unwrap_or_else(|| out.stdout.trim().lines().last().unwrap_or("")), 80)
                    ),
                }
            };
            s.error = problem.clone();
            s.repairs = round + 1;
            msgs.push(Msg::Assistant { text: reply, calls: vec![] });
            msgs.push(Msg::User(format!("{problem}\n\nFix it. Reply with the full corrected program in one ```python block and nothing else.")));
        }
        if s.answer.is_some() {
            s.error.clear();
        } else if !s.error.trim().is_empty() {
            // The gist of the failure: the exception line, or the complaint itself.
            let gist = s.error.lines().rev().find(|l| !l.trim().is_empty() && !l.starts_with("Hint:")).unwrap_or("").trim();
            let gist = util::take_chars(gist, 160);
            if !gist.is_empty() && !self.avoid.contains(&gist) && self.avoid.len() < 3 {
                self.avoid.push(gist);
            }
        }
        s.seconds = t0.elapsed().as_secs_f64();
        s
    }

    /// A step-by-step answer without a program: the tie-break, and the floor under the engine
    /// (it is never worse than asking the model directly).
    fn reasoned(&mut self) -> Sample {
        let t0 = Instant::now();
        let mut s = Sample { how: "reasoned", ..Default::default() };
        // The question and its files only: worked examples are programs, not reasoning.
        let own = self.user.rsplit("Now the question to answer:\n").next().unwrap_or(&self.user);
        let msg = format!("{}\n\nThink step by step, then give the final answer on the last line as `ANSWER: <value>`.", own.trim_end_matches(ASK).trim_end());
        match self.call("You are a careful assistant.", &[Msg::User(msg)], self.o.max_tokens, Some(0.2)) {
            Ok(r) => {
                s.model = self.v.served_by.clone();
                s.answer = labelled_answer(&r).filter(|a| !is_non_answer(a));
                if s.answer.is_none() {
                    s.error = "no ANSWER line in the reply".into();
                }
            }
            Err(e) => s.error = format!("{e}"),
        }
        s.seconds = t0.elapsed().as_secs_f64();
        s
    }
}

/// Group attempts that gave the same answer; each cluster is the indexes of its samples.
fn clusters(samples: &[Sample], question: &str) -> Vec<Vec<usize>> {
    let mut cs: Vec<Vec<usize>> = vec![];
    for (i, s) in samples.iter().enumerate() {
        let Some(a) = &s.answer else { continue };
        match cs.iter_mut().find(|c| same_answer(samples[c[0]].answer.as_deref().unwrap_or(""), a, question)) {
            Some(c) => c.push(i),
            None => cs.push(vec![i]),
        }
    }
    // Largest first; among equals, the one holding the earliest attempt.
    cs.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
    cs
}

/// The number of decimals a question asks for, when it says: "to the cent", "rounded to 2
/// decimals", "to the nearest whole number".
pub fn wanted_decimals(question: &str) -> Option<u32> {
    let q = question.to_lowercase();
    let word = |w: &str| match w {
        "one" | "1" => Some(1),
        "two" | "2" => Some(2),
        "three" | "3" => Some(3),
        "four" | "4" => Some(4),
        "zero" | "0" | "no" => Some(0),
        _ => None,
    };
    if let Some(c) = Regex::new(r"\b(\d|zero|no|one|two|three|four)\s+(?:decimal|dp\b)").unwrap().captures(&q) {
        return word(&c[1]);
    }
    if Regex::new(r"\b(to the cent|nearest cent|hundredths?)\b").unwrap().is_match(&q) {
        return Some(2);
    }
    if Regex::new(r"\b(nearest tenth|one tenth)\b").unwrap().is_match(&q) {
        return Some(1);
    }
    if Regex::new(r"\b(nearest (whole|integer|dollar|unit|euro|pound)|whole number|as an integer|no decimals?)\b").unwrap().is_match(&q) {
        return Some(0);
    }
    None
}

/// Members of a cluster can state the same value at different precision ("674.63" and "675").
/// Take the form the question asks for; when it does not say, the form most of them used, and
/// on a tie the more precise one (it can still be rounded; the coarse one cannot be undone).
fn pick_form(cluster: &[usize], samples: &[Sample], question: &str) -> Option<usize> {
    let decimals = |i: usize| samples[i].answer.as_deref().map(|a| atoms(a).0.iter().map(|n| n.decimals).max().unwrap_or(0)).unwrap_or(0);
    if let Some(want) = wanted_decimals(question) {
        if let Some(i) = cluster.iter().copied().find(|i| decimals(*i) == want) {
            return Some(i);
        }
    }
    let mut best: Option<(usize, u32, usize)> = None; // (count, decimals, first member)
    for &i in cluster {
        let d = decimals(i);
        let count = cluster.iter().filter(|j| decimals(**j) == d).count();
        if best.map(|(c, bd, _)| count > c || (count == c && d > bd)).unwrap_or(true) {
            best = Some((count, d, i));
        }
    }
    best.map(|(_, _, i)| i)
}

/// Enough agreement to stop sampling: two that agree with nothing against them, or a lead of two.
fn decided(cs: &[Vec<usize>]) -> bool {
    let top = cs.first().map(|c| c.len()).unwrap_or(0);
    let second = cs.get(1).map(|c| c.len()).unwrap_or(0);
    top >= 2 && (second == 0 || top >= second + 2)
}

/// Answer `question` about the files in `workspace` (paths relative to it).
pub fn answer(question: &str, workspace: &Path, files: &[String], o: &Opts) -> Result<Verdict> {
    if !pycode::available() {
        bail!("python3 is not installed, so programs cannot be run");
    }
    let chain = agent::resolve_chain(&o.model_id)?;
    let traj = match &o.trajectory {
        Some(p) => {
            if let Some(d) = p.parent() {
                std::fs::create_dir_all(d)?;
            }
            Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?)
        }
        None => None,
    };
    let goal = question.split("\n\nInput files (in your workspace):").next().unwrap_or(question).trim();
    let mut system = format!("{SYSTEM}{}", if files.is_empty() { NO_FILES.to_string() } else { cookbook(workspace, files) });
    if !o.notes.trim().is_empty() {
        system.push_str(&format!("\n\nNotes from earlier work on this project (use the ones that apply):\n{}", o.notes.trim()));
    }
    let shown = super::experience::examples_block(&o.examples);
    let user = if shown.is_empty() {
        format!("{goal}{}\n\n{ASK}", files_block(workspace, files))
    } else {
        format!("{}\n\n---\n\nNow the question to answer:\n{goal}{}\n\n{ASK}", shown.trim_start(), files_block(workspace, files))
    };
    let mut e = Engine { o, question: goal, workspace, chain: chain.clone(), system, user, traj, t0: Instant::now(), v: Verdict::default(), avoid: vec![] };
    e.ev(json!({"type": "session", "model": o.model_id, "depth": 0, "at": util::now_iso(), "engine": "precise"}));
    e.ev(json!({"type": "user", "text": e.user}));
    let n = o.samples.clamp(1, 9);
    let base_temp = chain[0].0.temperature.unwrap_or(0.2);
    let mut samples: Vec<Sample> = vec![];
    for i in 0..n {
        if crate::log::cancelled() || e.time_left().is_zero() {
            break;
        }
        // The first attempt is the model's best guess; later ones are sampled for diversity.
        let temp = if i == 0 { base_temp } else { base_temp.max(0.8) };
        let s = e.attempt(&chain, temp, "program");
        crate::say!("{}attempt {}: {}", o.log_prefix, i + 1, match &s.answer {
            Some(a) => format!("{} ({:.0}s{})", util::take_chars(a, 80), s.seconds, if s.repairs > 0 { ", after a fix" } else { "" }),
            None => format!("no answer — {}", util::take_chars(&s.error.replace('\n', " "), 100)),
        });
        samples.push(s);
        if decided(&clusters(&samples, goal)) {
            break;
        }
    }
    e.v.tried = samples.len();
    let mut cs = clusters(&samples, goal);
    let top = cs.first().map(|c| c.len()).unwrap_or(0);
    // No two attempts agree (or none answered): a step-by-step answer as the tie-break.
    if (top < 2 && n > 1 || top == 0) && !crate::log::cancelled() && !e.time_left().is_zero() {
        let s = e.reasoned();
        crate::say!("{}tie-break (step by step): {}", o.log_prefix, s.answer.as_deref().map(|a| util::take_chars(a, 80)).unwrap_or_else(|| "no answer".into()));
        samples.push(s);
        cs = clusters(&samples, goal);
    }
    // Still split: ask the stronger model once, and let it settle the matter.
    let top = cs.first().map(|c| c.len()).unwrap_or(0);
    let mut winner: Option<usize> = cs.first().map(|c| c[0]);
    if (top < 2 || cs.get(1).map(|c| c.len() == top).unwrap_or(false)) && !o.escalate_model.trim().is_empty() && !crate::log::cancelled() && !e.time_left().is_zero() {
        if let Ok(strong) = agent::resolve_chain(&o.escalate_model) {
            crate::log::event("heal", "escalate", json!({"to": o.escalate_model, "why": "the attempts did not agree"}));
            crate::say!("{}! attempts did not agree; asking {}", o.log_prefix, o.escalate_model);
            let mut s = e.attempt(&strong, 0.2, "escalated");
            if s.answer.is_none() {
                s.error = format!("escalation gave no answer: {}", s.error);
            }
            let got = s.answer.is_some();
            samples.push(s);
            if got {
                e.v.escalated = true;
                cs = clusters(&samples, goal);
                let last = samples.len() - 1;
                // Its cluster wins whatever its size.
                winner = cs.iter().find(|c| c.contains(&last)).map(|c| c[0]).or(Some(last));
            }
        }
    }
    if !e.v.escalated {
        winner = cs.first().map(|c| c[0]);
    }
    let mut v = std::mem::take(&mut e.v);
    v.answered = samples.iter().filter(|s| s.answer.is_some()).count();
    if let Some(w) = winner {
        let cluster = cs.iter().find(|c| c.contains(&w)).cloned().unwrap_or_else(|| vec![w]);
        // Show the computed member of the winning cluster when there is one.
        let rep = cluster.iter().copied().find(|i| !samples[*i].program.is_empty() && samples[*i].answer.is_some()).unwrap_or(w);
        v.answer = samples[pick_form(&cluster, &samples, goal).unwrap_or(w)].answer.clone();
        v.agree = cluster.len();
        v.how = samples[rep].how.to_string();
        v.program = samples[rep].program.clone();
        v.output = samples[rep].output.clone();
    } else {
        v.error = samples.iter().rev().find(|s| !s.error.is_empty()).map(|s| s.error.clone()).unwrap_or_else(|| "no attempt produced an answer".into());
    }
    v.samples = samples;
    e.ev(json!({"type": "vote", "answer": v.answer, "agree": v.agree, "answered": v.answered, "tried": v.tried, "confidence": v.confidence(), "escalated": v.escalated}));
    crate::log::event("accuracy", "vote", json!({"answer": v.answer.as_deref().map(|a| util::take_chars(a, 160)), "agree": v.agree, "answered": v.answered, "tried": v.tried, "confidence": v.confidence(), "escalated": v.escalated}));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_questions_are_told_from_open_goals() {
        for q in [
            "Using sales.csv, what is the total revenue (units x unit_price) across all rows?",
            "How many days are there between 2024-02-17 and 2024-11-05?",
            "Which region has the highest total revenue? Answer with the region name.",
            "Calculate the median of these numbers: 4, 8, 15.",
            "In access.log, how many requests have status=500?\n\nInput files (in your workspace): access.log",
            "What is 15% of 80? Write the answer to output/answer.md.",
        ] {
            assert!(is_exact_question(q), "{q}");
        }
        for q in [
            "create a hello world sleek html page",
            "i want a research document on dogs",
            "Write a report on sales.csv with the total revenue per region.",
            "Summarise the attached file into output/answer.md",
            "Explain why the sky is blue.",
            "Build a Python script that parses access.log",
            "Compare the two proposals and recommend one.",
            "Make a landing page. What is the best layout?",
        ] {
            assert!(!is_exact_question(q), "{q}");
        }
    }

    #[test]
    fn answers_are_compared_by_value() {
        let q = "Using sales.csv, what is the total revenue and which region has the highest revenue?";
        assert!(same_answer("6263.75", "$6,263.75", q));
        assert!(same_answer("The total revenue is 6263.75", "6263.75", q));
        assert!(same_answer("20.83", "20.8333333", "speed in m/s?"));
        assert!(same_answer("20.83 m/s", "20.83", "speed?"));
        assert!(!same_answer("6263.75", "6263.57", q));
        assert!(!same_answer("6263.75; East", "6263.75; West", q));
        assert!(same_answer("East", "east region", q));
        assert!(same_answer("The region is East.", "East", q));
        assert!(!same_answer("East", "West", q));
        assert!(same_answer("2024-11-05", "2024-11-05", "when?"));
        assert!(!same_answer("2024-11-05", "2024-11-06", "when?"));
        assert!(!same_answer("REQ-7731", "REQ-7713", "which request?"));
        assert!(same_answer("833 boxes", "833", "how many boxes?"));
        assert!(!same_answer("833", "849", "how many boxes?"));
        assert!(same_answer("0.30000000000000004", "0.3", "sum?"));
        assert!(same_answer("yes", "Yes.", "is it true?"));
    }

    #[test]
    fn labelled_answers_and_grounding() {
        assert_eq!(labelled_answer("work...\nANSWER: 42").unwrap(), "42");
        assert_eq!(labelled_answer("**Answer:** `East`").unwrap(), "East");
        assert_eq!(labelled_answer("Answer: 1\nmore\nFinal answer: 2").unwrap(), "2");
        assert_eq!(labelled_answer("**Final Answer:** ANSWER: 597.36").unwrap(), "597.36");
        assert!(labelled_answer("ANSWER: <value>").is_none());
        assert!(labelled_answer("no label here").is_none());
        let out = "Total revenue: 6263.75\nHighest region: East";
        assert!(grounded_in("6263.75; East", out));
        assert!(grounded_in("6,263.75", out));
        assert!(!grounded_in("1223", "849"));
        assert!(grounded_in("East", out));
        assert!(!grounded_in("West", out));
        for a in ["REQ-9894 not found in the log.", "None", "", "N/A", "Error: could not parse", "[]", "No matches found"] {
            assert!(is_non_answer(a), "{a}");
        }
        for a in ["0", "No", "633", "East", "none of the regions exceed 100 units: 0"] {
            assert!(!is_non_answer(a), "{a}");
        }
    }

    #[test]
    fn the_cookbook_is_written_for_the_files_at_hand() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("sales.csv"), "region,product,units,unit_price\nNorth,Widget,120,4.50\n").unwrap();
        std::fs::write(d.path().join("config.json"), r#"{"version": 3, "services": [{"name": "auth", "enabled": true, "timeout_s": 30}]}"#).unwrap();
        std::fs::write(d.path().join("access.log"), "2024-06-01 09:00:30 REQ-3475 GET /api/v1/items/123 status=200 time_ms=520\n".repeat(60)).unwrap();
        let files: Vec<String> = ["sales.csv", "config.json", "access.log"].iter().map(|s| s.to_string()).collect();
        let c = cookbook(d.path(), &files);
assert!(c.contains(r#"csv.DictReader(open("sales.csv"))"#) && c.contains("columns: region, product, units, unit_price"), "{c}");
        assert!(c.contains(r#"Number columns (units, unit_price): convert with float(row["units"])"#) && c.contains("Text columns (region, product): use as they are"), "{c}");
        assert!(c.contains(r#"json.load(open("config.json"))"#) && c.contains(r#"data["services"] is a list of dicts with keys: "#) && c.contains("timeout_s"), "{c}");
        assert!(c.contains(r#"open("access.log").read().splitlines()"#) && c.contains(r#"re.search(r"time_ms=(\d+)", line)"#), "{c}");
        let b = files_block(d.path(), &files);
        assert!(b.contains("sales.csv (2 lines, shown in full)"), "{b}");
        assert!(b.contains("access.log (60 lines in total; only a sample of the first") && b.contains("ALL of its lines"), "{b}");
        assert!(cookbook(d.path(), &[]).contains("exact names"));
        std::fs::write(d.path().join("orders.csv"), "order_id,customer_id,amount\nO1,C1,5.00\n").unwrap();
        std::fs::write(d.path().join("customers.csv"), "customer_id,name,city\nC1,Ana,Lyon\n").unwrap();
        let c = cookbook(d.path(), &["orders.csv".to_string(), "customers.csv".to_string()]);
        assert!(c.contains(r#"orders.csv and customers.csv share the column "customer_id""#), "{c}");
    }

    #[test]
    fn voting_stops_on_agreement_and_ranks_clusters() {
        let mk = |a: Option<&str>| Sample { answer: a.map(String::from), ..Default::default() };
        let q = "how many?";
        let s = vec![mk(Some("12")), mk(Some("12"))];
        assert!(decided(&clusters(&s, q)));
        let s = vec![mk(Some("12")), mk(Some("13"))];
        assert!(!decided(&clusters(&s, q)));
        let s = vec![mk(Some("12")), mk(Some("13")), mk(Some("13"))];
        let cs = clusters(&s, q);
        assert_eq!(cs[0], vec![1, 2]);
        assert!(!decided(&cs), "a 2-1 split keeps sampling while attempts remain");
        let s = vec![mk(Some("12")), mk(None), mk(Some("13")), mk(Some("13")), mk(Some("13"))];
        assert!(decided(&clusters(&s, q)));
        let s = vec![mk(Some("7")), mk(Some("8"))];
        assert_eq!(clusters(&s, q)[0], vec![0], "a tie goes to the earliest attempt");
        // one value at two precisions: the form most attempts used wins, then the earliest
        let s = vec![mk(Some("675")), mk(Some("674.63")), mk(Some("674.63"))];
        let cs = clusters(&s, "final price to the cent?");
        assert_eq!(cs[0].len(), 3);
        assert_eq!(pick_form(&cs[0], &s, "what is the price?"), Some(1));
        // the question's own precision decides, whichever attempt came first
        let s = vec![mk(Some("675")), mk(Some("674.63"))];
        let c = clusters(&s, q);
        assert_eq!(pick_form(&c[0], &s, "What is the final price in dollars, to the cent?"), Some(1));
        assert_eq!(pick_form(&c[0], &s, "What is the price, rounded to the nearest whole dollar?"), Some(0));
        assert_eq!(pick_form(&c[0], &s, "What is the price?"), Some(1), "no instruction: the more precise form");
        assert_eq!(wanted_decimals("rounded to 2 decimals"), Some(2));
        assert_eq!(wanted_decimals("to two decimal places"), Some(2));
        assert_eq!(wanted_decimals("How many boxes are there?"), None);
    }
}
