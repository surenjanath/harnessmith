//! Sleep: what a project does with its experience while nobody is waiting on it.
//!
//! The idea comes from "Language Models Need Sleep" (Behrouz, Hashemi, Javanmard and Mirrokni,
//! Google Research, 2026): split a learner's life into a *wake* phase, where it works and
//! collects experience cheaply, and an offline *sleep* phase with two stages — **consolidation**,
//! which moves what was learned into long-term memory, and **dreaming**, where the learner makes
//! up practice material, rehearses on it, and keeps what proves useful. The paper does this to a
//! model's weights. A harness cannot touch weights, so the same cycle is run on what a harness
//! does own, the project's experience (verified solutions kept as worked examples):
//!
//! ```text
//!   consolidate   merge examples of the same kind, forget the ones that never helped,
//!                 let unused memory notes fade
//!   dream         1. rehearse: turn a verified solution into a NEW practice task by changing
//!                    its input data (or the numbers in the question), and compute the right
//!                    answer by running the verified program on the changed data. Ask the model
//!                    cold; if it fails, ask again with the worked example. That measures what
//!                    each example is worth, which is this harness's stand-in for the paper's
//!                    filter on the training value of a synthetic example.
//!                 2. explore: have the model ask itself new questions about the project's
//!                    files, answer them with the voting engine, and keep the ones every
//!                    attempt agrees on, so a kind of question is already solved before it is
//!                    asked for real.
//! ```
//!
//! Ground truth in rehearsal comes from running a program that was already verified, never from
//! the model's own opinion of its answer, which is what keeps a dream from reinforcing a mistake.

use super::experience::{self, Exemplar, Store};
use super::{precise, pycode, sandbox::Sandbox};
use crate::util;
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct SleepOpts {
    pub model_id: String,
    /// Practice tasks to rehearse.
    pub dreams: usize,
    /// New questions the model may ask itself about the project's files.
    pub explore: usize,
    /// Attempts per rehearsal answer (1 keeps a sleep short).
    pub samples: usize,
    pub seed: u64,
    pub wall: Duration,
    /// "auto" | "bwrap" | "unshare" | "off"
    pub sandbox: String,
    /// Guidance from the project's base harness, shown in the "with experience" rehearsal.
    pub notes: String,
}

impl SleepOpts {
    pub fn new(model_id: &str) -> SleepOpts {
        SleepOpts { model_id: model_id.into(), dreams: 6, explore: 2, samples: 1, seed: 0, wall: Duration::from_secs(3600), sandbox: "auto".into(), notes: String::new() }
    }
}

/// SplitMix64, for repeatable perturbations.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A factor in [0.7, 1.5), never within 5% of 1.
    fn factor(&mut self) -> f64 {
        let f = 0.7 + (self.next() % 800) as f64 / 1000.0;
        if (f - 1.0).abs() < 0.05 { 1.21 } else { f }
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

/// A number changed by a factor, written with the decimals it had.
fn scaled(text: &str, f: f64) -> Option<String> {
    let clean = text.replace(',', "");
    let v: f64 = clean.parse().ok()?;
    let decimals = clean.split_once('.').map(|(_, d)| d.len()).unwrap_or(0);
    let mut nv = v * f;
    if decimals == 0 {
        nv = nv.round();
        if nv == v {
            nv = v + if v >= 0.0 { 1.0 } else { -1.0 };
        }
        return Some(format!("{}", nv as i64));
    }
    Some(format!("{nv:.decimals$}"))
}

/// Columns and keys that identify things rather than measure them stay as they are.
fn is_label(name: &str) -> bool {
    Regex::new(r"(?i)(^|[_\s\-])(id|ids|year|zip|code|date|phone|key|no|num|number|version|port|index|rank|status)$").unwrap().is_match(name.trim())
}

/// The same kind of file with other contents: data rows shuffled (and, when `drop_rows`, some
/// left out), measured numbers changed. Structure, headers, names and labels are kept.
pub fn perturb(kind: &str, body: &str, rng: &mut Rng, drop_rows: bool) -> String {
    let num = Regex::new(r"^-?\d[\d,]*(\.\d+)?$").unwrap();
    match kind {
        "csv" | "tsv" => {
            let sep = if kind == "tsv" { '\t' } else { ',' };
            let mut lines = body.lines().filter(|l| !l.trim().is_empty());
            let Some(header) = lines.next() else { return body.to_string() };
            let labels: Vec<bool> = header.split(sep).map(is_label).collect();
            let mut rows: Vec<String> = lines.map(String::from).collect();
            rng.shuffle(&mut rows);
            if drop_rows && rows.len() >= 6 {
                let keep = (rows.len() * 3 / 4).max(4);
                rows.truncate(keep);
            }
            let rows: Vec<String> = rows
                .iter()
                .map(|r| {
                    r.split(sep)
                        .enumerate()
                        .map(|(i, cell)| {
                            let c = cell.trim();
                            // Quoted cells and label columns are left alone.
                            if labels.get(i).copied().unwrap_or(false) || !num.is_match(c) || c.contains(',') {
                                return cell.to_string();
                            }
                            scaled(c, rng.factor()).unwrap_or_else(|| cell.to_string())
                        })
                        .collect::<Vec<_>>()
                        .join(&sep.to_string())
                })
                .collect();
            format!("{header}\n{}\n", rows.join("\n"))
        }
        "json" => {
            fn walk(v: &mut Value, key: &str, rng: &mut Rng) {
                match v {
                    Value::Number(n) if !is_label(key) => {
                        if let Some(i) = n.as_i64() {
                            let ni = ((i as f64) * rng.factor()).round() as i64;
                            *v = json!(if ni == i { i + 1 } else { ni });
                        } else if let Some(f) = n.as_f64() {
                            *v = json!((f * rng.factor() * 100.0).round() / 100.0);
                        }
                    }
                    Value::Array(a) => {
                        if a.iter().all(|x| x.is_object()) {
                            rng.shuffle(a);
                        }
                        for x in a.iter_mut() {
                            walk(x, key, rng);
                        }
                    }
                    Value::Object(m) => {
                        for (k, x) in m.iter_mut() {
                            walk(x, k, rng);
                        }
                    }
                    _ => {}
                }
            }
            match serde_json::from_str::<Value>(body) {
                Ok(mut v) => {
                    walk(&mut v, "", rng);
                    serde_json::to_string_pretty(&v).unwrap_or_else(|_| body.to_string()) + "\n"
                }
                Err(_) => body.to_string(),
            }
        }
        "log" | "text" => {
            let mut lines: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
            if lines.len() < 6 {
                return body.to_string();
            }
            rng.shuffle(&mut lines);
            if drop_rows {
                let keep = (lines.len() * 3 / 4).max(4);
                lines.truncate(keep);
            }
            lines.join("\n") + "\n"
        }
        _ => body.to_string(),
    }
}

/// A question with no files: change the numbers that the question and the verified program
/// share, in both, so the program still computes the right answer for the new question.
/// Only numbers of two or more digits that occur exactly once in each are touched.
pub fn perturb_question(question: &str, program: &str, rng: &mut Rng) -> Option<(String, String)> {
    let tok = Regex::new(r"\d[\d,]*(?:\.\d+)?").unwrap();
    let standalone = |text: &str, m: regex::Match| {
        let before = text[..m.start()].chars().next_back();
        let after = text[m.end()..].chars().next();
        let glued = |c: Option<char>| c.map(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == ':' || c == '/').unwrap_or(false);
        // "2024-02-17", "v2", "10:45" and decimals split by the regex are not plain quantities.
        !glued(before) && !glued(after) && before != Some('.') && !(after == Some('.') && text[m.end() + 1..].chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false))
    };
    let mut q = question.to_string();
    let mut p = program.to_string();
    let mut changed = 0;
    let found: Vec<String> = tok.find_iter(question).filter(|m| standalone(question, *m)).map(|m| m.as_str().trim_end_matches(',').to_string()).collect();
    for n in found {
        let plain = n.replace(',', "");
        if plain.replace('.', "").len() < 2 || question.matches(n.as_str()).count() != 1 {
            continue;
        }
        // The program writes it without thousands separators.
        let in_prog: Vec<regex::Match> = tok.find_iter(&p).filter(|m| m.as_str() == plain && standalone(&p, *m)).collect();
        if in_prog.len() != 1 {
            continue;
        }
        let Some(new) = scaled(&plain, rng.factor()) else { continue };
        if tok.find_iter(&p).any(|m| m.as_str() == new) || q.contains(new.as_str()) {
            continue;
        }
        let (s, e) = (in_prog[0].start(), in_prog[0].end());
        p = format!("{}{new}{}", &p[..s], &p[e..]);
        q = q.replacen(n.as_str(), &new, 1);
        changed += 1;
    }
    (changed > 0).then_some((q, p))
}

struct Dream {
    exemplar: String,
    question: String,
    truth: String,
    cold: Option<String>,
    with: Option<String>,
    /// "not needed" | "helped" | "no help" | "skipped"
    verdict: &'static str,
    why: String,
    seconds: f64,
}

impl Dream {
    fn to_json(&self) -> Value {
        json!({"exemplar": self.exemplar, "question": util::take_chars(&self.question, 300), "truth": self.truth, "cold": self.cold, "with": self.with, "verdict": self.verdict, "why": self.why, "seconds": (self.seconds * 10.0).round() / 10.0})
    }
}

/// Consolidation: one example per kind of question, and the ones rehearsal showed to be useless
/// are forgotten. Returns (merged, forgotten).
pub fn consolidate(store: &Store) -> Result<(usize, usize)> {
    store.update(|all| {
        let (mut merged, mut forgotten) = (0, 0);
        let key = |e: &Exemplar| format!("{}\n{}", experience::kind_key(&e.question), e.files.iter().map(|f| format!("{}|{}", f.kind, f.header)).collect::<Vec<_>>().join(";"));
        // Newest first, so the newest of each kind is the one kept.
        let mut order: Vec<usize> = (0..all.len()).filter(|i| all[*i].status == "active").collect();
        order.sort_by(|a, b| all[*b].pinned.cmp(&all[*a].pinned).then(all[*b].at.cmp(&all[*a].at)));
        let mut seen: Vec<(String, String)> = vec![];
        for i in order {
            let k = key(&all[i]);
            if let Some((_, keeper)) = seen.iter().find(|(x, _)| *x == k) {
                all[i].status = "archived".into();
                all[i].note = format!("merged into {keeper}: the same kind of question");
                merged += 1;
                continue;
            }
            seen.push((k, all[i].id.clone()));
            let e = &mut all[i];
            if !e.pinned && e.dreams >= 2 && e.helped == 0 && e.failed >= 2 {
                e.status = "archived".into();
                e.note = format!("forgotten: in {} rehearsals it never helped", e.dreams);
                forgotten += 1;
            }
        }
        Ok((merged, forgotten))
    })
}

/// One sleep: consolidate, rehearse, explore. Returns the journal (also saved under
/// `<project>/sleep/`).
pub fn sleep(project_root: &Path, o: &SleepOpts) -> Result<Value> {
    if !pycode::available() {
        bail!("python3 is not installed, so practice tasks cannot be checked");
    }
    let t0 = Instant::now();
    let store = Store::open(project_root);
    let before = store.load().iter().filter(|e| e.status == "active").count();
    crate::say!("sleep: consolidating {} example(s) of experience", before);
    crate::log::event("sleep", "start", json!({"examples": before, "dreams": o.dreams, "explore": o.explore, "model": o.model_id}));
    let (merged, forgotten) = consolidate(&store)?;
    let faded = super::memory::Store::open(project_root).fade().unwrap_or(0);
    if merged + forgotten + faded > 0 {
        crate::say!("  consolidated: {merged} merged, {forgotten} forgotten, {faded} memory note(s) faded");
    }
    let tmp_root = project_root.join("sleep").join("tmp");
    util::rmtree(&tmp_root);
    let mut rng = Rng(o.seed ^ 0x5EED_D2EA_4D00_0001 ^ (chrono::Utc::now().timestamp() as u64 / 3600));
    // Rehearse the least-rehearsed, newest examples first.
    let mut queue: Vec<Exemplar> = store.load().into_iter().filter(|e| e.status == "active" && !e.program.trim().is_empty()).collect();
    queue.sort_by(|a, b| a.dreams.cmp(&b.dreams).then(b.at.cmp(&a.at)));
    let mut dreams: Vec<Dream> = vec![];
    for (n, ex) in queue.iter().take(o.dreams).enumerate() {
        if crate::log::cancelled() || t0.elapsed() >= o.wall {
            break;
        }
        let d = rehearse(project_root, &tmp_root.join(format!("d{n}")), ex, o, &mut rng);
        crate::say!("  dream {}: {} — {}{}", n + 1, util::take_chars(&ex.question.replace('\n', " "), 70), d.verdict, if d.why.is_empty() { String::new() } else { format!(" ({})", util::take_chars(&d.why, 110)) });
        crate::log::event("sleep", "dream", d.to_json());
        if d.verdict != "skipped" {
            let (id, verdict) = (ex.id.clone(), d.verdict);
            store.update(|all| {
                if let Some(e) = all.iter_mut().find(|e| e.id == id) {
                    e.dreams += 1;
                    e.last_dream = util::now_iso();
                    match verdict {
                        "not needed" => e.cold_ok += 1,
                        "helped" => e.helped += 1,
                        _ => e.failed += 1,
                    }
                }
                Ok(())
            })?;
        }
        dreams.push(d);
    }
    // Explore: new questions about the files of recent goals.
    let mut explored: Vec<Value> = vec![];
    if o.explore > 0 && !crate::log::cancelled() && t0.elapsed() < o.wall {
        explored = explore(project_root, &tmp_root.join("explore"), &store, o);
    }
    let (m2, f2) = consolidate(&store)?;
    util::rmtree(&tmp_root);
    let done: Vec<&Dream> = dreams.iter().filter(|d| d.verdict != "skipped").collect();
    let count = |v: &str| done.iter().filter(|d| d.verdict == v).count();
    let after = store.load().iter().filter(|e| e.status == "active").count();
    let journal = json!({
        "at": util::now_iso(), "model": o.model_id, "seconds": t0.elapsed().as_secs(),
        "consolidated": {"merged": merged + m2, "forgotten": forgotten + f2, "notes_faded": faded, "examples_before": before, "examples_after": after},
        "rehearsed": done.len(), "right_cold": count("not needed"), "right_with_experience": count("not needed") + count("helped"), "helped": count("helped"), "no_help": count("no help"),
        "skipped": dreams.len() - done.len(),
        "dreams": dreams.iter().map(|d| d.to_json()).collect::<Vec<_>>(),
        "explored": explored,
        "learned": explored.iter().filter(|e| e["kept"] == true).count(),
    });
    let dir = project_root.join("sleep");
    std::fs::create_dir_all(&dir)?;
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    util::write(&dir.join(format!("{stamp}.json")), serde_json::to_string_pretty(&journal)?)?;
    util::write(&dir.join("latest.json"), serde_json::to_string_pretty(&journal)?)?;
    crate::say!(
        "sleep done in {}s: {} practice task(s) — {} right without help, {} more with experience, {} not solved; {} new solution(s) learned by exploring; {} example(s) kept",
        t0.elapsed().as_secs(),
        done.len(),
        count("not needed"),
        count("helped"),
        count("no help"),
        journal["learned"],
        after
    );
    crate::log::event("sleep", "done", json!({"rehearsed": done.len(), "right_cold": count("not needed"), "helped": count("helped"), "no_help": count("no help"), "learned": journal["learned"], "examples": after}));
    Ok(journal)
}

fn engine_opts(o: &SleepOpts, ws: &Path, time_left: Duration) -> precise::Opts {
    let mut po = precise::Opts::new(&o.model_id);
    po.samples = o.samples.clamp(1, 5);
    po.wall = time_left.min(Duration::from_secs(600)).max(Duration::from_secs(30));
    let sb = Sandbox::new(&o.sandbox, ws, &[], false);
    po.sandbox = (sb.kind != super::sandbox::Kind::Off).then_some(sb);
    po.log_prefix = "      ".into();
    po
}

/// The input files of the goal an example was learned from.
fn source_files(project_root: &Path, ex: &Exemplar) -> Option<PathBuf> {
    let d = project_root.join("tasks").join(&ex.goal).join("files");
    (!ex.goal.is_empty() && d.is_dir()).then_some(d)
}

/// One practice task built from one example, rehearsed cold and then (if needed) with it.
fn rehearse(project_root: &Path, ws: &Path, ex: &Exemplar, o: &SleepOpts, rng: &mut Rng) -> Dream {
    let t0 = Instant::now();
    let mut d = Dream { exemplar: ex.id.clone(), question: ex.question.clone(), truth: String::new(), cold: None, with: None, verdict: "skipped", why: String::new(), seconds: 0.0 };
    let skip = |mut d: Dream, why: &str| {
        d.why = why.into();
        d
    };
    if std::fs::create_dir_all(ws).is_err() {
        return skip(d, "could not create a practice workspace");
    }
    let mut question = ex.question.clone();
    let mut program = ex.program.clone();
    let mut files: Vec<String> = vec![];
    if !ex.files.is_empty() {
        let Some(src) = source_files(project_root, ex) else { return skip(d, "its input files are no longer there") };
        // Rows are dropped only from the largest table, so joins keep their partners.
        let largest = ex.files.iter().max_by_key(|f| f.lines).map(|f| f.name.clone()).unwrap_or_default();
        for f in &ex.files {
            let Ok(body) = std::fs::read(src.join(&f.name)) else { return skip(d, "its input files are no longer there") };
            let out = if f.kind == "other" { body } else { perturb(&f.kind, &String::from_utf8_lossy(&body), rng, f.name == largest).into_bytes() };
            if util::write(&ws.join(&f.name), out).is_err() {
                return skip(d, "could not write a practice file");
            }
            files.push(f.name.clone());
        }
    } else if let Some((q, p)) = perturb_question(&ex.question, &ex.program, rng) {
        question = q;
        program = p;
    }
    d.question = question.clone();
    // Ground truth: the verified program, run on the practice data.
    let sb = Sandbox::new(&o.sandbox, ws, &[], false);
    let sb = (sb.kind != super::sandbox::Kind::Off).then_some(sb);
    let truth = match pycode::run(ws, sb.as_ref(), &program, Duration::from_secs(30)) {
        Ok(out) if out.ok() => out.stdout.trim().to_string(),
        Ok(out) => return skip(d, &format!("its program does not run on changed data: {}", util::take_chars(&out.error.replace('\n', " "), 120))),
        Err(e) => return skip(d, &format!("could not run its program: {e}")),
    };
    let lines: Vec<&str> = truth.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() != 1 || util::char_len(lines[0]) > 200 {
        return skip(d, "its program prints more than one line, so there is no single answer to check against");
    }
    d.truth = lines[0].trim().to_string();
    // A program that printed a real figure on the original data and prints zero (or nothing
    // found) on the changed data has lost its footing (a hard-coded id, a row that was dropped):
    // that is no ground truth to hold the model to.
    let zero = |a: &str| precise::atoms(a).0.iter().all(|n| n.v == 0.0) && !precise::atoms(a).0.is_empty();
    if precise::is_non_answer(&d.truth) || (zero(&d.truth) && !zero(&ex.answer)) {
        return skip(d, "its program finds nothing on the changed data, so there is no reliable answer to check against");
    }
    let left = o.wall;
    // Cold: no worked example, no notes.
    let cold = precise::answer(&question, ws, &files, &engine_opts(o, ws, left));
    d.cold = cold.ok().and_then(|v| v.answer);
    if d.cold.as_deref().map(|a| precise::same_answer(a, &d.truth, &question)).unwrap_or(false) {
        d.verdict = "not needed";
        d.why = "the model solved it without the example".into();
    } else if crate::log::cancelled() {
        return skip(d, "cancelled");
    } else {
        let mut po = engine_opts(o, ws, left);
        po.examples = vec![(1.0, ex.clone())];
        po.notes = o.notes.clone();
        d.with = precise::answer(&question, ws, &files, &po).ok().and_then(|v| v.answer);
        if d.with.as_deref().map(|a| precise::same_answer(a, &d.truth, &question)).unwrap_or(false) {
            d.verdict = "helped";
            d.why = format!("wrong without it ({}), right with it", d.cold.as_deref().map(|a| util::take_chars(a, 40)).unwrap_or_else(|| "no answer".into()));
        } else {
            d.verdict = "no help";
            d.why = format!("expected {}, got {} cold and {} with the example", d.truth, d.cold.as_deref().map(|a| util::take_chars(a, 30)).unwrap_or_else(|| "nothing".into()), d.with.as_deref().map(|a| util::take_chars(a, 30)).unwrap_or_else(|| "nothing".into()));
        }
    }
    d.seconds = t0.elapsed().as_secs_f64();
    d
}

/// Dream up new questions about the files of recent goals, and keep the ones the voting engine
/// answers unanimously.
fn explore(project_root: &Path, ws_root: &Path, store: &Store, o: &SleepOpts) -> Vec<Value> {
    let mut out = vec![];
    let all = store.load();
    // The most recent example with files; its goal's inputs are the material.
    let mut with_files: Vec<&Exemplar> = all.iter().filter(|e| e.status == "active" && !e.files.is_empty() && source_files(project_root, e).is_some()).collect();
    with_files.sort_by(|a, b| b.at.cmp(&a.at));
    let Some(seed_ex) = with_files.first() else { return out };
    let Some(src) = source_files(project_root, seed_ex) else { return out };
    let ws = ws_root.to_path_buf();
    if util::copy_tree(&src, &ws).is_err() {
        return out;
    }
    let files: Vec<String> = util::files_sorted(&ws).iter().map(|p| util::rel(&ws, p)).collect();
    let known: Vec<String> = all.iter().filter(|e| e.status == "active").map(|e| e.question.clone()).collect();
    let asked = known.iter().filter(|q| seed_ex.files.iter().any(|f| q.contains(&f.name))).take(6).map(|q| format!("- {}", util::take_chars(&q.replace('\n', " "), 160))).collect::<Vec<_>>().join("\n");
    let prompt = format!(
        "Here are data files from a project:{}\n\nQuestions already asked about them:\n{}\n\nWrite {} NEW questions about these files. Each must have one exact answer (a number or a name) that a short program can compute from the files, and must name the file it uses. One question per line, each ending with a question mark. No numbering, no answers.",
        precise::files_block(&ws, &files),
        if asked.is_empty() { "- (none yet)".to_string() } else { asked },
        o.explore + 2
    );
    let Ok(chain) = super::agent::resolve_chain(&o.model_id) else { return out };
    let (conn, model, _) = &chain[0];
    let msgs = [super::client::Msg::User(prompt)];
    let reply = match super::client::chat(conn, model, &super::client::ChatReq { system: "You write practice questions about data files.", messages: &msgs, tools: &[], max_tokens: 400, temperature: Some(0.8) }, Duration::from_secs(300)) {
        Ok(r) => r.text,
        Err(_) => return out,
    };
    let strip = Regex::new(r"^\s*(?:[-*•]|\d+[.)])\s*").unwrap();
    let mut qs: Vec<String> = vec![];
    for l in reply.lines() {
        let q = strip.replace(l.trim(), "").trim().trim_matches('"').to_string();
        if !q.ends_with('?') || util::char_len(&q) < 20 || util::char_len(&q) > 300 || !precise::is_exact_question(&q) {
            continue;
        }
        let k = experience::kind_key(&q);
        if known.iter().any(|x| experience::kind_key(x) == k) || qs.iter().any(|x| experience::kind_key(x) == k) {
            continue;
        }
        qs.push(q);
    }
    let sigs: Vec<experience::FileSig> = files.iter().map(|f| experience::sig_of(&ws, f)).collect();
    for q in qs.into_iter().take(o.explore) {
        if crate::log::cancelled() {
            break;
        }
        let mut po = engine_opts(o, &ws, o.wall);
        po.samples = 3;
        let Ok(v) = precise::answer(&q, &ws, &files, &po) else { continue };
        let answer = v.answer.clone().unwrap_or_default();
        let computed: Vec<&precise::Sample> = v.samples.iter().filter(|s| s.answer.is_some() && s.how == "program").collect();
        let unanimous = computed.len() >= 2 && computed.iter().all(|s| precise::same_answer(s.answer.as_deref().unwrap_or(""), &answer, &q)) && v.output.lines().filter(|l| !l.trim().is_empty()).count() == 1;
        let mut kept = false;
        if unanimous && !v.program.trim().is_empty() {
            kept = store
                .record(Exemplar { goal: seed_ex.goal.clone(), question: q.clone(), files: sigs.clone(), program: v.program.clone(), output: v.output.clone(), answer: answer.clone(), agree: v.agree, answered: v.answered, model: v.served_by.clone(), status: "active".into(), origin: "dream".into(), ..Default::default() })
                .is_ok();
        }
        crate::say!("  explored: {} → {}", util::take_chars(&q, 80), if kept { format!("{} (kept: {} of {} attempts agree)", util::take_chars(&answer, 40), v.agree, v.answered) } else { "not kept (the attempts did not all agree)".into() });
        crate::log::event("sleep", "explored", json!({"question": util::take_chars(&q, 200), "answer": answer, "kept": kept}));
        out.push(json!({"question": q, "answer": answer, "kept": kept, "agree": v.agree, "answered": v.answered}));
    }
    out
}

/// The journals of earlier sleeps, newest first.
pub fn journals(project_root: &Path, limit: usize) -> Vec<Value> {
    let dir = project_root.join("sleep");
    let mut names: Vec<String> = std::fs::read_dir(&dir).map(|rd| rd.flatten().map(|e| e.file_name().to_string_lossy().to_string()).filter(|n| n.ends_with(".json") && n != "latest.json").collect()).unwrap_or_default();
    names.sort();
    names.reverse();
    names.into_iter().take(limit).filter_map(|n| serde_json::from_str(&std::fs::read_to_string(dir.join(n)).ok()?).ok()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbed_files_keep_their_shape_and_change_their_numbers() {
        let csv = "region,product,units,unit_price,order_id\nNorth,Widget,120,4.50,1001\nSouth,Gadget,40,12.50,1002\nEast,Widget,95,4.75,1003\nWest,Gadget,110,11.75,1004\nEast,Gadget,130,11.00,1005\nNorth,Gadget,75,12.00,1006\n";
        let mut rng = Rng(7);
        let p = perturb("csv", csv, &mut rng, true);
        let lines: Vec<&str> = p.lines().collect();
        assert_eq!(lines[0], "region,product,units,unit_price,order_id", "header kept");
        assert!(lines.len() >= 5 && lines.len() <= 7, "{p}");
        for l in &lines[1..] {
            let c: Vec<&str> = l.split(',').collect();
            assert_eq!(c.len(), 5);
            assert!(["North", "South", "East", "West"].contains(&c[0]));
            assert!(c[2].parse::<i64>().is_ok(), "whole numbers stay whole: {l}");
            assert_eq!(c[3].split('.').nth(1).map(|d| d.len()), Some(2), "decimals kept: {l}");
            assert!(csv.contains(&format!(",{}\n", c[4])), "id columns are not changed: {l}");
        }
        assert_ne!(p, csv);
        // the same seed gives the same file
        assert_eq!(perturb("csv", csv, &mut Rng(7), true), p);
        let j = perturb("json", r#"{"version": 3, "services": [{"name": "auth", "enabled": true, "timeout_s": 30}, {"name": "mail", "enabled": false, "timeout_s": 60}]}"#, &mut Rng(1), false);
        let v: Value = serde_json::from_str(&j).unwrap();
        assert_eq!(v["version"], 3, "labels kept");
        let svcs = v["services"].as_array().unwrap();
        assert_eq!(svcs.len(), 2);
        assert!(svcs.iter().all(|s| s["enabled"].is_boolean() && s["timeout_s"].is_i64() && s["name"].is_string()));
        assert!(svcs.iter().any(|s| s["timeout_s"] != 30 && s["timeout_s"] != 60));
        let log = (0..12).map(|i| format!("line {i} status=500")).collect::<Vec<_>>().join("\n");
        let pl = perturb("log", &log, &mut Rng(3), true);
        assert_eq!(pl.lines().count(), 9);
        assert!(pl.lines().all(|l| log.contains(l)), "log lines are kept whole");
    }

    #[test]
    fn question_numbers_change_together_with_the_program() {
        let q = "A warehouse had 1,250 boxes. It shipped 387 on Monday and 462 on Tuesday, then received 3 pallets of 144 boxes each. How many boxes does it have now?";
        let p = "start = 1250\nshipped = 387 + 462\nreceived = 3 * 144\nprint(start - shipped + received)";
        let (nq, np) = perturb_question(q, p, &mut Rng(5)).unwrap();
        assert_ne!(nq, q);
        assert!(nq.contains("3 pallets"), "single digits are left alone: {nq}");
        // every number changed in the question is the number now in the program
        let tok = Regex::new(r"\d[\d,]*").unwrap();
        let (qn, pn): (Vec<String>, Vec<String>) = (tok.find_iter(&nq).map(|m| m.as_str().replace(',', "")).collect(), tok.find_iter(&np).map(|m| m.as_str().to_string()).collect());
        for n in qn.iter().filter(|n| n.len() >= 2) {
            assert!(pn.contains(n), "{n} missing from the program: {np}");
        }
        // dates and codes are not quantities
        assert!(perturb_question("How many days from 2024-02-17 to 2024-11-05?", "from datetime import date\nprint((date(2024, 11, 5) - date(2024, 2, 17)).days)", &mut Rng(1)).is_none());
        // a number the program does not contain is left alone
        assert!(perturb_question("What is 15% of 80?", "print(0.15 * 8 * 10)", &mut Rng(1)).is_none());
    }

    #[test]
    fn consolidation_merges_kinds_and_forgets_what_never_helped() {
        let d = tempfile::tempdir().unwrap();
        let s = Store::open(d.path());
        let mk = |q: &str, at: &str| Exemplar { id: format!("x-{at}"), at: at.into(), question: q.into(), program: "print(1)".into(), answer: "1".into(), status: "active".into(), ..Default::default() };
        s.update(|all| {
            all.push(mk("How many days are there from 2024-02-17 to 2024-11-05?", "2026-01-01"));
            all.push(mk("How many days are there from 2023-01-09 to 2023-08-30?", "2026-02-01"));
            let mut useless = mk("What is the median of these numbers: 1, 2, 3?", "2026-01-05");
            (useless.dreams, useless.failed) = (3, 3);
            all.push(useless);
            let mut pinned = mk("Which is the third-largest of these numbers: 5, 6, 7?", "2026-01-06");
            (pinned.dreams, pinned.failed, pinned.pinned) = (3, 3, true);
            all.push(pinned);
            Ok(())
        })
        .unwrap();
        assert_eq!(consolidate(&s).unwrap(), (1, 1));
        let all = s.load();
        let active: Vec<&str> = all.iter().filter(|e| e.status == "active").map(|e| e.at.as_str()).collect();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&"2026-02-01"), "the newer of the two is kept");
        assert!(active.contains(&"2026-01-06"), "a pinned example is never forgotten");
        assert!(all.iter().any(|e| e.note.starts_with("merged into")) && all.iter().any(|e| e.note.starts_with("forgotten")));
    }
}
