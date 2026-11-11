//! The maker: pages and coding challenges out of small models.
//!
//! The same idea as the answer engine, for things that are built rather than computed. A small
//! model asked to deliver a page or a program through JSON tool calls loses the file to escaping
//! errors, stops half way, or never tests what it wrote. Here the harness drives the steps:
//!
//! ```text
//!   page   ask for the complete file in one fenced block -> write it -> check it against a
//!          design checklist -> send the list of misses back once -> keep the best candidate
//!   code   ask for the solution in one fenced block, and for assert-style tests -> run every
//!          candidate against every test -> repair the best one on its first failure -> keep
//!          the candidate that passes the most tests
//! ```
//!
//! Every check is run, not asked: a page is scored by what is in the file, a program by the
//! asserts it passes (its own and the other candidates', so a wrong test cannot crown a wrong
//! solution on its own).

use super::client::{self, ChatReq, Msg};
use super::sandbox::Sandbox;
use super::{agent, health, pycode};
use crate::util;
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Page,
    Code,
}

/// Is this goal a single page, or a Python coding task? Anything else stays on the agent loop.
pub fn kind(task: &str) -> Option<Kind> {
    let goal = task.split("\n\nInput files (in your workspace):").next().unwrap_or(task);
    if task.contains("\n\nInput files (in your workspace):") || util::char_len(goal) > 2500 {
        return None; // goals with inputs need the agent's tools
    }
    let g = goal.to_lowercase();
    let exts = crate::core::schedule::expected_exts(goal);
    // "a sleek hello world page" is a web page; "a one-page brief" is a document.
    let doc = Regex::new(r"\b(report|brief|essay|article|document|memo|letter|summary|pdf|docx|one[- ]page|\d+[- ]page|pages)\b").unwrap().is_match(&g);
    let page = Regex::new(r"\b(html|web ?page|landing page|website|home ?page)\b").unwrap().is_match(&g) || (Regex::new(r"\bpage\b").unwrap().is_match(&g) && !doc);
    if page && exts.iter().all(|e| matches!(*e, "html" | "css" | "js")) {
        return Some(Kind::Page);
    }
    let other_lang = Regex::new(r"\b(javascript|typescript|node\.?js|rust|golang|java|c\+\+|c#|ruby|php|swift|kotlin|bash script|shell script|sql)\b").unwrap().is_match(&g);
    let code = Regex::new(r"\b(coding (challenge|problem|task|exercise)|leetcode|(write|code|implement|create|build|make|solve|give me)\b[^.?!\n]{0,80}\b(function|program|script|class|algorithm|solution|code)\b|python (function|program|script))").unwrap().is_match(&g);
    if code && !other_lang && exts.iter().all(|e| *e == "py") {
        return Some(Kind::Code);
    }
    None
}

#[derive(Debug, Clone)]
pub struct Opts {
    pub model_id: String,
    pub candidates: usize,
    pub notes: String,
    pub sandbox: Option<Sandbox>,
    pub wall: Duration,
    pub trajectory: Option<PathBuf>,
    pub max_tokens: u32,
    pub gate: Option<super::gate::Gate>,
    pub log_prefix: String,
}

impl Opts {
    pub fn new(model_id: &str) -> Opts {
        Opts { model_id: model_id.into(), candidates: 3, notes: String::new(), sandbox: None, wall: Duration::from_secs(1200), trajectory: None, max_tokens: 3000, gate: None, log_prefix: "      ".into() }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Made {
    /// Workspace-relative file that was written (empty: nothing usable was produced).
    pub file: String,
    pub kind: String,
    /// Checks passed / checks run, for the candidate that was kept.
    pub passed: usize,
    pub total: usize,
    pub candidates: usize,
    /// What is still wrong with the kept candidate.
    pub open: Vec<String>,
    pub summary: String,
    pub calls: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub served_by: String,
}

impl Made {
    pub fn confidence(&self) -> &'static str {
        if self.file.is_empty() || self.total == 0 {
            "none"
        } else if self.passed == self.total {
            "high"
        } else if self.passed * 4 >= self.total * 3 {
            "medium"
        } else {
            "low"
        }
    }
    pub fn to_json(&self) -> Value {
        json!({"file": self.file, "kind": self.kind, "passed": self.passed, "total": self.total, "candidates": self.candidates, "open": self.open, "confidence": self.confidence(), "calls": self.calls, "model": self.served_by})
    }
}

struct Ctx<'a> {
    o: &'a Opts,
    chain: Vec<(super::connections::Connection, String, String)>,
    traj: Option<std::fs::File>,
    t0: Instant,
    m: Made,
}

impl<'a> Ctx<'a> {
    fn ev(&mut self, v: Value) {
        if let Some(f) = &mut self.traj {
            let _ = writeln!(f, "{v}");
        }
    }
    fn left(&self) -> Duration {
        self.o.wall.checked_sub(self.t0.elapsed()).unwrap_or_default()
    }
    fn call(&mut self, system: &str, msgs: &[Msg], temperature: f64) -> Result<String> {
        let mut last = String::from("no usable model");
        for (conn, model, id) in self.chain.clone() {
            if crate::log::cancelled() {
                bail!(client::CANCELLED);
            }
            if health::over_cap(&conn) || !health::reopens_in(&conn.name).is_zero() {
                continue;
            }
            let left = self.left();
            if left.is_zero() {
                bail!("time budget exhausted");
            }
            self.m.calls += 1;
            crate::log::event("step", "call", json!({"model": id, "step": self.m.calls, "depth": 0}));
            let max_tokens = conn.max_tokens.unwrap_or(self.o.max_tokens).max(self.o.max_tokens.min(conn.window() / 3)).min(conn.window() / 2).max(256);
            match client::chat(&conn, &model, &ChatReq { system, messages: msgs, tools: &[], max_tokens, temperature: Some(conn.temperature.unwrap_or(temperature)) }, left.max(Duration::from_secs(30))) {
                Ok(r) => {
                    self.m.input_tokens += r.input_tokens;
                    self.m.output_tokens += r.output_tokens;
                    let usd = conn.cost(&r);
                    self.m.cost_usd += usd;
                    health::record_cost(&conn.name, usd);
                    self.m.served_by = id.clone();
                    self.ev(json!({"type": "assistant", "text": r.text}));
                    crate::log::event("step", "turn", json!({"step": self.m.calls, "depth": 0, "in": self.m.input_tokens, "out": self.m.output_tokens, "model": id, "tools": Vec::<String>::new(), "say": util::take_chars(r.text.trim(), 120)}));
                    return Ok(r.text);
                }
                Err(e) => {
                    last = format!("{id}: {e}");
                    if last.contains(client::CANCELLED) {
                        bail!(client::CANCELLED);
                    }
                }
            }
        }
        bail!(last)
    }
}

/// The body of the first fenced block tagged with one of `langs` (or untagged); an unclosed
/// fence runs to the end of the reply.
pub fn fenced(reply: &str, langs: &[&str]) -> Option<String> {
    let re = Regex::new(r"(?s)```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\z)").unwrap();
    let mut untagged = None;
    for c in re.captures_iter(reply) {
        let tag = c[1].to_lowercase();
        if langs.contains(&tag.as_str()) && !c[2].trim().is_empty() {
            return Some(c[2].to_string());
        }
        if tag.is_empty() && untagged.is_none() && !c[2].trim().is_empty() {
            untagged = Some(c[2].to_string());
        }
    }
    untagged
}

/// The file the goal names for its deliverable, or `default`.
fn out_file(goal: &str, ext: &str, default: &str) -> String {
    let re = Regex::new(&format!(r"(?i)\b(?:output/)?([\w\-]+\.{ext})\b")).unwrap();
    re.captures(goal).map(|c| format!("output/{}", &c[1])).unwrap_or_else(|| format!("output/{default}"))
}

// ---- pages ----------------------------------------------------------------------------------

const PAGE_SYSTEM: &str = "You are a front-end designer who writes complete, polished single-file web pages.
Rules:
- Reply with the whole page in ONE ```html block and nothing else. No explanations.
- Start with <!DOCTYPE html>. Include <html lang>, <head> with <meta charset>, <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">, a <title>, and one <style> block; then <body>; end with </html>.
- All CSS goes in the <style> block. No external files, fonts, images or scripts from the internet.
- Design: define colours as CSS variables in :root, use a system font stack, centre the content with flexbox or grid, give the main element generous padding, rounded corners and a soft shadow, use a gradient or tinted background, and add one subtle transition or animation (for example on hover).
- Make it responsive with clamp() or a media query.
- Put in exactly the content the task asks for; no placeholder text.";

/// What a page has and lacks: (label of each passed check, description of each miss).
pub fn page_checks(goal: &str, html: &str) -> (Vec<&'static str>, Vec<String>) {
    let h = html.to_lowercase();
    let style: String = Regex::new(r"(?s)<style[^>]*>(.*?)</style>").unwrap().captures_iter(&h).map(|c| c[1].to_string()).collect::<Vec<_>>().join("\n");
    let mut ok = vec![];
    let mut miss = vec![];
    let mut check = |pass: bool, label: &'static str, fix: &str| if pass { ok.push(label) } else { miss.push(fix.to_string()) };
    check(h.trim_start().starts_with("<!doctype html"), "doctype", "the file must start with <!DOCTYPE html>");
    check(h.contains("</html>") && h.contains("<body") && h.contains("</body>"), "complete", "the page is cut off: it needs <body>…</body> and a closing </html>");
    check(h.contains("<title>") && !h.contains("<title></title>"), "title", "add a <title> in <head>");
    check(h.contains("name=\"viewport\"") || h.contains("name='viewport'") || h.contains("name=viewport"), "viewport", "add <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">");
    check(style.trim().len() >= 250, "styled", "the <style> block is missing or nearly empty: write real CSS for layout, typography, colours and spacing");
    check(style.contains("display:flex") || style.contains("display: flex") || style.contains("display:grid") || style.contains("display: grid") || style.contains("place-items"), "layout", "centre the content with flexbox or grid (display: flex / grid)");
    check(style.contains("font-family"), "typography", "set a font-family (a system font stack)");
    check(style.contains("--") && style.contains("var(--"), "variables", "define the colours as CSS variables in :root and use them with var(--name)");
    check(style.contains("gradient") || Regex::new(r"background(-color)?\s*:").unwrap().find_iter(&style).count() >= 2, "background", "give the page a gradient or tinted background");
    check(style.contains("border-radius") && style.contains("box-shadow"), "depth", "give the main element rounded corners (border-radius) and a soft shadow (box-shadow)");
    check(style.contains("transition") || style.contains("animation") || style.contains("@keyframes"), "motion", "add one subtle transition or animation (for example on :hover)");
    check(style.contains("@media") || style.contains("clamp(") || style.contains("vw") || style.contains("min("), "responsive", "make it responsive: use clamp() for sizes or add a @media query");
    check(!Regex::new(r"lorem ipsum|placeholder text|your (text|content) here|\[[a-z ]{3,30}\]").unwrap().is_match(&h), "real content", "replace the placeholder text with real content");
    // Words the goal puts in quotes, and "hello world" when it asks for one, must be on the page.
    let mut wanted: Vec<String> = Regex::new(r#""([^"]{2,60})"|'([^']{3,60})'"#).unwrap().captures_iter(goal).map(|c| c.get(1).or(c.get(2)).unwrap().as_str().to_lowercase()).collect();
    if goal.to_lowercase().contains("hello world") {
        wanted.push("hello".into());
    }
    let body = h.split("<body").nth(1).unwrap_or("");
    let missing: Vec<&String> = wanted.iter().filter(|w| !body.contains(w.as_str())).collect();
    check(missing.is_empty(), "content", &format!("the page must show: {}", missing.iter().map(|w| format!("\"{w}\"")).collect::<Vec<_>>().join(", ")));
    (ok, miss)
}

fn make_page(c: &mut Ctx, goal: &str, ws: &Path) -> Result<()> {
    let file = out_file(goal, "html?", "index.html");
    let system = if c.o.notes.trim().is_empty() { PAGE_SYSTEM.to_string() } else { format!("{PAGE_SYSTEM}\n\nNotes from earlier work on this project:\n{}", c.o.notes.trim()) };
    let ask = format!("{goal}\n\nReply with the complete page in one ```html block.");
    let mut best: Option<(usize, usize, String, Vec<String>)> = None;
    let n = c.o.candidates.clamp(1, 4);
    for i in 0..n {
        if crate::log::cancelled() || c.left().is_zero() {
            break;
        }
        let mut msgs = vec![Msg::User(ask.clone())];
        for round in 0..2 {
            let Ok(reply) = c.call(&system, &msgs, if i == 0 { 0.4 } else { 0.9 }) else { break };
            let html = fenced(&reply, &["html", "htm"]).or_else(|| reply.to_lowercase().contains("<!doctype html").then(|| reply[reply.to_lowercase().find("<!doctype html").unwrap()..].to_string()));
            let Some(html) = html else {
                msgs.push(Msg::Assistant { text: reply, calls: vec![] });
                msgs.push(Msg::User("Reply with the complete page in one ```html block and nothing else.".into()));
                continue;
            };
            let (ok, miss) = page_checks(goal, &html);
            c.ev(json!({"type": "tool", "name": "page_check", "input": {"candidate": i + 1, "round": round + 1}, "status": if miss.is_empty() { "ok" } else { "error" }, "output": if miss.is_empty() { format!("all {} checks passed", ok.len()) } else { miss.join("\n") }}));
            crate::say!("{}page candidate {} (round {}): {}/{} checks", c.o.log_prefix, i + 1, round + 1, ok.len(), ok.len() + miss.len());
            if best.as_ref().map(|b| ok.len() > b.0).unwrap_or(true) {
                best = Some((ok.len(), ok.len() + miss.len(), html.clone(), miss.clone()));
            }
            if miss.is_empty() || round == 1 {
                break;
            }
            msgs.push(Msg::Assistant { text: reply, calls: vec![] });
            msgs.push(Msg::User(format!("The page is not finished. Fix every point and reply with the complete corrected page in one ```html block:\n{}", miss.iter().map(|m| format!("- {m}")).collect::<Vec<_>>().join("\n"))));
        }
        c.m.candidates += 1;
        if best.as_ref().map(|b| b.0 == b.1).unwrap_or(false) {
            break; // a candidate that passes everything: no need for more
        }
    }
    let Some((passed, total, html, miss)) = best else { bail!("the model produced no page") };
    util::write(&ws.join(&file), html.trim_start())?;
    c.m.file = file;
    c.m.kind = "page".into();
    c.m.passed = passed;
    c.m.total = total;
    c.m.open = miss;
    c.m.summary = format!("A single-file page; it passes {passed} of {total} design and completeness checks.");
    Ok(())
}

// ---- code -----------------------------------------------------------------------------------

const CODE_SYSTEM: &str = "You are a careful Python programmer.
Rules:
- Reply with the complete solution in ONE ```python block and nothing else.
- Define exactly the function(s) the task asks for, with the names and arguments it gives. If it names none, write one function with a clear name.
- Standard library only. No input(), no reading files unless the task says so, no code that runs on import except definitions.
- Handle the edge cases the task mentions (empty input, zero, negative numbers, duplicates).";

const TEST_ASK: &str = "Now write tests for that task. Reply with ONE ```python block that contains only single-line assert statements calling the function(s), one per line, 6 to 10 of them. Start with the examples the task gives, then edge cases. Every expected value must be what a correct solution returns.";

const DRIVER: &str = r#"import json, sys, io, contextlib, signal
sols, asserts = json.load(open(sys.argv[1]))
def boom(*a):
    raise TimeoutError()
out = []
for path in sols:
    row = []
    try:
        src = open(path, encoding="utf-8").read()
    except Exception:
        src = "raise SystemExit"
    for a in asserts:
        g = {"__name__": "solution"}
        ok = False
        try:
            signal.signal(signal.SIGALRM, boom); signal.alarm(3)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(src, "solution.py", "exec"), g)
                exec(compile(a, "test", "exec"), g)
            ok = True
        except BaseException:
            ok = False
        finally:
            try: signal.alarm(0)
            except Exception: pass
        row.append(ok)
    out.append(row)
print(json.dumps(out))
"#;

/// Single-line asserts in a block of test code.
pub fn asserts_in(code: &str) -> Vec<String> {
    let mut v: Vec<String> = code.lines().map(|l| l.trim()).filter(|l| l.starts_with("assert ") && l.len() > 9 && l.matches('(').count() == l.matches(')').count()).map(String::from).collect();
    v.dedup();
    v.truncate(12);
    v
}

/// Run every solution against every assert: `matrix[s][a]`.
fn cross_run(ws: &Path, sandbox: Option<&Sandbox>, sols: &[String], asserts: &[String]) -> Result<Vec<Vec<bool>>> {
    let dir = ws.join(pycode::DIR);
    std::fs::create_dir_all(&dir)?;
    let mut paths = vec![];
    for (i, s) in sols.iter().enumerate() {
        let p = format!("{}/cand_{i}.py", pycode::DIR);
        util::write(&ws.join(&p), s)?;
        paths.push(p);
    }
    util::write(&dir.join("cases.json"), json!([paths, asserts]).to_string())?;
    let code = format!("import sys\nsys.argv = ['driver', '{}/cases.json']\n{DRIVER}", pycode::DIR);
    let out = pycode::run(ws, sandbox, &code, Duration::from_secs(20 + 4 * (sols.len() * asserts.len()) as u64))?;
    for p in &paths {
        let _ = std::fs::remove_file(ws.join(p));
    }
    let _ = std::fs::remove_file(dir.join("cases.json"));
    let line = out.stdout.lines().last().unwrap_or("");
    let m: Vec<Vec<bool>> = serde_json::from_str(line).map_err(|_| anyhow::anyhow!("the tests could not be run: {}", util::take_chars(&out.render(), 300)))?;
    Ok(m)
}

fn make_code(c: &mut Ctx, goal: &str, ws: &Path) -> Result<()> {
    if !pycode::available() {
        bail!("python3 is not installed");
    }
    let file = out_file(goal, "py", "solution.py");
    let system = if c.o.notes.trim().is_empty() { CODE_SYSTEM.to_string() } else { format!("{CODE_SYSTEM}\n\nNotes from earlier work on this project:\n{}", c.o.notes.trim()) };
    let ask = format!("{goal}\n\nReply with the complete solution in one ```python block.");
    let (mut sols, mut asserts): (Vec<String>, Vec<String>) = (vec![], vec![]);
    let n = c.o.candidates.clamp(1, 5);
    for i in 0..n {
        if crate::log::cancelled() || c.left().is_zero() {
            break;
        }
        let Ok(reply) = c.call(&system, &[Msg::User(ask.clone())], if i == 0 { 0.2 } else { 0.8 }) else { continue };
        let Some(code) = fenced(&reply, &["python", "py", "python3"]) else { continue };
        if let Some(v) = c.o.gate.as_ref().and_then(|g| g.check("python", &json!({"code": code}), None).filter(|_| g.mode != super::gate::Mode::Observe)) {
            crate::log::event("gate", "blocked", json!({"tool": "python", "rule": v.rule, "reason": v.reason}));
            continue;
        }
        // Tests from the same conversation, so they use this solution's function names.
        let msgs = vec![Msg::User(ask.clone()), Msg::Assistant { text: reply.clone(), calls: vec![] }, Msg::User(TEST_ASK.into())];
        if let Ok(t) = c.call(&system, &msgs, 0.3) {
            for a in asserts_in(&fenced(&t, &["python", "py"]).unwrap_or(t)) {
                if !asserts.contains(&a) {
                    asserts.push(a);
                }
            }
        }
        sols.push(code);
    }
    if sols.is_empty() {
        bail!("the model produced no program");
    }
    c.m.candidates = sols.len();
    let score = |m: &[Vec<bool>], s: usize| m.get(s).map(|r| r.iter().filter(|x| **x).count()).unwrap_or(0);
    let mut matrix = if asserts.is_empty() { vec![vec![]; sols.len()] } else { cross_run(ws, c.o.sandbox.as_ref(), &sols, &asserts)? };
    // A test no candidate passes is more likely a wrong test than three wrong programs.
    if sols.len() >= 2 {
        let keep: Vec<usize> = (0..asserts.len()).filter(|a| matrix.iter().any(|r| r[*a])).collect();
        if keep.len() >= 3 && keep.len() < asserts.len() {
            asserts = keep.iter().map(|a| asserts[*a].clone()).collect();
            matrix = matrix.iter().map(|r| keep.iter().map(|a| r[*a]).collect()).collect();
        }
    }
    let mut best = (0..sols.len()).max_by_key(|s| (score(&matrix, *s), usize::MAX - *s)).unwrap_or(0);
    c.ev(json!({"type": "tool", "name": "run_tests", "input": {"candidates": sols.len(), "tests": asserts.len()}, "status": "ok", "output": (0..sols.len()).map(|s| format!("candidate {}: {}/{} tests", s + 1, score(&matrix, s), asserts.len())).collect::<Vec<_>>().join("\n")}));
    crate::say!("{}code: {} candidate(s) × {} test(s); best passes {}", c.o.log_prefix, sols.len(), asserts.len(), score(&matrix, best));
    // One repair of the best candidate on the tests it fails.
    let failing: Vec<String> = asserts.iter().enumerate().filter(|(a, _)| !matrix[best].get(*a).copied().unwrap_or(true)).map(|(_, t)| t.clone()).collect();
    if !failing.is_empty() && !crate::log::cancelled() && !c.left().is_zero() {
        let msgs = vec![
            Msg::User(ask.clone()),
            Msg::Assistant { text: format!("```python\n{}\n```", sols[best].trim()), calls: vec![] },
            Msg::User(format!("These tests fail with that solution:\n{}\n\nIf the tests are right, fix the solution. Reply with the complete corrected solution in one ```python block.", failing.iter().take(5).map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n"))),
        ];
        if let Ok(reply) = c.call(&system, &msgs, 0.3) {
            if let Some(code) = fenced(&reply, &["python", "py", "python3"]) {
                sols.push(code);
                if let Ok(m) = cross_run(ws, c.o.sandbox.as_ref(), &sols, &asserts) {
                    matrix = m;
                    best = (0..sols.len()).max_by_key(|s| (score(&matrix, *s), usize::MAX - *s)).unwrap_or(best);
                    crate::say!("{}code: after a repair, best passes {}/{}", c.o.log_prefix, score(&matrix, best), asserts.len());
                }
            }
        }
    }
    util::write(&ws.join(&file), format!("{}\n", sols[best].trim()))?;
    let failing: Vec<String> = asserts.iter().enumerate().filter(|(a, _)| !matrix[best].get(*a).copied().unwrap_or(true)).map(|(_, t)| t.clone()).collect();
    c.m.file = file;
    c.m.kind = "code".into();
    c.m.passed = score(&matrix, best);
    c.m.total = asserts.len();
    c.m.open = failing.iter().take(5).map(|f| format!("fails: {f}")).collect();
    let passing: Vec<&String> = asserts.iter().enumerate().filter(|(a, _)| matrix[best].get(*a).copied().unwrap_or(false)).map(|(_, t)| t).collect();
    c.m.summary = format!(
        "{} candidate solution(s) were written and run against {} test(s); the one kept passes {}.\n\nTests it passes:\n\n```python\n{}\n```",
        c.m.candidates,
        asserts.len(),
        c.m.passed,
        passing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
    );
    Ok(())
}

/// Build the deliverable for a page or coding goal into `workspace/output/`.
pub fn make(task: &str, workspace: &Path, k: Kind, o: &Opts) -> Result<Made> {
    let goal = task.split("\n\nInput files (in your workspace):").next().unwrap_or(task).trim();
    let traj = match &o.trajectory {
        Some(p) => {
            if let Some(d) = p.parent() {
                std::fs::create_dir_all(d)?;
            }
            Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?)
        }
        None => None,
    };
    let mut c = Ctx { o, chain: agent::resolve_chain(&o.model_id)?, traj, t0: Instant::now(), m: Made::default() };
    c.ev(json!({"type": "session", "model": o.model_id, "depth": 0, "at": util::now_iso(), "engine": "maker"}));
    c.ev(json!({"type": "user", "text": goal}));
    match k {
        Kind::Page => make_page(&mut c, goal, workspace)?,
        Kind::Code => make_code(&mut c, goal, workspace)?,
    }
    crate::log::event("accuracy", "made", json!({"file": c.m.file, "kind": c.m.kind, "passed": c.m.passed, "total": c.m.total, "candidates": c.m.candidates, "confidence": c.m.confidence()}));
    c.ev(json!({"type": "made", "file": c.m.file, "passed": c.m.passed, "total": c.m.total}));
    Ok(c.m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goals_are_sorted_into_pages_code_and_the_rest() {
        assert_eq!(kind("create a hello world sleek html page"), Some(Kind::Page));
        assert_eq!(kind("code me a sleek designed hello world page"), Some(Kind::Page));
        assert_eq!(kind("write a one-page brief on dogs"), None);
        assert_eq!(kind("code me a sleek designed hello world web page"), Some(Kind::Page));
        assert_eq!(kind("Make a landing page for a coffee shop called \"Bean There\""), Some(Kind::Page));
        assert_eq!(kind("Write a Python function is_palindrome(s) that ignores case and punctuation."), Some(Kind::Code));
        assert_eq!(kind("Coding challenge: given a list of integers, return the two that sum to a target."), Some(Kind::Code));
        assert_eq!(kind("Implement a function fizzbuzz(n) in JavaScript"), None);
        assert_eq!(kind("i want a research document on dogs"), None);
        assert_eq!(kind("Write a script that summarises sales.csv\n\nInput files (in your workspace): sales.csv"), None);
    }

    #[test]
    fn page_checklist_tells_a_bare_page_from_a_designed_one() {
        let bare = "<html><body><h1>Hello, world!</h1></body></html>";
        let (ok, miss) = page_checks("create a hello world sleek html page", bare);
        assert!(ok.contains(&"content") && miss.len() >= 8, "{ok:?} {miss:?}");
        let good = r#"<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Hello</title>
<style>:root{--bg:#0f172a;--fg:#e2e8f0;--accent:#38bdf8}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;font-family:system-ui,sans-serif;background:linear-gradient(135deg,var(--bg),#1e293b);color:var(--fg)}
main{padding:clamp(2rem,6vw,4rem);border-radius:24px;box-shadow:0 20px 60px rgba(0,0,0,.4);background:rgba(255,255,255,.05);transition:transform .3s}main:hover{transform:translateY(-4px)}h1{color:var(--accent);font-size:clamp(2rem,8vw,4rem)}</style></head>
<body><main><h1>Hello, World!</h1></main></body></html>"#;
        let (ok, miss) = page_checks("create a hello world sleek html page", good);
        assert!(miss.is_empty(), "{miss:?}");
        assert_eq!(ok.len(), 14);
        assert_eq!(fenced(&format!("Here you go:\n```html\n{good}\n```\nEnjoy"), &["html"]).unwrap().trim(), good);
        assert_eq!(out_file("save it as output/hello.html please", "html?", "index.html"), "output/hello.html");
    }

    #[test]
    fn candidates_are_ranked_by_the_tests_they_pass() {
        if !pycode::available() {
            return;
        }
        let d = tempfile::tempdir().unwrap();
        let sols = vec!["def add(a, b):\n    return a - b".to_string(), "def add(a, b):\n    return a + b".to_string(), "def add(a, b):\n    while True: pass".to_string()];
        let asserts = asserts_in("# tests\nassert add(1, 2) == 3\nassert add(0, 0) == 0\nassert add(-1, 1) == 0\nprint('x')\nassert add(2, 2) == 5");
        assert_eq!(asserts.len(), 4);
        let m = cross_run(d.path(), None, &sols, &asserts).unwrap();
        assert_eq!(m[0], vec![false, true, false, false]);
        assert_eq!(m[1], vec![true, true, true, false]);
        assert_eq!(m[2], vec![false; 4], "an endless loop fails its tests instead of hanging");
    }
}
