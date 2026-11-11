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
    /// A written document about the goal's data files.
    Report,
}

/// Is this goal a single page, or a Python coding task? Anything else stays on the agent loop.
pub fn kind(task: &str) -> Option<Kind> {
    let goal = task.split("\n\nInput files (in your workspace):").next().unwrap_or(task);
    if util::char_len(goal) > 2500 {
        return None;
    }
    let g = goal.to_lowercase();
    let exts = crate::core::schedule::expected_exts(task);
    if let Some((_, inputs)) = task.split_once("\n\nInput files (in your workspace):") {
        // With data files: a written piece about them is a report; anything else needs the agent's tools.
        let data = inputs.split(',').any(|f| matches!(f.trim().rsplit('.').next().unwrap_or("").to_lowercase().as_str(), "csv" | "tsv" | "json" | "jsonl" | "log" | "txt"));
        let writes = Regex::new(r"\b(report|summary|summari[sz]e|brief|overview|analysis|analy[sz]e|write[- ]?up|describe|findings|memo|table)\b").unwrap().is_match(&g);
        return (data && writes && exts.is_empty() && inputs.split(',').count() <= 4).then_some(Kind::Report);
    }
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
- Structure: <body> holds exactly ONE <main> element and only centres it. Everything else goes inside <main>, stacked top to bottom (headline first, then sections), with a max-width around 720px.
- Design: define colours as CSS variables in :root, use a system font stack, centre <main> with flexbox or grid, give it generous padding, rounded corners and a soft shadow, use a smooth gradient or tinted page background, and add one subtle transition or animation (for example on hover).
- Readability comes first: text must contrast strongly with what is directly behind it. Use dark text (#111-#333) on a white or very light card, or white text on a dark card. Never put text in a colour close to its background.
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
    let (worst, pair) = worst_contrast(&style);
    check(worst >= 3.0, "contrast", &format!("text is hard to read: {pair} have a contrast of only {worst:.1}:1. Use dark text (#1f2937) on a white or very light surface, or white text on a dark one"));
    let kids = body_children(&h);
    check(kids <= 1 || !(Regex::new(r"body\s*\{[^}]*display\s*:\s*(flex|grid)").unwrap().is_match(&style)) || Regex::new(r"body\s*\{[^}]*flex-direction\s*:\s*column").unwrap().is_match(&style), "structure", &format!("<body> has {kids} elements side by side: wrap everything in ONE <main> element and stack the sections inside it top to bottom"));
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

/// sRGB colour from CSS: #rgb, #rrggbb, rgb()/rgba() (None when mostly transparent), common names.
fn parse_color(v: &str) -> Option<[f64; 3]> {
    let v = v.trim().to_lowercase();
    if let Some(h) = v.strip_prefix('#') {
        let h: String = h.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
        let full = match h.len() {
            3 | 4 => h.chars().take(3).flat_map(|c| [c, c]).collect::<String>(),
            6 | 8 => h[..6].to_string(),
            _ => return None,
        };
        let n = u32::from_str_radix(&full, 16).ok()?;
        return Some([((n >> 16) & 255) as f64, ((n >> 8) & 255) as f64, (n & 255) as f64]);
    }
    if let Some(c) = Regex::new(r"rgba?\(\s*([\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?").unwrap().captures(&v) {
        let alpha = c.get(4).map(|a| { let t = a.as_str(); if let Some(p) = t.strip_suffix('%') { p.parse::<f64>().unwrap_or(100.0) / 100.0 } else { t.parse::<f64>().unwrap_or(1.0) } }).unwrap_or(1.0);
        if alpha < 0.6 {
            return None;
        }
        return Some([c[1].parse().ok()?, c[2].parse().ok()?, c[3].parse().ok()?]);
    }
    let named = [("white", [255.0, 255.0, 255.0]), ("black", [0.0, 0.0, 0.0]), ("red", [255.0, 0.0, 0.0]), ("green", [0.0, 128.0, 0.0]), ("blue", [0.0, 0.0, 255.0]), ("gray", [128.0, 128.0, 128.0]), ("grey", [128.0, 128.0, 128.0]), ("navy", [0.0, 0.0, 128.0]), ("teal", [0.0, 128.0, 128.0]), ("orange", [255.0, 165.0, 0.0]), ("yellow", [255.0, 255.0, 0.0]), ("purple", [128.0, 0.0, 128.0]), ("pink", [255.0, 192.0, 203.0]), ("silver", [192.0, 192.0, 192.0]), ("whitesmoke", [245.0, 245.0, 245.0]), ("ivory", [255.0, 255.0, 240.0]), ("beige", [245.0, 245.0, 220.0]), ("brown", [165.0, 42.0, 42.0]), ("gold", [255.0, 215.0, 0.0]), ("crimson", [220.0, 20.0, 60.0])];
    let word = v.split(|c: char| !c.is_ascii_alphabetic()).find(|w| !w.is_empty())?;
    named.iter().find(|(n, _)| *n == word).map(|(_, c)| *c)
}

fn contrast(a: [f64; 3], b: [f64; 3]) -> f64 {
    let lum = |c: [f64; 3]| {
        let f = |x: f64| { let x = x / 255.0; if x <= 0.03928 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) } };
        0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])
    };
    let (x, y) = (lum(a), lum(b));
    (x.max(y) + 0.05) / (x.min(y) + 0.05)
}

/// The worst text-on-surface contrast the stylesheet sets up, and which pair it is. Text in a
/// rule is compared with that rule's own background when it has one; otherwise with the card
/// surfaces (backgrounds of rules other than html/body), or the page background when there are
/// none. Hover and other state rules are left out. Returns 21 when nothing can be compared.
pub fn worst_contrast(style: &str) -> (f64, String) {
    let mut vars: Vec<(String, String)> = vec![];
    for c in Regex::new(r"(--[\w-]+)\s*:\s*([^;}]+)").unwrap().captures_iter(style) {
        vars.push((c[1].to_string(), c[2].trim().to_string()));
    }
    let resolve = |v: &str| -> String {
        let mut out = v.to_string();
        for _ in 0..3 {
            for (k, val) in &vars {
                out = out.replace(&format!("var({k})"), val);
            }
        }
        out
    };
    let colors_in = |v: &str| -> Vec<[f64; 3]> {
        let v = resolve(v);
        Regex::new(r"#[0-9a-f]{3,8}\b|rgba?\([^)]*\)|\b[a-z]{3,10}\b").unwrap().find_iter(&v).filter_map(|m| parse_color(m.as_str())).collect()
    };
    let rule = Regex::new(r"([^{}]+)\{([^{}]*)\}").unwrap();
    let decl = |body: &str, prop: &str| -> Option<String> {
        Regex::new(&format!(r"(?:^|;)\s*{prop}\s*:\s*([^;]+)")).unwrap().captures_iter(body).last().map(|c| c[1].to_string())
    };
    let (mut page, mut cards): (Vec<[f64; 3]>, Vec<[f64; 3]>) = (vec![], vec![]);
    let mut texts: Vec<(String, [f64; 3], Vec<[f64; 3]>)> = vec![];
    for c in rule.captures_iter(style) {
        let sel = c[1].trim().to_string();
        if sel.starts_with('@') || sel.contains(':') && !sel.starts_with(":root") {
            continue;
        }
        let body = &c[2];
        // Gradient-filled text (background-clip: text) paints the letters, not a surface.
        let clipped = body.contains("background-clip") || Regex::new(r"(?:^|;)\s*color\s*:\s*transparent").unwrap().is_match(body);
        let bg: Vec<[f64; 3]> = if clipped { vec![] } else { decl(body, "background(?:-color|-image)?").map(|v| colors_in(&v)).unwrap_or_default() };
        if clipped {
            continue;
        }
        let has_own_text = decl(body, "color").is_some();
        let is_page = Regex::new(r"^(html|body|:root|\*)(\s*,\s*(html|body))*$").unwrap().is_match(&sel);
        // A rule that sets both its text and its background (a button, a badge) is its own
        // surface, not the surface other text sits on.
        if is_page { page.extend(bg.iter()) } else if !has_own_text { cards.extend(bg.iter()) }
        if let Some(fg) = decl(body, "color").and_then(|v| colors_in(&v).first().copied()) {
            if !is_page || !bg.is_empty() {
                texts.push((sel, fg, bg));
            } else {
                texts.push((sel, fg, vec![]));
            }
        }
    }
    let mut worst = (21.0, String::new());
    for (sel, fg, own) in texts {
        let is_page = sel == "body" || sel == "html";
        let against: &Vec<[f64; 3]> = if !own.is_empty() { &own } else if is_page || cards.is_empty() { &page } else { &cards };
        for bg in against {
            let r = contrast(fg, *bg);
            if r < worst.0 {
                worst = (r, format!("the text colour of `{sel}` (rgb {:.0},{:.0},{:.0}) and the background behind it (rgb {:.0},{:.0},{:.0})", fg[0], fg[1], fg[2], bg[0], bg[1], bg[2]));
            }
        }
    }
    worst
}

/// How many elements sit directly inside <body> (scripts and styles aside).
fn body_children(html: &str) -> usize {
    let Some(start) = html.find("<body") else { return 0 };
    let body = &html[start..];
    let body = &body[body.find('>').map(|i| i + 1).unwrap_or(0)..];
    let tag = Regex::new(r"<(/?)([a-z][a-z0-9]*)[^>]*?(/?)>").unwrap();
    let void = ["br", "hr", "img", "input", "meta", "link", "source", "wbr", "area", "base", "col", "embed", "track"];
    let (mut depth, mut kids) = (0i32, 0usize);
    for c in tag.captures_iter(body) {
        let (close, name, selfc) = (&c[1] == "/", &c[2], &c[3] == "/");
        if name == "body" || name == "html" {
            break;
        }
        if close {
            depth -= 1;
            continue;
        }
        if depth == 0 && !matches!(name, "script" | "style" | "noscript") {
            kids += 1;
        }
        if !selfc && !void.contains(&name) {
            depth += 1;
        }
    }
    kids
}

/// The design kit: a stylesheet and page frame that are known to look good, so a small model
/// only has to write the content. Its CSS design sense is the weak part; its HTML is fine.
const KIT_CSS: &str = ":root{--accent:ACCENT;--ink:#0f172a;--muted:#475569;--line:#e2e8f0;--card:#ffffff;--bg1:#f8fafc;--bg2:#e0e7ff}
*{box-sizing:border-box}
html{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:var(--ink);line-height:1.6}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:clamp(1rem,4vw,3rem);background:radial-gradient(1200px 600px at 10% -10%,rgba(99,102,241,.16),rgba(99,102,241,0)),linear-gradient(160deg,var(--bg1),var(--bg2))}
main{width:min(100%,760px);background:var(--card);border:1px solid var(--line);border-radius:28px;padding:clamp(1.75rem,5vw,3.5rem);box-shadow:0 30px 80px -30px rgba(15,23,42,.35),0 2px 6px rgba(15,23,42,.06);animation:rise .6s ease both}
h1{font-size:clamp(2.2rem,6vw,3.6rem);line-height:1.08;letter-spacing:-.03em;margin:0 0 .6rem;background:linear-gradient(120deg,var(--ink),var(--accent));-webkit-background-clip:text;background-clip:text;color:transparent}
h2{font-size:1.25rem;margin:2rem 0 .75rem;letter-spacing:-.01em;color:var(--ink)}
p{margin:.4rem 0 1rem;color:var(--muted);font-size:1.08rem}
ul{list-style:none;padding:0;margin:0;display:grid;gap:.5rem}
li{display:flex;justify-content:space-between;gap:1rem;padding:.8rem 1rem;border:1px solid var(--line);border-radius:14px;background:#f8fafc;color:var(--ink);transition:transform .2s ease,border-color .2s ease}
li:hover{transform:translateX(4px);border-color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.9rem}
.card{padding:1.1rem;border:1px solid var(--line);border-radius:18px;background:#f8fafc;color:var(--ink);transition:transform .2s ease,box-shadow .2s ease}
.card:hover{transform:translateY(-4px);box-shadow:0 14px 30px -18px rgba(15,23,42,.5)}
.card h3{margin:0 0 .25rem;font-size:1.05rem;color:var(--ink)}
.price,strong{color:var(--accent);font-weight:700}
.button{display:inline-block;margin-top:1rem;padding:.85rem 1.5rem;border-radius:999px;background:var(--accent);color:#ffffff;text-decoration:none;font-weight:600;transition:transform .2s ease,filter .2s ease}
.button:hover{transform:translateY(-2px);filter:brightness(1.08)}
footer{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.95rem}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@media (max-width:520px){main{border-radius:20px}li{flex-direction:column;gap:.1rem}}";

const KIT_SYSTEM: &str = "You write the content of a web page. The design (colours, layout, typography, animation) is already done; you supply only what goes inside it.
Reply in exactly this form and nothing else:

TITLE: <the browser tab title>
ACCENT: <one strong colour as #rrggbb that suits the topic, dark enough for white text on it>
```html
<the inner HTML of the page's <main> element>
```

Inside the block use only: <h1> (once, first), <p>, <h2>, <ul><li>, <strong>, <a class=\"button\" href=\"#\">, <footer>, and for items with a price or short facts either <li><span>name</span><span class=\"price\">$3.50</span></li> or <div class=\"grid\"><div class=\"card\"><h3>…</h3><p>…</p></div>…</div>.
No <html>, <head>, <body>, <main>, <style>, <script> or inline styles. Write only what the task asks for, with real text: no placeholders, and no invented sections, links, news or company details the task did not mention. A simple request gets a short page (a hello world page is a headline and a sentence or two).";

/// A page from the design kit: the model's content inside the harness's frame.
fn kit_page(reply: &str) -> Option<String> {
    let inner = fenced(reply, &["html", "htm"])?;
    // The frame already has the <main>: a wrapper the model added would draw a card in a card.
    let inner = Regex::new(r"(?is)^\s*<main[^>]*>(.*)</main>\s*$").unwrap().captures(&inner).map(|c| c[1].to_string()).unwrap_or(inner);
    let low = inner.to_lowercase();
    if low.contains("<style") || low.contains("<body") || low.contains("<html") || !low.contains("<h1") {
        return None;
    }
    let title = Regex::new(r"(?im)^\s*TITLE:\s*(.+)$").unwrap().captures(reply).map(|c| c[1].trim().to_string()).unwrap_or_else(|| "Page".into());
    // The accent must carry white button text; otherwise a safe default.
    let accent = Regex::new(r"(?im)^\s*ACCENT:\s*(#[0-9a-fA-F]{6})").unwrap().captures(reply).map(|c| c[1].to_string()).filter(|a| parse_color(a).map(|c| contrast(c, [255.0, 255.0, 255.0]) >= 3.5).unwrap_or(false)).unwrap_or_else(|| "#4f46e5".into());
    let esc = |t: &str| t.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    Some(format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{}</title>\n<style>\n{}\n</style>\n</head>\n<body>\n<main>\n{}\n</main>\n</body>\n</html>\n",
        esc(&title),
        KIT_CSS.replace("ACCENT", &accent),
        inner.trim()
    ))
}

fn make_page(c: &mut Ctx, goal: &str, ws: &Path) -> Result<()> {
    let file = out_file(goal, "html?", "index.html");
    // First choice: the design kit. The model writes content; the frame is known to look right.
    let mut msgs = vec![Msg::User(format!("{goal}\n\nWrite the page content in the form described."))];
    for round in 0..2 {
        let Ok(reply) = c.call(KIT_SYSTEM, &msgs, 0.4) else { break };
        let Some(html) = kit_page(&reply) else {
            msgs.push(Msg::Assistant { text: reply, calls: vec![] });
            msgs.push(Msg::User("Use exactly the form: a TITLE: line, an ACCENT: line, then one ```html block holding only the inner HTML of <main>, starting with an <h1>. No <style>, <body> or <html>.".into()));
            continue;
        };
        let (ok, miss) = page_checks(goal, &html);
        crate::say!("{}page from the design kit (round {}): {}/{} checks", c.o.log_prefix, round + 1, ok.len(), ok.len() + miss.len());
        c.ev(json!({"type": "tool", "name": "page_check", "input": {"candidate": "design kit", "round": round + 1}, "status": if miss.is_empty() { "ok" } else { "error" }, "output": if miss.is_empty() { format!("all {} checks passed", ok.len()) } else { miss.join("\n") }}));
        if miss.is_empty() || round == 1 && miss.len() <= 1 {
            util::write(&ws.join(&file), &html)?;
            c.m.file = file;
            c.m.kind = "page".into();
            c.m.candidates = 1;
            c.m.passed = ok.len();
            c.m.total = ok.len() + miss.len();
            c.m.open = miss;
            c.m.summary = format!("A single-file page: the content was written for this goal and set in the harness's design frame. It passes {} of {} design, readability and completeness checks.", c.m.passed, c.m.total);
            return Ok(());
        }
        msgs.push(Msg::Assistant { text: reply, calls: vec![] });
        msgs.push(Msg::User(format!("Not finished. Fix these and reply again in the same form:\n{}", miss.iter().map(|m| format!("- {m}")).collect::<Vec<_>>().join("\n"))));
    }
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

const TEST_ASK: &str = "Now write tests for that task. Reply with ONE ```python block that contains only single-line assert statements calling the function(s), one per line, 6 to 10 of them. Start with the examples the task gives, then write at least one test for EVERY requirement the task states, including how the result must be ordered or sorted (use an input whose natural order differs from the required order), then edge cases. Every expected value must be exactly what the task's rules give.";

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
    // Requirements review: tests the model wrote can miss a sentence of the task (an ordering,
    // an edge case). Ask once for a re-read; keep the revision only if it passes every test the
    // current best passes.
    if !crate::log::cancelled() && !c.left().is_zero() {
        let msgs = vec![
            Msg::User(ask.clone()),
            Msg::Assistant { text: format!("```python\n{}\n```", sols[best].trim()), calls: vec![] },
            Msg::User("Re-read the task sentence by sentence. For each requirement (names, arguments, return type, ordering or sorting of the result, edge cases) check that the solution does exactly that. If something is missing or different, fix it. Reply with the complete final solution in one ```python block and nothing else.".into()),
        ];
        if let Ok(reply) = c.call(&system, &msgs, 0.2) {
            if let Some(code) = fenced(&reply, &["python", "py", "python3"]).filter(|x| x.trim() != sols[best].trim()) {
                let mut both = vec![sols[best].clone(), code.clone()];
                if let Ok(m) = if asserts.is_empty() { Ok(vec![vec![], vec![]]) } else { cross_run(ws, c.o.sandbox.as_ref(), &both, &asserts) } {
                    let keeps = (0..asserts.len()).all(|a| !m[0][a] || m[1][a]);
                    // With no tests at all, the revision must at least run.
                    let runs = !asserts.is_empty() || pycode::run(ws, c.o.sandbox.as_ref(), &code, Duration::from_secs(10)).map(|o| o.ok()).unwrap_or(false);
                    if keeps && runs {
                        crate::say!("{}code: requirements review revised the solution (still passes every test)", c.o.log_prefix);
                        sols.push(both.remove(1));
                        best = sols.len() - 1;
                        matrix.push(m[1].clone());
                    }
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

// ---- reports on data -------------------------------------------------------------------------

const FACTS_SYSTEM: &str = "You prepare the figures for a written report by writing a short Python program. The program is run for you; what it prints becomes the report's facts.
Rules:
- Compute every figure the report will need (totals, per-group values, counts, averages, the largest and smallest) from the files. Never calculate in your head.
- Print each fact on its own line as `label: value`, with clear labels (for a per-group result, one line per group). Round money to 2 decimals.
- Standard library only (csv, json, math, statistics, collections, re, datetime). Simple step-by-step code with plain loops. No input().
- Do NOT write or open any file for writing, and do not write the report itself: only print() the facts. The report is written later from what you print.";

const WRITE_SYSTEM: &str = "You write clear, short business documents in Markdown.
Rules:
- Reply with the whole document in ONE ```markdown block and nothing else.
- Use ONLY the facts you are given. Copy each figure exactly as it appears; do not calculate new figures, do not round differently, do not invent any number.
- Say which is highest or lowest only as the facts' `Highest:` and `Lowest:` lines say; never rank things yourself.
- Structure: a `# Title` line; `## Summary` with one paragraph on the main finding; a `## ` section with a Markdown table when the task asks for one or the facts are per-group; `## Conclusion` with two or three sentences. Headings on their own lines, a blank line after each. No front matter, no `title:` labels.
- No placeholders, no template variables, no code.";

/// The per-group facts with the largest and the smallest value: (label, value as printed).
/// Totals, averages and counts are not groups.
pub fn ranking(facts: &str) -> (Option<(String, String)>, Option<(String, String)>) {
    let line = Regex::new(r"^\s*(.+?)\s*:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*%?\s*$").unwrap();
    let agg = Regex::new(r"(?i)\b(total|sum|overall|average|avg|mean|median|count|number of|all|grand)\b").unwrap();
    let mut groups: Vec<(String, String, f64)> = vec![];
    for l in facts.lines() {
        if let Some(c) = line.captures(l) {
            if agg.is_match(&c[1]) {
                continue;
            }
            if let Ok(v) = c[2].replace(',', "").parse::<f64>() {
                groups.push((c[1].trim().to_string(), c[2].to_string(), v));
            }
        }
    }
    if groups.len() < 3 {
        return (None, None);
    }
    let hi = groups.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    let lo = groups.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap()).unwrap();
    // A tie at either end leaves no single answer to hold the text to.
    if groups.iter().filter(|g| g.2 == hi.2).count() > 1 || groups.iter().filter(|g| g.2 == lo.2).count() > 1 {
        return (None, None);
    }
    (Some((hi.0.clone(), hi.1.clone())), Some((lo.0.clone(), lo.1.clone())))
}

/// A sentence that names a leader or a laggard other than the real one.
pub fn wrong_ranking(doc: &str, top: Option<&str>, bottom: Option<&str>) -> Option<String> {
    let hi = Regex::new(r"(?i)\b(leads?|leading|highest|largest|biggest|top|most|best|strongest|greatest)\b").unwrap();
    let lo = Regex::new(r"(?i)\b(lowest|smallest|least|worst|weakest|bottom|trails?)\b").unwrap();
    // The last word of a label ("East", "revenue for East" -> "east") identifies it in prose.
    let key = |l: &str| l.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).last().unwrap_or("").to_lowercase();
    for sent in doc.split(|c| c == '.' || c == '\n' || c == ';' || c == ',') {
        let s = sent.to_lowercase();
        let has = |name: Option<&str>| name.map(|n| !key(n).is_empty() && s.split(|c: char| !c.is_alphanumeric()).any(|w| w == key(n))).unwrap_or(false);
        if hi.is_match(&s) && !lo.is_match(&s) && has(bottom) && !has(top) {
            return Some(format!("this is wrong: \"{}\". The highest is {}, the lowest is {}", sent.trim(), top.unwrap_or("?"), bottom.unwrap_or("?")));
        }
        if hi.is_match(&s) && !lo.is_match(&s) && !has(top) && top.is_some() {
            // Names another group as the leader?
            if let Some(t) = top {
                if s.contains(" with ") || s.contains(" at ") || s.contains(" region") {
                    let mentions_other = Regex::new(r"\b[A-Z][a-z]+\b").unwrap().find_iter(sent).any(|m| !["The", "This", "It", "In", "A", "An", "Our"].contains(&m.as_str()) && m.as_str().to_lowercase() != key(t));
                    if mentions_other && Regex::new(r"(?i)\b(leads?|leading|highest|largest|top)\b").unwrap().is_match(&s) {
                        return Some(format!("this names the wrong leader: \"{}\". The highest is {}", sent.trim(), t));
                    }
                }
            }
        }
        if lo.is_match(&s) && !hi.is_match(&s) && has(top) && !has(bottom) {
            return Some(format!("this is wrong: \"{}\". The lowest is {}, the highest is {}", sent.trim(), bottom.unwrap_or("?"), top.unwrap_or("?")));
        }
    }
    None
}

fn make_report(c: &mut Ctx, task: &str, goal: &str, ws: &Path) -> Result<()> {
    if !pycode::available() {
        bail!("python3 is not installed");
    }
    let files: Vec<String> = task.split_once("\n\nInput files (in your workspace):").map(|(_, f)| f.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()).unwrap_or_default();
    let file = Regex::new(r"(?i)\boutput/([\w\-]+\.(?:md|txt))\b").unwrap().captures(goal).map(|m| format!("output/{}", &m[1])).unwrap_or_else(|| "output/report.md".into());
    // 1. The facts, computed.
    let system = format!("{FACTS_SYSTEM}{}", super::precise::cookbook(ws, &files));
    let ask = format!("A report will be written later about this:\n\"{goal}\"{}\n\nYour job now is only the figures. Write the Python program that print()s the facts that report needs. Reply with one ```python block and nothing else.", super::precise::files_block(ws, &files));
    let mut msgs = vec![Msg::User(ask)];
    let mut facts = String::new();
    for round in 0..3 {
        let Ok(reply) = c.call(&system, &msgs, if round == 0 { 0.2 } else { 0.6 }) else { break };
        let Some(code) = pycode::extract_code(&reply) else {
            msgs.push(Msg::Assistant { text: reply, calls: vec![] });
            msgs.push(Msg::User("Reply with the program in one ```python block.".into()));
            continue;
        };
        if let Some(v) = c.o.gate.as_ref().and_then(|g| g.check("python", &json!({"code": code}), None).filter(|_| g.mode != super::gate::Mode::Observe)) {
            bail!("blocked by the operator's gate (rule {})", v.rule);
        }
        let out = pycode::run(ws, c.o.sandbox.as_ref(), &code, Duration::from_secs(40))?;
        c.ev(json!({"type": "tool", "name": "python", "input": {"code": pycode::clean(&code)}, "status": if out.ok() { "ok" } else { "error" }, "output": util::take_chars(&out.render(), 3000)}));
        let lines = out.stdout.lines().filter(|l| !l.trim().is_empty()).count();
        if out.ok() && lines >= 2 && lines <= 80 {
            facts = out.stdout.trim().to_string();
            break;
        }
        let problem = if !out.ok() { format!("The program failed:\n{}", util::take_chars(&out.render(), 800)) } else if lines < 2 { "The program printed almost nothing. Do not write to a file: print() every fact the report needs, one `label: value` per line.".to_string() } else { format!("The program printed {lines} lines, too many for a short report. Print only summary facts (totals and per-group values), not every row.") };
        msgs.push(Msg::Assistant { text: reply, calls: vec![] });
        msgs.push(Msg::User(format!("{problem}\n\nFix it. Reply with the full corrected program in one ```python block.")));
    }
    if facts.is_empty() {
        bail!("no facts could be computed from the files");
    }
    // Rankings are where a small model slips with correct numbers in hand ("West leads" when
    // East is higher), so the harness works them out and states them as facts.
    let (top, bottom) = ranking(&facts);
    if let (Some(t), Some(b)) = (&top, &bottom) {
        facts.push_str(&format!("\nHighest: {} ({})\nLowest: {} ({})", t.0, t.1, b.0, b.1));
    }
    crate::say!("{}report: {} computed fact(s)", c.o.log_prefix, facts.lines().count());
    // 2. The document, written from those facts only, and 3. every figure checked.
    let mut ev = super::ground::Evidence::default();
    ev.load_inputs(ws, goal);
    ev.add_tool_output(&facts);
    let ask = format!("Task: {goal}\n\nFacts (computed from the files; use only these):\n{facts}\n\nWrite the document. Reply with one ```markdown block.");
    let mut msgs = vec![Msg::User(ask)];
    let mut best: Option<(usize, String, Vec<String>)> = None;
    let tmp = ws.join(pycode::DIR).join("draft");
    for round in 0..3 {
        let Ok(reply) = c.call(WRITE_SYSTEM, &msgs, if round == 0 { 0.3 } else { 0.5 }) else { break };
        let doc = fenced(&reply, &["markdown", "md"]).unwrap_or_else(|| reply.clone());
        util::rmtree(&tmp);
        util::write(&tmp.join("report.md"), doc.trim())?;
        let mut miss: Vec<String> = super::ground::unsupported(&tmp, &mut ev, 8).into_iter().map(|(_, n)| n).collect();
        let placeholders = Regex::new(r"\$\{[^}]*\}|\{[a-z_]+(:[^}]*)?\}|\[[A-Za-z ]{3,30}\]\s*$|TODO|lorem ipsum").unwrap().is_match(&doc);
        let words = doc.split_whitespace().count();
        let mut notes = vec![];
        if !miss.is_empty() {
            notes.push(format!("these figures are not among the facts or in the files: {}. Use only figures from the facts, copied exactly", miss.join(", ")));
        }
        if placeholders {
            notes.push("it contains template placeholders: write the actual values".into());
            miss.push("placeholder".into());
        }
        if words < 60 {
            notes.push(format!("it has only {words} words: write the title, the summary paragraph, the table and a conclusion"));
            miss.push("too short".into());
        }
        if let Some(wrong) = wrong_ranking(&doc, top.as_ref().map(|t| t.0.as_str()), bottom.as_ref().map(|b| b.0.as_str())) {
            notes.push(wrong);
            miss.push("ranking".into());
        }
        if !doc.lines().any(|l| l.starts_with("# ")) || doc.lines().filter(|l| l.starts_with("## ")).count() < 2 || Regex::new(r"(?im)^(title|summary|table|total|conclusion)\s*:").unwrap().is_match(&doc) {
            notes.push("format it as Markdown: one `# Title` line, then `## Summary`, `## ...` section headings on their own lines with a blank line after each; no `title:` or `summary:` labels and no front matter".into());
            miss.push("format".into());
        }
        if Regex::new(r"(?im)^#\s+\S*\.(md|txt|csv)\s*$").unwrap().is_match(&doc) {
            notes.push("the title is a file name: give the document a real title that says what it is about".into());
            miss.push("title".into());
        }
        if goal.to_lowercase().contains("table") && !doc.contains("|---") && !doc.contains("| ---") {
            notes.push("the task asks for a table: add a Markdown table".into());
            miss.push("no table".into());
        }
        c.ev(json!({"type": "tool", "name": "report_check", "input": {"round": round + 1}, "status": if notes.is_empty() { "ok" } else { "error" }, "output": if notes.is_empty() { "every figure is supported".to_string() } else { notes.join("\n") }}));
        crate::say!("{}report draft {}: {}", c.o.log_prefix, round + 1, if notes.is_empty() { "every figure is supported".to_string() } else { format!("{} problem(s)", notes.len()) });
        if best.as_ref().map(|b| miss.len() < b.0).unwrap_or(true) {
            best = Some((miss.len(), doc.clone(), notes.clone()));
        }
        if notes.is_empty() {
            break;
        }
        msgs.push(Msg::Assistant { text: reply, calls: vec![] });
        msgs.push(Msg::User(format!("Not finished:\n{}\n\nReply with the complete corrected document in one ```markdown block.", notes.iter().map(|n| format!("- {n}")).collect::<Vec<_>>().join("\n"))));
    }
    util::rmtree(&tmp);
    let Some((n_miss, doc, notes)) = best else { bail!("the model wrote no document") };
    util::write(&ws.join(&file), format!("{}\n", doc.trim()))?;
    c.m.file = file;
    c.m.kind = "report".into();
    c.m.candidates = 1;
    c.m.total = 4;
    c.m.passed = 4usize.saturating_sub(n_miss.min(4));
    c.m.open = notes;
    c.m.summary = format!("A document written from {} facts that a program computed from the input files; every figure in it was checked against those facts and the files.", facts.lines().count());
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
        Kind::Report => make_report(&mut c, task, goal, workspace)?,
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
        assert_eq!(kind("Write a script that parses sales.csv\n\nInput files (in your workspace): sales.csv"), None);
        assert_eq!(kind("Using sales.csv, write a short report to output/report.md on revenue per region, with a table.\n\nInput files (in your workspace): sales.csv"), Some(Kind::Report));
        assert_eq!(kind("Convert sales.csv to JSON\n\nInput files (in your workspace): sales.csv"), None);
        assert_eq!(kind("Fix the bug in app.py\n\nInput files (in your workspace): app.py"), None);
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
        assert_eq!(ok.len(), 16);
        // What a real 3B model produced, and a first checklist let through: blue text on a blue card.
        let unreadable = good.replace("main{padding", "main{background:#007bff;padding").replace("h1{color:var(--accent)", "h1{color:#0069d9");
        let (_, miss) = page_checks("create a hello world sleek html page", &unreadable);
        assert!(miss.iter().any(|m| m.contains("hard to read")), "{miss:?}");
        // ...and a body that lays three blocks out side by side.
        let row = good.replace("display:grid;place-items:center", "display:flex").replace("<main><h1>Hello, World!</h1></main>", "<h1>Hello, World!</h1><div>a</div><div>b</div>");
        let (_, miss) = page_checks("create a hello world sleek html page", &row);
        assert!(miss.iter().any(|m| m.contains("ONE <main>")), "{miss:?}");
        // The design kit's own frame passes every check with any content and a weak accent.
        let kit = kit_page("TITLE: Hello\nACCENT: #ffee88\n```html\n<h1>Hello, World!</h1>\n<p>A small page.</p>\n```").unwrap();
        let (ok, miss) = page_checks("create a hello world sleek html page", &kit);
        assert!(miss.is_empty() && ok.len() == 16, "{miss:?}");
        assert!(kit.contains("--accent:#4f46e5"), "an accent too pale for white text is replaced");
        assert!(kit_page("```html\n<style>x</style><h1>a</h1>\n```").is_none());
        assert_eq!(kit_page("```html\n<main>\n<h1>Hi</h1>\n</main>\n```").unwrap().matches("<main").count(), 1, "a wrapper <main> is unwrapped");
        assert!(contrast([255.0, 255.0, 255.0], [0.0, 0.0, 0.0]) > 20.9 && contrast([0.0, 123.0, 255.0], [0.0, 105.0, 217.0]) < 1.5);
        assert_eq!(fenced(&format!("Here you go:\n```html\n{good}\n```\nEnjoy"), &["html"]).unwrap().trim(), good);
        assert_eq!(out_file("save it as output/hello.html please", "html?", "index.html"), "output/hello.html");
    }

    #[test]
    fn rankings_are_computed_and_wrong_ones_are_caught() {
        let facts = "Total Revenue: 6263.75\nEast: 1881.25\nNorth: 1440.00\nSouth: 1350.00\nWest: 1592.50";
        let (top, bottom) = ranking(facts);
        assert_eq!((top.clone().unwrap().0.as_str(), bottom.clone().unwrap().0.as_str()), ("East", "South"));
        // What Qwen-3B wrote with the right numbers in front of it.
        let bad = "The revenue breakdown shows that the West region leads with $1592.50, followed by East at $1881.25. The South region has the lowest revenue.";
        assert!(wrong_ranking(bad, Some("East"), Some("South")).unwrap().contains("highest is East"));
        let good = "East leads with $1881.25, followed by West. South has the lowest revenue of $1350.00.";
        assert!(wrong_ranking(good, Some("East"), Some("South")).is_none());
        assert!(wrong_ranking("South has the highest revenue.", Some("East"), Some("South")).is_some());
        assert_eq!(ranking("Total: 5\nA: 1"), (None, None));
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
