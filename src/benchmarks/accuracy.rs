//! `harness2 accuracy`: does the harness make *your* model more accurate?
//!
//! A built-in check anyone can run against any connected model. Questions with one exact
//! answer are generated from a seed (arithmetic, percentages, dates, counting, statistics, and
//! lookups and sums over CSV / JSON / log files), so the ground truth is computed here, never
//! guessed, and no model has seen the questions. Each one is asked three ways:
//!
//! ```text
//!   plain   the model answers directly, thinking step by step (how a chat window uses it)
//!   agent   one rollout of the harness with the accuracy boost off (the general agent loop)
//!   boost   one rollout with the accuracy boost on (program-aided answers, voted)
//!   learned the boost in a project that has first worked through other questions of the same
//!           kinds (another seed) and slept on them: what experience adds
//! ```
//!
//! and the report says how often each way was right, per kind of question, with timings.

use crate::benchmarks::goal::GoalAdapter;
use crate::core::spec::BenchmarkAdapter;
use crate::llm::client::{self, ChatReq, Msg};
use crate::llm::{connections, precise};
use crate::util;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum Expect {
    Number { value: f64, tol: f64 },
    /// Any of these (lowercase) appearing as a whole word or phrase.
    Text(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct Question {
    pub id: String,
    pub category: &'static str,
    pub prompt: String,
    pub files: Vec<(String, String)>,
    pub expect: Expect,
}

impl Question {
    pub fn expected(&self) -> String {
        match &self.expect {
            Expect::Number { value, .. } => crate::llm::calc::format(*value),
            Expect::Text(t) => t.join(" | "),
        }
    }
}

/// SplitMix64: small, seedable, the same everywhere.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Inclusive range.
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
    fn pick<'a, T>(&mut self, v: &'a [T]) -> &'a T {
        &v[(self.next() % v.len() as u64) as usize]
    }
    /// A price with two decimals, in quarter steps.
    fn price(&mut self, lo: i64, hi: i64) -> f64 {
        self.int(lo * 4, hi * 4) as f64 / 4.0
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0 + x.signum() * 1e-7).round() / 100.0
}

fn num(v: f64, tol: f64) -> Expect {
    Expect::Number { value: v, tol }
}

const REGIONS: &[&str] = &["North", "South", "East", "West", "Central"];
const PRODUCTS: &[&str] = &["Widget", "Gadget", "Gizmo", "Bracket", "Sprocket"];
const CITIES: &[&str] = &["Lyon", "Paris", "Nice", "Lille", "Nantes"];
const NAMES: &[&str] = &["Ana", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea"];
const WORDS: &[&str] = &[
    "strawberry", "raspberry", "cranberry", "blueberry", "mississippi", "committee", "bookkeeper", "banana", "parallel", "successful", "necessary", "occurrence", "assessment", "possession", "tomorrow", "terrace",
    "mirror", "pepper", "coffee", "balloon", "giraffe", "lettuce", "cabbage", "broccoli", "zucchini", "referee",
];
const SERVICES: &[&str] = &["auth", "billing", "search", "mail", "reports", "cache", "gateway", "uploads", "metrics"];

fn sales_csv(r: &mut Rng) -> (String, Vec<(String, String, i64, f64)>) {
    loop {
        let n = r.int(10, 14) as usize;
        let mut rows = vec![];
        for _ in 0..n {
            rows.push((r.pick(REGIONS).to_string(), r.pick(PRODUCTS).to_string(), r.int(20, 250), r.price(2, 40)));
        }
        // A clear winner, so "which region is highest" has one answer.
        let mut by: BTreeMap<&str, f64> = BTreeMap::new();
        for (reg, _, u, p) in &rows {
            *by.entry(reg.as_str()).or_default() += *u as f64 * p;
        }
        let mut totals: Vec<f64> = by.values().cloned().collect();
        totals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if totals.len() >= 3 && totals[0] > totals[1] * 1.03 {
            let body = rows.iter().map(|(a, b, u, p)| format!("{a},{b},{u},{p:.2}")).collect::<Vec<_>>().join("\n");
            return (format!("region,product,units,unit_price\n{body}\n"), rows);
        }
    }
}

fn access_log(r: &mut Rng) -> (String, Vec<(String, i64, i64)>) {
    let n = r.int(48, 64) as usize;
    let mut ids: Vec<i64> = vec![];
    while ids.len() < n {
        let id = r.int(1000, 9999);
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let mut rows = vec![];
    let mut lines = vec![];
    for (i, id) in ids.iter().enumerate() {
        let status = *r.pick(&[200i64, 200, 200, 200, 404, 500, 500, 302]);
        let ms = r.int(12, 980);
        lines.push(format!("2024-06-{:02} {:02}:{:02}:{:02} REQ-{id} GET /api/v{}/items/{} status={status} time_ms={ms}", 1 + i % 9, 9 + i / 30, i % 60, r.int(10, 59), r.int(1, 3), r.int(1, 500)));
        rows.push((format!("REQ-{id}"), status, ms));
    }
    (lines.join("\n") + "\n", rows)
}

/// One question of `category`, deterministic in (`seed`, `category`, `k`).
fn make(category: &'static str, seed: u64, k: usize) -> Question {
    let h = category.bytes().fold(0xcbf2_9ce4_8422_2325u64, |a, b| (a ^ b as u64).wrapping_mul(0x0100_0000_01b3));
    let mut r = Rng(seed ^ h ^ ((k as u64 + 1) << 32));
    let id = format!("{category}-{}", k + 1);
    let q = |prompt: String, files: Vec<(String, String)>, expect: Expect| Question { id: id.clone(), category, prompt, files, expect };
    match category {
        "arithmetic" => {
            let (start, a, b, pallets, per, damaged) = (r.int(900, 3200), r.int(120, 480), r.int(150, 520), r.int(2, 6), r.int(48, 180), r.int(7, 40));
            let ans = start - a - b + pallets * per - damaged;
            q(format!("A warehouse had {start} boxes. It shipped {a} on Monday and {b} on Tuesday, then received {pallets} pallets of {per} boxes each, and wrote off {damaged} damaged boxes. How many boxes does it have now?"), vec![], num(ans as f64, 0.001))
        }
        "percent" => {
            let (price, disc, tax) = (r.int(300, 2500), r.int(5, 35), r.int(40, 95) as f64 / 10.0);
            let ans = round2(price as f64 * (1.0 - disc as f64 / 100.0) * (1.0 + tax / 100.0));
            q(format!("A laptop costs ${price}. It is discounted by {disc}%, then a {tax}% sales tax is applied to the discounted price. What is the final price in dollars, to the cent?"), vec![], num(ans, 0.0051))
        }
        "compound" => {
            let (p, rate, years) = (r.int(8, 60) * 100, r.int(15, 65) as f64 / 10.0, r.int(3, 9));
            let ans = round2(p as f64 * (1.0 + rate / 100.0).powi(years as i32));
            q(format!("An account starts with ${p} and earns {rate}% interest per year, compounded yearly. What is the balance after {years} years, to the cent?"), vec![], num(ans, 0.0051))
        }
        "dates" => {
            let d0 = chrono::NaiveDate::from_ymd_opt(2023, 1, 1).unwrap() + chrono::Duration::days(r.int(0, 700));
            let span = r.int(40, 420);
            let d1 = d0 + chrono::Duration::days(span);
            if k % 2 == 0 {
                q(format!("How many days are there from {d0} to {d1}? (Count the difference between the two dates.)"), vec![], num(span as f64, 0.001))
            } else {
                q(format!("What is the date {span} days after {d0}? Answer as YYYY-MM-DD."), vec![], Expect::Text(vec![d1.to_string()]))
            }
        }
        "letters" => {
            let mut ws: Vec<&str> = vec![];
            while ws.len() < 3 {
                let w = *r.pick(WORDS);
                if !ws.contains(&w) {
                    ws.push(w);
                }
            }
            let phrase = ws.join(" ");
            // A letter that does occur, more than once.
            let mut letters: Vec<char> = phrase.chars().filter(|c| c.is_alphabetic() && phrase.matches(*c).count() >= 3).collect();
            if letters.is_empty() {
                letters = phrase.chars().filter(|c| c.is_alphabetic()).collect();
            }
            letters.sort();
            letters.dedup();
            let letter = *r.pick(&letters);
            q(format!("How many times does the letter '{letter}' appear in the phrase \"{phrase}\"?"), vec![], num(phrase.matches(letter).count() as f64, 0.001))
        }
        "statistics" => {
            let n = if k % 2 == 0 { 12 } else { 11 };
            let v: Vec<i64> = (0..n).map(|_| r.int(11, 96)).collect();
            let list = v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
            if k % 2 == 0 {
                let mut s = v.clone();
                s.sort();
                let med = (s[n / 2 - 1] + s[n / 2]) as f64 / 2.0;
                q(format!("What is the median of these numbers: {list}?"), vec![], num(med, 0.001))
            } else {
                let mean = round2(v.iter().sum::<i64>() as f64 / n as f64);
                q(format!("What is the mean of these numbers, rounded to 2 decimals: {list}?"), vec![], num(mean, 0.0051))
            }
        }
        "ordering" => {
            let base = r.int(2, 8) * 1000;
            let mut v: Vec<i64> = vec![];
            while v.len() < 7 {
                let x = base + r.int(100, 999);
                if !v.contains(&x) {
                    v.push(x);
                }
            }
            let kth = r.int(2, 4) as usize;
            let mut s = v.clone();
            s.sort_by(|a, b| b.cmp(a));
            let ord = ["", "", "second", "third", "fourth"][kth];
            q(format!("Which is the {ord}-largest of these numbers: {}?", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")), vec![], num(s[kth - 1] as f64, 0.001))
        }
        "units" => {
            let (km, h, m) = (r.int(60, 420), r.int(1, 4), r.int(5, 55));
            let ans = round2(km as f64 * 1000.0 / (h * 3600 + m * 60) as f64);
            q(format!("A car travels {km} km in {h} hour(s) {m} minutes. What is its average speed in metres per second, rounded to 2 decimals?"), vec![], num(ans, 0.0051))
        }
        "csv_total" => {
            let (csv, rows) = sales_csv(&mut r);
            let total: f64 = rows.iter().map(|(_, _, u, p)| *u as f64 * p).sum();
            q("Using sales.csv, what is the total revenue (units x unit_price) across all rows?".into(), vec![("sales.csv".into(), csv)], num(round2(total), 0.0051))
        }
        "csv_group" => {
            let (csv, rows) = sales_csv(&mut r);
            let mut by: BTreeMap<String, f64> = BTreeMap::new();
            for (reg, _, u, p) in &rows {
                *by.entry(reg.clone()).or_default() += *u as f64 * p;
            }
            if k % 2 == 0 {
                let top = by.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0.clone();
                q("Using sales.csv, which region has the highest total revenue (units x unit_price, summed per region)? Answer with the region name.".into(), vec![("sales.csv".into(), csv)], Expect::Text(vec![top.to_lowercase()]))
            } else {
                let prod = rows[0].1.clone();
                let units: i64 = rows.iter().filter(|x| x.1 == prod).map(|x| x.2).sum();
                q(format!("Using sales.csv, how many units of {prod} were sold in total across all regions?"), vec![("sales.csv".into(), csv)], num(units as f64, 0.001))
            }
        }
        "log_count" => {
            let (log, rows) = access_log(&mut r);
            let n = rows.iter().filter(|x| x.1 == 500).count();
            q("In access.log, how many requests have status=500?".into(), vec![("access.log".into(), log)], num(n as f64, 0.001))
        }
        "log_lookup" => {
            let (log, rows) = access_log(&mut r);
            let (rid, _, ms) = rows[rows.len() * 2 / 3 + k % 5].clone();
            q(format!("In access.log, what is the time_ms value of request {rid}?"), vec![("access.log".into(), log)], num(ms as f64, 0.001))
        }
        "json" => {
            let mut names: Vec<&str> = SERVICES.to_vec();
            r.shuffle(&mut names);
            let svcs: Vec<Value> = names.iter().take(r.int(5, 7) as usize).map(|n| json!({"name": n, "enabled": r.int(0, 2) > 0, "timeout_s": r.int(1, 24) * 5, "retries": r.int(0, 5)})).collect();
            let cfg = serde_json::to_string_pretty(&json!({"version": 3, "services": svcs})).unwrap();
            let sum: i64 = svcs.iter().filter(|s| s["enabled"] == true).map(|s| s["timeout_s"].as_i64().unwrap()).sum();
            q("In config.json, what is the sum of timeout_s over all services whose enabled is true?".into(), vec![("config.json".into(), cfg + "\n")], num(sum as f64, 0.001))
        }
        "join" => loop {
            let nc = r.int(8, 10) as usize;
            let custs: Vec<(String, &str, &str)> = (0..nc).map(|i| (format!("C{:03}", i + 1), NAMES[i], *r.pick(CITIES))).collect();
            let city = custs[r.int(0, nc as i64 - 1) as usize].2;
            let orders: Vec<(String, String, f64)> = (0..r.int(20, 28)).map(|i| (format!("O{:04}", i + 1), custs[r.int(0, nc as i64 - 1) as usize].0.clone(), r.int(1500, 42000) as f64 / 100.0)).collect();
            let in_city: Vec<&String> = custs.iter().filter(|c| c.2 == city).map(|c| &c.0).collect();
            let hits: Vec<f64> = orders.iter().filter(|o| in_city.contains(&&o.1)).map(|o| o.2).collect();
            if in_city.len() < 2 || hits.len() < 3 {
                continue;
            }
            let c_csv = format!("customer_id,name,city\n{}\n", custs.iter().map(|c| format!("{},{},{}", c.0, c.1, c.2)).collect::<Vec<_>>().join("\n"));
            let o_csv = format!("order_id,customer_id,amount\n{}\n", orders.iter().map(|o| format!("{},{},{:.2}", o.0, o.1, o.2)).collect::<Vec<_>>().join("\n"));
            break q(
                format!("Using orders.csv and customers.csv, what is the total order amount for customers whose city is {city}?"),
                vec![("orders.csv".into(), o_csv), ("customers.csv".into(), c_csv)],
                num(round2(hits.iter().sum()), 0.0051),
            );
        },
        other => unreachable!("unknown category {other}"),
    }
}

pub const CATEGORIES: &[&str] = &["arithmetic", "percent", "compound", "dates", "letters", "statistics", "ordering", "units", "csv_total", "csv_group", "log_count", "log_lookup", "json", "join"];

/// The question set for a seed: `per_category` questions of every kind.
pub fn questions(seed: u64, per_category: usize) -> Vec<Question> {
    let mut v = vec![];
    for k in 0..per_category.max(1) {
        for c in CATEGORIES {
            v.push(make(c, seed, k));
        }
    }
    v
}

/// The part of a reply or deliverable that states the answer: an `ANSWER:` line when there is
/// one, otherwise the text before the first horizontal rule (the engine's write-up puts how it
/// got there after one).
fn candidate(text: &str) -> String {
    if let Some(a) = precise::labelled_answer(text) {
        return a;
    }
    // LaTeX habit of maths-trained models: the result in \boxed{...}.
    if let Some(c) = regex::Regex::new(r"\\boxed\{([^{}]+)\}").unwrap().captures_iter(text).last() {
        return c[1].to_string();
    }
    // The harness's write-up: the answer, a rule, then how it was reached.
    if let Some((head, _)) = text.split_once("\n---") {
        return util::take_chars(head.trim(), 400);
    }
    let t = text.trim();
    if util::char_len(t) <= 400 {
        return t.to_string();
    }
    // A long free-form reply concludes at its end: the last paragraph.
    let tail: String = t.chars().rev().take(320).collect::<Vec<_>>().into_iter().rev().collect();
    tail.rsplit("\n\n").find(|p| !p.trim().is_empty()).unwrap_or(&tail).trim().to_string()
}

/// Is `text` a correct answer to `q`? The same rule for every mode.
pub fn check(q: &Question, text: &str) -> bool {
    let cand = candidate(text);
    match &q.expect {
        Expect::Number { value, tol } => precise::atoms(&cand).0.iter().any(|n| (n.v - value).abs() <= *tol),
        Expect::Text(ok) => {
            let low = cand.to_lowercase();
            ok.iter().any(|t| {
                low.match_indices(t.as_str()).any(|(i, _)| {
                    let before = low[..i].chars().next_back().map(|c| c.is_alphanumeric()).unwrap_or(false);
                    let after = low[i + t.len()..].chars().next().map(|c| c.is_alphanumeric()).unwrap_or(false);
                    !before && !after
                })
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchOpts {
    pub model: String,
    pub modes: Vec<String>,
    pub seed: u64,
    pub per_category: usize,
    /// Only these categories or question ids (empty: all).
    pub only: Vec<String>,
    pub out: PathBuf,
    pub samples: usize,
    pub agent_steps: u32,
    pub timeout: u64,
    pub escalate_model: String,
    /// Practice tasks in the sleep that precedes the `learned` mode.
    pub dreams: usize,
}

fn plain_prompt(q: &Question) -> String {
    let mut s = q.prompt.clone();
    for (name, body) in &q.files {
        s.push_str(&format!("\n\n--- {name} ---\n{}", util::take_chars(body, 14_000)));
    }
    s.push_str("\n\nThink step by step, then give the final answer on the last line as `ANSWER: <value>`.");
    s
}

/// One question, one mode: (the answer text, seconds, a note such as the engine's confidence).
fn ask(q: &Question, mode: &str, o: &BenchOpts) -> Result<(String, f64, Value)> {
    let t0 = Instant::now();
    match mode {
        "plain" => {
            let (conn, model) = connections::resolve(o.model.split(',').next().unwrap_or("").trim())?;
            let msgs = [Msg::User(plain_prompt(q))];
            let r = client::chat(&conn, &model, &ChatReq { system: "You are a careful assistant.", messages: &msgs, tools: &[], max_tokens: conn.max_tokens.unwrap_or(1024).min(1500), temperature: None }, Duration::from_secs(o.timeout))?;
            Ok((r.text, t0.elapsed().as_secs_f64(), json!({"tokens_out": r.output_tokens})))
        }
        "agent" | "boost" | "learned" => {
            let root = o.out.join(mode);
            // `learned` answers from the experience the learning phase left, and nothing newer:
            // every question sees the same store.
            let frozen = root.join("experience.frozen.json");
            if mode == "learned" && frozen.is_file() {
                std::fs::copy(&frozen, root.join("experience.json"))?;
            }
            let mut s = GoalAdapter::open(&root)?.settings;
            s.solver_model = o.model.clone();
            s.improver_model = o.model.clone();
            s.selector_model = o.model.clone();
            s.solver_steps = o.agent_steps;
            s.solver_timeout = o.timeout;
            s.accuracy = if mode == "agent" { "off".into() } else { "on".into() };
            s.accuracy_samples = o.samples;
            s.escalate_model = if mode == "agent" { String::new() } else { o.escalate_model.clone() };
            // Each question stands alone: nothing remembered from the one before. Only the
            // `learned` mode reads (and, while learning, writes) the project's experience.
            s.memory_enabled = false;
            s.experience = mode == "learned";
            let adapter = GoalAdapter::with_settings(&root, s)?;
            let files: Vec<(String, Vec<u8>)> = q.files.iter().map(|(n, b)| (n.clone(), b.clone().into_bytes())).collect();
            let id = adapter.add_task(&q.prompt, "accuracy", "", &files)?;
            let rd = adapter.execute(&id, None, &format!("accuracy/{}/{id}", q.id), None)?;
            let mut text = adapter.deliverable_text(&rd).unwrap_or_default();
            if text.trim().is_empty() {
                text = std::fs::read_to_string(rd.join("final.md")).unwrap_or_default();
            }
            let usage: Value = std::fs::read_to_string(rd.join("usage.json")).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
            Ok((text, t0.elapsed().as_secs_f64(), json!({"steps": usage["steps"], "accuracy": usage["accuracy"], "error": usage["error"], "run_dir": rd.display().to_string()})))
        }
        m => bail!("unknown mode {m:?}: use plain, agent, boost or learned"),
    }
}

/// Run the benchmark (resuming from `<out>/results.jsonl`), print and save the report.
pub fn run(o: &BenchOpts) -> Result<Value> {
    std::fs::create_dir_all(&o.out)?;
    let wanted = |q: &Question| o.only.is_empty() || o.only.iter().any(|x| x == q.category || *x == q.id);
    let qs: Vec<Question> = questions(o.seed, o.per_category).into_iter().filter(wanted).collect();
    if qs.is_empty() {
        bail!("no questions match --only {:?}; categories: {}", o.only, CATEGORIES.join(", "));
    }
    let results_path = o.out.join("results.jsonl");
    let mut done: BTreeMap<(String, String), Value> = BTreeMap::new();
    for l in std::fs::read_to_string(&results_path).unwrap_or_default().lines() {
        if let Ok(v) = serde_json::from_str::<Value>(l) {
            if v["seed"].as_u64() == Some(o.seed) && v["model"] == json!(o.model) {
                done.insert((v["mode"].as_str().unwrap_or("").to_string(), v["id"].as_str().unwrap_or("").to_string()), v);
            }
        }
    }
    crate::say!("accuracy check: {} question(s) × {} mode(s) on {} (seed {})", qs.len(), o.modes.len(), o.model, o.seed);
    let mut training = Value::Null;
    for mode in &o.modes {
        if mode == "learned" {
            training = learn(o)?;
        }
        for q in &qs {
            if crate::log::cancelled() {
                bail!("cancelled by user");
            }
            let key = (mode.clone(), q.id.clone());
            if done.contains_key(&key) {
                continue;
            }
            let row = match ask(q, mode, o) {
                Ok((text, secs, note)) => {
                    let ok = check(q, &text);
                    crate::say!("  {} {:<6} {:<13} {:>5.0}s  got {:?}  expected {}", if ok { "✔" } else { "✘" }, mode, q.id, secs, util::take_chars(&candidate(&text).replace('\n', " "), 70), q.expected());
                    json!({"model": o.model, "seed": o.seed, "mode": mode, "id": q.id, "category": q.category, "ok": ok, "seconds": (secs * 10.0).round() / 10.0, "answer": util::take_chars(&candidate(&text), 300), "expected": q.expected(), "note": note, "text": util::take_chars(&text, 6000)})
                }
                Err(e) => {
                    if format!("{e}").contains("cancelled") {
                        bail!("cancelled by user");
                    }
                    crate::say!("  ✘ {:<6} {:<13} failed: {}", mode, q.id, util::take_chars(&format!("{e}"), 160));
                    json!({"model": o.model, "seed": o.seed, "mode": mode, "id": q.id, "category": q.category, "ok": false, "seconds": 0.0, "answer": "", "expected": q.expected(), "error": util::take_chars(&format!("{e}"), 400)})
                }
            };
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&results_path)?;
            use std::io::Write;
            writeln!(f, "{row}")?;
            done.insert(key, row);
        }
    }
    let mut report = summarise(o, &qs, &done);
    if !training.is_null() {
        report["training"] = training;
    }
    util::write(&o.out.join("report.json"), serde_json::to_string_pretty(&report)?)?;
    let md = render(&report);
    util::write(&o.out.join("report.md"), &md)?;
    crate::say!("\n{md}\nSaved to {}", o.out.join("report.md").display());
    Ok(report)
}

/// The learning phase before the `learned` mode: one question of every kind from another seed,
/// worked through with the boost (verified solutions become experience), then one sleep.
fn learn(o: &BenchOpts) -> Result<Value> {
    let root = o.out.join("learned");
    let marker = root.join("trained.json");
    if let Some(v) = std::fs::read_to_string(&marker).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) {
        return Ok(v);
    }
    let train_seed = o.seed + 1000;
    let qs: Vec<Question> = questions(train_seed, 1).into_iter().filter(|q| o.only.is_empty() || o.only.iter().any(|x| x == q.category || x.starts_with(&format!("{}-", q.category)))).collect();
    crate::say!("learning phase: {} other question(s) (seed {train_seed}), then a sleep", qs.len());
    let mut right = 0;
    for q in &qs {
        if crate::log::cancelled() {
            bail!("cancelled by user");
        }
        match ask(q, "learned", o) {
            Ok((text, secs, _)) => {
                let ok = check(q, &text);
                right += ok as usize;
                crate::say!("  {} learn  {:<13} {:>5.0}s", if ok { "✔" } else { "✘" }, q.id, secs);
            }
            Err(e) => crate::say!("  ✘ learn  {:<13} failed: {}", q.id, util::take_chars(&format!("{e}"), 120)),
        }
    }
    let mut so = crate::llm::sleep::SleepOpts::new(&o.model);
    so.dreams = o.dreams;
    so.explore = 0;
    so.seed = o.seed;
    let journal = crate::llm::sleep::sleep(&root, &so)?;
    let store = root.join("experience.json");
    if store.is_file() {
        std::fs::copy(&store, root.join("experience.frozen.json"))?;
    }
    let examples = crate::llm::experience::Store::open(&root).load().iter().filter(|e| e.status == "active").count();
    let v = json!({
        "seed": train_seed, "asked": qs.len(), "right": right, "examples": examples,
        "sleep": {"rehearsed": journal["rehearsed"], "right_cold": journal["right_cold"], "right_with_experience": journal["right_with_experience"], "helped": journal["helped"], "no_help": journal["no_help"], "skipped": journal["skipped"], "consolidated": journal["consolidated"]},
    });
    util::write(&marker, serde_json::to_string_pretty(&v)?)?;
    Ok(v)
}

fn summarise(o: &BenchOpts, qs: &[Question], done: &BTreeMap<(String, String), Value>) -> Value {
    let mut modes = vec![];
    for mode in &o.modes {
        let rows: Vec<&Value> = qs.iter().filter_map(|q| done.get(&(mode.clone(), q.id.clone()))).collect();
        let ok = rows.iter().filter(|r| r["ok"] == true).count();
        let secs: f64 = rows.iter().map(|r| r["seconds"].as_f64().unwrap_or(0.0)).sum();
        let mut by_cat: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for q in qs {
            if let Some(r) = done.get(&(mode.clone(), q.id.clone())) {
                let e = by_cat.entry(q.category).or_default();
                e.1 += 1;
                if r["ok"] == true {
                    e.0 += 1;
                }
            }
        }
        modes.push(json!({
            "mode": mode, "correct": ok, "asked": rows.len(), "accuracy": if rows.is_empty() { 0.0 } else { ok as f64 / rows.len() as f64 },
            "seconds": secs.round(), "avg_seconds": if rows.is_empty() { 0.0 } else { (secs / rows.len() as f64 * 10.0).round() / 10.0 },
            "by_category": by_cat.iter().map(|(c, (k, n))| json!({"category": c, "correct": k, "asked": n})).collect::<Vec<_>>(),
        }));
    }
    let rows: Vec<Value> = qs
        .iter()
        .map(|q| {
            let mut r = json!({"id": q.id, "category": q.category, "question": q.prompt, "expected": q.expected()});
            for mode in &o.modes {
                if let Some(d) = done.get(&(mode.clone(), q.id.clone())) {
                    r[mode.as_str()] = json!({"ok": d["ok"], "answer": d["answer"], "seconds": d["seconds"], "note": d["note"]});
                }
            }
            r
        })
        .collect();
    json!({"model": o.model, "seed": o.seed, "questions": qs.len(), "samples": o.samples, "at": util::now_iso(), "modes": modes, "rows": rows})
}

/// The report as Markdown: totals, then one line per kind of question.
pub fn render(report: &Value) -> String {
    let modes: Vec<&Value> = report["modes"].as_array().map(|a| a.iter().collect()).unwrap_or_default();
    let mut s = format!("# Accuracy check — {}\n\n{} questions with exact answers (seed {}).\n\n", report["model"].as_str().unwrap_or(""), report["questions"], report["seed"]);
    s.push_str("| mode | correct | accuracy | avg time |\n|---|---|---|---|\n");
    for m in &modes {
        s.push_str(&format!("| {} | {} / {} | {:.0}% | {:.0}s |\n", m["mode"].as_str().unwrap_or(""), m["correct"], m["asked"], m["accuracy"].as_f64().unwrap_or(0.0) * 100.0, m["avg_seconds"].as_f64().unwrap_or(0.0)));
    }
    let cats: Vec<String> = modes.first().and_then(|m| m["by_category"].as_array()).map(|a| a.iter().filter_map(|c| c["category"].as_str().map(String::from)).collect()).unwrap_or_default();
    if !cats.is_empty() {
        s.push_str(&format!("\n| kind of question | {} |\n|---|{}\n", modes.iter().map(|m| m["mode"].as_str().unwrap_or("").to_string()).collect::<Vec<_>>().join(" | "), "---|".repeat(modes.len())));
        for c in &cats {
            let cells: Vec<String> = modes
                .iter()
                .map(|m| m["by_category"].as_array().and_then(|a| a.iter().find(|x| x["category"] == json!(c))).map(|x| format!("{} / {}", x["correct"], x["asked"])).unwrap_or_else(|| "–".into()))
                .collect();
            s.push_str(&format!("| {c} | {} |\n", cells.join(" | ")));
        }
    }
    let t = &report["training"];
    if t.is_object() {
        s.push_str(&format!(
            "\n`learned` first worked through {} other questions (seed {}; {} right), which left {} worked example(s), then slept: {} practice task(s), {} right without help and {} more with its experience.\n",
            t["asked"], t["seed"], t["right"], t["examples"], t["sleep"]["rehearsed"], t["sleep"]["right_cold"], t["sleep"]["helped"]
        ));
    }
    s
}

/// The reports saved under `root` (one folder per run), newest first.
pub fn saved_reports(root: &Path) -> Vec<Value> {
    let mut v: Vec<Value> = std::fs::read_dir(root)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let mut r: Value = serde_json::from_str(&std::fs::read_to_string(e.path().join("report.json")).ok()?).ok()?;
                    r["name"] = json!(e.file_name().to_string_lossy());
                    Some(r)
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort_by(|a, b| b["at"].as_str().unwrap_or("").cmp(a["at"].as_str().unwrap_or("")));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questions_are_deterministic_and_answerable() {
        let a = questions(7, 2);
        let b = questions(7, 2);
        assert_eq!(a.len(), CATEGORIES.len() * 2);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.prompt, y.prompt);
            assert_eq!(x.files, y.files);
            assert_eq!(x.expected(), y.expected());
        }
        assert_ne!(questions(8, 1)[0].prompt, a[0].prompt, "another seed gives other questions");
        // the expected answer, written the way a model would, passes its own check
        for q in &a {
            let e = q.expected();
            let first = e.split(" | ").next().unwrap();
            assert!(check(q, &format!("Some reasoning with 17 and 3.5 in it.\nANSWER: {first}")), "{}: {e}", q.id);
            assert!(check(q, &format!("{first}\n\n---\n\n*Checked: 3 of 3 independent attempts gave this answer.*")), "{}: {e}", q.id);
            assert!(!check(q, "ANSWER: 123456789"), "{}", q.id);
            assert!(!check(q, ""), "{}", q.id);
        }
    }

    #[test]
    fn generated_answers_match_an_independent_computation() {
        // Recompute a few kinds from the generated files themselves.
        for seed in [1u64, 2, 3] {
            let qs = questions(seed, 2);
            for q in qs.iter().filter(|q| q.category == "csv_total") {
                let total: f64 = q.files[0].1.lines().skip(1).map(|l| l.split(',').collect::<Vec<_>>()).map(|c| c[2].parse::<f64>().unwrap() * c[3].parse::<f64>().unwrap()).sum();
                assert!(check(q, &format!("ANSWER: {total:.2}")), "{} {total}", q.id);
            }
            for q in qs.iter().filter(|q| q.category == "log_count") {
                let n = q.files[0].1.lines().filter(|l| l.contains("status=500")).count();
                assert!(check(q, &format!("ANSWER: {n}")), "{}", q.id);
            }
            for q in qs.iter().filter(|q| q.category == "letters") {
                let phrase = q.prompt.split('"').nth(1).unwrap();
                let letter = q.prompt.split('\'').nth(1).unwrap();
                assert!(check(q, &format!("ANSWER: {}", phrase.matches(letter).count())), "{}", q.id);
            }
        }
    }

    #[test]
    fn checking_reads_the_stated_answer_only() {
        let q = Question { id: "x".into(), category: "arithmetic", prompt: String::new(), files: vec![], expect: num(833.0, 0.001) };
        assert!(check(&q, "1250 - 387 = 863, then ... \nANSWER: 833 boxes"));
        assert!(!check(&q, "The total is 833 at first glance.\nANSWER: 849"), "an ANSWER line overrides numbers elsewhere");
        assert!(check(&q, "**833**\n\n---\n\n*Checked: 2 of 3 attempts.*"));
        assert!(!check(&q, "849\n\n---\n\n*Other answers: 833*"), "what follows the rule is not the answer");
        let t = Question { expect: Expect::Text(vec!["east".into()]), ..q.clone() };
        assert!(check(&t, "ANSWER: East"));
        assert!(check(&t, "The region is East."));
        assert!(!check(&t, "ANSWER: Northeast"));
        assert!(!check(&t, "ANSWER: West"));
        // a long step-by-step reply is read at its conclusion, however it labels it
        let long = format!("Step 1: 1250 - 387 = 863.\n{}\n\nSo the warehouse now has 833 boxes.", "More working with 849 and 17.\n".repeat(30));
        assert!(check(&q, &long));
        let wrong = format!("Step 1: the total is 833 so far.\n{}\n\nSo the warehouse now has 849 boxes.", "More working.\n".repeat(40));
        assert!(!check(&q, &wrong), "numbers in the working are not the answer");
        assert!(check(&q, &format!("{}\n\nThus \\[ \\boxed{{833}} \\]", "working\n".repeat(80))));
    }
}
