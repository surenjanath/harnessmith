//! The accuracy boost end to end, against scripted OpenAI-compatible servers that play a small
//! model: program-aided answers with voting, repairs, precision, escalation, experience (worked
//! examples), sleep (consolidate and dream), the accuracy check, and the agent loop's `python`
//! tool with the grounding check on figures.

use harness2::benchmarks::accuracy::{self, BenchOpts};
use harness2::benchmarks::goal::{GoalAdapter, GoalSettings};
use harness2::core::spec::BenchmarkAdapter;
use harness2::llm::{experience, sleep};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

type Script = Arc<dyn Fn(usize, &Value) -> Value + Send + Sync>;

/// A tiny HTTP server; `script(n, request)` answers the n-th request (1-based).
fn server(script: Script) -> (u16, Arc<Mutex<Vec<Value>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(vec![]));
    let s2 = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut len = 0usize;
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0; len];
            let _ = r.read_exact(&mut body);
            let req: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
            let n = {
                let mut g = s2.lock().unwrap();
                g.push(req.clone());
                g.len()
            };
            let b = script(n, &req).to_string();
            let _ = write!(s, "HTTP/1.1 200 X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}", b.len());
        }
    });
    (port, seen)
}

fn text(t: &str) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": t}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
}

fn tool(name: &str, args: Value) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": args.to_string()}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
}

fn program(code: &str) -> Value {
    text(&format!("```python\n{code}\n```"))
}

/// The last message of a request, as text.
fn last(req: &Value) -> String {
    req["messages"].as_array().and_then(|m| m.last()).and_then(|m| m["content"].as_str()).unwrap_or("").to_string()
}

fn conn(name: &str, port: u16) -> Value {
    json!({"name": name, "provider": "openai", "base_url": format!("http://127.0.0.1:{port}/v1"), "default_model": "m"})
}

fn project(root: &Path, solver: &str, accuracy: &str) -> GoalAdapter {
    let mut s = GoalSettings::default();
    s.solver_model = solver.into();
    s.improver_model = solver.into();
    s.selector_model = solver.into();
    s.accuracy = accuracy.into();
    s.sandbox = "off".into();
    s.memory_enabled = false;
    s.solver_steps = 12;
    GoalAdapter::with_settings(root, s).unwrap()
}

fn solve(a: &GoalAdapter, goal: &str, files: &[(&str, &str)], tag: &str) -> (PathBuf, String) {
    let files: Vec<(String, Vec<u8>)> = files.iter().map(|(n, b)| (n.to_string(), b.as_bytes().to_vec())).collect();
    let id = a.add_task(goal, "general", "", &files).unwrap();
    let rd = a.execute(&id, None, &format!("t/{tag}"), None).unwrap();
    let out = a.deliverable_text(&rd).unwrap_or_default();
    (rd, out)
}

fn read_json(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_default()).unwrap_or(Value::Null)
}

const SALES: &str = "region,product,units,unit_price\nNorth,Widget,120,4.50\nSouth,Gadget,40,12.50\nEast,Widget,95,4.75\nWest,Gadget,110,11.75\nEast,Gadget,130,11.00\nNorth,Gadget,75,12.00\n";
const TOTAL: &str = "import csv\nrows = list(csv.DictReader(open(\"sales.csv\")))\ntotal = 0\nfor row in rows:\n    total += float(row[\"units\"]) * float(row[\"unit_price\"])\nprint(round(total, 2))";
const Q_TOTAL: &str = "Using sales.csv, what is the total revenue (units x unit_price) across all rows?";

/// Every scenario shares HARNESS2_CONNECTIONS, so they run in one test.
#[test]
fn accuracy_boost_scenarios() {
    if !harness2::llm::pycode::available() {
        eprintln!("python3 not found: skipping");
        return;
    }
    // A: every attempt writes the right program.
    let (pa, seen_a) = server(Arc::new(|_, req: &Value| if last(req).contains("Write the Python program") { program(TOTAL) } else { text("ANSWER: 0") }));
    // B: a broken program, then one that rounds too far; the second attempt is exact.
    let (pb, seen_b) = server(Arc::new(|n, req: &Value| match n {
        1 => program("print(total)"),
        2 => {
            assert!(last(req).contains("NameError") && last(req).contains("Fix it"), "the repair request carries the error: {}", last(req));
            program("price = 1429 * (1 - 0.19) * (1 + 0.047)\nprint(round(price))")
        }
        _ => program("price = 1429 * (1 - 0.19) * (1 + 0.047)\nprint(round(price, 2))"),
    }));
    // C: three attempts, three answers, and a tie-break that agrees with none.
    let (pc, _) = server(Arc::new(|n, req: &Value| {
        if last(req).contains("Think step by step") {
            return text("Counting month by month...\nANSWER: 270");
        }
        program(&format!("print({})", [261, 263, 265][(n - 1) % 3]))
    }));
    // The stronger model the project may escalate to.
    let (pbig, seen_big) = server(Arc::new(|_, _| program("from datetime import date\nprint((date(2024, 11, 5) - date(2024, 2, 17)).days)")));
    // E: during sleep, wrong when alone, right when shown the worked example.
    let (pe, _) = server(Arc::new(|_, req: &Value| if last(req).contains("Solved earlier in this project") { program(TOTAL) } else { program("print(1)") }));
    // F: an oracle for the accuracy check (filled in below).
    let answers: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(vec![]));
    let a2 = answers.clone();
    let (pf, _) = server(Arc::new(move |_, req: &Value| {
        let msg = last(req);
        let known = a2.lock().unwrap();
        let Some((_, ans)) = known.iter().find(|(q, _)| msg.contains(q.as_str())) else { return text("ANSWER: unknown") };
        if msg.contains("Think step by step") { text(&format!("Working...\nANSWER: {ans}")) } else { program(&format!("print({ans:?})")) }
    }));
    // G: the agent loop: a made-up figure, sent back, computed with the python tool.
    let (pg, seen_g) = server(Arc::new(|n, req: &Value| match n {
        1 => tool("write_file", json!({"path": "output/answer.md", "content": "Revenue by region\n\nTotal revenue is $321.875.\n"})),
        2 => tool("finish", json!({"answer": "done"})),
        3 => {
            assert!(last(req).contains("321.875") && last(req).contains("python"), "finish was refused over the made-up figure: {}", last(req));
            tool("python", json!({"code": TOTAL}))
        }
        4 => {
            assert!(last(req).contains("5113.75"), "the python tool returned the total: {}", last(req));
            tool("write_file", json!({"path": "output/answer.md", "content": "Revenue by region\n\nTotal revenue is $5,113.75.\n"}))
        }
        _ => tool("finish", json!({"answer": "done"})),
    }));
    // H: a fenced Python block instead of a tool call is run.
    let (ph, _) = server(Arc::new(|n, req: &Value| match n {
        1 => text("I will compute it.\n```python\nprint(6 * 7)\n```"),
        2 => {
            assert!(last(req).contains("42"), "the block's output came back: {}", last(req));
            tool("write_file", json!({"path": "output/answer.md", "content": "Six sevens make 42.\n"}))
        }
        _ => text("Done."),
    }));

    let d = tempfile::tempdir().unwrap();
    let conns = json!({"connections": [conn("a", pa), conn("b", pb), conn("c", pc), conn("big", pbig), conn("e", pe), conn("f", pf), conn("g", pg), conn("h", ph)]});
    let cp = d.path().join("connections.json");
    std::fs::write(&cp, conns.to_string()).unwrap();
    std::env::set_var("HARNESS2_CONNECTIONS", &cp);
    std::env::set_var("HARNESS2_RUNS_DIR", d.path().join("runs"));
    harness2::log::set_quiet(true);

    // ---- A: two attempts agree, the answer is delivered with its provenance, and kept as experience.
    let root = d.path().join("proj");
    let a = project(&root, "a/m", "on");
    let (rd, out) = solve(&a, Q_TOTAL, &[("sales.csv", SALES)], "a1");
    assert!(out.starts_with("5113.75"), "{out}");
    assert!(out.contains("2 of 2 independent attempts gave this answer"), "{out}");
    assert!(out.contains("csv.DictReader"), "the program is shown: {out}");
    assert_eq!(seen_a.lock().unwrap().len(), 2, "two agreeing attempts are enough");
    let acc = read_json(&rd.join("accuracy.json"));
    assert_eq!((acc["confidence"].as_str(), acc["agree"].as_u64()), (Some("high"), Some(2)));
    assert_eq!(read_json(&rd.join("usage.json"))["accuracy"]["agree"], 2);
    let store = experience::Store::open(&root);
    assert_eq!(store.load().len(), 1, "the verified solution became a worked example");
    // The first request named the real file and its columns.
    let first = seen_a.lock().unwrap()[0].clone();
    let sys = first["messages"][0]["content"].as_str().unwrap().to_string();
    assert!(sys.contains(r#"csv.DictReader(open("sales.csv"))"#) && sys.contains("region,product,units,unit_price"), "{sys}");

    // ---- D: the same kind of question again gets the example, and the store still has one entry.
    let before = seen_a.lock().unwrap().len();
    let (_, out) = solve(&a, Q_TOTAL, &[("sales.csv", &SALES.replace("120,4.50", "200,4.50"))], "a2");
    assert!(out.starts_with("5473.75"), "{out}");
    let req = seen_a.lock().unwrap()[before].clone();
    assert!(last(&req).contains("Solved earlier in this project") && last(&req).contains("It printed: 5113.75"), "{}", last(&req));
    let all = store.load();
    assert_eq!(all.iter().filter(|e| e.status == "active").count(), 1);
    assert_eq!(all[0].uses, 1);

    // ---- B: repair, then an over-rounded answer; the precise form the question asks for wins.
    let b = project(&d.path().join("proj_b"), "b/m", "on");
    let (rd, out) = solve(&b, "A laptop costs $1429. It is discounted by 19%, then a 4.7% sales tax is applied to the discounted price. What is the final price in dollars, to the cent?", &[], "b1");
    assert!(out.starts_with("1211.89"), "{out}");
    assert_eq!(read_json(&rd.join("accuracy.json"))["samples"][0]["repairs"], 1);
    assert_eq!(seen_b.lock().unwrap().len(), 3);

    // ---- C: no agreement. Without a stronger model: the best guess, flagged. With one: settled.
    let q_days = "How many days are there from 2024-02-17 to 2024-11-05?";
    let c = project(&d.path().join("proj_c"), "c/m", "on");
    let (rd, out) = solve(&c, q_days, &[], "c1");
    assert!(out.starts_with("261"), "{out}");
    assert!(out.contains("Low confidence"), "{out}");
    assert_eq!(read_json(&rd.join("accuracy.json"))["confidence"], "low");
    assert!(experience::Store::open(&d.path().join("proj_c")).load().is_empty(), "a split vote is not experience");
    let mut c2 = project(&d.path().join("proj_c2"), "c/m", "on");
    c2.settings.escalate_model = "big/m".into();
    let c2 = GoalAdapter::with_settings(&d.path().join("proj_c2"), c2.settings.clone()).unwrap();
    let (rd, out) = solve(&c2, q_days, &[], "c2");
    assert!(out.starts_with("262"), "{out}");
    assert!(out.contains("stronger model"), "{out}");
    let acc = read_json(&rd.join("accuracy.json"));
    assert_eq!((acc["escalated"].as_bool(), acc["confidence"].as_str()), (Some(true), Some("high")));
    assert_eq!(seen_big.lock().unwrap().len(), 1, "the stronger model is asked once");

    // ---- accuracy off: the same question goes to the agent loop (which here just fails to act).
    assert!(!harness2::llm::precise::enabled("off", "a/m"));
    assert!(harness2::llm::precise::enabled("auto", "a/m"), "a local server counts as a small model");
    assert!(!harness2::llm::precise::enabled("auto", "mock/mock-1"));

    // ---- E: sleep. The example is rehearsed on changed data: wrong alone, right with it.
    let mut so = sleep::SleepOpts::new("e/m");
    so.dreams = 3;
    so.explore = 0;
    so.sandbox = "off".into();
    so.seed = 3;
    let j = sleep::sleep(&root, &so).unwrap();
    assert_eq!((j["rehearsed"].as_u64(), j["right_cold"].as_u64(), j["helped"].as_u64()), (Some(1), Some(0), Some(1)), "{j}");
    let dream = &j["dreams"][0];
    assert_eq!(dream["verdict"], "helped");
    assert_ne!(dream["truth"], "5473.75", "the practice data differs from the original: {dream}");
    assert_eq!(dream["cold"], "1");
    assert_eq!(dream["with"], dream["truth"]);
    let e = &store.load()[0];
    assert_eq!((e.dreams, e.helped, e.cold_ok, e.failed), (1, 1, 0, 0));
    assert!(e.value() > 0.9);
    assert!(root.join("sleep/latest.json").is_file() && !root.join("sleep/tmp").exists());
    assert_eq!(sleep::journals(&root, 5).len(), 1);

    // ---- F: the accuracy check, wired end to end with a model that knows the answers.
    let qs: Vec<accuracy::Question> = accuracy::questions(5, 1).into_iter().filter(|q| ["arithmetic", "csv_total", "dates"].contains(&q.category)).collect();
    *answers.lock().unwrap() = qs.iter().map(|q| (q.prompt.clone(), q.expected())).collect();
    let o = BenchOpts {
        model: "f/m".into(),
        modes: vec!["plain".into(), "boost".into()],
        seed: 5,
        per_category: 1,
        only: vec!["arithmetic".into(), "csv_total".into(), "dates".into()],
        out: d.path().join("check"),
        samples: 3,
        agent_steps: 8,
        timeout: 120,
        escalate_model: String::new(),
        dreams: 2,
    };
    let rep = accuracy::run(&o).unwrap();
    for m in rep["modes"].as_array().unwrap() {
        assert_eq!((m["correct"].as_u64(), m["asked"].as_u64()), (Some(3), Some(3)), "{m}");
    }
    let md = std::fs::read_to_string(d.path().join("check/report.md")).unwrap();
    assert!(md.contains("| plain | 3 / 3 | 100% |") && md.contains("| boost | 3 / 3 | 100% |"), "{md}");
    // Running it again answers nothing twice.
    assert_eq!(accuracy::run(&o).unwrap()["modes"][1]["correct"], 3);
    assert_eq!(accuracy::saved_reports(d.path()).len(), 1);

    // ---- G: a goal that is not one exact question runs on the agent loop, where a figure the
    // agent never computed is sent back, and the python tool supplies the real one.
    let g = project(&d.path().join("proj_g"), "g/m", "on");
    let (rd, out) = solve(&g, "Using sales.csv, write a few lines on revenue per region to output/answer.md.", &[("sales.csv", SALES)], "g1");
    assert!(out.contains("$5,113.75") && !out.contains("321.875"), "{out}");
    assert_eq!(seen_g.lock().unwrap().len(), 5);
    let traj = std::fs::read_to_string(rd.join("trajectory.jsonl")).unwrap();
    assert!(traj.contains("ungrounded_figures"), "the refusal is recorded as a self-healing action");
    assert_eq!(read_json(&rd.join("usage.json"))["review_open"], json!([]));
    // The task showed the small input file, and the tools included python and calc.
    let first = seen_g.lock().unwrap()[0].clone();
    assert!(first["messages"][1]["content"].as_str().unwrap().contains("North,Widget,120,4.50"), "small inputs are shown with the task");
    let names: Vec<&str> = first["tools"].as_array().unwrap().iter().filter_map(|t| t["function"]["name"].as_str()).collect();
    assert!(names.contains(&"python") && names.contains(&"calc"), "{names:?}");
    // With the boost off, the made-up figure would have gone through unchallenged, and the
    // tools are the ones from before.
    let off = project(&d.path().join("proj_off"), "a/m", "off");
    let _ = solve(&off, "Using sales.csv, write a few lines on revenue per region to output/answer.md.", &[("sales.csv", SALES)], "off1");
    let req = seen_a.lock().unwrap().last().cloned().unwrap();
    let names: Vec<&str> = req["tools"].as_array().unwrap().iter().filter_map(|t| t["function"]["name"].as_str()).collect();
    assert!(!names.contains(&"python") && !names.contains(&"calc"), "{names:?}");

    // ---- I: a full goal run. The first attempt's answer is verified, so it is adopted and the
    // improvement loop (improver sessions, more rollouts, a selector) never starts.
    let before = seen_a.lock().unwrap().len();
    let full = project(&d.path().join("proj_full"), "a/m", "on");
    let id = full.add_task(Q_TOTAL, "general", "", &[("sales.csv".to_string(), SALES.as_bytes().to_vec())]).unwrap();
    let res = harness2::cli::run_goal(&full, &id, 1, vec![harness2::core::schedule::mode_token("parallel").unwrap()]).unwrap();
    assert_eq!(res["status"], "ok", "{res}");
    let sel = &res["selections"]["agg"];
    assert_eq!((sel["chosen_label"].as_str(), sel["settled"].as_bool(), sel["chosen_is_base"].as_bool()), (Some("base1"), Some(true), Some(true)), "{sel}");
    assert!(sel["why"].as_str().unwrap().contains("verified"), "{sel}");
    assert!(res["deliverables"]["agg"]["text"].as_str().unwrap().starts_with("5113.75"));
    assert_eq!(seen_a.lock().unwrap().len() - before, 2, "two engine attempts and nothing else");
    // Running it again resumes from the summary instead of redoing anything.
    let again = harness2::cli::run_goal(&full, &id, 1, vec![harness2::core::schedule::mode_token("parallel").unwrap()]).unwrap();
    assert_eq!(again["selections"]["agg"]["chosen_label"], "base1");
    assert_eq!(seen_a.lock().unwrap().len() - before, 2, "a second run reuses the first");

    // ---- J: a page. The first draft is bare; the checklist sends it back; the fix is kept.
    const GOOD_PAGE: &str = "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Hello</title><style>:root{--bg:#0f172a;--fg:#e2e8f0;--accent:#38bdf8}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;font-family:system-ui,sans-serif;background:linear-gradient(135deg,var(--bg),#1e293b);color:var(--fg)}main{padding:clamp(2rem,6vw,4rem);border-radius:24px;box-shadow:0 20px 60px rgba(0,0,0,.4);background:rgba(255,255,255,.05);transition:transform .3s}main:hover{transform:translateY(-4px)}h1{color:var(--accent)}</style></head><body><main><h1>Hello, World!</h1></main></body></html>";
    let (pj, seen_j) = server(Arc::new(|n, req: &Value| {
        if n == 1 {
            return text("```html\n<html><body><h1>Hello, world!</h1></body></html>\n```");
        }
        assert!(last(req).contains("Use exactly the form"), "the form is asked for again: {}", last(req));
        text("TITLE: Hello\nACCENT: #0369a1\n```html\n<h1>Hello, World!</h1>\n<p>A sleek little page.</p>\n```")
    }));
    // ---- K: a coding task. The first candidate is wrong; the tests pick a right one.
    let (pk, _) = server(Arc::new(|n, req: &Value| {
        if last(req).contains("write tests") {
            return text("```python\nassert is_palindrome(\"Racecar\") == True\nassert is_palindrome(\"A man, a plan, a canal: Panama\") == True\nassert is_palindrome(\"hello\") == False\nassert is_palindrome(\"\") == True\n```");
        }
        if n == 1 {
            return program("def is_palindrome(s):\n    return s == s[::-1]");
        }
        program("def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]")
    }));
    // Register the two extra servers through the registry (it is cached after first use).
    let mut reg = harness2::llm::connections::load();
    for c in [conn("j", pj), conn("k", pk)] {
        reg.connections.push(serde_json::from_value(c).unwrap());
    }
    harness2::llm::connections::save(&reg).unwrap();
    let j = project(&d.path().join("proj_j"), "j/m", "on");
    let (rd, _) = solve(&j, "code me a sleek designed hello world page", &[], "j1");
    let page = std::fs::read_to_string(rd.join("workspace/output/index.html")).unwrap();
    assert!(page.starts_with("<!DOCTYPE html>") && page.contains("--accent:#0369a1") && page.contains("<h1>Hello, World!</h1>"), "{page}");
    assert_eq!(seen_j.lock().unwrap().len(), 2, "one reply in the wrong form, one in the right one");
    let u = read_json(&rd.join("usage.json"));
    assert_eq!((u["accuracy"]["confidence"].as_str(), u["accuracy"]["passed"].as_u64()), (Some("high"), Some(16)), "{u}");
    assert_eq!(u["review_open"], json!([]));
    let k = project(&d.path().join("proj_k"), "k/m", "on");
    let (rd, out) = solve(&k, "Write a Python function is_palindrome(s) that ignores case and punctuation.", &[], "k1");
    let sol = std::fs::read_to_string(rd.join("workspace/output/solution.py")).unwrap();
    assert!(sol.contains("isalnum"), "the candidate that passes the tests is kept: {sol}");
    assert!(out.contains("passes 4"), "{out}");
    assert_eq!(read_json(&rd.join("usage.json"))["accuracy"]["confidence"], "high");

    // ---- H: a fenced Python block written instead of a tool call is run, not ignored.
    let h = project(&d.path().join("proj_h"), "h/m", "on");
    let (rd, out) = solve(&h, "Write a line saying what six sevens make to output/answer.md.", &[], "h1");
    assert!(out.contains("42"), "{out}");
    assert!(std::fs::read_to_string(rd.join("trajectory.jsonl")).unwrap().contains("ran_code_block"));
}
