//! Self-healing behaviour of the agent loop, against scripted OpenAI-compatible servers:
//! failover along a model chain, waiting out an outage, the loop breaker, and the deadline
//! rescue.

use harness2::llm::agent::{self, AgentConfig};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

type Script = Arc<dyn Fn(usize, &Value) -> (u16, Value) + Send + Sync>;

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
            let (code, out) = script(n, &req);
            let b = out.to_string();
            let _ = write!(s, "HTTP/1.1 {code} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}", b.len());
        }
    });
    (port, seen)
}

fn reply_tool(name: &str, args: Value) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": args.to_string()}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
}

fn reply_text(t: &str) -> Value {
    json!({"choices": [{"message": {"role": "assistant", "content": t}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
}

fn setup(conns: Value) -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("connections.json");
    std::fs::write(&p, conns.to_string()).unwrap();
    std::env::set_var("HARNESS2_CONNECTIONS", &p);
    harness2::log::set_quiet(true);
    d
}

fn conn(name: &str, port: u16) -> Value {
    json!({"name": name, "provider": "openai", "base_url": format!("http://127.0.0.1:{port}/v1"), "default_model": "m"})
}

fn trajectory_kinds(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|e| e["type"] == "heal")
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect()
}

/// All self-healing scenarios share HARNESS2_CONNECTIONS, so they run in one test.
#[test]
fn self_healing_scenarios() {
    // A port with nothing listening: connection refused, fast.
    let dead = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();

    let (good, seen) = server(Arc::new(|n, _| if n == 1 { (200, reply_tool("write_file", json!({"path": "output/a.md", "content": "ok"}))) } else { (200, reply_text("Done.")) }));
    let (p2, seen2) = server(Arc::new(|_n, req: &Value| {
        let warned = req["messages"].as_array().unwrap().iter().any(|m| m["content"].as_str().map(|c| c.contains("[self-check] You have now made this exact call 3 times")).unwrap_or(false));
        if warned { (200, reply_text("Changed approach.")) } else { (200, reply_tool("list_files", json!({"path": "."}))) }
    }));
    let (p4, seen4) = server(Arc::new(|n, req: &Value| {
        if n == 1 {
            (200, reply_tool("csv_profile", json!({"path": "data.csv"})))
        } else {
            let msgs = req["messages"].as_array().unwrap();
            let got = msgs.iter().any(|m| m["content"].as_str().map(|c| c.contains("3 rows x 2 columns") && c.contains("max 7")).unwrap_or(false));
            (200, reply_text(if got { "profiled" } else { "no profile" }))
        }
    }));
    let (p5, _seen5) = server(Arc::new(|n, req: &Value| {
        if n == 1 {
            (200, reply_tool("bash", json!({"command": "sudo rm -rf /var/x"})))
        } else if n == 2 {
            let blocked = req["messages"].as_array().unwrap().iter().any(|m| m["content"].as_str().map(|c| c.contains("[gate] Blocked") && c.contains("`sudo`")).unwrap_or(false));
            (200, if blocked { reply_tool("write_file", json!({"path": "data.csv", "content": "overwritten"})) } else { reply_text("not blocked") })
        } else {
            let blocked = req["messages"].as_array().unwrap().iter().any(|m| m["content"].as_str().map(|c| c.contains("inputs-readonly")).unwrap_or(false));
            (200, reply_text(if blocked { "respected the gate" } else { "input was overwritten" }))
        }
    }));
    let (p3, _seen3) = server(Arc::new(|n, req: &Value| {
        let rescued = req["messages"].as_array().unwrap().iter().any(|m| m["content"].as_str().map(|c| c.contains("output/ is still empty")).unwrap_or(false));
        if rescued {
            (200, reply_tool("write_file", json!({"path": "output/answer.md", "content": "best effort"})))
        } else {
            (200, reply_tool("read_file", json!({"path": format!("notes{n}.txt")})))
        }
    }));
    let d = setup(json!({"connections": [conn("deadone", dead), conn("backup", good), conn("looper", p2), conn("slow", p3), conn("tooler", p4), conn("gated", p5)]}));
    let (d2, d3) = (&d, &d);

    // 1. Failover: the primary is down, the backup answers, the work completes.
    let t0 = std::time::Instant::now();
    let ws = d.path().join("ws1");
    std::fs::create_dir_all(&ws).unwrap();
    let mut cfg = AgentConfig::new("deadone/m, backup/m", "agent", &ws);
    cfg.allow_shell = false;
    cfg.trajectory = Some(d.path().join("t1.jsonl"));
    let out = agent::run(&cfg, "go").unwrap();
    assert_eq!(out.error, "", "{out:?}");
    assert_eq!(out.served_by, "backup/m");
    assert_eq!(std::fs::read_to_string(ws.join("output/a.md")).unwrap(), "ok");
    assert!(trajectory_kinds(&d.path().join("t1.jsonl")).contains(&"failover".to_string()));
    assert_eq!(seen.lock().unwrap().len(), 2);
    assert!(t0.elapsed().as_secs() < 5, "failover should be fast on a refused connection: {:?}", t0.elapsed());

    // 2. Loop breaker: the model repeats one call; the 3rd identical result carries a self-check,
    //    and the model then changes course.
    let ws2 = d2.path().join("ws2");
    std::fs::create_dir_all(&ws2).unwrap();
    let mut cfg = AgentConfig::new("looper/m", "agent", &ws2);
    cfg.trajectory = Some(d2.path().join("t2.jsonl"));
    let out = agent::run(&cfg, "go").unwrap();
    assert_eq!(out.final_text, "Changed approach.");
    assert_eq!(seen2.lock().unwrap().len(), 4, "three identical calls, then a different reply");
    assert_eq!(trajectory_kinds(&d2.path().join("t2.jsonl")), vec!["loop_break".to_string()]);

    // 3. Deadline rescue: with the step budget nearly gone and output/ empty, the model is told
    //    to deliver, and does.
    let ws3 = d3.path().join("ws3");
    std::fs::create_dir_all(ws3.join("output")).unwrap();
    let mut cfg = AgentConfig::new("slow/m", "agent", &ws3);
    cfg.max_steps = 8;
    cfg.rescue_dir = Some(ws3.join("output"));
    cfg.rescue_hint = "output/ is still empty. Write it now.".into();
    cfg.trajectory = Some(d3.path().join("t3.jsonl"));
    let _ = agent::run(&cfg, "go").unwrap();
    assert_eq!(std::fs::read_to_string(ws3.join("output/answer.md")).unwrap(), "best effort");
    let kinds = trajectory_kinds(&d3.path().join("t3.jsonl"));
    assert!(kinds.contains(&"deadline_rescue".to_string()), "{kinds:?}");
    assert!(kinds.contains(&"error_streak".to_string()), "{kinds:?}");

    // 4. Wired tools: a harness `tools/<name>.json` is offered to the model as a real tool and runs
    //    its command with the call's arguments on stdin.
    let ws4 = d.path().join("ws4");
    let h4 = ws4.join(".harness");
    for (rel, body) in harness2::substrate::native::default_base_tree() {
        let dst = h4.join(&rel);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::write(&dst, &body).unwrap();
        if rel.starts_with("scripts/") {
            std::fs::create_dir_all(ws4.join("scripts")).unwrap();
            std::fs::write(ws4.join(&rel), &body).unwrap();
        }
    }
    std::fs::write(ws4.join("data.csv"), "region,q1\nA,3\nB,7\nC,5\n").unwrap();
    let mut cfg = AgentConfig::new("tooler/m", "agent", &ws4);
    cfg.harness = Some(h4.clone());
    let out = agent::run(&cfg, "go").unwrap();
    let first = &seen4.lock().unwrap()[0];
    let names: Vec<&str> = first["tools"].as_array().unwrap().iter().filter_map(|t| t["function"]["name"].as_str()).collect();
    assert!(names.contains(&"csv_profile"), "{names:?}");
    assert_eq!(out.final_text, "profiled", "the tool result should reach the model");
    // and a harness without shell access does not get it
    let mut cfg = AgentConfig::new("tooler/m", "agent", &ws4);
    cfg.harness = Some(h4);
    cfg.allow_shell = false;
    assert!(!agent::tool_defs(&cfg).iter().any(|t| t.name == "csv_profile"));

    // 5. Guardrail gate: a dangerous command and a write to a task input are refused, the agent is
    //    told why, and the input is untouched.
    let ws5 = d.path().join("ws5");
    std::fs::create_dir_all(ws5.join("output")).unwrap();
    std::fs::write(ws5.join("data.csv"), "a\n1\n").unwrap();
    let mut cfg = AgentConfig::new("gated/m", "agent", &ws5);
    let mut g = harness2::llm::gate::Gate::new("enforce", &[]);
    g.protect_inputs(&ws5);
    cfg.gate = Some(g);
    cfg.trajectory = Some(d.path().join("t5.jsonl"));
    let out = agent::run(&cfg, "go").unwrap();
    assert_eq!(out.final_text, "respected the gate");
    assert_eq!(std::fs::read_to_string(ws5.join("data.csv")).unwrap(), "a\n1\n");
    let gates = std::fs::read_to_string(d.path().join("t5.jsonl")).unwrap().lines().filter(|l| l.contains("\"type\":\"gate\"")).count();
    assert_eq!(gates, 2);
}
