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
    let (p3, _seen3) = server(Arc::new(|n, req: &Value| {
        let rescued = req["messages"].as_array().unwrap().iter().any(|m| m["content"].as_str().map(|c| c.contains("output/ is still empty")).unwrap_or(false));
        if rescued {
            (200, reply_tool("write_file", json!({"path": "output/answer.md", "content": "best effort"})))
        } else {
            (200, reply_tool("read_file", json!({"path": format!("notes{n}.txt")})))
        }
    }));
    let d = setup(json!({"connections": [conn("deadone", dead), conn("backup", good), conn("looper", p2), conn("slow", p3)]}));
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
}
