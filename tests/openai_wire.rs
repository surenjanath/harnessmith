//! The agent loop against a strict OpenAI-compatible server (the shape llama.cpp, vLLM and
//! LM Studio validate): a text-emitted `<tool_call>` is recovered, executed, and the follow-up
//! request carries a string assistant content plus a `tool` message.

use harness2::llm::agent::{self, AgentConfig};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

fn serve(listener: TcpListener, seen: Arc<Mutex<Vec<Value>>>) {
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
            r.read_exact(&mut body).unwrap();
            let req: Value = serde_json::from_slice(&body).unwrap();
            // Strict validation, as llama.cpp's server does.
            let ok = req["messages"].as_array().unwrap().iter().all(|m| m["content"].is_string());
            let n = { let mut g = seen.lock().unwrap(); g.push(req.clone()); g.len() };
            let (code, out) = if !ok {
                ("500 Internal Server Error", json!({"error": {"message": "content must be a string"}}))
            } else if n == 1 {
                ("200 OK", json!({"choices": [{"message": {"role": "assistant", "content": "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"output/answer.md\", \"content\": \"north fell 25.0%\"}}\n</tool_call>"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 50, "completion_tokens": 20}}))
            } else {
                ("200 OK", json!({"choices": [{"message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 80, "completion_tokens": 2}}))
            };
            let b = out.to_string();
            let _ = write!(s, "HTTP/1.1 {code}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{b}", b.len());
        }
    });
}

#[test]
fn text_tool_call_round_trip_on_strict_server() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(vec![]));
    serve(listener, seen.clone());

    let d = tempfile::tempdir().unwrap();
    let conns = d.path().join("connections.json");
    std::fs::write(&conns, json!({"connections": [{"name": "strict", "provider": "openai", "base_url": format!("http://127.0.0.1:{port}/v1"), "default_model": "m"}]}).to_string()).unwrap();
    std::env::set_var("HARNESS2_CONNECTIONS", &conns);
    harness2::log::set_quiet(true);

    let ws = d.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let mut cfg = AgentConfig::new("strict/m", "You are an agent.", &ws);
    cfg.allow_shell = false;
    let out = agent::run(&cfg, "Write the answer.").unwrap();
    assert_eq!(out.error, "", "{:?}", out);
    assert_eq!(out.final_text, "Done.");
    assert_eq!(std::fs::read_to_string(ws.join("output/answer.md")).unwrap(), "north fell 25.0%");
    let reqs = seen.lock().unwrap();
    assert_eq!(reqs.len(), 2);
    let msgs = reqs[1]["messages"].as_array().unwrap();
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["tool_calls"][0]["function"]["name"], "write_file");
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!((out.input_tokens, out.output_tokens), (130, 22));
}
