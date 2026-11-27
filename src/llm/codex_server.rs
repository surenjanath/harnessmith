//! A long-lived `codex app-server` per Codex binary, spoken to over JSON-RPC on stdio.
//!
//! `codex exec` starts a process for every model call (about 1.5–2 s of a 4–4.5 s call). One
//! app-server process serves every call instead: each call is an ephemeral thread with the
//! harness's instructions, the same slimmed config as the exec path, and the same JSON output
//! schema, so replies are parsed exactly as before. Measured: ~2.3–2.9 s per call against
//! ~4–4.5 s, at the same ~4.3k input tokens. Threads are saved, not ephemeral (only saved
//! threads hit the prompt cache), and deleted when the call ends. Any failure of the server itself (cannot start,
//! exits, protocol error) makes the caller fall back to `codex exec`.

use super::client::{ChatReq, ChatResp, CANCELLED};
use super::codex_cli;
use anyhow::{anyhow, bail, Result};
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Errors of the server itself start with this; the caller falls back to `codex exec` on them.
pub const SERVER_ERR: &str = "codex app-server:";

struct Server {
    stdin: Mutex<ChildStdin>,
    child: Mutex<Child>,
    next: AtomicU64,
    alive: Arc<AtomicBool>,
    replies: Arc<Mutex<HashMap<u64, Sender<Value>>>>,
    threads: Arc<Mutex<HashMap<String, Sender<Value>>>>,
}

static SERVERS: Lazy<Mutex<HashMap<String, Arc<Server>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

impl Drop for Server {
    fn drop(&mut self) {
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

impl Server {
    fn start(bin: &str, cwd: &std::path::Path) -> Result<Arc<Server>> {
        let mut child = Command::new(bin)
            .arg("app-server")
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| anyhow!("{SERVER_ERR} cannot start {bin:?}: {e}"))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("{SERVER_ERR} no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("{SERVER_ERR} no stdout"))?;
        let srv = Arc::new(Server {
            stdin: Mutex::new(stdin),
            child: Mutex::new(child),
            next: AtomicU64::new(1),
            alive: Arc::new(AtomicBool::new(true)),
            replies: Arc::new(Mutex::new(HashMap::new())),
            threads: Arc::new(Mutex::new(HashMap::new())),
        });
        // The reader: replies by request id, notifications by thread id. Requests from the
        // server (approvals, tool calls) are refused: the harness runs its own tools. It holds a
        // weak reference, so dropping the last Server ends the process and the thread.
        let weak = Arc::downgrade(&srv);
        let (alive, replies, threads) = (srv.alive.clone(), srv.replies.clone(), srv.threads.clone());
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                let Ok(m) = serde_json::from_str::<Value>(&line) else { continue };
                let id = m.get("id").and_then(|i| i.as_u64());
                let is_reply = m.get("result").is_some() || m.get("error").is_some();
                if let (Some(id), true) = (id, is_reply) {
                    if let Some(tx) = replies.lock().unwrap().remove(&id) {
                        let _ = tx.send(m);
                    }
                } else if m.get("method").is_some() && m.get("id").is_some() {
                    if let Some(s) = weak.upgrade() {
                        let _ = s.write(&json!({"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32601, "message": "not supported by this client"}}));
                    }
                } else if let Some(tid) = m["params"]["threadId"].as_str() {
                    if let Some(tx) = threads.lock().unwrap().get(tid) {
                        let _ = tx.send(m.clone());
                    }
                }
            }
            alive.store(false, Ordering::SeqCst);
            // Wake everyone waiting: their senders are dropped.
            replies.lock().unwrap().clear();
            threads.lock().unwrap().clear();
        });
        let r = srv.request("initialize", json!({"clientInfo": {"name": "harnessmith", "version": env!("CARGO_PKG_VERSION")}}), Duration::from_secs(30))?;
        if r.get("error").is_some() {
            bail!("{SERVER_ERR} initialize failed: {}", r["error"]);
        }
        srv.write(&json!({"jsonrpc": "2.0", "method": "initialized", "params": {}}))?;
        Ok(srv)
    }

    fn write_raw(&self, line: &str) -> Result<()> {
        let mut w = self.stdin.lock().unwrap();
        w.write_all(line.as_bytes()).and_then(|_| w.write_all(b"\n")).and_then(|_| w.flush()).map_err(|e| anyhow!("{SERVER_ERR} write failed: {e}"))
    }

    fn write(&self, v: &Value) -> Result<()> {
        self.write_raw(&v.to_string())
    }

    fn send(&self, method: &str, params: Value) -> Result<(u64, Receiver<Value>)> {
        if !self.alive.load(Ordering::SeqCst) {
            bail!("{SERVER_ERR} the server has exited");
        }
        let id = self.next.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = channel();
        self.replies.lock().unwrap().insert(id, tx);
        self.write(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        Ok((id, rx))
    }

    fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let (id, rx) = self.send(method, params)?;
        match rx.recv_timeout(timeout) {
            Ok(v) => Ok(v),
            Err(_) => {
                self.replies.lock().unwrap().remove(&id);
                bail!("{SERVER_ERR} no reply to {method} within {}s", timeout.as_secs())
            }
        }
    }
}

fn server(bin: &str, cwd: &std::path::Path) -> Result<Arc<Server>> {
    let mut all = SERVERS.lock().unwrap();
    if let Some(s) = all.get(bin) {
        if s.alive.load(Ordering::SeqCst) {
            return Ok(s.clone());
        }
    }
    let s = Server::start(bin, cwd)?;
    all.insert(bin.to_string(), s.clone());
    Ok(s)
}

/// Whether to use the app-server (default on; `HARNESS2_CODEX_SERVER=0` uses `codex exec`).
pub fn enabled() -> bool {
    !std::env::var("HARNESS2_CODEX_SERVER").map(|v| v == "0" || v.eq_ignore_ascii_case("off")).unwrap_or(false)
}

/// The thread config: the exec path's switches, as one map.
fn config() -> Value {
    let mut m = serde_json::Map::new();
    for f in codex_cli::OFF_FEATURES {
        m.insert(format!("features.{f}"), json!(false));
    }
    for k in codex_cli::OFF_SETTINGS {
        m.insert(k.to_string(), json!(false));
    }
    m.insert("web_search".into(), json!("disabled"));
    Value::Object(m)
}

pub fn chat(bin: &str, model: &str, effort: &str, req: &ChatReq, limit: Duration) -> Result<ChatResp> {
    let home = codex_cli::home()?;
    let ws = home.join("ws");
    let srv = server(bin, &ws)?;
    let mut tparams = json!({"cwd": ws, "sandbox": "read-only", "approvalPolicy": "never", "ephemeral": false, "serviceName": "harnessmith", "baseInstructions": codex_cli::INSTRUCTIONS, "config": config()});
    if !model.is_empty() && model != "default" {
        tparams["model"] = json!(model);
    }
    let r = srv.request("thread/start", tparams, Duration::from_secs(60))?;
    if let Some(e) = r.get("error") {
        bail!("{SERVER_ERR} thread/start failed: {e}");
    }
    let tid = r["result"]["thread"]["id"].as_str().ok_or_else(|| anyhow!("{SERVER_ERR} thread/start returned no id"))?.to_string();
    let (tx, rx) = channel();
    srv.threads.lock().unwrap().insert(tid.clone(), tx);
    let guard = ThreadGuard { srv: srv.clone(), tid: tid.clone() };
    let with_tools = !req.tools.is_empty();
    let turn = json!({"threadId": tid, "input": [{"type": "text", "text": codex_cli::render(req)}], "outputSchema": codex_cli::schema(with_tools), "effort": effort});
    let r = srv.request("turn/start", turn, Duration::from_secs(60))?;
    if let Some(e) = r.get("error") {
        // A model or account problem, not the server's: reported like the exec path's.
        bail!("{}", model_error(&e.to_string()));
    }
    let turn_id = r["result"]["turn"]["id"].as_str().unwrap_or("").to_string();
    let (mut text, mut usage, mut err): (Option<String>, Option<Value>, Option<String>) = (None, None, None);
    let t0 = Instant::now();
    loop {
        if crate::log::cancelled() {
            let _ = srv.send("turn/interrupt", json!({"threadId": tid, "turnId": turn_id}));
            bail!(CANCELLED);
        }
        if t0.elapsed() > limit {
            let _ = srv.send("turn/interrupt", json!({"threadId": tid, "turnId": turn_id}));
            bail!("HTTP 504 from codex: no reply within {}s (turn interrupted; HARNESS2_CODEX_CALL_TIMEOUT raises the cap)", limit.as_secs());
        }
        let m = match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(m) => m,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => bail!("{SERVER_ERR} the server exited during the turn"),
        };
        let p = &m["params"];
        match m["method"].as_str().unwrap_or("") {
            "item/completed" if p["item"]["type"] == "agentMessage" => text = p["item"]["text"].as_str().map(|s| s.to_string()),
            "thread/tokenUsage/updated" => usage = Some(p["tokenUsage"]["last"].clone()),
            "error" if p["willRetry"] != true => err = Some(p["error"].to_string()),
            "turn/completed" => {
                if text.is_none() && !p["turn"]["error"].is_null() {
                    err = Some(p["turn"]["error"].to_string());
                }
                break;
            }
            _ => {}
        }
    }
    drop(guard);
    let Some(text) = text.filter(|t| !t.trim().is_empty()) else {
        bail!("{}", model_error(&err.unwrap_or_else(|| "the turn ended without a reply".into())));
    };
    let mut out = codex_cli::to_resp(&text, with_tools);
    if let Some(u) = usage {
        let cached = u["cachedInputTokens"].as_u64().unwrap_or(0);
        out.input_tokens = u["inputTokens"].as_u64().unwrap_or(0).saturating_sub(cached);
        out.cached_tokens = cached;
        out.output_tokens = u["outputTokens"].as_u64().unwrap_or(0) + u["reasoningOutputTokens"].as_u64().unwrap_or(0);
    }
    Ok(out)
}

/// Usage limits and sign-in problems read like HTTP errors, so the role's chain fails over.
fn model_error(why: &str) -> String {
    let low = why.to_lowercase();
    let code = if low.contains("usage limit") || low.contains("rate limit") || low.contains("429") {
        "HTTP 429"
    } else if low.contains("log in") || low.contains("login") || low.contains("401") || low.contains("unauthorized") {
        "HTTP 401"
    } else {
        "codex turn failed"
    };
    format!("{code} from codex: {}", crate::util::take_chars(why, 600))
}

/// Stops routing a finished thread's notifications.
struct ThreadGuard {
    srv: Arc<Server>,
    tid: String,
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        self.srv.threads.lock().unwrap().remove(&self.tid);
        // Saved (not ephemeral) threads keep the provider's prompt cache: measured, ephemeral
        // threads were never served from it (0 of 7.2k tokens cached), saved ones were (4–7k).
        // Deleting the thread keeps the user's Codex history free of harness calls.
        let _ = self.srv.send("thread/delete", json!({"threadId": self.tid}));
    }
}
