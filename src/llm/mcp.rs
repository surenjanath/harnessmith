//! MCP (Model Context Protocol) client over stdio: connect a project's MCP servers and offer
//! their tools to the solver next to the built-in and wired tools.
//!
//! Servers are configured by the operator in the project settings (not in the harness, so the
//! improver cannot add one). Each rollout starts its own server processes and stops them at the
//! end. Messages are newline-delimited JSON-RPC 2.0, as the MCP stdio transport specifies.

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const PROTOCOL_VERSION: &str = "2025-06-18";

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ServerCfg {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default = "yes")]
    pub enabled: bool,
    /// Ask a person before every call to this server's tools (gate "ask" flow).
    #[serde(default)]
    pub ask: bool,
}
fn yes() -> bool {
    true
}

#[derive(Debug, Clone)]
pub struct McpTool {
    pub server: String,
    /// The name the model sees: `mcp_<server>_<tool>`.
    pub exposed: String,
    pub name: String,
    pub description: String,
    pub schema: Value,
}

pub struct Client {
    pub cfg: ServerCfg,
    child: Child,
    stdin: Mutex<ChildStdin>,
    rx: Mutex<Receiver<Value>>,
    next_id: Mutex<u64>,
    pub tools: Vec<McpTool>,
}

fn clean(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

impl Client {
    /// Start the server, handshake, and list its tools.
    pub fn start(cfg: &ServerCfg, timeout: Duration) -> Result<Client> {
        if cfg.command.trim().is_empty() {
            bail!("MCP server {:?} has no command", cfg.name);
        }
        let mut c = Command::new(&cfg.command);
        c.args(&cfg.args).envs(&cfg.env).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
        let mut child = c.spawn().map_err(|e| anyhow!("cannot start MCP server {:?} ({}): {e}", cfg.name, cfg.command))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = channel::<Value>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<Value>(line.trim()) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
            }
        });
        let mut cl = Client { cfg: cfg.clone(), child, stdin: Mutex::new(stdin), rx: Mutex::new(rx), next_id: Mutex::new(1), tools: vec![] };
        cl.request(
            "initialize",
            json!({"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "harnessmith", "version": env!("CARGO_PKG_VERSION")}}),
            timeout,
        )?;
        cl.notify("notifications/initialized", json!({}))?;
        let mut tools = vec![];
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let r = cl.request("tools/list", params, timeout)?;
            for t in r["tools"].as_array().cloned().unwrap_or_default() {
                let name = t["name"].as_str().unwrap_or("").to_string();
                if name.is_empty() {
                    continue;
                }
                let mut exposed = format!("mcp_{}_{}", clean(&cfg.name), clean(&name));
                exposed.truncate(64);
                let mut schema = t.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object", "properties": {}}));
                if schema.get("type").and_then(|x| x.as_str()) != Some("object") {
                    schema = json!({"type": "object", "properties": {}});
                }
                tools.push(McpTool { server: cfg.name.clone(), exposed, name, description: t["description"].as_str().unwrap_or("").to_string(), schema });
            }
            cursor = r["nextCursor"].as_str().map(String::from);
            if cursor.is_none() || tools.len() > 500 {
                break;
            }
        }
        cl.tools = tools;
        Ok(cl)
    }

    fn send(&self, msg: &Value) -> Result<()> {
        let mut w = self.stdin.lock().unwrap();
        writeln!(w, "{msg}")?;
        w.flush()?;
        Ok(())
    }

    fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.send(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let id = {
            let mut n = self.next_id.lock().unwrap();
            *n += 1;
            *n
        };
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        let rx = self.rx.lock().unwrap();
        let until = Instant::now() + timeout;
        loop {
            let left = until.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(v) if v.get("id").and_then(|x| x.as_u64()) == Some(id) => {
                    if let Some(e) = v.get("error") {
                        bail!("MCP {} {method}: {}", self.cfg.name, e.get("message").and_then(|m| m.as_str()).unwrap_or("error"));
                    }
                    return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                }
                // Server-initiated requests (e.g. ping) get an empty result; notifications are ignored.
                Ok(v) if v.get("method").is_some() && v.get("id").is_some() => {
                    let _ = self.send(&json!({"jsonrpc": "2.0", "id": v["id"], "result": {}}));
                }
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => bail!("MCP server {:?} did not answer {method} within {}s", self.cfg.name, timeout.as_secs()),
                Err(RecvTimeoutError::Disconnected) => bail!("MCP server {:?} exited", self.cfg.name),
            }
        }
    }

    /// Call a tool; returns its text content (an MCP-level error becomes Err).
    pub fn call(&self, tool: &str, args: &Value, timeout: Duration) -> Result<String> {
        let r = self.request("tools/call", json!({"name": tool, "arguments": args}), timeout)?;
        let mut parts = vec![];
        for c in r["content"].as_array().cloned().unwrap_or_default() {
            match c["type"].as_str() {
                Some("text") => parts.push(c["text"].as_str().unwrap_or("").to_string()),
                Some("resource") => parts.push(c["resource"]["text"].as_str().map(String::from).unwrap_or_else(|| c["resource"].to_string())),
                Some(other) => parts.push(format!("[{other} content]")),
                None => {}
            }
        }
        if let Some(sc) = r.get("structuredContent").filter(|_| parts.is_empty()) {
            parts.push(sc.to_string());
        }
        let text = parts.join("\n");
        if r["isError"].as_bool().unwrap_or(false) {
            bail!("{text}");
        }
        Ok(if text.trim().is_empty() { "(no output)".into() } else { text })
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The started servers of one agent session.
#[derive(Default)]
pub struct Hub {
    pub clients: Vec<Client>,
    /// Servers that failed to start, with the reason (reported to the model and the log).
    pub failed: Vec<(String, String)>,
}

impl Hub {
    pub fn start(cfgs: &[ServerCfg]) -> Hub {
        let mut h = Hub::default();
        for c in cfgs.iter().filter(|c| c.enabled) {
            match Client::start(c, Duration::from_secs(30)) {
                Ok(cl) => h.clients.push(cl),
                Err(e) => h.failed.push((c.name.clone(), format!("{e}"))),
            }
        }
        h
    }
    pub fn tools(&self) -> Vec<&McpTool> {
        self.clients.iter().flat_map(|c| c.tools.iter()).collect()
    }
    pub fn find(&self, exposed: &str) -> Option<(&Client, &McpTool)> {
        self.clients.iter().find_map(|c| c.tools.iter().find(|t| t.exposed == exposed).map(|t| (c, t)))
    }
}

/// Start a server, list its tools, stop it (the UI's Test button).
pub fn probe(cfg: &ServerCfg) -> Result<Vec<(String, String)>> {
    let c = Client::start(cfg, Duration::from_secs(20))?;
    Ok(c.tools.iter().map(|t| (t.name.clone(), t.description.clone())).collect())
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Hub({} servers, {} tools)", self.clients.len(), self.tools().len())
    }
}
