//! One chat-with-tools call against any connected endpoint.
//!
//! Speaks the OpenAI Chat Completions protocol (which also covers Ollama, vLLM, LM Studio,
//! OpenRouter, Groq, DeepSeek and other compatible servers), the Anthropic Messages API, and
//! the Gemini generateContent API. A `mock` provider answers deterministically for offline
//! smoke tests.

use super::connections::Connection;
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema (type object, properties, required).
    pub parameters: Value,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
    /// Provider-specific data that must be echoed back (e.g. Gemini thought signatures).
    pub extra: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub id: String,
    pub name: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub enum Msg {
    User(String),
    Assistant { text: String, calls: Vec<ToolCall> },
    Tools(Vec<ToolResult>),
}

#[derive(Debug, Clone, Default)]
pub struct ChatResp {
    pub text: String,
    pub calls: Vec<ToolCall>,
    /// Uncached input tokens (cached ones are counted separately below).
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub stop: String,
    /// Input tokens served from the provider's prompt cache.
    pub cached_tokens: u64,
    /// Input tokens written to the cache (Anthropic).
    pub cache_write_tokens: u64,
}

pub struct ChatReq<'a> {
    pub system: &'a str,
    pub messages: &'a [Msg],
    pub tools: &'a [ToolDef],
    pub max_tokens: u32,
    /// Sampling temperature for this call; `None` uses the connection's setting (or the
    /// server's default). The answer engine sets it per sample: low for a best guess, higher
    /// for the independent attempts it votes over.
    pub temperature: Option<f64>,
}

/// Whether `url`'s host should skip the environment proxy (loopback, or listed in NO_PROXY).
fn bypass_proxy(url: &str) -> bool {
    let rest = url.split_once("://").map(|x| x.1).unwrap_or(url);
    let hostport = rest.split('/').next().unwrap_or("");
    let host = if hostport.starts_with('[') {
        hostport.trim_start_matches('[').split(']').next().unwrap_or("")
    } else {
        hostport.split(':').next().unwrap_or("")
    }
    .to_lowercase();
    if host == "localhost" || host.starts_with("127.") || host == "::1" || host == "0.0.0.0" || host == "host.docker.internal" {
        return true;
    }
    let np = std::env::var("NO_PROXY").or_else(|_| std::env::var("no_proxy")).unwrap_or_default();
    np.split(',').map(|x| x.trim().trim_start_matches('*').trim_start_matches('.').to_lowercase()).filter(|x| !x.is_empty()).any(|d| d == "*" || host == d || host.ends_with(&format!(".{d}")))
}

fn agent_for(url: &str, timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(timeout)
        .timeout_write(Duration::from_secs(120))
        .try_proxy_from_env(!bypass_proxy(url))
        .build()
}

/// The outcome of one HTTP attempt.
enum Sent {
    Ok(String),
    Status(u16, String),
    /// Transport failure; `true` when retrying is pointless (nothing listening, or cancelled).
    Transport(String, bool),
}

pub const CANCELLED: &str = "cancelled by user";

fn post(url: &str, headers: &[(&str, String)], body: &Value, timeout: Duration) -> Result<Value> {
    let mut last = String::new();
    for attempt in 0..4u32 {
        if attempt > 0 && crate::util::sleep_cancellable(Duration::from_secs(2u64.pow(attempt))) {
            bail!(CANCELLED);
        }
        // Plain-HTTP servers (local Ollama, llama.cpp, LM Studio...) go through a client whose
        // socket Stop can close mid-call, which also makes the server stop generating. HTTPS
        // calls run on a helper thread that Stop abandons.
        let sent = if url.starts_with("http://") && (bypass_proxy(url) || std::env::var("HTTP_PROXY").or_else(|_| std::env::var("http_proxy")).is_err()) {
            plain_post(url, headers, body, timeout)
        } else {
            ureq_post_cancellable(url, headers, body, timeout)
        };
        match sent {
            Sent::Ok(text) => {
                return serde_json::from_str(&text).map_err(|e| anyhow!("invalid JSON from {url}: {e}: {}", crate::util::take_chars(&text, 300)));
            }
            Sent::Status(code, b) => {
                last = format!("HTTP {code} from {url}: {}", crate::util::take_chars(&b, 600));
                if ![408, 409, 429, 500, 502, 503, 504, 529].contains(&code) {
                    bail!(last);
                }
            }
            Sent::Transport(msg, fatal) => {
                if msg == CANCELLED {
                    bail!(CANCELLED);
                }
                last = format!("transport error calling {url}: {msg}");
                if fatal {
                    bail!(last);
                }
            }
        }
    }
    bail!(last)
}

fn ureq_post_cancellable(url: &str, headers: &[(&str, String)], body: &Value, timeout: Duration) -> Sent {
    let (url_o, hs, body_o) = (url.to_string(), headers.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<Vec<_>>(), body.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut req = agent_for(&url_o, timeout).post(&url_o).set("content-type", "application/json");
        for (k, v) in &hs {
            req = req.set(k, v);
        }
        let r = match req.send_json(body_o) {
            Ok(resp) => match resp.into_string() {
                Ok(t) => Sent::Ok(t),
                Err(e) => Sent::Transport(format!("reading the reply: {e}"), false),
            },
            Err(ureq::Error::Status(code, resp)) => Sent::Status(code, resp.into_string().unwrap_or_default()),
            Err(ureq::Error::Transport(t)) => {
                let fatal = matches!(t.kind(), ureq::ErrorKind::ConnectionFailed | ureq::ErrorKind::Dns);
                Sent::Transport(format!("{t}"), fatal)
            }
        };
        let _ = tx.send(r);
    });
    // The read timeout is per socket read: a server that sends a byte now and then (keep-alive
    // whitespace, a stalled stream) keeps a request open for ever. Hold the whole request to the
    // caller's timeout, with a little slack for connecting and sending, and abandon the thread.
    let deadline = std::time::Instant::now() + timeout + Duration::from_secs(30);
    loop {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(r) => return r,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if crate::log::cancelled() {
                    return Sent::Transport(CANCELLED.into(), true);
                }
                if std::time::Instant::now() >= deadline {
                    return Sent::Transport(format!("no complete reply within {}s", timeout.as_secs() + 30), false);
                }
            }
            Err(_) => return Sent::Transport("request thread ended".into(), false),
        }
    }
}

/// A minimal HTTP/1.1 POST over a plain TCP socket, polled so Stop can close it at once.
fn plain_post(url: &str, headers: &[(&str, String)], body: &Value, timeout: Duration) -> Sent {
    use std::io::{Read, Write};
    use std::net::{Shutdown, TcpStream, ToSocketAddrs};
    let rest = &url["http://".len()..];
    let (hostport, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let hp = if hostport.contains(':') && !hostport.ends_with(']') { hostport.to_string() } else { format!("{hostport}:80") };
    let addr = match hp.to_socket_addrs().ok().and_then(|mut a| a.next()) {
        Some(a) => a,
        None => return Sent::Transport(format!("Dns Failed: cannot resolve {hostport}"), true),
    };
    let mut s = match TcpStream::connect_timeout(&addr, Duration::from_secs(10)) {
        Ok(s) => s,
        Err(e) => return Sent::Transport(format!("Connection Failed: Connect error: {e}"), true),
    };
    let payload = body.to_string();
    let mut head = format!("POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n", payload.len());
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    if let Err(e) = s.write_all(head.as_bytes()).and_then(|_| s.write_all(payload.as_bytes())) {
        return Sent::Transport(format!("sending the request: {e}"), false);
    }
    let _ = s.set_read_timeout(Some(Duration::from_millis(250)));
    let t0 = std::time::Instant::now();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 65536];
    loop {
        match s.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted) => {
                if crate::log::cancelled() {
                    let _ = s.shutdown(Shutdown::Both);
                    return Sent::Transport(CANCELLED.into(), true);
                }
                if t0.elapsed() > timeout {
                    let _ = s.shutdown(Shutdown::Both);
                    return Sent::Transport(format!("timed out reading response after {}s", timeout.as_secs()), false);
                }
            }
            Err(e) => return Sent::Transport(format!("reading the reply: {e}"), false),
        }
    }
    let Some(split) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
        return Sent::Transport("malformed HTTP reply (no header end)".into(), false);
    };
    let head = String::from_utf8_lossy(&buf[..split]).to_string();
    let mut body_b = buf[split + 4..].to_vec();
    let code: u16 = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let lower = head.to_lowercase();
    if lower.contains("transfer-encoding: chunked") {
        body_b = dechunk(&body_b);
    }
    let text = String::from_utf8_lossy(&body_b).to_string();
    if (200..300).contains(&code) {
        Sent::Ok(text)
    } else {
        Sent::Status(code, text)
    }
}

fn dechunk(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let Some(nl) = b[i..].windows(2).position(|w| w == b"\r\n") else { break };
        let size_s = String::from_utf8_lossy(&b[i..i + nl]).to_string();
        let size = usize::from_str_radix(size_s.split(';').next().unwrap_or("").trim(), 16).unwrap_or(0);
        i += nl + 2;
        if size == 0 || i + size > b.len() {
            break;
        }
        out.extend_from_slice(&b[i..i + size]);
        i += size + 2;
    }
    out
}

pub fn get_json(url: &str, headers: &[(&str, String)]) -> Result<Value> {
    let mut req = agent_for(url, Duration::from_secs(30)).get(url);
    for (k, v) in headers {
        req = req.set(k, v);
    }
    match req.call() {
        Ok(r) => Ok(serde_json::from_str(&r.into_string()?)?),
        Err(ureq::Error::Status(code, r)) => bail!("HTTP {code}: {}", crate::util::take_chars(&r.into_string().unwrap_or_default(), 400)),
        Err(e) => bail!("{e}"),
    }
}

static LIMITS: once_cell::sync::Lazy<std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<crate::core::proposer::Semaphore>>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

fn limiter(conn: &Connection) -> std::sync::Arc<crate::core::proposer::Semaphore> {
    let key = format!("{}|{}|{}", conn.name, conn.base(), conn.concurrency());
    LIMITS.lock().unwrap().entry(key).or_insert_with(|| std::sync::Arc::new(crate::core::proposer::Semaphore::new(conn.concurrency()))).clone()
}

/// One model call (queued behind the connection's concurrency limit).
/// Chat models worth trying on a hosted catalog (embedding, safety, reward, parsing, vision-only
/// and translation models filtered out), recommended agent models first.
pub fn chat_candidates(ids: &[String]) -> Vec<String> {
    let skip = regex::Regex::new(r"(?i)embed|safety|guard|reward|parse|clip|detector|calibration|retriever|deplot|kosmos|vila|neva|fuyu|translate|topic-control|riva|cosmos|starcoder|codegemma|codellama|diffusion|vision|vlm|chatqa|-vl-|omni|recurrentgemma|gemma-2b|llama2|minitron|synthetic").unwrap();
    let prefer = ["kimi", "nemotron-3-super", "nemotron-3-ultra", "deepseek-v4", "glm-5", "mistral-large-4", "minimax", "gpt-oss", "nemotron-3.5-lightning", "llama-3.1-nemotron-ultra", "gemma-4", "step-5", "qwen"];
    let mut v: Vec<String> = ids.iter().filter(|m| !skip.is_match(m)).cloned().collect();
    v.sort_by_key(|m| prefer.iter().position(|p| m.contains(p)).unwrap_or(prefer.len()));
    v
}

/// A reply saying this one model does not exist or is not enabled, not that the endpoint is down.
pub fn model_unavailable(err: &str) -> bool {
    let e = err.to_lowercase();
    (e.contains("http 404") || e.contains("http 400") || e.contains("http 403"))
        && (e.contains("not found for account") || e.contains("model_not_found") || e.contains("model not found") || e.contains("does not exist") || e.contains("no such model") || e.contains("unknown model") || e.contains("not available"))
}

/// Try one model with a tiny tool-calling request, without touching the connection's health
/// record (a catalog probe should not trip the breaker for models that do work).
pub fn probe(conn: &Connection, model: &str) -> Value {
    let t0 = std::time::Instant::now();
    let tools = [ToolDef { name: "ping".into(), description: "Reply by calling this tool.".into(), parameters: serde_json::json!({"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]}) }];
    let msgs = [Msg::User("Call the ping tool with word=\"ok\". Do not reply with text.".into())];
    let req = ChatReq { system: "You are a test harness. Use the tool.", messages: &msgs, tools: &tools, max_tokens: 200, temperature: None };
    let res = match conn.provider.as_str() {
        "openai" => openai(conn, model, &req, Duration::from_secs(20)),
        _ => chat(conn, model, &req, Duration::from_secs(20)),
    };
    match res {
        Ok(r) => serde_json::json!({"model": model, "ok": true, "tools": !r.calls.is_empty(), "ms": t0.elapsed().as_millis() as u64}),
        Err(e) => serde_json::json!({"model": model, "ok": false, "error": crate::util::take_chars(&format!("{e}"), 160), "ms": t0.elapsed().as_millis() as u64}),
    }
}

pub fn chat(conn: &Connection, model: &str, req: &ChatReq, timeout: Duration) -> Result<ChatResp> {
    // A hosted endpoint with no key would answer 401: say which setting is missing instead.
    if let Some(why) = conn.missing_key() {
        anyhow::bail!("{why}");
    }
    let lim = limiter(conn);
    let _slot = lim.acquire();
    let t0 = std::time::Instant::now();
    let res = match conn.provider.as_str() {
        "openai" => openai(conn, model, req, timeout),
        "ollama" => ollama(conn, model, req, timeout),
        "anthropic" => anthropic(conn, model, req, timeout),
        "gemini" => gemini(conn, model, req, timeout),
        "mock" => Ok(mock(req)),
        p => bail!("unknown provider {p:?}"),
    };
    let mut r = match res {
        Ok(r) => {
            super::health::record_ok(&conn.name, t0.elapsed(), r.input_tokens, r.output_tokens);
            r
        }
        Err(e) if format!("{e}").contains(CANCELLED) => {
            return Err(e);
        }
        // The endpoint answered but this model is not available (to this account): the
        // connection is fine, so do not count it against the breaker. The caller still fails
        // over to the next model in its chain.
        Err(e) if model_unavailable(&format!("{e}")) => {
            return Err(anyhow!("{e}\n(model {model:?} is not available on connection {:?}; pick another, e.g. with “Find free models” on the Models tab)", conn.name));
        }
        Err(e) => {
            if super::health::record_fail(&conn.name, &format!("{e}")) {
                crate::say!("    ! connection {} tripped its circuit breaker after {} failures in a row", conn.name, super::health::TRIP_AFTER);
            }
            return Err(e);
        }
    };
    r.text = strip_think(&r.text);
    if r.calls.is_empty() && !req.tools.is_empty() {
        r.calls = text_tool_calls(&r.text, req.tools);
        if !r.calls.is_empty() {
            r.text.clear();
        }
    }
    Ok(r)
}

/// Drop `<think>…</think>` reasoning that some local models put in the visible reply.
pub fn strip_think(t: &str) -> String {
    let re = regex::Regex::new(r"(?s)<think>.*?</think>").unwrap();
    let s = re.replace_all(t, "");
    // An unclosed think block at the start means the whole reply was reasoning.
    let s = if s.trim_start().starts_with("<think>") { "" } else { s.as_ref() };
    s.trim().to_string()
}

/// Recover tool calls a model wrote as text instead of using the tool-call channel:
/// `<tool_call>{json}</tool_call>` (Qwen/Hermes), `<|python_tag|>{json}` (Llama), a fenced
/// or bare JSON object with `name` + `arguments`/`parameters`. Only known tools count.
pub fn text_tool_calls(text: &str, tools: &[ToolDef]) -> Vec<ToolCall> {
    let known = |n: &str| tools.iter().any(|t| t.name == n);
    let mut cands: Vec<String> = vec![];
    let tc = regex::Regex::new(r"(?s)<tool_call>\s*(.*?)\s*(?:</tool_call>|$)").unwrap();
    for c in tc.captures_iter(text) {
        cands.push(c[1].to_string());
    }
    if cands.is_empty() {
        let fence = regex::Regex::new(r"(?s)```(?:json|tool_call)?\s*(\{.*?\})\s*```").unwrap();
        for c in fence.captures_iter(text) {
            cands.push(c[1].to_string());
        }
    }
    if cands.is_empty() {
        let t = text.trim().trim_start_matches("<|python_tag|>").trim();
        if t.starts_with('{') || t.starts_with('[') {
            cands.push(t.to_string());
        }
    }
    // Several bare objects mixed with prose: take each balanced {...} that names a known tool.
    if cands.len() <= 1 && text.contains("\"name\"") {
        let objs = balanced_objects(text, 8);
        if objs.len() > 1 || cands.is_empty() {
            cands = objs;
        }
    }
    let mut out = vec![];
    let mut push = |v: &Value| {
        // The name: `name`, or `function` as a string (Llama-1B style), or `function.name`.
        let name = v
            .get("name")
            .and_then(|n| n.as_str())
            .or_else(|| v.get("function").and_then(|f| f.as_str()))
            .or_else(|| v.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()))
            .or_else(|| v.get("tool").and_then(|n| n.as_str()));
        let Some(name) = name.filter(|n| known(n)) else { return };
        let mut args = v.get("arguments").or_else(|| v.get("parameters")).or_else(|| v.get("function").and_then(|f| f.get("arguments"))).or_else(|| v.get("args")).cloned().unwrap_or(json!({}));
        if let Value::String(s) = &args {
            args = lenient_json(s).unwrap_or(json!({}));
        }
        // Arguments wrapped the way a schema is ({"properties": {...}}): unwrap, unless the
        // tool really has a parameter of that name.
        let has_param = |p: &str| tools.iter().find(|t| t.name == name).map(|t| t.parameters["properties"].get(p).is_some()).unwrap_or(false);
        if !has_param("properties") {
            if let Some(inner) = args.get("properties").filter(|p| p.is_object()).cloned() {
                if args.as_object().map(|m| m.keys().all(|k| matches!(k.as_str(), "properties" | "required" | "type"))).unwrap_or(false) {
                    args = inner;
                }
            }
        }
        out.push(ToolCall { id: format!("text_{}", out.len()), name: name.to_string(), args, extra: None });
    };
    for c in cands {
        match lenient_json(&c) {
            Some(Value::Array(a)) => a.iter().for_each(&mut push),
            Some(v) => push(&v),
            // Trailing junk after a complete object ("...}}}" or "}\n}()"): take the object.
            None => {
                if let Some(v) = balanced_objects(&c, 1).first().and_then(|o| lenient_json(o)) {
                    push(&v);
                }
            }
        }
    }
    out
}

/// Top-level balanced `{...}` spans in free text (string-aware), at most `max`.
fn balanced_objects(text: &str, max: usize) -> Vec<String> {
    let (mut out, mut depth, mut start, mut in_str, mut esc) = (vec![], 0usize, 0usize, false, false);
    for (i, ch) in text.char_indices() {
        if in_str {
            if esc {
                esc = false;
            } else if ch == '\\' {
                esc = true;
            } else if ch == '"' {
                in_str = false;
            }
            continue;
        }
        match ch {
            '"' if depth > 0 => in_str = true,
            '{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    out.push(text[start..=i].to_string());
                    if out.len() >= max {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Parse JSON the way small models write it: raw newlines/tabs inside strings are escaped and
/// trailing commas dropped before giving up.
pub fn lenient_json(src: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(src) {
        return Some(v);
    }
    let mut out = String::with_capacity(src.len() + 16);
    let (mut in_str, mut esc) = (false, false);
    for ch in src.chars() {
        if in_str {
            if esc {
                esc = false;
                if !matches!(ch, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u') {
                    out.pop(); // an invalid escape such as \' : keep the character, drop the backslash
                }
                out.push(ch);
                continue;
            }
            match ch {
                '\\' => {
                    esc = true;
                    out.push(ch);
                }
                '"' => {
                    in_str = false;
                    out.push(ch);
                }
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push(' '),
                c => out.push(c),
            }
        } else {
            if ch == '"' {
                in_str = true;
            }
            out.push(ch);
        }
    }
    let re = regex::Regex::new(r",\s*([}\]])").unwrap();
    let fixed = re.replace_all(&out, "$1").into_owned();
    serde_json::from_str(&fixed).ok()
}

/// Ollama's native chat API: unlike its /v1 shim, it honours `num_ctx`, so long missions are
/// not silently cut to the default 2-4k token window.
fn ollama(conn: &Connection, model: &str, req: &ChatReq, timeout: Duration) -> Result<ChatResp> {
    let mut msgs = vec![];
    if !req.system.is_empty() {
        msgs.push(json!({"role": "system", "content": req.system}));
    }
    for m in req.messages {
        match m {
            Msg::User(t) => msgs.push(json!({"role": "user", "content": t})),
            Msg::Assistant { text, calls } => {
                let mut o = json!({"role": "assistant", "content": text});
                if !calls.is_empty() {
                    o["tool_calls"] = json!(calls.iter().map(|c| json!({"function": {"name": c.name, "arguments": c.args}})).collect::<Vec<_>>());
                }
                msgs.push(o);
            }
            Msg::Tools(rs) => {
                for r in rs {
                    msgs.push(json!({"role": "tool", "content": r.content, "tool_name": r.name}));
                }
            }
        }
    }
    let mut options = json!({"num_ctx": conn.window(), "num_predict": req.max_tokens});
    if let Some(t) = req.temperature.or(conn.temperature) {
        options["temperature"] = json!(t);
    }
    let mut body = json!({"model": model, "messages": msgs, "stream": false, "options": options});
    if !req.tools.is_empty() {
        body["tools"] = json!(req.tools.iter().map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}})).collect::<Vec<_>>());
    }
    let base = conn.base();
    let base = base.strip_suffix("/v1").unwrap_or(&base);
    let v = post(&format!("{base}/api/chat"), &[], &body, timeout)?;
    if let Some(e) = v.get("error").and_then(|e| e.as_str()) {
        bail!("ollama: {e}");
    }
    let msg = &v["message"];
    let mut out = ChatResp {
        text: msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string(),
        stop: v.get("done_reason").and_then(|d| d.as_str()).unwrap_or("").to_string(),
        input_tokens: v["prompt_eval_count"].as_u64().unwrap_or(0),
        output_tokens: v["eval_count"].as_u64().unwrap_or(0),
        ..Default::default()
    };
    for (i, c) in msg.get("tool_calls").and_then(|t| t.as_array()).cloned().unwrap_or_default().iter().enumerate() {
        let f = &c["function"];
        let args = match f.get("arguments").cloned().unwrap_or(json!({})) {
            Value::String(s) => serde_json::from_str(&s).unwrap_or(json!({})),
            o => o,
        };
        out.calls.push(ToolCall { id: format!("ollama_{i}"), name: f["name"].as_str().unwrap_or("").into(), args, extra: None });
    }
    Ok(out)
}

fn openai(conn: &Connection, model: &str, req: &ChatReq, timeout: Duration) -> Result<ChatResp> {
    let mut msgs = vec![];
    if !req.system.is_empty() {
        msgs.push(json!({"role": "system", "content": req.system}));
    }
    for m in req.messages {
        match m {
            Msg::User(t) => msgs.push(json!({"role": "user", "content": t})),
            Msg::Assistant { text, calls } => {
                // An empty string, not null: strict OpenAI-compatible servers (llama.cpp, vLLM
                // validators) reject a null assistant content even when tool_calls are present.
                let mut o = json!({"role": "assistant", "content": text});
                if !calls.is_empty() {
                    o["tool_calls"] = json!(calls
                        .iter()
                        .map(|c| json!({"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.args.to_string()}}))
                        .collect::<Vec<_>>());
                }
                msgs.push(o);
            }
            Msg::Tools(rs) => {
                for r in rs {
                    msgs.push(json!({"role": "tool", "tool_call_id": r.id, "content": r.content}));
                }
            }
        }
    }
    let base = conn.base();
    let mut body = json!({"model": model, "messages": msgs});
    if base.contains("api.openai.com") {
        body["max_completion_tokens"] = json!(req.max_tokens);
    } else {
        body["max_tokens"] = json!(req.max_tokens);
    }
    if let Some(t) = req.temperature.or(conn.temperature) {
        body["temperature"] = json!(t);
    }
    if !req.tools.is_empty() {
        body["tools"] = json!(req
            .tools
            .iter()
            .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}))
            .collect::<Vec<_>>());
    }
    let key = conn.key();
    let mut headers = vec![];
    if !key.is_empty() {
        headers.push(("authorization", format!("Bearer {key}")));
    }
    let v = post(&format!("{base}/chat/completions"), &headers, &body, timeout)?;
    let choice = v.get("choices").and_then(|c| c.get(0)).ok_or_else(|| anyhow!("no choices in response: {}", crate::util::take_chars(&v.to_string(), 400)))?;
    let msg = &choice["message"];
    // OpenAI caches long prompts automatically; prompt_tokens includes the cached part.
    let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0);
    let mut out = ChatResp {
        text: msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string(),
        stop: choice.get("finish_reason").and_then(|f| f.as_str()).unwrap_or("").to_string(),
        input_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0).saturating_sub(cached),
        output_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
        cached_tokens: cached,
        ..Default::default()
    };
    for (i, c) in msg.get("tool_calls").and_then(|t| t.as_array()).cloned().unwrap_or_default().iter().enumerate() {
        let f = &c["function"];
        let raw = f.get("arguments").cloned().unwrap_or(json!("{}"));
        let args = match raw {
            Value::String(s) => serde_json::from_str(&s).unwrap_or_else(|_| json!({"_raw": s})),
            o => o,
        };
        out.calls.push(ToolCall {
            id: c.get("id").and_then(|x| x.as_str()).map(|s| s.to_string()).unwrap_or_else(|| format!("call_{i}")),
            name: f.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            args,
            extra: None,
        });
    }
    Ok(out)
}

fn anthropic(conn: &Connection, model: &str, req: &ChatReq, timeout: Duration) -> Result<ChatResp> {
    let mut msgs: Vec<Value> = vec![];
    for m in req.messages {
        match m {
            Msg::User(t) => msgs.push(json!({"role": "user", "content": [{"type": "text", "text": t}]})),
            Msg::Assistant { text, calls } => {
                let mut content = vec![];
                if !text.trim().is_empty() {
                    content.push(json!({"type": "text", "text": text}));
                }
                for c in calls {
                    content.push(json!({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}));
                }
                if content.is_empty() {
                    content.push(json!({"type": "text", "text": "(no output)"}));
                }
                msgs.push(json!({"role": "assistant", "content": content}));
            }
            Msg::Tools(rs) => msgs.push(json!({"role": "user", "content": rs.iter().map(|r| json!({
                "type": "tool_result", "tool_use_id": r.id, "content": r.content, "is_error": r.is_error,
            })).collect::<Vec<_>>()})),
        }
    }
    // Prompt caching: breakpoints on the tools, the system prompt and the last two user turns,
    // so each turn re-reads the conversation so far from cache.
    let cache = conn.prompt_cache.unwrap_or(true);
    let eph = json!({"type": "ephemeral"});
    if cache {
        let user_idx: Vec<usize> = msgs.iter().enumerate().filter(|(_, m)| m["role"] == "user").map(|(i, _)| i).rev().take(2).collect();
        for i in user_idx {
            if let Some(last) = msgs[i]["content"].as_array_mut().and_then(|c| c.last_mut()) {
                last["cache_control"] = eph.clone();
            }
        }
    }
    let mut body = json!({"model": model, "max_tokens": req.max_tokens, "messages": msgs});
    if !req.system.is_empty() {
        body["system"] = if cache { json!([{"type": "text", "text": req.system, "cache_control": eph}]) } else { json!(req.system) };
    }
    if let Some(t) = req.temperature.or(conn.temperature) {
        body["temperature"] = json!(t.min(1.0));
    }
    if !req.tools.is_empty() {
        let mut tools: Vec<Value> = req.tools.iter().map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.parameters})).collect();
        if cache {
            if let Some(l) = tools.last_mut() {
                l["cache_control"] = eph.clone();
            }
        }
        body["tools"] = json!(tools);
    }
    let headers = vec![("x-api-key", conn.key()), ("anthropic-version", "2023-06-01".to_string())];
    let v = post(&format!("{}/v1/messages", conn.base()), &headers, &body, timeout)?;
    let mut out = ChatResp {
        stop: v.get("stop_reason").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        input_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: v["usage"]["output_tokens"].as_u64().unwrap_or(0),
        cached_tokens: v["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0),
        cache_write_tokens: v["usage"]["cache_creation_input_tokens"].as_u64().unwrap_or(0),
        ..Default::default()
    };
    for b in v.get("content").and_then(|c| c.as_array()).cloned().unwrap_or_default() {
        match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => out.text.push_str(b["text"].as_str().unwrap_or("")),
            Some("tool_use") => out.calls.push(ToolCall {
                id: b["id"].as_str().unwrap_or("").into(),
                name: b["name"].as_str().unwrap_or("").into(),
                args: b.get("input").cloned().unwrap_or(json!({})),
                extra: None,
            }),
            _ => {}
        }
    }
    Ok(out)
}

fn gemini(conn: &Connection, model: &str, req: &ChatReq, timeout: Duration) -> Result<ChatResp> {
    let mut contents: Vec<Value> = vec![];
    for m in req.messages {
        match m {
            Msg::User(t) => contents.push(json!({"role": "user", "parts": [{"text": t}]})),
            Msg::Assistant { text, calls } => {
                let mut parts = vec![];
                if !text.is_empty() {
                    parts.push(json!({"text": text}));
                }
                for c in calls {
                    parts.push(c.extra.clone().unwrap_or_else(|| json!({"functionCall": {"name": c.name, "args": c.args}})));
                }
                if parts.is_empty() {
                    parts.push(json!({"text": "(no output)"}));
                }
                contents.push(json!({"role": "model", "parts": parts}));
            }
            Msg::Tools(rs) => contents.push(json!({"role": "user", "parts": rs.iter().map(|r| json!({
                "functionResponse": {"name": r.name, "response": {"content": r.content, "is_error": r.is_error}}
            })).collect::<Vec<_>>()})),
        }
    }
    let mut body = json!({"contents": contents, "generationConfig": {"maxOutputTokens": req.max_tokens}});
    if let Some(t) = req.temperature.or(conn.temperature) {
        body["generationConfig"]["temperature"] = json!(t);
    }
    if !req.system.is_empty() {
        body["systemInstruction"] = json!({"parts": [{"text": req.system}]});
    }
    if !req.tools.is_empty() {
        body["tools"] = json!([{"functionDeclarations": req.tools.iter().map(|t| json!({"name": t.name, "description": t.description, "parameters": t.parameters})).collect::<Vec<_>>()}]);
    }
    let headers = vec![("x-goog-api-key", conn.key())];
    let url = format!("{}/v1beta/models/{model}:generateContent", conn.base());
    let v = post(&url, &headers, &body, timeout)?;
    let cand = v.get("candidates").and_then(|c| c.get(0)).ok_or_else(|| anyhow!("no candidates in response: {}", crate::util::take_chars(&v.to_string(), 400)))?;
    let mut out = ChatResp {
        stop: cand.get("finishReason").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        input_tokens: v["usageMetadata"]["promptTokenCount"].as_u64().unwrap_or(0).saturating_sub(v["usageMetadata"]["cachedContentTokenCount"].as_u64().unwrap_or(0)),
        cached_tokens: v["usageMetadata"]["cachedContentTokenCount"].as_u64().unwrap_or(0),
        output_tokens: v["usageMetadata"]["candidatesTokenCount"].as_u64().unwrap_or(0)
            + v["usageMetadata"]["thoughtsTokenCount"].as_u64().unwrap_or(0),
        ..Default::default()
    };
    for (i, p) in cand["content"]["parts"].as_array().cloned().unwrap_or_default().into_iter().enumerate() {
        if p.get("thought").and_then(|t| t.as_bool()).unwrap_or(false) {
            continue;
        }
        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
            out.text.push_str(t);
        }
        if let Some(fc) = p.get("functionCall") {
            out.calls.push(ToolCall {
                id: format!("gemini_{i}"),
                name: fc["name"].as_str().unwrap_or("").into(),
                args: fc.get("args").cloned().unwrap_or(json!({})),
                extra: Some(p.clone()),
            });
        }
    }
    Ok(out)
}

/// Deterministic offline stand-in: writes one answer file, then ends with a reply that every
/// stage of the method can parse (notes, a choice of candidate A, no library entry).
fn mock(req: &ChatReq) -> ChatResp {
    // Optional delay so demos and UI tests can watch a run move (HARNESS2_MOCK_DELAY_MS).
    if let Some(ms) = crate::config::env_int("HARNESS2_MOCK_DELAY_MS", 0).ok().filter(|m| *m > 0) {
        let _ = crate::util::sleep_cancellable(Duration::from_millis(ms.min(60_000) as u64));
    }
    let used_tools = req.messages.iter().any(|m| matches!(m, Msg::Tools(_)));
    let has_write = req.tools.iter().any(|t| t.name == "write_file");
    let task: String = req
        .messages
        .iter()
        .find_map(|m| if let Msg::User(t) = m { Some(t.clone()) } else { None })
        .unwrap_or_default();
    if has_write && !used_tools && !req.system.contains("MISSION.md") {
        return ChatResp {
            calls: vec![ToolCall {
                id: "mock_1".into(),
                name: "write_file".into(),
                args: json!({"path": "output/answer.md", "content": format!("Mock answer.\n\nTask excerpt: {}", crate::util::take_chars(&task, 200))}),
                extra: None,
            }],
            input_tokens: 10,
            output_tokens: 10,
            ..Default::default()
        };
    }
    ChatResp {
        text: "<notes>KEEP_BASE -- the mock model proposes no change.</notes>\n<why>Mock selection.</why>\n```choice\nA\n```\nNO_ENTRY".into(),
        input_tokens: 10,
        output_tokens: 10,
        stop: "stop".into(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn text_tool_calls_and_think() {
        use super::*;
        let tools = vec![ToolDef { name: "write_file".into(), description: String::new(), parameters: json!({}) }];
        let c = text_tool_calls("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a\"}}\n</tool_call>", &tools);
        assert_eq!(c[0].name, "write_file");
        assert_eq!(c[0].args["path"], "a");
        let c = text_tool_calls("{\"name\": \"write_file\", \"parameters\": {\"path\": \"b\"}}", &tools);
        assert_eq!(c[0].args["path"], "b");
        let c = text_tool_calls("```json\n{\"name\": \"write_file\", \"arguments\": \"{\\\"path\\\": \\\"c\\\"}\"}\n```", &tools);
        assert_eq!(c[0].args["path"], "c");
        assert!(text_tool_calls("{\"name\": \"rm_rf\"}", &tools).is_empty());
        // raw newline inside a string, and arguments given as a JSON string, as Qwen-3B writes it
        let c = text_tool_calls("<tool_call>\n{\"name\": \"write_file\", \"arguments\": \"{\\\"content\\\":\\\"line1\nline2\\\",\\\"path\\\":\\\"s.py\\\"}\"}\n</tool_call>", &tools);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].args["path"], "s.py");
        assert_eq!(c[0].args["content"], "line1\nline2");
        assert_eq!(lenient_json("{\"a\": [1, 2,],}").unwrap()["a"][1], 2);
        assert!(text_tool_calls("just prose", &tools).is_empty());
        // an extra closing brace after the call, as Qwen-3B wrote it three times in one session
        let c = text_tool_calls("I will write it now.\n<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"content\":\"```md\\n# T\\n```\", \"path\":\"output/answer.md\"}}}", &tools);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].args["path"], "output/answer.md");
        let c = text_tool_calls("<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.py\", \"content\": \"x\"}\n}()\n", &tools);
        assert_eq!(c.len(), 1, "junk after the object");
        // Llama-1B: `function` as the name, arguments wrapped like a schema
        let rf = vec![ToolDef { name: "read_file".into(), description: String::new(), parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}) }];
        let c = text_tool_calls("{\"function\": \"read_file\", \"parameters\": {\"properties\": {\"path\": \"sales.csv\"}}}", &rf);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].args["path"], "sales.csv");
        assert_eq!(strip_think("<think>hmm</think>\nanswer"), "answer");
    }

    #[test]
    fn proxy_bypass() {
        assert!(super::bypass_proxy("http://127.0.0.1:11434/v1"));
        assert!(super::bypass_proxy("http://localhost/v1"));
        assert!(!super::bypass_proxy("https://api.openai.com/v1"));
    }
}

/// List the models an endpoint serves (for the UI's model picker).
pub fn list_models(conn: &Connection) -> Result<Vec<String>> {
    let key = conn.key();
    let mut out: Vec<String> = match conn.provider.as_str() {
        "ollama" => {
            let base = conn.base();
            let base = base.strip_suffix("/v1").unwrap_or(&base).to_string();
            let v = get_json(&format!("{base}/api/tags"), &[])?;
            v["models"].as_array().cloned().unwrap_or_default().iter().filter_map(|m| m["name"].as_str().map(|s| s.to_string())).collect()
        }
        "openai" => {
            let mut h = vec![];
            if !key.is_empty() {
                h.push(("authorization", format!("Bearer {key}")));
            }
            let v = get_json(&format!("{}/models", conn.base()), &h)?;
            v["data"].as_array().cloned().unwrap_or_default().iter().filter_map(|m| m["id"].as_str().map(|s| s.to_string())).collect()
        }
        "anthropic" => {
            let v = get_json(&format!("{}/v1/models?limit=100", conn.base()), &[("x-api-key", key), ("anthropic-version", "2023-06-01".into())])?;
            v["data"].as_array().cloned().unwrap_or_default().iter().filter_map(|m| m["id"].as_str().map(|s| s.to_string())).collect()
        }
        "gemini" => {
            let v = get_json(&format!("{}/v1beta/models?pageSize=200", conn.base()), &[("x-goog-api-key", key)])?;
            v["models"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter_map(|m| m["name"].as_str().map(|s| s.trim_start_matches("models/").to_string()))
                .collect()
        }
        "mock" => vec!["mock-1".into()],
        p => bail!("unknown provider {p:?}"),
    };
    out.sort();
    Ok(out)
}

/// Probe the usual local model servers; returns those that answer, with their models.
pub fn discover_local() -> Vec<Value> {
    let cands: &[(&str, &str, &str)] = &[
        ("ollama", "ollama", "http://127.0.0.1:11434"),
        ("lmstudio", "openai", "http://127.0.0.1:1234/v1"),
        ("llamacpp", "openai", "http://127.0.0.1:8080/v1"),
        ("llamacpp-8081", "openai", "http://127.0.0.1:8081/v1"),
        ("vllm", "openai", "http://127.0.0.1:8000/v1"),
        ("textgen", "openai", "http://127.0.0.1:5000/v1"),
        ("localai", "openai", "http://127.0.0.1:8088/v1"),
    ];
    // Port open? Then ask for models; a server busy generating may not answer in time, and is
    // still reported (with a note) rather than hidden.
    let probe = |name: &str, provider: &str, base: &str| -> Option<Value> {
        let hostport = base.split("://").nth(1)?.split('/').next()?;
        let addr: std::net::SocketAddr = hostport.parse().ok()?;
        std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(400)).ok()?;
        let url = if provider == "ollama" { format!("{base}/api/tags") } else { format!("{base}/models") };
        let agent = ureq::AgentBuilder::new().timeout_connect(Duration::from_millis(800)).timeout_read(Duration::from_secs(8)).try_proxy_from_env(false).build();
        let listed: Option<Vec<String>> = (|| {
            let v: Value = serde_json::from_str(&agent.get(&url).call().ok()?.into_string().ok()?).ok()?;
            Some(if provider == "ollama" {
                v["models"].as_array()?.iter().filter_map(|m| m["name"].as_str().map(|s| s.to_string())).collect()
            } else {
                v["data"].as_array()?.iter().filter_map(|m| m["id"].as_str().map(|s| s.to_string())).collect()
            })
        })();
        let note = if listed.is_none() { "port is open but the model list did not come back (server busy, or not this kind of server)" } else { "" };
        Some(json!({"name": name, "provider": provider, "base_url": base, "models": listed.unwrap_or_default(), "note": note}))
    };
    crate::util::par_map(cands.to_vec(), cands.len(), |(name, provider, base)| probe(name, provider, base))
        .into_iter()
        .flatten()
        .collect()
}

/// A one-line round trip: proves the key, the URL and the model id all work.
pub fn ping(conn: &Connection, model: &str) -> Result<String> {
    let msgs = vec![Msg::User("Reply with the single word: ready".into())];
    let r = chat(conn, model, &ChatReq { system: "", messages: &msgs, tools: &[], max_tokens: 64, temperature: None }, Duration::from_secs(60))?;
    Ok(r.text.trim().to_string())
}

#[cfg(test)]
mod real_reply {
    /// The exact reply Qwen2.5-3B produced in a local test run, which strict JSON rejects.
    #[test]
    fn qwen_reply_with_raw_newlines_parses() {
        let text = include_str!("../../tests/golden/qwen_broken_call.txt");
        let tools = vec![super::ToolDef { name: "write_file".into(), description: String::new(), parameters: serde_json::json!({}) }];
        let c = super::text_tool_calls(text, &tools);
        assert_eq!(c.len(), 1, "{text}");
        assert_eq!(c[0].args["path"], "scripts/regions_analysis.py");
        assert!(c[0].args["content"].as_str().unwrap().contains("def find_percentage_change"));
    }
}

/// Embed texts with an embedding model id (`connection/model`). OpenAI-compatible servers use
/// `/embeddings`, Ollama `/api/embed`, Gemini `batchEmbedContents`; `mock` hashes words into a
/// small vector so offline tests can exercise semantic recall.
pub fn embed(model_id: &str, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(vec![]);
    }
    let (conn, model) = super::connections::resolve(model_id)?;
    let t = Duration::from_secs(60);
    let vecs: Vec<Vec<f32>> = match conn.provider.as_str() {
        "mock" => texts.iter().map(|s| mock_embed(s)).collect(),
        "ollama" => {
            let base = conn.base();
            let base = base.strip_suffix("/v1").unwrap_or(&base).to_string();
            let v = post(&format!("{base}/api/embed"), &[], &json!({"model": model, "input": texts}), t)?;
            v["embeddings"].as_array().cloned().unwrap_or_default().iter().map(to_f32).collect()
        }
        "gemini" => {
            let reqs: Vec<Value> = texts.iter().map(|s| json!({"model": format!("models/{model}"), "content": {"parts": [{"text": s}]}})).collect();
            let v = post(&format!("{}/v1beta/models/{model}:batchEmbedContents", conn.base()), &[("x-goog-api-key", conn.key())], &json!({"requests": reqs}), t)?;
            v["embeddings"].as_array().cloned().unwrap_or_default().iter().map(|e| to_f32(&e["values"])).collect()
        }
        "anthropic" => bail!("Anthropic has no embeddings API; use an OpenAI-compatible, Ollama or Gemini embedding model"),
        _ => {
            let key = conn.key();
            let mut h = vec![];
            if !key.is_empty() {
                h.push(("authorization", format!("Bearer {key}")));
            }
            let v = post(&format!("{}/embeddings", conn.base()), &h, &json!({"model": model, "input": texts}), t)?;
            v["data"].as_array().cloned().unwrap_or_default().iter().map(|d| to_f32(&d["embedding"])).collect()
        }
    };
    if vecs.len() != texts.len() || vecs.iter().any(|v| v.is_empty()) {
        bail!("embedding model returned {} vector(s) for {} text(s)", vecs.len(), texts.len());
    }
    Ok(vecs)
}

fn to_f32(v: &Value) -> Vec<f32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect()).unwrap_or_default()
}

fn mock_embed(s: &str) -> Vec<f32> {
    use sha2::Digest as _;
    let mut v = vec![0f32; 64];
    for w in s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
        // crude stemming so "delimiters" and "delimiter" meet
        let w = w.trim_end_matches('s');
        let h = sha2::Sha256::digest(w.as_bytes());
        v[(h[0] as usize) % 64] += 1.0;
    }
    v
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b) {
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 { 0.0 } else { d / (na.sqrt() * nb.sqrt()) }
}

#[cfg(test)]
mod prose_call_tests {
    use super::*;

    #[test]
    fn bare_objects_in_prose_become_calls() {
        let tools: Vec<ToolDef> = ["skill", "finish", "write_file"].iter().map(|n| ToolDef { name: n.to_string(), description: String::new(), parameters: json!({}) }).collect();
        let text = "Sure.\n\n{\n    \"type\": \"function\",\n    \"name\": \"skill\",\n    \"parameters\": {\n        \"name\": \"sleekhtml\"\n    }\n}\n\n{\n    \"type\": \"function\",\n    \"name\": \"finish\",\n    \"parameters\": {\"answer\": \"done {not a brace}\"}\n}\n";
        let calls = text_tool_calls(text, &tools);
        assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), vec!["skill", "finish"]);
        assert_eq!(calls[0].args["name"], "sleekhtml");
        assert_eq!(calls[1].args["answer"], "done {not a brace}");
    }
}
