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
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub stop: String,
}

pub struct ChatReq<'a> {
    pub system: &'a str,
    pub messages: &'a [Msg],
    pub tools: &'a [ToolDef],
    pub max_tokens: u32,
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

fn post(url: &str, headers: &[(&str, String)], body: &Value, timeout: Duration) -> Result<Value> {
    let mut last = String::new();
    for attempt in 0..4u32 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_secs(2u64.pow(attempt)));
        }
        let mut req = agent_for(url, timeout).post(url).set("content-type", "application/json");
        for (k, v) in headers {
            req = req.set(k, v);
        }
        match req.send_json(body.clone()) {
            Ok(resp) => {
                let text = resp.into_string()?;
                return serde_json::from_str(&text).map_err(|e| anyhow!("invalid JSON from {url}: {e}: {}", crate::util::take_chars(&text, 300)));
            }
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                last = format!("HTTP {code} from {url}: {}", crate::util::take_chars(&body, 600));
                if ![408, 409, 429, 500, 502, 503, 504, 529].contains(&code) {
                    bail!(last);
                }
            }
            Err(e) => last = format!("transport error calling {url}: {e}"),
        }
    }
    bail!(last)
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

/// One model call.
pub fn chat(conn: &Connection, model: &str, req: &ChatReq, timeout: Duration) -> Result<ChatResp> {
    match conn.provider.as_str() {
        "openai" | "ollama" => openai(conn, model, req, timeout),
        "anthropic" => anthropic(conn, model, req, timeout),
        "gemini" => gemini(conn, model, req, timeout),
        "mock" => Ok(mock(req)),
        p => bail!("unknown provider {p:?}"),
    }
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
                let mut o = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) }});
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
    if let Some(t) = conn.temperature {
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
    let mut out = ChatResp {
        text: msg.get("content").and_then(|c| c.as_str()).unwrap_or("").to_string(),
        stop: choice.get("finish_reason").and_then(|f| f.as_str()).unwrap_or("").to_string(),
        input_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        output_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
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
    let mut body = json!({"model": model, "max_tokens": req.max_tokens, "messages": msgs});
    if !req.system.is_empty() {
        body["system"] = json!(req.system);
    }
    if let Some(t) = conn.temperature {
        body["temperature"] = json!(t);
    }
    if !req.tools.is_empty() {
        body["tools"] = json!(req.tools.iter().map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.parameters})).collect::<Vec<_>>());
    }
    let headers = vec![("x-api-key", conn.key()), ("anthropic-version", "2023-06-01".to_string())];
    let v = post(&format!("{}/v1/messages", conn.base()), &headers, &body, timeout)?;
    let mut out = ChatResp {
        stop: v.get("stop_reason").and_then(|s| s.as_str()).unwrap_or("").to_string(),
        input_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: v["usage"]["output_tokens"].as_u64().unwrap_or(0),
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
    if let Some(t) = conn.temperature {
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
        input_tokens: v["usageMetadata"]["promptTokenCount"].as_u64().unwrap_or(0),
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
        "openai" | "ollama" => {
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

/// A one-line round trip: proves the key, the URL and the model id all work.
pub fn ping(conn: &Connection, model: &str) -> Result<String> {
    let msgs = vec![Msg::User("Reply with the single word: ready".into())];
    let r = chat(conn, model, &ChatReq { system: "", messages: &msgs, tools: &[], max_tokens: 64 }, Duration::from_secs(60))?;
    Ok(r.text.trim().to_string())
}
