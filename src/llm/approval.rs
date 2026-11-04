//! Human approval for risky tool calls (gate mode "ask", or an MCP server marked "ask"), and
//! outbound notifications.
//!
//! The agent thread files a request and waits; the UI lists pending requests with Approve /
//! Deny. No answer within the timeout (HARNESS2_APPROVAL_TIMEOUT, default 600 s) is a deny.
//! When HARNESS2_NOTIFY_URL is set, each request and each finished run is POSTed there as JSON
//! with a `text` field (Slack and Discord incoming webhooks, ntfy, or your own endpoint).

use once_cell::sync::Lazy;
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct Pending {
    pub id: String,
    pub created: String,
    pub tool: String,
    pub call: String,
    pub rule: String,
    pub reason: String,
    pub mode: String,
    pub node: String,
    #[serde(skip)]
    pub decision: Option<bool>,
}

static PENDING: Lazy<Mutex<Vec<Pending>>> = Lazy::new(|| Mutex::new(vec![]));

pub fn timeout() -> Duration {
    Duration::from_secs(crate::config::env_int("HARNESS2_APPROVAL_TIMEOUT", 600).unwrap_or(600).clamp(5, 86_400) as u64)
}

/// File a request and block until a person answers, the run is cancelled, or it times out.
pub fn ask(tool: &str, call: &str, rule: &str, reason: &str) -> (bool, String) {
    let ctx = crate::log::ctx();
    let id = format!("a{}", &crate::util::sha256_hex(format!("{call}{}", crate::util::now_iso()).as_bytes())[..10]);
    let p = Pending { id: id.clone(), created: crate::util::now_iso(), tool: tool.into(), call: call.into(), rule: rule.into(), reason: reason.into(), mode: ctx.mode.clone(), node: ctx.node.clone(), decision: None };
    PENDING.lock().unwrap().push(p.clone());
    crate::log::event("gate", "ask", json!({"id": id, "tool": tool, "call": call, "rule": rule, "reason": reason}));
    notify(&format!("harnessmith: approval needed at {} — {call} (rule {rule}: {reason})", if ctx.node.is_empty() { "a run" } else { &ctx.node }), json!({"event": "approval", "id": id, "call": call, "rule": rule}));
    let until = Instant::now() + timeout();
    let decision = loop {
        if let Some(d) = PENDING.lock().unwrap().iter().find(|x| x.id == id).and_then(|x| x.decision) {
            break Some(d);
        }
        if crate::log::cancelled() || Instant::now() >= until {
            break None;
        }
        std::thread::sleep(Duration::from_millis(400));
    };
    PENDING.lock().unwrap().retain(|x| x.id != id);
    match decision {
        Some(true) => (true, "approved by a person".into()),
        Some(false) => (false, "denied by a person".into()),
        None => (false, format!("no answer within {}s, so denied", timeout().as_secs())),
    }
}

pub fn list() -> Vec<Value> {
    PENDING.lock().unwrap().iter().filter(|p| p.decision.is_none()).map(|p| serde_json::to_value(p).unwrap_or(Value::Null)).collect()
}

pub fn decide(id: &str, allow: bool) -> bool {
    let mut g = PENDING.lock().unwrap();
    match g.iter_mut().find(|p| p.id == id && p.decision.is_none()) {
        Some(p) => {
            p.decision = Some(allow);
            true
        }
        None => false,
    }
}

/// POST `{text, ...extra}` to HARNESS2_NOTIFY_URL in the background; failures are only logged.
pub fn notify(text: &str, extra: Value) {
    let Some(url) = crate::config::env("HARNESS2_NOTIFY_URL").filter(|u| u.starts_with("http")) else { return };
    let mut body = json!({"text": text});
    if let (Some(b), Some(e)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in e {
            b.insert(k.clone(), v.clone());
        }
    }
    std::thread::spawn(move || {
        let r = ureq::post(&url).timeout(Duration::from_secs(15)).send_json(body);
        if let Err(e) = r {
            crate::say!("notify: could not reach HARNESS2_NOTIFY_URL: {e}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decision_unblocks_the_waiting_agent() {
        let h = std::thread::spawn(|| ask("bash", "bash(sudo ls)", "sudo", "privilege escalation"));
        let id = loop {
            if let Some(p) = list().into_iter().find(|p| p["call"] == "bash(sudo ls)") {
                break p["id"].as_str().unwrap().to_string();
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        assert!(decide(&id, true));
        assert_eq!(h.join().unwrap().0, true);
        assert!(!decide(&id, false), "answered requests are gone");
    }
}
