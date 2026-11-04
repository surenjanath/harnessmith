//! Connection health: what each model endpoint has been doing, and a circuit breaker.
//!
//! Every model call records its outcome here. After `TRIP_AFTER` consecutive failures a
//! connection's breaker opens for a cooldown that doubles on each re-trip (capped), during
//! which callers fail over to the next model in their chain. When the cooldown ends the
//! breaker is half-open: one call is let through, and success closes it.

use once_cell::sync::Lazy;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub const TRIP_AFTER: u32 = 3;
const BASE_COOLDOWN: Duration = Duration::from_secs(30);
const MAX_COOLDOWN: Duration = Duration::from_secs(600);

#[derive(Debug, Clone, Default)]
pub struct Stat {
    pub ok: u64,
    pub fail: u64,
    pub consecutive_fail: u32,
    pub trips: u32,
    pub open_until: Option<Instant>,
    pub last_error: String,
    pub last_ok: Option<Instant>,
    pub last_fail: Option<Instant>,
    /// Exponentially weighted averages over successful calls.
    pub latency_ms: f64,
    pub out_tok_s: f64,
    pub in_tokens: u64,
    pub out_tokens: u64,
}

static STATS: Lazy<Mutex<BTreeMap<String, Stat>>> = Lazy::new(|| Mutex::new(BTreeMap::new()));

fn ewma(old: f64, new: f64) -> f64 {
    if old == 0.0 { new } else { old * 0.7 + new * 0.3 }
}

pub fn record_ok(conn: &str, took: Duration, in_tok: u64, out_tok: u64) {
    let mut g = STATS.lock().unwrap();
    let s = g.entry(conn.to_string()).or_default();
    s.ok += 1;
    s.consecutive_fail = 0;
    s.open_until = None;
    s.last_ok = Some(Instant::now());
    s.latency_ms = ewma(s.latency_ms, took.as_millis() as f64);
    if took.as_secs_f64() > 0.0 && out_tok > 0 {
        s.out_tok_s = ewma(s.out_tok_s, out_tok as f64 / took.as_secs_f64());
    }
    s.in_tokens += in_tok;
    s.out_tokens += out_tok;
}

/// Record a failed call; returns true when this failure tripped the breaker.
pub fn record_fail(conn: &str, err: &str) -> bool {
    let mut g = STATS.lock().unwrap();
    let s = g.entry(conn.to_string()).or_default();
    s.fail += 1;
    s.consecutive_fail += 1;
    s.last_fail = Some(Instant::now());
    s.last_error = crate::util::take_chars(err, 300);
    if s.consecutive_fail >= TRIP_AFTER && s.open_until.map(|t| t <= Instant::now()).unwrap_or(true) {
        s.trips += 1;
        let cool = (BASE_COOLDOWN * 2u32.saturating_pow(s.trips.saturating_sub(1).min(5))).min(MAX_COOLDOWN);
        s.open_until = Some(Instant::now() + cool);
        return true;
    }
    false
}

/// Whether calls should skip this connection right now (breaker open, cooldown not over).
pub fn is_open(conn: &str) -> bool {
    STATS.lock().unwrap().get(conn).and_then(|s| s.open_until).map(|t| t > Instant::now()).unwrap_or(false)
}

/// Time until the breaker half-opens (zero if it is not open).
pub fn reopens_in(conn: &str) -> Duration {
    STATS
        .lock()
        .unwrap()
        .get(conn)
        .and_then(|s| s.open_until)
        .map(|t| t.saturating_duration_since(Instant::now()))
        .unwrap_or_default()
}

/// Close a breaker by hand (the Doctor's "reset" fix).
pub fn reset(conn: &str) {
    if let Some(s) = STATS.lock().unwrap().get_mut(conn) {
        s.consecutive_fail = 0;
        s.open_until = None;
    }
}

pub fn get(conn: &str) -> Stat {
    STATS.lock().unwrap().get(conn).cloned().unwrap_or_default()
}

/// The state shown in the UI.
pub fn state(s: &Stat) -> &'static str {
    match s.open_until {
        Some(t) if t > Instant::now() => "open",
        Some(_) => "half-open",
        None if s.consecutive_fail > 0 => "degraded",
        None if s.ok > 0 => "healthy",
        None => "unknown",
    }
}

pub fn snapshot() -> Value {
    let g = STATS.lock().unwrap();
    let ago = |t: Option<Instant>| t.map(|t| t.elapsed().as_secs()).map(Value::from).unwrap_or(Value::Null);
    Value::Object(
        g.iter()
            .map(|(k, s)| {
                (
                    k.clone(),
                    json!({
                        "state": state(s), "ok": s.ok, "fail": s.fail, "consecutive_fail": s.consecutive_fail, "trips": s.trips,
                        "reopens_in_s": s.open_until.map(|t| t.saturating_duration_since(Instant::now()).as_secs()),
                        "last_error": s.last_error, "last_ok_s_ago": ago(s.last_ok), "last_fail_s_ago": ago(s.last_fail),
                        "latency_ms": s.latency_ms.round(), "out_tok_s": (s.out_tok_s * 10.0).round() / 10.0,
                        "in_tokens": s.in_tokens, "out_tokens": s.out_tokens,
                    }),
                )
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn breaker_trips_and_resets() {
        let c = "test-breaker";
        assert!(!is_open(c));
        assert!(!record_fail(c, "x"));
        assert!(!record_fail(c, "x"));
        assert!(record_fail(c, "x"));
        assert!(is_open(c));
        assert_eq!(state(&get(c)), "open");
        record_ok(c, Duration::from_millis(100), 10, 10);
        assert!(!is_open(c));
        assert_eq!(state(&get(c)), "healthy");
    }
}
