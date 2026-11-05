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
    pub cached_tokens: u64,
}

static STATS: Lazy<Mutex<BTreeMap<String, Stat>>> = Lazy::new(|| Mutex::new(BTreeMap::new()));

fn ewma(old: f64, new: f64) -> f64 {
    if old == 0.0 { new } else { old * 0.7 + new * 0.3 }
}

/// Today's spend per connection (UTC day), persisted in `<runs>/spend.json`.
static SPEND: Lazy<Mutex<(String, BTreeMap<String, f64>)>> = Lazy::new(|| {
    let p = crate::config::get().runs.join("spend.json");
    let v: Value = std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
    let day = v["day"].as_str().unwrap_or("").to_string();
    let m = v["spend"].as_object().map(|o| o.iter().map(|(k, x)| (k.clone(), x.as_f64().unwrap_or(0.0))).collect()).unwrap_or_default();
    Mutex::new((day, m))
});

fn today() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

pub fn record_cost(conn: &str, usd: f64) {
    if usd <= 0.0 {
        return;
    }
    let mut g = SPEND.lock().unwrap();
    let t = today();
    if g.0 != t {
        *g = (t.clone(), BTreeMap::new());
    }
    *g.1.entry(conn.to_string()).or_default() += usd;
    let body = json!({"day": g.0, "spend": g.1});
    let _ = std::fs::write(crate::config::get().runs.join("spend.json"), body.to_string());
}

pub fn spent_today(conn: &str) -> f64 {
    let g = SPEND.lock().unwrap();
    if g.0 != today() { 0.0 } else { g.1.get(conn).copied().unwrap_or(0.0) }
}

/// Whether a connection has used up its daily cap.
pub fn over_cap(c: &super::connections::Connection) -> bool {
    c.daily_usd.map(|cap| cap > 0.0 && spent_today(&c.name) >= cap).unwrap_or(false)
}

pub fn record_cached(conn: &str, cached: u64) {
    if cached > 0 {
        STATS.lock().unwrap().entry(conn.to_string()).or_default().cached_tokens += cached;
    }
}

pub fn record_ok(conn: &str, took: Duration, in_tok: u64, out_tok: u64) {
    let was_open = STATS.lock().unwrap().get(conn).map(|s| s.open_until.is_some()).unwrap_or(false);
    record_ok_inner(conn, took, in_tok, out_tok);
    if was_open {
        save();
    }
}

fn record_ok_inner(conn: &str, took: Duration, in_tok: u64, out_tok: u64) {
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
        drop(g);
        save();
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
        s.last_error.clear();
        // A reset starts the record over, so old failures stop counting against it.
        s.ok = 0;
        s.fail = 0;
        s.trips = 0;
        s.last_fail = None;
    }
    // Persist at once, so a restart does not bring back a breaker someone just reset.
    save();
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

/// Save what matters across a restart: success/failure counts, speed, and open breakers
/// (as seconds left), so a server restart neither forgets a dead endpoint nor its measured speed.
pub fn save() {
    let g = STATS.lock().unwrap();
    let now = Instant::now();
    let m: serde_json::Map<String, Value> = g
        .iter()
        .map(|(k, s)| {
            (k.clone(), json!({"ok": s.ok, "fail": s.fail, "consecutive_fail": s.consecutive_fail, "trips": s.trips, "latency_ms": s.latency_ms, "out_tok_s": s.out_tok_s,
                "last_error": s.last_error, "open_for_s": s.open_until.map(|t| t.saturating_duration_since(now).as_secs()).unwrap_or(0)}))
        })
        .collect();
    drop(g);
    let _ = std::fs::write(crate::config::get().runs.join("health.json"), Value::Object(m).to_string());
}

/// Restore what `save` wrote (missing or broken file: start fresh).
pub fn load() {
    let Ok(text) = std::fs::read_to_string(crate::config::get().runs.join("health.json")) else { return };
    let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&text) else { return };
    let mut g = STATS.lock().unwrap();
    for (k, v) in m {
        let s = g.entry(k).or_default();
        let u = |f: &str| v[f].as_u64().unwrap_or(0);
        s.ok = u("ok");
        s.fail = u("fail");
        s.consecutive_fail = u("consecutive_fail") as u32;
        s.trips = u("trips") as u32;
        s.latency_ms = v["latency_ms"].as_f64().unwrap_or(0.0);
        s.out_tok_s = v["out_tok_s"].as_f64().unwrap_or(0.0);
        s.last_error = v["last_error"].as_str().unwrap_or("").into();
        let open = u("open_for_s");
        s.open_until = (open > 0).then(|| Instant::now() + Duration::from_secs(open));
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
                        "in_tokens": s.in_tokens, "out_tokens": s.out_tokens, "cached_tokens": s.cached_tokens,
                        "spent_today_usd": (spent_today(k) * 10000.0).round() / 10000.0,
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
