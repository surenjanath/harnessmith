//! Budget accounting: R = 4k rollouts and P = 3k improvement stages.
//!
//! Base executions count toward each mode. Retries are separate invocation counts; selection
//! and reflection are recorded outside this ledger.

use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// R = 4k: k shared base + 2k candidates + k finals.
pub fn expected_rollouts(k: usize) -> usize {
    4 * k
}

/// P = 3k: k INV + k DOM + k compositions.
pub fn expected_sessions(k: usize) -> usize {
    3 * k
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub kind: &'static str,
    pub label: String,
    pub usd: f64,
    pub cached: bool,
    pub attempts: u32,
}

/// Everything one mode spent on one task, recorded as it happens.
#[derive(Debug, Clone)]
pub struct Ledger {
    pub mode: String,
    pub task_id: String,
    pub k: usize,
    pub entries: Vec<Entry>,
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

impl Ledger {
    pub fn new(mode: &str, task_id: &str, k: usize) -> Ledger {
        Ledger { mode: mode.into(), task_id: task_id.into(), k, entries: vec![] }
    }

    pub fn rollout(&mut self, label: &str, usd: f64, cached: bool) {
        self.entries.push(Entry { kind: "rollout", label: label.into(), usd, cached, attempts: 1 });
    }

    pub fn session(&mut self, label: &str, usd: f64, cached: bool, attempts: u32) {
        self.entries.push(Entry { kind: "session", label: label.into(), usd, cached, attempts });
    }

    pub fn r(&self) -> usize {
        self.entries.iter().filter(|e| e.kind == "rollout").count()
    }
    pub fn p(&self) -> usize {
        self.entries.iter().filter(|e| e.kind == "session").count()
    }
    pub fn usd(&self) -> f64 {
        round6(self.entries.iter().map(|e| e.usd).sum())
    }
    fn usd_of(&self, kind: &str) -> f64 {
        round6(self.entries.iter().filter(|e| e.kind == kind).map(|e| e.usd).sum())
    }
    pub fn labels(&self, kind: &str) -> Vec<String> {
        self.entries.iter().filter(|e| e.kind == kind).map(|e| e.label.clone()).collect()
    }

    /// Fail unless the recorded entries show exactly R=4k and P=3k with no duplicate label.
    /// Duplicates are checked before the counts, because a duplicate can satisfy them.
    pub fn assert_identity(&self) -> Result<()> {
        let (want_r, want_p) = (expected_rollouts(self.k), expected_sessions(self.k));
        let (rl, sl) = (self.labels("rollout"), self.labels("session"));
        let mut dupes = BTreeSet::new();
        for l in &rl {
            if rl.iter().filter(|x| *x == l).count() > 1 {
                dupes.insert(l.clone());
            }
        }
        for l in &sl {
            if sl.iter().filter(|x| *x == l).count() > 1 {
                dupes.insert(l.clone());
            }
        }
        if dupes.is_empty() && self.r() == want_r && self.p() == want_p {
            return Ok(());
        }
        let none = |v: &[String]| if v.is_empty() { "(none)".to_string() } else { v.join(", ") };
        let mut detail = vec![
            format!("{}-{} {}: R={} (want {want_r}), P={} (want {want_p})", self.mode, self.k, self.task_id, self.r(), self.p()),
            format!("  rollouts: {}", none(&rl)),
            format!("  sessions: {}", none(&sl)),
        ];
        if !dupes.is_empty() {
            let d: Vec<String> = dupes.into_iter().collect();
            detail.push(format!(
                "  DUPLICATE labels: {} -- a repeated label means two stages wrote the same run id and one silently overwrote the other, so a candidate the ledger counts was never actually produced",
                super::tree::py_list(&d)
            ));
        }
        bail!("BudgetError: {}", detail.join("\n"))
    }

    pub fn to_json(&self) -> Value {
        json!({
            "mode": self.mode,
            "task_id": self.task_id,
            "k": self.k,
            "R": self.r(),
            "expected_R": expected_rollouts(self.k),
            "P": self.p(),
            "session_attempts": self.entries.iter().filter(|e| e.kind == "session").map(|e| e.attempts).sum::<u32>(),
            "expected_P": expected_sessions(self.k),
            "usd": self.usd(),
            "rollout_usd": self.usd_of("rollout"),
            "session_usd": self.usd_of("session"),
            "rollouts": self.labels("rollout"),
            "sessions": self.labels("session"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity() {
        let mut l = Ledger::new("agg", "t", 1);
        for r in ["base1", "inv1", "dom1", "agg1"] {
            l.rollout(r, 0.0, false);
        }
        for s in ["inv1", "dom1", "agg1"] {
            l.session(s, 0.0, false, 1);
        }
        l.assert_identity().unwrap();
        l.rollout("agg1", 0.0, false);
        assert!(l.assert_identity().unwrap_err().to_string().contains("DUPLICATE"));
    }
}
