//! Measurement of the run that selection chose, after it chose it.
//!
//! Nothing in the method imports this module. A judge failure is None, never a zero, and a
//! None is excluded from the denominator.

use super::spec::BenchmarkAdapter;
use crate::util;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// A score needs a numerator and a positive denominator; zero passes is valid.
pub fn has_grade(n_passed: Option<f64>, n_criteria: Option<f64>) -> bool {
    matches!((n_passed, n_criteria), (Some(_), Some(c)) if c > 0.0)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Scored {
    pub task_id: String,
    pub mode: String,
    pub k: usize,
    pub label: String,
    pub run_id: String,
    pub chosen_is_base: bool,
    #[serde(default)]
    pub selected_at: String,
    #[serde(default)]
    pub n_passed: Option<f64>,
    #[serde(default)]
    pub n_criteria: Option<f64>,
    #[serde(default)]
    pub all_pass: bool,
    #[serde(default)]
    pub raw: Value,
    #[serde(default)]
    pub error: String,
}

impl Scored {
    /// Whether the judge succeeded, not whether the run did well.
    pub fn graded(&self) -> bool {
        has_grade(self.n_passed, self.n_criteria)
    }
    pub fn frac(&self) -> Option<f64> {
        self.graded().then(|| self.n_passed.unwrap() / self.n_criteria.unwrap())
    }
    pub fn to_json(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap();
        v["frac"] = json!(self.frac());
        v["graded"] = json!(self.graded());
        v
    }
}

/// The mode token of a selection record, reading the legacy key too.
pub fn mode_of(sel: &Value) -> String {
    sel.get("mode").or_else(|| sel.get("arm")).and_then(|v| v.as_str()).unwrap_or("").to_string()
}

#[allow(clippy::too_many_arguments)]
pub fn judge_one(adapter: &dyn BenchmarkAdapter, task_id: &str, mode: &str, k: usize, label: &str, run_id: &str, run_dir: &Path, chosen_is_base: bool, selected_at: &str) -> Scored {
    let mut s = Scored {
        task_id: task_id.into(),
        mode: mode.into(),
        k,
        label: label.into(),
        run_id: run_id.into(),
        chosen_is_base,
        selected_at: selected_at.into(),
        ..Default::default()
    };
    match adapter.score(run_dir, task_id) {
        Err(e) => s.error = format!("{e}"),
        Ok(None) => s.error = "judge returned nothing".into(),
        Ok(Some(got)) if got.as_object().map(|o| o.is_empty()).unwrap_or(true) => s.error = "judge returned nothing".into(),
        Ok(Some(got)) => {
            s.n_passed = got.get("n_passed").and_then(|v| v.as_f64());
            s.n_criteria = got.get("n_criteria").and_then(|v| v.as_f64());
            s.all_pass = got
                .get("all_pass")
                .and_then(|v| v.as_bool())
                .unwrap_or(s.graded() && s.n_passed == s.n_criteria);
            s.raw = got;
        }
    }
    s
}

/// Judge each selection's chosen run, given Selection records.
pub fn judge_selections(adapter: &dyn BenchmarkAdapter, selections: &[Value], parallel: usize) -> Vec<Scored> {
    util::par_map(selections.to_vec(), parallel, |sel| {
        let g = |k: &str| sel.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let k = sel.get("k").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let base = sel.get("chosen_is_base").and_then(|v| v.as_bool()).unwrap_or(false);
        let rd = g("chosen_run_dir");
        if !sel.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) || g("chosen_label").is_empty() || rd.is_empty() {
            return Scored {
                task_id: g("task_id"),
                mode: mode_of(&sel),
                k,
                label: g("chosen_label"),
                run_id: g("chosen_run_id"),
                chosen_is_base: base,
                selected_at: g("selected_at"),
                error: "selection did not complete successfully".into(),
                ..Default::default()
            };
        }
        judge_one(adapter, &g("task_id"), &mode_of(&sel), k, &g("chosen_label"), &g("chosen_run_id"), &PathBuf::from(rd), base, &g("selected_at"))
    })
}

pub fn write(rows: &[Scored], path: &Path) -> Result<PathBuf> {
    let v: Vec<Value> = rows.iter().map(|r| r.to_json()).collect();
    util::write(path, serde_json::to_string_pretty(&v)?)?;
    Ok(path.to_path_buf())
}
