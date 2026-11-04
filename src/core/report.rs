//! Judged selections -> the figure each benchmark reports, plus pick provenance and warnings.
//!
//! Ungraded rows and tasks that never completed are excluded from the denominator rather than
//! counted as failures.

use super::judge::Scored;
use super::schedule::mode_name;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default)]
pub struct Summary {
    pub mode: String,
    pub k: usize,
    pub n: usize,
    pub n_ungraded: usize,
    pub score: Option<f64>,
    pub all_pass: Option<f64>,
    pub kept_base: usize,
    pub chose_final: usize,
    pub chose_candidate: usize,
    pub fell_back: usize,
    pub usd: f64,
    pub metrics: Vec<String>,
}

impl Summary {
    pub fn name(&self) -> String {
        format!("{} k={}", mode_name(&self.mode), self.k)
    }

    /// LAB reports partial credit and all-criteria pass; the others report one score.
    pub fn line(&self, bench: &str) -> String {
        if self.n == 0 {
            return format!("  {}: no graded task", self.name());
        }
        let figures = if bench == "lab" {
            format!("partial {:6.2}% | pass {:5.1}%", self.score.unwrap_or(0.0), self.all_pass.unwrap_or(0.0))
        } else {
            format!("score {:6.2}%", self.score.unwrap_or(0.0))
        };
        let pick = format!(
            " | picked output: final {} / candidate {} / base {}",
            self.chose_final, self.chose_candidate, self.kept_base
        );
        let fb = if self.fell_back > 0 { format!(" | fell back {}", self.fell_back) } else { String::new() };
        let ug = if self.n_ungraded > 0 { format!(" | ungraded {}", self.n_ungraded) } else { String::new() };
        format!("  {}: n={:>3} | {figures}{pick}{fb}{ug}", self.name(), self.n)
    }
}

/// Classify the selected output; each sequential checkpoint is a composed final.
fn kind(r: &Scored) -> &'static str {
    let lbl = r.label.as_str();
    if lbl.starts_with("base") {
        return "base";
    }
    match r.mode.as_str() {
        "agg" => {
            if lbl.starts_with("agg") { "final" } else { "candidate" }
        }
        "seq" => {
            if (1..=r.k).any(|i| lbl == format!("c{i}")) { "final" } else { "candidate" }
        }
        _ => "candidate",
    }
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// One Summary per (mode, k). Ungraded rows are counted, then excluded.
pub fn summarise(rows: &[Scored], fell_back: &BTreeMap<(String, usize), usize>, usd: &BTreeMap<(String, usize), f64>) -> Vec<Summary> {
    let keys: BTreeSet<(String, usize)> = rows.iter().map(|r| (r.mode.clone(), r.k)).collect();
    let mut out = vec![];
    for (mode, k) in keys {
        let group: Vec<&Scored> = rows.iter().filter(|r| r.mode == mode && r.k == k).collect();
        let good: Vec<&&Scored> = group.iter().filter(|r| r.graded()).collect();
        let units: BTreeSet<String> = good
            .iter()
            .filter_map(|r| r.raw.get("metric").and_then(|m| m.as_str()).map(|s| s.to_string()))
            .filter(|s| !s.is_empty())
            .collect();
        let mut s = Summary {
            mode: mode.clone(),
            k,
            n: good.len(),
            n_ungraded: group.len() - good.len(),
            kept_base: good.iter().filter(|r| r.chosen_is_base).count(),
            chose_final: good.iter().filter(|r| kind(r) == "final").count(),
            chose_candidate: good.iter().filter(|r| kind(r) == "candidate").count(),
            fell_back: *fell_back.get(&(mode.clone(), k)).unwrap_or(&0),
            usd: round2(*usd.get(&(mode.clone(), k)).unwrap_or(&0.0)),
            metrics: units.into_iter().collect(),
            ..Default::default()
        };
        if !good.is_empty() {
            let mean = good.iter().map(|r| r.frac().unwrap()).sum::<f64>() / good.len() as f64;
            s.score = Some(round2(100.0 * mean));
            s.all_pass = Some(round2(100.0 * good.iter().filter(|r| r.all_pass).count() as f64 / good.len() as f64));
        }
        out.push(s);
    }
    out
}

/// The printable summary block.
pub fn render(summaries: &[Summary], title: &str, bench: &str) -> String {
    let mut lines = vec![];
    if !title.is_empty() {
        lines.push(title.to_string());
        lines.push("=".repeat(40.max(title.chars().count())));
    }
    let mut sorted = summaries.to_vec();
    sorted.sort_by(|a, b| (a.k, &a.mode).cmp(&(b.k, &b.mode)));
    for s in &sorted {
        lines.push(s.line(bench));
    }
    let at1: BTreeMap<&str, &Summary> = summaries.iter().filter(|s| s.k == 1 && s.score.is_some()).map(|s| (s.mode.as_str(), s)).collect();
    if let (Some(a), Some(q)) = (at1.get("agg"), at1.get("seq")) {
        let gap = (a.score.unwrap() - q.score.unwrap()).abs();
        lines.push(String::new());
        lines.push("  NOTE  at k=1 the parallel and sequential modes are the same procedure (one INVARIANT draw, one DOMAIN draw, one composition).".into());
        lines.push(format!("        The {gap:.2} pt between them is therefore a NOISE FLOOR, not a result -- an effect at k>=2 has to clear it."));
    }
    for s in summaries {
        if s.n > 0 && (s.chose_final as f64) <= 0.25 * s.n as f64 {
            lines.push(String::new());
            lines.push(format!(
                "  WARNING  {}: the mode's own final was chosen on only {}/{} tasks. These records include outputs outside the published pool of {} composed finals.",
                s.name(), s.chose_final, s.n, s.k
            ));
        }
    }
    for s in summaries {
        if s.n > 0 && s.kept_base as f64 >= 0.8 * s.n as f64 {
            lines.push(String::new());
            lines.push(format!(
                "  WARNING  {}: selection chose a reference output on {}/{} tasks, outside the published pool of composed outputs.",
                s.name(), s.kept_base, s.n
            ));
        }
    }
    for s in summaries.iter().filter(|s| s.fell_back > 0) {
        lines.push(String::new());
        lines.push(format!(
            "  WARNING  {}: {} stage(s) fell back to the inherited harness. Inspect the stage errors when interpreting these results.",
            s.name(), s.fell_back
        ));
    }
    for s in summaries.iter().filter(|s| s.metrics.len() > 1) {
        lines.push(String::new());
        lines.push(format!(
            "  WARNING  {}: graded rows carry MIXED scoring units ({}). The score averages weighted points with unweighted criteria counts (about 20 pt apart on JobBench); re-judge the fallback rows before comparing this line with any published number.",
            s.name(), s.metrics.join(", ")
        ));
    }
    lines.join("\n")
}
