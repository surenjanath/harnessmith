//! Grounding: figures in a deliverable must come from somewhere.
//!
//! The commonest way a small model is wrong about data is the plainest: it writes a number it
//! never computed. This check knows every number the session has actually seen (the goal, the
//! input files, and the output of every tool call) and, before an agent may finish, looks for
//! figures in what it wrote that match none of them. Those are handed back with one
//! instruction: compute them. It is a deterministic check with no model call.

use super::precise::{atoms, Num};
use std::path::Path;

/// Every number the session has seen, kept sorted for range lookups.
#[derive(Debug, Default, Clone)]
pub struct Evidence {
    vals: Vec<f64>,
    sorted: bool,
    /// Whether the goal and the input files have been read in yet.
    pub loaded: bool,
    /// Tool outputs that carried numbers (the session computed or read something).
    pub tool_outputs: usize,
    /// Numbers found in the input files: with none, the goal is not about data.
    pub input_values: usize,
}

const MAX_VALUES: usize = 400_000;

impl Evidence {
    pub fn add_text(&mut self, text: &str) {
        if self.vals.len() >= MAX_VALUES {
            return;
        }
        for n in atoms(text).0 {
            self.vals.push(n.v);
            // A fraction stated as a percentage, and the reverse.
            if n.v.abs() <= 1.0 && n.v != 0.0 {
                self.vals.push(n.v * 100.0);
            }
        }
        self.sorted = false;
    }

    /// A tool's output: evidence, and a sign that the session works with data.
    pub fn add_tool_output(&mut self, text: &str) {
        let before = self.vals.len();
        self.add_text(text);
        if self.vals.len() > before {
            self.tool_outputs += 1;
        }
    }

    /// The goal text and every input file of the workspace (text files up to 2 MB in all).
    pub fn load_inputs(&mut self, workspace: &Path, task: &str) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        self.add_text(task);
        let mut budget = 2_000_000usize;
        for p in crate::util::files_sorted(workspace) {
            let rel = crate::util::rel(workspace, &p);
            if rel.starts_with("output/") || rel.starts_with(".harness/") || rel.starts_with(".hx/") || crate::util::in_pycache(&p) {
                continue;
            }
            let Ok(bytes) = std::fs::read(&p) else { continue };
            if bytes.len() > budget || bytes[..bytes.len().min(4096)].contains(&0) {
                continue;
            }
            budget -= bytes.len();
            let before = self.vals.len();
            self.add_text(&String::from_utf8_lossy(&bytes));
            self.input_values += self.vals.len() - before;
        }
    }

    /// Is `n` (allowing for it having been rounded to the decimals it shows) among the evidence?
    pub fn supports(&mut self, n: &Num) -> bool {
        if !self.sorted {
            self.vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted = true;
        }
        let tol = 0.5 * 10f64.powi(-(n.decimals.min(9) as i32)) + 1e-9 * n.v.abs().max(1.0);
        let lo = self.vals.partition_point(|v| *v < n.v - tol);
        self.vals.get(lo).map(|v| (*v - n.v).abs() <= tol).unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.vals.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }
}

/// Numbers that are part of ordinary prose rather than findings: small counts, 100 (as in
/// "100%"), and years.
fn incidental(n: &Num) -> bool {
    n.decimals == 0 && (n.v.abs() <= 12.0 || n.v == 100.0 || (1900.0..=2100.0).contains(&n.v))
}

/// Prose without the parts whose numbers are not claims: code blocks, links, and the
/// harness's own provenance notes.
fn claims_text(body: &str) -> String {
    let mut out = String::new();
    let mut in_code = false;
    for l in body.lines() {
        if l.trim_start().starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            continue;
        }
        out.push_str(l);
        out.push('\n');
    }
    let url = regex::Regex::new(r"https?://\S+").unwrap();
    url.replace_all(&out, " ").into_owned()
}

/// Figures in the written deliverables (`.md` / `.txt` under `out_dir`) that the evidence does
/// not support: (file, the figure as a number). At most `max` are returned.
pub fn unsupported(out_dir: &Path, ev: &mut Evidence, max: usize) -> Vec<(String, String)> {
    let mut bad: Vec<(String, String)> = vec![];
    for p in crate::util::files_sorted(out_dir) {
        let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        if !matches!(ext.as_str(), "md" | "markdown" | "txt") {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(&p) else { continue };
        let rel = format!("output/{}", crate::util::rel(out_dir, &p));
        for n in atoms(&claims_text(&body)).0 {
            if incidental(&n) || ev.supports(&n) {
                continue;
            }
            let shown = crate::llm::calc::format(n.v);
            if !bad.iter().any(|(_, s)| *s == shown) {
                bad.push((rel.clone(), shown));
            }
            if bad.len() >= max {
                return bad;
            }
        }
    }
    bad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn made_up_figures_are_found_and_computed_ones_pass() {
        let d = tempfile::tempdir().unwrap();
        let ws = d.path();
        std::fs::create_dir_all(ws.join("output")).unwrap();
        std::fs::write(ws.join("sales.csv"), "region,units,unit_price\nNorth,120,4.50\nSouth,200,4.25\n").unwrap();
        let mut ev = Evidence::default();
        ev.load_inputs(ws, "Using sales.csv, write a short report on revenue.");
        // The model wrote a total it never computed.
        std::fs::write(ws.join("output/answer.md"), "# Revenue\n\nTotal revenue is $321.875 across 2 regions. North sold 120 units at $4.50.\n").unwrap();
        let bad = unsupported(&ws.join("output"), &mut ev, 5);
        assert_eq!(bad, vec![("output/answer.md".to_string(), "321.875".to_string())]);
        // After computing, the same figure style passes: rounded, with a thousands separator, as a percentage.
        ev.add_tool_output("1390.0\n0.3885");
        assert_eq!(ev.tool_outputs, 1);
        std::fs::write(ws.join("output/answer.md"), "# Revenue\n\nTotal revenue is $1,390 (North: 38.85% of it). In 2024 the top 3 held.\n\n```python\nx = 99999\n```\nSee https://example.com/q/123456\n").unwrap();
        assert!(unsupported(&ws.join("output"), &mut ev, 5).is_empty());
        // Rounding in the write-up is fine; a different value is not.
        std::fs::write(ws.join("output/answer.md"), "About 1390.0 in total, i.e. 1.39 thousand... or 1391?").unwrap();
        let bad = unsupported(&ws.join("output"), &mut ev, 5);
        assert_eq!(bad.iter().map(|b| b.1.as_str()).collect::<Vec<_>>(), vec!["1.39", "1391"]);
        // Other file types are not prose.
        std::fs::write(ws.join("output/answer.md"), "ok").unwrap();
        std::fs::write(ws.join("output/page.html"), "<style>p{margin:17px}</style>").unwrap();
        assert!(unsupported(&ws.join("output"), &mut ev, 5).is_empty());
    }
}
