//! A deterministic local adapter for exercising the real adaptation workflow.
//!
//! Same as the reference demo: two ledger tasks, a base harness whose summing script drops
//! credits, scripted editors that fix it, and real Python executions of the edited script.

use crate::core::proposer::{CallOpts, Propose, ProposerError, Reply};
use crate::core::spec::{channel_map, BenchPrompts, BenchSpec, BenchmarkAdapter, ChannelSpec};
use crate::core::tree::{self, SurfaceSpec, Tree};
use crate::substrate::opencode::opencode_surface;
use crate::{config, util};
use anyhow::{bail, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

pub const BASE_SCRIPT: &str = "import csv\nimport sys\nfrom decimal import Decimal\n\nwith open(sys.argv[1]) as source:\n    amounts = [Decimal(row['amount']) for row in csv.DictReader(source)]\ntotal = sum((amount for amount in amounts if amount > 0), Decimal(0))\nprint(f'Total: {total:.2f}')\n";

pub struct DemoAdapter {
    pub root: PathBuf,
    spec: BenchSpec,
    pub execution_count: AtomicUsize,
}

impl DemoAdapter {
    pub fn new(root: &Path) -> Result<DemoAdapter> {
        let channel = ChannelSpec::new("opencode", "local/deterministic")?;
        let mut spec = BenchSpec::new(
            "demo",
            root.join("runs"),
            "workspace_config",
            BenchPrompts::work("reconcile signed amounts in a transaction ledger").checked()?,
            channel_map(channel.clone(), channel),
        );
        spec.leak_globs = vec!["reference_answer.txt".into()];
        let a = DemoAdapter { root: root.to_path_buf(), spec: spec.checked()?, execution_count: AtomicUsize::new(0) };
        let surface = opencode_surface();
        let mut base: Tree = surface.always_on.iter().map(|n| (n.to_string(), String::new())).collect();
        base.insert("systemprompt.md".into(), "Reconcile the ledger using python scripts/summarize.py.\n".into());
        base.insert("scripts/summarize.py".into(), BASE_SCRIPT.into());
        tree::write_tree(&surface, &base, &a.base_harness()?, false)?;
        for (task_id, amounts) in [("ledger-1", "100.25\n-20.10\n5.00\n"), ("ledger-2", "35.50\n-10.25\n-5.00\n")] {
            let src = a.task_files(task_id).unwrap();
            util::write(&src.join("ledger.csv"), format!("amount\n{amounts}"))?;
        }
        Ok(a)
    }
}

pub fn python() -> Result<PathBuf> {
    util::which("python3").or_else(|| util::which("python")).ok_or_else(|| anyhow::anyhow!("the demo executes its harness scripts with Python; install python3"))
}

impl BenchmarkAdapter for DemoAdapter {
    fn spec(&self) -> &BenchSpec {
        &self.spec
    }
    fn substrate_name(&self) -> &str {
        "oc"
    }
    fn surface(&self) -> SurfaceSpec {
        opencode_surface()
    }
    fn tasks(&self, _split: &str) -> Result<Vec<String>> {
        Ok(vec!["ledger-1".into(), "ledger-2".into()])
    }
    fn domain_of(&self, _task_id: &str) -> String {
        "accounting".into()
    }
    fn task_prompt(&self, _task_id: &str) -> Result<String> {
        Ok("Sum every signed amount in ledger.csv. Return Total: with two decimal places.".into())
    }
    fn task_files(&self, task_id: &str) -> Option<PathBuf> {
        Some(self.root.join("sources").join(task_id))
    }
    fn base_harness(&self) -> Result<PathBuf> {
        Ok(self.root.join("base"))
    }
    fn validate_tree(&self, t: &Tree) -> Result<()> {
        if !t.contains_key("scripts/summarize.py") {
            bail!("The local executor requires scripts/summarize.py");
        }
        Ok(())
    }
    fn execute(&self, task_id: &str, harness_dir: Option<&Path>, run_id: &str, timeout: Option<u64>) -> Result<PathBuf> {
        self.execution_count.fetch_add(1, Ordering::SeqCst);
        let run_dir = self.root.join("executions").join(run_id);
        std::fs::create_dir_all(&run_dir)?;
        let base = self.base_harness()?;
        let script = harness_dir.unwrap_or(&base).join("scripts/summarize.py");
        let mut cmd = std::process::Command::new(python()?);
        cmd.arg(&script).arg(self.task_files(task_id).unwrap().join("ledger.csv"));
        config::apply_env(&mut cmd, &config::scrub_env(None, &[]));
        let out = util::run_with_timeout(cmd, None, Duration::from_secs(timeout.unwrap_or(10)))?;
        if out.code != Some(0) {
            bail!("CalledProcessError: script exited {:?}: {}", out.code, out.stderr.trim());
        }
        util::write(&run_dir.join("answer.txt"), &out.stdout)?;
        util::write(&run_dir.join("trajectory.md"), format!("[ACTION] Run scripts/summarize.py on ledger.csv\n[OBSERVATION]\n{}", out.stdout))?;
        Ok(run_dir)
    }
    fn delivered(&self, run_dir: &Path) -> bool {
        run_dir.join("answer.txt").is_file()
    }
    fn trajectory(&self, run_dir: &Path, _obs_trunc: usize) -> Result<String> {
        util::read_lossy(&run_dir.join("trajectory.md"))
    }
    fn deliverable_text(&self, run_dir: &Path) -> Result<String> {
        util::read_lossy(&run_dir.join("answer.txt"))
    }
    fn deliverable_files(&self, run_dir: &Path) -> Option<BTreeMap<String, String>> {
        let mut m = BTreeMap::new();
        m.insert("answer.txt".into(), self.deliverable_text(run_dir).ok()?);
        Some(m)
    }
}

/// Scripted edits and selection; no model service or benchmark judge is used.
pub struct DemoProposer {
    spec: ChannelSpec,
    calls: Mutex<Vec<Reply>>,
}

impl DemoProposer {
    pub fn new(spec: ChannelSpec) -> DemoProposer {
        DemoProposer { spec, calls: Mutex::new(vec![]) }
    }
}

/// Sum of the `amount` column, formatted with two decimals (exact decimal arithmetic in
/// hundredths, which is all the demo ledgers use).
fn signed_total(csv: &str) -> String {
    let mut cents: i64 = 0;
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let v = line.trim();
        let neg = v.starts_with('-');
        let v = v.trim_start_matches('-');
        let (w, f) = v.split_once('.').unwrap_or((v, "0"));
        let f = format!("{f:0<2}");
        let c = w.parse::<i64>().unwrap_or(0) * 100 + f[..2].parse::<i64>().unwrap_or(0);
        cents += if neg { -c } else { c };
    }
    let sign = if cents < 0 { "-" } else { "" };
    format!("{sign}{}.{:02}", cents.abs() / 100, cents.abs() % 100)
}

impl Propose for DemoProposer {
    fn spec(&self) -> &ChannelSpec {
        &self.spec
    }
    fn audit_dir(&self) -> Option<PathBuf> {
        None
    }
    fn calls(&self) -> Vec<Reply> {
        self.calls.lock().unwrap().clone()
    }
    fn call(&self, _mission: &str, context_dir: Option<&Path>, label: &str, _o: CallOpts) -> Result<Reply, ProposerError> {
        let err = |e: anyhow::Error| ProposerError { message: format!("[{label}] demo: {e}"), attempts: 1, cost_usd: 0.0 };
        let ctx = context_dir.ok_or_else(|| err(anyhow::anyhow!("no context")))?;
        let mut workspace = None;
        let text = if label.starts_with("select.") {
            let expected = signed_total(&util::read_lossy(&ctx.join("source/ledger.csv")).map_err(err)?);
            let mut cands = util::files_sorted(&ctx.join("candidates"));
            cands.retain(|p| p.extension().map(|e| e == "md").unwrap_or(false));
            let chosen = cands
                .iter()
                .find(|p| util::read_lossy(p).map(|t| t.contains(&format!("Total: {expected}"))).unwrap_or(false))
                .map(|p| p.file_stem().unwrap().to_string_lossy().to_string())
                .ok_or_else(|| err(anyhow::anyhow!("StopIteration: no candidate matches the source total")))?;
            format!("<why>The signed total matches the source ledger.</why>\n```choice\n{chosen}\n```")
        } else if label.starts_with("experience.") {
            let a = util::read_lossy(&ctx.join("harness/scripts/summarize.py")).map_err(err)?;
            let b = util::read_lossy(&ctx.join("base_harness/scripts/summarize.py")).map_err(err)?;
            let changed = a != b;
            format!(
                "<edits>\ncomponent: scripts/summarize.py\ncandidates: inv1\nchange: dropped the positive-only filter from the amount sum\nobserved: the probe total included the credits the base run omitted\nverdict: {}\n</edits>\n\n{}",
                if changed { "helped" } else { "no observed effect" },
                if changed {
                    "<experience>\ncue: When a ledger includes credits and debits\nlesson: Include both signs when summing in scripts/summarize.py.\nevidence: The scripts/summarize.py row -- removing the positive-only filter changed the probe total to include the credits omitted by the base trajectory.\nconfidence: high for this observed arithmetic contrast\n</experience>"
                } else {
                    "NO_ENTRY"
                }
            )
        } else {
            // The real proposer works on a copy; edit the harness in place like a session would.
            let ws = ctx.join("harness");
            let script = ws.join("scripts/summarize.py");
            let body = util::read_lossy(&script).map_err(err)?;
            util::write(&script, body.replace(" if amount > 0", "")).map_err(err)?;
            workspace = Some(ws);
            "<notes>Keep signed amounts: the base total omitted credits in the source ledger.</notes>".into()
        };
        let r = Reply { label: label.into(), text, ok: true, attempts: 1, workspace, ..Default::default() };
        self.calls.lock().unwrap().push(r.clone());
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn totals() {
        assert_eq!(super::signed_total("amount\n100.25\n-20.10\n5.00\n"), "85.15");
        assert_eq!(super::signed_total("amount\n35.50\n-10.25\n-5.00\n"), "20.25");
    }
}
