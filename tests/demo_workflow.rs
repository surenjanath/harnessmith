//! The whole method on the deterministic demo: both modes, k=2, two tasks, library threading,
//! the budget identity, selection of the corrected harness, and resume.

use harness2::benchmarks::demo::{DemoAdapter, DemoProposer};
use harness2::core::proposer::Propose;
use harness2::core::schedule::{self, RunOpts};
use harness2::core::spec::BenchmarkAdapter;
use serde_json::Value;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[test]
fn demo_runs_both_modes_and_resumes() {
    if harness2::util::which("python3").is_none() {
        eprintln!("skipping: the demo executes its harness script with python3");
        return;
    }
    harness2::log::set_quiet(true);
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    let adapter = DemoAdapter::new(&root).unwrap();
    let prop: Arc<dyn Propose> = Arc::new(DemoProposer::new(adapter.spec().channels["proposer"].clone()));
    let o = RunOpts::default();
    let tasks = adapter.tasks("all").unwrap();
    let runs = adapter.spec().runs_root.clone();
    let lib = root.join("library/k2");
    let r = schedule::run_domain("accounting", &tasks, 2, &adapter, &prop, &runs, &lib, &o);
    assert_eq!(r["done"], 2, "{r}");
    assert_eq!(adapter.execution_count.load(Ordering::SeqCst), 28);
    assert_eq!(prop.calls().len(), 32);

    for t in ["ledger-1", "ledger-2"] {
        let s: Value = serde_json::from_str(&std::fs::read_to_string(runs.join("demo-oc-agg2").join(t).join("summary.json")).unwrap()).unwrap();
        assert_eq!(s["status"], "ok");
        for m in ["agg", "seq"] {
            let ledger = &s["modes"][m]["ledger"];
            assert_eq!((ledger["R"].as_u64(), ledger["P"].as_u64()), (Some(8), Some(6)));
            assert_eq!(harness2::core::ledger::check_saved(ledger), None, "{ledger}");
            assert_eq!((ledger["live_R"].as_u64(), ledger["dead"].as_array().map(|a| a.len())), (Some(8), Some(0)));
            let rd = s["selections"][m]["chosen_run_dir"].as_str().unwrap();
            let answer = std::fs::read_to_string(std::path::Path::new(rd).join("answer.txt")).unwrap();
            let want = if t == "ledger-1" { "Total: 85.15" } else { "Total: 20.25" };
            assert_eq!(answer.trim(), want);
        }
    }
    let exp = std::fs::read_to_string(lib.join("agg/accounting/harnesses/ledger-1/EXPERIENCE.md")).unwrap();
    assert!(exp.contains("- **provenance**: measured"), "{exp}");
    assert!(runs.join("demo-oc-agg2/ledger-2/context/inv1/library/INDEX.md").is_file());

    let before = adapter.execution_count.load(Ordering::SeqCst);
    let r2 = schedule::run_domain("accounting", &tasks, 2, &adapter, &prop, &runs, &lib, &o);
    assert_eq!(r2["tasks"][0]["status"], "skipped");
    assert_eq!(adapter.execution_count.load(Ordering::SeqCst), before);
}
