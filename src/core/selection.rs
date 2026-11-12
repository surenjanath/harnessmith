//! Choose an existing anonymized final output, then reflect on its refinement evidence.

use super::context;
use super::library;
use super::prompts as P;
use super::proposer::{CallOpts, Propose, ProposerError};
use super::pyrandom::PyRandom;
use super::spec::{resolve_prompts, BenchPrompts, BenchmarkAdapter, RunView};
use crate::util;
use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha1::{Digest, Sha1};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

static CHOICE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```choice\s*\n?\s*([A-Za-z0-9_-]+)\s*\n?\s*```").unwrap());
static WHY: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<why>(.*?)</why>").unwrap());
static EXP: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<experience>(.*?)</experience>").unwrap());
static EDITS: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<edits>(.*?)</edits>").unwrap());

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Selection {
    pub task_id: String,
    pub mode: String,
    pub k: usize,
    pub chosen_id: String,
    pub chosen_label: String,
    pub chosen_run_id: String,
    pub chosen_is_base: bool,
    pub why: String,
    pub id_map: BTreeMap<String, String>,
    pub dropped: Vec<String>,
    pub n_candidates: usize,
    pub ok: bool,
    pub fallback: bool,
    pub error: String,
    pub usd: f64,
    pub experience: Map<String, Value>,
    pub trivial: bool,
    pub evidence: String,
}

impl Selection {
    pub fn new(task_id: &str, mode: &str, k: usize) -> Selection {
        Selection { task_id: task_id.into(), mode: mode.into(), k, evidence: "deliverable".into(), ..Default::default() }
    }
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap()
    }
    pub fn experience_value(&self) -> Value {
        Value::Object(self.experience.clone())
    }
}

/// anon id -> label, by canonical sort then a seeded shuffle: stable across reruns.
pub fn assign_ids(labels: &[String], task_id: &str, k: usize, mode: &str) -> BTreeMap<String, String> {
    let digest = hex::encode(Sha1::digest(format!("{task_id}|{k}|{mode}").as_bytes()));
    let seed = i128::from_str_radix(&digest[..12], 16).unwrap();
    let mut ordered = labels.to_vec();
    ordered.sort();
    PyRandom::from_int(seed).shuffle(&mut ordered);
    ordered.into_iter().enumerate().map(|(i, l)| (anon_id(i), l)).collect()
}

/// A..Z, AA..AZ, BA... -- always accepted by the choice parser.
pub fn anon_id(index: usize) -> String {
    let mut out = String::new();
    let mut n = index + 1;
    while n > 0 {
        let rem = (n - 1) % 26;
        n = (n - 1) / 26;
        out.insert(0, (b'A' + rem as u8) as char);
    }
    out
}

fn is_base(label: &str) -> bool {
    label.starts_with("base")
}

/// The one home for selection-pool construction: exactly the k finals of `mode`.
pub fn selection_pool(mode: &str, k: usize, rollouts: &BTreeMap<String, RunView>, _base: &[RunView]) -> BTreeMap<String, RunView> {
    let labels: Vec<String> = if mode == "agg" {
        (1..=k).map(|i| format!("agg{i}")).collect()
    } else {
        (1..=k).map(|t| format!("c{t}")).collect()
    };
    labels.into_iter().filter_map(|l| rollouts.get(&l).map(|r| (l, r.clone()))).collect()
}

/// Drop non-answers before anonymising, recording each drop and its cause in `dropped`.
fn render_pool(candidates: &BTreeMap<String, RunView>, adapter: &dyn BenchmarkAdapter, sel: &mut Selection) -> (BTreeMap<String, RunView>, BTreeMap<String, String>) {
    let mut pool = BTreeMap::new();
    let mut artifacts = BTreeMap::new();
    for (lbl, rv) in candidates {
        if !(rv.ok && rv.delivered) {
            sel.dropped.push(lbl.clone());
            continue;
        }
        match adapter.deliverable_text(&rv.run_dir) {
            Err(e) => sel.dropped.push(util::take_chars(&format!("{lbl} (render failed: {e})"), 200)),
            Ok(r) if r.trim().is_empty() => sel.dropped.push(lbl.clone()),
            Ok(r) => {
                pool.insert(lbl.clone(), rv.clone());
                artifacts.insert(lbl.clone(), r);
            }
        }
    }
    if pool.is_empty() && !candidates.is_empty() && adapter.spec().state_deliverable {
        for (lbl, rv) in candidates {
            if !rv.ok {
                continue;
            }
            match adapter.trajectory(&rv.run_dir, adapter.spec().obs_trunc) {
                Err(e) => sel.dropped.push(util::take_chars(&format!("{lbl} (trajectory failed: {e})"), 200)),
                Ok(body) => {
                    if crate::core::context::is_absent_evidence(&body) {
                        continue;
                    }
                    pool.insert(lbl.clone(), rv.clone());
                    artifacts.insert(
                        lbl.clone(),
                        format!("(this task produced no file deliverable; the candidate is shown as its own session transcript)\n\n{body}"),
                    );
                }
            }
        }
        if !pool.is_empty() {
            sel.evidence = "trajectory".into();
            sel.dropped.retain(|d| !pool.contains_key(d));
        }
    }
    (pool, artifacts)
}

fn assets_note(adapter: &dyn BenchmarkAdapter, anon: &str, names: &[String]) -> String {
    match adapter.deliverable_assets_note(names) {
        Some(n) => n.replace("{anon}", anon),
        None => format!(
            "     Rendered evidence for this candidate is in `{anon}/`: {}. OPEN THESE -- they show what the artifact actually does when run, which its source alone does not tell you.\n",
            names.join(", ")
        ),
    }
}

fn assets_line(adapter: &dyn BenchmarkAdapter, shown: &str) -> String {
    adapter.deliverable_assets_line(shown).unwrap_or_else(|| {
        format!(
            "    context/candidates/<id>/     RENDERED EVIDENCE for that candidate ({shown}) -- what the\n                                artifact actually DOES when run. Source alone does not show this.\n"
        )
    })
}

/// Write one anonymised, untrimmed file per candidate; returns the asset names copied in.
fn write_candidates(sel: &Selection, artifacts: &BTreeMap<String, String>, pool: &BTreeMap<String, RunView>, cand_dir: &Path, adapter: &dyn BenchmarkAdapter) -> Result<BTreeSet<String>> {
    let mut asset_names = BTreeSet::new();
    for (anon, lbl) in &sel.id_map {
        let text = &artifacts[lbl];
        let assets: Vec<PathBuf> = adapter.deliverable_assets(&pool[lbl].run_dir).unwrap_or_default().into_iter().filter(|a| a.is_file()).collect();
        let mut note = String::new();
        if !assets.is_empty() {
            let adir = cand_dir.join(anon);
            std::fs::create_dir_all(&adir)?;
            let mut names = vec![];
            for a in &assets {
                let n = a.file_name().unwrap().to_string_lossy().to_string();
                std::fs::copy(a, adir.join(&n))?;
                names.push(n);
            }
            note = assets_note(adapter, anon, &names);
            asset_names.extend(names);
        }
        let header = format!(
            "<!-- candidate {anon}: {} chars, {} lines. COMPLETE -- nothing trimmed. Navigate with grep/offsets; evaluating a candidate means opening the part that determines whether it works.\n{note}-->\n",
            context::thousands(util::char_len(text)),
            context::thousands(text.matches('\n').count() + 1)
        );
        util::write(&cand_dir.join(format!("{anon}.md")), format!("{header}{text}"))?;
    }
    Ok(asset_names)
}

fn evidence_block(asset_names: &BTreeSet<String>, adapter: &dyn BenchmarkAdapter, reference_only: bool) -> (String, String) {
    if asset_names.is_empty() {
        return (String::new(), String::new());
    }
    let shown = asset_names.iter().take(4).cloned().collect::<Vec<_>>().join(", ");
    let a_evid = format!(
        "context/candidates/<id>/        {}: {shown}\n",
        if reference_only { "raw originals per candidate" } else { "rendered evidence per candidate" }
    );
    (assets_line(adapter, &shown), a_evid)
}

/// The neutral stand-in after a selector failure: a delivered base, else the first final.
fn fallback(sel: &mut Selection, pool: &BTreeMap<String, RunView>, pool_kind: &str, exc: &str) -> bool {
    let base: Vec<&String> = pool.keys().filter(|l| is_base(l)).collect();
    if let Some(b) = base.first() {
        sel.chosen_label = (*b).clone();
    } else if pool_kind == "finals" {
        sel.chosen_label = pool.keys().next().unwrap().clone();
    } else {
        sel.error = format!("{}; no delivered base artifact was available for a neutral fallback", util::take_chars(exc, 350));
        return false;
    }
    sel.chosen_id = sel.id_map.iter().find(|(_, l)| **l == sel.chosen_label).map(|(a, _)| a.clone()).unwrap_or_default();
    sel.fallback = true;
    sel.error = util::take_chars(exc, 400);
    true
}

/// Write selection.json, carrying a disk failure on the record rather than raising.
fn persist(sel: &mut Selection, root: &Path, run_dir: &str) {
    let mut rec = sel.to_json();
    rec["chosen_run_dir"] = json!(run_dir);
    if let Err(e) = util::write(&root.join("selection.json"), serde_json::to_string_pretty(&rec).unwrap()) {
        sel.ok = false;
        sel.error = util::take_chars(&format!("{} | could not persist selection: {e}", sel.error), 400);
    }
}

/// Pick one output. A selector failure falls back to a delivered base and never raises;
/// an adapter bug does raise, because it is a bug and not a selection outcome.
#[allow(clippy::too_many_arguments)]
pub fn run_selection(task_id: &str, k: usize, mode: &str, candidates: &BTreeMap<String, RunView>, adapter: &dyn BenchmarkAdapter, prop: &dyn Propose, work_dir: &Path, pool_kind: &str) -> Result<Selection> {
    let mut sel = Selection::new(task_id, mode, k);
    let bp = resolve_prompts(adapter, task_id);
    let (pool, artifacts) = render_pool(candidates, adapter, &mut sel);
    sel.n_candidates = pool.len();
    if pool.is_empty() {
        sel.error = "no candidate produced an artifact".into();
        crate::log::at("select", "select");
        crate::log::event("select", "failed", json!({"error": sel.error, "dropped": sel.dropped}));
        return Ok(sel);
    }
    sel.id_map = assign_ids(&pool.keys().cloned().collect::<Vec<_>>(), task_id, k, mode);
    let root = work_dir.join("selection").join(format!("{mode}{k}"));
    util::rmtree(&root);
    std::fs::create_dir_all(&root)?;
    util::write(&root.join("task.md"), adapter.task_prompt(task_id)?)?;
    context::copy_source(adapter, task_id, &root)?;
    let cand_dir = root.join("candidates");
    std::fs::create_dir_all(&cand_dir)?;
    let asset_names = write_candidates(&sel, &artifacts, &pool, &cand_dir, adapter)?;
    let ids = sel.id_map.keys().cloned().collect::<Vec<_>>().join(", ");
    let reference_only = adapter.deliverable_assets_reference_only();
    let (a_line, a_evid) = evidence_block(&asset_names, adapter, reference_only);
    let mission = [
        P::select(&bp, pool.len(), pool_kind, &a_line)?,
        format!(
            "# YOUR EVIDENCE\n\nIt is on disk. Read it -- do not answer from this message alone.\n\ncontext/task.md                 the instructions\ncontext/source/                 task source materials, when supplied\ncontext/candidates/<id>.md      one file per candidate: {ids}\n{a_evid}\nOpen every candidate before choosing{}. Choose exactly one of: {ids}.",
            if !asset_names.is_empty() && !reference_only { ", INCLUDING the rendered evidence in its directory" } else { "" }
        ),
    ]
    .join("\n\n");
    context::assert_no_grader_verdicts(&root, context::GRADER_VERDICT_MARKERS)?;
    crate::log::at("select", "select");
    crate::log::event("select", "start", json!({"n": pool.len(), "ids": sel.id_map}));
    crate::say!("  [select.{mode}{k}] choosing among {} candidates: {ids}", pool.len());
    let outcome: Result<(), (String, f64)> = match prop.call(&mission, Some(&root), &format!("select.{mode}{k}"), CallOpts::default()) {
        Err(ProposerError { message, cost_usd, .. }) => Err((message, cost_usd)),
        Ok(reply) => {
            sel.usd += reply.cost_usd;
            match CHOICE.captures(&reply.text) {
                None => Err(("no ```choice block in the reply".into(), 0.0)),
                Some(m) => {
                    let anon = m[1].trim().to_string();
                    if let Some(lbl) = sel.id_map.get(&anon).cloned() {
                        sel.why = util::take_chars(WHY.captures(&reply.text).map(|w| w[1].trim().to_string()).unwrap_or_default().as_str(), 2000);
                        sel.chosen_id = anon;
                        sel.chosen_label = lbl;
                        sel.ok = true;
                        Ok(())
                    } else {
                        Err((format!("chose {anon:?}, which is not one of {ids}"), 0.0))
                    }
                }
            }
        }
    };
    if let Err((msg, cost)) = outcome {
        sel.usd += cost;
        if !fallback(&mut sel, &pool, pool_kind, &msg) {
            return Ok(sel);
        }
    }
    sel.chosen_run_id = pool[&sel.chosen_label].run_id.clone();
    sel.chosen_is_base = is_base(&sel.chosen_label);
    let rd = pool.get(&sel.chosen_label).map(|r| r.run_dir.to_string_lossy().to_string()).unwrap_or_default();
    crate::say!("  [select.{mode}{k}] chose {} ({}){}", sel.chosen_id, sel.chosen_label, if sel.fallback { " [fallback]" } else { "" });
    crate::log::event("select", "done", json!({"chosen": sel.chosen_label, "id": sel.chosen_id, "fallback": sel.fallback, "why": util::take_chars(&sel.why, 800), "error": sel.error}));
    persist(&mut sel, &root, &rd);
    Ok(sel)
}

/// Deliver the single k=1 final without a selector session.
pub fn trivial_selection(task_id: &str, k: usize, mode: &str, candidates: &BTreeMap<String, RunView>, adapter: &dyn BenchmarkAdapter, work_dir: &Path) -> Result<Selection> {
    if candidates.len() > 1 {
        anyhow::bail!(
            "trivial_selection expects at most one candidate (the k=1 final), got {:?}",
            candidates.keys().collect::<Vec<_>>()
        );
    }
    let mut sel = Selection::new(task_id, mode, k);
    sel.trivial = true;
    let (pool, _a) = render_pool(candidates, adapter, &mut sel);
    sel.n_candidates = pool.len();
    if pool.is_empty() {
        sel.error = "no candidate produced an artifact (finals-only pool has no base fallback)".into();
        return Ok(sel);
    }
    let label = pool.keys().next().unwrap().clone();
    sel.ok = true;
    sel.chosen_id = "A".into();
    sel.chosen_label = label.clone();
    sel.id_map.insert("A".into(), label.clone());
    sel.chosen_run_id = pool[&label].run_id.clone();
    sel.chosen_is_base = is_base(&label);
    sel.why = "trivial pool: single final at k=1; delivered without a selector session".into();
    let root = work_dir.join("selection").join(format!("{mode}{k}"));
    util::rmtree(&root);
    std::fs::create_dir_all(&root)?;
    let rd = pool[&label].run_dir.to_string_lossy().to_string();
    persist(&mut sel, &root, &rd);
    crate::log::at("select", "select");
    crate::log::event("select", "done", json!({"chosen": label, "id": "A", "trivial": true, "why": sel.why}));
    Ok(sel)
}

/// Learn from harness edits, decisions, trajectories, and deliverables after adoption.
#[allow(clippy::too_many_arguments)]
pub fn reflect_adoption(task_id: &str, sel: &mut Selection, candidates: &BTreeMap<String, RunView>, adapter: &dyn BenchmarkAdapter, prop: &dyn Propose, work_dir: &Path, decisions: Option<&BTreeMap<String, String>>) -> Result<Map<String, Value>> {
    let Some(chosen) = candidates.get(&sel.chosen_label).cloned() else { return Ok(Map::new()) };
    if !sel.ok {
        return Ok(Map::new());
    }
    let root = work_dir.join("reflection").join(format!("{}{}", sel.mode, sel.k));
    let harnesses = work_dir.join("harnesses");
    let pool: Vec<context::PoolEntry> = candidates
        .iter()
        .filter(|(l, _)| !l.starts_with("base"))
        .map(|(l, r)| (l.clone(), harnesses.join(l), Some(r.clone())))
        .collect();
    let rolls: Vec<RunView> = candidates.values().cloned().collect();
    let evidence = context::build(&root, adapter, task_id, &rolls, Some(&harnesses.join(&sel.chosen_label)), None, &pool, decisions, None)?;
    util::copy_tree(&adapter.base_harness()?, &root.join("base_harness"))?;
    util::write(
        &root.join("adoption.json"),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({"chosen_label": sel.chosen_label, "candidate_ids": sel.id_map, "reason": sel.why}))?
        ),
    )?;
    context::assert_no_leak_paths(&root, &adapter.spec().leak_globs)?;
    context::assert_no_grader_verdicts(&root, context::GRADER_VERDICT_MARKERS)?;
    let experience = distil(task_id, sel, adapter, prop, &root, None, &evidence.layout())?;
    sel.experience = experience.clone();
    let rd = chosen.run_dir.to_string_lossy().to_string();
    persist(sel, &work_dir.join("selection").join(format!("{}{}", sel.mode, sel.k)), &rd);
    Ok(experience)
}

/// Extract a supported lesson from the complete task-local improvement record.
fn distil(task_id: &str, sel: &mut Selection, adapter: &dyn BenchmarkAdapter, prop: &dyn Propose, root: &Path, bp: Option<BenchPrompts>, layout: &str) -> Result<Map<String, Value>> {
    let bp = bp.unwrap_or_else(|| resolve_prompts(adapter, task_id));
    let mission = [
        P::experience(&bp),
        format!(
            "# WHAT HAPPENED\n\nHarness {} was adopted. The selection reasoning was:\n\n{}\n\ncontext/adoption.json maps any anonymous IDs in that reasoning to rollout labels. Compare context/harness/ with context/base_harness/ and the candidates in context/pool/. Read the decisions and execution contrasts before writing.\n\n{layout}",
            sel.chosen_label, sel.why
        ),
    ]
    .join("\n\n");
    let exp_root = root.with_file_name(format!("{}.experience", root.file_name().unwrap().to_string_lossy()));
    util::rmtree(&exp_root);
    util::copy_tree_filtered(root, &exp_root, |rel| !rel.split('/').any(|p| p == "selection.json"))?;
    context::assert_no_grader_verdicts(&exp_root, context::GRADER_VERDICT_MARKERS)?;
    crate::log::at("learn", "learn");
    crate::log::event("learn", "start", json!({}));
    crate::say!("  [experience.{}{}] reflecting on the adopted harness", sel.mode, sel.k);
    // Every way out records why no lesson was kept, with the edit rows (upstream #143).
    let none_kept = |outcome: &str, error: &str, edits: Option<&String>| -> Map<String, Value> {
        crate::log::event("learn", "done", json!({"entry": false, "outcome": outcome}));
        let mut m = Map::new();
        m.insert("outcome".into(), json!(outcome));
        if !error.is_empty() {
            m.insert("error".into(), json!(util::take_chars(error, 600)));
        }
        if let Some(e) = edits {
            m.insert("edits".into(), json!(e));
        }
        m
    };
    let reply = match prop.call(&mission, Some(&exp_root), &format!("experience.{}{}", sel.mode, sel.k), CallOpts::default()) {
        Err(e) => {
            sel.usd += e.cost_usd;
            crate::log::event("learn", "failed", json!({"error": util::take_chars(&e.message, 600)}));
            return Ok(none_kept("session_failed", &e.message, None));
        }
        Ok(r) => r,
    };
    sel.usd += reply.cost_usd;
    let edits = EDITS.captures(&reply.text).map(|c| c[1].trim().to_string());
    if let Some(e) = &edits {
        util::write(&root.join("edits.md"), format!("{e}\n"))?;
    }
    let bad = edits.as_deref().map(library::bad_verdicts).unwrap_or_default();
    let Some(m) = EXP.captures(&reply.text) else {
        let declined = reply.text.lines().any(|l| l.trim() == "NO_ENTRY");
        let error = if bad.is_empty() { String::new() } else { format!("verdicts outside the four allowed: {}", bad.join(", ")) };
        return Ok(none_kept(if declined { "declined" } else { "malformed" }, &error, edits.as_ref()));
    };
    let mut out = Map::new();
    for line in m[1].trim().lines() {
        if let Some((key, val)) = line.split_once(':') {
            let key = key.trim().to_lowercase();
            if ["cue", "lesson", "evidence", "confidence"].contains(&key.as_str()) {
                out.insert(key, json!(val.trim()));
            }
        }
    }
    let missing: Vec<&str> = ["cue", "lesson", "evidence", "confidence"].into_iter().filter(|k| !out.get(*k).and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false)).collect();
    if !missing.is_empty() || !bad.is_empty() {
        let mut why = vec![];
        if !missing.is_empty() {
            why.push(format!("missing {}", missing.join(", ")));
        }
        if !bad.is_empty() {
            why.push(format!("verdicts outside the four allowed: {}", bad.join(", ")));
        }
        return Ok(none_kept("malformed", &why.join("; "), edits.as_ref()));
    }
    let prov = provenance(out["evidence"].as_str().unwrap_or(""), edits.as_deref().unwrap_or(""));
    out.insert("provenance".into(), json!(prov));
    if prov == "unverified" {
        return Ok(none_kept("unverified", "", edits.as_ref()));
    }
    crate::log::event("learn", "done", json!({"entry": true, "cue": out.get("cue"), "lesson": out.get("lesson"), "provenance": prov}));
    if let Some(e) = edits {
        out.insert("edits".into(), json!(e));
    }
    let mut probe = BTreeMap::new();
    probe.insert("EXPERIENCE.md".to_string(), Value::Object(out.clone()).to_string());
    context::assert_tree_has_no_grader_verdicts(&probe)?;
    Ok(out)
}

/// A run label of this task: base1, inv2, dom1, agg3, c1, ...
static RUN_LABEL: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(base|inv|dom|agg|c)\d+\b").unwrap());

/// Where a lesson's evidence comes from (upstream #143, ask 3), read from what the
/// reflection already writes, so the prompt stays as upstream has it:
/// - `measured`: the evidence names a run of this task, or an edit row whose `candidates`
///   names one.
/// - `upstream`: it rests on an earlier library entry instead.
/// - `unverified`: neither; such a lesson is not stored.
pub fn provenance(evidence: &str, edits: &str) -> &'static str {
    if RUN_LABEL.is_match(evidence) {
        return "measured";
    }
    let mut component = String::new();
    for line in edits.lines() {
        let line = line.trim();
        if let Some(c) = line.strip_prefix("component:") {
            component = c.trim().trim_matches('`').to_string();
        } else if let Some(c) = line.strip_prefix("candidates:") {
            if !component.is_empty() && evidence.contains(component.as_str()) && RUN_LABEL.is_match(c) {
                return "measured";
            }
        }
    }
    let low = evidence.to_lowercase();
    if ["library", "harnesses/", "earlier task", "previous task", "prior task"].iter().any(|w| low.contains(w)) {
        return "upstream";
    }
    "unverified"
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn provenance_reads_the_citations() {
        let rows = "component: scripts/summarize.py\ncandidates: inv1\nverdict: helped";
        assert_eq!(provenance("the scripts/summarize.py row", rows), "measured");
        assert_eq!(provenance("agg2 diverged at step 4", ""), "measured");
        assert_eq!(provenance("the library entry for ledger-1 showed this", ""), "upstream");
        assert_eq!(provenance("it seems right", rows), "unverified");
        assert_eq!(provenance("the cinv1 thing", ""), "unverified");
    }
    #[test]
    fn verdicts_are_checked() {
        assert!(library::bad_verdicts("verdict: helped\nverdict: confounded -- two edits\nverdict: `no observed effect`").is_empty());
        assert_eq!(library::bad_verdicts("verdict: probably helped\nverdict: helpedish"), vec!["probably helped", "helpedish"]);
    }
    #[test]
    fn anon_ids() {
        assert_eq!(anon_id(0), "A");
        assert_eq!(anon_id(25), "Z");
        assert_eq!(anon_id(26), "AA");
        assert_eq!(anon_id(52), "BA");
    }
    #[test]
    fn ids_match_python() {
        // python: assign_ids(["agg1","agg2","agg3"], task_id="t", k=3, mode="agg")
        let got = assign_ids(&["agg1".into(), "agg2".into(), "agg3".into()], "t", 3, "agg");
        let want: BTreeMap<String, String> = serde_json::from_str(include_str!("../../tests/golden/assign_ids.json")).unwrap();
        assert_eq!(got, want);
    }
}
