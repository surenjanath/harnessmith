//! Every prompt the method sends. Benchmark text arrives as values, never as branches.
//!
//! The stage prompts are identical across benchmarks; only the `BenchPrompts` slots differ.
//! The substrate enters exactly once per authoring prompt, as the `# WHAT YOU CAN CHANGE`
//! surface block selected by the SurfaceSpec's name. No prompt here names a rubric, a score or
//! a judge. The long constant blocks are embedded verbatim from the reference implementation.

use super::spec::BenchPrompts;
use anyhow::{bail, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

pub const VERSION: &str = "harness2-1";

pub const ANSWER_FREE: &str = include_str!("prompt_text/answer_free.txt");
pub const ANSWER_FREE_SELECT: &str = include_str!("prompt_text/answer_free_select.txt");
pub const EVIDENCE_BOUNDARY: &str = include_str!("prompt_text/evidence_boundary.txt");
pub const POOL_MIXED_SELECT: &str = include_str!("prompt_text/pool_mixed_select.txt");
pub const POOL_FINALS_SELECT: &str = include_str!("prompt_text/pool_finals_select.txt");
pub const EVIDENCE_BOUNDARY_SELECT: &str = include_str!("prompt_text/evidence_boundary_select.txt");
const SURFACE_OPENCODE: &str = include_str!("prompt_text/surface_opencode.txt");
const SURFACE_CODEX: &str = include_str!("prompt_text/surface_codex.txt");
const SURFACE_NATIVE: &str = include_str!("prompt_text/surface_native.txt");
pub const GENERALIZE: &str = include_str!("prompt_text/generalize.txt");
const KEEP_BASE: &str = include_str!("prompt_text/keep_base.txt");
const INV_TEMPLATE: &str = include_str!("prompt_text/inv_template.txt");
const DOM_TEMPLATE: &str = include_str!("prompt_text/dom_template.txt");
const TTS_SURFACE_HEAD_OPENCODE: &str = include_str!("prompt_text/tts_surface_head_opencode.txt");
const TTS_SURFACE_HEAD_CODEX: &str = include_str!("prompt_text/tts_surface_head_codex.txt");
const TTS_SURFACE_HEAD_NATIVE: &str = include_str!("prompt_text/tts_surface_head_native.txt");
const TTS_INV: &str = include_str!("prompt_text/tts_inv.txt");
const TTS_DOM: &str = include_str!("prompt_text/tts_dom.txt");
const TTS_GENERALIZE: &str = include_str!("prompt_text/tts_generalize.txt");
const TTS_HOW: &str = include_str!("prompt_text/tts_how.txt");
pub const CONFORMANCE_BLOCK: &str = include_str!("prompt_text/conformance_block.txt");
pub const VALUE_TAIL: &str = include_str!("prompt_text/value_tail.txt");

/// Collapse blank runs left by an empty optional slot.
fn tidy(text: &str) -> String {
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
    format!("{}\n", RE.replace_all(text, "\n\n").trim())
}

/// Single-pass `str.format` over named `{slot}`s (values are never re-scanned).
fn fmt(template: &str, slots: &[(&str, &str)]) -> String {
    static RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\{([A-Za-z_]+)\}").unwrap());
    RE.replace_all(template, |c: &regex::Captures| {
        slots
            .iter()
            .find(|(k, _)| *k == &c[1])
            .map(|(_, v)| v.to_string())
            .unwrap_or_else(|| c[0].to_string())
    })
    .into_owned()
}

fn work(bp: &BenchPrompts) -> String {
    format!("# THE WORK\n\n{}", bp.work.trim())
}

fn lens(bp: &BenchPrompts) -> String {
    let body = bp.domain_guidance.trim();
    if body.is_empty() {
        String::new()
    } else {
        format!("# DOMAIN AND TASK LENS\n\n{body}")
    }
}

/// An exact span of a SURFACE block, so no pack-specific copy of it can drift.
fn surface_part(surface_md: &str, start: &str, end: Option<&str>) -> String {
    let i = surface_md.find(start).expect("surface span start");
    let j = end.map(|e| surface_md.find(e).expect("surface span end")).unwrap_or(surface_md.len());
    surface_md[i..j].trim().to_string()
}

/// The per-substrate prompt fragments, resolved once.
struct SurfaceText {
    surface_md: String,
    inv: String,
    dom: String,
    tts_surface: String,
}

static SURFACE_TEXTS: Lazy<HashMap<&'static str, SurfaceText>> = Lazy::new(|| {
    let mut m = HashMap::new();
    let oc_tts = [
        TTS_SURFACE_HEAD_OPENCODE.to_string(),
        surface_part(SURFACE_OPENCODE, "THREE RULES THAT DISCARD", Some("THE DELIVERABLE'S SHAPE")),
        surface_part(SURFACE_OPENCODE, "# HOW TO MAKE A CHANGE", None),
    ]
    .join("\n\n");
    let cx_tts = [
        TTS_SURFACE_HEAD_CODEX.to_string(),
        surface_part(SURFACE_CODEX, "A skill only exists", Some("THE DELIVERABLE'S SHAPE")),
        surface_part(SURFACE_CODEX, "# HOW TO MAKE A CHANGE", None),
    ]
    .join("\n\n");
    let nx_tts = [
        TTS_SURFACE_HEAD_NATIVE.to_string(),
        surface_part(SURFACE_NATIVE, "THREE RULES THAT DISCARD", Some("THE DELIVERABLE'S SHAPE")),
        surface_part(SURFACE_NATIVE, "# HOW TO MAKE A CHANGE", None),
    ]
    .join("\n\n");
    m.insert(
        "opencode",
        SurfaceText {
            surface_md: SURFACE_OPENCODE.into(),
            inv: INV_TEMPLATE.replace(
                "__LOOP_LEVER__",
                "`plugin/<name>.ts` hook -- for example, preventing exit before a {deliverable} exists",
            ),
            dom: DOM_TEMPLATE.replace(
                "__TRIGGER_SENTENCE__",
                "A skill exists only once its name appears in the always-on text: write that trigger line in the same edit.",
            ),
            tts_surface: oc_tts,
        },
    );
    m.insert(
        "codex",
        SurfaceText {
            surface_md: SURFACE_CODEX.into(),
            inv: INV_TEMPLATE.replace(
                "__LOOP_LEVER__",
                "hook in `.codex/hooks.json` -- for example, blocking a finish before a {deliverable} exists, with a corrective reason",
            ),
            dom: DOM_TEMPLATE.replace(
                "__TRIGGER_SENTENCE__",
                "A skill exists only once its name appears in `AGENTS.md` as a trigger: write that trigger line in the same edit.",
            ),
            tts_surface: cx_tts,
        },
    );
    m.insert(
        "native",
        SurfaceText {
            surface_md: SURFACE_NATIVE.into(),
            inv: INV_TEMPLATE.replace(
                "__LOOP_LEVER__",
                "hook in `hooks.json` -- for example, blocking a finish before a {deliverable} exists, with a corrective reason",
            ),
            dom: DOM_TEMPLATE.replace(
                "__TRIGGER_SENTENCE__",
                "A skill exists only once its name appears in the always-on text: write that trigger line in the same edit.",
            ),
            tts_surface: nx_tts,
        },
    );
    m
});

fn st(surface: &str) -> Result<&'static SurfaceText> {
    match SURFACE_TEXTS.get(surface) {
        Some(s) => Ok(s),
        None => {
            let mut known: Vec<_> = SURFACE_TEXTS.keys().collect();
            known.sort();
            bail!("no prompt surface for {surface:?}; known: {known:?}")
        }
    }
}

fn yardstick(bp: &BenchPrompts) -> Option<String> {
    let y = bp.yardstick.trim();
    (!y.is_empty()).then(|| format!("# WHAT GOOD PROCESS LOOKS LIKE HERE\n\n{y}"))
}

fn tts_propose(bp: &BenchPrompts, dim: &str, s: &SurfaceText, k: usize, index: usize) -> String {
    let levers = if dim == "INV" { bp.inv_levers.trim() } else { bp.dom_levers.trim() };
    let block = fmt(
        if dim == "INV" { TTS_INV } else { TTS_DOM },
        &[("deliverable", &bp.deliverable_word), ("domain", &bp.domain_word), ("levers", levers)],
    );
    let mut where_ = "Do not hedge toward an average or try to cover every base -- commit to the single most defensible version of your dimension. A later stage compares and merges.".to_string();
    if k > 1 {
        where_ = format!(
            "You are candidate {index} of {k} on this dimension, drawn independently from the same evidence and never shown the others. {where_}"
        );
    }
    let mut parts = vec![
        format!(
            "You are the {} harness editor for one task in this {}. You have the CURRENT real task, FULL rollouts of it under the current (base) harness, the current harness files, and this area's library of what was adopted for earlier tasks.",
            if dim == "INV" { "INVARIANT" } else { "DOMAIN" },
            bp.domain_word
        ),
        work(bp),
        lens(bp),
        block,
        fmt(
            TTS_GENERALIZE,
            &[("domain_word", &bp.domain_word), ("DOMAIN_WORD_UPPER", &bp.domain_word.to_uppercase())],
        ),
        TTS_HOW.to_string(),
        where_,
    ];
    if let Some(y) = yardstick(bp) {
        parts.push(y);
    }
    parts.extend([ANSWER_FREE.into(), EVIDENCE_BOUNDARY.into(), s.tts_surface.clone()]);
    tidy(&parts.join("\n\n"))
}

fn tts_aggregate(bp: &BenchPrompts, s: &SurfaceText, k: usize) -> String {
    let n = 2 * k;
    let plural = if k > 1 { "s" } else { "" };
    let mut parts = vec![
        "You are the integration judge and synthesizer. What you emit is what gets ADOPTED: it runs on this task, and if selected it enters this area's library where later tasks read it. Every rule that bound the candidate proposers binds you -- merging an edit does not launder it.".to_string(),
        work(bp),
        lens(bp),
        format!(
            "# WHAT YOU HAVE\n\n{k} rollout{plural} of this task under the unmodified (base) harness, and {n} dimension-scoped candidate harnesses built on that same base -- {k} on the INVARIANT dimension (tool and procedure machinery) and {k} on the DOMAIN dimension (what a professional in this field knows). Each candidate was rolled out ONCE and you have that rollout. All of it is in your workspace:\n\n    context/task.md               the real task\n    context/harness/              the base harness files, and the tree you edit\n    context/rollouts/<label>.md   FULL trajectories, untruncated\n    context/deliverables/         what each base rollout actually PRODUCED, one entry per rollout\n    context/pool/<label>/         one per candidate: `harness/` (its files), `rollout.md`\n                                  and `produced.md` (what its rollout delivered)\n    context/library/              earlier tasks' adopted harnesses (browse INDEX.md)\n\nOpen what each side PRODUCED, not only its trajectory: judging an edit without reading its product is guessing."
        ),
        format!(
            "# YOUR JOB\n\n1. Judge each candidate edit from the evidence: did its rollout show it helped, hurt, or did nothing versus the base rollouts? The task's user request is the yardstick -- completeness against what was asked counts as much as correctness of what was produced. An edit whose skill was never loaded or whose rule was ignored changed nothing, whatever it says on paper.\n2. Synthesize ONE final harness containing only what the evidence supports: merge the good parts of both dimensions, resolve overlaps rather than concatenating, keep always-on text tiny, and keep every retained skill's trigger line narrow. Adopting a single candidate wholesale is valid when it is simply the best state.\n\nKEEPING THE BASE UNCHANGED IS A VALID OUTCOME -- if no candidate demonstrably improved on base, leave the tree as it is and say KEEP_BASE in your notes. {KEEP_BASE}"
        ),
        if k > 1 {
            format!("You are one of {k} synthesizers composing a final from this same evidence in parallel, and you will not see the others. Do not aim for the consensus answer -- commit to the harness you can defend.")
        } else {
            String::new()
        },
    ];
    if let Some(y) = yardstick(bp) {
        parts.push(y);
    }
    parts.extend([
        ANSWER_FREE.into(),
        EVIDENCE_BOUNDARY.into(),
        s.tts_surface.clone(),
        "# NOTES FORMAT\n\nIn your single `<notes>` block, one line per decision:\n\n    kept|revised|dropped -> <which edit> -> <the evidence: what the trajectories showed, with counts where sets are involved>\n\nA decision without evidence attached is a decision the next reader cannot audit.".into(),
    ]);
    tidy(&parts.join("\n\n"))
}

fn select_looking_at(bp: &BenchPrompts, assets_note: &str) -> String {
    format!(
        "# WHAT YOU ARE LOOKING AT\n\n{}    context/task.md              the request, verbatim\n    context/source/             the material the work was done from\n    context/candidates/<id>.md  what that run PRODUCED -- the {}, rendered\n{}\nYou are choosing between ARTIFACTS. You are not told how any of them was produced and you cannot see the runs that made them, so judge each one on what it is, against the request and the source -- never on how impressive or thorough it sounds.",
        if bp.prompt_pack == "tts" {
            "Each candidate is what one agent run produced for the SAME task, in full.\n\n"
        } else {
            ""
        },
        bp.deliverable_word,
        assets_note
    )
}

fn tts_select(bp: &BenchPrompts, n: usize, pool_kind: &str, assets_note: &str) -> String {
    let d = &bp.deliverable_word;
    tidy(&[
        format!("You are choosing which of {n} candidate ANSWERS to a workplace task should be delivered to the person who asked for this work."),
        work(bp),
        (if pool_kind == "mixed" { POOL_MIXED_SELECT } else { POOL_FINALS_SELECT }).to_string(),
        select_looking_at(bp, assets_note),
        format!("# HOW TO DECIDE, IN THIS ORDER\n\n1. COVERAGE. Is every {d} the instructions name present, and does each contain every section, field, value or analysis they ask for? Count what the request explicitly asks to be produced, computed, stated or explained, and prefer the candidate that answers the most of those points.\n\n2. CORRECTNESS. Are the values internally consistent -- totals matching their components, figures agreeing between files, counts matching what the source actually contains? Where candidates disagree on a number, check `context/source/` and prefer the one whose derivation the material supports. Reject invented specifics; where the source is silent, a candidate that says so is safer than one that fills the gap confidently.\n\n3. FIDELITY. Required formats, file names and structure, and no placeholder, TODO, unresolved note, broken reference or empty required field. A renderer-truncation marker inserted for this comparison is not a defect in the original artifact; judge only what is actually visible.\n\nLength and polish are not merits in themselves. The candidates are independent attempts of differing quality, better than each other in different respects, and it is entirely possible that the plainest one is the best answer. Pick the best available answer, not the most elaborate one."),
        ANSWER_FREE_SELECT.into(),
        EVIDENCE_BOUNDARY_SELECT.into(),
        "# OUTPUT FORMAT\n\nFirst the reasoning, then the choice as the very last thing:\n\n<why>\nAt most 5 lines. Name the criterion that decided it, and the closest runner-up with the specific thing that cost it the pick.\n</why>\n\n```choice\n<the id of the single candidate to submit>\n```".into(),
    ].join("\n\n"))
}

/// The job for one INV or DOM candidate. `round_t` set => sequential mode, pass t.
pub fn propose(bp: &BenchPrompts, dim: &str, surface: &str, k: usize, index: usize, round_t: Option<usize>) -> Result<String> {
    if dim != "INV" && dim != "DOM" {
        bail!("unknown proposal dimension {dim:?}; expected 'INV' or 'DOM'");
    }
    let s = st(surface)?;
    if bp.prompt_pack == "tts" && round_t.is_none() {
        return Ok(tts_propose(bp, dim, s, k, index));
    }
    let levers = if dim == "INV" { bp.inv_levers.trim() } else { bp.dom_levers.trim() };
    let block = fmt(
        if dim == "INV" { &s.inv } else { &s.dom },
        &[("deliverable", &bp.deliverable_word), ("domain", &bp.domain_word), ("levers", levers)],
    );
    let mut where_ = if k > 1 {
        format!("You are candidate {index} of {k} on this dimension, drawn independently from the same evidence.")
    } else {
        String::new()
    };
    if let Some(t) = round_t.filter(|t| *t > 0) {
        where_ += &format!(
            "\n\nThis is round {t}. You are editing the checkpoint the previous round produced, not the original harness, and the evidence includes every earlier round. In this stage `KEEP_BASE` means leave that CURRENT checkpoint unchanged; it never resets to the original harness."
        );
    }
    let mut how = vec![
        "# HOW TO DECIDE WHAT TO CHANGE\n".to_string(),
        "Every edit must trace to a moment in a trajectory. For each one ask: which run did what, and would this text have changed it? If you cannot name the moment, the edit is speculation and costs context for nothing. Prefer, in order:".into(),
        "  1. a DIVERGENCE -- the runs disagreed, so the agent can already do it and just needs telling. This is the cheapest win.".into(),
        "  2. a repeated concrete failure -- the same error or omission in every run.".into(),
        "  3. a risk visible in a trajectory but not yet realised.".into(),
        "Do not fix what the runs already got right, and do not restate what the harness already says clearly; duplicated instruction dilutes the rest.".into(),
    ];
    if !where_.trim().is_empty() {
        how.push(String::new());
        how.push(where_.trim().to_string());
    }
    how.push(String::new());
    how.push("Do not hedge toward an average or try to cover every base -- commit to the single most defensible version of your dimension. A later stage compares and merges.".into());
    let mut parts = vec![
        "You are adapting an autonomous agent's harness.".to_string(),
        work(bp),
        lens(bp),
        block,
        fmt(GENERALIZE, &[("domain_word", &bp.domain_word), ("DOMAIN_WORD_UPPER", &bp.domain_word.to_uppercase())]),
        how.join("\n"),
    ];
    if let Some(y) = yardstick(bp) {
        parts.push(y);
    }
    parts.extend([ANSWER_FREE.into(), EVIDENCE_BOUNDARY.into(), s.surface_md.clone()]);
    Ok(tidy(&parts.join("\n\n")))
}

/// agg-k: compose ONE final from all base evidence and all 2k candidates.
pub fn aggregate(bp: &BenchPrompts, surface: &str, k: usize, _index: usize) -> Result<String> {
    let s = st(surface)?;
    if bp.prompt_pack == "tts" {
        return Ok(tts_aggregate(bp, s, k));
    }
    let n = 2 * k;
    let plural = if k > 1 { "s" } else { "" };
    let are = if k > 1 { "s are" } else { " is" };
    let mut parts = vec![
        "You are composing THE final harness for one piece of work. What you emit is what gets ADOPTED: it runs on this task, and if selected it enters this area's library where later tasks read it. Every rule that bound the candidate proposers binds you -- you are the last gate, and merging an edit launders it.".to_string(),
        work(bp),
        lens(bp),
        format!("# WHAT YOU HAVE\n\n{k} rollout{plural} under the unmodified harness, and {n} candidate harnesses -- {k} on the INV dimension (how the agent works) and {k} on the DOM dimension (what a practitioner knows). Each candidate was rolled out ONCE and you have that rollout.\n\nThe candidates were sampled separately but share the same task, evidence and likely model biases. Agreement is support, not independent proof; contradiction is a reason to inspect the trajectories, not to vote."),
        format!("# HOW TO WEIGH THE EVIDENCE\n\nThe {k} base rollout{are} draws from ONE distribution -- read them as a set, not as {k} separate stories.\n\n- A failure that repeats across the base set is a real property of the base harness and the strongest thing you can act on. A defect in 1/{k} is noise, and an edit aimed at it is overfitting to one sample.\n- Each candidate has ONE rollout, so a single good or bad outcome is weak evidence either way. An edit earns its place when a SPECIFIC MOMENT in a trajectory shows the mechanism -- 'this looks like good practice' is not evidence; name the moment.\n- Judge the downside as heavily as the upside. The costliest failure on this work is a run that produces nothing; an edit that helps one rollout and risks emptiness elsewhere is a bad edit.\n- IF ANY ROLLOUT ASSOCIATED WITH AN EDIT PRODUCED NOTHING, say so explicitly in your notes and treat that edit as UNSAFE unless the rest of the evidence is overwhelming. Never merge an edit from a non-delivering rollout without stating why."),
        format!("# YOUR JOB\n\nCompose ONE harness. All of these are valid, judged only on what the rollouts support:\n- MERGE across candidates -- the expected outcome when an INV edit and a DOM edit each earn their place. Resolve the overlaps rather than concatenating.\n- ADOPT ONE WHOLESALE if a single candidate is simply the best state.\n- KEEP_BASE.\n\n{KEEP_BASE}"),
        if k > 1 {
            format!("You are one of {k} aggregators composing a final from this same evidence in parallel, and you will not see the others. Do not aim for the consensus answer -- commit to the harness you can defend.")
        } else {
            String::new()
        },
    ];
    if let Some(y) = yardstick(bp) {
        parts.push(y);
    }
    parts.extend([
        ANSWER_FREE.into(),
        EVIDENCE_BOUNDARY.into(),
        s.surface_md.clone(),
        format!("# NOTES FORMAT\n\nIn your single `<notes>` block, one line per decision, in this exact shape:\n\n    kept|revised|dropped -> <which edit> -> <the trajectory moment or base-set pattern that decided it, with counts where sets are involved, e.g. 'failed in {}/{k} base rollouts'>\n\nA decision without evidence attached is a decision the next reader cannot audit.", (k.saturating_sub(1)).max(2)),
    ]);
    Ok(tidy(&parts.join("\n\n")))
}

/// seq-k: compose checkpoint C_t from this round's candidates and every earlier round.
pub fn compose_round(bp: &BenchPrompts, surface: &str, t: usize, k: usize) -> Result<String> {
    let s = st(surface)?;
    let final_ = t >= k;
    let parts = [
        format!("You are composing checkpoint {t} of {k} in a sequential adaptation loop."),
        work(bp),
        lens(bp),
        format!(
            "# WHAT YOU HAVE\n\nThe rollouts under the unmodified harness (the shared evidence, which does not change between rounds), this round's INV and DOM candidates with their rollouts, and every earlier round's candidates, checkpoints and trajectories. The previous checkpoint's own rollout is this round's on-policy evidence: it shows what the harness you are editing actually does.{}",
            if t == 1 {
                "\n\nThis is the FIRST round, so the only evidence is the base rollouts and this round's two candidates."
            } else {
                ""
            }
        ),
        format!(
            "# YOUR JOB\n\n{}All of these are valid:\n- MERGE across rounds -- including a candidate REJECTED earlier, if a later rollout now argues for it.\n- REPRODUCE an earlier checkpoint verbatim, if that state was simply the best one and later rounds drifted away from it.\n- ADOPT this round's best candidate.\n- KEEP_BASE.\n\n{KEEP_BASE}\n\nHere `KEEP_BASE` means keep the CURRENT checkpoint in `context/harness/` unchanged; it does not restore the original harness. In `<notes>`, identify which rounds' edits survive and which were dropped or restored.",
            if final_ {
                "Compose THE FINAL harness, from the WHOLE history rather than just this round.\n".to_string()
            } else {
                format!("Compose checkpoint {t}, which the next round will build on.\n")
            }
        ),
        ANSWER_FREE.into(),
        EVIDENCE_BOUNDARY.into(),
        s.surface_md.clone(),
    ];
    Ok(tidy(&parts.join("\n\n")))
}

fn checks(bp: &BenchPrompts) -> String {
    let body = bp.selection_checks.trim();
    if body.is_empty() {
        String::new()
    } else {
        format!("# WHAT CORRECTNESS MEANS FOR THIS WORK\n\n{body}")
    }
}

/// Blind selection over candidate outputs, substrate-free; `pool_kind` picks the anonymity
/// block, `assets_note` names any per-candidate rendered evidence selection copied.
pub fn select(bp: &BenchPrompts, n: usize, pool_kind: &str, assets_note: &str) -> Result<String> {
    if pool_kind != "mixed" && pool_kind != "finals" {
        bail!("unknown pool_kind {pool_kind:?}; expected 'mixed' or 'finals'");
    }
    if bp.prompt_pack == "tts" {
        return Ok(tts_select(bp, n, pool_kind, assets_note));
    }
    let d = &bp.deliverable_word;
    let lead = bp.selection_lead.trim();
    let decide = format!(
        "# HOW TO DECIDE, IN THIS ORDER\n\n{}1. DELIVERY. Is every {d} named by the instructions present, at the requested path, in the requested form, with every named part?\n\n{}3. FIDELITY. CHECK THE CLAIMS AGAINST `context/source/`, which holds the material the work was done from. Open it and verify the facts, data, names, figures and constraints a candidate asserts -- do not take its word for them, and do not settle for whether it sounds authoritative. Reject invented specifics. Where candidates disagree, the source decides it; where the source is silent, prefer the candidate that says so over the one that fills the gap confidently.\n\n4. INTERNAL CONSISTENCY. Does the artifact agree with itself? Totals that match their rows, figures that agree between a summary and the table it summarises, cross-references that resolve, counts that match the number of items actually present. A number that contradicts another number in the same artifact is a defect even when you cannot tell which one is wrong.\n\n5. OPERATIONAL FITNESS. Is this the right kind of artifact, and can the intended person or system actually use, run, open, import or act on it without reconstructing missing work? Judge this as the requester's downstream consumer, not as a reviewer wanting to be convinced -- an addition that helps a human read the file but breaks an importer is a cost, not a benefit.\n\n6. FINISH. No placeholders, TODOs, unresolved notes, broken references, empty required fields, visibly incomplete sections, or content that does not match its declared format. A renderer-truncation marker inserted for this comparison is not a defect in the original artifact; judge only what is actually visible.\n\n{}Length and polish are not merits in themselves. A shorter, correct, complete, exactly-shaped answer beats a longer one with a gap or an embellishment.",
        if lead.is_empty() { String::new() } else { format!("0. {lead}\n\n") },
        if bp.selection_conformance == "value_only" { "" } else { CONFORMANCE_BLOCK },
        if bp.selection_conformance == "full" { "" } else { VALUE_TAIL },
    );
    Ok(tidy(&[
        format!("You are choosing which of {n} candidate answers should be delivered to the person who asked for this work."),
        work(bp),
        lens(bp),
        (if pool_kind == "mixed" { POOL_MIXED_SELECT } else { POOL_FINALS_SELECT }).to_string(),
        select_looking_at(bp, assets_note),
        decide,
        checks(bp),
        ANSWER_FREE_SELECT.into(),
        EVIDENCE_BOUNDARY_SELECT.into(),
        "# OUTPUT FORMAT\n\nFirst the reasoning, then the choice as the very last thing:\n\n<why>\nAt most 6 lines. Name the criterion that decided it, and the closest runner-up with the specific thing that cost it the pick. If you rejected a candidate on CONFORMANCE, say which field, column or section.\n</why>\n\n```choice\n<the id of the single candidate to submit>\n```".into(),
    ].join("\n\n")))
}

/// Two records: what each harness edit did, then the one lesson that transfers.
pub fn experience(bp: &BenchPrompts) -> String {
    tidy(&[
        "You are reading the complete record of one improvement round: a base harness, the edits proposed against it, the executions those edited harnesses produced, and which output was finally submitted. Write two things for a library that future work in this area reads before it makes its own edits.".to_string(),
        work(bp),
        lens(bp),
        "# THE ADOPTED CANDIDATE IS NOT THE ONLY EVIDENCE\n\nSelection compared DELIVERABLES with the harnesses hidden, so the choice says which output read best -- not which edit worked. Treat every candidate as evidence. An edit that appeared only in a rejected candidate may still be the one that changed behavior for the better, and an edit inside the adopted candidate may have contributed nothing to why it was adopted. Read the trajectories, not the verdict.".into(),
        "# WHAT EACH EDIT DID\n\nDiff every candidate harness against `context/base_harness/`. For each component any candidate changed, write one row: the component, what the edit actually changed, the observed difference in execution it produced, and a verdict.\n\nVerdicts are `helped`, `hurt`, `no observed effect` and `confounded`. Use `confounded` -- and prefer it -- whenever a candidate changed several components at once and no execution isolates one of them, or when a single rollout is all that stands behind the difference. A confounded row is a useful record; an invented attribution poisons the library for every later task. Cite the run and the point in the trajectory where behavior diverges. If two candidates changed the same component in different ways, say which forms were tried and what each produced.".into(),
        "# THE ONE LESSON THAT TRANSFERS\n\nNow write at most one durable entry, and only if the edit rows support it.\n\nThe `cue` describes the SHAPE OF SITUATION the lesson applies to, so it can fire on a different matter. Not this task's facts -- the pattern. The `lesson` is the operation to perform on a harness, stated so it can be applied and checked, and it must name the component it concerns.\n\nToo generic is useless ('be thorough'); too specific never fires again ('the fee rose to 250 on Jan 1'). Aim between: 'when an instrument performs several sequential acts and one of them terminates the actor's own authority, order them so the terminating act comes last'.\n\nThe `evidence` field cites the edit rows the lesson rests on. A lesson resting only on `confounded` rows is a guess: narrow it to what the record does show, or decline. Adoption alone never establishes that an edit helped or will transfer. If the record shows no concrete, reusable execution contrast, write the single line NO_ENTRY in place of the <experience> block -- the edit rows are still worth writing.".into(),
        ANSWER_FREE.into(),
        EVIDENCE_BOUNDARY.into(),
        "# OUTPUT FORMAT\n\n<edits>\ncomponent: <name as it appears in the harness tree>\ncandidates: <the labels that carried this edit>\nchange: <what the edit did>\nobserved: <the execution difference, and where you saw it>\nverdict: <helped|hurt|no observed effect|confounded>\n(repeat these five lines for each changed component)\n</edits>\n\n<experience>\ncue: <the shape of situation this applies to>\nlesson: <the transferable operation, naming the component>\nevidence: <the edit rows this rests on, and the execution contrast>\nconfidence: <high|medium|low, and why>\n</experience>".into(),
    ].join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn bps() -> Vec<(&'static str, BenchPrompts)> {
        vec![
            ("plain", BenchPrompts::work("reconcile signed amounts in a transaction ledger")),
            (
                "full",
                BenchPrompts {
                    work: "draft legal memos for a client".into(),
                    domain_word: "practice area".into(),
                    inv_levers: "Lever A: read every file.".into(),
                    dom_levers: "Lever B: cite the clause.".into(),
                    yardstick: "Good process reads first.".into(),
                    deliverable_word: "memo".into(),
                    domain_guidance: "Antitrust work needs market definition.".into(),
                    selection_checks: "Check citations.".into(),
                    selection_lead: "Prefer the one that answers.".into(),
                    selection_conformance: "value_only".into(),
                    ..Default::default()
                },
            ),
            (
                "tts",
                BenchPrompts {
                    work: "office work".into(),
                    prompt_pack: "tts".into(),
                    yardstick: "Y.".into(),
                    domain_guidance: "G.".into(),
                    ..Default::default()
                },
            ),
        ]
    }

    /// Every prompt is byte-identical to the reference implementation's output.
    #[test]
    fn prompts_match_reference_golden() {
        let gold: Value =
            serde_json::from_str(include_str!("../../tests/golden/prompts.json")).unwrap();
        let mut checked = 0;
        for (bn, bp) in bps() {
            for surf in ["opencode", "codex"] {
                for k in [1usize, 3] {
                    for dim in ["INV", "DOM"] {
                        let g = &gold[format!("propose|{bn}|{surf}|{dim}|k{k}")];
                        assert_eq!(propose(&bp, dim, surf, k, 1, None).unwrap(), g.as_str().unwrap(), "propose {bn} {surf} {dim} k{k}");
                        let g = &gold[format!("propose_seq|{bn}|{surf}|{dim}|k{k}")];
                        assert_eq!(propose(&bp, dim, surf, k, 2, Some(2)).unwrap(), g.as_str().unwrap(), "propose_seq {bn} {surf} {dim} k{k}");
                        checked += 2;
                    }
                    let g = &gold[format!("aggregate|{bn}|{surf}|k{k}")];
                    assert_eq!(aggregate(&bp, surf, k, 1).unwrap(), g.as_str().unwrap(), "aggregate {bn} {surf} k{k}");
                    for t in [1, k] {
                        let g = &gold[format!("compose|{bn}|{surf}|t{t}|k{k}")];
                        assert_eq!(compose_round(&bp, surf, t, k).unwrap(), g.as_str().unwrap(), "compose {bn} {surf} t{t} k{k}");
                    }
                    checked += 3;
                }
            }
            for kind in ["mixed", "finals"] {
                let g = &gold[format!("select|{bn}|{kind}")];
                assert_eq!(select(&bp, 3, kind, "    note line\n").unwrap(), g.as_str().unwrap(), "select {bn} {kind}");
                checked += 1;
            }
            assert_eq!(experience(&bp), gold[format!("experience|{bn}")].as_str().unwrap());
            checked += 1;
        }
        assert!(checked >= 80);
    }

    #[test]
    fn native_surface_builds() {
        let bp = BenchPrompts::work("answer a question");
        let p = propose(&bp, "INV", "native", 1, 1, None).unwrap();
        assert!(p.contains("hooks.json"));
        let p = propose(&BenchPrompts { prompt_pack: "tts".into(), ..bp }, "DOM", "native", 2, 1, None).unwrap();
        assert!(p.contains("EDITABLE HARNESS SURFACE"));
    }
}
