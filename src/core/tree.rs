//! The c1..c8 component tree: read, validate, write, diff — parametrised by `SurfaceSpec`.
//!
//! Everything that differs between agent frameworks is data on `SurfaceSpec` (paths, caps,
//! frontmatter rules, README seeds) plus a per-surface hooks-schema validator; the shared
//! logic exists once. The edit contract is omitted-is-inherited: a proposal emits only what it
//! changes.

use crate::util;
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

/// rel path -> content. Ordered so that iteration is deterministic.
pub type Tree = BTreeMap<String, String>;

pub const RESERVED_SCRIPTS: &[&str] = &["read_doc.py", "pack.py", "unpack.py"];

const NAME: &str = r"[a-z0-9][a-z0-9_\-]*";

/// The editable surface of one agent framework: paths, caps and per-surface rules.
#[derive(Debug, Clone)]
pub struct SurfaceSpec {
    pub name: &'static str,
    pub always_on: Vec<&'static str>,
    pub always_on_max: usize,
    pub always_on_cap_reason: &'static str,
    pub skill_dir: &'static str,
    /// "must_match" or "overrides_dir".
    pub skill_name_rule: &'static str,
    /// Contains `{name}`.
    pub trigger_hint: &'static str,
    pub autotrigger_file: &'static str,
    /// Contains `{name}` and `{desc}`.
    pub trigger_line: &'static str,
    pub subagent_requires_mode: bool,
    pub plugin_dir: Option<&'static str>,
    pub hooks_file: Option<&'static str>,
    pub hook_events: Vec<&'static str>,
    pub compile_skill_scripts: bool,
    pub reject_empty_always_on: bool,
    /// Directory of wired tools (`<dir>/<name>.json`) the agent calls as first-class tools.
    /// Only the native substrate has them; the reference surfaces leave this `None`.
    pub tool_dir: Option<&'static str>,
    pub stubs: Vec<(&'static str, &'static str)>,
}

struct Rx {
    skill: Regex,
    skill_script: Regex,
    subagent: Regex,
    script: Regex,
    script_ref: Regex,
    plugin: Option<Regex>,
    tool: Option<Regex>,
}

fn rx(s: &SurfaceSpec) -> Rx {
    let sd = regex::escape(s.skill_dir);
    Rx {
        skill: Regex::new(&format!(r"^{sd}/({NAME})/SKILL\.md$")).unwrap(),
        skill_script: Regex::new(&format!(r"^{sd}/{NAME}/scripts/[A-Za-z0-9_.\-]+$")).unwrap(),
        subagent: Regex::new(&format!(r"^sub_agents/({NAME})/AGENT\.md$")).unwrap(),
        script: Regex::new(r"^scripts/[a-z0-9_][a-z0-9_]*\.py$").unwrap(),
        script_ref: Regex::new(&format!(r"(?:python3?\s+)((?:{sd}/[\w\-]+/)?scripts/[\w.\-]+\.py)"))
            .unwrap(),
        plugin: s
            .plugin_dir
            .map(|p| Regex::new(&format!(r"^{}/({NAME})\.ts$", regex::escape(p))).unwrap()),
        tool: s.tool_dir.map(|d| Regex::new(&format!(r"^{}/({NAME})\.json$", regex::escape(d))).unwrap()),
    }
}

/// Normalise a relative path, rejecting escapes before it is joined to a directory.
pub fn normalize_rel(rel: &str) -> Result<String> {
    let r = rel.trim().replace('\\', "/");
    if r.starts_with('/') || r.split('/').any(|p| p == "..") {
        bail!("path escapes the harness: {rel:?}");
    }
    let mut r = r.as_str();
    while let Some(x) = r.strip_prefix("./") {
        r = x;
    }
    Ok(r.trim_matches('/').to_string())
}

/// Whether this path is on the editable surface. A predicate: it never fails.
pub fn is_allowed(surface: &SurfaceSpec, rel: &str) -> bool {
    let Ok(rel) = normalize_rel(rel) else { return false };
    if surface.always_on.contains(&rel.as_str()) {
        return true;
    }
    if surface.hooks_file == Some(rel.as_str()) {
        return true;
    }
    let r = rx(surface);
    if r.skill.is_match(&rel)
        || r.skill_script.is_match(&rel)
        || r.subagent.is_match(&rel)
        || r.script.is_match(&rel)
    {
        return true;
    }
    r.plugin.map(|p| p.is_match(&rel)).unwrap_or(false) || r.tool.map(|t| t.is_match(&rel)).unwrap_or(false)
}

/// Built-in tool names a wired tool may not shadow.
pub const BUILTIN_TOOLS: &[&str] = &["list_files", "read_file", "write_file", "edit_file", "delete_file", "finish", "bash", "skill", "task", "remember", "recall"];

/// Check every `tools/<name>.json`: valid JSON, `name` matching the file, a description, a command,
/// an object schema, and any `scripts/x.py` it runs present in the harness.
fn check_tools(tree: &Tree, r: &Rx, errs: &mut Vec<String>) -> usize {
    let Some(rt) = &r.tool else { return 0 };
    let mut n = 0;
    for (rel, src) in tree {
        let Some(c) = rt.captures(rel) else { continue };
        n += 1;
        let stem = c[1].to_string();
        let v: Value = match serde_json::from_str(src) {
            Ok(v) => v,
            Err(e) => {
                errs.push(format!("{rel} is not valid JSON: {e}"));
                continue;
            }
        };
        let name = v.get("name").and_then(|x| x.as_str()).unwrap_or("");
        if name != stem {
            errs.push(format!("{rel}: `name` must be {stem:?} (the file name), got {name:?}"));
        }
        if BUILTIN_TOOLS.contains(&stem.as_str()) {
            errs.push(format!("{rel}: {stem:?} is a built-in tool name; pick another"));
        }
        if v.get("description").and_then(|x| x.as_str()).map(|d| d.trim().is_empty()).unwrap_or(true) {
            errs.push(format!("{rel}: needs a non-empty `description` saying when to call it"));
        }
        let cmd = v.get("command").and_then(|x| x.as_str()).unwrap_or("");
        if cmd.trim().is_empty() {
            errs.push(format!("{rel}: needs a `command` (run with bash in the workspace; arguments arrive as JSON on stdin)"));
        }
        if let Some(p) = v.get("parameters") {
            if p.get("type").and_then(|t| t.as_str()) != Some("object") {
                errs.push(format!("{rel}: `parameters` must be a JSON Schema with \"type\": \"object\""));
            }
        }
        for c in r.script_ref.captures_iter(cmd) {
            if !tree.contains_key(&c[1]) {
                errs.push(format!("{rel} runs `{}` but that file is not in the harness", &c[1]));
            }
        }
    }
    n
}

pub fn skill_names(surface: &SurfaceSpec, tree: &Tree) -> Vec<String> {
    let r = rx(surface);
    let set: BTreeSet<String> =
        tree.keys().filter_map(|k| r.skill.captures(k).map(|c| c[1].to_string())).collect();
    set.into_iter().collect()
}

pub fn subagent_names(surface: &SurfaceSpec, tree: &Tree) -> Vec<String> {
    let r = rx(surface);
    let set: BTreeSet<String> =
        tree.keys().filter_map(|k| r.subagent.captures(k).map(|c| c[1].to_string())).collect();
    set.into_iter().collect()
}

fn stub_paths(surface: &SurfaceSpec) -> HashSet<&'static str> {
    surface.stubs.iter().map(|(p, _)| *p).collect()
}

/// rel -> content for every file on the editable surface, README stubs excluded.
/// Pass `rejected` to collect paths that exist on disk but are off the surface.
pub fn read_tree(surface: &SurfaceSpec, d: &Path, mut rejected: Option<&mut Vec<String>>) -> Tree {
    let mut out = Tree::new();
    if !d.is_dir() {
        return out;
    }
    let stubs = stub_paths(surface);
    for p in util::files_sorted(d) {
        if util::in_pycache(&p) {
            continue;
        }
        let rel = util::rel(d, &p);
        if stubs.contains(rel.as_str()) {
            continue;
        }
        if !is_allowed(surface, &rel) {
            if let Some(r) = rejected.as_deref_mut() {
                r.push(rel);
            }
            continue;
        }
        out.insert(rel, util::read_lossy(&p).unwrap_or_default());
    }
    out
}

pub fn is_tree(surface: &SurfaceSpec, d: &Path) -> bool {
    d.is_dir() && surface.always_on.iter().any(|a| d.join(a).is_file())
}

/// Materialise an exact editable tree, with always-on files and optional stubs present.
/// Files outside the editable surface are preserved; editable files already there are not.
pub fn write_tree(surface: &SurfaceSpec, tree: &Tree, dest: &Path, stubs: bool) -> Result<PathBuf> {
    std::fs::create_dir_all(dest)?;
    let stub_set = stub_paths(surface);
    let mut existing = util::files_sorted(dest);
    existing.reverse();
    for p in existing {
        let rel = util::rel(dest, &p);
        if is_allowed(surface, &rel) || stub_set.contains(rel.as_str()) {
            std::fs::remove_file(&p)?;
        }
    }
    for a in &surface.always_on {
        let p = dest.join(a);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !tree.contains_key(*a) && !p.exists() {
            std::fs::write(&p, "")?;
        }
    }
    for (rel, content) in tree {
        let p = dest.join(normalize_rel(rel)?);
        util::write(&p, content)?;
    }
    if stubs {
        for (rel, body) in &surface.stubs {
            let p = dest.join(rel);
            if !p.exists() {
                util::write(&p, body)?;
            }
        }
    }
    Ok(dest.to_path_buf())
}

pub fn always_on_text(surface: &SurfaceSpec, tree: &Tree) -> String {
    surface
        .always_on
        .iter()
        .map(|a| tree.get(*a).cloned().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn frontmatter(body: &str) -> String {
    util::take_chars(body, 600)
}

/// Give every skill a trigger line in always-on text; return the names repaired.
/// A line that would push the body over the always-on cap is skipped instead.
pub fn autotrigger(surface: &SurfaceSpec, tree: &mut Tree, always_on_max: Option<usize>) -> Vec<String> {
    let cap = always_on_max.unwrap_or(surface.always_on_max);
    let mut body = always_on_text(surface, tree);
    let mut fixed = vec![];
    let desc_re = Regex::new(r"(?m)^description:\s*(.+)$").unwrap();
    for name in skill_names(surface, tree) {
        if body.contains(&name) {
            continue;
        }
        let fm = frontmatter(tree.get(&format!("{}/{name}/SKILL.md", surface.skill_dir)).map(|s| s.as_str()).unwrap_or(""));
        let desc = desc_re
            .captures(&fm)
            .map(|c| c[1].trim().trim_end_matches('.').to_string())
            .unwrap_or_else(|| format!("the {name} workflow applies"));
        let line = surface.trigger_line.replace("{name}", &name).replace("{desc}", &desc);
        let cur = tree.get(surface.autotrigger_file).cloned().unwrap_or_default();
        if util::char_len(&always_on_text(surface, tree)) + util::char_len(&line) + 2 > cap {
            continue;
        }
        let new = if cur.trim().is_empty() { line } else { format!("{}\n{}", cur.trim_end(), line) };
        tree.insert(surface.autotrigger_file.to_string(), new);
        body = always_on_text(surface, tree);
        fixed.push(name);
    }
    fixed
}

/// Validate the hooks file against the schema the framework dispatches; a malformed one never
/// fires rather than erroring, so the shape is enforced where the proposer sees it.
pub fn check_hooks(surface: &SurfaceSpec, src: &str, errs: &mut Vec<String>) {
    let hf = surface.hooks_file.unwrap_or("hooks.json");
    let doc: Value = match serde_json::from_str(src) {
        Ok(v) => v,
        Err(e) => {
            errs.push(format!("{hf} is not valid JSON: {e}"));
            return;
        }
    };
    let hooks = doc.get("hooks").and_then(|h| h.as_object());
    let Some(hooks) = hooks.filter(|h| !h.is_empty()) else {
        errs.push(format!(r#"{hf} needs a top-level {{"hooks": {{"<Event>": [...]}}}} object"#));
        return;
    };
    let mut events: Vec<&str> = surface.hook_events.clone();
    events.sort();
    for (event, entries) in hooks {
        if !surface.hook_events.contains(&event.as_str()) {
            errs.push(format!(
                "{hf}: unknown event {event:?}; the framework dispatches {} (PascalCase)",
                events.join(", ")
            ));
            continue;
        }
        let Some(entries) = entries.as_array() else {
            errs.push(format!("{hf}: {event} must map to a LIST of matcher entries"));
            continue;
        };
        for (i, entry) in entries.iter().enumerate() {
            let Some(hs) = entry.get("hooks").and_then(|h| h.as_array()) else {
                errs.push(format!("{hf}: {event}[{i}] needs a `hooks` list"));
                continue;
            };
            for h in hs {
                if h.get("type").and_then(|t| t.as_str()) != Some("command") {
                    errs.push(format!(
                        "{hf}: {event}[{i}] hook type must be \"command\" — async/prompt/agent hooks are skipped as 'not supported yet'"
                    ));
                } else if h.get("command").map(value_str).unwrap_or_default().trim().is_empty() {
                    errs.push(format!("{hf}: {event}[{i}] has an empty `command`"));
                }
            }
        }
    }
}

fn value_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        o => o.to_string(),
    }
}

/// Byte-compile Python source with the host interpreter, if there is one. Without Python the
/// check is skipped (the reference implementation runs inside Python and always has it).
pub fn py_compiles(rel: &str, src: &str, errs: &mut Vec<String>) {
    let Some(py) = util::which("python3").or_else(|| util::which("python")) else { return };
    let code = "import sys\nsrc=sys.stdin.read()\ntry:\n    compile(src, sys.argv[1], 'exec')\nexcept SyntaxError as e:\n    print(f'SyntaxError: {e.msg} (line {e.lineno})'); sys.exit(1)\nexcept ValueError as e:\n    print(f'ValueError: {e}'); sys.exit(1)\n";
    let mut cmd = std::process::Command::new(py);
    cmd.arg("-I").arg("-c").arg(code).arg(rel);
    if let Ok(out) = util::run_with_timeout(cmd, Some(src), std::time::Duration::from_secs(30)) {
        if out.code == Some(1) {
            let detail = util::take_chars(out.stdout.trim(), 160);
            errs.push(format!("{rel} does not compile: {detail}"));
        }
    }
}

fn check_skills(surface: &SurfaceSpec, tree: &Tree, body: &str, errs: &mut Vec<String>) -> Vec<String> {
    let skills = skill_names(surface, tree);
    let desc = Regex::new(r"(?m)^description:\s*\S").unwrap();
    let name_re = Regex::new(r"(?m)^name:\s*(\S+)\s*$").unwrap();
    for name in &skills {
        let sd = surface.skill_dir;
        let fm = frontmatter(&tree[&format!("{sd}/{name}/SKILL.md")]);
        if !desc.is_match(&fm) {
            errs.push(format!(
                "{sd}/{name}/SKILL.md needs a `description:` in its frontmatter, or the skill is dropped SILENTLY at load"
            ));
        }
        let m_name = name_re.captures(&fm).map(|c| c[1].to_string());
        if surface.skill_name_rule == "must_match" {
            if m_name.as_deref() != Some(name.as_str()) {
                errs.push(format!(
                    "{sd}/{name}/SKILL.md needs frontmatter `name: {name}` matching its directory (a mismatch is dropped SILENTLY at load)"
                ));
            }
        } else if let Some(m) = m_name.filter(|m| m != name) {
            errs.push(format!(
                "{sd}/{name}/SKILL.md frontmatter `name: {m}` OVERRIDES the directory name at load, so every trigger line naming `{name}` stops matching; drop the `name:` line or make it `name: {name}`"
            ));
        }
        if !body.contains(name.as_str()) {
            errs.push(format!(
                "skill '{name}' is never named in the always-on text, so the agent will not know to invoke it — an untriggered skill is dead weight. {}",
                surface.trigger_hint.replace("{name}", name)
            ));
        }
    }
    skills
}

fn check_subagents(surface: &SurfaceSpec, tree: &Tree, allow: bool, errs: &mut Vec<String>) -> Vec<String> {
    let subs = subagent_names(surface, tree);
    if !subs.is_empty() && !allow {
        errs.push(format!("sub_agents are not available on this deployment: {}", py_list(&subs)));
    }
    let mode = Regex::new(r"(?m)^mode:\s*subagent\s*$").unwrap();
    let desc = Regex::new(r"(?m)^description:\s*\S").unwrap();
    for name in &subs {
        let fm = frontmatter(&tree[&format!("sub_agents/{name}/AGENT.md")]);
        if surface.subagent_requires_mode && !mode.is_match(&fm) {
            errs.push(format!("sub_agents/{name}/AGENT.md needs frontmatter `mode: subagent`"));
        }
        if !desc.is_match(&fm) {
            errs.push(format!("sub_agents/{name}/AGENT.md needs a `description:`"));
        }
    }
    subs
}

/// Python's repr of a list of strings, for messages that quote one.
pub fn py_list(items: &[String]) -> String {
    format!("[{}]", items.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(", "))
}

fn check_script_refs(surface: &SurfaceSpec, tree: &Tree, r: &Rx, reserved: &HashSet<String>, errs: &mut Vec<String>) {
    let have: HashSet<&String> =
        tree.keys().filter(|k| r.script.is_match(k) || r.skill_script.is_match(k)).collect();
    for (rel, src) in tree {
        if !(r.skill.is_match(rel) || surface.always_on.contains(&rel.as_str())) {
            continue;
        }
        for c in r.script_ref.captures_iter(src) {
            let reference = c[1].to_string();
            let base = reference.rsplit('/').next().unwrap_or("").to_string();
            if reserved.contains(&base) {
                continue;
            }
            if !have.contains(&reference) {
                errs.push(format!("{rel} invokes `{reference}` but that file is not in the harness"));
            }
        }
    }
}

/// Validate; on failure the error lists every problem. On success, return a census.
pub fn validate(
    surface: &SurfaceSpec,
    tree: &Tree,
    allow_subagents: bool,
    always_on_max: Option<usize>,
    reserved_scripts: &HashSet<String>,
) -> Result<Value> {
    let mut errs: Vec<String> = vec![];
    let r = rx(surface);
    let cap = always_on_max.unwrap_or(surface.always_on_max);
    let body = always_on_text(surface, tree);
    let blen = util::char_len(&body);
    if blen > cap {
        errs.push(format!(
            "always-on content is {blen} chars, over the {cap} cap; {}",
            surface.always_on_cap_reason
        ));
    }
    if surface.reject_empty_always_on && body.trim().is_empty() {
        errs.push(format!(
            "{} is empty — it is the only always-on file on this framework, so an empty one is a harness that does not exist",
            surface.always_on[0]
        ));
    }
    let skills = check_skills(surface, tree, &body, &mut errs);
    let subs = check_subagents(surface, tree, allow_subagents, &mut errs);
    for (rel, src) in tree {
        if r.script.is_match(rel)
            || (surface.compile_skill_scripts && r.skill_script.is_match(rel) && rel.ends_with(".py"))
        {
            py_compiles(rel, src, &mut errs);
        }
    }
    if let Some(hf) = surface.hooks_file {
        if let Some(src) = tree.get(hf) {
            check_hooks(surface, src, &mut errs);
        }
    }
    check_script_refs(surface, tree, &r, reserved_scripts, &mut errs);
    let n_tools = check_tools(tree, &r, &mut errs);
    if !errs.is_empty() {
        bail!("{}", errs.join(" | "));
    }
    let mut out = json!({
        "always_on_chars": blen,
        "n_files": tree.len(),
        "skills": skills.len(),
        "sub_agents": subs.len(),
        "scripts": tree.keys().filter(|k| r.script.is_match(k)).count(),
    });
    if let Some(p) = &r.plugin {
        out["plugins"] = json!(tree.keys().filter(|k| p.is_match(k)).count());
    }
    if let Some(hf) = surface.hooks_file {
        out["hooks"] = json!(if tree.contains_key(hf) { 1 } else { 0 });
    }
    if r.tool.is_some() {
        out["tools"] = json!(n_tools);
    }
    Ok(out)
}

pub fn reserved_default() -> HashSet<String> {
    RESERVED_SCRIPTS.iter().map(|s| s.to_string()).collect()
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DiffSummary {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub modified: Vec<String>,
    pub always_on_char_delta: i64,
}

impl DiffSummary {
    pub fn unchanged(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }
}

pub fn diff_summary(surface: &SurfaceSpec, before: &Tree, after: &Tree) -> DiffSummary {
    DiffSummary {
        added: after.keys().filter(|k| !before.contains_key(*k)).cloned().collect(),
        removed: before.keys().filter(|k| !after.contains_key(*k)).cloned().collect(),
        modified: before
            .iter()
            .filter(|(k, v)| after.get(*k).map(|a| a != *v).unwrap_or(false))
            .map(|(k, _)| k.clone())
            .collect(),
        always_on_char_delta: util::char_len(&always_on_text(surface, after)) as i64
            - util::char_len(&always_on_text(surface, before)) as i64,
    }
}

/// Pairwise line-Jaccard over candidate trees, so collapse to one body is visible.
pub fn diversity(trees: &BTreeMap<String, Tree>) -> Value {
    let names: Vec<&String> = trees.keys().collect();
    if names.len() < 2 {
        return json!({"n": names.len(), "n_distinct": names.len(), "mean_line_jaccard": 1.0});
    }
    let lines = |t: &Tree| -> HashSet<String> {
        t.iter()
            .flat_map(|(rel, c)| {
                c.lines().filter(|l| !l.trim().is_empty()).map(move |l| format!("{rel}:{l}"))
            })
            .collect()
    };
    let sets: Vec<HashSet<String>> = names.iter().map(|n| lines(&trees[*n])).collect();
    let mut sims = vec![];
    for i in 0..names.len() {
        for j in i + 1..names.len() {
            let u = sets[i].union(&sets[j]).count();
            let inter = sets[i].intersection(&sets[j]).count();
            sims.push(if u > 0 { inter as f64 / u as f64 } else { 1.0 });
        }
    }
    let distinct: HashSet<Vec<(&String, &String)>> =
        trees.values().map(|t| t.iter().collect()).collect();
    let mean = sims.iter().sum::<f64>() / sims.len() as f64;
    json!({
        "n": names.len(),
        "n_distinct": distinct.len(),
        "mean_line_jaccard": (mean * 10000.0).round() / 10000.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::substrate::opencode::opencode_surface;

    #[test]
    fn allowed_paths() {
        let s = opencode_surface();
        assert!(is_allowed(&s, "systemprompt.md"));
        assert!(is_allowed(&s, "skills/doc_review/SKILL.md"));
        assert!(is_allowed(&s, "scripts/check_totals.py"));
        assert!(is_allowed(&s, "plugin/guard.ts"));
        assert!(!is_allowed(&s, "skills/Doc/SKILL.md"));
        assert!(!is_allowed(&s, "../x"));
        assert!(!is_allowed(&s, "AGENTS.md"));
    }

    #[test]
    fn untriggered_skill_is_rejected_and_autotrigger_repairs_it() {
        let s = opencode_surface();
        let mut t = Tree::new();
        t.insert("systemprompt.md".into(), "Do the work.".into());
        t.insert(
            "skills/ledger/SKILL.md".into(),
            "---\nname: ledger\ndescription: reconciling a ledger.\n---\nbody".into(),
        );
        let e = validate(&s, &t, true, None, &reserved_default()).unwrap_err().to_string();
        assert!(e.contains("never named in the always-on text"));
        let fixed = autotrigger(&s, &mut t, None);
        assert_eq!(fixed, vec!["ledger".to_string()]);
        assert!(t["guardrails.md"].contains("When reconciling a ledger, load the `ledger` skill."));
        validate(&s, &t, true, None, &reserved_default()).unwrap();
    }

    #[test]
    fn missing_script_reference_is_rejected() {
        let s = opencode_surface();
        let mut t = Tree::new();
        t.insert("systemprompt.md".into(), "Run python scripts/missing.py first.".into());
        let e = validate(&s, &t, true, None, &reserved_default()).unwrap_err().to_string();
        assert!(e.contains("invokes `scripts/missing.py`"));
    }

    #[test]
    fn write_and_read_roundtrip_skips_stubs() {
        let s = opencode_surface();
        let d = tempfile::tempdir().unwrap();
        let mut t = Tree::new();
        t.insert("systemprompt.md".into(), "x".into());
        write_tree(&s, &t, d.path(), true).unwrap();
        assert!(d.path().join("skills/README.md").is_file());
        let back = read_tree(&s, d.path(), None);
        assert_eq!(back.len(), 4); // the four always-on files
        assert_eq!(back["systemprompt.md"], "x");
    }
}
