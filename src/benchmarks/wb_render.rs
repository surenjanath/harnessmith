//! The renderer for what a WB rollout PRODUCED: agent-authored bytes in, text out.
//!
//! `admit()` is a positive allowlist of four locations, one per subset, and fails on anything
//! else. `render`/`flat_text` are the selector's view (contents), `listing` the proposer's
//! (names only). xlsx/pptx are rendered by a small Python helper (openpyxl / python-pptx), and
//! a missing renderer is an error rather than a silent stub.

use crate::{config, util};
use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub fn allowlist(subset: &str) -> Option<&'static [&'static str]> {
    Some(match subset {
        "office" => &["verifier/raw_artifacts/**", "verifier/agent.patch"],
        "web" => &["verifier/agent.workspace.tar.gz"],
        "code" => &["verifier/agent.patch"],
        "sec" => &["artifacts/logs/artifacts/**"],
        _ => return None,
    })
}

/// Return the trial-relative path, or fail. Every read in this module goes through here.
pub fn admit(trial: &Path, path: &Path, subset: &str) -> Result<String> {
    let Some(pats) = allowlist(subset) else { bail!("no input allowlist for subset {subset:?}; known: [\"code\", \"office\", \"sec\", \"web\"]") };
    let t = std::fs::canonicalize(trial).unwrap_or_else(|_| trial.to_path_buf());
    let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let Ok(r) = p.strip_prefix(&t) else { bail!("{} is not inside the trial dir {}", path.display(), trial.display()) };
    let rel = util::rel(Path::new(""), r);
    for pat in pats {
        if let Some(head) = pat.strip_suffix("/**") {
            if rel == head || rel.starts_with(&format!("{head}/")) {
                return Ok(rel);
            }
        } else if rel == *pat {
            return Ok(rel);
        }
    }
    bail!(
        "wb_render refuses to open {rel:?} for subset {subset:?}: the allowlist is {pats:?}. Everything else in a WB trial directory is grader output written by the verifier stage of the same trial, and reading it is the leak this module replaces."
    )
}

fn read_bytes(trial: &Path, path: &Path, subset: &str) -> Result<Vec<u8>> {
    admit(trial, path, subset)?;
    Ok(std::fs::read(path)?)
}

const PROVENANCE_HEAD: &[&str] = &["verifier-owned", "verifier-extracted", "verifier-generated", "for the judge", "judge rubric", "judge modality", "deterministic verifier status"];
const PROVENANCE_HEAD_CHARS: usize = 4000;
const VERDICT_ANY: &[&str] = &["deterministic checks:", "pass_rate", "passed_count", "failed_checks", "test_pass_rate"];
const PROVENANCE_NAME_PREFIXES: &[&str] = &["judge_evidence"];

/// Reason string if this admitted file looks verifier-authored, else None.
pub fn is_verifier_authored(name: &str, text: Option<&str>, subset: &str) -> Option<String> {
    if subset == "code" || subset == "web" {
        return None;
    }
    let ln = name.to_lowercase();
    for pre in PROVENANCE_NAME_PREFIXES {
        if ln.starts_with(pre) {
            return Some(format!("filename begins with {pre:?}, the verifier's evidence naming convention"));
        }
    }
    let text = text?;
    let head = util::take_chars(text, PROVENANCE_HEAD_CHARS).to_lowercase();
    for p in PROVENANCE_HEAD {
        if head.contains(p) {
            return Some(format!("file head declares verifier provenance ({p:?})"));
        }
    }
    let low = text.to_lowercase();
    VERDICT_ANY.iter().find(|p| low.contains(*p)).map(|p| format!("contains a grader verdict marker ({p:?})"))
}

const TEXT_SUFFIXES: &[&str] = &[".md", ".txt", ".json", ".csv", ".tsv", ".yaml", ".yml", ".html", ".htm", ".log", ".py", ".js", ".sh", ".xml", ".ini", ".toml"];
const IMAGE_SUFFIXES: &[&str] = &[".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"];
const WEB_EXTRA: &[&str] = &[".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".css"];
const BINARY: &[&str] = &[".xlsx", ".pptx"];

fn suffix(name: &str) -> String {
    let base = name.rsplit('/').next().unwrap_or(name);
    match base.rfind('.') {
        Some(i) if i > 0 => base[i..].to_lowercase(),
        _ => String::new(),
    }
}

fn text(data: &[u8]) -> String {
    String::from_utf8_lossy(data).into_owned()
}

fn image_stub(size: usize) -> String {
    format!(
        "(image, {} bytes -- not inlined)\nImages are named and sized, never rendered into text: a base64 blob or an OCR guess in an evidence file is noise to a reader and invites reasoning about pixels nobody actually looked at.",
        crate::core::context::thousands(size)
    )
}

fn binary_stub(name: &str, size: usize) -> String {
    let s = suffix(name);
    format!("({} file, {} bytes -- binary, not rendered)", if s.is_empty() { "no-extension".into() } else { s }, crate::core::context::thousands(size))
}

/// The interpreter that renders xlsx/pptx (needs openpyxl and python-pptx).
pub fn render_python() -> PathBuf {
    config::env("HARNESS2_RENDER_PYTHON")
        .map(PathBuf::from)
        .or_else(|| util::which("python3"))
        .unwrap_or_else(|| PathBuf::from("python3"))
}

const RENDER_PY: &str = r###"import json, sys
from pathlib import Path
_XLSX_NO_CACHE = "(no cached value; a data_only reader sees this cell as EMPTY)"
def _xlsx_cell(raw, cached):
    if raw is None:
        return ""
    s = raw if isinstance(raw, str) else str(raw)
    if not isinstance(raw, str):
        if type(raw).__name__ in ("ArrayFormula", "DataTableFormula"):
            s = getattr(raw, "text", None) or str(raw)
        else:
            return s
    if not s.startswith("="):
        return s
    if cached is None or (isinstance(cached, str) and cached == ""):
        return f"formula:{s} -> {_XLSX_NO_CACHE}"
    return f"formula:{s} = {cached}"
def _xlsx_text(path, display_name):
    import openpyxl
    wb_f = openpyxl.load_workbook(path, data_only=False)
    wb_v = openpyxl.load_workbook(path, data_only=True)
    out = [f"# Workbook: {display_name}"]
    for idx, ws_f in enumerate(wb_f.worksheets, start=1):
        ws_v = wb_v[ws_f.title] if ws_f.title in wb_v.sheetnames else None
        out.append("")
        out.append(f"## Sheet {idx}: {ws_f.title}")
        out.append(f"Dimensions: {ws_f.max_row} rows x {ws_f.max_column} columns")
        rows_v = list(ws_v.iter_rows(values_only=True)) if ws_v is not None else []
        for r, row_f in enumerate(ws_f.iter_rows(values_only=True), start=1):
            row_v = rows_v[r - 1] if r - 1 < len(rows_v) else ()
            cells = [_xlsx_cell(raw, row_v[c] if c < len(row_v) else None) for c, raw in enumerate(row_f)]
            if not any(x.strip() for x in cells):
                continue
            out.append(f"R{r}: " + " | ".join(cells))
    return "\n".join(out)
def _shape_lines(shape):
    lines = []
    try:
        if getattr(shape, "shape_type", None) is not None and hasattr(shape, "shapes"):
            for sub in shape.shapes:
                lines += _shape_lines(sub)
            return lines
        if getattr(shape, "has_table", False):
            for row in shape.table.rows:
                lines.append(" | ".join(c.text for c in row.cells))
            return lines
        if getattr(shape, "has_text_frame", False):
            body = shape.text_frame.text
            if body.strip():
                lines.append(body)
    except Exception as exc:
        lines.append(f"(shape unreadable: {type(exc).__name__}: {exc})")
    return lines
def _pptx_text(path, display_name):
    from pptx import Presentation
    prs = Presentation(str(path))
    out = [f"# Deck: {display_name}", f"Slides: {len(prs.slides)}"]
    for n, slide in enumerate(prs.slides, start=1):
        out.append("")
        out.append(f"## slide {n}")
        wrote = False
        for shape in slide.shapes:
            for line in _shape_lines(shape):
                out.append(line)
                wrote = True
        if not wrote:
            out.append("(no text frames on this slide)")
    return "\n".join(out)
req = json.loads(sys.stdin.read())
texts = {}
for it in req["items"]:
    p = Path(it["path"])
    fn = _xlsx_text if p.suffix.lower() == ".xlsx" else _pptx_text
    texts[it["key"]] = fn(p, it["name"])
print(json.dumps({"texts": texts}))
"###;

/// {key -> text} for every xlsx/pptx in `items`, through ONE renderer call.
fn render_binaries(items: &[(String, PathBuf, String)]) -> Result<BTreeMap<String, String>> {
    if items.is_empty() {
        return Ok(BTreeMap::new());
    }
    let py = render_python();
    let payload = json!({"items": items.iter().map(|(k, p, n)| json!({"key": k, "path": p, "name": n})).collect::<Vec<_>>()});
    let mut c = std::process::Command::new(&py);
    c.arg("-I").arg("-c").arg(RENDER_PY);
    let o = util::run_with_timeout(c, Some(&payload.to_string()), Duration::from_secs(1800))
        .map_err(|e| anyhow!("RenderBackendError: the renderer interpreter {} could not run ({e}). Install openpyxl and python-pptx for it (or set HARNESS2_RENDER_PYTHON). Refusing to degrade a workbook to '(binary, not rendered)': that is indistinguishable from a rollout that produced nothing.", py.display()))?;
    if o.code != Some(0) {
        bail!("RenderBackendError: wb_render renderer ({}) failed: {}", py.display(), util::tail_chars(o.stderr.trim(), 1200));
    }
    let v: Value = serde_json::from_str(&o.stdout).map_err(|_| anyhow!("RenderBackendError: renderer returned non-JSON: {:?}", util::take_chars(&o.stdout, 400)))?;
    Ok(v["texts"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect()).unwrap_or_default())
}

/// (name, byte size, rendered text, was-scannable-as-text), in stable name order.
type Entry = (String, usize, String, bool);

fn entries(trial: &Path, subset: &str) -> Result<Vec<Entry>> {
    match subset {
        "office" => entries_dir(trial, subset, &trial.join("verifier/raw_artifacts")),
        "sec" => entries_dir(trial, subset, &trial.join("artifacts/logs/artifacts")),
        "web" => entries_tar(trial, subset, &trial.join("verifier/agent.workspace.tar.gz")),
        "code" => entries_patch(trial, subset, &trial.join("verifier/agent.patch")),
        s => bail!("wb_render has no handler for subset {s:?}"),
    }
}

fn entries_dir(trial: &Path, subset: &str, root: &Path) -> Result<Vec<Entry>> {
    if !root.is_dir() {
        return Ok(vec![]);
    }
    admit(trial, root, subset)?;
    let (mut plain, mut bins, mut sizes) = (vec![], vec![], BTreeMap::new());
    for f in util::files_sorted(root) {
        let rel = util::rel(root, &f);
        let data = read_bytes(trial, &f, subset)?;
        sizes.insert(rel.clone(), data.len());
        let suf = suffix(&rel);
        if IMAGE_SUFFIXES.contains(&suf.as_str()) {
            plain.push((rel, data.len(), image_stub(data.len()), false));
        } else if TEXT_SUFFIXES.contains(&suf.as_str()) {
            plain.push((rel, data.len(), text(&data), true));
        } else if BINARY.contains(&suf.as_str()) {
            bins.push((rel, f.clone(), f.file_name().unwrap().to_string_lossy().to_string()));
        } else {
            plain.push((rel.clone(), data.len(), binary_stub(&rel, data.len()), false));
        }
    }
    let rendered = render_binaries(&bins)?;
    for (rel, _, _) in bins {
        let body = rendered.get(&rel).cloned().unwrap_or_default();
        plain.push((rel.clone(), sizes[&rel], body, true));
    }
    plain.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(plain)
}

fn tar_members(tb: &Path) -> Result<Vec<(String, u64, Option<Vec<u8>>)>> {
    let f = std::fs::File::open(tb)?;
    let mut ar = tar::Archive::new(flate2::read::GzDecoder::new(f));
    let mut out = vec![];
    for e in ar.entries().map_err(|e| anyhow!("RenderBackendError: web workspace tarball unreadable: {}: {e}", tb.display()))? {
        let mut e = e.map_err(|e| anyhow!("RenderBackendError: web workspace tarball unreadable: {}: {e}", tb.display()))?;
        if !e.header().entry_type().is_file() {
            continue;
        }
        let name = e.path()?.to_string_lossy().to_string();
        let size = e.header().size()?;
        let suf = suffix(&name);
        let want = TEXT_SUFFIXES.contains(&suf.as_str()) || WEB_EXTRA.contains(&suf.as_str());
        let body = if want && !IMAGE_SUFFIXES.contains(&suf.as_str()) {
            let mut b = vec![];
            e.read_to_end(&mut b)?;
            Some(b)
        } else {
            None
        };
        out.push((name, size, body));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

fn entries_tar(trial: &Path, subset: &str, tb: &Path) -> Result<Vec<Entry>> {
    if !tb.is_file() {
        return Ok(vec![]);
    }
    admit(trial, tb, subset)?;
    Ok(tar_members(tb)?
        .into_iter()
        .map(|(name, size, body)| {
            let suf = suffix(&name);
            if IMAGE_SUFFIXES.contains(&suf.as_str()) {
                (name, size as usize, image_stub(size as usize), false)
            } else if let Some(b) = body {
                (name, size as usize, text(&b), true)
            } else {
                let stub = binary_stub(&name, size as usize);
                (name, size as usize, stub, false)
            }
        })
        .collect())
}

fn touched(body: &str) -> Vec<String> {
    body.lines().filter(|l| l.starts_with("diff --git ") && l.contains(" b/")).map(|l| l.split_once(" b/").unwrap().1.to_string()).collect()
}

fn entries_patch(trial: &Path, subset: &str, patch: &Path) -> Result<Vec<Entry>> {
    if !patch.is_file() {
        return Ok(vec![]);
    }
    let body = text(&read_bytes(trial, patch, subset)?);
    let t = touched(&body);
    let head = if t.is_empty() { String::new() } else { format!("## Files touched\n{}\n\n", t.iter().map(|x| format!("- {x}")).collect::<Vec<_>>().join("\n")) };
    Ok(vec![("agent.patch".into(), body.len(), format!("{head}## Diff\n{body}"), true)])
}

fn kept(subset: &str, es: Vec<Entry>) -> Vec<Entry> {
    es.into_iter()
        .filter(|(n, _, b, scanned)| {
            let base = n.rsplit('/').next().unwrap_or(n);
            is_verifier_authored(base, if *scanned { Some(b) } else { None }, subset).is_none()
        })
        .collect()
}

/// {filename -> rendered text} for one rollout. Agent-authored bytes only.
pub fn render(trial: &Path, subset: &str) -> Result<BTreeMap<String, String>> {
    Ok(render_report(trial, subset)?.0)
}

/// `render` plus the quarantine log.
pub fn render_report(trial: &Path, subset: &str) -> Result<(BTreeMap<String, String>, Vec<String>)> {
    let mut out = BTreeMap::new();
    let mut withheld = vec![];
    for (name, _s, body, scanned) in entries(trial, subset)? {
        let base = name.rsplit('/').next().unwrap_or(&name).to_string();
        if let Some(r) = is_verifier_authored(&base, if scanned { Some(&body) } else { None }, subset) {
            withheld.push(format!("{name}: {r}"));
            continue;
        }
        out.insert(name, body);
    }
    Ok((out, withheld))
}

/// The same rendering as one string, untrimmed.
pub fn flat_text(trial: &Path, subset: &str) -> Result<String> {
    let es = kept(subset, entries(trial, subset)?);
    if es.is_empty() {
        return Ok(String::new());
    }
    let mut parts = vec![format!(
        "## Files produced\n{}",
        es.iter().map(|(n, s, _, _)| format!("- {n} ({} bytes)", crate::core::context::thousands(*s))).collect::<Vec<_>>().join("\n")
    )];
    parts.extend(es.iter().map(|(n, _, b, _)| format!("## {n}\n{b}")));
    Ok(parts.join("\n\n"))
}

/// Rendered artifact BODY characters, excluding flat_text's index.
pub fn body_chars(trial: &Path, subset: &str) -> usize {
    entries(trial, subset).map(|es| kept(subset, es).iter().map(|(_, _, b, _)| util::char_len(b.trim())).sum()).unwrap_or(0)
}

/// Quarantine-passing RAW binary artifacts: [(display name, path)], office and sec only.
pub fn raw_files(trial: &Path, subset: &str) -> Result<Vec<(String, PathBuf)>> {
    let root = match subset {
        "office" => trial.join("verifier/raw_artifacts"),
        "sec" => trial.join("artifacts/logs/artifacts"),
        _ => return Ok(vec![]),
    };
    if !root.is_dir() {
        return Ok(vec![]);
    }
    admit(trial, &root, subset)?;
    let withheld: HashSet<String> = entries(trial, subset)?
        .into_iter()
        .filter(|(n, _, b, sc)| is_verifier_authored(n.rsplit('/').next().unwrap_or(n), if *sc { Some(b) } else { None }, subset).is_some())
        .map(|(n, ..)| n)
        .collect();
    let mut out = vec![];
    for f in util::files_sorted(&root) {
        let rel = util::rel(&root, &f);
        if !BINARY.contains(&suffix(&rel).as_str()) {
            continue;
        }
        admit(trial, &f, subset)?;
        if withheld.contains(&rel) || is_verifier_authored(rel.rsplit('/').next().unwrap_or(&rel), None, subset).is_some() {
            continue;
        }
        out.push((rel, f));
    }
    Ok(out)
}

/// The deliverable NAMES only: the PROPOSER's view.
pub fn listing(trial: &Path, subset: &str) -> Result<Vec<String>> {
    match subset {
        "office" | "code" => {
            let patch = trial.join("verifier/agent.patch");
            if !patch.is_file() {
                return Ok(vec![]);
            }
            let body = text(&read_bytes(trial, &patch, subset)?);
            let mut names: Vec<String> = touched(&body).into_iter().map(|s| s.trim().to_string()).collect();
            names.sort();
            names.dedup();
            Ok(names)
        }
        "web" => {
            let tb = trial.join("verifier/agent.workspace.tar.gz");
            if !tb.is_file() {
                return Ok(vec![]);
            }
            admit(trial, &tb, subset)?;
            let mut v: Vec<String> = tar_members(&tb)?.into_iter().map(|(n, s, _)| format!("{n} ({} bytes)", crate::core::context::thousands(s as usize))).collect();
            v.sort();
            Ok(v)
        }
        "sec" => {
            let root = trial.join("artifacts/logs/artifacts");
            if !root.is_dir() {
                return Ok(vec![]);
            }
            admit(trial, &root, subset)?;
            let mut v: Vec<String> = util::files_sorted(&root)
                .iter()
                .map(|f| format!("{} ({} bytes)", util::rel(&root, f), crate::core::context::thousands(std::fs::metadata(f).map(|m| m.len() as usize).unwrap_or(0))))
                .collect();
            v.sort();
            Ok(v)
        }
        s => bail!("wb_render has no listing for subset {s:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allowlist_and_quarantine() {
        let d = tempfile::tempdir().unwrap();
        let t = d.path();
        util::write(&t.join("verifier/raw_artifacts/report.md"), "hello world").unwrap();
        util::write(&t.join("verifier/raw_artifacts/judge_evidence.md"), "x").unwrap();
        util::write(&t.join("verifier/score.json"), "{}").unwrap();
        assert!(admit(t, &t.join("verifier/score.json"), "office").is_err());
        let (files, held) = render_report(t, "office").unwrap();
        assert_eq!(files.keys().collect::<Vec<_>>(), vec!["report.md"]);
        assert_eq!(held.len(), 1);
        assert!(flat_text(t, "office").unwrap().contains("## Files produced\n- report.md (11 bytes)"));
    }
    #[test]
    fn patch_listing() {
        let d = tempfile::tempdir().unwrap();
        util::write(&d.path().join("verifier/agent.patch"), "diff --git a/x.py b/x.py\n+1\n").unwrap();
        assert_eq!(listing(d.path(), "code").unwrap(), vec!["x.py"]);
    }
}
