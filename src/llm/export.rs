//! `export_pdf`: turn a Markdown or HTML deliverable into a real PDF with no extra installs.
//!
//! Headless Chrome/Chromium, when present, prints the rendered HTML (best quality). Otherwise a
//! small built-in writer lays the Markdown out as text: headings, paragraphs, lists and tables
//! as rows, Helvetica on A4, page breaks and page numbers.

use anyhow::{bail, Result};
use std::path::Path;
use std::time::Duration;

/// A browser that can print to PDF, if one is installed.
pub fn chrome() -> Option<String> {
    let mac = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Chromium.app/Contents/MacOS/Chromium", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"];
    for p in mac {
        if Path::new(p).exists() {
            return Some(p.into());
        }
    }
    for b in ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"] {
        if let Ok(o) = std::process::Command::new("bash").arg("-c").arg(format!("command -v {b}")).output() {
            if o.status.success() {
                return Some(String::from_utf8_lossy(&o.stdout).trim().to_string());
            }
        }
    }
    None
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn inline_html(s: &str) -> String {
    let s = esc(s);
    let s = regex::Regex::new(r"`([^`]+)`").unwrap().replace_all(&s, "<code>$1</code>");
    let s = regex::Regex::new(r"\*\*([^*]+)\*\*").unwrap().replace_all(&s, "<b>$1</b>");
    regex::Regex::new(r"(^|[^*])\*([^*\s][^*]*)\*").unwrap().replace_all(&s, "$1<i>$2</i>").to_string()
}

/// Markdown to a styled, printable HTML page.
pub fn md_to_html(md: &str, title: &str) -> String {
    let lines: Vec<&str> = md.lines().collect();
    let mut h = String::new();
    let mut i = 0;
    let is_item = |l: &str| regex::Regex::new(r"^\s*([-*+]|\d+[.)])\s+").unwrap().is_match(l);
    while i < lines.len() {
        let l = lines[i];
        let t = l.trim();
        if t.starts_with("```") {
            let mut b = vec![];
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with("```") {
                b.push(lines[i]);
                i += 1;
            }
            i += 1;
            h += &format!("<pre>{}</pre>", esc(&b.join("\n")));
            continue;
        }
        if let Some(c) = regex::Regex::new(r"^(#{1,6})\s+(.*)").unwrap().captures(t) {
            let n = c[1].len().min(4);
            h += &format!("<h{n}>{}</h{n}>", inline_html(&c[2]));
            i += 1;
            continue;
        }
        if regex::Regex::new(r"^(-{3,}|\*{3,}|_{3,})$").unwrap().is_match(t) {
            h += "<hr>";
            i += 1;
            continue;
        }
        if t.starts_with('|') {
            let mut rows = vec![];
            while i < lines.len() && lines[i].trim().starts_with('|') {
                rows.push(lines[i].trim());
                i += 1;
            }
            h += "<table>";
            for (j, r) in rows.iter().filter(|r| !regex::Regex::new(r"^\|[\s:|-]+\|?$").unwrap().is_match(r)).enumerate() {
                let tag = if j == 0 { "th" } else { "td" };
                h += "<tr>";
                for c in r.trim_matches('|').split('|') {
                    h += &format!("<{tag}>{}</{tag}>", inline_html(c.trim()));
                }
                h += "</tr>";
            }
            h += "</table>";
            continue;
        }
        if is_item(l) {
            let ordered = t.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false);
            let tag = if ordered { "ol" } else { "ul" };
            h += &format!("<{tag}>");
            let strip = regex::Regex::new(r"^\s*([-*+]|\d+[.)])\s+").unwrap();
            while i < lines.len() && is_item(lines[i]) {
                h += &format!("<li>{}</li>", inline_html(&strip.replace(lines[i], "")));
                i += 1;
            }
            h += &format!("</{tag}>");
            continue;
        }
        if t.is_empty() {
            i += 1;
            continue;
        }
        let mut p = vec![];
        while i < lines.len() && !lines[i].trim().is_empty() && !lines[i].trim().starts_with('#') && !lines[i].trim().starts_with('|') && !lines[i].trim().starts_with("```") && !is_item(lines[i]) {
            p.push(lines[i].trim());
            i += 1;
        }
        if p.is_empty() {
            p.push(t);
            i += 1;
        }
        h += &format!("<p>{}</p>", inline_html(&p.join(" ")));
    }
    format!(
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>{}</title><style>@page{{size:A4;margin:22mm 20mm}}body{{font:11pt/1.5 Georgia,serif;color:#111}}h1{{font:600 22pt Helvetica,Arial,sans-serif;margin:0 0 12pt}}h2{{font:600 15pt Helvetica,Arial,sans-serif;margin:18pt 0 6pt;border-bottom:1px solid #ccc}}h3,h4{{font:600 12pt Helvetica,Arial,sans-serif;margin:12pt 0 4pt}}table{{border-collapse:collapse;margin:8pt 0;width:100%}}td,th{{border:1px solid #bbb;padding:4pt 6pt;text-align:left;vertical-align:top;font-size:10pt}}th{{background:#f0f0f0}}pre{{background:#f5f5f5;padding:8pt;font-size:9pt;white-space:pre-wrap}}code{{font-size:9.5pt}}</style></head><body>{h}</body></html>",
        esc(title)
    )
}

/// Characters Helvetica's WinAnsi encoding has, everything else mapped to a close ASCII.
fn ansi(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' => "'".to_string(),
            '\u{201C}' | '\u{201D}' => "\"".to_string(),
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' => "-".to_string(),
            '\u{2014}' => "--".to_string(),
            '\u{2022}' => "*".to_string(),
            '\u{2026}' => "...".to_string(),
            '\u{00A0}' | '\u{202F}' | '\u{2009}' => " ".to_string(),
            '\u{2265}' => ">=".to_string(),
            '\u{2264}' => "<=".to_string(),
            '\u{2192}' => "->".to_string(),
            c if (c as u32) < 128 => c.to_string(),
            _ => "?".to_string(),
        })
        .collect::<String>()
        .replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
}

/// The built-in writer: Markdown laid out as text in a valid PDF.
pub fn md_to_pdf(md: &str) -> Vec<u8> {
    // (text, font size, bold, indent)
    let mut blocks: Vec<(String, f32, bool, f32)> = vec![];
    let strip = |s: &str| {
        let s = regex::Regex::new(r"\*\*|__|`").unwrap().replace_all(s, "");
        regex::Regex::new(r"\*([^*\n]+)\*").unwrap().replace_all(&s, "$1").to_string()
    };
    let mut para = String::new();
    let flush = |para: &mut String, blocks: &mut Vec<(String, f32, bool, f32)>| {
        if !para.trim().is_empty() {
            blocks.push((para.trim().to_string(), 11.0, false, 0.0));
        }
        para.clear();
    };
    for l in md.lines() {
        let t = l.trim();
        if t.is_empty() || regex::Regex::new(r"^(-{3,}|\*{3,})$").unwrap().is_match(t) || t.starts_with("```") {
            flush(&mut para, &mut blocks);
            continue;
        }
        if let Some(c) = regex::Regex::new(r"^(#{1,6})\s+(.*)").unwrap().captures(t) {
            flush(&mut para, &mut blocks);
            let size = match c[1].len() { 1 => 20.0, 2 => 15.0, _ => 12.5 };
            blocks.push((strip(&c[2]), size, true, 0.0));
            continue;
        }
        if t.starts_with('|') {
            flush(&mut para, &mut blocks);
            if regex::Regex::new(r"^\|[\s:|-]+\|?$").unwrap().is_match(t) {
                continue;
            }
            let cells: Vec<String> = t.trim_matches('|').split('|').map(|c| strip(c.trim())).collect();
            blocks.push((cells.join("  |  "), 10.0, false, 6.0));
            continue;
        }
        if let Some(c) = regex::Regex::new(r"^\s*([-*+]|\d+[.)])\s+(.*)").unwrap().captures(l) {
            flush(&mut para, &mut blocks);
            let bullet = if c[1].chars().next().map(|x| x.is_ascii_digit()).unwrap_or(false) { c[1].to_string() } else { "*".into() };
            blocks.push((format!("{bullet} {}", strip(&c[2])), 11.0, false, 14.0));
            continue;
        }
        if !para.is_empty() {
            para.push(' ');
        }
        para.push_str(&strip(t));
    }
    flush(&mut para, &mut blocks);

    let (w, h, margin) = (595.0f32, 842.0f32, 56.0f32);
    let mut pages: Vec<String> = vec![];
    let mut cur = String::new();
    let mut y = h - margin;
    let new_page = |pages: &mut Vec<String>, cur: &mut String, y: &mut f32| {
        pages.push(std::mem::take(cur));
        *y = h - margin;
    };
    for (text, size, bold, indent) in &blocks {
        let lead = size * 1.4;
        let space_before = if *bold { size * 0.9 } else { size * 0.35 };
        if *bold && y - space_before - lead * 2.0 < margin {
            new_page(&mut pages, &mut cur, &mut y);
        }
        y -= space_before;
        // Average Helvetica glyph ~0.5em (0.55 for bold).
        let per_line = ((w - 2.0 * margin - indent) / (size * if *bold { 0.55 } else { 0.5 })).max(10.0) as usize;
        let mut line = String::new();
        let mut lines = vec![];
        for word in text.split_whitespace() {
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > per_line {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        if !line.is_empty() {
            lines.push(line);
        }
        for ln in lines {
            if y - lead < margin {
                new_page(&mut pages, &mut cur, &mut y);
            }
            y -= lead;
            cur += &format!("BT /{} {size} Tf {} {y:.1} Td ({}) Tj ET\n", if *bold { "F2" } else { "F1" }, margin + indent, ansi(&ln));
        }
    }
    if !cur.is_empty() || pages.is_empty() {
        pages.push(cur);
    }
    let n = pages.len();
    // Objects: 1 catalog, 2 pages, 3 F1, 4 F2, then per page: page + content.
    let mut objs: Vec<String> = vec![
        "<< /Type /Catalog /Pages 2 0 R >>".into(),
        format!("<< /Type /Pages /Kids [{}] /Count {n} >>", (0..n).map(|i| format!("{} 0 R", 5 + i * 2)).collect::<Vec<_>>().join(" ")),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".into(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>".into(),
    ];
    for (i, body) in pages.iter().enumerate() {
        let content = format!("{body}BT /F1 9 Tf {:.1} 30 Td (Page {} of {n}) Tj ET\n", w / 2.0 - 22.0, i + 1);
        objs.push(format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {} 0 R >>", 6 + i * 2));
        objs.push(format!("<< /Length {} >>\nstream\n{content}endstream", content.len()));
    }
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = vec![];
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for off in offsets {
        out.extend(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).as_bytes());
    out
}

/// Write `dest` (a .pdf) from `src` (.md, .markdown, .html, .txt). Returns how it was made.
pub fn export(src: &Path, dest: &Path) -> Result<String> {
    let body = crate::util::read_lossy(src)?;
    let ext = src.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    if !matches!(ext.as_str(), "md" | "markdown" | "html" | "htm" | "txt") {
        bail!("export_pdf converts .md, .html or .txt files, not .{ext}");
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let title = body.lines().find(|l| l.trim_start().starts_with('#')).map(|l| l.trim_start_matches('#').trim().to_string()).unwrap_or_else(|| "Document".into());
    if let Some(bin) = chrome() {
        let html = if ext.starts_with("htm") { body.clone() } else { md_to_html(&body, &title) };
        let tmp = tempfile::Builder::new().suffix(".html").tempfile()?;
        std::fs::write(tmp.path(), html)?;
        let profile = tempfile::tempdir()?;
        let mut c = std::process::Command::new(&bin);
        c.args(["--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer"])
            .arg(format!("--user-data-dir={}", profile.path().display()))
            .arg(format!("--print-to-pdf={}", dest.display()))
            .arg(format!("file://{}", tmp.path().display()));
        let r = crate::util::run_with_timeout(c, None, Duration::from_secs(90));
        if r.is_ok() && std::fs::read(dest).map(|b| b.starts_with(b"%PDF")).unwrap_or(false) {
            return Ok(format!("rendered with headless {}", Path::new(&bin).file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default()));
        }
    }
    let text = if ext.starts_with("htm") {
        let no_tags = regex::Regex::new(r"(?s)<(script|style)[^>]*>.*?</(script|style)>|<[^>]+>").unwrap().replace_all(&body, "\n");
        no_tags.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    } else {
        body
    };
    std::fs::write(dest, md_to_pdf(&text))?;
    Ok("laid out with the built-in writer (plain styling; install Chrome or Chromium for full rendering)".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_writer_makes_a_valid_multi_page_pdf() {
        let mut md = String::from("# Title\n\nIntro paragraph with “quotes” and — dashes.\n\n## Section\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n");
        for i in 0..200 {
            md += &format!("\nParagraph {i} with enough words to wrap across the line at least once in the layout.\n");
        }
        let pdf = md_to_pdf(&md);
        let s = String::from_utf8_lossy(&pdf);
        assert!(pdf.starts_with(b"%PDF-1.4"));
        assert!(s.contains("%%EOF"));
        assert!(s.contains("/Count ") && !s.contains("/Count 1 "), "should span several pages");
        assert!(s.contains("(Title) Tj"));
    }

    #[test]
    fn markdown_becomes_html() {
        let h = md_to_html("# T\n\n**bold** text\n\n- a\n- b\n\n| x | y |\n|---|---|\n| 1 | 2 |", "T");
        assert!(h.contains("<h1>T</h1>") && h.contains("<b>bold</b>") && h.contains("<li>a</li>") && h.contains("<td>1</td>"));
    }
}
