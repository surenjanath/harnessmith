//! Deliverable review: before an agent may finish, check what it wrote against what the goal
//! asked for. Deterministic checks only (no model call), so they are cheap and never wrong in
//! the same way the model is: a missing file type, a cut-off HTML page, an unstyled "sleek"
//! page, a "research document" of a few sentences, an empty file, broken JSON or CSV.

use regex::Regex;
use std::path::Path;

fn words(s: &str) -> usize {
    s.split_whitespace().filter(|w| w.chars().any(|c| c.is_alphanumeric())).count()
}

/// Problems with `out_dir` for this goal; empty when nothing obvious is wrong.
pub fn review(goal: &str, out_dir: &Path) -> Vec<String> {
    let g = goal.to_lowercase();
    let files: Vec<std::path::PathBuf> = crate::util::files_sorted(out_dir);
    let rel = |p: &Path| format!("output/{}", crate::util::rel(out_dir, p));
    let mut issues = vec![];
    if files.is_empty() {
        return vec!["output/ is empty: nothing has been delivered yet".into()];
    }
    let ext = |p: &Path| p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    // 1. The file types the goal names.
    for want in crate::core::schedule::expected_exts(goal) {
        if !files.iter().any(|p| ext(p) == want || (want == "html" && ext(p) == "htm")) {
            issues.push(format!("the goal asks for a .{want} file, but output/ has none (it has: {})", files.iter().map(|p| rel(p)).collect::<Vec<_>>().join(", ")));
        }
    }
    let styled_wanted = ["sleek", "modern", "beautiful", "stylish", "styled", "polished", "responsive", "elegant", "nice looking", "professional", "landing page", "design"].iter().any(|w| g.contains(w));
    let long_wanted = ["research", "report", "document", "essay", "article", "guide", "whitepaper", "analysis", "study", "overview", "brief"].iter().any(|w| g.contains(w));
    let has_css_file = files.iter().any(|p| ext(p) == "css");
    let mut text_words = 0;
    for p in &files {
        let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        if size == 0 {
            issues.push(format!("{} is empty", rel(p)));
            continue;
        }
        let body = std::fs::read_to_string(p).unwrap_or_default();
        match ext(p).as_str() {
            "html" | "htm" => {
                let b = body.to_lowercase();
                if !b.contains("</html>") || !b.contains("<body") {
                    issues.push(format!("{} looks cut off or incomplete: a full page needs <!DOCTYPE html>, <head>, <body> and a closing </html>. Rewrite the whole file with write_file", rel(p)));
                }
                if styled_wanted && !b.contains("<style") && !b.contains(".css") && !has_css_file && !b.contains("style=") {
                    issues.push(format!("{} has no styling, but the goal asks for a {} result: add CSS (layout, typography, colours, spacing)", rel(p), ["sleek", "modern", "beautiful", "stylish", "polished", "responsive", "elegant", "professional"].iter().find(|w| g.contains(*w)).unwrap_or(&"designed")));
                }
                if b.contains("</html>") && !b.trim_start().starts_with("<!doctype html") {
                    issues.push(format!("{} must start with <!DOCTYPE html>; put any <style> inside <head>, not before the doctype", rel(p)));
                }
                if b.contains("<script") && !b.contains("</script>") {
                    issues.push(format!("{} has an unclosed <script> tag", rel(p)));
                }
            }
            "json" => {
                if serde_json::from_str::<serde_json::Value>(&body).is_err() {
                    issues.push(format!("{} is not valid JSON", rel(p)));
                }
            }
            "csv" => {
                let counts: Vec<usize> = body.lines().filter(|l| !l.trim().is_empty()).take(200).map(|l| l.split(',').count()).collect();
                if counts.len() > 1 && counts.iter().any(|c| *c != counts[0]) && !body.contains('"') {
                    issues.push(format!("{} has rows with different numbers of columns", rel(p)));
                }
            }
            "md" | "markdown" => {
                text_words += words(&body);
                let tags = Regex::new(r"(?i)<(h[1-6]|p|ul|ol|li|div|br)\b").unwrap().find_iter(&body).count();
                if tags > 4 && !body.lines().any(|l| l.trim_start().starts_with('#')) {
                    issues.push(format!("{} is written with HTML tags (<h2>, <p>…); a Markdown file needs Markdown: '## ' headings, blank-line paragraphs, '- ' lists. Rewrite it with write_file", rel(p)));
                }
            }
            "txt" => text_words += words(&body),
            _ => {}
        }
        let t = body.trim_start();
        if t.starts_with("{\"status\":\"ERROR\"") || t.starts_with("ERROR:") || t.starts_with("Error:") {
            issues.push(format!("{} contains an error message instead of the deliverable", rel(p)));
        }
    }
    // 2. Depth: a "research document" of a few sentences is not one.
    let has_doc = files.iter().any(|p| matches!(ext(p).as_str(), "md" | "txt" | "markdown" | "html" | "docx" | "pdf"));
    let deep = ["research", "report", "whitepaper", "study", "analysis"].iter().any(|w| g.contains(w));
    let min_words = if deep { 600 } else { 350 };
    if long_wanted && has_doc && text_words > 0 && text_words < min_words && !files.iter().any(|p| matches!(ext(p).as_str(), "html" | "docx" | "pdf")) {
        issues.push(format!("the written deliverable has only {text_words} words; the goal asks for a {} — cover each part in real depth with headings, specifics and sources", ["research document", "report", "document", "essay", "article", "guide", "analysis", "study", "overview", "brief"].iter().find(|w| g.contains(*w)).unwrap_or(&"document")));
    }
    issues
}

/// One concrete next action for a model that keeps calling finish: small models act on a single
/// instruction far better than on a list of problems.
pub fn next_step(issues: &[String], out_dir: &Path) -> String {
    let main = crate::util::files_sorted(out_dir)
        .into_iter()
        .filter(|p| p.extension().map(|e| matches!(e.to_string_lossy().as_ref(), "md" | "txt" | "markdown")).unwrap_or(false))
        .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .map(|p| format!("output/{}", crate::util::rel(out_dir, &p)))
        .unwrap_or_else(|| "output/report.md".into());
    let first = issues.first().map(|s| s.as_str()).unwrap_or("");
    let step = if first.contains("words") {
        format!("NEXT STEP: call append_file on {main} to add ONE complete new section (a heading plus at least 150 words of specific content). Repeat append_file for each missing part. Calling finish again without new content will be refused.")
    } else if first.contains("cut off") || first.contains("no styling") || first.contains("DOCTYPE") || first.contains(".html") {
        "NEXT STEP: call write_file on output/index.html with the COMPLETE page: <!DOCTYPE html>, <html>, <head> with <meta charset> and an inline <style> block (layout, typography, colours, spacing), <body> with the content, and </html>. Then read it back.".to_string()
    } else if first.contains("output/ is empty") {
        "NEXT STEP: call write_file to create the deliverable under output/.".to_string()
    } else {
        "NEXT STEP: fix the first point above with write_file, read the file back, then fix the next one. Calling finish without changes will be refused.".to_string()
    };
    step
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catches_the_hello_world_failure_and_passes_a_good_page() {
        let d = tempfile::tempdir().unwrap();
        let goal = "create a hello world sleek html page";
        std::fs::write(d.path().join("hello_world.html"), "<html><head><title>Hello World</title></head><body><h1>Hello, world!</h1><caption>").unwrap();
        let i = review(goal, d.path());
        assert!(i.iter().any(|x| x.contains("cut off")), "{i:?}");
        assert!(i.iter().any(|x| x.contains("no styling")), "{i:?}");
        std::fs::write(d.path().join("hello_world.html"), "<style>h1{}</style><!DOCTYPE html><html><head></head><body><h1>Hello</h1></body></html>").unwrap();
        assert!(review(goal, d.path()).iter().any(|x| x.contains("must start with <!DOCTYPE html>")));
        std::fs::write(d.path().join("hello_world.html"), "<!DOCTYPE html><html><head><style>body{font-family:system-ui}</style></head><body><h1>Hello</h1></body></html>").unwrap();
        assert!(review(goal, d.path()).is_empty());
    }

    #[test]
    fn wrong_type_short_docs_and_errors() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("answer.md"), "Dogs are mammals. They bark.").unwrap();
        let i = review("i want a research document on dogs", d.path());
        assert!(i.iter().any(|x| x.contains("only 5 words")), "{i:?}");
        let i = review("make a landing page in html", d.path());
        assert!(i.iter().any(|x| x.contains(".html file")), "{i:?}");
        std::fs::write(d.path().join("answer.md"), format!("<h2>History</h2><p>{}</p><h2>Breeds</h2><p>x</p><ul><li>a</li></ul>", "word ".repeat(700))).unwrap();
        assert!(review("i want a research document on dogs", d.path()).iter().any(|x| x.contains("written with HTML tags")));
        std::fs::write(d.path().join("answer.md"), "{\"status\":\"ERROR\",\"data\":{}}").unwrap();
        assert!(review("anything", d.path()).iter().any(|x| x.contains("error message")));
    }
}
