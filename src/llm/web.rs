//! Opt-in web access for the solver (project setting `web_access`).
//!
//! `web_fetch` reads a page as text. `web_search` needs a search API key in the environment:
//! `BRAVE_API_KEY` (Brave Search) or `TAVILY_API_KEY` (Tavily). Without one, only fetch exists.

use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::time::Duration;

const UA: &str = "harnessmith/0.1 (+https://github.com/surenjanath/harnessmith)";

/// Which search backend is configured, if any.
pub fn search_backend() -> Option<&'static str> {
    if std::env::var("BRAVE_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
        Some("brave")
    } else if std::env::var("TAVILY_API_KEY").map(|v| !v.is_empty()).unwrap_or(false) {
        Some("tavily")
    } else {
        None
    }
}

/// Visible text of an HTML page, whitespace collapsed.
pub fn html_text(html: &str) -> String {
    let re_drop = regex::Regex::new(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</(script|style|noscript|svg|nav|footer|header)>").unwrap();
    let s = re_drop.replace_all(html, " ");
    let s = regex::Regex::new(r"(?i)<(br|/p|/div|/h[1-6]|/li|/tr)[^>]*>").unwrap().replace_all(&s, "\n");
    let s = regex::Regex::new(r"<[^>]+>").unwrap().replace_all(&s, " ");
    let s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&nbsp;", " ");
    let s = regex::Regex::new(r"[ \t]+").unwrap().replace_all(&s, " ");
    regex::Regex::new(r"\n\s*\n+").unwrap().replace_all(&s, "\n\n").trim().to_string()
}

pub fn fetch(url: &str) -> Result<String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        bail!("web_fetch needs an http(s) URL");
    }
    let r = ureq::get(url).set("User-Agent", UA).timeout(Duration::from_secs(30)).call().map_err(|e| anyhow::anyhow!("fetch failed: {e}"))?;
    let ct = r.header("content-type").unwrap_or("").to_lowercase();
    let mut body = String::new();
    std::io::Read::read_to_string(&mut std::io::Read::take(r.into_reader(), 3_000_000), &mut body).ok();
    let text = if ct.contains("html") || body.trim_start().starts_with('<') { html_text(&body) } else { body };
    Ok(format!("{url}\n\n{}", crate::util::take_chars(&text, 20_000)))
}

pub fn search(q: &str, n: usize) -> Result<String> {
    let n = n.clamp(1, 10);
    let rows: Vec<(String, String, String)> = match search_backend() {
        Some("brave") => {
            let key = std::env::var("BRAVE_API_KEY").unwrap_or_default();
            let v: Value = ureq::get("https://api.search.brave.com/res/v1/web/search")
                .query("q", q)
                .query("count", &n.to_string())
                .set("X-Subscription-Token", &key)
                .set("Accept", "application/json")
                .timeout(Duration::from_secs(30))
                .call()?
                .into_json()?;
            v["web"]["results"].as_array().cloned().unwrap_or_default().iter().map(|r| (r["title"].as_str().unwrap_or("").into(), r["url"].as_str().unwrap_or("").into(), html_text(r["description"].as_str().unwrap_or("")))).collect()
        }
        Some(_) => {
            let key = std::env::var("TAVILY_API_KEY").unwrap_or_default();
            let v: Value = ureq::post("https://api.tavily.com/search").timeout(Duration::from_secs(30)).send_json(json!({"api_key": key, "query": q, "max_results": n}))?.into_json()?;
            v["results"].as_array().cloned().unwrap_or_default().iter().map(|r| (r["title"].as_str().unwrap_or("").into(), r["url"].as_str().unwrap_or("").into(), crate::util::take_chars(r["content"].as_str().unwrap_or(""), 300))).collect()
        }
        None => bail!("web search is not configured (set BRAVE_API_KEY or TAVILY_API_KEY); use web_fetch on a known URL"),
    };
    if rows.is_empty() {
        return Ok("No results.".into());
    }
    Ok(rows.iter().enumerate().map(|(i, (t, u, d))| format!("{}. {t}\n   {u}\n   {d}", i + 1)).collect::<Vec<_>>().join("\n"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn strips_markup_and_scripts() {
        let t = super::html_text("<html><head><style>x{}</style><script>evil()</script></head><body><h1>Title</h1><p>A &amp; B</p></body></html>");
        assert!(t.contains("Title") && t.contains("A & B") && !t.contains("evil"), "{t}");
    }
}
