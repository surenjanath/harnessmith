//! The goal assistant: turn a rough request ("a research document on dogs") into a detailed
//! brief the solver can act on, with grading criteria and a domain. It uses the project's
//! improver model (the first reachable one in its chain); with no model it falls back to a
//! structured template so the button always helps.

use crate::llm::client::{self, ChatReq, Msg};
use anyhow::Result;
use serde_json::{json, Value};
use std::time::Duration;

const SYSTEM: &str = r#"You turn a person's rough request into a clear, complete brief for an autonomous agent that will do the work in a sandboxed workspace (it can read input files, run Python, and write files; it may have no internet access unless tools provide it).

Write the brief the way an expert in the field would scope it:
- Objective: one or two sentences on what success looks like and who it is for.
- Scope: what to cover, broken into concrete sections or sub-questions; include angles, places and perspectives the person probably wants but did not mention (history, regions, variants, risks, costs, comparisons, recent developments...).
- Sources: where the information should come from (the input files first, then named kinds of sources: standards bodies, official statistics, peer-reviewed reviews, manufacturer docs...), and how to cite them; say what to do when something cannot be verified.
- Deliverables: exact files under output/ (e.g. output/report.md, output/summary.md, output/data.csv), their structure, length and format.
- Quality bar: accuracy rules, what to avoid, how to check the work before finishing.

Keep the person's intent; do not invent facts about their situation. If something important is ambiguous, pick a sensible default and list the question.

Reply with ONLY a JSON object:
{"goal": "<the full brief, markdown, 150-450 words>", "rubric": ["<5-8 checkable grading criteria>"], "domain": "<one or two lowercase words>", "deliverables": ["output/..."], "questions": ["<up to 3 questions worth asking the person>"], "title": "<a short title>"}"#;

/// Expand a rough goal. Returns {goal, rubric, domain, deliverables, questions, title, model}.
pub fn expand(model_chain: &str, text: &str, has_files: &[String]) -> Result<Value> {
    let text = text.trim();
    if text.is_empty() {
        anyhow::bail!("write a few words about what you want first");
    }
    let mut user = format!("The person wrote:\n\n{text}");
    if !has_files.is_empty() {
        user.push_str(&format!("\n\nThey are attaching these input files: {}", has_files.join(", ")));
    }
    let mut last_err = String::new();
    for id in model_chain.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let Ok((conn, model)) = crate::llm::connections::resolve(id) else { continue };
        if conn.provider == "mock" {
            continue;
        }
        let msgs = vec![Msg::User(user.clone())];
        match client::chat(&conn, &model, &ChatReq { system: SYSTEM, messages: &msgs, tools: &[], max_tokens: 2500, temperature: None }, Duration::from_secs(300)) {
            Ok(r) => {
                let body = r.text.trim();
                let body = body.split_once("```json").map(|(_, b)| b.split("```").next().unwrap_or(b)).or_else(|| body.split_once("```").map(|(_, b)| b.split("```").next().unwrap_or(b))).unwrap_or(body);
                let start = body.find('{').unwrap_or(0);
                let end = body.rfind('}').map(|e| e + 1).unwrap_or(body.len());
                if let Some(mut v) = client::lenient_json(&body[start..end]).filter(|v| v["goal"].as_str().map(|g| g.len() > 40).unwrap_or(false)) {
                    v["model"] = json!(id);
                    return Ok(v);
                }
                last_err = format!("{id} did not return the expected JSON");
            }
            Err(e) => last_err = format!("{id}: {e}"),
        }
    }
    let mut v = template(text, has_files);
    v["model"] = json!("template");
    if !last_err.is_empty() {
        v["note"] = json!(format!("No model could expand it ({}); this is the built-in template, edit it to taste.", crate::util::take_chars(&last_err, 200)));
    }
    Ok(v)
}

/// Offline fallback: a structured brief built from the request itself.
pub fn template(text: &str, files: &[String]) -> Value {
    let lower = text.to_lowercase();
    let research = ["research", "report", "document", "study", "review", "analysis", "paper", "brief"].iter().any(|w| lower.contains(w));
    let data = ["csv", "excel", "spreadsheet", "data", "table", "chart"].iter().any(|w| lower.contains(w)) || files.iter().any(|f| f.ends_with(".csv") || f.ends_with(".xlsx"));
    // "i want a research document on dogs" -> "A research document on dogs"
    let lead = regex::Regex::new(r"(?i)^\s*(please\s+)?(i\s+(want|need|would\s+like)(\s+you)?(\s+to)?|can\s+you|could\s+you|make\s+me|give\s+me)\s+").unwrap();
    let t = lead.replace(text.trim(), "").trim_end_matches(['.', '!', '?']).to_string();
    let mut c = t.chars();
    let topic = match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => text.to_string(),
    };
    let main = if research { "output/report.md" } else if data { "output/analysis.md" } else { "output/answer.md" };
    let mut deliverables = vec![main.to_string(), "output/summary.md".to_string()];
    if data {
        deliverables.push("output/data.csv".into());
    }
    let inputs = if files.is_empty() { String::new() } else { format!("\n\nStart from the input files ({}) and treat them as the primary source.", files.join(", ")) };
    let goal = format!(
        "## Objective\n{topic}. Produce a well-organised, accurate deliverable a non-expert can use, with the reasoning and sources visible.{inputs}\n\n\
## Scope\n- Background and key definitions\n- The main sub-topics, each with its own section\n- Different perspectives, regions or variants where they matter\n- Practical implications: costs, risks, trade-offs\n- Open questions and what is uncertain\n\n\
## Sources\nUse the input files first. Prefer primary and authoritative sources (official statistics, standards bodies, peer-reviewed reviews). Cite each non-obvious claim inline. If something cannot be verified, say so instead of guessing.\n\n\
## Deliverables\n- `{main}`: the full {}, with headings, a short introduction and a conclusion\n- `output/summary.md`: a one-page summary with the key findings as bullets{}\n\n\
## Quality bar\nNo invented facts or figures. Check every number against its source. Read the files back before finishing.",
        if research { "report" } else { "answer" },
        if data { "\n- `output/data.csv`: the cleaned table behind any figures" } else { "" },
    );
    json!({
        "title": crate::util::take_chars(&topic, 60),
        "goal": goal,
        "rubric": [
            "Covers every section listed under Scope",
            "Every non-obvious claim has a source or is marked as uncertain",
            "No invented figures; numbers match their sources",
            format!("{main} exists, is well structured with headings, and reads clearly"),
            "output/summary.md gives the key findings on one page",
            "States limitations and open questions",
        ],
        "domain": if research { "research" } else if data { "data" } else { "general" },
        "deliverables": deliverables,
        "questions": ["Who is the audience, and how long should it be?", "Are there regions, time periods or sources to focus on or avoid?"],
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn template_expands_a_research_request() {
        let v = super::template("I want a research document on dogs", &[]);
        let g = v["goal"].as_str().unwrap();
        assert!(g.contains("## Scope") && g.contains("output/report.md"));
        assert_eq!(v["domain"], "research");
        assert!(v["rubric"].as_array().unwrap().len() >= 5);
    }
}
