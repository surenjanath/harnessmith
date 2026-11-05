import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { execFileSync } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const script = JSON.parse(readFileSync(join(ROOT, "script.json"), "utf-8"));
const SCREENS = join(ROOT, "screenshots");

const B = script.brand;
// "mobile" -> narrow phone-frame media, side-by-side with copy.
// "web" -> wide browser-chrome-frame media, stacked above the copy.
const PLATFORM = script.platform === "web" ? "web" : "mobile";

// CUSTOMIZE: for each scene id in script.json, optionally provide a richer
// written kicker/body/bullets for the PDF (beyond the spoken narration).
// Any scene id left out just falls back to scene.narration as the body with
// no bullet list — so this can start as {} and be filled in per app.
const DETAILS = {
  overview: { kicker: "One gate, one queue, a recursion per goal",
    body: "The Overview draws the whole system and updates live while goals run. Each goal is checked at a deterministic gate, waits in a queue, and gets its own run slot with its own workspace and copy of the harness.",
    bullets: ["Solver attempts are probed; an improver proposes edits on two dimensions (how the agent works, what a practitioner knows) and composes them", "A blind selector picks the best output without knowing which harness made it", "A lesson is written to the library for later goals in the same domain", "The live bar shows what is running, on which model, and any model that is down"] },
  "run-goal": { kicker: "From a sentence to a queued run",
    body: "Describe the outcome you want, or start from a template. Grading criteria are optional but unlock scoring, A/B tests and the Results tab.",
    bullets: ["Templates: research report, landing page, data brief, PDF document, small script", "The estimate splits runs, time, tokens and cost by role before you start", "A warning appears when the improver is a small local model, the usual weak spot", "Cmd/Ctrl+Enter starts the run; a browser notification says when it finishes"] },
  models: { kicker: "Pick models that fit your machine",
    body: "The advisor reads your CPU, memory and GPU, lists every local model that fits, and measures real speed with Benchmark.",
    bullets: ["Recommended solver, improver, selector and embedding model", "One-click setups: All local, Local solver + API improver, Local solver + NVIDIA improver, All NVIDIA", "Each role can list fallbacks after a comma, so a failing model hands over automatically"] },
  connections: { kicker: "Keys stay in your .env file",
    body: "A connection is a named endpoint such as nvidia, ollama or openrouter. Model ids are written connection/model.",
    bullets: ["Type NAME=key (or just the key) in API key: it is written to the private, git-ignored .env file, never to connections.json", "After saving, the form only says whether the key is set; no part of the key is shown", "Save & test checks that the model really answers; Find free NVIDIA models shows which hosted models your account can use", "A model your account cannot use is skipped for the next one in the chain without blocking the connection"] },
  history: { kicker: "Every goal, searchable",
    body: "History groups goals by day. Each entry shows status, rubric score, run time and age; arrow keys move between goals and each goal has its own link.",
    bullets: ["Search, status filter (running, needs review, done, failed, cancelled) and domain filter", "A summary card: last run, score against the base attempt, time, number of runs", "The full goal and its criteria fold away; every run of the goal is listed"] },
  journey: { kicker: "How a run moved through the pipeline",
    body: "The journey map redraws the run: base runs, the INV and DOM proposals with their probes, the composed harness, and the blind selection.",
    bullets: ["Each box shows its role, what it did, steps and time", "The adopted path is drawn as a striped trunk", "Self-heals (failovers, loop stops, rescues) are badged on the box where they happened", "Click a box for its timeline, filtered to errors and self-heals if you like"] },
  output: { kicker: "Honest results",
    body: "Markdown deliverables are rendered, with a source view. The deliverable review runs on every output.",
    bullets: ["Missing file types, fake PDFs, placeholders such as [Your Name] and citations without references are flagged", "A run whose output fails review, or scores under 50% against the rubric, ends as needs review instead of done"] },
  compare: { kicker: "What adaptation changed",
    body: "Compare shows the first base attempt beside the adopted output with changed lines highlighted.",
    bullets: ["A/B test: base harness vs adopted harness, three times each, graded, with mean, spread and a verdict", "Model shootout: the same goal on several solver models, side by side", "Report (.md) downloads a shareable summary; Delete moves a goal to the project's trash"] },
  results: { kicker: "Is adapting the harness helping?",
    body: "For goals with criteria and a judge model, the adopted output and the unmodified base attempt are both graded after each run, three gradings each with a majority vote per criterion.",
    bullets: ["Adapted wins, ties and losses, and the average lift in points", "How often the regression guard kept the base attempt", "Spend and time per goal"] },
  library: { kicker: "Lessons that carry over",
    body: "After each run a reflection writes a lesson: the cue, the lesson, the evidence and a confidence. Later goals in the same domain read the library before proposing edits.",
    bullets: ["Pin a lesson you trust: it is listed first for later improvers", "Retire one that no longer helps: it is moved aside, not deleted"] },
  tools: { kicker: "What the agent can call",
    body: "The base harness is where every goal in a project starts: system prompt, guardrails, memory, tool guidance, skills, scripts and wired tools.",
    bullets: ["export_pdf turns Markdown or HTML into a real PDF (headless Chrome when installed, a built-in writer otherwise)", "web_fetch and web_search when web access is on for the project", "Wired tools are JSON files under tools/ that run a command; the improver can add them too"] },
  doctor: { kicker: "Checks and safe fixes",
    body: "The Doctor leads with a summary and lists only what needs attention, each with a next step. The rest is grouped and folded.",
    bullets: ["Models & keys: every role's model reachable, the model exists, breaker state, API key present", "Runs: failed or needs-review runs in the last 24 hours, stuck or interrupted runs", "One-click fixes: reset a breaker, resume runs, restore a base harness, make .env private", "Safe fixes also run at startup and every 10 minutes"] },
};

// CUSTOMIZE: path to the target app's real logo file (PNG/SVG with a
// transparent or light background — it renders inside a white pill badge,
// see .mark CSS below). Copy the app's actual logo asset into
// walkthrough/assets/logo.png and reference it here, e.g.:
//   const LOGO_PATH = join(ROOT, "assets", "logo.png");
const LOGO_PATH = join(ROOT, "assets", "logo.png");
const fmtDate = new Date().toLocaleDateString("en-US", {
  year: "numeric",
  month: "long",
  day: "numeric",
});

// A scene's "redact" rects ([x, y, w, h] as fractions of the screenshot) are blurred in a copy of the
// image, so personal data hidden in the video is hidden in the PDF too.
function redacted(scene) {
  const src = join(SCREENS, scene.image);
  if (!scene.redact || !scene.redact.length) return src;
  const buf = readFileSync(src), W = buf.readUInt32BE(16), H = buf.readUInt32BE(20);
  const dir = join(SCREENS, ".redacted");
  mkdirSync(dir, { recursive: true });
  const out = join(dir, scene.image);
  let chain = "", last = "0:v";
  scene.redact.forEach(([x, y, w, h], k) => {
    const [px, py, pw, ph] = [x * W, y * H, w * W, h * H].map(Math.round);
    chain += `[0:v]crop=${pw}:${ph}:${px}:${py},gblur=sigma=14[b${k}];[${last}][b${k}]overlay=${px}:${py}[o${k}];`;
    last = `o${k}`;
  });
  execFileSync("ffmpeg", ["-loglevel", "error", "-y", "-i", src, "-filter_complex", chain.slice(0, -1), "-map", `[${last}]`, out]);
  return out;
}

const sectionsHtml = script.scenes
  .map((scene, i) => {
    const d = DETAILS[scene.id] || {};
    const imgPath = scene.image ? redacted(scene) : null;
    const num = String(i + 1).padStart(2, "0");
    const media =
      PLATFORM === "web"
        ? imgPath
          ? `<div class="browser-frame"><div class="browser-bar"><span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>${scene.url ? `<span class="browser-url">${scene.url}</span>` : ""}</div><img src="file://${imgPath}" /></div>`
          : `<div class="browser-frame placeholder"></div>`
        : imgPath
          ? `<div class="phone-frame"><img src="file://${imgPath}" /></div>`
          : `<div class="phone-frame placeholder"></div>`;
    return `
  <div class="page section-page">
    <section class="scene ${PLATFORM === "web" ? "web" : i % 2 === 1 ? "reverse" : ""}">
      <div class="scene-media">
        ${media}
      </div>
      <div class="scene-copy">
        <div class="scene-num">${num} / ${String(script.scenes.length).padStart(2, "0")}</div>
        <div class="kicker">${d.kicker || scene.eyebrow}</div>
        <h2>${scene.title}</h2>
        <p class="lead">${d.body || scene.narration}</p>
        ${
          d.bullets
            ? `<ul>${d.bullets.map((b) => `<li>${b}</li>`).join("")}</ul>`
            : ""
        }
      </div>
    </section>
    <div class="footer-strip"><span>${script.appName}</span><span>${num} / ${String(script.scenes.length).padStart(2, "0")} — ${scene.title}</span></div>
  </div>`;
  })
  .join("\n");

const tocHtml = script.scenes
  .map((scene, i) => {
    const num = String(i + 1).padStart(2, "0");
    const d = DETAILS[scene.id] || {};
    return `<div class="toc-row"><span class="toc-num">${num}</span><span class="toc-title">${scene.title}</span><span class="toc-kicker">${d.kicker || scene.eyebrow}</span></div>`;
  })
  .join("\n");

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>${script.appName} — App Walkthrough</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --primary: ${B.primary};
    --primary-light: ${B.primaryLight};
    --primary-dark: ${B.primaryDark};
    --gold: ${B.gold};
    --bg: ${B.bg};
    --surface: ${B.surface};
    --text: ${B.textPrimary};
    --text-secondary: ${B.textSecondary};
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: 'Manrope', sans-serif;
    color: var(--text);
    background: var(--bg);
    font-size: 12.5px;
  }
  h1, h2, h3 { font-family: 'Outfit', sans-serif; margin: 0; }
  @page { size: A4; margin: 0; }

  .page { width: 210mm; height: 297mm; position: relative; page-break-after: always; page-break-inside: avoid; overflow: hidden; }
  .page:last-child { page-break-after: auto; }

  /* Cover */
  .cover {
    background: radial-gradient(circle at 50% 15%, var(--primary-light) 0%, var(--primary) 55%, var(--primary-dark) 100%);
    color: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40mm 20mm;
  }
  .mark {
    background: #fff; border-radius: 999px;
    padding: 7mm 14mm; margin-bottom: 14mm;
    box-shadow: 0 6px 18px -6px rgba(0,0,0,0.35);
    display: inline-flex;
  }
  .mark img { height: 12mm; display: block; }
  .mark.small { padding: 5mm 10mm; margin-bottom: 8mm; }
  .mark.small img { height: 8mm; }
  .cover .eyebrow {
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(255,255,255,.8);
    font-size: 12px;
    margin-bottom: 8mm;
  }
  .cover h1 {
    font-weight: 300;
    font-size: 46px;
    letter-spacing: -1px;
    line-height: 1.15;
    margin-bottom: 6mm;
  }
  .cover .tagline {
    font-size: 16px;
    color: rgba(255,255,255,0.85);
    font-weight: 500;
    margin-bottom: 26mm;
  }
  .cover .meta {
    font-size: 11px;
    color: rgba(255,255,255,0.6);
    letter-spacing: 1px;
  }
  .cover .divider { width: 48px; height: 2px; background: rgba(255,255,255,.6); margin: 10mm 0; }

  /* TOC */
  .toc-page { padding: 26mm 22mm; background: var(--surface); }
  .toc-page .kicker-top {
    font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    color: var(--primary); font-size: 11px; margin-bottom: 4mm;
  }
  .toc-page h1 { font-size: 30px; font-weight: 500; color: var(--text); margin-bottom: 14mm; }
  .toc-row {
    display: flex; align-items: baseline; gap: 6mm;
    padding: 4.2mm 0; border-bottom: 1px solid #E5E7EB;
  }
  .toc-num { font-family: 'Outfit'; font-weight: 600; color: var(--gold); font-size: 13px; width: 10mm; }
  .toc-title { font-family: 'Outfit'; font-weight: 500; font-size: 14px; flex: 1; color: var(--text); }
  .toc-kicker { font-size: 10px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; }

  /* Section pages */
  .section-page { background: var(--surface); padding: 16mm 16mm; }
  .scene {
    display: flex;
    align-items: center;
    gap: 12mm;
    height: 265mm;
  }
  .scene.reverse { flex-direction: row-reverse; }
  .scene-media { flex: 0 0 68mm; display: flex; justify-content: center; }
  .phone-frame {
    width: 62mm;
    border-radius: 7mm;
    background: #0B0F0C;
    padding: 2.2mm;
    box-shadow: 0 10px 30px -8px rgba(18,32,24,0.4);
  }
  .phone-frame img { width: 100%; display: block; border-radius: 5.2mm; }
  .phone-frame.placeholder { height: 130mm; }

  /* Web platform: stacked layout, wide browser-chrome frame on top */
  .scene.web { flex-direction: column; align-items: stretch; gap: 8mm; height: auto; }
  .scene.web .scene-media { flex: none; width: 100%; }
  .browser-frame {
    width: 100%; max-height: 150mm; border-radius: 4mm; overflow: hidden;
    background: #E2E4EA; box-shadow: 0 10px 30px -8px rgba(18,32,24,0.35);
    display: flex; flex-direction: column;
  }
  /* object-fit: contain — a web screenshot's aspect ratio is unpredictable
     (a 16:9 viewport capture vs. a tall full-page capture), so this always
     letterboxes inside the capped height instead of blowing out the page. */
  .browser-frame img { width: 100%; flex: 1; min-height: 0; object-fit: contain; background: #fff; display: block; }
  .browser-frame.placeholder { height: 90mm; }
  .browser-bar { display: flex; align-items: center; gap: 3mm; padding: 2.6mm 4mm; background: #EDEEF2; }
  .browser-bar .dot { width: 2.6mm; height: 2.6mm; border-radius: 50%; display: inline-block; }
  .browser-bar .dot.r { background: #FF5F57; }
  .browser-bar .dot.y { background: #FEBC2E; }
  .browser-bar .dot.g { background: #28C840; }
  .browser-url {
    margin-left: 2mm; background: #fff; border-radius: 999px; padding: 1.3mm 4mm;
    font-size: 9px; color: var(--text-secondary); font-weight: 600;
  }

  .scene-copy { flex: 1; }
  .scene-num {
    font-family: 'Outfit'; font-weight: 600; color: var(--gold);
    font-size: 11px; letter-spacing: 1px; margin-bottom: 3mm;
  }
  .kicker {
    font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
    color: var(--primary); font-size: 10.5px; margin-bottom: 3mm;
  }
  .scene-copy h2 { font-size: 24px; font-weight: 500; color: var(--text); margin-bottom: 5mm; line-height: 1.2; }
  .scene-copy .lead { font-size: 12.5px; line-height: 1.65; color: var(--text); margin-bottom: 5mm; }
  .scene-copy ul { margin: 0; padding-left: 4.5mm; }
  .scene-copy li { font-size: 11.5px; line-height: 1.55; color: var(--text-secondary); margin-bottom: 2mm; }
  .scene-copy li::marker { color: var(--gold); }

  /* Closing */
  .closing {
    background: radial-gradient(circle at 50% 85%, var(--primary-light) 0%, var(--primary) 55%, var(--primary-dark) 100%);
    color: #fff; display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center; padding: 40mm 24mm;
  }
  .closing h2 { font-size: 30px; font-weight: 300; margin-bottom: 6mm; }
  .closing p { font-size: 13px; color: rgba(255,255,255,0.8); max-width: 130mm; line-height: 1.7; }
  .closing .tag { color: rgba(255,255,255,.85); font-weight: 600; letter-spacing: 2px; text-transform: uppercase; font-size: 11px; margin-top: 10mm; }

  .footer-strip {
    position: absolute; bottom: 10mm; left: 16mm; right: 16mm;
    display: flex; justify-content: space-between;
    font-size: 9px; color: var(--text-secondary); letter-spacing: 0.5px;
  }
</style>
</head>
<body>

  <div class="page cover">
    <div class="mark"><img src="file://${LOGO_PATH}" /></div>
    <div class="eyebrow">Complete App Walkthrough</div>
    <h1>${script.appName}</h1>
    <div class="tagline">${script.tagline}</div>
    <div class="divider"></div>
    <div class="meta">Feature Guide &amp; Screen-by-Screen Reference · ${fmtDate}</div>
  </div>

  <div class="page toc-page">
    <div class="kicker-top">Contents</div>
    <h1>Everything in this guide</h1>
    ${tocHtml}
    <div class="footer-strip"><span>${script.appName}</span><span>Page 2</span></div>
  </div>

  ${sectionsHtml}

  <div class="page closing">
    <div class="mark small"><img src="file://${LOGO_PATH}" /></div>
    <h2>Thank you</h2>
    <p>This guide covered every major screen in ${script.appName}, from describing a goal and connecting a model to reading the results, the lessons it learned, and the Doctor's checks. Keys stay in your own .env file; nothing in this guide shows one.</p>
    <div class="tag">${script.tagline}</div>
  </div>

</body>
</html>`;

writeFileSync(join(ROOT, "docs", "walkthrough.html"), html);
console.log("wrote docs/walkthrough.html —", script.scenes.length, "sections");
