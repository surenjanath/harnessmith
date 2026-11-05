// Capture walkthrough material from a running web app through Chrome's DevTools protocol (no dependencies).
//
//   node capture.mjs shots.json            stills (+ element rects for highlights, + live clips)
//   node capture.mjs shots.json --probe    also list the blocks visible in each shot, to pick highlight targets from
//
// Each shot: { name, url, wait?, js?, jsWait?, sel?|y?, pad?, settle?, marks?, record? }
//   sel / y   scroll an element (or a pixel offset) to the top of the viewport before capturing
//   marks     { key: "css selector" | "text=Heading text" }  -> screenshots/marks.json, used as scene focus targets.
//             "text=..." takes the block (parent element) of the first heading containing that text; append "^" to climb further.
//   record    { seconds, actions?: [{ at, js }] }  -> screenshots/<name>.mp4 (a live clip); the still is its last frame
// Env: BASE (default http://localhost:3000), VIEW=1440x810 (CSS px), OUT=1920 (still width in px).
import { spawn, execFileSync } from "node:child_process";
import { writeFileSync, readFileSync, mkdtempSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const BASE = process.env.BASE || "http://localhost:3000";
const [W, H] = (process.env.VIEW || "1440x810").split("x").map(Number);
const OUT_W = Number(process.env.OUT || 1920), SCALE = OUT_W / W, PORT = 9333;
const PROBE = process.argv.includes("--probe");
const shots = JSON.parse(readFileSync(process.argv[2], "utf8"));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
mkdirSync("screenshots", { recursive: true });
const marksPath = join("screenshots", "marks.json");
const marks = existsSync(marksPath) ? JSON.parse(readFileSync(marksPath, "utf8")) : {};

const chrome = spawn("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "wt-cap-"))}`,
  `--window-size=${W},${H}`, "--hide-scrollbars", "--no-first-run", "about:blank",
], { stdio: "ignore" });

let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json`)).json()).find((t) => t.type === "page"); } catch (_) {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map(); let onFrame = null;
ws.onmessage = (e) => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  else if (m.method === "Page.screencastFrame") { send("Page.screencastFrameAck", { sessionId: m.params.sessionId }); onFrame?.(m.params); }
};
const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Page.enable");
await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: process.env.SCHEME || "light" }] });
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: SCALE, mobile: false });

// Runs in the page: resolve a mark spec to a viewport rect, as fractions of the viewport.
const RECT_FN = `(spec)=>{
  let el=null, up=0; while(spec.endsWith('^')){ up++; spec=spec.slice(0,-1); }
  if(spec.startsWith('text=')){ const t=spec.slice(5);
    const h=[...document.querySelectorAll('h1,h2,h3,h4,legend,summary,.sheet-title')].find(e=>e.textContent.includes(t)); el=h&&h.parentElement; }
  else el=document.querySelector(spec);
  for(let i=0;i<up&&el;i++) el=el.parentElement;
  if(!el) return null; const r=el.getBoundingClientRect();
  const x0=Math.max(0,r.left), y0=Math.max(0,r.top), x1=Math.min(innerWidth,r.right), y1=Math.min(innerHeight,r.bottom);
  if(x1-x0<8||y1-y0<8) return null;
  return [x0/innerWidth,y0/innerHeight,(x1-x0)/innerWidth,(y1-y0)/innerHeight].map(v=>+v.toFixed(4)); }`;
const PROBE_FN = `(()=>{ const out=[], seen=new Set(), A=innerWidth*innerHeight;
  const add=(el,how)=>{ if(!el||seen.has(el)) return; seen.add(el); const r=el.getBoundingClientRect();
    const x0=Math.max(0,r.left), y0=Math.max(0,r.top), x1=Math.min(innerWidth,r.right), y1=Math.min(innerHeight,r.bottom);
    if(x1-x0<40||y1-y0<24) return; const a=(x1-x0)*(y1-y0)/A; if(a<0.012||a>0.6) return;
    out.push(how+'  ['+[x0/innerWidth,y0/innerHeight,(x1-x0)/innerWidth,(y1-y0)/innerHeight].map(v=>v.toFixed(2)).join(',')+']  '+(el.innerText||'').replace(/\\s+/g,' ').slice(0,46)); };
  document.querySelectorAll('h1,h2,h3,h4,.sheet-title').forEach(h=>{ const t=h.textContent.trim().replace(/\\s+/g,' ').slice(0,28); add(h.parentElement,'text='+t); add(h.parentElement&&h.parentElement.parentElement,'text='+t+'^'); });
  document.querySelectorAll('[id]').forEach(e=>add(e,'#'+e.id));
  document.querySelectorAll('table,form,canvas,svg,ul,section,figure').forEach(e=>add(e,e.tagName.toLowerCase()+(e.className&&typeof e.className==='string'?'.'+e.className.trim().split(/\\s+/).join('.'):'')));
  return out.join('\\n'); })()`;

for (const s of shots) {
  await send("Page.navigate", { url: "about:blank" });
  await sleep(150);
  await send("Page.navigate", { url: BASE + s.url });
  await sleep(s.wait ?? 2500);
  if (s.js) { await evaluate(s.js); await sleep(s.jsWait ?? 1500); }
  if (s.sel) await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(s.sel)}); if(e) window.scrollTo(0, e.getBoundingClientRect().top + scrollY - ${s.pad ?? 24});})()`);
  else if (s.y != null) await evaluate(`window.scrollTo(0, ${s.y})`);
  await sleep(s.settle ?? 700);

  if (s.record) {
    const dir = mkdtempSync(join(tmpdir(), "wt-rec-")), frames = [];
    onFrame = (p) => { const f = join(dir, `f${String(frames.length).padStart(5, "0")}.jpg`); writeFileSync(f, Buffer.from(p.data, "base64")); frames.push({ f, t: p.metadata.timestamp }); };
    await send("Page.startScreencast", { format: "jpeg", quality: 95, everyNthFrame: 1 });
    const t0 = Date.now();
    for (const a of [...(s.record.actions || [])].sort((p, q) => p.at - q.at)) {
      await sleep(Math.max(0, a.at * 1000 - (Date.now() - t0)));
      await evaluate(a.js);
    }
    await sleep(Math.max(0, s.record.seconds * 1000 - (Date.now() - t0)));
    await send("Page.stopScreencast"); onFrame = null;
    const total = (Date.now() - t0) / 1000;
    let list = "ffconcat version 1.0\n";
    frames.forEach((fr, i) => { const d = (i + 1 < frames.length ? frames[i + 1].t : frames[0].t + total) - fr.t; list += `file '${fr.f}'\nduration ${Math.max(0.001, d).toFixed(4)}\n`; });
    if (frames.length) list += `file '${frames[frames.length - 1].f}'\n`;
    writeFileSync(join(dir, "list.txt"), list);
    execFileSync("ffmpeg", ["-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", join(dir, "list.txt"),
      "-vf", `fps=30,scale=${OUT_W}:${Math.round(H * SCALE)}:flags=lanczos`, "-pix_fmt", "yuv420p", "-crf", "16", join("screenshots", s.name + ".mp4")]);
    rmSync(dir, { recursive: true, force: true });
    console.log(s.name, `clip: ${frames.length} frames over ${total.toFixed(1)} s`);
  }

  for (const [k, spec] of Object.entries(s.marks || {})) {
    const r = await evaluate(`(${RECT_FN})(${JSON.stringify(spec)})`);
    if (r) marks[`${s.name}.${k}`] = r; else console.warn(`  ! mark not found or off-screen: ${s.name}.${k} (${spec})`);
  }
  const info = await evaluate(`JSON.stringify({h: document.documentElement.scrollHeight, y: Math.round(scrollY), hash: location.hash})`);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join("screenshots", s.name + ".png"), Buffer.from(shot.result.data, "base64"));
  console.log(s.name, info);
  if (PROBE) console.log(await evaluate(PROBE_FN), "\n");
}
writeFileSync(marksPath, JSON.stringify(marks, null, 1));
ws.close(); chrome.kill();
