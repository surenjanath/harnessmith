// Reads ../../script.json (the single source of truth for narration/copy) and writes src/scenes.ts.
// Re-run this any time script.json, the screenshots, the clips or the audio change. It also:
//   - copies screenshots, clips, the logo and optional music into public/
//   - sets each scene's length from the measured narration audio (../audioDurations.json)
//   - turns narration word timestamps (../captions.json) into caption pages
//   - resolves each focus target ("rect" or "mark") and its "cue" phrase into a time within the scene
import { readFileSync, writeFileSync, existsSync, mkdirSync, copyFileSync } from "fs";
import { execFileSync } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const VIDEO = join(__dirname, "..");
const ROOT = join(VIDEO, ".."); // walkthrough/
const SHOTS = join(ROOT, "screenshots");
const readJson = (p, fallback) => (existsSync(p) ? JSON.parse(readFileSync(p, "utf-8")) : fallback);

const script = readJson(join(ROOT, "script.json"));
const durations = readJson(join(ROOT, "audioDurations.json"), {});
const captions = readJson(join(ROOT, "captions.json"), {});
const marks = readJson(join(SHOTS, "marks.json"), {});

const FPS = 30;
const AUDIO_DELAY = 0.45; // narration starts this long into a scene, clear of the transition
const TAIL = 0.75; // breathing room after the last word
const TRANSITION_FRAMES = 16;
const CUE_LEAD = 0.3; // the picture moves slightly before the word that names it

// "mobile" -> vertical phone-frame video (1080x1920). "web" -> landscape browser-frame video (1920x1080).
const PLATFORM = script.platform === "web" ? "web" : "mobile";
const [VIDEO_WIDTH, VIDEO_HEIGHT] = PLATFORM === "web" ? [1920, 1080] : [1080, 1920];

for (const d of ["screens", "clips", "audio"]) mkdirSync(join(VIDEO, "public", d), { recursive: true });
const pngSize = (p) => { const b = readFileSync(p); return [b.readUInt32BE(16), b.readUInt32BE(20)]; };
const norm = (w) => w.toLowerCase().replace(/[^a-z0-9]+/g, "");

// Find where a spoken phrase starts, in seconds from the start of the narration audio.
function cueTime(words, phrase) {
  const want = phrase.split(/\s+/).map(norm).filter(Boolean);
  const have = words.map((w) => norm(w.t));
  for (let i = 0; i + want.length <= have.length; i++) {
    if (want.every((w, k) => have[i + k] === w)) return words[i].s;
  }
  return null;
}

// Short caption lines: break at sentence ends, at commas once a line is filling up, and at a length limit.
function pagesOf(words) {
  const pages = []; let from = 0, chars = 0;
  const LIMIT = PLATFORM === "web" ? 58 : 34;
  words.forEach((w, i) => {
    chars += w.t.length + 1;
    const last = i === words.length - 1;
    const sentence = /[.!?:]$/.test(w.t), comma = /[,;]$/.test(w.t);
    const nextLen = last ? 0 : words[i + 1].t.length + 1;
    if (last || sentence || (comma && chars > LIMIT * 0.55) || chars + nextLen > LIMIT) {
      pages.push({ from, to: i, s: words[from].s, e: w.e });
      from = i + 1; chars = 0;
    }
  });
  return pages;
}

let warned = 0;
const warn = (...a) => { warned++; console.warn("  !", ...a); };

const scenes = script.scenes.map((s) => {
  const words = captions[s.id] || [];
  const audio = durations[s.id];
  const seconds = audio ? Math.round((AUDIO_DELAY + audio + TAIL) * FPS) / FPS : s.seconds;

  let imgW = 16, imgH = 9;
  if (s.image) {
    const src = join(SHOTS, s.image);
    if (!existsSync(src)) warn(`${s.id}: missing screenshot ${s.image}`);
    else { copyFileSync(src, join(VIDEO, "public", "screens", s.image)); [imgW, imgH] = pngSize(src); }
  }

  const base = s.image ? s.image.replace(/\.[a-z]+$/i, "") : "";
  const raw = (s.focus || []).map((f) => {
    const rect = f.rect || marks[`${base}.${f.mark}`];
    if (!rect) { warn(`${s.id}: focus "${f.mark || f.label}" has no rect and no captured mark`); return null; }
    let at = typeof f.at === "number" ? f.at : null;
    if (at === null && f.cue) {
      const t = cueTime(words, f.cue);
      if (t === null) warn(`${s.id}: cue "${f.cue}" is not in the narration`);
      else at = Math.max(0.3, t + AUDIO_DELAY - CUE_LEAD);
    }
    return { ...f, rect, at };
  }).filter(Boolean);
  // Any focus still without a time is spread evenly through the scene.
  raw.forEach((f, i) => { if (f.at === null) f.at = 1 + ((seconds - 2.5) * i) / Math.max(1, raw.length); });
  raw.sort((a, b) => a.at - b.at);
  const focus = raw.map((f, i) => {
    const next = raw[i + 1]?.at ?? Infinity;
    const end = Math.min(next, f.at + (f.hold ?? 6.5), seconds - 0.7);
    const [x, y, w, h] = f.rect;
    return { x, y, w, h, label: f.label || "", at: +f.at.toFixed(2), end: +Math.max(end, f.at + 0.8).toFixed(2), cursor: Boolean(f.cursor), ...(f.zoom ? { zoom: f.zoom } : {}) };
  });

  let clip = null;
  if (s.clip) {
    const src = join(SHOTS, s.clip);
    if (!existsSync(src)) warn(`${s.id}: missing clip ${s.clip}`);
    else {
      copyFileSync(src, join(VIDEO, "public", "clips", s.clip));
      const frames = Number(execFileSync("ffprobe", ["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", src]).toString().trim());
      let start = 0.5;
      if (s.clipCue) { const t = cueTime(words, s.clipCue); if (t === null) warn(`${s.id}: clipCue "${s.clipCue}" is not in the narration`); else start = Math.max(0.3, t + AUDIO_DELAY - CUE_LEAD); }
      clip = { src: s.clip, frames, start: +start.toFixed(2) };
    }
  }

  return {
    id: s.id, image: s.image || null, eyebrow: s.eyebrow, title: s.title, narration: s.narration,
    seconds, ...(s.url ? { url: s.url } : {}),
    imgW, imgH, focus, clip, redact: s.redact || [], words, pages: pagesOf(words),
  };
});

for (const [src, dst] of [["logo.png", "logo.png"], ["music.mp3", "music.mp3"]]) {
  if (existsSync(join(ROOT, "assets", src))) copyFileSync(join(ROOT, "assets", src), join(VIDEO, "public", dst));
}
const HAS_MUSIC = existsSync(join(VIDEO, "public", "music.mp3"));

const frames = scenes.map((s) => Math.round(s.seconds * FPS));
const TOTAL_FRAMES = frames.reduce((a, b) => a + b, 0) - TRANSITION_FRAMES * Math.max(0, scenes.length - 1);
const B = script.brand;

const out = `// AUTO-GENERATED by scripts/genScenes.mjs from ../script.json. Do not hand-edit;
// edit script.json and re-run the generator instead.
export type Focus = { x: number; y: number; w: number; h: number; label: string; at: number; end: number; cursor: boolean; zoom?: number };
export type Word = { t: string; s: number; e: number };
export type Page = { from: number; to: number; s: number; e: number };
export type Clip = { src: string; frames: number; start: number };
export type Scene = {
  id: string;
  image: string | null;
  eyebrow: string;
  title: string;
  narration: string;
  seconds: number;
  url?: string; // shown in the browser-chrome address bar when PLATFORM === "web"
  imgW: number;
  imgH: number;
  focus: Focus[]; // regions the camera visits, as fractions of the screenshot, timed to the narration
  clip: Clip | null; // a live recording that plays in place of the still
  redact: number[][]; // [x, y, w, h] fractions to blur (personal data)
  words: Word[];
  pages: Page[];
};

export const FPS = ${FPS};
export const PLATFORM: "mobile" | "web" = "${PLATFORM}";
export const VIDEO_WIDTH = ${VIDEO_WIDTH};
export const VIDEO_HEIGHT = ${VIDEO_HEIGHT};
export const AUDIO_DELAY = ${AUDIO_DELAY};
export const TRANSITION_FRAMES = ${TRANSITION_FRAMES};
export const TOTAL_FRAMES = ${TOTAL_FRAMES};

export const APP_NAME = ${JSON.stringify(script.appName)};
export const TAGLINE = ${JSON.stringify(script.tagline)};

// Video options from script.json ("video": {...}); all optional.
export const OPTIONS = {
  captions: ${script.video?.captions !== false},
  motion: ${script.video?.motion !== false}, // false = static screens with plain crossfades
  transition: ${JSON.stringify(script.video?.transition || "slide")} as "slide" | "wipe" | "fade",
  logo: ${JSON.stringify(script.video?.logo || "pill")} as "pill" | "tile", // pill: wordmark on a light badge. tile: a square app icon
  music: ${HAS_MUSIC},
  musicVolume: ${script.video?.musicVolume ?? 0.07},
};

export const BRAND = ${JSON.stringify({
  primary: B.primary, primaryLight: B.primaryLight, primaryDark: B.primaryDark, gold: B.gold, silver: B.silver,
  platinum: B.platinum, bg: B.bg, surface: B.surface, textPrimary: B.textPrimary, textSecondary: B.textSecondary, success: B.success,
}, null, 2)};

export const scenes: Scene[] = ${JSON.stringify(scenes)};

export const totalSeconds = TOTAL_FRAMES / FPS;
`;

writeFileSync(join(VIDEO, "src", "scenes.ts"), out);
const nFocus = scenes.reduce((a, s) => a + s.focus.length, 0), nClips = scenes.filter((s) => s.clip).length;
console.log(`wrote src/scenes.ts: ${scenes.length} scenes, ${nFocus} focus moves, ${nClips} clips, ${(TOTAL_FRAMES / FPS).toFixed(1)} s${HAS_MUSIC ? ", with music" : ""}${Object.keys(captions).length ? "" : " (no captions.json yet: run generate_audio.py for captions and cues)"}`);
if (warned) console.log(`${warned} warning(s) above`);
