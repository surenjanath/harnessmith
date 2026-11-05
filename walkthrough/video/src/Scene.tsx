import React from "react";
import {
  AbsoluteFill,
  Audio,
  Freeze,
  Img,
  OffthreadVideo,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import type { Scene as SceneType } from "./scenes";
import { APP_NAME, AUDIO_DELAY, BRAND, FPS, OPTIONS, scenes } from "./scenes";
import { OUTFIT, MANROPE } from "./fonts";

// Motion in these scenes always points at something: the camera moves to the region the narrator is
// describing, a label names it, captions follow the voice, and live clips show the app's own animation.
// Set "video": {"motion": false} in script.json for static screens with plain crossfades.

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const ON_DARK = "#f2f2ee";

// 0 -> 1 without overshoot, starting at `atSeconds` into the scene.
function useEase() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (atSeconds: number, seconds = 0.8) =>
    spring({ frame: frame - atSeconds * fps, fps, config: { damping: 200 }, durationInFrames: Math.round(seconds * fps) });
}

const hasAudio = (id: string) => {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const manifest = require("./audioManifest.json");
    return Boolean(manifest[id]);
  } catch {
    return false;
  }
};

const Narration: React.FC<{ id: string }> = ({ id }) =>
  hasAudio(id) ? (
    <Sequence from={Math.round(AUDIO_DELAY * FPS)} layout="none">
      <Audio src={staticFile(`audio/${id}.mp3`)} />
    </Sequence>
  ) : null;

// "pill": a wordmark (usually dark on transparent) on a light badge. "tile": a square app icon.
const LogoBadge: React.FC<{ width?: number }> = ({ width = 220 }) =>
  OPTIONS.logo === "tile" ? (
    <Img
      src={staticFile("logo.png")}
      style={{ width, height: width, display: "block", borderRadius: width * 0.2, border: "1px solid rgba(255,255,255,0.26)" }}
    />
  ) : (
    <div
      style={{
        backgroundColor: "#FFFFFF",
        borderRadius: 999,
        padding: `${width * 0.045}px ${width * 0.085}px`,
        boxShadow: "0 10px 24px -8px rgba(0,0,0,0.35)",
        display: "inline-flex",
      }}
    >
      <Img src={staticFile("logo.png")} style={{ width, display: "block" }} />
    </div>
  );

// ---------- the screen: screenshot or live clip, with a camera, highlights, labels and a pointer ----------

const Pointer: React.FC<{ x: number; y: number; opacity: number; click: number }> = ({ x, y, opacity, click }) => (
  <div style={{ position: "absolute", left: x, top: y, opacity, pointerEvents: "none" }}>
    <div
      style={{
        position: "absolute",
        left: -34,
        top: -34,
        width: 68,
        height: 68,
        borderRadius: 34,
        border: `3px solid ${BRAND.gold}`,
        transform: `scale(${0.3 + click * 0.9})`,
        opacity: click > 0 && click < 1 ? 1 - click : 0,
      }}
    />
    <svg width="34" height="34" viewBox="0 0 24 24" style={{ position: "absolute", left: -4, top: -3, filter: "drop-shadow(0 3px 5px rgba(0,0,0,0.35))", transform: `scale(${1 - 0.12 * Math.sin(Math.PI * clamp(click * 2.5, 0, 1))})` }}>
      <path d="M5 3l14 8.2-6.3 1.5L9.4 19 5 3z" fill="#111" stroke="#fff" strokeWidth="1.4" strokeLinejoin="round" />
    </svg>
  </div>
);

export const Screen: React.FC<{
  scene: SceneType;
  width: number;
  height: number;
  fit: "contain" | "cover-top"; // contain: letterbox a web capture. cover-top: fill a phone frame from the top
}> = ({ scene, width: VW, height: VH, fit }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const ease = useEase();
  const ratio = scene.imgW / scene.imgH;
  const CW = fit === "contain" ? Math.min(VW, VH * ratio) : VW;
  const CH = CW / ratio;
  const baseCy = Math.min(0.5, VH / (2 * CH));

  // Camera: a weighted blend of the regions in focus. Each fades in at its cue and out when the next begins.
  const motion = OPTIONS.motion;
  const drift = motion ? interpolate(frame, [0, durationInFrames], [0, 0.022]) : 0;
  let s = 1 + drift, cx = 0.5, cy = baseCy;
  const weights = scene.focus.map((f) => {
    if (!motion) return 0;
    const w = ease(f.at, 0.9) - ease(f.end, 0.9);
    const z = f.zoom ?? clamp(Math.min((0.8 * VW) / (f.w * CW), (0.8 * VH) / (f.h * CH)), 1.06, 1.9);
    s += w * (z - 1);
    cx += w * (f.x + f.w / 2 - 0.5);
    cy += w * (f.y + f.h / 2 - baseCy);
    return w;
  });
  const hx = VW / (2 * CW * s), hy = VH / (2 * CH * s);
  cx = hx >= 0.5 ? 0.5 : clamp(cx, hx, 1 - hx);
  cy = hy >= 0.5 ? 0.5 : clamp(cy, hy, 1 - hy);
  const px = (X: number) => VW / 2 + (X - cx) * CW * s;
  const py = (Y: number) => VH / 2 + (Y - cy) * CH * s;

  const clip = scene.clip;
  const clipFrame = clip ? clamp(frame - Math.round(clip.start * fps), 0, clip.frames - 1) : 0;
  const fill: React.CSSProperties = { position: "absolute", left: 0, top: 0, width: CW, height: CH, display: "block" };

  return (
    <div style={{ width: VW, height: VH, position: "relative", overflow: "hidden", backgroundColor: BRAND.bg }}>
      <div style={{ position: "absolute", left: 0, top: 0, width: CW, height: CH, transformOrigin: "0 0", transform: `translate(${px(0)}px, ${py(0)}px) scale(${s})` }}>
        {scene.image ? <Img src={staticFile(`screens/${scene.image}`)} style={fill} /> : null}
        {clip ? (
          <Freeze frame={clipFrame}>
            <OffthreadVideo src={staticFile(`clips/${clip.src}`)} muted style={fill} />
          </Freeze>
        ) : null}
        {scene.redact.map(([x, y, w, h], i) => (
          <div key={i} style={{ position: "absolute", left: x * CW, top: y * CH, width: w * CW, height: h * CH, backdropFilter: "blur(9px)", backgroundColor: "rgba(140,140,135,0.18)", borderRadius: 4 }} />
        ))}
      </div>

      {scene.focus.map((f, i) => {
        const w = weights[i];
        if (w < 0.004) return null;
        const pad = 10;
        const left = px(f.x) - pad, top = py(f.y) - pad;
        const bw = f.w * CW * s + pad * 2, bh = f.h * CH * s + pad * 2;
        const labelAbove = top > 62;
        const pointerIn = ease(f.at + 0.15, 0.8);
        const tx = left + bw * 0.5, ty = top + Math.min(bh * 0.5, 60);
        return (
          <React.Fragment key={i}>
            <div
              style={{
                position: "absolute", left, top, width: bw, height: bh, borderRadius: 10,
                border: `2.5px solid ${BRAND.gold}`, opacity: w,
                boxShadow: `0 0 0 4000px rgba(16,16,15,${0.3 * w})`,
              }}
            />
            {f.label ? (
              <div
                style={{
                  position: "absolute",
                  left: clamp(left, 8, VW - 8 - f.label.length * 12.5 - 44),
                  top: labelAbove ? top - 50 : Math.min(top + bh + 10, VH - 50),
                  opacity: w,
                  transform: `translateY(${(1 - w) * (labelAbove ? 10 : -10)}px)`,
                  backgroundColor: BRAND.primary, color: ON_DARK,
                  fontFamily: MANROPE, fontWeight: 600, fontSize: 21, lineHeight: 1,
                  padding: "10px 16px 11px", borderRadius: 8, whiteSpace: "nowrap",
                  display: "flex", alignItems: "center", gap: 10,
                  boxShadow: "0 8px 20px -8px rgba(0,0,0,0.5)",
                }}
              >
                <span style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: BRAND.gold, display: "block" }} />
                {f.label}
              </div>
            ) : null}
            {f.cursor ? (
              <Pointer
                x={tx + (1 - pointerIn) * 220}
                y={ty + (1 - pointerIn) * 160}
                opacity={Math.min(1, w * 1.6)}
                click={clamp((frame / fps - (f.at + 1.0)) / 0.55, 0, 1)}
              />
            ) : null}
          </React.Fragment>
        );
      })}
    </div>
  );
};

// ---------- captions: one short line at a time, each word lighting up as it is spoken ----------

export const Captions: React.FC<{ scene: SceneType; dark?: boolean; fontSize?: number }> = ({ scene, dark, fontSize = 30 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (!OPTIONS.captions || !scene.pages.length) return null;
  const t = frame / fps - AUDIO_DELAY;
  let page = scene.pages[0];
  for (const p of scene.pages) if (t >= p.s - 0.12) page = p;
  if (t < scene.pages[0].s - 0.3 || t > page.e + 0.9) return null;
  const appear = interpolate(t, [page.s - 0.12, page.s + 0.1], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const ink = dark ? ON_DARK : BRAND.textPrimary;
  const quiet = dark ? "rgba(242,242,238,0.42)" : BRAND.platinum;
  return (
    <div style={{ fontFamily: MANROPE, fontWeight: 500, fontSize, lineHeight: 1.3, textAlign: "center", opacity: appear, transform: `translateY(${(1 - appear) * 8}px)` }}>
      {scene.words.slice(page.from, page.to + 1).map((w, i) => {
        const now = t >= w.s && t < w.e + 0.04;
        return (
          <span key={i} style={{ color: t >= w.s ? ink : quiet, fontWeight: now ? 700 : 500, whiteSpace: "pre" }}>
            {w.t + " "}
          </span>
        );
      })}
    </div>
  );
};

// ---------- title cards (intro / outro): the tour's own screens drift behind the title ----------

const Montage: React.FC = () => {
  const frame = useCurrentFrame();
  const shots = scenes.filter((s) => s.image);
  if (!shots.length) return null;
  const rows = [0, 1, 2];
  const tileH = 330;
  return (
    <AbsoluteFill style={{ transform: "rotate(-7deg) scale(1.35)", opacity: 0.3, justifyContent: "center", gap: 26 }}>
      {rows.map((r) => {
        const list = shots.filter((_, i) => i % rows.length === r);
        const loop = [...list, ...list, ...list];
        const dir = r % 2 === 0 ? -1 : 1;
        return (
          <div key={r} style={{ display: "flex", gap: 26, transform: `translateX(${dir * frame * 0.9 - (dir < 0 ? 200 : 1500)}px)` }}>
            {loop.map((s, i) => (
              <Img key={i} src={staticFile(`screens/${s.image}`)} style={{ height: tileH, width: (tileH * s.imgW) / s.imgH, borderRadius: 12, flexShrink: 0, objectFit: "cover" }} />
            ))}
          </div>
        );
      })}
    </AbsoluteFill>
  );
};

export const TitleCard: React.FC<{ scene: SceneType }> = ({ scene }) => {
  const ease = useEase();
  const { width, height } = useVideoConfig();
  const words = scene.title.split(" ");
  const logoIn = ease(0.1, 0.9);
  const big = Math.round(Math.min(width * 0.042, 84));
  return (
    <AbsoluteFill style={{ backgroundColor: BRAND.primaryDark }}>
      <Narration id={scene.id} />
      {OPTIONS.motion ? <Montage /> : null}
      <AbsoluteFill style={{ backgroundImage: `radial-gradient(ellipse at 50% 45%, ${BRAND.primary}e6 0%, ${BRAND.primaryDark}f2 60%, ${BRAND.primaryDark} 100%)` }} />
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 34, padding: "0 80px" }}>
          <div style={{ opacity: logoIn, transform: `scale(${0.82 + 0.18 * logoIn})` }}>
            <LogoBadge width={OPTIONS.logo === "tile" ? 150 : 260} />
          </div>
          <div style={{ fontFamily: OUTFIT, fontWeight: 500, fontSize: big, letterSpacing: -big * 0.03, lineHeight: 1.08, color: "#FFFFFF", textAlign: "center" }}>
            {words.map((w, i) => {
              const p = OPTIONS.motion ? ease(0.45 + i * 0.09, 0.7) : 1;
              return (
                <span key={i} style={{ display: "inline-block", whiteSpace: "pre", opacity: p, transform: `translateY(${(1 - p) * 34}px)` }}>
                  {w + (i < words.length - 1 ? " " : "")}
                </span>
              );
            })}
          </div>
          <div style={{ width: 72 * ease(0.9, 0.8), height: 2, backgroundColor: BRAND.gold }} />
          <div style={{ fontFamily: MANROPE, fontWeight: 500, fontSize: 24, color: BRAND.silver, letterSpacing: 3 + 3 * (1 - ease(1.0, 1.2)), textTransform: "uppercase", textAlign: "center", opacity: ease(1.0, 0.9) }}>
            {scene.eyebrow}
          </div>
        </div>
      </AbsoluteFill>
      <div style={{ position: "absolute", left: 120, right: 120, bottom: height * 0.085 }}>
        <Captions scene={scene} dark fontSize={28} />
      </div>
    </AbsoluteFill>
  );
};

// Heading block shared by both layouts: eyebrow and title rise in at the start of each scene.
const Heading: React.FC<{ scene: SceneType; titleSize: number; align?: "left" | "center" }> = ({ scene, titleSize, align = "left" }) => {
  const ease = useEase();
  const a = OPTIONS.motion ? ease(0.1, 0.6) : 1, b = OPTIONS.motion ? ease(0.2, 0.7) : 1;
  return (
    <div style={{ textAlign: align }}>
      <div style={{ fontFamily: MANROPE, fontWeight: 600, fontSize: 16, letterSpacing: 3, color: BRAND.silver, textTransform: "uppercase", opacity: a, transform: `translateY(${(1 - a) * 14}px)` }}>
        {scene.eyebrow}
      </div>
      <div style={{ marginTop: 14, fontFamily: OUTFIT, fontWeight: 500, fontSize: titleSize, lineHeight: 1.14, letterSpacing: -titleSize * 0.022, color: ON_DARK, opacity: b, transform: `translateY(${(1 - b) * 22}px)` }}>
        {scene.title}
      </div>
    </div>
  );
};

// ---------- mobile: vertical video, phone frame ----------

export const ScreenShowcase: React.FC<{ scene: SceneType; index: number }> = ({ scene }) => {
  const ease = useEase();
  const rise = OPTIONS.motion ? ease(0, 0.8) : 1;
  return (
    <AbsoluteFill style={{ backgroundColor: BRAND.bg }}>
      <Narration id={scene.id} />
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 300, background: `linear-gradient(135deg, ${BRAND.primary} 0%, ${BRAND.primaryLight} 100%)` }} />
      <AbsoluteFill style={{ justifyContent: "flex-start", alignItems: "center", paddingTop: 50 }}>
        <LogoBadge width={OPTIONS.logo === "tile" ? 64 : 128} />
        <div style={{ marginTop: 22, padding: "0 70px" }}>
          <Heading scene={scene} titleSize={44} align="center" />
        </div>
        <div
          style={{
            marginTop: 34, width: 620, height: 1250, borderRadius: 56, backgroundColor: "#0B0F0C", padding: 14,
            boxShadow: "0 40px 80px -20px rgba(0,0,0,0.5)",
            transform: `translateY(${(1 - rise) * 40}px) scale(${0.97 + 0.03 * rise})`,
          }}
        >
          <div style={{ width: 592, height: 1222, borderRadius: 44, overflow: "hidden" }}>
            <Screen scene={scene} width={592} height={1222} fit="cover-top" />
          </div>
        </div>
      </AbsoluteFill>
      <div style={{ position: "absolute", left: 60, right: 60, bottom: 110 }}>
        <Captions scene={scene} fontSize={38} />
      </div>
    </AbsoluteFill>
  );
};

// ---------- web: landscape video, side panel and a browser window ----------

const BrowserChrome: React.FC<{ url?: string; width: number; height: number; children: React.ReactNode }> = ({ url, width, height, children }) => (
  <div style={{ width, height, borderRadius: 14, backgroundColor: "#d4d4cf", boxShadow: "0 30px 70px -24px rgba(0,0,0,0.45)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
    <div style={{ height: 46, flexShrink: 0, display: "flex", alignItems: "center", gap: 20, padding: "0 18px", backgroundColor: "#dedede" }}>
      <div style={{ display: "flex", gap: 8 }}>
        <div style={{ width: 12, height: 12, borderRadius: 6, background: "#FF5F57" }} />
        <div style={{ width: 12, height: 12, borderRadius: 6, background: "#FEBC2E" }} />
        <div style={{ width: 12, height: 12, borderRadius: 6, background: "#28C840" }} />
      </div>
      {url ? (
        <div style={{ flex: 1, maxWidth: 440, backgroundColor: "#f1f1f0", borderRadius: 999, padding: "6px 18px", fontFamily: MANROPE, fontWeight: 500, fontSize: 15, color: "#5f5f5c" }}>
          {url}
        </div>
      ) : null}
    </div>
    <div style={{ flex: 1, position: "relative" }}>{children}</div>
  </div>
);

export const WebScreenShowcase: React.FC<{ scene: SceneType; index: number }> = ({ scene, index }) => {
  const ease = useEase();
  const SIDE = 440, FRAME_W = 1360, BAR = 46, FRAME_H = BAR + 765;
  const rise = OPTIONS.motion ? ease(0, 0.8) : 1;
  return (
    <AbsoluteFill style={{ backgroundColor: BRAND.bg }}>
      <Narration id={scene.id} />
      <div
        style={{
          position: "absolute", left: 0, top: 0, bottom: 0, width: SIDE,
          background: `linear-gradient(160deg, ${BRAND.primary} 0%, ${BRAND.primaryDark} 100%)`,
          display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 48px",
        }}
      >
        <div style={{ position: "absolute", left: 48, top: 56, display: "flex", alignItems: "center", gap: 16 }}>
          <LogoBadge width={OPTIONS.logo === "tile" ? 52 : 120} />
          {OPTIONS.logo === "tile" ? (
            <div style={{ fontFamily: MANROPE, fontWeight: 500, fontSize: 19, lineHeight: 1.2, color: ON_DARK, maxWidth: 190 }}>{APP_NAME}</div>
          ) : null}
        </div>
        <Heading scene={scene} titleSize={40} />
        <div style={{ position: "absolute", left: 48, bottom: 52, fontFamily: MANROPE, fontWeight: 500, fontSize: 16, color: BRAND.platinum, fontVariantNumeric: "tabular-nums" }}>
          {String(index + 1).padStart(2, "0")} / {String(scenes.length).padStart(2, "0")}
        </div>
      </div>

      {/* Plain positioned div, not AbsoluteFill: AbsoluteFill hardcodes width: 100%, so overriding
          just `left` on it pushes this region off the right edge instead of shrinking it. */}
      <div style={{ position: "absolute", left: SIDE + 60, top: 54, transform: `translateY(${(1 - rise) * 26}px) scale(${0.985 + 0.015 * rise})`, transformOrigin: "50% 60%" }}>
        <BrowserChrome url={scene.url} width={FRAME_W} height={FRAME_H}>
          <Screen scene={scene} width={FRAME_W} height={FRAME_H - BAR} fit="contain" />
        </BrowserChrome>
      </div>

      <div style={{ position: "absolute", left: SIDE + 60, width: FRAME_W, top: 54 + FRAME_H + 44 }}>
        <Captions scene={scene} />
      </div>
    </AbsoluteFill>
  );
};
