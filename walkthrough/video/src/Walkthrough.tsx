import React from "react";
import { AbsoluteFill, Audio, interpolate, staticFile, useCurrentFrame } from "remotion";
import { TransitionSeries, linearTiming, springTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { wipe } from "@remotion/transitions/wipe";
import { scenes, FPS, Scene, BRAND, PLATFORM, OPTIONS, TRANSITION_FRAMES, TOTAL_FRAMES } from "./scenes";
import { TitleCard, ScreenShowcase, WebScreenShowcase } from "./Scene";

const isTitle = (s: Scene) => s.id === "intro" || s.id === "outro";

// A change of chapter (a new eyebrow) gets a directional move; scenes inside a chapter crossfade,
// so the heading panel holds still while the screen changes.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function transitionBetween(a: Scene, b: Scene): { presentation: any; timing: any } {
  const calm = !OPTIONS.motion || OPTIONS.transition === "fade" || isTitle(a) || isTitle(b) || a.eyebrow === b.eyebrow;
  if (calm) return { presentation: fade(), timing: linearTiming({ durationInFrames: TRANSITION_FRAMES }) };
  const timing = springTiming({ config: { damping: 200 }, durationInFrames: TRANSITION_FRAMES });
  const direction = PLATFORM === "web" ? "from-right" : "from-bottom";
  return { presentation: OPTIONS.transition === "wipe" ? wipe({ direction }) : slide({ direction }), timing };
}

// One segment per chapter, filling as the video plays.
const ProgressBar: React.FC<{ starts: number[]; frames: number[]; list: Scene[] }> = ({ starts, frames, list }) => {
  const frame = useCurrentFrame();
  const body = list.map((s, i) => ({ s, i })).filter(({ s }) => !isTitle(s));
  if (!body.length) return null;
  const from = starts[body[0].i], to = starts[body[body.length - 1].i] + frames[body[body.length - 1].i];
  const chapters: { start: number; end: number }[] = [];
  body.forEach(({ s, i }, k) => {
    const end = starts[i] + frames[i] - (i < list.length - 1 ? TRANSITION_FRAMES : 0);
    if (k > 0 && body[k - 1].s.eyebrow === s.eyebrow) chapters[chapters.length - 1].end = end;
    else chapters.push({ start: starts[i], end });
  });
  const visible = interpolate(frame, [from, from + 12, to - TRANSITION_FRAMES - 10, to - TRANSITION_FRAMES], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const box: React.CSSProperties = PLATFORM === "web" ? { left: 500, width: 1360, bottom: 30 } : { left: 60, right: 60, bottom: 54 };
  return (
    <div style={{ position: "absolute", ...box, height: 4, display: "flex", gap: 5, opacity: visible }}>
      {chapters.map((c, k) => (
        <div key={k} style={{ flexGrow: c.end - c.start, flexBasis: 0, backgroundColor: "rgba(120,120,115,0.28)", borderRadius: 2, overflow: "hidden" }}>
          <div style={{ width: `${interpolate(frame, [c.start, c.end], [0, 100], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })}%`, height: "100%", backgroundColor: BRAND.textPrimary }} />
        </div>
      ))}
    </div>
  );
};

export const Walkthrough: React.FC<{ scenes: Scene[] }> = ({ scenes: list }) => {
  const frames = list.map((s) => Math.round(s.seconds * FPS));
  const starts: number[] = [];
  frames.reduce((at, f, i) => { starts[i] = at; return at + f - TRANSITION_FRAMES; }, 0);
  const Showcase = PLATFORM === "web" ? WebScreenShowcase : ScreenShowcase;

  return (
    <AbsoluteFill style={{ backgroundColor: BRAND.bg }}>
      <TransitionSeries>
        {list.map((scene, index) => (
          <React.Fragment key={scene.id}>
            {index > 0 ? <TransitionSeries.Transition {...transitionBetween(list[index - 1], scene)} /> : null}
            <TransitionSeries.Sequence durationInFrames={frames[index]}>
              {isTitle(scene) ? <TitleCard scene={scene} /> : <Showcase scene={scene} index={index} />}
            </TransitionSeries.Sequence>
          </React.Fragment>
        ))}
      </TransitionSeries>
      <ProgressBar starts={starts} frames={frames} list={list} />
      {OPTIONS.music ? (
        // A bed under the narration: quiet throughout, fading in and out with the video.
        <Audio
          src={staticFile("music.mp3")}
          loop
          volume={(f) => OPTIONS.musicVolume * interpolate(f, [0, 45, TOTAL_FRAMES - 75, TOTAL_FRAMES - 5], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })}
        />
      ) : null}
    </AbsoluteFill>
  );
};

export { scenes };
