import React from "react";
import { Composition } from "remotion";
import { Walkthrough } from "./Walkthrough";
import { FPS, scenes, TOTAL_FRAMES, VIDEO_WIDTH, VIDEO_HEIGHT } from "./scenes";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Walkthrough"
      component={Walkthrough}
      durationInFrames={TOTAL_FRAMES}
      fps={FPS}
      width={VIDEO_WIDTH}
      height={VIDEO_HEIGHT}
      defaultProps={{ scenes }}
    />
  );
};
