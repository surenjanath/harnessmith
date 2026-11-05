// Scene lengths now come straight from the measured audio inside genScenes.mjs
// (it reads ../audioDurations.json), so this just re-runs the generator. Kept so older instructions still work.
await import("./genScenes.mjs");
