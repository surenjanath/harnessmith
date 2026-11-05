import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

ROOT = Path(__file__).parent
script = json.loads((ROOT / "script.json").read_text())

AUDIO_DIR = ROOT / "video" / "public" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# CUSTOMIZE: lang_code must match the voice's accent family:
#   'a' = American English (af_bella, af_heart, af_nicole, af_sarah, am_adam, am_michael)
#   'b' = British English  (bf_emma, bf_isabella, bm_george, bm_lewis)
# SPEED: 1.0 = natural pace; 1.15-1.25 reads as "a bit faster" without sounding rushed.
pipeline = KPipeline(lang_code="a")
VOICE = "am_michael"
SPEED = 1.08
SR = 24000

manifest = {}
durations = {}
captions = {}  # scene id -> [{"t": word, "s": start seconds, "e": end seconds}], used for captions and narration-synced cues

for scene in script["scenes"]:
    scene_id = scene["id"]
    text = scene["narration"]
    chunks, words, offset = [], [], 0.0
    for result in pipeline(text, voice=VOICE, speed=SPEED):
        audio = np.asarray(result.audio)
        for tok in (getattr(result, "tokens", None) or []):
            if tok.start_ts is None or tok.end_ts is None:
                continue
            if words and not any(c.isalnum() for c in tok.text):
                words[-1]["t"] += tok.text  # punctuation rides on the word before it
                continue
            words.append({"t": tok.text, "s": round(offset + tok.start_ts, 3), "e": round(offset + tok.end_ts, 3)})
        chunks.append(audio)
        offset += len(audio) / SR

    full = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    wav_path = AUDIO_DIR / f"{scene_id}.wav"
    sf.write(wav_path, full, SR)
    mp3_path = AUDIO_DIR / f"{scene_id}.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_path)],
        check=True,
        capture_output=True,
    )
    wav_path.unlink()

    manifest[scene_id] = True
    durations[scene_id] = round(len(full) / SR, 2)
    captions[scene_id] = words
    print(f"{scene_id}: {durations[scene_id]:.2f}s, {len(words)} words", flush=True)

(ROOT / "video" / "src").mkdir(parents=True, exist_ok=True)
(ROOT / "video" / "src" / "audioManifest.json").write_text(json.dumps(manifest, indent=2))
(ROOT / "audioDurations.json").write_text(json.dumps(durations, indent=2))
(ROOT / "captions.json").write_text(json.dumps(captions))
print("done")
