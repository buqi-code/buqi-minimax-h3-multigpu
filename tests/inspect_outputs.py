"""Inspect generated AV artifacts; decoded pixels are not a DiT parity proof."""
import json
import pathlib

import av
import numpy as np
from PIL import Image
from bootstrap import setup

root = pathlib.Path(__file__).resolve().parents[1] / "results"
output_root = setup() / "output"
for directory in sorted(root.iterdir()):
    history_path = directory / "history.json"
    if not history_path.exists():
        continue
    history = json.loads(history_path.read_text())
    job = next(iter(history.values()), {})
    files = job.get("outputs", {}).get("save", {}).get("images", [])
    for entry in files:
        path = output_root / entry["subfolder"] / entry["filename"]
        with av.open(str(path)) as container:
            frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        with av.open(str(path)) as container:
            audio = [f.to_ndarray().astype(np.float32) for f in container.decode(audio=0)]
        pixels = np.stack(frames)
        audio_finite = all(bool(np.isfinite(f).all()) for f in audio)
        report = {"file": str(path), "frames": len(frames), "shape": list(frames[0].shape),
                  "pixel_min": int(pixels.min()), "pixel_max": int(pixels.max()),
                  "pixel_std": float(pixels.std()), "audio_finite": audio_finite,
                  "audio_peak": max(float(np.abs(f).max()) for f in audio),
                  "audio_rms": float(np.sqrt(sum(float((f*f).sum()) for f in audio) / sum(f.size for f in audio)))}
        for index in (0, len(frames)//2, len(frames)-1):
            Image.fromarray(frames[index]).save(directory / f"frame{index}.png")
        (directory / "media.json").write_text(json.dumps(report, indent=2))
        print(directory.name, json.dumps(report), flush=True)
