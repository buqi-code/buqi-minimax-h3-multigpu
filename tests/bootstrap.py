"""Locate an existing ComfyUI checkout without embedding a user's home path."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def setup():
    candidates = [Path(os.environ["COMFYUI_ROOT"])] if os.environ.get("COMFYUI_ROOT") else [
        ROOT.parent / "ComfyUI", ROOT.parent.parent,
    ]
    for root in candidates:
        if (root / "comfy/ldm/minimax/model.py").is_file():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(ROOT))
            return root
    raise RuntimeError("Set COMFYUI_ROOT to the existing reviewed ComfyUI checkout")
