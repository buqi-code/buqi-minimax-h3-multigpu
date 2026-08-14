"""Parity self-test: verify the multi-GPU path is bit-identical to single-GPU.

Submits the same small job twice to a running ComfyUI server -- once through the
stock UNETLoader (world size 1) and once through MiniMaxH3SPUNETLoader -- and
compares the SHA-256 of the produced videos. Identical hashes prove the Ulysses
sequence-parallel path changes no numbers.

Usage (server must already be running with all SP GPUs visible):
    python tests/selftest.py --server http://127.0.0.1:18188 --sp 2 \
        --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
        --image your_first_frame.png

Requires aiohttp. Exits 0 on bit-identical output.
"""
import argparse
import asyncio
import hashlib
import json
import uuid

import aiohttp

PROMPT_TEXT = (
    "A transparent RGB gaming mouse sits on a dark desk mat. The internal RGB lighting "
    "pulses slowly from cyan to magenta while the camera performs a smooth slow orbit "
    "around it. Ambient hum of a quiet room with soft mechanical clicks."
)


def build_prompt(a, sp):
    if sp > 1:
        unet = {"class_type": "MiniMaxH3SPUNETLoader", "inputs": {
            "unet_name": a.unet, "weight_dtype": "default", "world_size": sp, "devices": a.devices}}
    else:
        unet = {"class_type": "UNETLoader", "inputs": {"unet_name": a.unet, "weight_dtype": "default"}}
    return {
        "unet": unet,
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": a.clip, "type": "minimax", "device": "default"}},
        "vae_video": {"class_type": "VAELoader", "inputs": {"vae_name": a.video_vae}},
        "vae_audio": {"class_type": "VAELoader", "inputs": {"vae_name": a.audio_vae}},
        "loadimage": {"class_type": "LoadImage", "inputs": {"image": a.image}},
        "i2v": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae_video", 0], "prompt": PROMPT_TEXT,
            "width": a.width, "height": a.height, "length": 124, "first_frame": ["loadimage", 0]}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["unet", 0], "conditioning": ["i2v", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "scheduler": {"class_type": "BasicScheduler", "inputs": {
            "model": ["unet", 0], "scheduler": "simple", "steps": a.steps, "denoise": 1.0}},
        "custom": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": ["scheduler", 0], "latent_image": ["i2v", 1]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["custom", 0], "vae": ["vae_video", 0]}},
        "decode_audio": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["custom", 0], "vae": ["vae_audio", 0]}},
        "createvideo": {"class_type": "CreateVideo", "inputs": {
            "images": ["decode", 0], "audio": ["decode_audio", 0], "fps": 24.0, "bit_depth": 8}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["createvideo", 0], "filename_prefix": f"video/selftest_sp{sp}",
            "format": "auto", "codec": "auto"}},
    }


async def run_one(session, a, sp):
    client_id = str(uuid.uuid4())
    prompt = build_prompt(a, sp)
    async with session.ws_connect(f"{a.server}/ws?clientId={client_id}") as ws:
        async with session.post(f"{a.server}/prompt", json={"prompt": prompt, "client_id": client_id}) as r:
            resp = await r.json()
        if "prompt_id" not in resp:
            raise RuntimeError(f"submit failed: {json.dumps(resp)[:500]}")
        pid = resp["prompt_id"]
        print(f"[sp{sp}] queued {pid}", flush=True)
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            data = m.get("data", {})
            if data.get("prompt_id") != pid:
                continue
            if m.get("type") == "executing" and data.get("node") is None:
                break
            if m.get("type") == "execution_error":
                raise RuntimeError(f"[sp{sp}] execution_error: {json.dumps(data)[:600]}")
        hist = await (await session.get(f"{a.server}/history/{pid}")).json()
        entry = hist.get(pid)
        if entry is None or entry.get("status", {}).get("status_str") != "success":
            raise RuntimeError(f"[sp{sp}] job failed: {json.dumps(entry)[:500] if entry else 'no history entry'}")
        out = entry["outputs"]["save"]
        files = out.get("videos") or out.get("images")
        if not files:
            raise RuntimeError(f"[sp{sp}] SaveVideo produced no file: {json.dumps(out)[:300]}")
        fn = files[0]
        params = {"filename": fn["filename"], "subfolder": fn.get("subfolder", ""), "type": fn.get("type", "output")}
        blob = await (await session.get(f"{a.server}/view", params=params)).read()
        return hashlib.sha256(blob).hexdigest(), len(blob)


async def main_async(a):
    async with aiohttp.ClientSession() as session:
        h1, n1 = await run_one(session, a, 1)
        print(f"[sp1] sha256={h1} ({n1} bytes)")
        h2, n2 = await run_one(session, a, a.sp)
        print(f"[sp{a.sp}] sha256={h2} ({n2} bytes)")
    if h1 == h2:
        print(f"PASS: sp{a.sp} output is bit-identical to single-GPU")
        return 0
    print(f"FAIL: hashes differ\n  sp1  {h1}\n  sp{a.sp}  {h2}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18188")
    ap.add_argument("--sp", type=int, default=2)
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--unet", required=True)
    ap.add_argument("--clip", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    ap.add_argument("--video-vae", default="minimax_h3_video_vae_fp16.safetensors")
    ap.add_argument("--audio-vae", default="minimax_h3_audio_vae_fp32.safetensors")
    ap.add_argument("--image", required=True, help="first-frame image already in ComfyUI/input")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42424242)
    a = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(a)))


if __name__ == "__main__":
    main()
