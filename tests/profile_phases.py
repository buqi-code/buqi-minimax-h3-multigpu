"""Phase-level profiler: per-node wall time for a MiniMax-H3 job.

Uses the server's websocket "executing" events to attribute wall time to each
node (text encode inside MiniMaxH3ImageToVideo, SamplerCustomAdvanced denoise
loop, VAE decodes, ...). Prints a breakdown table; appends JSON lines to
--out for later analysis.
"""
import argparse
import asyncio
import json
import sys
import time
import uuid

import aiohttp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from selftest import build_prompt  # noqa: E402


async def run_one(session, a, sp, run_idx=0):
    client_id = str(uuid.uuid4())
    prompt = build_prompt(a, sp)
    # defeat the server's execution cache so every phase is really measured
    prompt["i2v"]["inputs"]["prompt"] += " " * run_idx
    marks = []  # (node_id_or_None, t)
    async with session.ws_connect(f"{a.server}/ws?clientId={client_id}") as ws:
        async with session.post(f"{a.server}/prompt", json={"prompt": prompt, "client_id": client_id}) as r:
            resp = await r.json()
        if "prompt_id" not in resp:
            raise RuntimeError(f"submit failed: {json.dumps(resp)[:500]}")
        pid = resp["prompt_id"]
        t_submit = time.perf_counter()
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            data = m.get("data", {})
            if data.get("prompt_id") != pid:
                continue
            if m.get("type") == "executing":
                node = data.get("node")
                marks.append((node, time.perf_counter()))
                if node is None:
                    break
            elif m.get("type") == "execution_error":
                raise RuntimeError(f"[sp{sp}] execution_error: {json.dumps(data)[:600]}")
    hist = await (await session.get(f"{a.server}/history/{pid}")).json()
    entry = hist.get(pid)
    if entry is None or entry.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"[sp{sp}] job failed: {json.dumps(entry)[:500] if entry else 'no history entry'}")
    classes = {nid: node["class_type"] for nid, node in prompt.items()}
    rows = []
    for i in range(len(marks) - 1):
        node, t0 = marks[i]
        t1 = marks[i + 1][1]
        if node is None:
            continue
        rows.append((classes.get(node, node), t1 - t0))
    total = marks[-1][1] - t_submit if marks else 0.0
    return rows, total


async def main_async(a):
    results = []
    async with aiohttp.ClientSession() as session:
        for run_idx, sp in enumerate(a.modes):
            rows, total = await run_one(session, a, sp, run_idx)
            print(f"\n=== sp{sp} {a.width}x{a.height} len={a.length} steps={a.steps} ===  total {total:.1f}s")
            for cls, dt in sorted(rows, key=lambda r: -r[1]):
                print(f"  {cls:<28} {dt:8.1f}s  {100*dt/total:5.1f}%")
            rec = {"sp": sp, "width": a.width, "height": a.height, "length": a.length,
                   "steps": a.steps, "total": total,
                   "nodes": {cls: dt for cls, dt in rows}}
            results.append(rec)
    if a.out:
        with open(a.out, "a") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")
        print(f"\nappended {len(results)} records to {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18188")
    ap.add_argument("--modes", default="1,2", help="comma-separated world sizes to run")
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--unet", required=True)
    ap.add_argument("--clip", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    ap.add_argument("--video-vae", default="minimax_h3_video_vae_fp16.safetensors")
    ap.add_argument("--audio-vae", default="minimax_h3_audio_vae_fp32.safetensors")
    ap.add_argument("--image", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42424242)
    ap.add_argument("--out", default="/root/profile_results.jsonl")
    a = ap.parse_args()
    a.modes = [int(x) for x in a.modes.split(",")]
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
