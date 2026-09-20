"""Local API baseline using upstream's example and current official ComfyUI nodes.

Local test harness, not part of buqi-code's original implementation.
Each run records the exact prompt, events, GPU telemetry and final history.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import time
import uuid

import aiohttp
from benchmark import write_benchmark


async def main(world=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=world, required=world is None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--length", type=int, default=22)
    ap.add_argument("--lora", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--seed", type=int, default=42424242)
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--model", default="minimax_h3_fl2va_pruned_int8_convrot.safetensors")
    ap.add_argument("--warmup", choices=("cold", "warm"), default="cold")
    ap.add_argument("--verification", choices=("enabled", "disabled", "unknown"), default="unknown",
                    help="Declare server MINIMAX_SP_VERIFY setting; unknown prevents speedup claims")
    a = ap.parse_args()
    if world is not None and a.world != world:
        ap.error(f"this entry point requires world_size={world}")
    if pathlib.Path(a.name).name != a.name or a.name in (".", ".."):
        ap.error("name must be a single directory name")
    root = pathlib.Path(__file__).resolve().parents[1]
    outdir = root / "results" / a.name
    outdir.mkdir(parents=True, exist_ok=False)
    prompt = json.loads((root / "examples/workflow_api_2gpu.json").read_text())
    prompt["unet"]["inputs"].update(unet_name=a.model, world_size=a.world)
    prompt["i2v"]["inputs"].update(prompt="A red sailboat moves slowly across a calm lake at sunset. Gentle water sounds, no dialogue.",
                                  width=a.width, height=a.height, length=a.length)
    if a.image:
        prompt["loadimage"]["inputs"]["image"] = a.image
    else:
        del prompt["loadimage"]
        del prompt["i2v"]["inputs"]["first_frame"]
    prompt["scheduler"]["inputs"]["steps"] = a.steps
    prompt["noise"]["inputs"]["noise_seed"] = a.seed
    if a.lora:
        prompt["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet", 0], "lora_name": a.lora, "strength_model": 1.0}}
        prompt["scheduler"]["inputs"]["model"] = ["lora", 0]
        prompt["guider"]["inputs"]["model"] = ["lora", 0]
    prompt["save"]["inputs"] = {"video": ["createvideo", 0], "filename_prefix": "minimax_sp/" + a.name,
                                  "format": "auto", "format.codec": "auto"}
    (outdir / "prompt.json").write_text(json.dumps(prompt, indent=2))
    spec = importlib.util.find_spec("comfyui_workflow_templates_json")
    if spec is None:
        raise RuntimeError("Install the workflow templates required by your ComfyUI requirements.txt")
    official = pathlib.Path(next(iter(spec.submodule_search_locations))) / "templates/video_minimax_h3_t2v.json"
    (outdir / "provenance.json").write_text(json.dumps({
        "official_template": str(official), "sha256": hashlib.sha256(official.read_bytes()).hexdigest(),
        "note": "API graph uses the same official H3 loaders/conditioning/res_multistep/simple/AV decode chain; dimensions, seed, steps and optional LoRA are explicit in prompt.json. UI subgraph/switch nodes are resolved to this selected branch."
    }, indent=2))
    client_id = str(uuid.uuid4())
    start = time.perf_counter()
    stop = asyncio.Event()

    async def telemetry():
        with (outdir / "gpu.csv").open("w") as f:
            f.write("elapsed,index,memory_used_mib,utilization_percent\n")
            while not stop.is_set():
                p = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                                          "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
                stdout, _ = await p.communicate()
                for line in stdout.decode().splitlines():
                    f.write(f"{time.perf_counter()-start:.3f},{line}\n")
                f.flush()
                try:
                    await asyncio.wait_for(stop.wait(), 1)
                except asyncio.TimeoutError:
                    pass

    async with aiohttp.ClientSession(trust_env=False) as session:
        async with session.get(a.url + "/system_stats") as response:
            response.raise_for_status()
            system_stats = await response.json()
        argv = system_stats["system"].get("argv", [])
        source_files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((root / "minimax_sp").rglob("*.py"))}
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        (outdir / "run_metadata.json").write_text(json.dumps({
            "system_stats": system_stats, "server_argv": argv, "warmup": a.warmup,
            "verification": a.verification,
            "source_commit": revision, "source_files_sha256": source_files,
            "memory_mode": "legacy" if "--disable-dynamic-vram" in argv else "native_default",
        }, indent=2))
        async with session.ws_connect(f"{a.url}/ws?clientId={client_id}", receive_timeout=600) as ws:
            async with session.post(a.url + "/prompt", json={"prompt": prompt, "client_id": client_id}) as r:
                result = await r.json()
                if r.status != 200:
                    (outdir / "error.json").write_text(json.dumps(result, indent=2))
                    raise RuntimeError(result)
            prompt_id = result["prompt_id"]
            print("SUBMITTED", a.name, prompt_id, flush=True)
            task = asyncio.create_task(telemetry())
            with (outdir / "events.jsonl").open("w") as log:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(msg.data)
                    elapsed = time.perf_counter() - start
                    log.write(json.dumps({"elapsed": elapsed, **event}) + "\n")
                    log.flush()
                    kind, data = event["type"], event["data"]
                    if kind in ("executing", "progress", "execution_error", "execution_success"):
                        print(f"{elapsed:.2f}s", kind, json.dumps(data)[:1200], flush=True)
                    if data.get("prompt_id") == prompt_id and (kind in ("execution_error", "execution_success", "execution_interrupted")
                            or (kind == "executing" and data.get("node") is None)):
                        break
            stop.set()
            await task
            async with session.get(f"{a.url}/history/{prompt_id}") as r:
                history = await r.json()
            (outdir / "history.json").write_text(json.dumps(history, indent=2))
            write_benchmark()
            if history.get(prompt_id, {}).get("status", {}).get("status_str") != "success":
                raise RuntimeError(f"benchmark failed; inspect {outdir}")
            print("FINISHED", a.name, f"{time.perf_counter()-start:.2f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
