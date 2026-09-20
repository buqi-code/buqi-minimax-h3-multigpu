"""Destructive-to-test-worker checks on an otherwise idle, dedicated test server.

Does not kill the server. Requires its explicit PID and resolves/verifies its
unique worker child before the kill. Never use against a shared ComfyUI instance.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import shutil
import time
import uuid

import aiohttp


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--dedicated-test-server", action="store_true", required=True)
    parser.add_argument("--prefix", default="pr")
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "results"
    template = json.loads(args.template.read_text())
    run_index = 0
    seed_offset = uuid.uuid4().int % 1_000_000_000

    async with aiohttp.ClientSession(trust_env=False) as session:
        async def queue():
            async with session.get(args.url + "/queue") as response:
                response.raise_for_status()
                return await response.json()

        async def idle():
            state = await queue()
            if state["queue_running"] or state["queue_pending"]:
                raise RuntimeError("dedicated test server must have an empty queue")

        async def run(name, interrupt=False, expect="success"):
            nonlocal run_index
            await idle()
            run_index += 1
            name = args.prefix + "_" + name.removeprefix("pr_")
            directory = root / name
            directory.mkdir(exist_ok=False)
            prompt = json.loads(json.dumps(template))
            prompt["noise"]["inputs"]["noise_seed"] += seed_offset + run_index
            prompt["scheduler"]["inputs"]["steps"] = 20 if interrupt else 4
            prompt["save"]["inputs"]["filename_prefix"] = "minimax_sp/" + name
            (directory / "prompt.json").write_text(json.dumps(prompt, indent=2))
            metadata = json.loads(args.template.with_name("run_metadata.json").read_text())
            metadata["purpose"] = "controlled_runtime_recovery"
            (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
            client = str(uuid.uuid4())
            start = time.perf_counter()
            requested = False
            events = []
            async with session.ws_connect(f"{args.url}/ws?clientId={client}", receive_timeout=180) as ws:
                async with session.post(args.url + "/prompt", json={"prompt": prompt, "client_id": client}) as response:
                    response.raise_for_status()
                    prompt_id = (await response.json())["prompt_id"]
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    if event["data"].get("prompt_id") != prompt_id:
                        continue
                    events.append({"elapsed": time.perf_counter() - start, **event})
                    with (directory / "events.jsonl").open("a") as stream:
                        stream.write(json.dumps(events[-1]) + "\n")
                    if interrupt and not requested and event["type"] == "progress":
                        state = await queue()
                        if state["queue_pending"] or any(item[1] != prompt_id for item in state["queue_running"]):
                            raise RuntimeError("another prompt appeared; refusing to interrupt shared work")
                        async with session.post(args.url + "/interrupt", json={"prompt_id": prompt_id}) as response:
                            response.raise_for_status()
                        requested = True
                    if event["type"] in ("execution_success", "execution_error", "execution_interrupted"):
                        break
            # Terminal event can precede history publication by one event-loop tick.
            history = {}
            for _ in range(50):
                async with session.get(f"{args.url}/history/{prompt_id}") as response:
                    history = await response.json()
                if prompt_id in history:
                    break
                await asyncio.sleep(0.1)
            (directory / "history.json").write_text(json.dumps(history, indent=2))
            terminal = events[-1]["type"] if events else None
            wanted = {"success": "execution_success", "error": "execution_error", "interrupt": "execution_interrupted"}[expect]
            if terminal != wanted or prompt_id not in history:
                raise RuntimeError(f"{name}: expected {wanted}, got {terminal}; see {directory}")
            if expect == "success" and not any(event["type"] == "progress" for event in events):
                raise RuntimeError(f"{name}: no denoise progress; cached outputs cannot prove recovery")
            print(name, "PASS", terminal, flush=True)
            return {"run": name, "terminal": terminal, "seconds": time.perf_counter() - start}

        records = [await run("pr_interrupt", interrupt=True, expect="interrupt"),
                   await run("pr_after_interrupt")]
        await idle()
        # Cancellation may have discarded the original worker. Resolve the unique
        # current worker child of this exact server before the intentional kill.
        # ComfyUI spawns from its execution thread, not necessarily the main TID.
        children = set()
        for child_list in Path(f"/proc/{args.server_pid}/task").glob("*/children"):
            children.update(child_list.read_text().split())
        candidates = [pid for pid in children if any(part.endswith(b"/sp_worker.py") for part in
                      Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0"))]
        if len(candidates) != 1:
            raise RuntimeError("expected one current worker child of the dedicated server")
        worker_pid = int(candidates[0])
        process = Path(f"/proc/{worker_pid}")
        argv = (process / "cmdline").read_bytes().split(b"\0")
        if not any(part.endswith(b"/sp_worker.py") for part in argv):
            raise RuntimeError("target PID is not a MiniMax SP worker")
        parent = next(line.split()[1] for line in (process / "status").read_text().splitlines() if line.startswith("PPid:"))
        if int(parent) != args.server_pid:
            raise RuntimeError("worker PID does not belong to the specified test server")
        worker_log = args.log_dir / "minimax_sp_worker1.log"
        if worker_log.exists():
            shutil.copy2(worker_log, args.log_dir / "worker_before_restart.log")
        os.kill(worker_pid, signal.SIGTERM)
        await asyncio.sleep(0.5)
        records.append(await run("pr_dead_worker", expect="error"))
        records.append(await run("pr_after_worker_restart"))
        (root / "pr_runtime_recovery.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
