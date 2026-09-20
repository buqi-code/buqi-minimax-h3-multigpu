"""Local benchmark schema. Historical measurements are not new regression runs."""
import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1] / "results"


def summarize(directory):
    prompt = json.loads((directory / "prompt.json").read_text())
    inputs = prompt["i2v"]["inputs"]
    events_path = directory / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    history_path = directory / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    job_id, job = next(iter(history.items()), (None, {}))
    events = [e for e in events if job_id is None or e["data"].get("prompt_id") == job_id]
    progress = [e for e in events if e["type"] == "progress" and e["data"].get("node") == "custom"]
    intervals = [b["elapsed"] - a["elapsed"] for a, b in zip(progress, progress[1:])]
    done = [e["elapsed"] for e in events if e["type"] == "execution_success"]
    metadata_path = directory / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    gpu = {}
    if (directory / "gpu.csv").exists():
        with (directory / "gpu.csv").open() as stream:
            for row in csv.DictReader(stream):
                device = gpu.setdefault(row["index"].strip(), {"peak_mib": 0, "max_util_percent": 0})
                device["peak_mib"] = max(device["peak_mib"], int(row["memory_used_mib"]))
                device["max_util_percent"] = max(device["max_util_percent"], int(row["utilization_percent"]))
    success = job.get("status", {}).get("status_str") == "success"
    return {
        "run": directory.name, "status": "success" if success else "failed_or_incomplete",
        "evidence": "recorded_run" if metadata else "historical_local_compatibility_run",
        "model": prompt["unet"]["inputs"]["unet_name"],
        "weight_dtype": prompt["unet"]["inputs"]["weight_dtype"],
        "lora": prompt.get("lora", {}).get("inputs"),
        "resolution": [inputs["width"], inputs["height"]], "frames": inputs["length"],
        "steps": prompt["scheduler"]["inputs"]["steps"], "seed": prompt["noise"]["inputs"]["noise_seed"],
        "world_size": prompt["unet"]["inputs"]["world_size"],
        "gpu": metadata.get("system_stats", {}).get("devices", "historical: RTX 2080 Ti x2, see REPORT.md"),
        "runtime": {"total_seconds": done[-1] if done and success else None,
                    "steady_step_mean_seconds": statistics.mean(intervals) if intervals else None,
                    "step_intervals_seconds": intervals, "steps_completed": len(progress),
                    "metadata": metadata},
        "VRAM": gpu, "speedup": None, "outputs": job.get("outputs", {}),
    }


def comparable_prompt(path):
    graph = json.loads((path / "prompt.json").read_text())
    graph["unet"]["inputs"].pop("world_size")
    graph["save"]["inputs"].pop("filename_prefix")
    return graph


def compare(records, single, multi, root):
    a, b = records[single], records[multi]
    if a["world_size"] != 1 or b["world_size"] <= 1 or any(r["status"] != "success" for r in (a, b)):
        raise ValueError("speedup requires successful world_size=1 and multi-GPU runs")
    if comparable_prompt(root / single) != comparable_prompt(root / multi):
        raise ValueError("speedup requires identical prompts except world_size and output name")
    ma, mb = a["runtime"]["metadata"], b["runtime"]["metadata"]
    historical = (single, multi) == ("baseline_w1_480p", "baseline_w2_480p") and not ma and not mb
    if not historical:
        fields = ("server_argv", "warmup", "verification", "memory_mode", "source_files_sha256")
        if not ma or not mb or any(not ma.get(k) or ma.get(k) != mb.get(k) for k in fields):
            raise ValueError("speedup requires matching runtime / warmup / verification metadata")
        if ma.get("verification") == "unknown" or ma.get("warmup") != "warm":
            raise ValueError("speedup requires warm runs and explicit verification metadata")
    ta, tb = a["runtime"], b["runtime"]
    if any(not r.get(k) or r[k] <= 0 for r in (ta, tb)
           for k in ("total_seconds", "steady_step_mean_seconds")):
        raise ValueError("speedup requires positive total and steady-step measurements")
    b["speedup"] = {"baseline": single, "total": ta["total_seconds"] / tb["total_seconds"],
                    "steady_step": ta["steady_step_mean_seconds"] / tb["steady_step_mean_seconds"],
                    "conditions_source": "REPORT.md (legacy, warmed pair)" if historical else "run_metadata.json"}


def write_benchmark(root=ROOT, pairs=()):
    records = {d.name: summarize(d) for d in sorted(root.iterdir()) if d.is_dir() and (d / "prompt.json").exists()}
    for single, multi in pairs:
        compare(records, single, multi, root)
    output = {"schema_version": 1, "units": {"runtime": "seconds", "VRAM": "MiB"},
              "timing": "total includes uncached work; steady steps exclude first progress interval",
              "runs": list(records.values())}
    (root / "benchmark.json").write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, action="append", default=[], metavar=("SINGLE", "MULTI"))
    args = parser.parse_args()
    write_benchmark(pairs=args.compare)
