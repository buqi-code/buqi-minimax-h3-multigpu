"""Export only review-relevant measurements, not machine paths or full histories."""
import json
from pathlib import Path
import re

from benchmark import ROOT, write_benchmark


def main():
    benchmark = write_benchmark(pairs=[("baseline_w1_480p", "baseline_w2_480p")])
    runs = []
    for run in benchmark["runs"]:
        if not run["run"].startswith("pr_") and run["run"] not in ("baseline_w1_480p", "baseline_w2_480p"):
            continue
        metadata = run["runtime"]["metadata"]
        media_path = ROOT / run["run"] / "media.json"
        media = json.loads(media_path.read_text()) if media_path.exists() else None
        if media is not None:
            media.pop("file", None)  # Keep local absolute output paths private.
        runs.append({
            "run": run["run"], "status": run["status"], "model": run["model"],
            "resolution": run["resolution"], "frames": run["frames"], "steps": run["steps"],
            "seed": run["seed"], "world_size": run["world_size"],
            "source_commit": metadata.get("source_commit"),
            "source_files_sha256": metadata.get("source_files_sha256"),
            "lora": run["lora"]["lora_name"] if run["lora"] else None,
            "runtime_seconds": run["runtime"]["total_seconds"],
            "steady_step_seconds": run["runtime"]["steady_step_mean_seconds"],
            "VRAM": run["VRAM"], "speedup": run["speedup"], "evidence": run["evidence"],
            "decoded_media": media,
        })
    log = "\n".join(path.read_text() for path in sorted(ROOT.glob("pr_runtime*/comfyui_sp.log")))
    ack = [{"keys": int(n), "sha256": digest} for n, digest in re.findall(
        r"all ranks acknowledged (\d+) patched keys, sha256=([0-9a-f]+)", log)]
    parity = [{"stream": stream, "max_abs": float(delta), "relative": float(relative)}
              for stream, delta, relative in re.findall(
                  r"\[verify\] (video|audio) max_abs=([^ ]+) relative=([^\s]+)", log)]
    recovery_path = ROOT / "pr_runtime_recovery.json"
    unit_results = {}
    for name, filename in (("api", "pr_api_unit.log"), ("patch_sync", "pr_lora_unit_final.log"),
                           ("algorithm_preservation", "pr_algorithm_unchanged.log"),
                           ("benchmark", "pr_benchmark_unit.log")):
        text = (ROOT / filename).read_text()
        count = re.search(r"Ran (\d+) tests?", text)
        unit_results[name] = {"passed": bool(re.search(r"^OK$", text, re.M)),
                              "tests": int(count[1]) if count else None}
    parity_log = (ROOT / "pr_api_parity_final.log").read_text()
    unit_results["tiny_two_rank_parity"] = {"passed_stream_comparisons": parity_log.count(" PASS max_abs ")}
    evidence = {"schema_version": 1,
                "upstream": "buqi-code/buqi-minimax-h3-multigpu@bce083929c135bdabd41679da3ddf6f409336ed3",
                "comfyui": "99073836d45f66053c45ba8564984e6def9cebba",
                "hardware": "2 x RTX 2080 Ti 22528 MiB, SM75, NV2",
                "unit_tests": unit_results,
                "note": "pr_* runs are correctness regressions with verification overhead, not speedup benchmarks. Historical baseline is explicitly separate. Expected interrupt/dead-worker runs are not successful generations.",
                "patch_ack_sequence": ack, "native_parity_sequence": parity,
                "runtime_recovery": json.loads(recovery_path.read_text()) if recovery_path.exists() else None,
                "initial_cancellation_failure": json.loads((ROOT / "pr_cancellation_failure.json").read_text()),
                "runs": runs}
    output = Path(__file__).resolve().parents[1] / "evidence/pr_validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2))
    print(output)


if __name__ == "__main__":
    main()
