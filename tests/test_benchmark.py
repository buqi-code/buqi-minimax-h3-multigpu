"""No GPU needed: prevent invalid comparisons from becoming speedup claims."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from benchmark import compare


class BenchmarkTests(unittest.TestCase):
    def records(self):
        metadata = {"server_argv": ["main.py"], "warmup": "warm", "verification": "enabled",
                    "memory_mode": "native_default", "source_files_sha256": {"sp.py": "hash"}}
        single = {"world_size": 1, "status": "success", "runtime": {
            "metadata": metadata, "total_seconds": 20, "steady_step_mean_seconds": 2}}
        multi = copy.deepcopy(single)
        multi["world_size"] = 2
        multi["runtime"]["total_seconds"] = 10
        multi["runtime"]["steady_step_mean_seconds"] = 1
        return {"single": single, "multi": multi}

    def test_matching_pair(self):
        records = self.records()
        with patch("benchmark.comparable_prompt", return_value={}):
            compare(records, "single", "multi", Path("results"))
        self.assertEqual(records["multi"]["speedup"]["total"], 2)

    def test_different_runtime_rejected(self):
        for key, value in (("warmup", "cold"), ("verification", "disabled"),
                           ("memory_mode", "legacy"), ("source_files_sha256", {"sp.py": "changed"})):
            records = self.records()
            records["multi"]["runtime"]["metadata"][key] = value
            with patch("benchmark.comparable_prompt", return_value={}):
                with self.assertRaises(ValueError):
                    compare(records, "single", "multi", Path("results"))

    def test_failed_run_rejected(self):
        records = self.records()
        records["multi"]["status"] = "failed_or_incomplete"
        with self.assertRaises(ValueError):
            compare(records, "single", "multi", Path("results"))

    def test_missing_provenance_rejected_on_both_runs(self):
        records = self.records()
        for record in records.values():
            del record["runtime"]["metadata"]["source_files_sha256"]
        with patch("benchmark.comparable_prompt", return_value={}):
            with self.assertRaises(ValueError):
                compare(records, "single", "multi", Path("results"))

    def test_missing_step_measurements_rejected(self):
        records = self.records()
        records["multi"]["runtime"]["steady_step_mean_seconds"] = None
        with patch("benchmark.comparable_prompt", return_value={}):
            with self.assertRaises(ValueError):
                compare(records, "single", "multi", Path("results"))


if __name__ == "__main__":
    unittest.main()
