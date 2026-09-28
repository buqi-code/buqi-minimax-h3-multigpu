"""Release metadata checks that do not import ComfyUI or torch."""

from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def test_registry_metadata(self):
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(data["project"]["version"], "0.1.0")
        self.assertEqual(data["project"]["dependencies"], [])
        self.assertEqual(data["project"]["license"]["file"], "LICENSE")
        self.assertEqual(data["project"]["urls"]["Repository"],
                         "https://github.com/buqi-code/buqi-minimax-h3-multigpu")
        self.assertEqual(data["tool"]["comfy"]["PublisherId"], "buqi-code")
        self.assertTrue(data["tool"]["comfy"]["DisplayName"])
        compatibility = data["tool"]["minimax-sp"]["compatibility"]
        self.assertEqual(compatibility["minimum-h3-api-commit"], "8d534945")
        self.assertEqual(compatibility["tested-comfyui-commit"],
                         "7fbcfa8be9a8f47cf905ec47978b5bd754959ea7")
        self.assertEqual(compatibility["runtime-check"], "minimax_sp.check_h3_api")

    def test_registry_archive_excludes_tests_and_models(self):
        patterns = set((ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines())
        self.assertIn("tests/", patterns)
        self.assertIn("*.log", patterns)
        self.assertIn("*.safetensors", patterns)


if __name__ == "__main__":
    unittest.main()
