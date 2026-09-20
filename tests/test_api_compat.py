"""Boundary checks; numerical behavior is tested by test_current_api.py."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bootstrap import setup
setup()
import minimax_sp as node


class APICompatibilityTests(unittest.TestCase):
    def test_current_api(self):
        node.check_h3_api()

    def test_old_forward_rejected(self):
        with patch.object(node, "MiniMaxH3Model", SimpleNamespace(_forward=lambda x: x)):
            with self.assertRaisesRegex(RuntimeError, "denoise masks"):
                node.check_h3_api()

    def test_old_layout_rejected(self):
        with patch.object(node, "PackedLayout", lambda frame_count: None):
            with self.assertRaisesRegex(RuntimeError, "PackedLayout"):
                node.check_h3_api()

    def test_unverified_revision_warns_instead_of_hash_lock(self):
        with patch.object(node.subprocess, "check_output", return_value="another-commit\n"):
            with self.assertLogs(level="WARNING") as logs:
                node.check_h3_api()
        self.assertIn("unverified", " ".join(logs.output))

    def test_single_gpu_does_not_create_group(self):
        with patch.object(node.folder_paths, "get_full_path_or_raise", return_value="model.safetensors"), \
                patch.object(node.comfy.sd, "load_diffusion_model", return_value=object()), \
                patch.object(node.sp_group, "get_group") as create:
            node.MiniMaxH3SPUNETLoader.execute("model", "default", 1)
            create.assert_not_called()

    def test_allgather_rejected_before_model_load(self):
        with patch.dict(os.environ, {"MINIMAX_SP_EXCHANGE": "allgather"}), \
                patch.object(node.comfy.sd, "load_diffusion_model") as load:
            with self.assertRaisesRegex(RuntimeError, "head-sliced"):
                node.MiniMaxH3SPUNETLoader.execute("model", "default", 2)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
