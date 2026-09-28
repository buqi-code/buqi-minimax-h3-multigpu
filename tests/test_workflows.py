"""Static checks for shipped UI and API workflow examples."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
API_FILES = {
    "t2v": "workflow_api_t2v_2gpu.json",
    "i2v_first": "workflow_api_2gpu.json",
    "i2v_first_last": "workflow_api_i2v_first_last_2gpu.json",
    "addguide": "workflow_api_addguide_2gpu.json",
    "r2v": "workflow_api_r2v_2gpu.json",
    "turbo": "workflow_api_turbo_lora_2gpu.json",
}


class WorkflowTests(unittest.TestCase):
    def load(self, name):
        with (EXAMPLES / name).open(encoding="utf-8") as handle:
            return json.load(handle)

    def test_api_examples_use_current_models_and_standard_vae_decode(self):
        for label, filename in API_FILES.items():
            with self.subTest(workflow=label):
                graph = self.load(filename)
                self.assertEqual(graph["unet"]["class_type"], "MiniMaxH3SPUNETLoader")
                self.assertEqual(graph["unet"]["inputs"]["weight_dtype"], "default")
                self.assertIn("int8_convrot", graph["unet"]["inputs"]["unet_name"])
                self.assertEqual(graph["vae_video"]["inputs"]["vae_name"],
                                 "minimax_h3_video_vae_int8_convrot.safetensors")
                self.assertEqual(graph["decode_video"]["class_type"], "VAEDecode")
                self.assertNotIn("example.png", json.dumps(graph))
                self.assert_references_exist(graph)

    def test_required_api_variants_are_distinct(self):
        self.assertNotIn("first_frame", self.load(API_FILES["t2v"])["conditioning"]["inputs"])
        self.assertIn("first_frame", self.load(API_FILES["i2v_first"])["conditioning"]["inputs"])
        first_last = self.load(API_FILES["i2v_first_last"])["conditioning"]["inputs"]
        self.assertIn("first_frame", first_last)
        self.assertIn("last_frame", first_last)
        self.assertEqual(self.load(API_FILES["addguide"])["guide"]["class_type"], "MiniMaxH3AddGuide")
        r2v = self.load(API_FILES["r2v"])
        self.assertEqual(r2v["conditioning"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertIn("ref_images.ref_image_0", r2v["conditioning"]["inputs"])
        self.assertNotIn("ref_image_0", r2v["conditioning"]["inputs"])
        self.assertEqual(self.load(API_FILES["turbo"])["turbo"]["class_type"],
                         "LoraLoaderModelOnly")

    def test_ui_example_is_official_template_with_only_sp_loader(self):
        workflow = self.load("workflow_ui_2gpu.json")
        nodes = workflow["definitions"]["subgraphs"][0]["nodes"]
        types = [node["type"] for node in nodes]
        self.assertIn("MiniMaxH3SPUNETLoader", types)
        self.assertNotIn("UNETLoader", types)
        self.assertIn("VAEDecode", types)
        loader = next(node for node in nodes if node["type"] == "MiniMaxH3SPUNETLoader")
        self.assertEqual(loader["widgets_values"][-2:], [2, "auto"])
        self.assertIn("int8_convrot", json.dumps(workflow))
        self.assertNotIn("example.png", json.dumps(workflow))

    def assert_references_exist(self, graph):
        for node in graph.values():
            for value in node.get("inputs", {}).values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    self.assertIn(value[0], graph)
                    self.assertIsInstance(value[1], int)


if __name__ == "__main__":
    unittest.main()
