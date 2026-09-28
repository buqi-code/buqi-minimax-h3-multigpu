"""Boundary checks; numerical behavior is tested by test_current_api.py."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from bootstrap import setup
setup()
import minimax_sp as node
from minimax_sp import sp_forward as spf
from comfy.ldm.modules.attention import AttentionTensorContainer


class APICompatibilityTests(unittest.TestCase):
    def test_current_api(self):
        node.check_h3_api()

    def test_old_forward_rejected(self):
        with patch.object(node, "MiniMaxH3Model", SimpleNamespace(_forward=lambda x: x)):
            with self.assertRaisesRegex(RuntimeError, "_forward is missing"):
                node.check_h3_api()

    def test_old_layout_rejected(self):
        with patch.object(node, "PackedLayout", lambda frame_count: None):
            with self.assertRaisesRegex(RuntimeError, "frame_count"):
                node.check_h3_api()

    def test_attention_without_comfy_backend_rejected(self):
        with patch.object(node, "Attention", object):
            with self.assertRaisesRegex(RuntimeError, "comfy_attention"):
                node.check_h3_api()

    def test_block_without_attention_adapter_rejected(self):
        block = SimpleNamespace(forward=lambda self, x: x)
        with patch.object(node, "DiTBlock", block):
            with self.assertRaisesRegex(RuntimeError, "does not accept attention"):
                node.check_h3_api()

    def test_old_final_layer_rejected(self):
        final = SimpleNamespace(forward=lambda self, x, t_emb, video_seg, audio_seg: x)
        with patch.object(node, "FinalLayer", final):
            with self.assertRaisesRegex(RuntimeError, "FinalLayer.forward is missing"):
                node.check_h3_api()

    def test_attention_uses_public_containers_and_preferred_backend(self):
        q = torch.randn(1, 2, 3, 4)
        k = torch.randn(1, 2, 3, 4)
        v = torch.randn(1, 2, 3, 4)
        preferred = object()

        def optimized(q_container, k_container, v_container, heads, **kwargs):
            self.assertIsInstance(q_container, AttentionTensorContainer)
            self.assertIsInstance(k_container, AttentionTensorContainer)
            self.assertIsInstance(v_container, AttentionTensorContainer)
            self.assertIs(q_container.peek(), q)
            self.assertIs(k_container.peek(), k)
            self.assertIs(v_container.peek(), v)
            self.assertEqual(heads, 2)
            self.assertIs(kwargs["preferred_attention"], preferred)
            return q_container.take()

        with patch.object(spf, "optimized_attention", side_effect=optimized):
            out = spf.run_optimized_attention(SimpleNamespace(comfy_attention=preferred), q, k, v, 2, {})
        self.assertIs(out, q)

    def test_sp_block_delegates_to_current_block_forward(self):
        class Block:
            def __init__(self):
                self.attn = object()
                self.args = None

            def __call__(self, x, t_emb, mod_segments, rope_freqs,
                         transformer_options=None, attention=None):
                self.args = (x, t_emb, mod_segments, rope_freqs, transformer_options)
                return attention(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

        block = Block()
        x, t_emb, rope = torch.randn(2, 3), torch.randn(1, 3), torch.randn(1)
        segments = [(0, 2, torch.tensor([0, 1]))]
        options = {"block_index": 0}
        ctx = object()
        expected = torch.randn(2, 3)
        with patch.object(spf, "sp_attention", return_value=expected) as attention:
            out = spf.sp_block(block, x, t_emb, segments, rope, ctx, options)
        self.assertIs(out, expected)
        for actual, original in zip(block.args, (x, t_emb, segments, rope, options)):
            self.assertIs(actual, original)
        attention.assert_called_once()
        for actual, original in zip(attention.call_args.args, (block.attn, x, rope, ctx, options)):
            self.assertIs(actual, original)

    def test_transformer_patches_rejected(self):
        for key in ("patches_replace", "patches"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                spf.validate_transformer_options({key: {"dit": object()}})

    def test_threaded_multigpu_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "threaded MultiGPU"):
            spf.validate_transformer_options({"multigpu_thread_device": torch.device("cpu")})

    def test_unsynchronized_attention_options_rejected(self):
        for key in ("optimized_attention_override", "custom_attention_mode"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, key):
                spf.validate_transformer_options({key: object()})

    def test_standard_pdd_options_and_empty_patch_maps_allowed(self):
        spf.validate_transformer_options({
            "patches_replace": {},
            "patches": {},
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
            "minimax_h3_sigma_shift_video": 7.0,
            "minimax_h3_sigma_shift_audio": 3.0,
            "prefetch_dynamic_vbars": True,
        })

    def test_auto_devices_follow_cuda_visible_devices(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=True):
            self.assertEqual(node.resolve_devices("auto", 2), ["2", "3"])

    def test_device_mapping_rejects_duplicates_and_short_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,2"}, clear=True):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                node.resolve_devices("auto", 2)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}, clear=True):
            with self.assertRaisesRegex(ValueError, "need 2"):
                node.resolve_devices("auto", 2)

    def test_explicit_device_override_uses_physical_visible_ids(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,4"}, clear=True):
            self.assertEqual(node.resolve_devices("2,4", 2), ["2", "4"])
            with self.assertRaisesRegex(ValueError, "primary visible device"):
                node.resolve_devices("3,2", 2)
            with self.assertRaisesRegex(ValueError, "physical identifiers"):
                node.resolve_devices("0,1", 2)
            with self.assertRaisesRegex(ValueError, "world_size"):
                node.resolve_devices("2", 2)

    def test_minimax_device_override_is_validated_against_visibility(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,4", "MINIMAX_SP_DEVICES": "2,4"}, clear=True):
            self.assertEqual(node.resolve_devices("auto", 2), ["2", "4"])

    def test_single_gpu_does_not_create_group(self):
        with patch.object(node.folder_paths, "get_full_path_or_raise", return_value="model.safetensors"), \
                patch.object(node.comfy.sd, "load_diffusion_model", return_value=object()), \
                patch.object(node.sp_group, "get_group") as create:
            node.MiniMaxH3SPUNETLoader.execute("model", "default", 1)
            create.assert_not_called()

    def test_deprecated_vae_node_is_official_decode_alias(self):
        vae = SimpleNamespace(decode=Mock(return_value=torch.zeros(1, 3, 2, 2)))
        samples = {"samples": torch.zeros(1, 2, 1, 1, 1)}
        with patch.object(node, "_vae_deprecation_logged", False), self.assertLogs(level="WARNING") as logs:
            node.MiniMaxH3SPVAEDecode.execute(samples, vae)
            node.MiniMaxH3SPVAEDecode.execute(samples, vae)
        self.assertEqual(vae.decode.call_count, 2)
        self.assertEqual(sum("deprecated" in message for message in logs.output), 1)


if __name__ == "__main__":
    unittest.main()
