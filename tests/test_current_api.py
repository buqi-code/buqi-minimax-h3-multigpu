"""Small CPU/Gloo parity checks for the current H3 conditioning and mask contract."""
from datetime import timedelta
import os
import unittest

import torch
import torch.distributed as dist

from bootstrap import setup
setup()
import comfy.ops
from comfy.ldm.minimax.model import MiniMaxH3Model
from minimax_sp import sp_forward as spf
from minimax_sp import sp_group as spg


class TorchrunParityNotice(unittest.TestCase):
    @unittest.skipUnless("RANK" in os.environ and "WORLD_SIZE" in os.environ,
                         "distributed parity runs via torchrun, not unittest discovery")
    def test_torchrun_entrypoint(self):
        self.skipTest("torchrun executes main() directly")


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"test_current_api.py requires exactly 2 ranks, got {world}")
    torch.manual_seed(17)
    model = MiniMaxH3Model(hidden_size=64, num_layers=2, token_refiner_num_layers=1,
                           num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=128,
                           text_dim=64, timestep_input_dim=16, time_embed_hidden_size=64,
                           time_embed_dim=32, dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast)
    model.requires_grad_(False)
    for p in model.parameters():
        p.uniform_(-0.05, 0.05)
    model.rope.inv_freq.fill_(0.01)
    x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
    t = torch.tensor([500.0])
    context = torch.randn(1, 3, 64)
    image = torch.randn(1, 24, 1, 4, 4)
    audio = torch.randn(1, 32, 2, 2)
    cases = {
        "t2v": ({}, None, None, {}),
        "i2v": ({"keyframes": [{"resolved_frame_index": 0, "latent": image}], "cond_video_latents": [image]}, None, None, {}),
        "audio_guide": ({"keyframes": [{"resolved_frame_index": 1, "audio_latent": audio}], "cond_audio_latents": [audio]}, None, None, {}),
        "reference": ({"refs": [{"kind": "image", "latent_h": 4, "latent_w": 4}, {"kind": "audio", "ref_audio_t": 2}],
                       "cond_video_latents": [image], "cond_audio_latents": [audio]}, None, None, {}),
        "masks": ({}, torch.linspace(0, 1, 32).reshape(1, 1, 2, 4, 4), torch.linspace(0, 1, 6).reshape(1, 1, 2, 3), {}),
        "pdd": ({}, None, None, {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0]),
                                  "minimax_h3_sigma_shift_video": 7.0,
                                  "minimax_h3_sigma_shift_audio": 3.0}),
    }
    for name, (payload, vm, am, options) in cases.items():
        ref = model._forward(x, t, context, transformer_options=options.copy(),
                             minimax_payload=payload, denoise_mask=vm, audio_denoise_mask=am)
        meta = spg.build_meta(x, t, context, options, payload)
        wire_payload = spg.payload_from_meta(meta, {"tags": None, "cond_video": payload.get("cond_video_latents", []),
                                                   "cond_audio": payload.get("cond_audio_latents", [])})
        transformer_options = options.copy()
        got = spf.sp_forward(model, x, t, context, transformer_options, wire_payload,
                             rank, world, None, vm, am)
        if transformer_options.get("block_index") != 1 or "minimax_h3_layout" not in transformer_options:
            raise AssertionError("SP forward did not publish the current H3 transformer options")
        if rank == 0:
            for label, actual, expected in zip(("video", "audio"), got, ref):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
                print(name, label, "PASS", "max_abs", float((actual-expected).abs().max()), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
