"""Numerical parity of the sequence-parallel DiT against the single-GPU forward.

Compares the velocity tensors the DiT actually produces, so nothing is hidden by
the VAE or by lossy video encoding. Runs the official MiniMaxH3Model._forward as
the reference on rank 0 and the sequence-parallel forward on all ranks, for both
exchange strategies, and reports max absolute deviation.

    torchrun --nproc_per_node=2 tests/latent_parity.py --unet <file.safetensors>
"""
import argparse
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ.get("COMFY_ROOT", "/root/comfy/ComfyUI"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_inputs(dit, args, device, dtype):
    torch.manual_seed(1234)
    lat_t, lat_h, lat_w = args.latent_t, args.height // 16, args.width // 16
    video = torch.randn(1, dit.latents_dim, lat_t, lat_h, lat_w,
                        dtype=torch.float32, device=device)
    audio = torch.randn(1, 32, 2, args.audio_t, dtype=torch.float32, device=device)
    context = torch.randn(1, args.text_len, 5120, dtype=dtype, device=device)
    timestep = torch.full((1,), 500.0, dtype=torch.float32, device=device)
    return [video, audio], timestep, context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--latent-t", type=int, default=32)
    ap.add_argument("--audio-t", type=int, default=216)
    ap.add_argument("--text-len", type=int, default=64)
    args = ap.parse_args()

    dist.init_process_group("nccl", timeout=timedelta(minutes=40),
                            device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    import comfy.model_management
    import comfy.sd
    import folder_paths
    from minimax_sp import sp_forward as spf

    path = folder_paths.get_full_path_or_raise("diffusion_models", args.unet)
    patcher = comfy.sd.load_diffusion_model(path)
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    dit = patcher.model.diffusion_model
    dtype = next(p for p in dit.parameters()).dtype

    x, timestep, context = build_inputs(dit, args, device, dtype)
    inner = dit.blocks[0].attn.heads * dit.blocks[0].attn.head_dim

    with torch.no_grad():
        ref = None
        if rank == 0:
            out = dit._forward(x, timestep, context, transformer_options={})
            ref = [t.float().clone() for t in out]

        results = {}
        for force_ag in (True, False):
            if force_ag and not spf.use_allgather(world, dit.hidden_size, inner):
                continue
            orig = spf.use_allgather
            spf.use_allgather = lambda *a, **k: force_ag
            try:
                got = spf.sp_forward(dit, x, timestep, context, {}, None, rank, world, None)
            finally:
                spf.use_allgather = orig
            if rank == 0:
                results["all_gather" if force_ag else "all_to_all"] = [t.float() for t in got]

    if rank == 0:
        print(f"world={world} shapes video={tuple(ref[0].shape)} audio={tuple(ref[1].shape)}")
        for name, got in results.items():
            for label, r, g in zip(("video", "audio"), ref, got):
                d = (g - r).abs()
                rel = d.max().item() / max(r.abs().max().item(), 1e-12)
                print(f"  vs 1gpu  {name:<11} {label:<5} max_abs={d.max().item():.3e} "
                      f"rel={rel:.3e} exact={torch.equal(g, r)} nonzero={int((d > 0).sum())}"
                      f"/{d.numel()}")
        if len(results) == 2:
            ag, a2a = results["all_gather"], results["all_to_all"]
            for label, g1, g2 in zip(("video", "audio"), ag, a2a):
                d = (g1 - g2).abs()
                print(f"  ag vs a2a            {label:<5} max_abs={d.max().item():.3e} "
                      f"exact={torch.equal(g1, g2)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
