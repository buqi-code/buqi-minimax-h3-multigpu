"""Numerical parity of the sharded video VAE decode against the single-GPU decode.

The sharded path spreads decode_temporal's chunks over the ranks and feeds the
results back into the official assembly. Each chunk is a pure function of a slice
of the latent, so the result should match exactly -- but the chunk runs on a
different GPU, and convolution algorithm selection can in principle differ, so
verify rather than assume.

    torchrun --nproc_per_node=2 tests/vae_parity.py --vae <file.safetensors>
"""
import argparse
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, os.environ.get("COMFY_ROOT", "/root/comfy/ComfyUI"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="minimax_h3_video_vae_fp16.safetensors")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--latent-t", type=int, default=32)
    ap.add_argument("--handover", action="store_true",
                    help="rebuild the worker VAE from rank 0's broadcast weights, as the group does")
    args = ap.parse_args()

    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", timeout=timedelta(minutes=40),
                            device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)

    import comfy.sd
    import comfy.utils
    import folder_paths
    from minimax_sp import sp_vae as spv

    if args.handover and rank != 0:
        import comfy.ldm.minimax.vae

        box = [None]
        dist.broadcast_object_list(box, src=0)
        state = {}
        for name, shape, dtype in box[0]:
            t = torch.empty(shape, dtype=getattr(torch, dtype), device=device)
            dist.broadcast(t, src=0)
            state[name] = t
        fsm = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE()
        fsm.load_state_dict(state)
        vae_dtype = next(iter(state.values())).dtype
        fsm = fsm.to(device=device, dtype=vae_dtype).eval()

        class _V:
            pass
        vae = _V()
        vae.first_stage_model = fsm
        vae.vae_dtype = vae_dtype
        print(f"rank {rank}: rebuilt VAE from handover, dtype {vae_dtype}")
    else:
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("vae", args.vae)))
        fsm = vae.first_stage_model.to(device=device, dtype=vae.vae_dtype).eval()
        print(f"rank {rank}: vae dtype {vae.vae_dtype}, loaded from file")
        if args.handover:
            sd = fsm.state_dict()
            manifest = [(k, list(v.shape), str(v.dtype).replace("torch.", "")) for k, v in sd.items()]
            dist.broadcast_object_list([manifest], src=0)
            for k, _, _ in manifest:
                dist.broadcast(sd[k].to(device).contiguous(), src=0)

    torch.manual_seed(4321)
    z = torch.randn(1, 24, args.latent_t, args.height // 16, args.width // 16,
                    dtype=vae.vae_dtype, device=device)

    pad_tokens, bounds = spv.chunk_plan(fsm, z.shape[2])
    with torch.no_grad():
        prepared = spv.prepare_latent(fsm, z, pad_tokens)
        mine = spv.decode_local(fsm, prepared, bounds, rank, world)
        # rank 0 decodes every chunk itself; the workers ship theirs over. Comparing
        # the two answers is what actually tests cross-GPU determinism.
        if rank == 0:
            local_all = {i: fsm._adaptive_decode(prepared[:, :, t0:t1])
                         for i, (t0, t1) in enumerate(bounds)}

    if rank == 0:
        print(f"world={world} latent_t={z.shape[2]} chunks={len(bounds)} pad_tokens={pad_tokens}")
        worst = 0.0
        nbad = 0
        for i, (t0, t1) in enumerate(bounds):
            owner = i % world
            ref = local_all[i]
            if owner == 0:
                got = mine[i]
            else:
                got = torch.empty(spv.chunk_shape(fsm, prepared, t0, t1),
                                  dtype=ref.dtype, device=device)
                dist.recv(got, src=owner)
            d = (got.float() - ref.float()).abs()
            same = torch.equal(got, ref)
            nbad += 0 if same else 1
            worst = max(worst, d.max().item())
            print(f"  chunk {i} (rank {owner}): exact={same} max_abs={d.max().item():.3e} "
                  f"differing={int((d > 0).sum())}/{d.numel()}")
        print(f"worst max_abs across chunks: {worst:.3e}")
        print("PASS: chunks decoded on other GPUs are bit-identical" if nbad == 0
              else f"FAIL: {nbad} chunks differ from rank 0's own decode")
    else:
        for i in sorted(mine):
            dist.send(mine[i].contiguous(), dst=0)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
