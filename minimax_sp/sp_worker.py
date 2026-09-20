"""Worker process for the MiniMax-H3 sequence-parallel group (rank >= 1).

Launched with CUDA_VISIBLE_DEVICES=<rank>, so its single visible GPU is cuda:0 and
ComfyUI's model management needs no per-device plumbing. Holds only the DiT: no
text encoder, no VAE. Parks on a gloo broadcast waiting for work.
"""

import argparse
import json
import logging
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--unet", required=True)
    ap.add_argument("--weight-dtype", default="default")
    ap.add_argument("--comfy-root", required=True)
    ap.add_argument("--runtime-options", default="{}")
    a = ap.parse_args()

    sys.path.insert(0, a.comfy_root)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from comfy.cli_args import args, enables_dynamic_vram
    for key, value in json.loads(a.runtime_options).items():
        setattr(args, key, value)
    if enables_dynamic_vram():
        import comfy_aimdo.control
        headroom = None if args.reserve_vram is None else int(args.reserve_vram * 2**30)
        comfy_aimdo.control.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
    # rendezvous before loading 21GB of weights so the TCP init doesn't time out
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{a.port}",
                            rank=a.rank, world_size=a.world, timeout=timedelta(minutes=40),
                            device_id=torch.device("cuda", 0))
    obj_pg = dist.new_group(backend="gloo", timeout=timedelta(minutes=40))
    torch.cuda.set_device(0)

    import comfy.model_management
    import comfy.sd
    if enables_dynamic_vram():
        import comfy_aimdo.control
        import comfy.memory_management
        import comfy.model_patcher
        if not comfy_aimdo.control.init_devices([(0, int(args.vram_headroom * 2**30))]):
            raise RuntimeError("worker could not initialize native DynamicVRAM")
        # Same worker-local startup selection as main.py, not a class rewrite.
        comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
        comfy.memory_management.aimdo_enabled = True
    import folder_paths
    from minimax_sp import sp_forward as spf
    from minimax_sp import sp_group as spg
    from minimax_sp import sp_vae as spv
    from minimax_sp.sp_patches import PatchManifest, WorkerPatches

    model_options = {}
    if a.weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif a.weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif a.weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    path = a.unet if os.path.isabs(a.unet) else folder_paths.get_full_path_or_raise("diffusion_models", a.unet)
    stream = os.environ.get("MINIMAX_SP_STREAM") == "1"
    logging.info(f"rank {a.rank}: loading {a.unet} (stream_weights={stream})")
    patcher = comfy.sd.load_diffusion_model(path, model_options=model_options)
    comfy.model_management.load_models_gpu([patcher], force_full_load=not stream)
    dit = patcher.model.diffusion_model
    device = patcher.load_device
    logging.info(f"rank {a.rank}: loaded on {device}, "
                 f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")

    dist.barrier()
    logging.info(f"rank {a.rank}: ready")

    steps = 0
    vae_model = None
    patches = WorkerPatches(patcher, a.rank)
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0, group=obj_pg)
        meta = box[0]
        if meta is None or meta.get("op") == "shutdown":
            logging.info(f"rank {a.rank}: shutdown after {steps} forwards")
            break

        tensors = {}
        if meta.get("op") in ("prepare", "patches"):
            try:
                if meta["op"] == "prepare":
                    reply = patches.prepare(meta["memory_required"], meta["version"])
                else:
                    manifest = PatchManifest.from_message(meta)
                    reply = patches.apply(manifest)
                    logging.info("rank %s: applied %d patched keys, sha256=%s",
                                 a.rank, len(manifest.keys), manifest.sha256)
            except Exception as exc:
                logging.exception("worker patch preparation failed")
                reply = {"rank": a.rank, "error": str(exc)}
            dist.all_gather_object([None] * a.world, reply, group=obj_pg)
            continue
        if meta.get("op") == "vae_load":
            import comfy.ldm.minimax.vae

            vae_model = comfy.ldm.minimax.vae.MiniMaxH3VideoVAE()
            state = {}
            for name, shape, dtype in meta["manifest"]:
                t = torch.empty(shape, dtype=getattr(torch, dtype), device=device)
                dist.broadcast(t, src=0)
                state[name] = t
            vae_model.load_state_dict(state)
            vae_dtype = next(iter(state.values())).dtype if state else torch.float16
            vae_model = vae_model.to(device=device, dtype=vae_dtype).eval()
            del state
            dist.barrier()
            logging.info(f"rank {a.rank}: video VAE ready, "
                         f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")
            continue

        if meta.get("op") == "vae_decode":
            if vae_model is None:
                logging.error(f"rank {a.rank}: vae_decode before vae_load")
                break
            z = spg.alloc_from_meta(meta["z"], device)
            dist.broadcast(z, src=0)
            pad_tokens, bounds = spv.chunk_plan(vae_model, z.shape[2])
            with torch.no_grad():
                prepared = spv.prepare_latent(vae_model, z, pad_tokens)
                mine = spv.decode_local(vae_model, prepared, bounds, a.rank, a.world)
            for i in sorted(mine):
                dist.send(mine[i].contiguous(), dst=0)
            del z, prepared, mine
            continue

        patches.require_ready()
        video = spg.alloc_from_meta(meta["video"], device)
        audio = spg.alloc_from_meta(meta["audio"], device)
        timestep = spg.alloc_from_meta(meta["timestep"], device)
        context = spg.alloc_from_meta(meta["context"], device)
        denoise_mask = spg.alloc_from_meta(meta["denoise_mask"], device)
        audio_denoise_mask = spg.alloc_from_meta(meta["audio_denoise_mask"], device)
        tensors["tags"] = spg.alloc_from_meta(meta["tags"], device)
        tensors["cond_video"] = [spg.alloc_from_meta(m, device) for m in meta["cond_video"]]
        tensors["cond_audio"] = [spg.alloc_from_meta(m, device) for m in meta["cond_audio"]]
        for t in [video, audio, timestep, context, tensors["tags"],
                  *tensors["cond_video"], *tensors["cond_audio"], denoise_mask, audio_denoise_mask]:
            if t is not None:
                dist.broadcast(t, src=0)

        payload = spg.payload_from_meta(meta, tensors)
        with torch.no_grad():
            spf.sp_forward(dit, [video, audio], timestep, context, dict(meta["options"]),
                           payload, a.rank, a.world, None, denoise_mask, audio_denoise_mask)
        steps += 1

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
