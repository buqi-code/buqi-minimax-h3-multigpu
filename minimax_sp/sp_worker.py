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


def worker_forward(meta, previous_operation_id, rank, world, device, control_pg, tensor_pg,
                   dit, patches, spg, spf):
    expected = spg.forward_status(meta)
    tensors = {}
    try:
        spg.validate_operation_id(meta["operation_id"], previous_operation_id)
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
        tensors["sample_sigmas"] = spg.alloc_from_meta(meta["sample_sigmas"], device)
        local_status = expected
    except Exception as exc:
        local_status = spg.forward_status(meta, ready=False, error=exc)

    replies = [None] * world
    dist.all_gather_object(replies, local_status, group=control_pg)
    spg.validate_forward_status(expected, replies)

    for tensor in [video, audio, timestep, context, tensors["tags"],
                   *tensors["cond_video"], *tensors["cond_audio"], tensors["sample_sigmas"],
                   denoise_mask, audio_denoise_mask]:
        if tensor is not None:
            dist.broadcast(tensor, src=0, group=tensor_pg)

    payload = spg.payload_from_meta(meta, tensors)
    options = dict(meta["options"])
    if tensors["sample_sigmas"] is not None:
        options["sample_sigmas"] = tensors["sample_sigmas"]
    with torch.no_grad():
        spf.sp_forward(dit, [video, audio], timestep, context, options,
                       payload, rank, world, tensor_pg, denoise_mask, audio_denoise_mask)
    return meta["operation_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--unet", required=True)
    ap.add_argument("--weight-dtype", default="default")
    ap.add_argument("--comfy-root", required=True)
    ap.add_argument("--startup-timeout", type=int, required=True)
    ap.add_argument("--collective-timeout", type=int, required=True)
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

    store = None
    control_pg = None
    tensor_pg = None
    owns_default_pg = False
    try:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        store = dist.TCPStore("127.0.0.1", a.port, a.world, False,
                              timeout=timedelta(seconds=a.startup_timeout), wait_for_workers=False)
        dist.init_process_group("gloo", store=store, rank=a.rank, world_size=a.world,
                                timeout=timedelta(seconds=a.startup_timeout))
        owns_default_pg = True
        control_pg = dist.group.WORLD
        torch.cuda.set_device(0)
        tensor_pg = dist.new_group(backend="nccl", timeout=timedelta(seconds=a.collective_timeout))

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

        loaded = [None] * a.world
        dist.all_gather_object(loaded, {"loaded": True}, group=control_pg)
        if any(reply != {"loaded": True} for reply in loaded):
            raise RuntimeError(f"MiniMax SP worker load failed: {loaded}")
        spg.startup_probe(a.rank, a.world, tensor_pg, device)
        logging.info(f"rank {a.rank}: ready")

        steps = 0
        previous_operation_id = 0
        patches = WorkerPatches(patcher, a.rank)
        while True:
            box = [None]
            dist.broadcast_object_list(box, src=0, group=control_pg)
            meta = box[0]
            if meta is None or meta.get("op") == "shutdown":
                logging.info(f"rank {a.rank}: shutdown after {steps} forwards")
                break

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
                    logging.error("worker patch preparation failed: %s", exc)
                    reply = {"rank": a.rank, "error": str(exc)}
                dist.all_gather_object([None] * a.world, reply, group=control_pg)
                continue
            previous_operation_id = worker_forward(
                meta, previous_operation_id, a.rank, a.world, device, control_pg, tensor_pg,
                dit, patches, spg, spf)
            steps += 1
    finally:
        initialized = dist.is_initialized()
        try:
            if initialized and tensor_pg is not None:
                dist.destroy_process_group(tensor_pg)
        finally:
            tensor_pg = None
            try:
                if initialized and owns_default_pg and control_pg is not None:
                    dist.destroy_process_group(control_pg)
            finally:
                control_pg = None
                store = None


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("MiniMax SP worker failed: %s", exc)
        raise SystemExit(1)
