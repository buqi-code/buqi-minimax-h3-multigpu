"""Worker process for the MiniMax-H3 sequence-parallel group (rank >= 1).

Launched with CUDA_VISIBLE_DEVICES=<rank>, so its single visible GPU is cuda:0 and
ComfyUI's model management needs no per-device plumbing. Holds only the DiT: no
text encoder, no VAE. Parks on a gloo broadcast waiting for work.
"""

import argparse
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
    a = ap.parse_args()

    sys.path.insert(0, a.comfy_root)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # rendezvous before loading 21GB of weights so the TCP init doesn't time out
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{a.port}",
                            rank=a.rank, world_size=a.world, timeout=timedelta(minutes=40),
                            device_id=torch.device("cuda", 0))
    obj_pg = dist.new_group(backend="gloo", timeout=timedelta(minutes=40))
    torch.cuda.set_device(0)

    import comfy.model_management
    import comfy.sd
    import folder_paths
    from minimax_sp import sp_forward as spf
    from minimax_sp import sp_group as spg

    model_options = {}
    if a.weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif a.weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif a.weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    path = folder_paths.get_full_path_or_raise("diffusion_models", a.unet)
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
    while True:
        box = [None]
        dist.broadcast_object_list(box, src=0, group=obj_pg)
        meta = box[0]
        if meta is None or meta.get("op") == "shutdown":
            logging.info(f"rank {a.rank}: shutdown after {steps} forwards")
            break

        tensors = {}
        video = spg.alloc_from_meta(meta["video"], device)
        audio = spg.alloc_from_meta(meta["audio"], device)
        timestep = spg.alloc_from_meta(meta["timestep"], device)
        context = spg.alloc_from_meta(meta["context"], device)
        tensors["tags"] = spg.alloc_from_meta(meta["tags"], device)
        tensors["cond_video"] = [spg.alloc_from_meta(m, device) for m in meta["cond_video"]]
        tensors["cond_audio"] = [spg.alloc_from_meta(m, device) for m in meta["cond_audio"]]
        for t in [video, audio, timestep, context, tensors["tags"],
                  *tensors["cond_video"], *tensors["cond_audio"]]:
            if t is not None:
                dist.broadcast(t, src=0)

        payload = spg.payload_from_meta(meta, tensors)
        with torch.no_grad():
            spf.sp_forward(dit, [video, audio], timestep, context, dict(meta["options"]),
                           payload, a.rank, a.world, None)
        steps += 1

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
