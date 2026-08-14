"""Rank-0 side of the MiniMax-H3 Ulysses sequence-parallel group.

Rank 0 is the ComfyUI process itself; ranks 1..P-1 are thin worker processes that
hold only the DiT and do nothing but sequence-parallel forwards. Each worker gets
CUDA_VISIBLE_DEVICES=<rank> so ComfyUI's own device plumbing stays untouched.

Per step, rank 0 broadcasts a small metadata dict over a gloo subgroup and the
input tensors over NCCL, then all ranks run the identical sp_forward. State is
sent every step on purpose: no cross-step caching means no cache-invalidation
class of bugs.
"""

import atexit
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from . import sp_forward as spf

REF_META_KEYS = ("kind", "latent_h", "latent_w", "latent_t", "ref_audio_t")
TO_WORKER_OPTIONS = ("minimax_h3_sigma_shift_video", "minimax_h3_sigma_shift_audio")

_GROUPS = {}


def tensor_meta(t):
    if t is None:
        return None
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}


def alloc_from_meta(meta, device):
    if meta is None:
        return None
    return torch.empty(meta["shape"], dtype=getattr(torch, meta["dtype"]), device=device)


def build_meta(x, timestep, context, transformer_options, payload):
    payload = payload or {}
    cond_v = payload.get("cond_video_latents", []) or []
    cond_a = payload.get("cond_audio_latents", []) or []
    tags = payload.get("text_token_tags")
    return {
        "op": "forward",
        "video": tensor_meta(x[0]),
        "audio": tensor_meta(x[1]),
        "timestep": tensor_meta(timestep),
        "context": tensor_meta(context),
        "tags": tensor_meta(tags),
        "cond_video": [tensor_meta(t) for t in cond_v],
        "cond_audio": [tensor_meta(t) for t in cond_a],
        "keyframes": [{"resolved_frame_index": kf["resolved_frame_index"]}
                      for kf in (payload.get("keyframes") or [])] or None,
        "refs": [{k: r[k] for k in REF_META_KEYS if k in r}
                 for r in (payload.get("refs") or [])] or None,
        "frame_count": payload.get("frame_count"),
        "seed": payload.get("seed", 0),
        "visual_cond_noise_aug": payload.get("visual_cond_noise_aug", spf.VISUAL_COND_TIMESTEP),
        "audio_cond_noise_aug": payload.get("audio_cond_noise_aug", spf.AUDIO_COND_TIMESTEP),
        "options": {k: transformer_options[k] for k in TO_WORKER_OPTIONS if k in transformer_options},
    }


def payload_from_meta(meta, tensors):
    p = {
        "seed": meta["seed"],
        "visual_cond_noise_aug": meta["visual_cond_noise_aug"],
        "audio_cond_noise_aug": meta["audio_cond_noise_aug"],
    }
    if meta["keyframes"]:
        p["keyframes"] = meta["keyframes"]
    if meta["refs"]:
        p["refs"] = meta["refs"]
    if meta["frame_count"] is not None:
        p["frame_count"] = meta["frame_count"]
    if tensors["tags"] is not None:
        p["text_token_tags"] = tensors["tags"]
    if tensors["cond_video"]:
        p["cond_video_latents"] = tensors["cond_video"]
    if tensors["cond_audio"]:
        p["cond_audio_latents"] = tensors["cond_audio"]
    return p


def ordered_tensors(x, timestep, context, payload):
    payload = payload or {}
    out = [x[0], x[1], timestep, context, payload.get("text_token_tags")]
    out += list(payload.get("cond_video_latents", []) or [])
    out += list(payload.get("cond_audio_latents", []) or [])
    return out


class SPGroup:
    def __init__(self, world, unet_name, weight_dtype, devices, port=29511):
        self.world = world
        self.unet_name = unet_name
        self.procs = []
        self.log_paths = {}
        self.n_forward = 0
        comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp_worker.py")
        log_dir = os.environ.get("MINIMAX_SP_LOGDIR") or tempfile.gettempdir()
        for r in range(1, world):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = devices[r]
            env["PYTHONPATH"] = comfy_root + os.pathsep + env.get("PYTHONPATH", "")
            log_path = os.path.join(log_dir, f"minimax_sp_worker{r}.log")
            self.log_paths[r] = log_path
            log = open(log_path, "w")
            self.procs.append(subprocess.Popen(
                [sys.executable, script, "--rank", str(r), "--world", str(world),
                 "--port", str(port), "--unet", unet_name, "--weight-dtype", weight_dtype,
                 "--comfy-root", comfy_root],
                env=env, stdout=log, stderr=log, cwd=comfy_root))
            logging.info(f"[minimax_sp] spawned worker rank {r} on physical GPU {devices[r]}")

        if not dist.is_initialized():
            dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                                    rank=0, world_size=world, timeout=timedelta(minutes=40),
                                    device_id=torch.device("cuda", 0))
        self.obj_pg = dist.new_group(backend="gloo", timeout=timedelta(minutes=40))
        logging.info("[minimax_sp] process group up, waiting for workers to load weights...")
        dist.barrier()
        logging.info(f"[minimax_sp] all {world} ranks ready")
        atexit.register(self.shutdown)

    def check_alive(self):
        for r, p in enumerate(self.procs, start=1):
            if p.poll() is not None:
                raise RuntimeError(f"[minimax_sp] worker rank {r} died (exit {p.returncode}), "
                                   f"see {self.log_paths[r]}")

    def forward(self, dit, x, timestep, context, transformer_options, payload):
        self.check_alive()
        profile = os.environ.get("MINIMAX_SP_PROFILE")
        try:
            t_disp = time.perf_counter()
            meta = build_meta(x, timestep, context, transformer_options, payload)
            dist.broadcast_object_list([meta], src=0, group=self.obj_pg)
            t_meta = time.perf_counter()
            device = x[0].device
            for t in ordered_tensors(x, timestep, context, payload):
                if t is not None:
                    dist.broadcast(t.to(device).contiguous(), src=0)
            if profile:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = spf.sp_forward(dit, x, timestep, context, transformer_options, payload,
                                 0, self.world, None)
            if profile:
                torch.cuda.synchronize()
                logging.info(f"[minimax_sp][profile] step {self.n_forward}: "
                             f"meta_bcast={1000 * (t_meta - t_disp):.1f}ms "
                             f"tensor_bcast={1000 * (t0 - t_meta):.1f}ms "
                             f"sp_forward={1000 * (time.perf_counter() - t0):.1f}ms")
                self.n_forward += 1
            return out
        except Exception:
            # ranks are desynchronized past this point; drop the group so the next
            # prompt respawns a clean one instead of hanging on a collective
            logging.exception("[minimax_sp] forward failed, tearing down the group")
            self.destroy()
            raise

    def destroy(self):
        self.shutdown()
        for key, group in list(_GROUPS.items()):
            if group is self:
                del _GROUPS[key]

    def shutdown(self):
        if not self.procs:
            return
        try:
            dist.broadcast_object_list([{"op": "shutdown"}], src=0, group=self.obj_pg)
        except Exception:
            pass
        for p in self.procs:
            try:
                p.wait(timeout=15)
            except Exception:
                p.kill()
        self.procs = []
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def get_group(world, unet_name, weight_dtype, devices=None):
    key = (world, unet_name, weight_dtype, tuple(devices) if devices else None)
    if key not in _GROUPS:
        if _GROUPS:
            raise RuntimeError("[minimax_sp] a different SP group is already running; restart ComfyUI "
                               f"to change it (active: {list(_GROUPS)[0]})")
        if devices is None:
            raise ValueError("[minimax_sp] devices must be provided for a new SP group")
        _GROUPS[key] = SPGroup(world, unet_name, weight_dtype, devices)
    return _GROUPS[key]
