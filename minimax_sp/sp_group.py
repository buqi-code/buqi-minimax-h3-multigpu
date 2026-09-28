"""Rank-0 side of the MiniMax-H3 Ulysses sequence-parallel group.

Rank 0 is the ComfyUI process itself; ranks 1..P-1 are thin worker processes that
hold only the DiT and do nothing but sequence-parallel forwards. Each worker gets
CUDA_VISIBLE_DEVICES=<rank> so ComfyUI's own device plumbing stays untouched.

Per step, rank 0 broadcasts a small metadata dict over the Gloo control group and
the input tensors over a dedicated NCCL group, then all ranks run the identical
sp_forward. State is sent every step on purpose: no cross-step caching means no
cache-invalidation class of bugs.
"""

import atexit
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta

import torch
import torch.distributed as dist

import comfy.model_management
from comfy.cli_args import args as comfy_args
import folder_paths

from . import sp_forward as spf
from . import sp_patches as patch_sync

REF_META_KEYS = ("kind", "latent_h", "latent_w", "latent_t", "ref_audio_t")
TO_WORKER_OPTIONS = ("minimax_h3_sigma_shift_video", "minimax_h3_sigma_shift_audio", "prefetch_dynamic_vbars")

_GROUPS = {}


def positive_timeout(name, default):
    value = os.environ.get(name, str(default))
    try:
        seconds = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer number of seconds, got {value!r}") from exc
    if seconds <= 0:
        raise ValueError(f"{name} must be a positive integer number of seconds, got {value!r}")
    return timedelta(seconds=seconds)


def tensor_meta(t):
    if t is None:
        return None
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}


def forward_tensor_metadata(meta):
    tensors = [meta["video"], meta["audio"], meta["timestep"], meta["context"], meta["tags"]]
    tensors += meta["cond_video"]
    tensors += meta["cond_audio"]
    tensors += [meta["sample_sigmas"], meta["denoise_mask"], meta["audio_denoise_mask"]]
    return tensors


def forward_status(meta, ready=True, error=None):
    metadata = forward_tensor_metadata(meta)
    status = {
        "ready": ready,
        "operation_id": meta["operation_id"],
        "tensor_count": sum(item is not None for item in metadata),
        "tensor_metadata": metadata,
    }
    if error is not None:
        status["error"] = str(error)
    return status


def validate_forward_status(expected, replies):
    problems = []
    for rank, reply in enumerate(replies):
        if not isinstance(reply, dict):
            problems.append(f"rank {rank} returned {reply!r}")
            continue
        if reply.get("ready") is not True:
            problems.append(f"rank {rank} is not ready: {reply.get('error', reply)!r}")
            continue
        for key in ("operation_id", "tensor_count", "tensor_metadata"):
            if reply.get(key) != expected[key]:
                problems.append(f"rank {rank} {key} differs")
    if problems:
        raise RuntimeError("MiniMax SP forward control-plane mismatch before tensor collectives: " + "; ".join(problems))


def validate_operation_id(operation_id, previous):
    expected = previous + 1
    if operation_id != expected:
        raise RuntimeError(f"MiniMax SP worker expected operation_id {expected}, got {operation_id}")


def startup_probe(rank, world, tensor_pg, device):
    try:
        reduced = torch.tensor([rank + 1], dtype=torch.float32, device=device)
        dist.all_reduce(reduced, group=tensor_pg)
        expected_sum = world * (world + 1) / 2
        if reduced.item() != expected_sum:
            raise RuntimeError(f"all_reduce returned {reduced.item()}, expected {expected_sum}")

        source = torch.arange(world, dtype=torch.int64, device=device) + rank * world
        received = torch.empty_like(source)
        dist.all_to_all_single(received, source, group=tensor_pg)
        expected = torch.arange(world, dtype=torch.int64, device=device) * world + rank
        if not torch.equal(received, expected):
            raise RuntimeError(f"all_to_all_single returned {received.tolist()}, expected {expected.tolist()}")
    except Exception as exc:
        raise RuntimeError(
            "MiniMax SP NCCL startup probe failed; check per-worker GPU visibility, GPU P2P support, "
            f"and NCCL logs (set NCCL_DEBUG=INFO): {exc}") from exc


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
        "sample_sigmas": tensor_meta(transformer_options.get("sample_sigmas")),
        "keyframes": [{"resolved_frame_index": kf["resolved_frame_index"]}
                      for kf in (payload.get("keyframes") or [])] or None,
        "refs": [{k: r[k] for k in REF_META_KEYS if k in r}
                 for r in (payload.get("refs") or [])] or None,
        "frame_count": payload.get("frame_count"),
        "seed": payload.get("seed", 0),
        "visual_cond_noise_aug": payload.get("visual_cond_noise_aug", spf.VISUAL_COND_TIMESTEP),
        "audio_cond_noise_aug": payload.get("audio_cond_noise_aug", spf.AUDIO_COND_TIMESTEP),
        "options": {k: transformer_options[k] for k in TO_WORKER_OPTIONS if k in transformer_options},
        "layout": payload.get("layout") or spf.PackedLayout(
            context.shape[1], x[0].shape[2], (x[0].shape[3]+1)//2*2,
            (x[0].shape[4]+1)//2*2, x[1].shape[-1],
            keyframes=payload.get("keyframes"), refs=payload.get("refs")),
    }


def payload_from_meta(meta, tensors):
    p = {
        "layout": meta["layout"],
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


def ordered_tensors(x, timestep, context, transformer_options, payload):
    payload = payload or {}
    out = [x[0], x[1], timestep, context, payload.get("text_token_tags")]
    out += list(payload.get("cond_video_latents", []) or [])
    out += list(payload.get("cond_audio_latents", []) or [])
    out.append(transformer_options.get("sample_sigmas"))
    return out


class SPGroup:
    def __init__(self, world, unet_name, weight_dtype, devices, port=0):
        if dist.is_initialized():
            raise RuntimeError("MiniMax SP cannot adopt an existing process group; use a separate ComfyUI process")
        self.world = world
        self.unet_name = unet_name
        self.procs = []
        self.log_paths = {}
        self.n_forward = 0
        self.operation_id = 0
        self.patch_uuid = None
        self.patches_ready = False
        self.verify_pending = False
        self.control_pg = None
        self.tensor_pg = None
        self.store = None
        self._owns_default_pg = False
        self.startup_timeout = positive_timeout("MINIMAX_SP_STARTUP_TIMEOUT", 300)
        self.collective_timeout = positive_timeout("MINIMAX_SP_COLLECTIVE_TIMEOUT", 120)
        try:
            self._start(world, unet_name, weight_dtype, devices, port)
        except (Exception, comfy.model_management.InterruptProcessingException):
            self.destroy()
            raise

    def _start(self, world, unet_name, weight_dtype, devices, port):
        # A fresh store avoids reusing stale rendezvous keys after cancellation.
        # Bind port 0 before spawning, rather than probing/releasing a free port.
        self.store = dist.TCPStore("127.0.0.1", port, world, True,
                                   timeout=self.startup_timeout, wait_for_workers=False)
        port = self.store.port
        comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        runtime = {k: getattr(comfy_args, k) for k in (
            "force_fp16", "fp16_unet", "fp32_unet", "bf16_unet", "disable_dynamic_vram",
            "disable_comfy_compiler", "disable_cuda_graphs", "use_pytorch_cross_attention",
            "disable_triton_backend", "highvram", "lowvram", "reserve_vram",
            "vram_headroom", "disable_nvml_pressure")}
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sp_worker.py")
        log_dir = os.environ.get("MINIMAX_SP_LOGDIR") or tempfile.gettempdir()
        for r in range(1, world):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = devices[r]
            env["PYTHONPATH"] = comfy_root + os.pathsep + env.get("PYTHONPATH", "")
            env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
            env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
            log_path = os.path.join(log_dir, f"minimax_sp_worker{r}.log")
            self.log_paths[r] = log_path
            with open(log_path, "w") as log:
                self.procs.append(subprocess.Popen(
                    [sys.executable, script, "--rank", str(r), "--world", str(world),
                     "--port", str(port), "--unet", folder_paths.get_full_path_or_raise("diffusion_models", unet_name),
                     "--weight-dtype", weight_dtype, "--comfy-root", comfy_root,
                     "--startup-timeout", str(int(self.startup_timeout.total_seconds())),
                     "--collective-timeout", str(int(self.collective_timeout.total_seconds())),
                     "--runtime-options", json.dumps(runtime)],
                    env=env, stdout=log, stderr=log, cwd=comfy_root))
            logging.info(f"[minimax_sp] spawned worker rank {r} on physical GPU {devices[r]}")

        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
        dist.init_process_group("gloo", store=self.store, rank=0, world_size=world,
                                timeout=self.startup_timeout)
        self._owns_default_pg = True
        self.control_pg = dist.group.WORLD
        self.tensor_pg = dist.new_group(backend="nccl", timeout=self.collective_timeout)
        logging.info("[minimax_sp] process groups up, waiting for workers to load weights...")
        loaded = [None] * world
        dist.all_gather_object(loaded, {"loaded": True}, group=self.control_pg)
        if any(reply != {"loaded": True} for reply in loaded):
            raise RuntimeError(f"MiniMax SP worker load failed: {loaded}")
        self._startup_probe()
        logging.info(f"[minimax_sp] all {world} ranks ready")
        atexit.register(self.shutdown)

    def _startup_probe(self):
        startup_probe(0, self.world, self.tensor_pg, torch.device("cuda", 0))

    def check_alive(self):
        for r, p in enumerate(self.procs, start=1):
            if p.poll() is not None:
                message = f"[minimax_sp] worker rank {r} died (exit {p.returncode}), see {self.log_paths[r]}"
                self.destroy()
                raise RuntimeError(message)

    def sync_patches(self, patcher):
        self.patches_ready = False
        self.check_alive()
        version = patch_sync.patch_version(patcher)
        try:
            dist.broadcast_object_list([{"op": "prepare", "memory_required": patch_sync.memory_budget(patcher),
                                         "version": version}], src=0, group=self.control_pg)
            ready = [None] * self.world
            dist.all_gather_object(ready, {"ready": True}, group=self.control_pg)
            if any(reply != {"ready": True} for reply in ready):
                raise RuntimeError(f"MiniMax SP worker preparation failed: {ready}")
            if version != self.patch_uuid:
                with patch_sync.export_patches(patcher) as manifest:
                    dist.broadcast_object_list([manifest.message()], src=0, group=self.control_pg)
                    replies = [None] * self.world
                    dist.all_gather_object(replies, manifest.ready(0), group=self.control_pg)
                    patch_sync.validate_replies(manifest, replies, self.world)
                self.patch_uuid = version
                self.verify_pending = os.environ.get("MINIMAX_SP_VERIFY") == "1"
                logging.info("[minimax_sp] all ranks acknowledged %d patched keys, sha256=%s",
                             len(manifest.keys), manifest.sha256)
            self.patches_ready = True
        except (Exception, comfy.model_management.InterruptProcessingException):
            self.destroy()
            raise

    def forward(self, dit, x, timestep, context, transformer_options, payload,
                denoise_mask=None, audio_denoise_mask=None):
        self.check_alive()
        if not self.patches_ready:
            raise RuntimeError("MiniMax SP denoise prohibited: patches have not been acknowledged")
        spf.validate_transformer_options(transformer_options)
        profile = os.environ.get("MINIMAX_SP_PROFILE")
        try:
            t_disp = time.perf_counter()
            meta = build_meta(x, timestep, context, transformer_options, payload)
            meta["operation_id"] = self.operation_id + 1
            meta["denoise_mask"] = tensor_meta(denoise_mask)
            meta["audio_denoise_mask"] = tensor_meta(audio_denoise_mask)
            dist.broadcast_object_list([meta], src=0, group=self.control_pg)
            expected = forward_status(meta)
            replies = [None] * self.world
            dist.all_gather_object(replies, expected, group=self.control_pg)
            validate_forward_status(expected, replies)
            self.operation_id = meta["operation_id"]
            t_meta = time.perf_counter()
            device = x[0].device
            if transformer_options.get("sample_sigmas") is not None:
                transformer_options["sample_sigmas"] = transformer_options["sample_sigmas"].to(device)
            for t in [*ordered_tensors(x, timestep, context, transformer_options, payload), denoise_mask, audio_denoise_mask]:
                if t is not None:
                    dist.broadcast(t.to(device).contiguous(), src=0, group=self.tensor_pg)
            if profile:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = spf.sp_forward(dit, x, timestep, context, transformer_options, payload,
                                 0, self.world, self.tensor_pg, denoise_mask, audio_denoise_mask)
            if self.verify_pending:
                # Verification only: compare the actual loaded ranks to native H3 before accepting a step.
                reference = type(dit)._forward(dit, x, timestep, context, transformer_options.copy(),
                                                minimax_payload=payload, denoise_mask=denoise_mask,
                                                audio_denoise_mask=audio_denoise_mask)
                for label, actual, expected in zip(("video", "audio"), out, reference):
                    delta = (actual.float() - expected.float()).abs().max().item()
                    relative = delta / max(expected.float().abs().max().item(), 1e-12)
                    logging.info("[minimax_sp][verify] %s max_abs=%.8g relative=%.8g", label, delta, relative)
                    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all() or relative > 1e-3:
                        raise RuntimeError(f"[minimax_sp] {label} differs from native H3: relative={relative}")
                self.verify_pending = False
            if profile:
                torch.cuda.synchronize()
                logging.info(f"[minimax_sp][profile] step {self.n_forward}: "
                             f"meta_bcast={1000 * (t_meta - t_disp):.1f}ms "
                             f"tensor_bcast={1000 * (t0 - t_meta):.1f}ms "
                             f"sp_forward={1000 * (time.perf_counter() - t0):.1f}ms")
                self.n_forward += 1
            return out
        except comfy.model_management.InterruptProcessingException:
            logging.info("[minimax_sp] forward cancelled; rebuilding the group on the next prompt")
            self.destroy()
            raise
        except Exception as exc:
            logging.error("[minimax_sp] forward failed; rebuilding the group on the next prompt: %s", exc)
            self.destroy()
            raise

    def destroy(self):
        # Do not broadcast shutdown into a failed collective.
        self.patches_ready = False
        atexit.unregister(self.shutdown)
        for process in getattr(self, "procs", []):
            if process.poll() is None:
                process.kill()
            process.wait()
        self.procs = []

        tensor_pg = getattr(self, "tensor_pg", None)
        control_pg = getattr(self, "control_pg", None)
        owns_default = getattr(self, "_owns_default_pg", False)
        initialized = dist.is_initialized()
        try:
            if initialized and tensor_pg is not None:
                dist.destroy_process_group(tensor_pg)
        finally:
            self.tensor_pg = None
            try:
                if initialized and owns_default and control_pg is not None:
                    dist.destroy_process_group(control_pg)
            finally:
                self.control_pg = None
                self._owns_default_pg = False
                self.store = None
                for key, group in list(_GROUPS.items()):
                    if group is self:
                        del _GROUPS[key]

    def shutdown(self):
        if not self.procs:
            return
        try:
            dist.broadcast_object_list([{"op": "shutdown"}], src=0, group=self.control_pg)
        except Exception:
            pass
        for p in self.procs:
            try:
                p.wait(timeout=15)
            except Exception:
                p.kill()
        self.destroy()


def active_group():
    """The SP group currently running, if any. Only one exists per process."""
    return next(iter(_GROUPS.values()), None)


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
