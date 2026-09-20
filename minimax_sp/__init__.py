"""Ulysses sequence-parallel MiniMax-H3 DiT loader for ComfyUI.

Drop-in replacement for UNETLoader that additionally spawns world_size-1 worker
processes (one per extra GPU) and routes every diffusion-model forward through
a sequence-parallel path. Mathematically equivalent to single-GPU sampling:
same latents, same video/audio output.

Launch ComfyUI with all target GPUs visible, e.g. for 2 GPUs:
    python main.py --cuda-device 0,1 --highvram
"""

import inspect
import logging
import subprocess
import os
import sys

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.sd
import folder_paths

from . import sp_group
from . import sp_vae

from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout
from comfy.patcher_extension import CallbacksMP
import comfyui_version


def check_h3_api():
    required = {"minimax_payload", "denoise_mask", "audio_denoise_mask"}
    if not required.issubset(inspect.signature(MiniMaxH3Model._forward).parameters):
        raise RuntimeError("MiniMax SP needs the current H3 forward with audio/video denoise masks")
    if "frame_count" in inspect.signature(PackedLayout).parameters:
        raise RuntimeError("MiniMax SP compatibility branch requires the current PackedLayout API")
    root = os.path.dirname(os.path.abspath(folder_paths.__file__))
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    logging.info("[minimax_sp] ComfyUI %s commit=%s MiniMaxH3Model%s",
                 comfyui_version.__version__, commit, inspect.signature(MiniMaxH3Model))
    if commit != "99073836d45f66053c45ba8564984e6def9cebba":
        logging.warning("[minimax_sp] this ComfyUI revision is unverified; run tests/test_current_api.py "
                        "and enable MINIMAX_SP_VERIFY=1 before relying on outputs")



def resolve_devices(devices, world):
    if devices and devices.strip() and devices.strip().lower() != "auto":
        picked = [d.strip() for d in devices.split(",")]
        if len(picked) != world:
            raise ValueError(f'devices "{devices}" lists {len(picked)} GPUs, world_size is {world}')
        return picked
    env = os.environ.get("MINIMAX_SP_DEVICES")
    if env:
        picked = [d.strip() for d in env.split(",")]
        if len(picked) < world:
            raise ValueError(f"MINIMAX_SP_DEVICES lists {len(picked)} devices, need {world}")
        return picked[:world]
    return [str(i) for i in range(world)]


class MiniMaxH3SPUNETLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPUNETLoader",
            display_name="MiniMax H3 Multi-GPU Loader (Ulysses SP)",
            category="advanced/multigpu",
            description="Loads the MiniMax-H3 DiT and shards the packed sequence across "
                        "world_size GPUs (Ulysses all-to-all). Numerically equivalent to "
                        "single-GPU sampling.",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]),
                io.Int.Input("world_size", default=2, min=1, max=8,
                             tooltip="GPUs to shard across. Must divide the 56 attention heads "
                                     "(valid: 1, 2, 4, 7, 8)."),
                io.String.Input("devices", default="auto",
                                tooltip='Comma-separated physical CUDA ids, e.g. "0,1". The first id '
                                        'must be the GPU ComfyUI itself runs on. "auto" uses '
                                        'MINIMAX_SP_DEVICES if set, else the first world_size GPUs.'),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype, world_size, devices="auto") -> io.NodeOutput:
        if world_size > 1:
            check_h3_api()
            if os.environ.get("MINIMAX_SP_EXCHANGE", "all_to_all") != "all_to_all":
                raise RuntimeError("Use MINIMAX_SP_EXCHANGE=all_to_all; head-sliced QKV is not validated")
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
        if world_size <= 1:
            return io.NodeOutput(model)

        if sys.platform == "win32":
            raise RuntimeError("multi-GPU SP needs NCCL, which is not available on native Windows; "
                               "run ComfyUI inside WSL2 or on Linux")

        dit = model.get_model_object("diffusion_model")
        if not isinstance(dit, MiniMaxH3Model):
            raise RuntimeError("MiniMax SP loader requires a MiniMaxH3Model checkpoint")
        heads = dit.blocks[0].attn.heads
        if heads % world_size:
            raise ValueError(f"world_size {world_size} must divide the {heads} attention heads "
                             "(valid: 1, 2, 4, 7, 8)")

        picked = resolve_devices(devices, world_size)
        if torch.cuda.device_count() < world_size:
            raise ValueError(f"world_size is {world_size} but only {torch.cuda.device_count()} GPUs "
                             "are visible; start ComfyUI with e.g. --cuda-device 0,1")

        sp_group.get_group(world_size, unet_name, weight_dtype, picked)

        model = model.clone()
        dit = model.get_model_object("diffusion_model")

        def sp_entry(x, timestep, context, transformer_options=None, minimax_payload=None,
                     denoise_mask=None, audio_denoise_mask=None, **kwargs):
            group = sp_group.get_group(world_size, unet_name, weight_dtype, picked)
            return group.forward(dit, x, timestep, context, transformer_options or {}, minimax_payload,
                                 denoise_mask, audio_denoise_mask)

        def sync_patches(patcher):
            group = sp_group.get_group(world_size, unet_name, weight_dtype, picked)
            group.sync_patches(patcher)

        model.add_callback_with_key(CallbacksMP.ON_PRE_RUN, "minimax_sp_lora", sync_patches)
        # ModelPatcher installs/restores the inner call without bypassing outer wrappers.
        model.add_object_patch("diffusion_model._forward", sp_entry)
        logging.info(f"[minimax_sp] sequence parallel enabled, world_size={world_size} devices={picked}")
        return io.NodeOutput(model)


class MiniMaxH3SPVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPVAEDecode",
            display_name="MiniMax H3 Multi-GPU VAE Decode",
            category="advanced/multigpu",
            description="Drop-in VAEDecode that spreads the video VAE's temporal chunks "
                        "across the GPUs of a running MiniMax H3 SP group. Falls back to "
                        "the stock decode when that is not possible. Costs each worker the "
                        "video VAE weights (~5 GB) on top of the DiT.",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, samples, vae) -> io.NodeOutput:
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        group = sp_group.active_group()
        images = None
        if group is None:
            logging.info("[minimax_sp] no SP group running, decoding the VAE on one GPU")
        elif group.world > 1 and sp_vae.is_supported(vae):
            images = group.vae_decode(vae, latent)
            if images is None:
                logging.info("[minimax_sp] too few temporal chunks to shard, "
                             "decoding the VAE on one GPU")
        if images is None:
            images = vae.decode(latent)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return io.NodeOutput(images)


class MiniMaxSPExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SPUNETLoader, MiniMaxH3SPVAEDecode]


async def comfy_entrypoint() -> ComfyExtension:
    return MiniMaxSPExtension()
