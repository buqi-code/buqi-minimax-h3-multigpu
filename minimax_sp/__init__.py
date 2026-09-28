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
import os
import sys

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.ops
import comfy.sd
import folder_paths

from . import sp_group
from .sp_forward import validate_transformer_options

from comfy.ldm.minimax.model import Attention, DiTBlock, FinalLayer, MiniMaxH3Model, PackedLayout
from comfy.patcher_extension import CallbacksMP
import comfyui_version


def check_h3_api():
    problems = []
    forward_required = {"x", "timestep", "context", "transformer_options", "minimax_payload",
                        "denoise_mask", "audio_denoise_mask"}
    missing = forward_required.difference(inspect.signature(MiniMaxH3Model._forward).parameters)
    if missing:
        problems.append(f"MiniMaxH3Model._forward is missing {', '.join(sorted(missing))}")
    if "frame_count" in inspect.signature(PackedLayout).parameters:
        problems.append("PackedLayout still requires frame_count")
    try:
        attention = Attention(1, 1, 1, 1e-6, operations=comfy.ops.disable_weight_init)
    except TypeError:
        attention = None
    if attention is None or not hasattr(attention, "comfy_attention"):
        problems.append("Attention does not expose comfy_attention")
    if "attention" not in inspect.signature(DiTBlock.forward).parameters:
        problems.append("DiTBlock.forward does not accept attention")
    final_required = {"sigma", "sample_sigmas", "shifts"}
    final_missing = final_required.difference(inspect.signature(FinalLayer.forward).parameters)
    if final_missing:
        problems.append(f"FinalLayer.forward is missing {', '.join(sorted(final_missing))}")
    if problems:
        raise RuntimeError("MiniMax SP requires the H3 API introduced by ComfyUI commit 8d534945: "
                           + "; ".join(problems) + ". Update ComfyUI and restart it.")
    logging.info("[minimax_sp] compatible ComfyUI %s MiniMaxH3Model%s",
                 comfyui_version.__version__, inspect.signature(MiniMaxH3Model))


def _parse_device_list(value, source):
    picked = [item.strip() for item in value.split(",")]
    if not picked or any(not item for item in picked):
        raise ValueError(f"{source} must be a comma-separated list of CUDA device identifiers")
    if len(set(picked)) != len(picked):
        raise ValueError(f"{source} contains duplicate CUDA devices: {value}")
    return picked


def resolve_devices(devices, world):
    visible_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = None
    if visible_value is not None:
        visible = _parse_device_list(visible_value, "CUDA_VISIBLE_DEVICES")
        if visible == ["-1"]:
            raise ValueError("CUDA_VISIBLE_DEVICES exposes no CUDA devices")

    explicit = devices and devices.strip() and devices.strip().lower() != "auto"
    override = os.environ.get("MINIMAX_SP_DEVICES") if not explicit else None
    if explicit or override:
        source = "devices" if explicit else "MINIMAX_SP_DEVICES"
        value = devices if explicit else override
        picked = _parse_device_list(value, source)
        if len(picked) != world:
            raise ValueError(f'{source} "{value}" lists {len(picked)} GPUs, world_size is {world}')
    elif visible is not None:
        if len(visible) < world:
            raise ValueError(f"CUDA_VISIBLE_DEVICES lists {len(visible)} devices, need {world}")
        picked = visible[:world]
    else:
        picked = [str(i) for i in range(world)]

    if visible is not None:
        missing = [device for device in picked if device not in visible]
        if missing:
            raise ValueError(f"SP worker devices must be physical identifiers from CUDA_VISIBLE_DEVICES; missing {missing}")
        if picked[0] != visible[0]:
            raise ValueError(f"first SP device {picked[0]} must match ComfyUI's primary visible device {visible[0]}")
    elif torch.cuda.is_available() and picked[0] != str(torch.cuda.current_device()):
        raise ValueError(f"first SP device {picked[0]} must match ComfyUI's current CUDA device {torch.cuda.current_device()}")
    return picked


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
                                tooltip='Comma-separated physical CUDA ids, e.g. "2,3". The first id '
                                        'must be ComfyUI\'s primary visible GPU. "auto" uses '
                                        'MINIMAX_SP_DEVICES, then CUDA_VISIBLE_DEVICES.'),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype, world_size, devices="auto") -> io.NodeOutput:
        if world_size > 1:
            check_h3_api()
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
            transformer_options = transformer_options or {}
            validate_transformer_options(transformer_options)
            group = sp_group.get_group(world_size, unet_name, weight_dtype, picked)
            return group.forward(dit, x, timestep, context, transformer_options, minimax_payload,
                                 denoise_mask, audio_denoise_mask)

        def sync_patches(patcher):
            group = sp_group.get_group(world_size, unet_name, weight_dtype, picked)
            group.sync_patches(patcher)

        model.add_callback_with_key(CallbacksMP.ON_PRE_RUN, "minimax_sp_lora", sync_patches)
        # ModelPatcher installs/restores the inner call without bypassing outer wrappers.
        model.add_object_patch("diffusion_model._forward", sp_entry)
        logging.info(f"[minimax_sp] sequence parallel enabled, world_size={world_size} devices={picked}")
        return io.NodeOutput(model)


_vae_deprecation_logged = False


class MiniMaxH3SPVAEDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SPVAEDecode",
            display_name="MiniMax H3 VAE Decode (Deprecated Alias)",
            category="advanced/multigpu",
            description="Compatibility alias for old workflows. Uses the official vae.decode path; "
                        "replace it with the standard VAEDecode node.",
            inputs=[
                io.Latent.Input("samples"),
                io.Vae.Input("vae"),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, samples, vae) -> io.NodeOutput:
        global _vae_deprecation_logged
        if not _vae_deprecation_logged:
            logging.warning("[minimax_sp] MiniMaxH3SPVAEDecode is deprecated; using official vae.decode")
            _vae_deprecation_logged = True
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        return io.NodeOutput(images)


class MiniMaxSPExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SPUNETLoader, MiniMaxH3SPVAEDecode]


async def comfy_entrypoint() -> ComfyExtension:
    return MiniMaxSPExtension()
