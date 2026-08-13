"""Ulysses sequence-parallel MiniMax-H3 DiT loader for ComfyUI.

Drop-in replacement for UNETLoader that additionally spawns world_size-1 worker
processes (one per extra GPU) and routes every diffusion-model forward through
a sequence-parallel path. Mathematically equivalent to single-GPU sampling:
same latents, same video/audio output.

Launch ComfyUI with all target GPUs visible, e.g. for 2 GPUs:
    python main.py --cuda-device 0,1 --highvram
"""

import logging
import sys

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.sd
import folder_paths

from . import sp_group

try:
    from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: F401
except ImportError:
    raise RuntimeError(
        "buqi-minimax-h3-multigpu requires a ComfyUI build that ships the "
        "MiniMax-H3 model (comfy.ldm.minimax.model); update ComfyUI to >= 0.30.0"
    )


def resolve_devices(devices, world):
    if devices and devices.strip() and devices.strip().lower() != "auto":
        picked = [d.strip() for d in devices.split(",")]
        if len(picked) != world:
            raise ValueError(f'devices "{devices}" lists {len(picked)} GPUs, world_size is {world}')
        return picked
    import os
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

        heads = model.model.diffusion_model.blocks[0].attn.heads
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

        def sp_entry(x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
            group = sp_group.get_group(world_size, unet_name, weight_dtype, picked)
            return group.forward(dit, x, timestep, context, transformer_options, minimax_payload)

        # patch _forward instead of adding a DIFFUSION_MODEL wrapper: a wrapper that
        # never calls executor() short-circuits the chain and silently disables other
        # wrappers on the same hook. forward() resolves _forward per call, so this
        # keeps the chain intact with the sharded path as its inner call.
        model.add_object_patch("diffusion_model._forward", sp_entry)
        logging.info(f"[minimax_sp] sequence parallel enabled, world_size={world_size} devices={picked}")
        return io.NodeOutput(model)


class MiniMaxSPExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3SPUNETLoader]


async def comfy_entrypoint() -> ComfyExtension:
    return MiniMaxSPExtension()
