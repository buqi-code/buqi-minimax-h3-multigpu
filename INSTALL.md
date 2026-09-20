# Install the compatibility branch

This is a local modification of buqi-code/buqi-minimax-h3-multigpu (MIT), not a new algorithm.
The branch is `fix/comfyui-h3-compat` in the
[RagnarokChan fork](https://github.com/RagnarokChan/buqi-minimax-h3-multigpu).
Do not expect upstream main to contain these changes until merged.

To obtain the compatibility branch, clone it beside an existing ComfyUI checkout:

```bash
git clone --branch fix/comfyui-h3-compat https://github.com/RagnarokChan/buqi-minimax-h3-multigpu.git
```

If you already have this checkout, keep it; do not clone over an existing folder.

## Existing local checkout

Use the Python environment that already runs ComfyUI. No new packages are needed.
Keep this repository beside ComfyUI, then link only its node subpackage:

```bash
ln -s /absolute/path/buqi-minimax-h3-multigpu/minimax_sp /absolute/path/ComfyUI/custom_nodes/minimax_sp
```

Skip that command if the correct link already exists. Do not overwrite an existing
installation or install duplicate copies. The whole repository must remain available
because workers import the subpackage from it.

```bash
conda activate comfyui
cd /absolute/path/buqi-minimax-h3-multigpu
COMFYUI_ROOT=/absolute/path/ComfyUI bash run_comfy.sh
```

`COMFYUI_PYTHON` can select an explicit Python executable instead of the activated
environment. The launcher has no hard-coded user home or conda path. It defaults
to port 8188, visible GPUs 0,1, native DynamicVRAM, PyTorch attention, all-to-all,
and first-forward numerical verification. Use the normal native VAEDecode nodes.

Choose `MiniMaxH3SPUNETLoader`, `world_size=2`, `weight_dtype=default` in the
official workflow. Use the installed INT8 ConvRot FL2VA checkpoint and native
video/audio VAEs. Qwen runs on rank0 and is not distributed by this node.

On SM75, `--fp16-unet` does not force H3 compute to FP16: current native H3 selects
FP32. Do not force BF16/FP16 compute or infer support from a storage dtype.
For two-rank Turbo on this 64 GiB RAM machine, keep DynamicVRAM enabled; eager
legacy loading previously exhausted host RAM. SP replicates weights; it does not
pool VRAM. The optional author's SP VAE is not part of the validated configuration.

## Reproduce checks

```bash
python tests/test_api_compat.py
python tests/test_lora_sync.py
python -m torch.distributed.run --standalone --nproc_per_node=2 tests/test_current_api.py
python tests/baseline_single_gpu.py --name my_single --verification enabled
python tests/baseline_multi_gpu.py --name my_multi --verification enabled
```

Use distinct run names; the harness refuses to overwrite results. For timing
comparisons, start fresh servers separately, use identical model/prompt/runtime,
warm up equally and explicitly record `--warmup warm`. Small smoke runs are
correctness checks, not speedup benchmarks. `--verification enabled` records the
server setting; it does not enable verification on the server itself.

```bash
python tests/benchmark.py --compare my_single my_multi
```

The comparison rejects mismatched prompts, source hashes, runtime options or
uncontrolled warmup. `results/benchmark.json` contains unified records, including
failed/incomplete runs. Raw histories/events/GPU samples remain under each run.
Historical 480p baseline measurements are explicitly marked as historical.

## Unsupported / unverified

Only all-to-all is enabled on this branch. The author's experimental head-sliced
all-gather helpers remain in source for attribution/review, but cannot be selected
by the runtime. FP8/NVFP4 H3 formats and a full FP16 H3 checkpoint have not been
validated by these tests. Existing dtype UI options are preserved, not endorsements
of those formats on SM75. Hooks, custom patch functions, attention replacements,
and ComfyUI thread-based MultiGPU clones are not synchronized by this worker path.

Never expose the NCCL/Gloo rendezvous to untrusted peers. Patch manifests and
temporary pickle files are trusted same-user, same-host IPC, not an authentication
or sandbox boundary. SHA256 detects transfer corruption, not malicious peers.
