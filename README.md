# buqi-minimax-h3-multigpu

**English** | [中文](README_zh.md) | [日本語](README_ja.md)

A ComfyUI custom node that runs the MiniMax H3 DiT with Ulysses sequence parallelism across multiple NVIDIA GPUs.

## Compatibility and status

- Requires the MiniMax H3 API introduced at or after ComfyUI commit **`8d534945`**; startup uses a runtime capability check instead of a release-version claim.
- Tested against the current ComfyUI 0.37 development commit **`7fbcfa8be9a8f47cf905ec47978b5bd754959ea7`**.
- The old ComfyUI 0.30 integration belongs to older releases; the main branch no longer claims compatibility with it.
- Platform: Linux or WSL2 with PyTorch/NCCL. Native Windows multi-GPU is not supported.
- `world_size` must divide MiniMax H3's 56 attention heads: **1, 2, 4, 7, or 8**.

The CPU suite checks package discovery, current API contracts, fail-fast behavior, patch synchronization, and two-rank Gloo parity. Current-commit GPU execution still requires the manual self-hosted acceptance workflow or the commands below; do not interpret CPU tests as a GPU performance claim.

## Install

Clone the repository directly as one ComfyUI custom-node directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
```

Restart ComfyUI. No extra pip package is required by this node; PyTorch and ComfyUI are intentionally not declared as pip dependencies.

## Models and INT8

Current example filenames are:

- DiT: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Reference DiT: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Video VAE: `minimax_h3_video_vae_int8_convrot.safetensors`
- Audio VAE: `minimax_h3_audio_vae_fp32.safetensors`
- Turbo LoRA: `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

For a pre-quantized INT8/ConvRot checkpoint, select **`weight_dtype=default`**. Its quantization metadata is loaded by ComfyUI; do not force another dtype in the SP loader.

Use the standard **`VAEDecode`** node for both official INT8 and FP16 video VAEs. This preserves ComfyUI's quantized metadata, streaming behavior, and memory policy. The old `MiniMaxH3SPVAEDecode` node ID remains only so existing workflows load: it logs one deprecation warning and calls the same official `vae.decode` path; it no longer performs parallel VAE decode.

| VAE path | Supported behavior |
|---|---|
| Standard `VAEDecode` + official INT8 or FP16 video VAE | Recommended; official decode and streaming path |
| Legacy `MiniMaxH3SPVAEDecode` node | Compatibility alias for standard `vae.decode`; deprecated and not parallel |
| Audio VAE | Standard `VAEDecodeAudio`; not sharded by this project |

## Supported workflow surface

| Feature | Status |
|---|---|
| T2V | Supported by the current H3 conditioning path |
| I2V first frame | Supported |
| I2V first + last frame | Supported |
| R2V and chained `MiniMaxH3AddGuide` | Supported |
| Denoise masks | Supported |
| PDD/static model options | Supported |
| Static Turbo LoRA loaded before execution | Supported; patches are synchronized before denoising |
| Fun ControlNet | Not supported; rejected before distributed execution |
| Sparse Attention | Not supported; rejected before distributed execution |
| Dynamic hooks/patches | Not supported; rejected before distributed execution |
| Stacking with ComfyUI MultiGPU/threaded MultiGPU | Not supported; rejected before distributed execution |

“Supported” describes the capability-checked H3 interface. Run the GPU acceptance tests for the exact hardware, model files, and workflow combination you deploy.

## SP versus DP

- **Sequence parallelism (this project):** one prompt is split across GPUs by sequence/head communication. Every rank holds the full DiT weights. It targets lower latency for one generation; it does not divide model-weight VRAM.
- **Data parallelism:** each GPU runs a separate prompt. It increases throughput, not the latency of one prompt, and needs an external queue/orchestrator.

Do not stack this SP loader with another MultiGPU wrapper.

## Start and device mapping

Two GPUs:

```bash
cd /path/to/ComfyUI
MINIMAX_SP_DEVICES=0,1 python main.py --cuda-device 0,1 --highvram
```

Then replace only `UNETLoader` with `MiniMaxH3SPUNETLoader`. Keep standard `VAEDecode`, especially with the INT8 video VAE.

`devices="auto"` reads `MINIMAX_SP_DEVICES` first, then the physical identifiers in `CUDA_VISIBLE_DEVICES` (including values set by `--cuda-device 2,3`). It does not replace those with logical `0,1`. Explicit mappings must be unique, contain exactly `world_size` visible physical identifiers, and start with ComfyUI's primary visible GPU. `world_size=1` behaves as a normal single-GPU load.

## Examples

| File | Format | Coverage |
|---|---|---|
| [`examples/workflow_ui_2gpu.json`](examples/workflow_ui_2gpu.json) | UI | Direct copy of the checked-out official “Image to Video (MiniMax H3)” blueprint with only `UNETLoader` replaced by `MiniMaxH3SPUNETLoader`; optional first/last inputs cover T2V, first-frame I2V, first+last I2V, and its static Turbo LoRA switch |
| [`examples/workflow_api_t2v_2gpu.json`](examples/workflow_api_t2v_2gpu.json) | API | T2V |
| [`examples/workflow_api_2gpu.json`](examples/workflow_api_2gpu.json) | API | First-frame I2V |
| [`examples/workflow_api_i2v_first_last_2gpu.json`](examples/workflow_api_i2v_first_last_2gpu.json) | API | First+last-frame I2V |
| [`examples/workflow_api_r2v_2gpu.json`](examples/workflow_api_r2v_2gpu.json) | API | R2V reference image |
| [`examples/workflow_api_addguide_2gpu.json`](examples/workflow_api_addguide_2gpu.json) | API | Arbitrary-frame AddGuide |
| [`examples/workflow_api_turbo_lora_2gpu.json`](examples/workflow_api_turbo_lora_2gpu.json) | API | Static 8-step Turbo LoRA |

The API variants follow the node schemas in the tested development commit but are not claimed as upstream-exported templates. For a deployment-specific API graph, import the UI workflow, configure and validate it with your installed models, enable ComfyUI Developer Mode, then use **Save (API Format)**.

API image names beginning with `REPLACE_WITH_` are intentional placeholders. Put your own files in `ComfyUI/input` and replace those values before submission. The UI workflow has no repository image dependency.

## Verification

CPU checks:

```bash
cd /path/to/ComfyUI/custom_nodes/buqi-minimax-h3-multigpu
COMFYUI_ROOT=/path/to/ComfyUI PYTHONPATH=/path/to/ComfyUI \
  python -m unittest discover -s tests -p 'test_*.py' -v
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 tests/test_current_api.py
python -m compileall -q __init__.py minimax_sp tests
python -c 'import json,pathlib; [json.loads(p.read_text()) for p in pathlib.Path("examples").glob("*.json")]'
```

Two-GPU checks:

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/latent_parity.py --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
COMFYUI_ROOT=/path/to/ComfyUI python tests/runtime_recovery.py \
  --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors --devices auto
```

The parity and lifecycle scripts exit nonzero on failure. `tests/selftest.py` is an end-to-end server smoke test and also propagates submission/execution failures, but encoded-video hashes are informational rather than a numerical parity check.

## Timeouts, logs, and troubleshooting

| Variable | Default | Purpose |
|---|---:|---|
| `MINIMAX_SP_DEVICES` | `CUDA_VISIBLE_DEVICES` | Optional physical device mapping used by `devices=auto` |
| `MINIMAX_SP_STARTUP_TIMEOUT` | 300 s | Worker/process-group startup timeout |
| `MINIMAX_SP_COLLECTIVE_TIMEOUT` | 120 s | NCCL collective timeout |
| `MINIMAX_SP_LOGDIR` | system temp directory | Worker log directory (`minimax_sp_worker<N>.log`) |
| `MINIMAX_SP_VERIFY=1` | off | Compare the first SP result with native H3 before accepting it |
| `MINIMAX_SP_PROFILE=1` | off | Forward timing logs |
| `MINIMAX_SP_PROFILE_OPS=1` | off | Per-region timing logs |

Common failures:

- **Node is missing after clone:** confirm the repository directory itself is under `custom_nodes`, contains the root `__init__.py`, and inspect the ComfyUI startup traceback.
- **Native Windows error:** use WSL2 or Linux; NCCL is required.
- **GPU visibility/P2P/startup probe failure:** verify `--cuda-device`, `MINIMAX_SP_DEVICES`, `nvidia-smi`, and NCCL/P2P availability. Increase `MINIMAX_SP_STARTUP_TIMEOUT` only if model loading is genuinely slow.
- **Collective timeout or worker exit:** inspect every worker log in `MINIMAX_SP_LOGDIR`; the rank-specific traceback is authoritative. Fix the underlying OOM, device mapping, or unsupported patch instead of repeatedly extending the timeout.
- **Deprecated VAE node warning:** replace `MiniMaxH3SPVAEDecode` with standard `VAEDecode`; both now call the official decode path.
- **Changed model/dtype/world size:** restart ComfyUI before creating a different SP group.

## License

MIT; see [LICENSE](LICENSE).
