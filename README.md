# buqi-minimax-h3-multigpu

**English** | [中文](README_zh.md) | [日本語](README_ja.md)

Multi-GPU (sequence-parallel) inference for **MiniMax-H3** in ComfyUI — one node,
drop-in replacement for `UNETLoader`, **bit-identical output**, no quality loss.

MiniMax-H3 generates video **and audio** jointly from a single packed-token DiT.
This node shards that packed sequence across 2/4/7/8 GPUs with the
[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) all-to-all scheme: every
GPU computes exact full attention for a subset of heads, so the math is
unchanged — the multi-GPU result is bit-identical to single-GPU sampling
(verified; run `tests/selftest.py` to prove it on your machine).

Designed for the community's common setups first: **2 GPUs** (works from 1 to 8).

## Requirements

- Linux (or WSL2 on Windows — NCCL is not available on native Windows)
- ComfyUI **>= 0.30.0** (the release that ships `comfy.ldm.minimax.model`); the
  loader refuses to import on older builds
- MiniMax-H3 model files (DiT, Qwen3-VL text encoder, video/audio VAEs) installed
  as usual into `diffusion_models/`, `text_encoders/`, `vae/` — get them from the
  official MiniMax-H3 release; this repo contains code only
- PyTorch with NCCL (stock ComfyUI wheels include it)
- `world_size` must divide the 56 attention heads: **1, 2, 4, 7, 8**

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
```

No extra Python dependencies.

## Quick start (2 GPUs)

1. Start ComfyUI with both GPUs visible:

   ```bash
   python main.py --cuda-device 0,1 --highvram
   ```

2. Replace `UNETLoader` with **MiniMax H3 Multi-GPU Loader (Ulysses SP)** in your
   workflow. `world_size=2` is the default; leave `devices=auto`.

3. Queue a prompt. First run spawns one worker process per extra GPU and loads a
   copy of the DiT on each (~21 GB fp8 per card); later prompts reuse them.

An API-format example is in [`examples/workflow_api_2gpu.json`](examples/workflow_api_2gpu.json);
a ready-to-import graph workflow (drag & drop into the ComfyUI canvas) is
[`examples/workflow_ui_2gpu.json`](examples/workflow_ui_2gpu.json).

## Node inputs

| input | default | meaning |
|---|---|---|
| `unet_name` | — | H3 DiT checkpoint, same as `UNETLoader` |
| `weight_dtype` | `default` | same options as `UNETLoader`; fp8 recommended |
| `world_size` | `2` | GPUs to shard across (1/2/4/7/8). `1` = plain single-GPU load |
| `devices` | `auto` | physical CUDA ids, e.g. `"0,1"`. First id must be the GPU ComfyUI itself runs on. `auto` = `MINIMAX_SP_DEVICES` env if set, else the first `world_size` GPUs |

`world_size=1` passes through untouched, so a workflow with this node also runs
on a single-GPU machine.

## VRAM & resolution guidance

Sequence parallelism replicates the full DiT on every GPU (weights are not
split), so per-card VRAM is the same as single-GPU — the win is speed, and it
also lets you hold larger activations.

| weights | DiT per card | practical cards |
|---|---|---|
| fp8 (recommended) | ~21 GB | 24 GB cards up to 720p, comfortable at 480p |
| bf16 (`default` dtype from bf16 checkpoint) | ~40 GB | 48 GB+ cards |

Measured on RTX PRO 5000 (48 GB), fp8, 20 steps, end-to-end seconds:

| shape | 1 GPU | SP2 | speedup |
|---|---|---|---|
| 480p × 5s | 89.3 | 62.3 | 1.43× |
| 720p × 5s | 300.8 | 223.5 | 1.35× |
| 720p × 10s | 867.6 | 403.7 | 2.15× |
| 1080p × 5s | 1127.4 | 375.6 | 3.00× |

SP2's sub-linear scaling is expected: with only 2 ranks the per-step all-to-all
overhead is amortized over less work, and sampling itself is cheap relative to
VAE decode at short durations. Longer videos and higher resolutions scale better
(measured up to 6.9× on 8 GPUs).

PCIe / no-NVLink machines work fine (tested on RTX 5090); expect slightly lower
scaling than the NVLink numbers above.

## Resolution constraints

Heights/widths follow the stock MiniMax-H3 rules: canvas is rounded to 32 px and
the latent (px/16) must be divisible by the patch (2), so stick to the standard
grids — 832×480, 1280×736, 1920×1088 etc. `height=720` is **not** a valid canvas
(latent height 45 is odd); use 736.

## Verify correctness on your machine

```bash
# with the server running:
python custom_nodes/buqi-minimax-h3-multigpu/tests/selftest.py \
    --server http://127.0.0.1:18188 --sp 2 \
    --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
    --image your_first_frame.png
```

Runs one single-GPU job and one 2-GPU job with identical inputs and compares
SHA-256 of the output videos. Expect `PASS: sp2 output is bit-identical`.

## Environment variables

| variable | meaning |
|---|---|
| `MINIMAX_SP_DEVICES` | fallback device list when `devices=auto`, e.g. `0,1` |
| `MINIMAX_SP_LOGDIR` | worker log directory (default: system temp) |

## Troubleshooting

- **"world_size N must divide the 56 attention heads"** — use 1/2/4/7/8.
- **worker died, see .../minimax_sp_worker1.log** — usually VRAM; lower the
  resolution or use an fp8 checkpoint. The log has the real traceback.
- **hang at "process group up, waiting for workers to load weights"** — workers
  load ~21 GB each; first run can take a couple of minutes. Check worker logs.
- **`devices` mismatch** — the first id in `devices` must be the GPU ComfyUI runs
  on (`--cuda-device`).
- One SP group per ComfyUI process: changing checkpoint/dtype/world_size needs a
  server restart.

## How it works

The H3 DiT processes one packed sequence `[text|cond|audio|video]` with 56
attention heads. Ulysses SP keeps the sequence row-sharded across ranks; every
per-token op (patch proj, AdaLN, RoPE, MLP) is row-local and runs without
communication. Only attention crosses ranks, via two all-to-all exchanges that
swap the head dim for the sequence dim — each rank then runs complete, exact
attention over the *full* sequence for 56/P heads. No approximation, no mask,
no loss.

## License

MIT (see [LICENSE](LICENSE)).
