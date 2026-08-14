# buqi-minimax-h3-multigpu

**English** | [中文](README_zh.md) | [日本語](README_ja.md)

Multi-GPU (sequence-parallel) inference for **MiniMax-H3** in ComfyUI — one custom
node, drop-in replacement for `UNETLoader`, **no approximation and no precision
loss**.

MiniMax-H3 is a joint video+audio generation model: one packed-token DiT produces
the video *and* its soundtrack (text-to-video / image-to-video with synchronized
audio). This node shards that packed sequence across 2/4/7/8 GPUs with the
[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) scheme: every GPU computes
exact full attention for a subset of heads, so the math is unchanged.

### On accuracy

Every GPU still evaluates exact attention over the whole sequence — nothing is
approximated, quantized further, masked, or cached. The result is therefore
*mathematically* equivalent to single-GPU sampling, but it is **not bit-identical**
to it: attention runs over 56/`world_size` heads per rank, and the attention
kernel picks its reduction order from the head count, so floating-point rounding
lands differently. Measured on the DiT's own output, the deviation from a single
GPU is **≤ 4.2e-6 relative** (audio: exactly 0) — three orders of magnitude below
bfloat16 resolution (7.8e-3), with no systematic bias.

Verify it yourself with `tests/latent_parity.py`, which compares the velocity
tensors the DiT produces rather than a hash of the encoded video (H.264 is lossy
and hides differences of this size in either direction).

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

Measured on 2× RTX PRO 5000 Blackwell (72 GB, PCIe 5.0, **no NVLink**), fp8,
832×480, 20 steps:

| stage | 1 GPU | SP2 | speedup |
|---|---|---|---|
| denoise loop | 78.3 s | 42.3 s | **1.85×** |
| VAE decode (not parallelized) | 5.9 s | 5.9 s | 1.00× |
| end-to-end | 89.1 s | 55.3 s | **1.61×** |

Where the remaining gap goes, from the op-level profile (`MINIMAX_SP_PROFILE_OPS=1`,
per step, per rank): attention+output exchange 856 ms (timed together because they
overlap), MLP 657 ms, gather+qkv_proj 325 ms, out_proj 83 ms, modulation+norms
76 ms, QK-norm+RoPE 27 ms. The transformer GEMMs already run at 270–370 TFLOPS
(fp8) and attention at ~176 TFLOPS, so there is nothing left to win in the math
itself — the only cost that does not shard is the inter-GPU exchange, which
started at ~21 % of a step (481 ms) and is now largely hidden behind compute.

End-to-end scaling is capped by the parts that still run on one GPU: VAE decode
(5.9 s here) and video muxing. Longer clips and higher resolutions therefore scale
better, since the denoise loop grows while that tail stays flat.

Larger shapes (1080p and up) additionally benefit from activations being split
across ranks, which can push a single GPU into offloading while SP2 stays
resident — those cases can look super-linear for that reason, not because
sequence parallelism exceeds its own limit.

## Resolution constraints

Heights/widths follow the stock MiniMax-H3 rules: canvas is rounded to 32 px and
the latent (px/16) must be divisible by the patch (2), so stick to the standard
grids — 832×480, 1280×736, 1920×1088 etc. `height=720` is **not** a valid canvas
(latent height 45 is odd); use 736.

## Verify correctness on your machine

The meaningful check compares what the DiT actually produces, on the GPUs you own:

```bash
cd custom_nodes/buqi-minimax-h3-multigpu
torchrun --nproc_per_node=2 tests/latent_parity.py \
    --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
```

It runs the stock single-GPU `_forward` as the reference and the sequence-parallel
forward on the same inputs, for both exchange strategies, and prints the maximum
absolute and relative deviation. Expect `ag vs a2a … exact=True` (the two
strategies agree exactly) and a `vs 1gpu` relative deviation around 4e-6.

There is also an end-to-end smoke test that renders one job each way and compares
the output videos:

```bash
python tests/selftest.py --server http://127.0.0.1:18188 --sp 2 \
    --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
    --image your_first_frame.png
```

Treat its hash comparison as indicative only: H.264 is lossy, so it can both hide
real differences and report differences that no viewer can see. Use
`latent_parity.py` for anything you care about.

## Profiling

```bash
MINIMAX_SP_PROFILE_OPS=1 MINIMAX_SP_PROFILE=1 python main.py --cuda-device 0,1 --highvram
```

Logs a per-step breakdown by region (attention, MLP, gather, exchange, …) plus the
per-step dispatch cost, which is what the tuning above was based on.
`tests/profile_phases.py` does the same at node granularity over the websocket API.

## Environment variables

| variable | meaning |
|---|---|
| `MINIMAX_SP_DEVICES` | fallback device list when `devices=auto`, e.g. `0,1` |
| `MINIMAX_SP_LOGDIR` | worker log directory (default: system temp) |
| `MINIMAX_SP_AG_CHUNKS` | sub-chunks the hidden-state gather is split into so it overlaps the projection (default 4; 1 disables overlap) |
| `MINIMAX_SP_ATTN_CHUNKS` | head chunks attention is split into so the output exchange overlaps it (default 4; 1 disables overlap) |
| `MINIMAX_SP_PROFILE` | log per-step dispatch and forward timings |
| `MINIMAX_SP_PROFILE_OPS` | log the per-step breakdown by region |

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
per-token op (patch proj, modulation, RoPE, MLP) is row-local and runs without
communication. Only attention crosses ranks: each rank ends up computing complete,
exact attention over the *full* sequence for 56/`world_size` heads.

There are two ways to get the heads and the sequence to line up, and which one is
cheaper depends on `world_size`:

- **all-to-all** (the classic Ulysses form): project Q/K/V for all 56 heads on the
  local rows, then swap the head dim for the sequence dim. Moves
  `3 * inner / world` bytes per row.
- **all_gather** (used when `world_size <= 4`): broadcast the modulated hidden
  state instead, then project it through only this rank's head rows of
  `qkv_proj`. Moves `hidden` bytes per row, and Q/K/V come out already spanning
  the full sequence, so the three transpose+copy steps disappear entirely.

For H3 (`hidden` 5376, `inner` 7168) the two cost the same at `world_size == 4`
and all_gather wins below it — measured at 480p on PCIe: 6.04 ms → 2.34 ms per
block for the exchange, and the achieved bandwidth rises from 24.9 to 32.2 GB/s
because there is no longer a permute in the way. Above 4 the all-to-all path is
kept.

Row-slicing `qkv_proj` is exact rather than approximate: the fp8 weight carries a
single per-tensor scale, so selecting rows leaves every output element the same
dot product it was. It costs one extra copy of `1/world` of that weight per card
(~2.9 GB at fp8, `world_size=2`).

Finally both transfers are overlapped with the compute that surrounds them. The
gather is issued as several asynchronous sub-transfers (`MINIMAX_SP_AG_CHUNKS`,
default 4) so each chunk's projection runs while the next chunk is still in
flight, and attention is computed in head chunks (`MINIMAX_SP_ATTN_CHUNKS`,
default 4) so each chunk's output exchange flies while the next chunk is
computed. Together these are worth ~150 ms per step at 480p. Neither changes any
arithmetic — heads are independent in attention, and rows are independent in the
projection — and `tests/latent_parity.py` reports both bit-identical at 1, 2, 4
and 8 chunks.

## License

MIT (see [LICENSE](LICENSE)).

---

Keywords: MiniMax H3, MiniMax-H3, ComfyUI, ComfyUI custom node, multi-GPU, 多卡,
多卡并行, 并行推理, sequence parallelism, Ulysses, video generation, audio
generation, 视频生成, 音画同步, ビデオ生成, H3 加速, multi GPU inference.
