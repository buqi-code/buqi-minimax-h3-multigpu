# buqi-minimax-h3-multigpu

**English** | [中文](README_zh.md) | [日本語](README_ja.md)

Real multi-GPU inference for **MiniMax-H3** in ComfyUI — not just the denoise
loop, the **VAE decode** is parallelized too. Two drop-in nodes,
**no approximation and no precision loss**.

MiniMax-H3 is a joint video+audio generation model: one packed-token DiT
produces the video *and* its soundtrack (text-to-video / image-to-video with
synchronized audio), then a video VAE turns the latents into pixels. This repo
spreads both stages across 2/4/7/8 GPUs:

- **`MiniMaxH3SPUNETLoader`** — replaces `UNETLoader`. Shards the DiT's packed
  sequence across ranks with [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509):
  every rank computes exact full attention for a subset of heads, so the math
  is unchanged. Denoise **1.94×** on 2 GPUs, **6.65×** on 8 (below).
- **`MiniMaxH3SPVAEDecode`** — optional replacement for `VAEDecode`. Spreads
  the video VAE's temporal chunks across the same ranks. VAE decode
  **1.70×** on 2 GPUs, **5.99×** on 8.
- End-to-end **1.95×** on 2 GPUs at 480p and **6.28×** on 8 GPUs at 1080p;
  profile-driven, PCIe-friendly (measured without NVLink).

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

Measured on up to 8× RTX PRO 5000 Blackwell (72 GB, PCIe 5.0, **no NVLink**), fp8,
124 frames, 20 steps. Every number is steady state — the first sharded decode of
a session pays a one-time weight handover, so each configuration was measured on
a second run:

| shape | GPUs | 1 GPU | multi-GPU | end-to-end | denoise | VAE | cost per second of output |
|---|---|---|---|---|---|---|---|
| 832×480 | 2 | 89.7 s | 46.7 s | **1.95×** | 1.94× | 1.70× | **1.03×** |
| 832×480 | 4 | 91.2 s | 27.0 s | **3.37×** | 3.41× | 3.27× | **1.19×** |
| 832×480 | 8 | 91.1 s | 19.2 s | **4.74×** | 4.86× | 5.99× | 1.69× |
| 1280×736 | 8 | 301.6 s | 57.8 s | **5.22×** | 5.46× | 5.11× | 1.53× |
| 1920×1088 | 8 | 1131.8 s | 180.2 s | **6.28×** | 6.65× | 5.09× | **1.27×** |

The last column is `GPUs / speedup` — how much more GPU-time a second of video
costs than on one card. Two GPUs are nearly free (+3 %) and four are cheap
(+19 %); eight only pay off on big canvases, because there the denoise loop grows
while the fixed tail (text encoder, muxing) stays flat and the exchange stays
proportional to the work.

The VAE rows need `MiniMaxH3SPVAEDecode` (see below). Its ceiling is set by how
the temporal chunks divide — these clips have 7, so two GPUs split them 4/3 and
cannot beat 1.75×, while eight GPUs get one chunk each.

Where the denoise step goes at 480p on 2 GPUs, from the op-level profile
(`MINIMAX_SP_PROFILE_OPS=1`, per step, per rank): attention+output exchange
856 ms (timed together because they overlap), MLP 657 ms, gather+qkv_proj 325 ms,
out_proj 83 ms, modulation+norms 76 ms, QK-norm+RoPE 27 ms. The transformer GEMMs
already run at 270–370 TFLOPS (fp8) and attention at ~176 TFLOPS, so there is
nothing left to win in the math itself — the only cost that does not shard is the
inter-GPU exchange, which started at ~21 % of a step (481 ms) and is now largely
hidden behind compute.

At 1080p a single GPU also starts paying for memory pressure that the sharded run
avoids, which is part of why that row scales best.

## Resolution constraints

Heights/widths follow the stock MiniMax-H3 rules: canvas is rounded to 32 px and
the latent (px/16) must be divisible by the patch (2), so stick to the standard
grids — 832×480, 1280×736, 1920×1088 etc. `height=720` is **not** a valid canvas
(latent height 45 is odd); use 736.

## Multi-GPU VAE decode (optional)

Replace `VAEDecode` with **MiniMax H3 Multi-GPU VAE Decode**
(`MiniMaxH3SPVAEDecode`) — same two inputs, same output — and the video VAE's
temporal chunks are spread over the same GPUs the SP group already holds.

`decode_temporal` walks the latent one chunk at a time and each chunk only reads a
slice of it, so the chunks are independent. Only that leaf work is distributed:
every spatial tile blend, temporal blend and canvas write stays in the official
code path on rank 0, which is fed the finished chunks. The result is bit-identical
(verified below).

The workers receive the VAE weights over NCCL from rank 0 on first use rather than
loading a file themselves, so the sharded decode always uses exactly the
checkpoint you loaded, including custom paths. That transfer is ~4.85 GiB and
costs ~2.3 s once per session.

This is opt-in because it is not free: each worker holds the video VAE (~5 GB) on
top of the DiT, so ~26 GB per card at fp8. On 24 GB cards, keep using the stock
`VAEDecode`. The node falls back to the stock decode, with a log line, whenever
sharding does not apply: no SP group running, `world_size` 1, a VAE that is not
the H3 video VAE, or a latent with too few chunks to split.

## Verify correctness on your machine

Two tests compare tensors, which is the only meaningful level. The DiT:

```bash
cd custom_nodes/buqi-minimax-h3-multigpu
torchrun --nproc_per_node=2 tests/latent_parity.py \
    --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
```

Runs the stock single-GPU `_forward` as the reference and the sequence-parallel
forward on the same inputs, for both exchange strategies. Expect
`ag vs a2a … exact=True` and a `vs 1gpu` relative deviation around 4e-6.

And the VAE:

```bash
torchrun --nproc_per_node=2 tests/vae_parity.py --handover
```

Has each worker decode its chunks and rank 0 decode the same slices itself, then
compares them, with `--handover` rebuilding the worker VAE from rank 0's
broadcast weights exactly as the group does. Expect every chunk `exact=True`.

### Do not compare encoded video

`tests/selftest.py` runs a job both ways end to end, but it is only a smoke test.
Hashing the output video proves nothing: the encoded container is **not
byte-reproducible**. Three runs of the identical stock pipeline, same seed and
same inputs, produced three different hashes (`529324c6…` at 736142 bytes,
`c1c1c01c…` and `1bd76f6c…` both at 736204 bytes). A hash comparison there will
report failures that do not exist, and can hide ones that do.

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

Both nodes shard **whole stages**, not just kernels; nothing is approximated or
quantized further.

**DiT (denoise loop)** — the H3 DiT processes one packed sequence
`[text|cond|audio|video]` with 56 attention heads. Ulysses SP keeps the sequence
row-sharded across ranks; every per-token op (patch proj, modulation, RoPE, MLP)
is row-local and runs without communication. Only attention crosses ranks —
each rank computes exact full attention over the *full* sequence for
56/`world_size` heads. Two exchange strategies are supported and picked by
world size:

- `world_size <= 4` broadcasts the modulated hidden state and projects it
  through only this rank's head rows of `qkv_proj`. Moves `hidden` bytes per
  row and the three transpose+copy steps of the classic form disappear.
  Row-slicing the fp8 weight is exact because its scale is per-tensor.
- `world_size >= 7` keeps the classic all-to-all form
  (`3 * inner / world` bytes per row).

For H3 (`hidden` 5376, `inner` 7168) they cost the same at world 4 and
`all_gather` wins below it (6.04 ms → 2.34 ms per block on PCIe, and achieved
bandwidth rises 24.9 → 32.2 GB/s because there is no longer a permute in the
way). Both transfers overlap with the compute around them: the gather is issued
as async sub-transfers (`MINIMAX_SP_AG_CHUNKS`, default 4) so each chunk's
projection runs while the next chunk is still in flight, and attention is done
in head chunks (`MINIMAX_SP_ATTN_CHUNKS`, default 4) so each chunk's output
exchange flies while the next chunk is computed. Together those hide ~150 ms
per step. Neither changes any arithmetic; `tests/latent_parity.py` confirms
bit-identical output at 1, 2, 4 and 8 chunks.

**VAE decode** — `decode_temporal` walks the latent one chunk at a time and
each chunk only reads a slice of it, so the chunks are independent. Only that
leaf work is distributed round-robin across ranks; every spatial tile blend,
temporal blend and canvas write stays in the official code path on rank 0.
Workers receive the VAE weights over NCCL from rank 0 on first use (~4.85 GiB,
~2.3 s once per session), so the sharded decode always uses exactly the
checkpoint that was loaded.

**Tried and dropped, so nobody has to repeat it.** Profiling ruled these out:
AdaLN precomputation (upstream pruned weights already factor `t_dim` to an
8-dim curve basis; the whole branch is 44 M params and 0.06 ms per step),
fold-in of the text encoder (measured 1.0 s total on the short prompt),
swapping the gloo control channel for a persistent NCCL one (per-step meta
broadcast is 0.3 ms), and fusing modulation/norm into a Triton kernel (the
whole modulation+norm region is 3.4 % of a step). Details in the commit
history.

## License

MIT (see [LICENSE](LICENSE)).

---

Keywords: MiniMax H3, MiniMax-H3, ComfyUI, ComfyUI custom node, multi-GPU, 多卡,
多卡并行, 并行推理, sequence parallelism, Ulysses, video generation, audio
generation, 视频生成, 音画同步, ビデオ生成, H3 加速, multi GPU inference.
