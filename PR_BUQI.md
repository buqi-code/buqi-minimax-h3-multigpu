# PR draft: Fix current H3 conditioning and synchronize worker patches

Draft only — not posted. Target: buqi-code/buqi-minimax-h3-multigpu main.

## Summary

Keep the existing rank0/worker/NCCL/Ulysses architecture while adapting the forward
to current ComfyUI H3. Preserve the native world_size=1 path and all attribution.

- Carry the current PackedLayout and audio/video conditioning/masks to workers;
  follow the native unscaled audio-velocity and final-layer contracts.
- Synchronize the actual downstream ModelPatcher's LoRA patches/strengths before
  denoising. Verify payload SHA256, patch keys, version and rank acknowledgments.
  The worker loop delegates patch internals to one shared module.
- Resolve the actual ComfyUI root and checkpoint path. Match worker runtime flags
  and initialize ComfyUI's existing DynamicVRAM in the worker.
- Use the author's all-to-all path; do not use head-sliced QKV copies for row-scaled
  INT8 / changing LoRA. Do not alter the Ulysses attention or sequence partition.
- Stop on patch failures, discard dead workers, preserve numerical verification.
- Catch ComfyUI cancellation (a BaseException), release failed groups/workers,
  and use a fresh rendezvous store so subsequent prompts can recreate the group.

Related reports: #5 (current API/layout), #1 (INT8), #2 (installation instructions).
This does not claim to resolve the unrelated RTX 5090 collective-stall report.

## Validation

Initial forward/patch candidate: `e5f08a1`; final lifecycle fixes: `9bfaf92`.
ComfyUI `99073836`, PyTorch 2.14.0+cu130, two 22528 MiB RTX 2080 Ti (SM75), NV2.
Tiny native-vs-SP T2V/I2V/reference/audio/mask comparisons pass (max abs 7.45e-9).
The candidate completed fresh 832×480 / 124-frame 4-step and 8-step Turbo runs,
with 208 verified patch keys on each rank, then Base clearing (zero keys) and I2V.
Native single-GPU bypass and repeated two-GPU generation also passed.
Controlled cancellation and idle-worker death both permit actual generation
after worker recreation without restarting ComfyUI. Earlier failures and their
fixes are retained in CHANGELOG_LOCAL.md and evidence/pr_validation.json.

The earlier local compatibility experiment completed INT8 ConvRot + FP16 video
VAE and both 4/8-step Turbo LoRAs at 832×480 / 124 frames. Its warmed 20-step Base
baseline was 935.28 s single vs 534.30 s dual (1.75× total / 1.86× steady step).
Those are historical timings, not a new benchmark of this candidate.

## Scope / limitations

No ComfyUI core/dependency changes. Current native H3 uses FP32 compute on SM75;
FP16 storage is not FP16 compute. Only the named INT8 H3 format is validated here.
No full ref2va, full FP16 checkpoint, FP8/NVFP4 H3, SP VAE, dynamic hooks, attention
replacements or >2-rank claims. Static trusted same-host patch IPC is not a network
security boundary. Startup timeout and in-collective process death still need
separate lifecycle work; an idle-worker recovery test does not cover those cases.

This work derives from buqi-code's MIT implementation. I would appreciate review
of the current H3 assumptions and wider GPU coverage. I also intend to discuss a
separate H3 SP integration proposal with ComfyUI, with attribution retained;
this PR does not introduce a new general MultiGPU framework.
