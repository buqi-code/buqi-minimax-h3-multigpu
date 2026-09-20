# Upstream plan — scoped compatibility first

Source: [buqi-code/buqi-minimax-h3-multigpu](https://github.com/buqi-code/buqi-minimax-h3-multigpu),
MIT, Copyright (c) 2026 buqi-code. Original baseline: `bce0839`.
Do not describe Ulysses or this forward as locally invented.

## Review classification

| Owner | Keep from author (A) | Necessary local compatibility (B) | Possible framework work, not implemented (C) |
| --- | --- | --- | --- |
| __init__.py | Existing node IDs, native world_size=1, ModelPatcher object patch | Masks, actual downstream patcher ON_PRE_RUN, API diagnostics | A public model-specific SP integration point |
| sp_forward.py | Uneven row partition, QKV/output all-to-all, attention/MLP math | Current PackedLayout, masks/audio/reference time labels and final-layer contract | Share native H3 preprocessing where maintainers find an appropriate API |
| sp_group.py | Rank0 orchestration, NCCL tensors, Gloo control | Complete layout transport, patch ACK gate, worker options, teardown on failure | Distributed ownership, cancellation and error propagation only if accepted as shared needs |
| sp_worker.py | One process per extra GPU, independent DiT replica | Same model path/runtime, existing DynamicVRAM startup, verified patches | Public worker bootstrap/model reload factory |
| sp_patches.py | New local boundary, not upstream original | Trusted patch export/manifest/apply/ACK, no denoise before consistency | Public patch-only snapshot/reset contract, if a general use case exists |
| sp_vae.py | Author's optional temporal-shard algorithm, unchanged | Not enabled in baseline | No VAE framework changes proposed now |

## Branches and scope

- `archive/engineering-wip-20260921` (`5728bbb`): preserves the larger experimental
  compat/ decomposition and early reports. Not the submission branch.
- `fix/comfyui-h3-compat`: starts directly from upstream; keeps the original file
  layout and adds only one shared patch-sync boundary. It is a newly tested
  candidate, not an exact reconstruction of the earlier uncommitted success state.
- The earlier workspace and logs are also preserved in a local tar archive.

Removed from this candidate: whole-H3-file hash lock, extra wrapper-only adapters,
custom cleanup callback using a global active group, blanket FP8 UI rejection,
and automatic changes to the optional VAE's selection policy. Unknown ComfyUI
revisions warn; known incompatible signatures fail. Numerical tests remain the
semantic compatibility check. No core ComfyUI source is changed.

## Track A — custom node / buqi PR

1. Keep author history and MIT attribution; fork under RagnarokChan.
2. Submit the compatibility code with focused regression tests. Keep benchmark
   tooling/docs in a separate commit and archive experiments out of the PR.
3. Reference existing issues, rather than duplicating them:
   [#5 API/layout mismatch](https://github.com/buqi-code/buqi-minimax-h3-multigpu/issues/5),
   [#1 INT8 question](https://github.com/buqi-code/buqi-minimax-h3-multigpu/issues/1),
   [#2 node installation](https://github.com/buqi-code/buqi-minimax-h3-multigpu/issues/2).
4. Invite the author to review API assumptions and broader-device support. Do not
   claim the unrelated 5090 communication report is fixed by our SM75 evidence.

## Track B — ComfyUI proposal, not a framework transplant

The inspected current `comfy/multigpu.py` uses a same-process thread pool and
`ModelPatcher.deepclone_multigpu()` uses loader factories. Neither is a drop-in
owner of the author's cross-process NCCL group. Replacing the worker with these
would be a different architecture, not compatibility work.

Before a core implementation, ask maintainers whether cross-process SP belongs
in core and who should own group/device/model lifetimes. Search existing PRs and
discuss a narrow H3 integration proposal with evidence. The local inspection is
for ComfyUI `99073836`; re-check target HEAD before proposing exact code changes.

H3 should own packed sequence layout, modality masks and Ulysses integration.
Only demonstrably reusable process/device/model lifecycle primitives should go
in shared code. An H3-only requirement is not a reason to build a general
distributed framework. Native H3 already has an attention argument in DiTBlock,
but an attention-only hook does not replace the complete sequence-sharded forward
(including MLP); evaluate this explicitly rather than assuming it solves SP.

Current worker-local ModelPatcher field assignments / DynamicVRAM bootstrapping
are acknowledged integration debt. Wrapping them does not make them public APIs.
Do not submit those monkey patches as a proposed core design. The custom node can
remain usable while the core boundary is discussed; author review and ComfyUI
discussion can proceed without requiring simultaneous merges.

## Release gates

- Commit-tied native/SP parity and actual two-GPU INT8 generation.
- Base → 4step → 8step → Base; hash/keys/version/rank consistency; repeated runs.
- Negative ACK prevents denoise; cancellation and idle-worker failure/recovery.
- Preserve logs of failures, report limitations, never relabel historical timings
  as measurements of a newly edited commit.
- No claims for full FP16 H3, FP8, NVFP4 H3, full ref2va, SP VAE or arbitrary hooks
  without separate tests. FP16 kernel tests are not full-model verification.

Remaining architectural risks for reviewer discussion include startup timeout,
single-group ownership, same-host trusted pickle transport, full model replication,
and behavior if a process dies inside a collective. These are not solved merely
by passing an idle-worker recovery test.

## Publication and privacy

The submission branch retains author history. Local contributions use the
GitHub-provided noreply identity; no personal email or raw machine histories
belong in the public branch. Publish only the sanitized evidence summary.
The archive branch is a local safety copy, not a branch to push with this PR.
PR_BUQI.md contains the author-facing compatibility PR text.
COMFYUI_PROPOSAL.md is still a draft, not a posted ComfyUI discussion.
