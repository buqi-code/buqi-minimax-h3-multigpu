# Discussion draft: H3 sequence-parallel integration boundary

Draft only — not posted. This is a design discussion before a core PR, not a
proposal to copy a custom node and its worker monkey patches into ComfyUI.

We have a working compatibility adaptation of buqi-code's MIT MiniMax H3 Ulysses
implementation: rank0 is ComfyUI, rank1 is a separate worker, NCCL exchanges packed
sequence/head partitions. On two SM75 GPUs, INT8 ConvRot H3 + native VAE and Turbo
LoRA work. The earlier warmed 832×480 / 124-frame / 20-step experiment showed
1.75× end-to-end speedup; current submission verification is tracked separately.

The node currently mirrors H3 forward and installs one ModelPatcher-owned inner
_forward object patch. LoRA synchronization requires a patch-only snapshot/reset
that is not exposed as a public worker API. The worker also repeats the relevant
native DynamicVRAM startup. These are concrete integration costs, not reasons to
invent a broad distributed framework in this PR.

Questions for maintainers:

1. Is a cross-process H3 SP implementation an acceptable integration direction,
   or should it remain a custom node with a small supported H3 extension seam?
2. Where should shared process-group/device ownership and cancellation live if
   cross-process inference is accepted? The existing MultiGPUThreadPool and model
   deepclones are same-process facilities, not direct replacements for NCCL ranks.
3. Can model-specific H3 preprocessing/segment modulation be reused without a full
   forward fork? Attention injection alone does not preserve MLP sequence sharding.
4. Should a patch-only snapshot/version/application interface be public, or should
   each worker rebuild models from a maintained native loader description?

Suggested staging: first agree on the H3 boundary, then submit only the minimal
native H3/refactor and tests needed for it. Shared lifecycle work should be a
separate proposal only if justified by more than this one model. Keep layout,
modality masks and Ulysses integration with H3, not in a model-agnostic GPU manager.

Evidence and source: buqi-code/buqi-minimax-h3-multigpu, existing API compatibility
issue #5 and INT8 question #1. Attribution and MIT notices must remain. Searches
also found ComfyUI issue #15262 about FP16 overflow on V100; it is related dtype
context, not a bug we claim to fix. Recheck current upstream discussions/PRs before
posting, and do not imply maintainers have already accepted this architecture.
