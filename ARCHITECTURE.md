# Architecture and attribution

Original implementation: buqi-code, MIT. Local work fixes compatibility and patch
consistency; it does not replace the original Ulysses algorithm.

## Native ComfyUI H3

```text
Loader → ModelPatcher → H3 Forward → Attention → H3 output
```

## Original buqi implementation

```text
SP Loader → Rank0 + worker replicas → sp_forward fork → Ulysses → H3 output
```

## Current compatibility candidate

```text
Loader → ModelPatcher → ON_PRE_RUN: actual downstream LoRA patches
                            │
                         Patch export
                            ↓
                  Manifest: version + SHA256 + keys
                            ↓
                  Worker verify → apply → ACK(rank)
                            ↓
                  Rank0 checks every ACK; failure stops here
                            ↓
                  SP runtime → author's all-to-all Ulysses → H3 output
                            ↓
                  Rank0 native audio/video VAE → output
```

world_size=1 returns the native model before worker/group creation. world_size=2
uses NCCL for tensors and Gloo for control. Each rank owns a full DiT replica with
native offloading; text encoder and normal VAE decode stay on rank0.

Cancellation or a detected dead worker invalidates the patch-ready gate, kills
the remaining owned workers and releases process-group/store references. The
next prompt recreates workers using a freshly bound TCPStore and repeats patch
verification. Failed construction is cleaned up too. This is scoped recovery,
not a general fault-tolerant distributed framework.

The single `diffusion_model._forward` object patch is installed/restored by
ModelPatcher. Replacing it with a wrapper that bypasses executor would suppress
other wrappers, so removal is not automatically an improvement. Downstream
LoRA clones retain the ON_PRE_RUN callback. `sp_patches.py` alone knows patch
representation and version fields; the worker dispatch loop handles protocol ops.

Private dependency boundaries are documented rather than hidden: H3's internal
helpers in sp_forward, ModelPatcher patch/version fields in sp_patches, native
DynamicVRAM bootstrapping in sp_worker, and the author's temporary VAE callback
replacement in sp_group. The optional VAE fallback now calls the saved original
method instead of recursively calling its own replacement; this is not a claim
that full SP VAE has been validated.
