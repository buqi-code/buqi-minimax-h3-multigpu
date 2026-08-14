"""Shard the MiniMax-H3 video VAE decode across the sequence-parallel group.

decode_temporal() walks the latent one temporal chunk at a time and each chunk
only reads a slice of z, so the chunks are independent and can be spread over the
ranks that already hold the group. Only `_adaptive_decode` is farmed out: every
spatial tile blend, temporal blend and canvas write stays in the official code
path on rank 0, which is fed the precomputed chunks.

Workers are handed the VAE weights over NCCL rather than a filename, so the
sharded decode always uses exactly the checkpoint the user loaded.

Pinned to ComfyUI 0.30.0's comfy/ldm/minimax/vae.py: chunk_plan and
prepare_latent mirror the padding, chunk bounds and denormalization of
decode()/decode_temporal() and must be re-checked when that file changes.
"""

import torch


def is_supported(vae):
    fsm = getattr(vae, "first_stage_model", None)
    return all(hasattr(fsm, a) for a in
               ("_adaptive_decode", "tokens_chunk_size", "token_overlap", "token_drop",
                "vae_ratio", "vae_ratio_t", "latents_mean", "latents_std"))


def chunk_plan(fsm, z_len):
    """Mirror decode_temporal's padding and per-chunk token bounds."""
    pseudo = z_len + fsm.token_drop
    pad_tokens = 0
    remainder = pseudo % fsm.tokens_chunk_size
    if remainder != 0:
        pad_tokens = fsm.tokens_chunk_size - remainder
        pseudo += pad_tokens
    num_chunks = pseudo // fsm.tokens_chunk_size - int(fsm.token_drop > 0)
    if num_chunks < 1:
        pad_tokens += fsm.tokens_chunk_size
        num_chunks += 1

    padded_len = z_len + pad_tokens
    bounds = []
    for i in range(num_chunks):
        t0 = i * fsm.tokens_chunk_size
        t1 = min(t0 + fsm.tokens_chunk_size + fsm.token_overlap, padded_len)
        bounds.append((t0, t1))
    return pad_tokens, bounds


def prepare_latent(fsm, z, pad_tokens):
    """Apply decode()'s denormalization then decode_temporal()'s tail padding."""
    mean = fsm.latents_mean.view(1, -1, 1, 1, 1).to(z)
    std = fsm.latents_std.view(1, -1, 1, 1, 1).to(z)
    z = z * std + mean
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
    return z


def chunk_shape(fsm, z, t0, t1):
    """Pixel shape _adaptive_decode returns for one chunk: 4 frames per token, 16x spatial."""
    return (z.shape[0], 3, (t1 - t0) * fsm.vae_ratio_t,
            z.shape[3] * fsm.vae_ratio, z.shape[4] * fsm.vae_ratio)


def decode_local(fsm, z_prepared, bounds, rank, world):
    """Decode the chunks this rank owns. Round-robin keeps the load even."""
    out = {}
    for i, (t0, t1) in enumerate(bounds):
        if i % world == rank:
            out[i] = fsm._adaptive_decode(z_prepared[:, :, t0:t1])
    return out
