"""Ulysses sequence-parallel forward for MiniMax-H3.

The packed sequence is split by rows across ranks. Every per-token op (patch
proj, AdaLN, RoPE, MLP, out_proj) is row-local and needs no communication; only
attention does, via two all-to-all exchanges that trade the head dim for the
sequence dim. Attention math is therefore unchanged: each rank runs complete
attention over the full sequence for 56/P of the heads.

Row splits are deliberately uneven (base + remainder) instead of padding the
sequence, so no attention mask is needed and results stay numerically equivalent
to the single-GPU path.

Originally pinned by buqi-code to ComfyUI 0.30.0. Local compatibility changes
target ComfyUI 99073836 (0.36.0): see AUDIT.md and CHANGES_LOCAL.md.
sp_forward mirrors MiniMaxH3Model._forward and must be re-checked when it changes.
"""

import copy
import dataclasses
import logging
import os
import time

import torch
import torch.distributed as dist

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.quant_ops
from comfy.ldm.minimax.model import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    PackedLayout,
    mask_row_values,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    time_shift_sigma,
    unpack_audio,
    unpatchify_video,
)
from comfy.ldm.modules.attention import optimized_attention

PROFILE_OPS = bool(os.environ.get("MINIMAX_SP_PROFILE_OPS"))
AG_CHUNKS = max(1, int(os.environ.get("MINIMAX_SP_AG_CHUNKS", "4")))
ATTN_CHUNKS = max(1, int(os.environ.get("MINIMAX_SP_ATTN_CHUNKS", "4")))
_prof_acc = {}


class region:
    """Accumulate wall time per named region; only active under MINIMAX_SP_PROFILE_OPS."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if PROFILE_OPS:
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if PROFILE_OPS:
            torch.cuda.synchronize()
            _prof_acc[self.name] = _prof_acc.get(self.name, 0.0) + time.perf_counter() - self.t0
        return False


def prof_report(total):
    rows = sorted(_prof_acc.items(), key=lambda kv: -kv[1])
    body = "  ".join(f"{k}={v * 1000:.0f}ms" for k, v in rows)
    logging.info(f"[minimax_sp][ops] step={total * 1000:.0f}ms  {body}")
    _prof_acc.clear()


def use_allgather(world, hidden, inner):
    """Broadcasting h beats swapping Q/K/V while hidden < 3*inner/world.

    For H3 (hidden 5376, inner 7168) the two cost the same bytes at world 4 and
    below that the all_gather path additionally removes three full permute copies.
    """
    return 1 < world <= (3 * inner) // hidden


def head_sharded_qkv(proj, world, rank):
    """Row-slice qkv_proj down to this rank's heads, keeping fp8 storage.

    The weight carries one per-tensor scale, so selecting rows is exact: every
    output element remains the dot product it was on a single GPU. Cached on the
    module so it is freed with the model.
    """
    cached = getattr(proj, "_sp_shard", None)
    if cached is not None and cached[0] == (world, rank):
        return cached[1]

    w = proj.weight
    inner = w.shape[0] // 3
    span = inner // world
    lo = rank * span
    idx = torch.cat([torch.arange(lo, lo + span, device=w.device) + off * inner
                     for off in range(3)])

    sub = copy.copy(proj)
    sub._parameters = dict(proj._parameters)
    del sub._parameters["weight"]
    if type(w).__name__ == "QuantizedTensor":
        qd = w._qdata[idx].contiguous()
        params = dataclasses.replace(w._params, orig_shape=(qd.shape[0], w.shape[1]))
        sub.weight = type(w)(qd, w._layout_cls, params)
    else:
        sub.weight = w[idx].contiguous()
    sub.out_features = idx.numel()
    proj._sp_shard = ((world, rank), sub)
    return sub


def gather_rows_full(local, ctx):
    """[s_local, C] -> [S_total, C] in rank order, tolerating uneven row splits."""
    c = local.shape[1]
    maxn = max(ctx.splits)
    if min(ctx.splits) == maxn:
        out = torch.empty(ctx.world * maxn, c, dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(out, local.contiguous(), group=ctx.group)
        return out
    buf = torch.zeros(maxn, c, dtype=local.dtype, device=local.device)
    buf[:local.shape[0]] = local
    out = torch.empty(ctx.world * maxn, c, dtype=local.dtype, device=local.device)
    dist.all_gather_into_tensor(out, buf, group=ctx.group)
    out = out.view(ctx.world, maxn, c)
    return torch.cat([out[r, :ctx.splits[r]] for r in range(ctx.world)], dim=0)


class SPContext:
    """Row-sharding descriptor for one forward pass."""

    def __init__(self, rank, world, group, seq_len):
        self.rank = rank
        self.world = world
        self.group = group
        self.seq_len = seq_len
        base, rem = divmod(seq_len, world)
        self.splits = [base + (1 if r < rem else 0) for r in range(world)]
        self.start = sum(self.splits[:rank])
        self.stop = self.start + self.splits[rank]

    @property
    def local(self):
        return self.splits[self.rank]


def post_heads_to_seq(t, ctx, pending):
    """[s_local, H, D] -> [S_total, H/P, D], transfer posted asynchronously.

    Appends (work, send) to pending: the send buffer has to stay referenced until
    the transfer completes. Posting rather than blocking lets the Q, K and V
    exchanges pipeline against each other's permute copies.
    """
    s_local, heads, dim = t.shape
    p = ctx.world
    hp = heads // p
    chunk = hp * dim
    send = t.reshape(s_local, p, chunk).transpose(0, 1).contiguous().view(-1)
    out = torch.empty(ctx.seq_len * chunk, dtype=t.dtype, device=t.device)
    work = dist.all_to_all_single(
        out, send,
        output_split_sizes=[n * chunk for n in ctx.splits],
        input_split_sizes=[s_local * chunk] * p,
        group=ctx.group,
        async_op=True,
    )
    pending.append((work, send))
    return out.view(ctx.seq_len, hp, dim)


def a2a_scatter_seq_gather_heads(t, ctx):
    """[S_total, (H/P)*D] -> [s_local, H*D]"""
    chunk = t.shape[1]
    p = ctx.world
    s_local = ctx.local
    out = torch.empty(p * s_local * chunk, dtype=t.dtype, device=t.device)
    dist.all_to_all_single(
        out, t.contiguous().view(-1),
        output_split_sizes=[s_local * chunk] * p,
        input_split_sizes=[n * chunk for n in ctx.splits],
        group=ctx.group,
    )
    return out.view(p, s_local, chunk).transpose(0, 1).reshape(s_local, p * chunk)


def gather_and_project(attn_proj, x, ctx, chunks, out_dim):
    """all_gather h and project it, with the transfers overlapping the projection.

    Chunk i holds the same row window from every rank, so its projected rows are
    written back at each rank's global offset and the result lands in exactly the
    row order a single GPU would produce. Row splits are uneven when the sequence
    does not divide by world, and all_gather needs equal contributions, so the
    local block is padded up to the largest split and the padding is dropped on
    write-back.
    """
    s_local, hidden = x.shape
    if chunks <= 1:
        return attn_proj(gather_rows_full(x, ctx))

    maxn = max(ctx.splits)
    if s_local == maxn:
        src = x
    else:
        src = torch.zeros(maxn, hidden, dtype=x.dtype, device=x.device)
        src[:s_local] = x

    span = -(-maxn // chunks)
    starts = [sum(ctx.splits[:r]) for r in range(ctx.world)]
    pending = []
    for a in range(0, maxn, span):
        b = min(a + span, maxn)
        buf = torch.empty(ctx.world * (b - a), hidden, dtype=x.dtype, device=x.device)
        src_chunk = src[a:b].contiguous()
        work = dist.all_gather_into_tensor(buf, src_chunk, group=ctx.group, async_op=True)
        pending.append((a, b, buf, work, src_chunk))

    out = torch.empty(ctx.seq_len, out_dim, dtype=x.dtype, device=x.device)
    for a, b, buf, work, _ in pending:
        work.wait()
        piece = attn_proj(buf)
        m = b - a
        for r in range(ctx.world):
            n = min(b, ctx.splits[r]) - a
            if n > 0:
                out[starts[r] + a:starts[r] + a + n] = piece[r * m:r * m + n]
    return out


def attention_and_exchange(q, k, v, ctx, transformer_options, chunks, hp, dim):
    """Attention in head chunks, with each chunk's output exchange in flight while
    the next chunk is computed.

    Heads are independent, so chunking the attention is exact. The exchange hands
    every rank the rows it owns for the chunk's heads; those land at that rank's
    head offset plus the chunk offset, which is the column order the single
    unchunked exchange produced.
    """
    inner = ctx.world * hp * dim
    if chunks <= 1:
        with region("attn"):
            out = optimized_attention(q, k, v, hp, mask=None, skip_reshape=True,
                                      transformer_options=transformer_options)
        with region("a2a_out"):
            return a2a_scatter_seq_gather_heads(out.squeeze(0), ctx)

    span = -(-hp // chunks)
    pending = []
    with region("attn+a2a_out"):
        for h0 in range(0, hp, span):
            h1 = min(h0 + span, hp)
            out = optimized_attention(q[:, h0:h1], k[:, h0:h1], v[:, h0:h1], h1 - h0,
                                      mask=None, skip_reshape=True,
                                      transformer_options=transformer_options)
            cd = (h1 - h0) * dim
            send = out.squeeze(0).contiguous().view(-1)
            buf = torch.empty(ctx.world * ctx.local * cd, dtype=q.dtype, device=q.device)
            work = dist.all_to_all_single(
                buf, send,
                output_split_sizes=[ctx.local * cd] * ctx.world,
                input_split_sizes=[n * cd for n in ctx.splits],
                group=ctx.group, async_op=True)
            pending.append((h0, cd, buf, work, send))

        out_full = torch.empty(ctx.local, inner, dtype=q.dtype, device=q.device)
        for h0, cd, buf, work, _ in pending:
            work.wait()
            piece = buf.view(ctx.world, ctx.local, cd)
            for r in range(ctx.world):
                off = r * hp * dim + h0 * dim
                out_full[:, off:off + cd] = piece[r]
    return out_full


def sp_attention_allgather(attn, x, rope_full, ctx, transformer_options, chunks=None):
    """Ulysses attention that broadcasts h instead of swapping Q/K/V.

    Each rank projects the whole sequence through only its own head rows, so Q/K/V
    come out already gathered over the sequence: no head/seq transpose, no QKV
    all-to-all, and only this rank's slice of qkv_proj has to be resident.
    """
    heads, dim = attn.heads, attn.head_dim
    hp = heads // ctx.world
    proj = head_sharded_qkv(attn.qkv_proj, ctx.world, ctx.rank)
    with region("gather+qkv_proj"):
        qkv = gather_and_project(proj, x, ctx, AG_CHUNKS if chunks is None else chunks,
                                 3 * hp * dim)
    s = qkv.shape[0]
    with region("qknorm+rope"):
        q, k, v = qkv.split(hp * dim, dim=-1)
        v = v.view(s, hp, dim)
        if rope_full is not None:
            q = q.view(1, s, hp, dim)
            k = k.view(1, s, hp, dim)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_full.shape[-3] * 2
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_full, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, hp, dim))
            k = attn.k_norm(k.view(s, hp, dim))
    out = attention_and_exchange(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), ctx, transformer_options,
        ATTN_CHUNKS, hp, dim)
    with region("out_proj"):
        return attn.out_proj(out)


def sp_attention(attn, x, rope_freqs, ctx, transformer_options):
    s = x.shape[0]
    heads, dim = attn.heads, attn.head_dim
    with region("qkv_proj+qknorm+rope"):
        q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
        v = v.view(s, heads, dim)
        if rope_freqs is not None:
            q = q.view(1, s, heads, dim)
            k = k.view(1, s, heads, dim)
            qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
            kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, heads, dim))
            k = attn.k_norm(k.view(s, heads, dim))

    hp = heads // ctx.world
    with region("a2a_qkv"):
        pending = []
        q = post_heads_to_seq(q, ctx, pending)
        k = post_heads_to_seq(k, ctx, pending)
        v = post_heads_to_seq(v, ctx, pending)
        for work, _ in pending:
            work.wait()
        pending.clear()
    out = attention_and_exchange(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), ctx, transformer_options,
        ATTN_CHUNKS, hp, dim)
    with region("out_proj"):
        return attn.out_proj(out)


def sp_block(block, x, t_emb, mod_segments, rope_freqs, ctx, transformer_options, allgather=False):
    from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift

    with region("adaln+norm+mod"):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        h = _mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
    attn = sp_attention_allgather if allgather else sp_attention
    attn_out = attn(block.attn, h, rope_freqs, ctx, transformer_options)
    with region("adaln+norm+mod"):
        x = _mod_gate(x, gate_msa, attn_out, mod_segments)
        h = _mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
    with region("mlp"):
        mlp_out = block.mlp(h)
    with region("adaln+norm+mod"):
        return _mod_gate(x, gate_mlp, mlp_out, mod_segments)


def shard_segments(segments, start, stop):
    """Clip global (start, stop, row) triples to a rank's row window, in local coords."""
    out = []
    for a, b, row in segments:
        lo, hi = max(a, start), min(b, stop)
        if hi > lo:
            out.append((lo - start, hi - start, row[lo-a:hi-a] if torch.is_tensor(row) else row))
    return out


def _gather_rows(local, counts, ctx, out_dim, device):
    """Padded all_gather of variable-length row blocks, concatenated in rank order."""
    maxn = max(counts)
    buf = torch.zeros(maxn, out_dim, dtype=torch.float32, device=device)
    if local is not None:
        buf[:local.shape[0]] = local
    gathered = torch.empty(ctx.world * maxn, out_dim, dtype=torch.float32, device=buf.device)
    dist.all_gather_into_tensor(gathered, buf, group=ctx.group)
    gathered = gathered.view(ctx.world, maxn, out_dim)
    return torch.cat([gathered[r, :counts[r]] for r in range(ctx.world) if counts[r] > 0], dim=0)


def sp_forward(dit, x, timestep, context, transformer_options, minimax_payload, rank, world, group,
               denoise_mask=None, audio_denoise_mask=None):
    """Sequence-parallel mirror of MiniMaxH3Model._forward.

    Returns [video_velocity, audio_velocity] on rank 0, None elsewhere.
    """
    t_fwd = time.perf_counter()
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, dit.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                              keyframes=payload.get("keyframes"), refs=payload.get("refs"))

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", dit.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", dit.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
    # Current ComfyUI labels masked rows at their own stream sigma.
    video_rows_t = audio_rows_t = None
    if denoise_mask is not None:
        m = mask_row_values(denoise_mask[0, 0].float(), latent_t, lat_h, lat_w)
        if m is not None:
            rows_t = (1.0 - m * sigma_v.to(m.device)).clamp(max=max(t_v, VISUAL_COND_TIMESTEP))
            if rows_t.unique().numel() == 1:
                seg_t["video"] = float(rows_t[0])
            else:
                video_rows_t = rows_t
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].float().reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            rows_t = (1.0 - m * (1.0 - t_a)).clamp(max=max(t_a, AUDIO_COND_TIMESTEP))
            if rows_t.unique().numel() == 1:
                seg_t["audio"] = float(rows_t[0])
            else:
                audio_rows_t = rows_t
    unique_t = sorted({t_v, t_a} | {seg_t[k] for _, _, k in layout.segments}
                      | (set(video_rows_t.unique().tolist()) if video_rows_t is not None else set())
                      | (set(audio_rows_t.unique().tolist()) if audio_rows_t is not None else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2}

    def rows_to_mod_index(rows_t, tag):
        levels = rows_t.unique()
        base = torch.tensor([t_row[v] * 3 + tag for v in levels.tolist()], dtype=torch.long, device=rows_t.device)
        return base[torch.searchsorted(levels, rows_t)]

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        elif kind == "video" and video_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(video_rows_t, 0)))
        elif kind == "audio" and audio_rows_t is not None:
            mod_segments.append((a, b, rows_to_mod_index(audio_rows_t, 2)))
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
    audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
    if video_rows_t is not None:
        video_seg = (*video_seg[:2], rows_to_mod_index(video_rows_t, 0) // 3)
    if audio_rows_t is not None:
        audio_seg = (*audio_seg[:2], rows_to_mod_index(audio_rows_t, 0) // 3)

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), dit.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = dit._cond_video_rows(payload, device)
    cond_audio_rows = dit._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = dit.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = dit.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != dit.hidden_size:
        text_states = dit.token_refiner(dit.condition_proj(text_states),
                                        transformer_options=transformer_options)

    ctx = SPContext(rank, world, group, layout.seq_len)
    # embedding assembly is row-local and cheap; every rank builds only its own window
    h = torch.empty(ctx.local, dit.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        lo, hi = max(a, ctx.start), min(b, ctx.stop)
        if hi > lo:
            if kind == "text":
                h[lo - ctx.start:hi - ctx.start] = text_states[lo - a:hi - a]
            elif kind in ("cond", "ref_img", "video"):
                h[lo - ctx.start:hi - ctx.start] = video_embed[voff + (lo - a):voff + (hi - a)]
            else:
                h[lo - ctx.start:hi - ctx.start] = audio_embed[aoff + (lo - a):aoff + (hi - a)]
        if kind in ("cond", "ref_img", "video"):
            voff += n
        elif kind != "text":
            aoff += n

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if dit.use_adaln_curves:
        table = comfy.model_management.cast_to(dit.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    else:
        t_emb = dit.time_embedder(t_vals).to(dtype)


    rope_freqs = rope_rotation_table(dit.rope_freqs(layout.position_ids, device), dtype)
    local_segments = shard_segments(mod_segments, ctx.start, ctx.stop)

    if os.environ.get("MINIMAX_SP_EXCHANGE", "all_to_all") != "all_to_all":
        raise RuntimeError("MiniMax SP requires all_to_all: head-sliced QKV is not validated for INT8 / LoRA")
    allgather = False
    rope_for_blocks = rope_freqs[:, ctx.start:ctx.stop].contiguous()

    if PROFILE_OPS:
        torch.cuda.synchronize()
        _prof_acc["pre"] = time.perf_counter() - t_fwd

    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(dit.blocks), device, transformer_options)
    for block in dit.blocks:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        h = sp_block(block, h, t_emb, local_segments, rope_for_blocks, ctx, transformer_options,
                     allgather=allgather)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

    def counts_for(seg):
        a, b, _ = seg
        out = []
        s = 0
        for n in ctx.splits:
            lo, hi = max(a, s), min(b, s + n)
            out.append(max(0, hi - lo))
            s += n
        return out

    with region("final+gather"):
        def local_seg(seg):
            a, b, row = seg
            lo, hi = max(a, ctx.start), min(b, ctx.stop)
            if hi <= lo:
                return (0, 0, row[:0] if torch.is_tensor(row) else row)
            return (lo-ctx.start, hi-ctx.start, row[lo-a:hi-a] if torch.is_tensor(row) else row)

        v_local, a_local = dit.final_layer(h, t_emb, local_seg(video_seg), local_seg(audio_seg),
                                          sigma_v, transformer_options.get("sample_sigmas"), (shift_v, shift_a))

        v = _gather_rows(v_local, counts_for(video_seg), ctx, dit.final_layer.video_out.out_features, device)
        a = _gather_rows(a_local, counts_for(audio_seg), ctx, dit.final_layer.audio_out.out_features, device)
    if PROFILE_OPS and rank == 0:
        prof_report(time.perf_counter() - t_fwd)
    if rank != 0:
        return None

    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, dit.latents_dim, dit.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
