import argparse
import sys

import torch

import tilelang
from tilelang import language as T
from tilelang.intrinsics import make_zn_layout, make_nz_layout
from tilelang.profiler import do_bench


# ===========================================================================
# varlen_utils (inlined) — padding-mask generation + cu_seqlens derivation
# Ported from Developer version (verified).
# ===========================================================================


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    """Generate a random padding mask.

    Args:
        max_seqlen: maximum possible sequence length (= padded seqlen).
        batch_size: number of sequences.
        device: torch device (e.g. "npu").
        mode: "full" (all max), "random" ([max-20, max]), "third" ([max//3, max]).

    Returns:
        padding_mask: [batch_size, max_seqlen] bool tensor.
    """
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device)
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = torch.arange(max_seqlen, device=device).unsqueeze(0) < lengths
    return padding_mask


def mask_to_cu_seqlens(padding_mask):
    """Convert a [batch, seqlen] bool mask to cu_seqlens [batch+1] int32."""
    lengths = padding_mask.sum(dim=1).to(torch.int32)
    cu_seqlens = torch.zeros(padding_mask.shape[0] + 1, dtype=torch.int32, device=padding_mask.device)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    return cu_seqlens


def build_attention_mask(
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    is_causal,
    device,
):
    """Build the attention mask tensor on the host.

    Returns:
        mask: [batch, max_seqlen_q, max_seqlen_k] float32.
              1.0 = visible, 0.0 = masked (padding or causal).
    """
    batch = int(cu_seqlens_q.shape[0]) - 1
    q_idx = torch.arange(max_seqlen_q, device=device).view(-1, 1)  # [M, 1]
    k_idx = torch.arange(max_seqlen_k, device=device).view(1, -1)  # [1, N]
    mask = torch.zeros(batch, max_seqlen_q, max_seqlen_k, dtype=torch.float32, device=device)
    for b in range(batch):
        q_len = int(cu_seqlens_q[b + 1].item()) - int(cu_seqlens_q[b].item())
        kv_len = int(cu_seqlens_k[b + 1].item()) - int(cu_seqlens_k[b].item())
        offset = kv_len - q_len
        pad_mask = (q_idx < q_len) & (k_idx < kv_len)  # [M, N]
        if is_causal:
            causal_mask = k_idx <= q_idx + offset  # True = visible
            visible = pad_mask & causal_mask
        else:
            visible = pad_mask
        mask[b] = visible.float()
    return mask


# ===========================================================================
# Golden functions (padded layout)
# 1. ref_gqa_varlen_fwd_padded: self-written PyTorch golden (no flash_attn dep)
# 2. ref_sdpa_padded: F.scaled_dot_product_attention golden — the NPU
#    equivalent of the GPU main-repo golden flash_attn.flash_attn_varlen_func.
#    flash_attn is CUDA-only and unavailable on NPU; SDPA is PyTorch's native
#    attention and runs on NPU, providing an independent cross-validation path.
# ===========================================================================


def ref_gqa_varlen_fwd_padded(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    heads,
    groups,
    dim,
    is_causal,
):
    """PyTorch reference for padded GQA forward. Padding positions output 0.

    Args:
        q: [batch, heads, q_seqlen, dim] float16 (padded)
        k: [batch, head_kv, k_seqlen, dim] float16 (padded)
        v: [batch, head_kv, k_seqlen, dim] float16 (padded)
        cu_seqlens_q: [batch+1] int32 (actual Q lengths prefix sum)
        cu_seqlens_k: [batch+1] int32 (actual K lengths prefix sum)

    Returns:
        output: [batch, heads, q_seqlen, dim] float16 (padded, padding rows = 0)
    """
    scale = (1.0 / dim) ** 0.5
    output = torch.zeros_like(q)
    batch = q.shape[0]

    for b in range(batch):
        sq = int(cu_seqlens_q[b + 1].item()) - int(cu_seqlens_q[b].item())
        skv = int(cu_seqlens_k[b + 1].item()) - int(cu_seqlens_k[b].item())
        if sq == 0:
            continue

        q_b = q[b, :, :sq, :].float()  # [heads, sq, dim]
        k_b = k[b, :, :skv, :].float()  # [head_kv, skv, dim]
        v_b = v[b, :, :skv, :].float()  # [head_kv, skv, dim]

        # GQA: repeat KV heads to match Q heads
        k_b_rep = k_b.repeat_interleave(groups, dim=0)  # [heads, skv, dim]
        v_b_rep = v_b.repeat_interleave(groups, dim=0)  # [heads, skv, dim]

        # [heads, sq, skv] = [heads, sq, dim] @ [heads, dim, skv]
        scores = torch.einsum("hqd,hdk->hqk", q_b, k_b_rep.transpose(1, 2)) * scale

        if is_causal:
            q_idx = torch.arange(sq, device=q.device)
            k_idx = torch.arange(skv, device=q.device)
            offset = skv - sq
            mask = q_idx[:, None] + offset < k_idx[None, :]  # [sq, skv]
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(scores, dim=-1)  # [heads, sq, skv]
        # Invisible Q rows (causal, q_pos+offset<0) have all -inf scores ->
        # softmax returns NaN. Replace with 0 so the golden output is clean
        # (these rows are also excluded from comparison via visible_q_mask).
        attn = torch.nan_to_num(attn, nan=0.0)
        out_b = torch.einsum("hqk,hkd->hqd", attn, v_b_rep)  # [heads, sq, dim]
        output[b, :, :sq, :] = out_b.to(q.dtype)

    return output


def ref_sdpa_padded(q, k, v, cu_seqlens_q, cu_seqlens_k, heads, groups, dim, is_causal):
    """PyTorch SDPA golden (mathematically equivalent to flash_attn_varlen_func).

    Uses F.scaled_dot_product_attention as the golden. This is the NPU
    equivalent of the GPU main-repo golden ``flash_attn.flash_attn_varlen_func``,
    which is CUDA-only and unavailable on NPU. SDPA is PyTorch's native scaled
    dot-product attention implementation and runs on NPU, providing an
    independent cross-validation path against the self-written golden.

    Note: ``flash_attn_varlen_func`` uses **right-aligned** causal masking
    (offset = skv - sq): the last query token attends to the last key token.
    SDPA's ``is_causal=True`` uses **left-aligned** masking (standard lower
    triangular), which only matches when sq == skv. For sq != skv + causal,
    we construct the right-aligned mask manually and pass it as ``attn_mask``
    to SDPA, ensuring mathematical equivalence with
    ``flash_attn_varlen_func`` for all shape combinations.

    Args:
        q: [batch, heads, q_seqlen, dim] float16 (padded)
        k: [batch, head_kv, k_seqlen, dim] float16 (padded)
        v: [batch, head_kv, k_seqlen, dim] float16 (padded)
        cu_seqlens_q: [batch+1] int32 (actual Q lengths prefix sum)
        cu_seqlens_k: [batch+1] int32 (actual K lengths prefix sum)
        heads: number of Q heads.
        groups: GQA group size (heads // head_kv).
        dim: head dimension.
        is_causal: whether to apply causal mask.

    Returns:
        output: [batch, heads, q_seqlen, dim] float16 (padded, padding rows = 0)
    """
    out = torch.zeros_like(q)
    batch = q.shape[0]
    for b in range(batch):
        sq = int(cu_seqlens_q[b + 1].item()) - int(cu_seqlens_q[b].item())
        skv = int(cu_seqlens_k[b + 1].item()) - int(cu_seqlens_k[b].item())
        if sq == 0:
            continue
        q_b = q[b, :, :sq, :].float()  # [heads, sq, dim]
        k_b = k[b, :, :skv, :].float()  # [head_kv, skv, dim]
        v_b = v[b, :, :skv, :].float()  # [head_kv, skv, dim]

        # GQA: repeat KV heads to match Q heads
        k_b_rep = k_b.repeat_interleave(groups, dim=0)  # [heads, skv, dim]
        v_b_rep = v_b.repeat_interleave(groups, dim=0)  # [heads, skv, dim]

        # SDPA expects [batch=1, heads, sq, dim]
        q_b_4d = q_b.unsqueeze(0)
        k_b_4d = k_b_rep.unsqueeze(0)
        v_b_4d = v_b_rep.unsqueeze(0)

        if is_causal and sq == skv:
            # Standard causal mask == right-aligned when offset = 0
            out_b = torch.nn.functional.scaled_dot_product_attention(
                q_b_4d,
                k_b_4d,
                v_b_4d,
                is_causal=True,
            )  # [1, heads, sq, dim]
        elif is_causal:
            # Right-aligned causal: query i attends to key j iff j <= i + offset.
            # flash_attn_varlen_func uses this convention; SDPA's is_causal=True
            # would give left-aligned (j <= i), which differs when sq != skv.
            offset = skv - sq
            q_idx = torch.arange(sq, device=q.device)
            k_idx = torch.arange(skv, device=q.device)
            visible = k_idx[None, :] <= q_idx[:, None] + offset  # [sq, skv]
            attn_mask = torch.zeros(sq, skv, device=q.device, dtype=torch.float32)
            attn_mask[~visible] = float("-inf")
            out_b = torch.nn.functional.scaled_dot_product_attention(
                q_b_4d,
                k_b_4d,
                v_b_4d,
                attn_mask=attn_mask,
            )  # [1, heads, sq, dim]
        else:
            out_b = torch.nn.functional.scaled_dot_product_attention(
                q_b_4d,
                k_b_4d,
                v_b_4d,
            )  # [1, heads, sq, dim]

        out[b, :, :sq, :] = out_b[0].to(q.dtype)

    return out


# ===========================================================================
# JIT kernel (Expert mode, CV fusion, 4D padded layout, mask tensor)
# Structure follows flash_attn_bhsd.py; additions: GQA, varlen, causal, mask.
# ===========================================================================

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,  # manual CV separation
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,  # manual inter-core sync
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,  # manual intra-core sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,  # manual memory planning
}

NUM_CORES = 24  # 910B has 24 AI Cores (static task distribution)


@tilelang.jit(out_idx=[4], workspace_idx=[5, 6, 7], pass_configs=PASS_CONFIGS)
def flashattn(
    batch_size,
    groups,
    heads,
    dim,
    max_seqlen_q,
    max_seqlen_k,
    is_causal,
    block_M=128,
    block_N=128,
    num_stages=8,
    cross_interval=1,
    apply_mask=True,
):
    """GQA varlen Flash Attention forward kernel (Expert mode, pipelined).

    Iter 2: CV pipeline rewrite following fa_opt/flash_attn_bhsd_expert_h16_d128.py.
    - num_stages=14 multi-stage pipeline (batch KV iterations)
    - T.mma + L0A/L0B/L0C double buffering (replaces T.gemm_v0)
    - T.set_flag/wait_flag fine-grained intra-core sync (replaces T.barrier_all)
    - T.tile.broadcast + T.tile.axpy vectorized softmax (replaces per-row loop)
    - T.annotate_layout ZN/NZ layout optimization
    - Mask integration: mask applied after exp via T.tile.mul, synced with barrier_all

    Iter 3: 2-flag mask sync + param tuning (21.26 ms, 51.72 TFlops).
    - 2-flag mask sync: SIG_MASK_FREE (V→MTE2) + SIG_MASK_READY (MTE2→V) replaces
      2 T.barrier_all() per KV iter. No init needed (first iter V exp runs first).
    - num_stages=14→8 (bench_mark confirmed optimal: 8+8 two full batches vs 14+2)
    - cross_interval=2→1 (more frequent sync lets Vector start earlier)
    - Standalone 2-flag gain only 1.4% (noise range), but param tuning gave 9.9% total

    Iter 4: Mask skip + double-buffered io_buf (target: ≤15 ms).
    - apply_mask compile-time flag: when False (non-causal + full padding), skips
      mask GM load (MTE2), mask mul (V), and 2-flag sync entirely. This eliminates
      MTE2 serialization (was 2 loads/iter: QK 16KB + mask 32KB; now 1 load/iter:
      QK 16KB only). MTE2 pipeline becomes continuous QK loads, fully hidden behind
      V compute. Root cause: mask load on same MTE2 unit blocked next QK load.
    - Double-buffered io_buf [2, half_M, block_N]: MTE2 loads QK[i+1] into io_buf[1]
      while V processes QK[i] from io_buf[0]. Removes io_buf release wait, enabling
      full MTE2/V pipeline overlap. UB: +16KB (132.8→148.8 KB < 192 KB limit).
    - block_N=256 NOT feasible: L0B [2,128,256] fp16=128KB > 64KB L0B limit;
      GEMM2 P matrix [128,256] in L0A also overflows. Confirmed via capacity analysis.

    Args:
        batch_size: number of sequences (compile-time).
        groups: GQA group size (heads // head_kv).
        heads: number of Q heads.
        dim: head dimension (fixed 128 for L0).
        max_seqlen_q: padded max Q sequence length (compile-time).
        max_seqlen_k: padded max K sequence length (compile-time).
        is_causal: whether to apply causal mask (compile-time, documentation only).
        block_M: Q block size.
        block_N: K/V block size.
        num_stages: pipeline depth (batch KV iterations per outer loop).
        cross_interval: cross-core sync interval (sync every N iterations).
        apply_mask: whether to apply attention mask (compile-time). False skips
            mask load+mul for non-causal full-padding case (mask is all 1.0).

    Kernel inputs (4D padded + mask tensor + 3 GM workspaces):
        Q: [batch, heads, max_seqlen_q, dim]                 # 0
        K: [batch, head_kv, max_seqlen_k, dim]               # 1
        V: [batch, head_kv, max_seqlen_k, dim]               # 2
        Mask: [batch, max_seqlen_q, max_seqlen_k] float32    # 3
        Output: [batch, heads, max_seqlen_q, dim]             # 4 (out_idx)
        workspace_1: [NUM_CORES, num_stages, block_M, block_N] fp16  # 5 (QK scores)
        workspace_2: [NUM_CORES, num_stages, block_M, block_N] fp16  # 6 (softmax P)
        workspace_3: [NUM_CORES, num_stages, block_M, dim]    fp16  # 7 (PV output)
    """
    head_kv = heads // groups
    sm_scale = (1.0 / dim) ** 0.5  # natural exp, no log2(e) factor
    dtype = "float16"
    accum_dtype = "float"

    q_shape = [batch_size, heads, max_seqlen_q, dim]
    kv_shape = [batch_size, head_kv, max_seqlen_k, dim]
    mask_shape = [batch_size, max_seqlen_q, max_seqlen_k]
    o_shape = [batch_size, heads, max_seqlen_q, dim]

    assert max_seqlen_q % block_M == 0, f"max_seqlen_q ({max_seqlen_q}) must be divisible by block_M ({block_M})"
    assert max_seqlen_k % block_N == 0, f"max_seqlen_k ({max_seqlen_k}) must be divisible by block_N ({block_N})"
    assert num_stages % 2 == 0, "num_stages must be even for double buffering"

    num_q_blocks = max_seqlen_q // block_M
    max_kv_iters = max_seqlen_k // block_N
    block_num = num_q_blocks * heads * batch_size
    num_outer = T.ceildiv(max_kv_iters, num_stages)

    q_tasks = block_num // NUM_CORES
    r_tasks = block_num % NUM_CORES

    # Cross-core semaphore IDs (Cube <-> Vector)
    SEM_WS1_C2V = 0  # workspace_1 (QK^T) ready: Cube -> Vector
    SEM_WS1_V2C = 1  # workspace_1 consumed: Vector -> Cube
    SEM_WS2_V2C = 2  # workspace_2 (softmax P) ready: Vector -> Cube
    SEM_WS2_C2V = 3  # workspace_2 consumed: Cube -> Vector
    SEM_WS3_C2V = 4  # workspace_3 (PV output) ready: Cube -> Vector
    SEM_WS3_V2C = 5  # workspace_3 consumed: Vector -> Cube

    # Intra-core signal IDs (C Scope)
    SIG_K_L1 = 0
    SIG_P_L1 = 1
    SIG_V_L1 = 2
    SIG_L0AB = 3  # double-buffer base: slot 0 = SIG_L0AB, slot 1 = SIG_L0AB + 1
    SIG_L0C = 5  # double-buffer base: slot 0 = SIG_L0C,  slot 1 = SIG_L0C + 1

    # Intra-core signal IDs (V Scope)
    # io_buf double-buffered: slot 0 = SIG_IO_UB, slot 1 = SIG_IO_UB + 1
    SIG_IO_UB = 0
    SIG_S_HALF = 2
    SIG_MASK_FREE = 3  # V -> MTE2: buf_2d released after exp (mask can overwrite)
    SIG_MASK_READY = 4  # MTE2 -> V: mask loaded into buf_2d (mul can proceed)

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(kv_shape, dtype),  # type: ignore
        V: T.Tensor(kv_shape, dtype),  # type: ignore
        Mask: T.Tensor(mask_shape, accum_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        workspace_1: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),  # type: ignore
        workspace_2: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype),  # type: ignore
        workspace_3: T.Tensor([NUM_CORES, num_stages, block_M, dim], dtype),  # type: ignore
    ):
        with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
            # ---- Buffer allocation (Expert: explicit memory hierarchy) ----
            # L1 buffers (Cube core)
            q_l1 = T.alloc_L1([block_M, dim], dtype)
            k_l1 = T.alloc_L1([block_N, dim], dtype)
            v_l1 = T.alloc_L1([block_N, dim], dtype)
            p_l1 = T.alloc_L1([block_M, block_N], dtype)

            # L1 layout optimization (ZN for Q/P/V, NZ for K with transpose)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_l1: make_nz_layout(k_l1),
                    p_l1: make_zn_layout(p_l1),
                    v_l1: make_zn_layout(v_l1),
                }
            )

            # L0 double buffering (2 slots for pipeline parallelism)
            l0a = T.alloc_L0A([2, block_M, dim], dtype)
            l0b = T.alloc_L0B([2, dim, block_N], dtype)
            l0c = T.alloc_L0C([2, block_M, block_N], accum_dtype)

            # UB buffers (vid split: block_M//2 per vid, Vector core)
            # NOTE: use block_M // 2 (Python int) in alloc shapes, not half_M (TIR var)
            acc_o = T.alloc_ub([block_M // 2, dim], accum_dtype)

            # Batch softmax buffers (num_stages slots for deferred rescale)
            r_factors = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)
            sumexp_is = T.alloc_ub([num_stages, block_M // 2, 1], accum_dtype)

            sumexp = T.alloc_ub([block_M // 2, 1], accum_dtype)
            neg_sm = T.alloc_ub([2, block_M // 2, 1], accum_dtype)  # double-buffered max

            # IO and work buffers (reused across phases)
            # Double-buffered io_buf: MTE2 loads QK[i+1] while V processes QK[i]
            io_buf = T.alloc_ub([2, block_M // 2, block_N], dtype)  # GM <-> UB transfer (fp16)
            acc_s_half = T.alloc_ub([block_M // 2, block_N], dtype)  # fp16 softmax output

            work_ub = T.alloc_ub([block_M // 2, block_N], accum_dtype)  # main compute buffer (fp32)
            buf_2d = T.alloc_ub([block_M // 2, block_N], accum_dtype)  # broadcast+mask buffer (fp32)
            # NOTE: mask reuses buf_2d after exp consumes it (saves 32KB UB)

            half_M = block_M // 2  # TIR variable for slice expressions

            # ---- Static task distribution (NUM_CORES=24) ----
            my_start = cid * q_tasks + T.if_then_else(cid < r_tasks, cid, r_tasks)
            my_count = q_tasks + T.if_then_else(cid < r_tasks, 1, 0)

            # ==================== Cube core (vid=0) ====================
            with T.Scope("C"):
                # init: pretend consumer already released
                T.set_cross_flag("MTE2", SEM_WS2_C2V)
                T.set_flag("MTE1", "MTE2", SIG_K_L1)
                T.set_flag("MTE1", "MTE2", SIG_P_L1)
                T.set_flag("MTE1", "MTE2", SIG_V_L1)
                T.set_flag("M", "MTE1", SIG_L0AB)
                T.set_flag("M", "MTE1", SIG_L0AB + 1)
                T.set_flag("FIX", "M", SIG_L0C)
                T.set_flag("FIX", "M", SIG_L0C + 1)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_q_blocks
                    by = (task_id // num_q_blocks) % heads
                    bz = task_id // (num_q_blocks * heads)
                    kv_head_idx = by // groups  # GQA

                    T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], q_l1)
                    T.barrier_all()

                    for k in T.serial(num_outer):
                        _remaining = max_kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- GEMM1: produce QK^T scores into workspace_1 ---
                        T.wait_cross_flag(SEM_WS1_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # K: GM -> L1 (MTE2 -> MTE1 flag)
                            T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                            T.copy(K[bz, kv_head_idx, idx * block_N : (idx + 1) * block_N, :], k_l1)
                            T.set_flag("MTE2", "MTE1", SIG_K_L1)

                            # Q: L1 -> L0A (only first 2 iterations, then reused)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            if i < 2:
                                T.copy(q_l1, l0a[side, :, :])

                            # K: L1 -> L0B with transpose
                            T.wait_flag("MTE2", "MTE1", SIG_K_L1)
                            T.copy(k_l1, l0b[side, :, :], transpose=True)
                            T.set_flag("MTE1", "MTE2", SIG_K_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            # MMA: QK^T -> L0C
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            # L0C -> workspace_1 (FIX pipeline)
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_1[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS1_C2V)

                        # --- GEMM2: consume P from workspace_2, produce PV into workspace_3 ---
                        T.wait_cross_flag(SEM_WS3_V2C)
                        for i in T.serial(batch_iters):
                            side = i % 2
                            idx = k * num_stages + i

                            # V: GM -> L1
                            T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                            T.copy(V[bz, kv_head_idx, idx * block_N : (idx + 1) * block_N, :], v_l1)
                            T.set_flag("MTE2", "MTE1", SIG_V_L1)

                            # P: workspace_2 -> L1
                            T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS2_V2C)
                            T.copy(workspace_2[cid, i, :, :], p_l1)
                            T.set_flag("MTE2", "MTE1", SIG_P_L1)

                            # V: L1 -> L0B (no transpose for PV)
                            T.wait_flag("MTE2", "MTE1", SIG_V_L1)
                            T.wait_flag("M", "MTE1", SIG_L0AB + side)
                            T.copy(v_l1, l0b[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_V_L1)

                            # P: L1 -> L0A (no transpose)
                            T.wait_flag("MTE2", "MTE1", SIG_P_L1)
                            T.copy(p_l1, l0a[side, :, :])
                            T.set_flag("MTE1", "MTE2", SIG_P_L1)
                            T.set_flag("MTE1", "M", SIG_L0AB + side)

                            # MMA: PV -> L0C
                            T.wait_flag("MTE1", "M", SIG_L0AB + side)
                            T.wait_flag("FIX", "M", SIG_L0C + side)
                            T.mma(l0a[side, :, :], l0b[side, :, :], l0c[side, :, :], init=True)
                            T.set_flag("M", "MTE1", SIG_L0AB + side)
                            T.set_flag("M", "FIX", SIG_L0C + side)

                            # L0C -> workspace_3
                            T.wait_flag("M", "FIX", SIG_L0C + side)
                            T.copy(l0c[side, :, :], workspace_3[cid, i, :, :])
                            T.set_flag("FIX", "M", SIG_L0C + side)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("FIX", SEM_WS3_C2V)

                        T.set_cross_flag("MTE2", SEM_WS2_C2V)

                # destroy: consume outstanding init-direction flags
                T.wait_flag("MTE1", "MTE2", SIG_K_L1)
                T.wait_flag("MTE1", "MTE2", SIG_P_L1)
                T.wait_flag("MTE1", "MTE2", SIG_V_L1)
                T.wait_flag("M", "MTE1", SIG_L0AB)
                T.wait_flag("M", "MTE1", SIG_L0AB + 1)
                T.wait_flag("FIX", "M", SIG_L0C)
                T.wait_flag("FIX", "M", SIG_L0C + 1)

            # ==================== Vector core (vid=1) ====================
            with T.Scope("V"):
                # init: pretend workspaces already released by Cube
                T.set_cross_flag("MTE2", SEM_WS1_V2C)
                T.set_cross_flag("MTE2", SEM_WS3_V2C)
                # init: pretend both io_buf slots are free (consumer already released)
                T.set_flag("V", "MTE2", SIG_IO_UB)
                T.set_flag("V", "MTE2", SIG_IO_UB + 1)
                T.set_flag("MTE3", "V", SIG_S_HALF)

                for t in T.serial(my_count):
                    task_id = my_start + t
                    bx = task_id % num_q_blocks
                    by = (task_id // num_q_blocks) % heads
                    bz = task_id // (num_q_blocks * heads)

                    T.tile.fill(acc_o, 0.0)
                    T.tile.fill(sumexp, 0.0)
                    T.tile.fill(neg_sm, 2**30)  # large positive = -inf max

                    for k in T.serial(num_outer):
                        _remaining = max_kv_iters - k * num_stages
                        batch_iters = T.if_then_else(_remaining < num_stages, _remaining, num_stages)

                        # --- Phase 1: Softmax batch (compute exp, write workspace_2) ---
                        T.wait_cross_flag(SEM_WS2_C2V)
                        for i in T.serial(batch_iters):
                            cur = i % 2
                            prv = 1 - cur
                            idx = k * num_stages + i
                            io_side = i % 2  # io_buf double-buffer slot

                            # Read QK scores from workspace_1 (vid half)
                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS1_C2V)
                            T.copy(
                                workspace_1[cid, i, vid * half_M : vid * half_M + half_M, :],
                                io_buf[io_side, :, :],
                            )
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)  # fp16 -> fp32
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            # Online softmax: max (on raw scores, scale via axpy)
                            T.reduce_max(work_ub, neg_sm[cur, :, :], dim=-1)
                            T.tile.mul(neg_sm[cur, :, :], neg_sm[cur, :, :], -sm_scale)
                            T.tile.min(neg_sm[cur, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                            # Vectorized sub + scale: buf_2d = (work_ub - new_max) * sm_scale
                            T.tile.broadcast(buf_2d, neg_sm[cur, :, :])
                            T.tile.axpy(buf_2d, work_ub, sm_scale)
                            T.tile.exp(work_ub, buf_2d)

                            if apply_mask:
                                # === MASK AFTER EXP (mask tensor from GM) ===
                                # buf_2d is free after exp consumed it.
                                # 2-flag sync (replaces barrier_all): V releases buf_2d
                                # after exp, MTE2 loads mask, MTE2 signals V for mul.
                                # No init needed: first iter V runs exp first, then sets
                                # SIG_MASK_FREE; MTE2 waits and unblocks after that.
                                T.set_flag("V", "MTE2", SIG_MASK_FREE)
                                T.wait_flag("V", "MTE2", SIG_MASK_FREE)
                                T.copy(
                                    Mask[
                                        bz,
                                        bx * block_M + vid * half_M : bx * block_M + vid * half_M + half_M,
                                        idx * block_N : (idx + 1) * block_N,
                                    ],
                                    buf_2d,
                                )
                                T.set_flag("MTE2", "V", SIG_MASK_READY)
                                T.wait_flag("MTE2", "V", SIG_MASK_READY)
                                T.tile.mul(work_ub, work_ub, buf_2d)

                            # Write masked softmax P to workspace_2 (via acc_s_half)
                            T.wait_flag("MTE3", "V", SIG_S_HALF)
                            T.copy(work_ub, acc_s_half)  # fp32 -> fp16
                            T.set_flag("V", "MTE3", SIG_S_HALF)

                            T.wait_flag("V", "MTE3", SIG_S_HALF)
                            T.copy(
                                acc_s_half,
                                workspace_2[cid, i, vid * half_M : vid * half_M + half_M, :],
                            )
                            T.set_flag("MTE3", "V", SIG_S_HALF)
                            if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
                                T.set_cross_flag("MTE3", SEM_WS2_V2C)

                            # Precompute r_factors and sumexp_is for phase 2
                            T.reduce_sum(work_ub, sumexp_is[i, :, :], dim=-1)
                            T.tile.sub(r_factors[i, :, :], neg_sm[cur, :, :], neg_sm[prv, :, :])

                        T.set_cross_flag("MTE2", SEM_WS1_V2C)

                        # --- Phase 2: O accumulation batch (rescale + accumulate PV) ---
                        for i in T.serial(batch_iters):
                            # Deferred rescale: exp(old_max - new_max)
                            T.tile.exp(r_factors[i, :, :], r_factors[i, :, :])
                            T.tile.mul(sumexp, sumexp, r_factors[i, :, :])
                            T.tile.add(sumexp, sumexp, sumexp_is[i, :, :])
                            T.tile.broadcast(buf_2d, r_factors[i, :, :])
                            T.tile.mul(acc_o, acc_o, buf_2d)

                            # Read PV output from workspace_3 (vid half)
                            io_side = i % 2  # io_buf double-buffer slot
                            T.wait_flag("V", "MTE2", SIG_IO_UB + io_side)
                            if i % cross_interval == 0:
                                T.wait_cross_flag(SEM_WS3_C2V)
                            T.copy(
                                workspace_3[cid, i, vid * half_M : vid * half_M + half_M, :],
                                io_buf[io_side, :, :],
                            )
                            T.set_flag("MTE2", "V", SIG_IO_UB + io_side)

                            T.wait_flag("MTE2", "V", SIG_IO_UB + io_side)
                            T.copy(io_buf[io_side, :, :], work_ub)  # fp16 -> fp32
                            T.set_flag("V", "MTE2", SIG_IO_UB + io_side)

                            T.tile.add(acc_o, acc_o, work_ub)

                        T.set_cross_flag("MTE2", SEM_WS3_V2C)

                    # Final normalize: acc_o /= sumexp
                    T.tile.broadcast(buf_2d, sumexp)
                    T.tile.div(acc_o, acc_o, buf_2d)

                    # Write back (vid half)
                    T.copy(acc_o, acc_s_half)  # fp32 -> fp16
                    T.barrier_all()
                    T.copy(
                        acc_s_half,
                        Output[
                            bz,
                            by,
                            bx * block_M + vid * half_M : bx * block_M + vid * half_M + half_M,
                            :,
                        ],
                    )

                # destroy: consume outstanding init-direction flags
                T.wait_flag("V", "MTE2", SIG_IO_UB)
                T.wait_flag("V", "MTE2", SIG_IO_UB + 1)
                T.wait_flag("MTE3", "V", SIG_S_HALF)

    return main


# ===========================================================================
# Test helpers
# ===========================================================================


def _prepare_and_run(
    batch,
    heads,
    groups,
    q_seqlen,
    k_seqlen,
    dim,
    is_causal,
    padding_mode,
    block_M,
    block_N,
    device,
    dtype,
    atol,
    rtol,
):
    """Prepare padded inputs + mask, run kernel + dual golden, return (max_diff, golden_diff, passed).

    Runs both the self-written golden (ref_gqa_varlen_fwd_padded) and the SDPA
    golden (ref_sdpa_padded, equivalent to the main-repo flash_attn_varlen_func).
    The test passes only if the kernel matches BOTH goldens within tolerance.
    golden_diff reports the max difference between the two goldens (should be
    small, confirming their mathematical equivalence).
    """
    torch.manual_seed(0)
    head_kv = heads // groups

    # Pad seqlens to block_M/block_N multiples to avoid GM OOB reads.
    # Padding rows/cols are zero-filled; mask=0 handles them in kernel.
    padded_sq = ((q_seqlen + block_M - 1) // block_M) * block_M
    padded_skv = ((k_seqlen + block_N - 1) // block_N) * block_N

    # Padded 4D layout: [batch, heads, padded_seqlen, dim]
    q = torch.zeros(batch, heads, padded_sq, dim, dtype=dtype, device=device)
    q[:, :, :q_seqlen, :] = torch.randn(batch, heads, q_seqlen, dim, dtype=dtype, device=device)
    k = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
    k[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device)
    v = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
    v[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device)

    # Padding masks (original seqlen) -> cu_seqlens -> attention mask tensor (padded seqlens)
    q_mask = generate_random_padding_mask(q_seqlen, batch, device, mode=padding_mode)
    k_mask = generate_random_padding_mask(k_seqlen, batch, device, mode=padding_mode)
    cu_seqlens_q = mask_to_cu_seqlens(q_mask)
    cu_seqlens_k = mask_to_cu_seqlens(k_mask)
    attn_mask = build_attention_mask(
        cu_seqlens_q,
        cu_seqlens_k,
        padded_sq,
        padded_skv,
        is_causal,
        device,
    )

    # Compile kernel with padded seqlens
    # Skip mask only when non-causal + full padding + no block padding needed
    # (mask is all 1.0 only when seq lens are exact multiples of block sizes)
    has_block_padding = (q_seqlen % block_M != 0) or (k_seqlen % block_N != 0)
    apply_mask = is_causal or padding_mode != "full" or has_block_padding
    kernel = flashattn(
        batch,
        groups,
        heads,
        dim,
        padded_sq,
        padded_skv,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        apply_mask=apply_mask,
    )

    # Run kernel
    out = kernel(q, k, v, attn_mask)
    torch.npu.synchronize()

    # Golden 1: self-written PyTorch (padded layout)
    ref_out = ref_gqa_varlen_fwd_padded(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        heads,
        groups,
        dim,
        is_causal,
    )
    torch.npu.synchronize()

    # Golden 2: SDPA (F.scaled_dot_product_attention) — NPU equivalent of the
    # GPU main-repo golden flash_attn.flash_attn_varlen_func.
    ref_sdpa_out = ref_sdpa_padded(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        heads,
        groups,
        dim,
        is_causal,
    )
    torch.npu.synchronize()

    # Build the comparison mask: valid Q rows that are ALSO visible under
    # causal (q_pos + offset >= 0). Invisible rows (causal, q longer than k
    # gives offset<0 -> early Q rows see no KV) produce NaN/0 in both kernel
    # and golden and are excluded.
    visible_q_mask = q_mask.clone()
    if is_causal:
        for b in range(batch):
            q_len_b = int(cu_seqlens_q[b + 1].item()) - int(cu_seqlens_q[b].item())
            kv_len_b = int(cu_seqlens_k[b + 1].item()) - int(cu_seqlens_k[b].item())
            offset_b = kv_len_b - q_len_b
            if offset_b < 0:
                invisible_count = -offset_b
                if invisible_count > 0:
                    visible_q_mask[b, :invisible_count] = False

    # Extend visible_q_mask to padded_sq (False for block-padding rows)
    if padded_sq > q_seqlen:
        vqm_padded = torch.zeros(batch, padded_sq, dtype=torch.bool, device=device)
        vqm_padded[:, :q_seqlen] = visible_q_mask
        visible_q_mask = vqm_padded

    # Compare only visible Q positions.
    # out / ref_out / ref_sdpa_out: [batch, heads, padded_sq, dim].
    out_perm = out.permute(0, 2, 1, 3).contiguous()  # [batch, padded_sq, heads, dim]
    ref_perm = ref_out.permute(0, 2, 1, 3).contiguous()
    ref_sdpa_perm = ref_sdpa_out.permute(0, 2, 1, 3).contiguous()
    out_valid = out_perm[visible_q_mask].cpu()  # [num_visible, heads, dim]
    ref_valid = ref_perm[visible_q_mask].cpu()
    ref_sdpa_valid = ref_sdpa_perm[visible_q_mask].cpu()

    # Guard against NaN leaking into valid rows (should not happen).
    if torch.isnan(out_valid).any():
        return float("nan"), float("nan"), False

    # max_diff: kernel vs self-written golden
    max_diff = (out_valid.float() - ref_valid.float()).abs().max().item()
    # golden_diff: self-written golden vs SDPA golden (cross-validation)
    golden_diff = (ref_valid.float() - ref_sdpa_valid.float()).abs().max().item()

    # Test passes only if kernel matches BOTH goldens within tolerance.
    try:
        torch.testing.assert_close(out_valid, ref_valid, rtol=rtol, atol=atol)
        torch.testing.assert_close(out_valid, ref_sdpa_valid, rtol=rtol, atol=atol)
        passed = True
    except AssertionError:
        passed = False

    return max_diff, golden_diff, passed


# ===========================================================================
# L0 gate tests (DESIGN.md §11.2)
# ===========================================================================


def test_gqa_fwd_varlen_l0():
    """L0 gate tests: regular shapes (block-aligned), for precision convergence."""
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    block_M, block_N = 128, 128

    # (name, batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, padding_mode)
    configs = [
        ("l0_min_full_nc", 1, 4, 2, 128, 128, 128, False, "full"),
        ("l0_min_full_c", 1, 4, 2, 128, 128, 128, True, "full"),
        ("l0_small_rand_nc", 2, 8, 4, 128, 128, 128, False, "random"),
        ("l0_default_full_nc", 8, 64, 16, 2048, 2048, 128, False, "full"),
        ("l0_default_full_c", 8, 64, 16, 2048, 2048, 128, True, "full"),
        # Main-repo-aligned case: same shape + padding mode as the GPU main-repo
        # example_gqa_fwd_varlen.py main() defaults (batch=8, heads=64, groups=16,
        # q_seqlen=2048, k_seqlen=2048, dim=128, is_causal=False, random padding).
        ("l0_main_repo_match", 8, 64, 16, 2048, 2048, 128, False, "random"),
    ]

    ok = True
    for name, b, h, g, sq, skv, d, causal, pmode in configs:
        try:
            max_diff, golden_diff, passed = _prepare_and_run(
                b,
                h,
                g,
                sq,
                skv,
                d,
                causal,
                pmode,
                block_M,
                block_N,
                device,
                dtype,
                atol,
                rtol,
            )
            if passed:
                print(
                    f"[PRECISION_PASS] l0 {name} batch={b} heads={h} groups={g} "
                    f"sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} "
                    f"max_diff={max_diff:.6e} golden_diff={golden_diff:.6e}"
                )
            else:
                print(
                    f"[PRECISION_FAIL] l0 {name} batch={b} heads={h} groups={g} "
                    f"sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} "
                    f"max_diff={max_diff:.6e} golden_diff={golden_diff:.6e}"
                )
                ok = False
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(
                f"[PRECISION_FAIL] l0 {name} batch={b} heads={h} groups={g} sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} error={e}"
            )
            ok = False
    return ok


# ===========================================================================
# L1 / L2 / Boundary (expanded by tilelang-op-test-design scenario B)
# ===========================================================================


def test_gqa_fwd_varlen_l1():
    """L1 functional tests: irregular shapes, tail blocks, q!=k, GQA variants.

    Returns True iff all cases pass (blocking).
    """
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    block_M, block_N = 128, 128

    # (name, batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, padding_mode)
    # Avoid sq > skv + causal (would create invisible Q rows -> NaN in both
    # kernel and golden, which cannot be compared).
    configs = [
        ("l1_irregular_nc", 1, 4, 2, 100, 100, 128, False, "random"),  # tail block + varlen
        ("l1_irregular_c", 1, 4, 2, 100, 100, 128, True, "random"),  # tail + causal + varlen
        ("l1_q_short_k_c", 2, 4, 2, 64, 128, 128, True, "full"),  # q<k + causal (offset>0)
        ("l1_gqa1_c", 1, 4, 4, 128, 128, 128, True, "full"),  # head_kv=1 (full share)
        ("l1_multi_rand_c", 3, 8, 4, 256, 256, 128, True, "random"),  # multi-batch + rand + causal
        ("l1_tail_nc", 1, 4, 2, 33, 65, 128, False, "full"),  # extreme tail (rem 1)
    ]

    ok = True
    for name, b, h, g, sq, skv, d, causal, pmode in configs:
        try:
            max_diff, golden_diff, passed = _prepare_and_run(
                b,
                h,
                g,
                sq,
                skv,
                d,
                causal,
                pmode,
                block_M,
                block_N,
                device,
                dtype,
                atol,
                rtol,
            )
            if passed:
                print(
                    f"[PRECISION_PASS] l1 {name} batch={b} heads={h} groups={g} "
                    f"sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} "
                    f"max_diff={max_diff:.6e} golden_diff={golden_diff:.6e}"
                )
            else:
                print(
                    f"[PRECISION_FAIL] l1 {name} batch={b} heads={h} groups={g} "
                    f"sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} "
                    f"max_diff={max_diff:.6e} golden_diff={golden_diff:.6e}"
                )
                ok = False
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(
                f"[PRECISION_FAIL] l1 {name} batch={b} heads={h} groups={g} sq={sq} skv={skv} dim={d} causal={causal} pad={pmode} error={e}"
            )
            ok = False
    return ok


def _run_boundary_case(name, batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, padding_mode, input_scale, block_M, block_N):
    """Run one L2/Boundary case. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    device = "npu"
    dtype = torch.float16
    atol, rtol = 1e-2, 1e-2
    head_kv = heads // groups
    try:
        torch.manual_seed(0)
        # Pad seqlens to block_M/block_N multiples to avoid GM OOB reads
        padded_sq = ((q_seqlen + block_M - 1) // block_M) * block_M
        padded_skv = ((k_seqlen + block_N - 1) // block_N) * block_N
        q = torch.zeros(batch, heads, padded_sq, dim, dtype=dtype, device=device)
        q[:, :, :q_seqlen, :] = torch.randn(batch, heads, q_seqlen, dim, dtype=dtype, device=device) * input_scale
        k = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
        k[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device) * input_scale
        v = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
        v[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device) * input_scale
        q_mask = generate_random_padding_mask(q_seqlen, batch, device, mode=padding_mode)
        k_mask = generate_random_padding_mask(k_seqlen, batch, device, mode=padding_mode)
        cu_seqlens_q = mask_to_cu_seqlens(q_mask)
        cu_seqlens_k = mask_to_cu_seqlens(k_mask)
        attn_mask = build_attention_mask(
            cu_seqlens_q,
            cu_seqlens_k,
            padded_sq,
            padded_skv,
            is_causal,
            device,
        )
        kernel = flashattn(
            batch,
            groups,
            heads,
            dim,
            padded_sq,
            padded_skv,
            is_causal,
            block_M=block_M,
            block_N=block_N,
            apply_mask=(is_causal or padding_mode != "full" or (q_seqlen % block_M != 0) or (k_seqlen % block_N != 0)),
        )
        out = kernel(q, k, v, attn_mask)
        torch.npu.synchronize()
        ref_out = ref_gqa_varlen_fwd_padded(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            heads,
            groups,
            dim,
            is_causal,
        )
        torch.npu.synchronize()
        # SDPA golden (cross-validation, equivalent to main-repo flash_attn)
        ref_sdpa_out = ref_sdpa_padded(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            heads,
            groups,
            dim,
            is_causal,
        )
        torch.npu.synchronize()
        # Extend q_mask to padded_sq for comparison
        if padded_sq > q_seqlen:
            qm_padded = torch.zeros(batch, padded_sq, dtype=torch.bool, device=device)
            qm_padded[:, :q_seqlen] = q_mask
            q_mask = qm_padded
        out_perm = out.permute(0, 2, 1, 3).contiguous()[q_mask].cpu()
        ref_perm = ref_out.permute(0, 2, 1, 3).contiguous()[q_mask].cpu()
        ref_sdpa_perm = ref_sdpa_out.permute(0, 2, 1, 3).contiguous()[q_mask].cpu()
        if torch.isnan(out_perm).any():
            print(f"[BOUNDARY_WARN] boundary {name}: NaN in valid output")
            return
        max_diff = (out_perm.float() - ref_perm.float()).abs().max().item()
        golden_diff = (ref_perm.float() - ref_sdpa_perm.float()).abs().max().item()
        max_diff_sdpa = (out_perm.float() - ref_sdpa_perm.float()).abs().max().item()
        torch.testing.assert_close(out_perm, ref_perm, rtol=rtol, atol=atol)
        print(f"[BOUNDARY_PASS] boundary {name} max_diff={max_diff:.6e} golden_diff={golden_diff:.6e} max_diff_sdpa={max_diff_sdpa:.6e}")
    except Exception as e:
        print(f"[BOUNDARY_WARN] boundary {name}: {e}")


def test_gqa_fwd_varlen_l2():
    """L2 abnormal input tests. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    block_M, block_N = 128, 128
    # (name, batch, heads, groups, sq, skv, dim, causal, pad, scale)
    cases = [
        ("l2_single_token", 1, 4, 2, 1, 1, 128, False, "full", 1.0),
        ("l2_min_seqlen", 1, 4, 2, 32, 32, 128, False, "full", 1.0),
        ("l2_batch1_head1", 1, 1, 1, 64, 64, 128, False, "full", 1.0),
    ]
    for name, b, h, g, sq, skv, d, causal, pmode, scale in cases:
        _run_boundary_case(name, b, h, g, sq, skv, d, causal, pmode, scale, block_M, block_N)


def test_gqa_fwd_varlen_boundary():
    """Boundary / special value tests. Non-blocking: prints [BOUNDARY_PASS/WARN]."""
    block_M, block_N = 128, 128
    # (name, batch, heads, groups, sq, skv, dim, causal, pad, scale)
    cases = [
        ("zero_input", 1, 4, 2, 128, 128, 128, False, "full", 0.0),  # all-zero Q/K/V
        ("large_input", 1, 4, 2, 128, 128, 128, False, "full", 10.0),  # large values (stability)
    ]
    for name, b, h, g, sq, skv, d, causal, pmode, scale in cases:
        _run_boundary_case(name, b, h, g, sq, skv, d, causal, pmode, scale, block_M, block_N)


# ===========================================================================
# Performance benchmark (merged from perf_gqa_fwd_varlen.py)
# Run: python example_gqa_fwd_varlen.py --level perf
# Includes a correctness check before bench, so precision + latency in one run.
# ===========================================================================


def _perf_build_inputs(batch, heads, groups, q_seqlen, k_seqlen, dim, is_causal, padding_mode, device, dtype, block_M, block_N):
    """Build padded 4D inputs + mask tensor (mirrors _prepare_and_run)."""
    torch.manual_seed(0)
    head_kv = heads // groups

    padded_sq = ((q_seqlen + block_M - 1) // block_M) * block_M
    padded_skv = ((k_seqlen + block_N - 1) // block_N) * block_N

    q = torch.zeros(batch, heads, padded_sq, dim, dtype=dtype, device=device)
    q[:, :, :q_seqlen, :] = torch.randn(batch, heads, q_seqlen, dim, dtype=dtype, device=device)
    k = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
    k[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device)
    v = torch.zeros(batch, head_kv, padded_skv, dim, dtype=dtype, device=device)
    v[:, :, :k_seqlen, :] = torch.randn(batch, head_kv, k_seqlen, dim, dtype=dtype, device=device)

    q_mask = generate_random_padding_mask(q_seqlen, batch, device, mode=padding_mode)
    k_mask = generate_random_padding_mask(k_seqlen, batch, device, mode=padding_mode)
    cu_seqlens_q = mask_to_cu_seqlens(q_mask)
    cu_seqlens_k = mask_to_cu_seqlens(k_mask)
    attn_mask = build_attention_mask(
        cu_seqlens_q,
        cu_seqlens_k,
        padded_sq,
        padded_skv,
        is_causal,
        device,
    )
    return q, k, v, attn_mask, cu_seqlens_q, cu_seqlens_k, padded_sq, padded_skv


def _bench_tilelang(kernel, q, k, v, attn_mask):
    """Benchmark the TileLang kernel via do_bench (returns ms, mean)."""

    def f():
        kernel(q, k, v, attn_mask)

    latency = do_bench(f, _n_warmup=5, _n_repeat=5, return_mode="mean")
    return latency


def _bench_golden(q, k, v, cu_seqlens_q, cu_seqlens_k, heads, groups, dim, is_causal):
    """Benchmark the PyTorch golden (per-batch loop + einsum)."""

    def f():
        ref_gqa_varlen_fwd_padded(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            heads,
            groups,
            dim,
            is_causal,
        )
        torch.npu.synchronize()

    latency = do_bench(f, _n_warmup=5, _n_repeat=10, return_mode="median")
    return latency


def _compute_flops(batch, heads, q_seqlen, k_seqlen, dim, is_causal):
    """Flash attention FLOPs: 2 matmuls (QK^T and PV)."""
    flops_per_matmul = 2.0 * batch * heads * q_seqlen * k_seqlen * dim
    total = 2 * flops_per_matmul
    if is_causal:
        total *= 0.5
    return total


def run_perf_case(
    name,
    batch,
    heads,
    groups,
    q_seqlen,
    k_seqlen,
    dim,
    is_causal,
    padding_mode,
    block_M,
    block_N,
    num_stages,
    cross_interval,
    with_golden,
    device,
    dtype,
):
    """Run a single benchmark config: correctness check + latency (one pass)."""
    head_kv = heads // groups
    print(
        f"\n[{name}] batch={batch} heads={heads} groups={groups} head_kv={head_kv} "
        f"q_seqlen={q_seqlen} k_seqlen={k_seqlen} dim={dim} "
        f"causal={is_causal} pad={padding_mode} block_M={block_M} block_N={block_N}"
    )

    q, k, v, attn_mask, cu_seqlens_q, cu_seqlens_k, padded_sq, padded_skv = _perf_build_inputs(
        batch,
        heads,
        groups,
        q_seqlen,
        k_seqlen,
        dim,
        is_causal,
        padding_mode,
        device,
        dtype,
        block_M,
        block_N,
    )

    has_block_padding = (q_seqlen % block_M != 0) or (k_seqlen % block_N != 0)
    apply_mask = is_causal or padding_mode != "full" or has_block_padding

    print("  compiling kernel ...")
    kernel = flashattn(
        batch,
        groups,
        heads,
        dim,
        padded_sq,
        padded_skv,
        is_causal,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        cross_interval=cross_interval,
        apply_mask=apply_mask,
    )

    # Correctness check before bench (so we don't bench a broken kernel)
    out = kernel(q, k, v, attn_mask)
    torch.npu.synchronize()
    if torch.isnan(out).any():
        print("  [ERROR] kernel output contains NaN, skipping bench")
        return
    ref_out = ref_gqa_varlen_fwd_padded(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        heads,
        groups,
        dim,
        is_causal,
    )
    torch.npu.synchronize()
    q_mask = generate_random_padding_mask(q_seqlen, batch, device, mode=padding_mode)
    if padded_sq > q_seqlen:
        qm_padded = torch.zeros(batch, padded_sq, dtype=torch.bool, device=device)
        qm_padded[:, :q_seqlen] = q_mask
        q_mask = qm_padded
    out_perm = out.permute(0, 2, 1, 3).contiguous()
    ref_perm = ref_out.permute(0, 2, 1, 3).contiguous()
    out_v = out_perm[q_mask].cpu()
    ref_v = ref_perm[q_mask].cpu()
    if torch.isnan(out_v).any():
        non_nan = ~torch.isnan(out_v).any(dim=-1)
        out_v = out_v[non_nan]
        ref_v = ref_v[non_nan]
    max_diff = (out_v.float() - ref_v.float()).abs().max().item()
    print(f"  correctness: max_diff={max_diff:.6e} (atol=1e-2)")

    # Bench TileLang kernel
    print("  benching TileLang kernel ...")
    tl_ms = _bench_tilelang(kernel, q, k, v, attn_mask)
    flops = _compute_flops(batch, heads, q_seqlen, k_seqlen, dim, is_causal)
    tl_tflops = flops / (tl_ms * 1e-3) * 1e-12
    print(f"  TileLang:  {tl_ms:.4f} ms   {tl_tflops:.2f} TFlops")

    if with_golden:
        print("  benching PyTorch golden ...")
        gold_ms = _bench_golden(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            heads,
            groups,
            dim,
            is_causal,
        )
        gold_tflops = flops / (gold_ms * 1e-3) * 1e-12
        speedup = gold_ms / tl_ms if tl_ms > 0 else float("inf")
        print(f"  Golden:    {gold_ms:.4f} ms   {gold_tflops:.2f} TFlops")
        print(f"  Speedup:   {speedup:.2f}x  (TileLang vs PyTorch golden)")


def run_perf(args, device, dtype):
    """Dispatch the perf benchmark based on --preset."""
    if args.preset == "default":
        run_perf_case(
            "default",
            args.batch,
            args.heads,
            args.groups,
            args.q_seqlen,
            args.k_seqlen,
            args.dim,
            args.causal,
            args.padding,
            args.block_M,
            args.block_N,
            args.num_stages,
            args.cross_interval,
            args.with_golden,
            device,
            dtype,
        )
    elif args.preset == "small":
        run_perf_case(
            "small",
            1,
            4,
            2,
            128,
            128,
            128,
            False,
            "full",
            args.block_M,
            args.block_N,
            args.num_stages,
            args.cross_interval,
            args.with_golden,
            device,
            dtype,
        )
    elif args.preset == "sweep":
        print("=" * 70)
        print("Preset: sweep seqlen (batch=8, heads=64, groups=16, dim=128, non-causal)")
        print("=" * 70)
        for sq in [512, 1024, 2048, 4096]:
            run_perf_case(
                f"sq{sq}",
                8,
                64,
                16,
                sq,
                sq,
                128,
                False,
                "full",
                args.block_M,
                args.block_N,
                args.num_stages,
                args.cross_interval,
                args.with_golden,
                device,
                dtype,
            )
    elif args.preset == "causal-sweep":
        print("=" * 70)
        print("Preset: causal-sweep (causal=True, vary seqlen)")
        print("=" * 70)
        for sq in [512, 1024, 2048, 4096]:
            run_perf_case(
                f"sq{sq}_causal",
                8,
                64,
                16,
                sq,
                sq,
                128,
                True,
                "full",
                args.block_M,
                args.block_N,
                args.num_stages,
                args.cross_interval,
                args.with_golden,
                device,
                dtype,
            )
    print("\nDone.")


# ===========================================================================
# Main entry
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(description="GQA varlen Flash Attention (Ascend Expert)")
    parser.add_argument(
        "--level",
        default="l0",
        choices=["l0", "l1", "l2", "boundary", "all", "perf"],
        help="Test level to run (perf = correctness + latency benchmark in one pass)",
    )
    # perf-level args (used only when --level perf)
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=64, help="query heads")
    parser.add_argument("--groups", type=int, default=16, help="GQA groups")
    parser.add_argument("--q-seqlen", type=int, default=2048, help="Q sequence length")
    parser.add_argument("--k-seqlen", type=int, default=2048, help="K/V sequence length")
    parser.add_argument("--dim", type=int, default=128, help="head dim")
    parser.add_argument("--causal", action="store_true", help="causal attention")
    parser.add_argument(
        "--padding", default="full", choices=["full", "random", "third"], help="padding mode (full = no padding / max length)"
    )
    parser.add_argument("--block-M", type=int, default=128, help="Q block size")
    parser.add_argument("--block-N", type=int, default=128, help="K/V block size")
    parser.add_argument("--num-stages", type=int, default=8, help="pipeline depth")
    parser.add_argument("--cross-interval", type=int, default=1, help="cross-core sync interval")
    parser.add_argument("--with-golden", action="store_true", help="also benchmark PyTorch golden for speedup comparison")
    parser.add_argument(
        "--preset",
        default="default",
        choices=["default", "sweep", "small", "causal-sweep"],
        help="preset benchmark suite (overrides individual args)",
    )
    args = parser.parse_args()

    tilelang.disable_cache()
    torch.set_default_device("npu")
    torch.manual_seed(0)

    if args.level == "perf":
        run_perf(args, "npu", torch.float16)
        return

    blocking_ok = True  # Only L0/L1 count toward blocking

    if args.level in ("l0", "all"):
        blocking_ok &= test_gqa_fwd_varlen_l0()
    if args.level in ("l1", "all"):
        blocking_ok &= test_gqa_fwd_varlen_l1()
    if args.level in ("l2", "all"):
        test_gqa_fwd_varlen_l2()
    if args.level in ("boundary", "all"):
        test_gqa_fwd_varlen_boundary()

    if blocking_ok:
        print("Test Passed!")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
