import math
import torch
import triton
import triton.language as tl

# Experimental: Dont use in prod.
# Implement MLA deepseek style https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf
@triton.autotune(
    configs=[
        triton.Config({'BN': 32, 'BM': 32}, num_stages=1, num_warps=4),
        triton.Config({'BN': 64, 'BM': 32}, num_stages=2, num_warps=4),
        triton.Config({'BN': 32, 'BM': 64}, num_stages=2, num_warps=4),
    ],
    key=['N', 'rank', 'rope']
)
@triton.jit
def _mqa_enc_kernel(
    Q, K, V, O, pads, scale,
    qSb, qSh, qSn, qSd,
    kSb, kSm, kSd,
    vSb, vSm, vSd,
    oSb, oSh, oSn, oSd,
    N, rank: tl.constexpr, rope: tl.constexpr,
    BN: tl.constexpr, BM: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    nOff = tl.program_id(2) * BN

    pads += b
    p = tl.load(pads, mask=True)
    if (nOff + BN) <= p:
        return

    Q += b * qSb + h * qSh + nOff * qSn
    K += b * kSb + p * kSm
    V += b * vSb
    O += b * oSb + h * oSh + nOff * oSn

    nn = tl.arange(0, BN)
    mm = tl.arange(0, BM)
    dt = Q.type.element_ty

    qLo = Q + nn[:, None] * qSn + tl.arange(0, rank)[None, :]
    qRo = Q + nn[:, None] * qSn + rank + tl.arange(0, rope)[None, :]
    kLo = K + mm[None, :] * kSm + tl.arange(0, rank)[:, None]
    kRo = K + mm[None, :] * kSm + rank + tl.arange(0, rope)[:, None]

    maskN = (nOff + nn) < N
    ql = tl.load(qLo, mask=maskN[:, None], other=0.)
    qr = tl.load(qRo, mask=maskN[:, None], other=0.)

    accum = tl.zeros((BN, rank), dtype=tl.float32)
    mx = tl.full((BN,), float('-inf'), dtype=tl.float32)
    sf = tl.zeros((BN,), dtype=tl.float32)

    limit = nOff + BN
    for mStart in range(p, limit, BM):
        maskM = (mStart + mm) < N
        kl = tl.load(kLo, mask=maskM[None, :], other=0.)
        kr = tl.load(kRo, mask=maskM[None, :], other=0.)

        sc = tl.dot(qr, kr) + tl.dot(ql, kl)
        sc = sc * scale

        # Causal mask
        valid = (nOff + nn)[:, None] >= (mStart + mm)[None, :]
        sc = tl.where(valid & maskM[None, :], sc, float('-inf'))

        m_ij = tl.max(sc, axis=1)
        new_mx = tl.maximum(mx, m_ij)
        alpha = tl.exp(mx - new_mx)
        esc = tl.exp(sc - new_mx[:, None])

        sf = sf * alpha + tl.sum(esc, axis=-1)
        accum = accum * alpha[:, None]
        accum = tl.dot(esc.to(dt), tl.trans(kl, 1, 0), acc=accum)
        mx = new_mx

        kLo += BM * kSm
        kRo += BM * kSm

    accum = accum / sf[:, None]
    outPtr = O + nn[:, None] * oSn + tl.arange(0, rank)[None, :]
    tl.store(outPtr, accum.to(dt), mask=maskN[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BM': 32, 'BH': 16}, num_stages=1, num_warps=4),
        triton.Config({'BM': 64, 'BH': 16}, num_stages=1, num_warps=4),
        triton.Config({'BM': 32, 'BH': 32}, num_stages=2, num_warps=4),
    ],
    key=['M', 'rank', 'rope']
)
@triton.jit
def _mqa_dec_kernel(
    Q, K, V, O, pads, scale,
    qSb, qSh, qSn, qSd,
    kSb, kSm, kSd,
    vSb, vSm, vSd,
    oSb, oSh, oSn, oSd,
    M, H, rank: tl.constexpr, rope: tl.constexpr,
    BH: tl.constexpr, BM: tl.constexpr
):
    """Single-step decode (N=1)."""
    b = tl.program_id(0)
    hOff = tl.program_id(1) * BH

    pads += b
    p = tl.load(pads, mask=True)

    Q += b * qSb + hOff * qSh
    K += b * kSb + p * kSm
    V += b * vSb
    O += b * oSb + hOff * oSh

    hh = tl.arange(0, BH)
    mm = tl.arange(0, BM)
    dt = Q.type.element_ty

    # Pointers
    qLo = Q + hh[:, None] * qSh + tl.arange(0, rank)[None, :]
    qRo = Q + hh[:, None] * qSh + rank + tl.arange(0, rope)[None, :]
    kLo = K + mm[None, :] * kSm + tl.arange(0, rank)[:, None]
    kRo = K + mm[None, :] * kSm + rank + tl.arange(0, rope)[:, None]

    maskH = (hOff + hh) < H
    ql = tl.load(qLo, mask=maskH[:, None], other=0.)
    qr = tl.load(qRo, mask=maskH[:, None], other=0.)

    accum = tl.zeros((BH, rank), dtype=tl.float32)
    mx = tl.full((BH,), float('-inf'), dtype=tl.float32)
    sf = tl.zeros((BH,), dtype=tl.float32)

    for mStart in range(p, M, BM):
        maskM = (mStart + mm) < M
        kl = tl.load(kLo, mask=maskM[None, :], other=0.)
        kr = tl.load(kRo, mask=maskM[None, :], other=0.)

        sc = (tl.dot(qr, kr) + tl.dot(ql, kl)) * scale
        sc = tl.where(maskM[None, :], sc, float('-inf'))

        m_ij = tl.max(sc, axis=1)
        new_mx = tl.maximum(mx, m_ij)
        alpha = tl.exp(mx - new_mx)
        esc = tl.exp(sc - new_mx[:, None])

        sf = sf * alpha + tl.sum(esc, axis=-1)
        accum = accum * alpha[:, None]
        accum = tl.dot(esc.to(dt), tl.trans(kl, 1, 0), acc=accum)
        mx = new_mx

        kLo += BM * kSm
        kRo += BM * kSm

    accum /= sf[:, None]
    outPtr = O + hh[:, None] * oSh + tl.arange(0, rank)[None, :]
    tl.store(outPtr, accum.to(dt), mask=maskH[:, None])



@triton.autotune(
    configs=[
        triton.Config({'BM': 32, 'BH': 32, 'BK': 64},  num_stages=2, num_warps=4),
        triton.Config({'BM': 32, 'BH': 16, 'BK': 64},  num_stages=2, num_warps=4),
        triton.Config({'BM': 64, 'BH': 16, 'BK': 64},  num_stages=1, num_warps=4),
    ],
    key=['M', 'rank', 'rope']
)
@triton.jit
def _compute_qk (
    Q, K, QK, pads, scale,
    qSb, qSh, qSn, qSd,
    kSb, kSm, kSd,
    M, H, rank: tl.constexpr, rope: tl.constexpr,
    BH: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr
):
    """Stage1: Q*K^T -> QK buffer."""
    b = tl.program_id(0)
    hOff = tl.program_id(1) * BH
    mOff = tl.program_id(2) * BM

    pads += b
    p = tl.load(pads, mask=True)
    if (mOff + BM) <= p:
        return

    Q += b * qSb + hOff * qSh
    K += b * kSb + mOff * kSm
    QK += b * (M * H) + hOff * M + mOff

    hh = tl.arange(0, BH)
    mm = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    dt = Q.type.element_ty

    # LoRA / RoPE pointers
    qLo = Q + hh[:, None] * qSh + kk[None, :]
    qRo = Q + hh[:, None] * qSh + rank + tl.arange(0, rope)[None, :]
    kLo = K + mm[None, :] * kSm + kk[:, None]
    kRo = K + mm[None, :] * kSm + rank + tl.arange(0, rope)[:, None]

    maskH = (hOff + hh) < H
    maskM = (mOff + mm) < M

    qk = tl.zeros((BH, BM), dtype=tl.float32)
    # RoPE chunk
    qr = tl.load(qRo, mask=maskH[:, None], other=0.)
    kr = tl.load(kRo, mask=maskM[None, :], other=0.)
    qk = tl.dot(qr, kr, acc=qk)

    steps = (rank + BK - 1) // BK
    for _ in range(steps):
        ql = tl.load(qLo, mask=maskH[:, None], other=0.)
        kl = tl.load(kLo, mask=maskM[None, :], other=0.)
        qk = tl.dot(ql, kl, acc=qk)
        qLo += BK
        kLo += BK

    qk *= scale
    ptr = QK + hh[:, None] * M + mm[None, :]
    tl.store(ptr, qk, mask=maskH[:, None] & maskM[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BM': 32, 'BH': 16, 'BK': 64},  num_stages=2, num_warps=4),
        triton.Config({'BM': 64, 'BH': 16, 'BK': 64},  num_stages=2, num_warps=4),
        triton.Config({'BM': 32, 'BH': 32, 'BK': 128}, num_stages=2, num_warps=4),
    ],
    key=['M', 'rank', 'rope']
)
@triton.jit
def _compute_attn(
    QK, V, O, pads,
    vSb, vSm, vSd,
    oSb, oSh, oSn, oSd,
    M, H, rank: tl.constexpr, rope: tl.constexpr,
    BH: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr
):
    b = tl.program_id(0)
    hOff = tl.program_id(1) * BH
    kOff = tl.program_id(2) * BK

    pads += b
    p = tl.load(pads, mask=True)

    QK += b * (M * H) + hOff * M + p
    V += b * vSb + p * vSm + kOff
    O += b * oSb + hOff * oSh + kOff

    hh = tl.arange(0, BH)
    mm = tl.arange(0, BM)
    kk = tl.arange(0, BK)
    dt = O.type.element_ty

    qkPtr = QK + hh[:, None] * M + mm[None, :]
    vPtr = V + mm[:, None] * vSm + kk[None, :]

    maskH = (hOff + hh) < H

    accum = tl.zeros((BH, BK), dtype=tl.float32)
    mx = tl.full((BH,), float('-inf'), dtype=tl.float32)
    sf = tl.zeros((BH,), dtype=tl.float32)

    for startM in range(p, M, BM):
        maskM = (startM + mm) < M
        sc = tl.load(qkPtr, mask=maskM[None, :] & maskH[:, None], other=float('-inf'))

        m_ij = tl.max(sc, axis=1)
        new_mx = tl.maximum(mx, m_ij)
        alpha = tl.exp(mx - new_mx)
        esc = tl.exp(sc - new_mx[:, None])

        sf = sf * alpha + tl.sum(esc, axis=-1)
        accum = accum * alpha[:, None]

        vv = tl.load(vPtr, mask=maskM[:, None], other=0.)
        accum = tl.dot(esc.to(dt), vv, acc=accum)
        mx = new_mx

        qkPtr += BM
        vPtr += BM * vSm

    accum /= sf[:, None]
    outPtr = O + hh[:, None] * oSh + kk[None, :]
    tl.store(outPtr, accum, mask=maskH[:, None])


def triton_mqa(q, k, v, scale=None, attention_mask=None, two_stage_decode=None):
    """
    Multi-Query Attention (LoRA + RoPE):
      - If N>1 => encode path
      - If N=1 => decode path (single-kernel or two-stage)
    """
    B, H, N, D = q.shape
    M = k.shape[-2]
    if D == 288:  # e.g. 256 + 32
        rank, rope = 256, 32
    elif D == 576:  # e.g. 512 + 64
        rank, rope = 512, 64
    else:
        rank = 2 ** int(math.log2(D - 1))
        rope = D - rank

    assert scale is not None, "Must provide a softmax scale"
    if k.dim() == 4:  # squeeze if [B,1,M,D]
        k = k.squeeze(1)
    if v.dim() == 4:
        v = v.squeeze(1)

    pads = (M - attention_mask.sum(-1)) if attention_mask is not None \
           else torch.zeros((B,), dtype=torch.int32, device=q.device)
    out = torch.empty(B, H, N, rank, dtype=q.dtype, device=q.device)

    if N > 1:
        grid = lambda meta: (B, H, triton.cdiv(N, meta['BN']))
        _mqa_enc_kernel[grid](
            q, k, v, out, pads, scale,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            N, rank, rope
        )
    else:
        use_two_stage = two_stage_decode if two_stage_decode is not None else (B <= 4)
        if use_two_stage:
            qk_buf = torch.empty(B, H, M, dtype=torch.float32, device=q.device)
            grid1 = lambda meta: (
                B,
                triton.cdiv(H, meta['BH']),
                triton.cdiv(M, meta['BM'])
            )
            _compute_qk[grid1](
                q, k, qk_buf, pads, scale,
                *q.stride(), *k.stride(),
                M, H, rank, rope
            )
            grid2 = lambda meta: (
                B,
                triton.cdiv(H, meta['BH']),
                triton.cdiv(rank, meta['BK'])
            )
            _compute_attn[grid2](
                qk_buf, v, out, pads,
                *v.stride(), *out.stride(),
                M, H, rank, rope
            )
        else:
            grid = lambda meta: (B, triton.cdiv(H, meta['BH']))
            _mqa_dec_kernel[grid](
                q, k, v, out, pads, scale,
                *q.stride(), *k.stride(), *v.stride(), *out.stride(),
                M, H, rank, rope
            )

    return out