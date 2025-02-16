from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

@dataclass
class AttentionShape:
    batch_size: int
    num_heads: int
    seq_length: int
    head_dim: int

    @classmethod
    def from_tensor(cls, t: torch.Tensor, name: str) -> 'AttentionShape':
        # Expect shape (B, H, N, D)
        if t.dim() != 4:
            raise ValueError(
                f"Expected {name} to have 4 dims (B, H, N, D); got {t.shape}"
            )
        return cls(*t.shape)

def validate_attention_shapes(qs: AttentionShape, ks: AttentionShape, vs: AttentionShape):
    if qs.batch_size != ks.batch_size or ks.batch_size != vs.batch_size:
        raise ValueError("Batch sizes of q, k, v must match.")
    if ks.seq_length != vs.seq_length:
        raise ValueError("Key and value sequence lengths must match.")
    # Either all heads match, or k,v each have 1 head while q has many
    if not (
        (ks.num_heads == vs.num_heads == 1 and qs.num_heads > 1)
        or (qs.num_heads == ks.num_heads == vs.num_heads)
    ):
        raise ValueError(
            "Heads must match, or k and v must each have exactly 1 head while q has more."
        )

def compute_scores(query, key, scale, mask=None):
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if mask is not None:
        scores.masked_fill_(~mask, float('-inf'))
    return F.softmax(scores, dim=-1, dtype=torch.float32)

def torch_mqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
    decode_one: bool = False
) -> torch.Tensor:
    
    q_shape = AttentionShape.from_tensor(q, "q")
    k_shape = AttentionShape.from_tensor(k, "k")
    v_shape = AttentionShape.from_tensor(v, "v")
    validate_attention_shapes(q_shape, k_shape, v_shape)

    if k_shape.num_heads == v_shape.num_heads == 1 and q_shape.num_heads > 1:
        k = k.expand(-1, q_shape.num_heads, -1, -1)
        v = v.expand(-1, q_shape.num_heads, -1, -1)

    if scale is None:
        scale = 1.0

    if attention_mask is not None:
        attention_mask = attention_mask.to(dtype=torch.bool)

    if q_shape.seq_length > 1:
        causal_mask = torch.tril(
            torch.ones(q_shape.seq_length, k_shape.seq_length,
                       device=q.device, dtype=torch.bool)
        )[None, None, :, :]

        if attention_mask is not None:
            mask = causal_mask & attention_mask
        else:
            mask = causal_mask

        attn_weights = compute_scores(q, k, scale, mask).to(q.dtype)
        return torch.matmul(attn_weights, v)

    if decode_one:
        outs = []
        for b_idx in range(q_shape.batch_size):
            mask_b = attention_mask[b_idx] if attention_mask is not None else None
            attn_weights_b = compute_scores(q[b_idx], k[b_idx], scale, mask_b).to(q.dtype)
            outs.append(torch.matmul(attn_weights_b, v[b_idx]))
        return torch.stack(outs, dim=0)
    else:
        attn_weights = compute_scores(q, k, scale, attention_mask).to(q.dtype)
        return torch.matmul(attn_weights, v)
