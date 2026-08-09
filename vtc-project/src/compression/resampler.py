"""
Perceiver-resampler-style compression, the approach used by earlier
MiniCPM-V versions: a fixed, learnable set of query vectors attends over
the full patch token sequence via one cross-attention layer, producing a
fixed-size compressed output regardless of how many tokens came in.

This is the one method in the project with trainable parameters (the
queries + attention weights), so it needs a short training pass on the
eval task itself (or a lightweight reconstruction/distillation objective)
before it's evaluated. Kept intentionally tiny — one cross-attention
layer, no deep stack — since the goal is a fast, fair baseline, not a
production compressor.
"""

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    def __init__(self, embed_dim: int, num_queries: int, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, D) -> returns (B, num_queries, D)"""
        b = tokens.shape[0]
        q = self.norm_q(self.queries).unsqueeze(0).expand(b, -1, -1)
        kv = self.norm_kv(tokens)
        out, _ = self.attn(q, kv, kv)
        return self.norm_out(out + q)


def build_resampler_for_ratio(embed_dim: int, num_patches: int, keep_ratio: float) -> PerceiverResampler:
    num_queries = max(1, round(num_patches * keep_ratio))
    return PerceiverResampler(embed_dim=embed_dim, num_queries=num_queries)
