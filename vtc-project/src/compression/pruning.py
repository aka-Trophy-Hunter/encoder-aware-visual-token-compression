"""
Training-free token compression, following the VisionZip paradigm:

1. Score every patch token by its L2 norm (a cheap proxy for "how much
   this token's activation stands out" — high-norm tokens tend to carry
   salient/foreground information).
2. Keep the top-k highest-norm tokens as "dominant" tokens, unchanged.
3. Merge the remaining low-norm tokens into a small number of "contextual"
   tokens via average pooling, so background/context isn't discarded
   entirely, just compressed.

No parameters, no training — this is the cheapest baseline and the one
most likely to behave differently across encoder types, since it directly
depends on what "high norm" means for that encoder's learned features.
"""

import torch


def prune_tokens(tokens: torch.Tensor, keep_ratio: float, context_frac: float = 0.25) -> torch.Tensor:
    """
    tokens: (B, N, D) patch tokens
    keep_ratio: fraction of the original N tokens to keep in total
    context_frac: fraction of the *kept* budget spent on merged context
                  tokens rather than dominant tokens

    Returns: (B, K, D) compressed tokens, K = round(N * keep_ratio)
    """
    b, n, d = tokens.shape
    k_total = max(1, round(n * keep_ratio))
    k_context = max(1, round(k_total * context_frac))
    k_dominant = max(1, k_total - k_context)

    norms = tokens.norm(dim=-1)  # (B, N)
    order = norms.argsort(dim=-1, descending=True)  # (B, N)

    dominant_idx = order[:, :k_dominant]  # (B, k_dominant)
    rest_idx = order[:, k_dominant:]  # (B, N - k_dominant)

    dominant = torch.gather(
        tokens, 1, dominant_idx.unsqueeze(-1).expand(-1, -1, d)
    )  # (B, k_dominant, D)

    # merge the "rest" into k_context tokens by chunked average pooling
    rest = torch.gather(
        tokens, 1, rest_idx.unsqueeze(-1).expand(-1, -1, d)
    )  # (B, N - k_dominant, D)

    chunks = torch.chunk(rest, k_context, dim=1)
    context = torch.stack([c.mean(dim=1) for c in chunks], dim=1)  # (B, k_context, D)

    return torch.cat([dominant, context], dim=1)  # (B, k_total, D)
