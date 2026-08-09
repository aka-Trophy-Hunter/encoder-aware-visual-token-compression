"""
Early / intra-encoder token merging, in the spirit of MiniCPM-V 4.6's move
away from post-hoc perceiver resampling toward compressing tokens *inside*
the ViT (an "intra-ViT early compression" technique from LLaVA-UHD v4).

We approximate this without modifying the encoder's internals (which would
require retraining the backbone) by merging spatially adjacent tokens on
the patch grid *before* they're treated as a flat sequence — a 2x2
neighborhood average pool over the patch grid, applied once, right after
the tokens leave the encoder. This keeps the "compress early / compress by
spatial locality" property that distinguishes this paradigm from the
norm-based selection in pruning.py, without needing to retrain the ViT
itself.

Note: this is a simplification for a small-scale student project, not a
reproduction of MiniCPM-V 4.6's actual internals — call this out explicitly
in the writeup.
"""

import math
import torch


def early_merge(tokens: torch.Tensor, grid_size: int, keep_ratio: float) -> torch.Tensor:
    """
    tokens: (B, N, D) patch tokens, N == grid_size * grid_size
    keep_ratio: approximate fraction of tokens to keep; internally rounded
                to the nearest valid pooling stride (e.g. 2x2 -> 25%, so
                exact ratios like 0.5 fall back to the nearest achievable one)
    """
    b, n, d = tokens.shape
    assert n == grid_size * grid_size, "tokens must form a square grid"

    # choose a pooling stride that gets closest to the requested keep_ratio
    target_tokens = max(1, round(n * keep_ratio))
    best_stride, best_diff = 1, float("inf")
    for stride in range(1, grid_size + 1):
        if grid_size % stride != 0:
            continue
        new_grid = grid_size // stride
        diff = abs(new_grid * new_grid - target_tokens)
        if diff < best_diff:
            best_stride, best_diff = stride, diff

    if best_stride == 1:
        return tokens  # no valid compression possible below this ratio

    grid = tokens.view(b, grid_size, grid_size, d)
    pooled = grid.unfold(1, best_stride, best_stride).unfold(2, best_stride, best_stride)
    # pooled: (B, new_h, new_w, D, stride, stride)
    pooled = pooled.mean(dim=(-1, -2))  # (B, new_h, new_w, D)
    new_grid = grid_size // best_stride
    return pooled.reshape(b, new_grid * new_grid, d)
