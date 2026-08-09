"""
Evaluates one (encoder, compression method, keep_ratio) combination.

Task: zero-shot-style classification via nearest-centroid matching in the
compressed token space (mean-pooled) — kept deliberately simple so the
*compression method* is the thing being measured, not a downstream
classifier's capacity to compensate for it. Swap in real zero-shot CLIP
text-matching or a retrieval metric once you move past the scaffolding
stage (see README "Status").
"""

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.compression.pruning import prune_tokens
from src.compression.early_merge import early_merge
from src.compression.resampler import build_resampler_for_ratio


def compress(tokens: torch.Tensor, method: str, keep_ratio: float, embed_dim: int):
    if keep_ratio >= 1.0:
        return tokens
    if method == "pruning":
        return prune_tokens(tokens, keep_ratio)
    if method == "early_merge":
        n = tokens.shape[1]
        grid_size = int(math.isqrt(n))
        if grid_size * grid_size != n:
            raise ValueError("early_merge requires a square patch grid")
        return early_merge(tokens, grid_size, keep_ratio)
    if method == "resampler":
        # NOTE: untrained resampler used here as a structural placeholder.
        # For a fair comparison, train the resampler briefly (a few epochs
        # of a reconstruction or contrastive objective) before scoring —
        # see README "Status" for what's stubbed vs. finished.
        resampler = build_resampler_for_ratio(embed_dim, tokens.shape[1], keep_ratio)
        resampler.eval()
        with torch.no_grad():
            return resampler(tokens)
    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def run_eval(encoder, method: str, keep_ratio: float, loader) -> float:
    """Returns nearest-centroid classification accuracy on the loader."""
    class_sums = {}
    class_counts = {}
    all_feats, all_labels = [], []

    for images, labels in tqdm(loader, desc=f"{encoder.name}/{method}/{keep_ratio}"):
        tokens = encoder(images)
        compressed = compress(tokens, method, keep_ratio, encoder.embed_dim)
        pooled = compressed.mean(dim=1)  # (B, D) — simple mean pool to a single vector
        pooled = F.normalize(pooled, dim=-1)

        all_feats.append(pooled)
        all_labels.append(labels)

    feats = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    # build per-class centroids, then classify each sample by nearest centroid
    # (excluding itself would be more rigorous — fine for a v1 sanity check)
    unique = labels.unique()
    centroids = torch.stack([feats[labels == c].mean(dim=0) for c in unique])
    centroids = F.normalize(centroids, dim=-1)

    sims = feats @ centroids.T  # (N, C)
    preds = unique[sims.argmax(dim=-1)]

    return (preds == labels).float().mean().item()
