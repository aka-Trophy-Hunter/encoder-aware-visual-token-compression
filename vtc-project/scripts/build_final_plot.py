"""
Builds the final tokens-vs-accuracy comparison plot from all three
compression methods, across both encoders (CLIP, DINOv2).

Data below is transcribed directly from the Colab sweep results
(see results/*.csv for the per-method files saved during the project).
"""

import matplotlib.pyplot as plt

# ---- Pruning (3-seed mean, with error bars) ----
pruning_clip = {
    1.0: (0.0823, 0.0121),
    0.5: (0.0753, 0.0101),
    0.25: (0.0600, 0.0069),
    0.1: (0.0567, 0.0050),
}
pruning_dino = {
    1.0: (0.6997, 0.0067),
    0.5: (0.6860, 0.0061),
    0.25: (0.6557, 0.0059),
    0.1: (0.6163, 0.0163),
}

# ---- Early-merge (max-pool eval, single run - coarse ratios only) ----
early_merge_clip = {
    1.0: 0.0760,
    1/49: 0.0660,
}
early_merge_dino = {
    1.0: 0.6990,
    0.25: 0.7150,
    1/16: 0.7120,
    1/64: 0.7030,
}

# ---- Resampler (single trained instance per encoder, single run) ----
resampler_clip = {12/49: 0.0440}
resampler_dino = {64/256: 0.7210}


def plot_encoder(ax, title, pruning, early_merge, resampler, ylabel):
    # pruning: mean +/- std
    xs = sorted(pruning.keys())
    means = [pruning[x][0] for x in xs]
    stds = [pruning[x][1] for x in xs]
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label="Pruning (VisionZip-style)", color="#d62728")

    # early_merge: no error bars (single run)
    xs2 = sorted(early_merge.keys())
    ys2 = [early_merge[x] for x in xs2]
    ax.plot(xs2, ys2, marker="s", linestyle="--", label="Early-merge (intra-encoder)", color="#1f77b4")

    # resampler: single point per encoder
    xs3 = list(resampler.keys())
    ys3 = list(resampler.values())
    ax.scatter(xs3, ys3, marker="^", s=90, label="Resampler (learned)", color="#2ca02c", zorder=5)

    ax.set_xlabel("Token keep ratio")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_encoder(
    axes[0], "CLIP ViT-B/32 (contrastive)",
    pruning_clip, early_merge_clip, resampler_clip,
    ylabel="Nearest-centroid accuracy"
)
plot_encoder(
    axes[1], "DINOv2 ViT-S/14 (self-supervised)",
    pruning_dino, early_merge_dino, resampler_dino,
    ylabel="Nearest-centroid accuracy"
)

fig.suptitle(
    "FOMO: Does compression tolerance depend on encoder type?\n"
    "(CIFAR-100 subset, nearest-centroid classification, 3-seed mean where noted)",
    fontsize=11
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("/home/claude/vtc-project/results/fomo_comparison.png", dpi=160)
print("Saved plot.")
