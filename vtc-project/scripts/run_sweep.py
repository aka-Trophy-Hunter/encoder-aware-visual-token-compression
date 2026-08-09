"""
Runs the full (encoder x method x keep_ratio) grid and saves:
  - results/sweep_results.csv
  - results/tokens_vs_accuracy.png

Usage:
    python scripts/run_sweep.py \
        --encoders clip dinov2 \
        --methods pruning resampler early_merge \
        --ratios 1.0 0.75 0.5 0.25 \
        --num-samples 1000 \
        --output-dir results/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt

from src.encoders import build_encoder
from src.data import get_eval_loader
from src.eval import run_eval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoders", nargs="+", default=["clip", "dinov2"])
    p.add_argument("--methods", nargs="+", default=["pruning", "resampler", "early_merge"])
    p.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.75, 0.5, 0.25])
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if os_has_cuda() else "cpu")
    p.add_argument("--output-dir", default="results/")
    return p.parse_args()


def os_has_cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for enc_name in args.encoders:
        encoder = build_encoder(enc_name, device=args.device)
        loader = get_eval_loader(args.num_samples, encoder.preprocess if hasattr(encoder, "preprocess") else None,
                                  batch_size=args.batch_size)
        for method in args.methods:
            for ratio in args.ratios:
                acc = run_eval(encoder, method, ratio, loader)
                rows.append({
                    "encoder": enc_name,
                    "method": method,
                    "keep_ratio": ratio,
                    "accuracy": acc,
                })
                print(f"[{enc_name}/{method}/{ratio}] accuracy={acc:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    plot_path = os.path.join(args.output_dir, "tokens_vs_accuracy.png")
    plot_results(df, plot_path)
    print(f"Saved {plot_path}")


def plot_results(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    for (enc, method), group in df.groupby(["encoder", "method"]):
        group = group.sort_values("keep_ratio")
        ax.plot(group["keep_ratio"], group["accuracy"], marker="o", label=f"{enc} / {method}")

    ax.set_xlabel("Token keep ratio")
    ax.set_ylabel("Accuracy (nearest-centroid)")
    ax.set_title("Compression tolerance by encoder type and method")
    ax.invert_xaxis()  # so it reads left (aggressive compression) to right (none)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    main()
