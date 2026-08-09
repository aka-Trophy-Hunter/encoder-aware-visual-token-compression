"""
Small eval dataset loader.

Default: a subset of CIFAR-100 used as a zero-shot classification proxy
task (fast to download, small enough to run the full sweep in minutes).
This is a placeholder for a stronger choice later on — e.g. a small
COCO or Flickr8k retrieval subset would demonstrate grounded
image-text understanding more convincingly, and matches the kind of
task Dr. Khan's lab actually cares about (see README). Swap it in here
once you're ready to move past the scaffolding stage.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


@dataclass
class EvalBatch:
    images: torch.Tensor
    labels: torch.Tensor
    class_names: list


def get_eval_loader(
    num_samples: int,
    preprocess: Callable,
    batch_size: int = 32,
    root: str = "./data",
) -> DataLoader:
    full = datasets.CIFAR100(root=root, train=False, download=True, transform=preprocess)
    subset_indices = list(range(min(num_samples, len(full))))
    subset = torch.utils.data.Subset(full, subset_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    loader.class_names = full.classes  # type: ignore[attr-defined]
    return loader
