# ============================================================
# CONSOLIDATED SETUP CELL
# Run this single cell any time the runtime resets.
# Rebuilds: packages, encoders, dataset, splits, compression
# methods, eval functions, and the multi-seed sweep runner.
# ============================================================

get_ipython().system('pip install -q torch torchvision open_clip_torch timm')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import statistics
import open_clip
from torchvision import datasets

# ---------------- Encoders ----------------

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
model = model.visual
model.eval()

dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dino.eval()


def clip_encoder_fn(images):
    x = model.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    cls = model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device)
    x = torch.cat([cls, x], dim=1)
    x = x + model.positional_embedding.to(x.dtype)
    x = model.ln_pre(x)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_post(x)
    return x[:, 1:, :]


def dino_encoder_fn(images):
    with torch.no_grad():
        features = dino.forward_features(images)
        return features["x_norm_patchtokens"]


def get_tokens(images):
    with torch.no_grad():
        x = model.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
    return x


# ---------------- Dataset + splits ----------------

test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=preprocess)

torch.manual_seed(42)
train_indices_v2 = torch.randperm(len(test_set))[:1500]
train_subset_v2 = torch.utils.data.Subset(test_set, train_indices_v2.tolist())
train_loader_v2 = DataLoader(train_subset_v2, batch_size=32, shuffle=True)

remaining = list(set(range(len(test_set))) - set(train_indices_v2.tolist()))
eval_indices = torch.tensor(remaining)[torch.randperm(len(remaining))[:1000]]
eval_subset = torch.utils.data.Subset(test_set, eval_indices.tolist())
eval_loader = DataLoader(eval_subset, batch_size=32, shuffle=False)

print("Dataset ready. Train:", len(train_subset_v2), "Eval:", len(eval_subset))


# ---------------- Compression methods ----------------

def prune_tokens_with_context(tokens, keep_ratio, context_frac=0.25):
    b, n, d = tokens.shape
    k_total = round(n * keep_ratio)
    k_context = max(1, round(k_total * context_frac))
    k_dominant = k_total - k_context

    norms = tokens.norm(dim=-1)
    order = norms.argsort(dim=-1, descending=True)

    dominant_idx = order[:, :k_dominant]
    rest_idx = order[:, k_dominant:]

    dominant = torch.gather(tokens, 1, dominant_idx.unsqueeze(-1).expand(-1, -1, d))
    rest = torch.gather(tokens, 1, rest_idx.unsqueeze(-1).expand(-1, -1, d))

    chunks = torch.chunk(rest, k_context, dim=1)
    context = torch.stack([c.mean(dim=1) for c in chunks], dim=1)

    return torch.cat([dominant, context], dim=1)


def early_merge(tokens, grid_size, stride):
    b, n, d = tokens.shape
    assert n == grid_size * grid_size, "tokens don't form a clean square grid"
    grid = tokens.reshape(b, grid_size, grid_size, d)
    blocks = grid.unfold(1, stride, stride).unfold(2, stride, stride)
    merged = blocks.mean(dim=(-1, -2))
    new_grid_size = grid_size // stride
    return merged.reshape(b, new_grid_size * new_grid_size, d)


class PerceiverResampler(nn.Module):
    def __init__(self, embed_dim, num_queries, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        b = tokens.shape[0]
        q = self.norm_q(self.queries).unsqueeze(0).expand(b, -1, -1)
        kv = self.norm_kv(tokens)
        out, _ = self.attn(q, kv, kv)
        return self.norm_out(out + q)


class PerceiverDecoder(nn.Module):
    def __init__(self, embed_dim, num_output_tokens, num_heads=8):
        super().__init__()
        self.output_queries = nn.Parameter(torch.randn(num_output_tokens, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, compressed_tokens):
        b = compressed_tokens.shape[0]
        q = self.output_queries.unsqueeze(0).expand(b, -1, -1)
        out, _ = self.attn(q, compressed_tokens, compressed_tokens)
        return out


# NOTE: resampler/decoder need re-training after a reset - they are NOT
# reconstructed with trained weights here. Re-run your training loop after
# this cell if you need the trained resampler (see "Resampler" section of
# your notebook history). A fresh, untrained instance is created below so
# later cells referencing `resampler` don't immediately error - retrain
# before trusting any resampler results.
resampler = PerceiverResampler(embed_dim=768, num_queries=12)
decoder = PerceiverDecoder(embed_dim=768, num_output_tokens=49)


# ---------------- Eval pipeline ----------------

def extract_features(loader, encoder_fn, method, keep_ratio, grid_size=None, stride=None, projection=None):
    all_feats, all_labels = [], []
    for images, labels in loader:
        with torch.no_grad():
            tokens = encoder_fn(images)
            if method == "pruning":
                compressed = prune_tokens_with_context(tokens, keep_ratio)
            elif method == "resampler":
                compressed = resampler(tokens)
            elif method == "early_merge":
                compressed = early_merge(tokens, grid_size=grid_size, stride=stride)
            else:
                raise ValueError(f"unknown method: {method}")

            pooled = compressed.mean(dim=1)
            if projection is not None:
                pooled = pooled @ projection
            pooled = F.normalize(pooled, dim=-1)

        all_feats.append(pooled)
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def nearest_centroid_accuracy(train_feats, train_labels, eval_feats, eval_labels):
    unique_classes = train_labels.unique()
    centroids = torch.stack([
        train_feats[train_labels == c].mean(dim=0) for c in unique_classes
    ])
    centroids = F.normalize(centroids, dim=-1)
    similarities = eval_feats @ centroids.T
    predicted_idx = similarities.argmax(dim=-1)
    predicted_labels = unique_classes[predicted_idx]
    correct = (predicted_labels == eval_labels).float()
    return correct.mean().item()


def run_sweep_config(encoder_fn, method, keep_ratio, grid_size=None, stride=None, projection=None, seeds=[1, 2, 3]):
    accs = []
    for seed in seeds:
        torch.manual_seed(seed)
        remaining_pool = list(set(range(len(test_set))) - set(train_indices_v2.tolist()))
        eval_idx = torch.tensor(remaining_pool)[torch.randperm(len(remaining_pool))[:1000]]
        eval_subset_s = torch.utils.data.Subset(test_set, eval_idx.tolist())
        eval_loader_s = DataLoader(eval_subset_s, batch_size=32, shuffle=False)

        train_feats, train_labels = extract_features(
            train_loader_v2, encoder_fn, method, keep_ratio,
            grid_size=grid_size, stride=stride, projection=projection
        )
        eval_feats, eval_labels = extract_features(
            eval_loader_s, encoder_fn, method, keep_ratio,
            grid_size=grid_size, stride=stride, projection=projection
        )

        acc = nearest_centroid_accuracy(train_feats, train_labels, eval_feats, eval_labels)
        accs.append(acc)

    return statistics.mean(accs), statistics.stdev(accs)


print("Setup complete. All functions and classes ready.")
print("REMINDER: resampler is UNTRAINED after this reset - retrain before using its results.")
