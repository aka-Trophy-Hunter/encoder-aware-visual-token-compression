"""
Wraps CLIP ViT-B/32 (contrastive) and DINOv2 ViT-S/14 (self-supervised)
behind one shared interface: given a batch of images, return the raw
patch token sequence *before* any pooling, so compression methods can
operate on it directly.

Both encoders are kept frozen throughout the project — we're studying
how their token structure responds to compression, not fine-tuning them.
"""

import torch
import torch.nn as nn


class EncoderWrapper(nn.Module):
    """Common interface: forward(images) -> (B, N, D) patch tokens."""

    name: str
    embed_dim: int
    num_patches: int

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CLIPEncoder(EncoderWrapper):
    """CLIP ViT-B/32 vision tower, contrastive objective (image-text alignment)."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = model.visual.to(device)
        self.model.eval()
        self.preprocess = preprocess
        self.device = device
        self.name = "clip-vit-b32"
        self.embed_dim = self.model.transformer.width
        self.num_patches = (self.model.grid_size[0] * self.model.grid_size[1])

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # open_clip's visual tower normally pools internally; we hook the
        # patch embeddings + transformer manually to keep the full token seq.
        x = self.model.conv1(images.to(self.device))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, N, D)
        cls = self.model.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)
        return x[:, 1:, :]  # drop CLS, return patch tokens only


class DINOv2Encoder(EncoderWrapper):
    """DINOv2 ViT-S/14, self-supervised objective (patch-level self-distillation)."""

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
        self.model.eval()
        self.device = device
        self.name = "dinov2-vit-s14"
        self.embed_dim = self.model.embed_dim
        self.num_patches = None  # depends on input resolution

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(images.to(self.device))
        return feats["x_norm_patchtokens"]  # (B, N, D), CLS already excluded


def build_encoder(name: str, device: str = "cpu") -> EncoderWrapper:
    if name == "clip":
        return CLIPEncoder(device=device)
    if name == "dinov2":
        return DINOv2Encoder(device=device)
    raise ValueError(f"Unknown encoder: {name}")
