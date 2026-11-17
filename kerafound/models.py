"""KERAFound encoders (ViT-L/16) and the classifier used in the KERAFound benchmark.

Every encoder returns (gap, cls): the mean of the final-layer patch tokens after the final norm (GAP, used for linear
probing and fine-tuning in the paper) and the normed CLS token.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

HF_ORG = "R2Squared"
MODELS = {
    "KERAFound-DINOv3-OCT":     dict(kind="dinov3", modality="AS-OCT B-scans"),
    "KERAFound-DINOv3-Placido": dict(kind="dinov3", modality="Placido-disc topography"),
    "KERAFound-DINOv3-Photo":   dict(kind="dinov3", modality="Anterior segment colour photographs"),
    "KERAFound-MAE-OCT":        dict(kind="mae", modality="AS-OCT B-scans"),
    "KERAFound-MAE-Placido":    dict(kind="mae", modality="Placido-disc topography"),
}


def weights_path(model, weights=None):
    """Local model.safetensors: `weights` if given (file or folder), else downloaded from the gated HuggingFace repo."""
    if weights:
        p = Path(weights)
        return p / "model.safetensors" if p.is_dir() else p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(f"{HF_ORG}/{model}", "model.safetensors"))


class DinoEncoder(nn.Module):
    def __init__(self, drop_path=0.0):
        super().__init__()
        from dinov3.models.vision_transformer import DinoVisionTransformer
        # ViT-L/16 as dinov3_vitl16 (4 storage tokens, RoPE), without the masked K bias and RoPE coordinate augmentation
        self.vit = DinoVisionTransformer(
            img_size=224, patch_size=16, in_chans=3, pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=None, pos_embed_rope_dtype="fp32", embed_dim=1024, depth=24, num_heads=16,
            ffn_ratio=4, qkv_bias=True, drop_path_rate=drop_path, layerscale_init=1e-5, norm_layer="layernorm",
            ffn_layer="mlp", ffn_bias=True, proj_bias=True, n_storage_tokens=4, mask_k_bias=False)

    def forward(self, x):
        out = self.vit.forward_features(x)
        return out["x_norm_patchtokens"].mean(1), out["x_norm_clstoken"]


class MaeEncoder(nn.Module):
    def __init__(self, drop_path=0.0):
        super().__init__()
        import timm
        self.vit = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, drop_path_rate=drop_path)

    def forward(self, x):
        t = self.vit.forward_features(x)      # final norm applied
        return t[:, 1:].mean(1), t[:, 0]


def build_encoder(model, weights=None, drop_path=0.0):
    """Build a KERAFound encoder and load its pretrained weights (strict)."""
    from safetensors.torch import load_file
    if model not in MODELS:
        raise ValueError(f"unknown model {model}; choose from {list(MODELS)}")
    enc = (DinoEncoder if MODELS[model]["kind"] == "dinov3" else MaeEncoder)(drop_path)
    enc.vit.load_state_dict(load_file(weights_path(model, weights)), strict=True)
    return enc


class Classifier(nn.Module):
    """GAP -> LayerNorm -> Linear, as in the benchmark."""
    def __init__(self, encoder, n_classes=2):
        super().__init__()
        self.encoder = encoder
        self.head_norm = nn.LayerNorm(1024)
        self.head = nn.Linear(1024, n_classes)
        nn.init.trunc_normal_(self.head.weight, std=2e-5); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.head_norm(self.encoder(x)[0]))


def param_groups(model, lr, weight_decay, layer_decay, depth=24):
    """Layer-wise lr decay (as in BEiT/MAE/RETFound): embeddings = layer 0, block i = i+1, norm/head = depth+1."""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        n0 = n.replace("encoder.vit.", "")
        if n0.startswith("blocks."): lid = int(n0.split(".")[1]) + 1
        elif n0.startswith(("cls_token", "pos_embed", "patch_embed", "storage_tokens", "mask_token", "rope_embed", "reg_token")): lid = 0
        else: lid = depth + 1
        no_wd = p.ndim == 1 or n0.endswith(("cls_token", "pos_embed", "storage_tokens", "mask_token"))
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = dict(params=[], weight_decay=0.0 if no_wd else weight_decay, lr_scale=layer_decay ** (depth + 1 - lid), lr=lr)
        groups[key]["params"].append(p)
    return list(groups.values())
