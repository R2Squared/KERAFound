#!/usr/bin/env python
"""Export encoder-only KERAFound weights (fp32 safetensors) from the training checkpoints.
DINOv3: teacher backbone (DINO/iBOT heads dropped). MAE: encoder (decoder, mask token and optimiser state dropped).
Run inside the benchmark code directory (needs kf.config); writes <out>/<name>/model.safetensors + config.json."""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
sys.path.insert(0, ".")
from kf.config import ENCODERS

OUT = Path(sys.argv[1])
NAMES = {"kf_dino_oct": "KERAFound-DINOv3-OCT", "kf_dino_placido": "KERAFound-DINOv3-Placido", "kf_dino_photo": "KERAFound-DINOv3-Photo",
         "kf_mae_oct": "KERAFound-MAE-OCT", "kf_mae_placido": "KERAFound-MAE-Placido"}
for key, name in NAMES.items():
    cfg = ENCODERS[key]; sd = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
    if cfg["kind"] == "dinov3":
        sd = {k[len("backbone."):]: v for k, v in sd["teacher"].items() if k.startswith("backbone.")}
    else:
        sd = {k: v for k, v in sd.get("model", sd).items() if not k.startswith("decoder") and k != "mask_token"}
    sd = {k: v.float().contiguous() for k, v in sd.items()}
    d = OUT / name; d.mkdir(parents=True, exist_ok=True); save_file(sd, d / "model.safetensors")
    json.dump(dict(name=name, kind=cfg["kind"], arch="vit_large_patch16_224", embed_dim=1024, img_size=224,
                   mask_k_bias=cfg.get("mask_k_bias"), norm=cfg.get("norm"), n_tensors=len(sd),
                   n_params=sum(v.numel() for v in sd.values())), open(d / "config.json", "w"), indent=1)
    print(name, len(sd), "tensors", sum(v.numel() for v in sd.values()) / 1e6, "M params", flush=True)
