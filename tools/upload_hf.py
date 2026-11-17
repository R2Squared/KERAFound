#!/usr/bin/env python
"""Create the gated HuggingFace model repositories and upload the exported KERAFound weights (maintainers only).

  huggingface-cli login                       # token with write access to the R2Squared namespace
  python tools/upload_hf.py release/ [--org R2Squared]
Each <release>/<model>/ folder (from export_weights.py) holds model.safetensors and config.json.
"""
import argparse, json
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent.parent
CARD = """---
license: {license}
{license_extra}library_name: pytorch
pipeline_tag: image-feature-extraction
tags: [ophthalmology, cornea, anterior-segment, foundation-model, vit-large, {kind}]
extra_gated_prompt: "KERAFound is released for research use only. It is not a medical device and must not be used for clinical decision-making. By requesting access you agree to the licence below, not to attempt to re-identify any individual, and to cite KERAFound in any publication."
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I agree to use the model for research purposes only: checkbox
---

# {name}

ViT-L/16 encoder (about 303M parameters, 224x224 input, 1,024-d features) for **{modality}**, from the KERAFound
family of corneal and anterior segment foundation models. {origin}

- `model.safetensors`: encoder weights only, in fp32.
- `config.json`: architecture details.

Usage, linear probing and fine-tuning scripts: https://github.com/{org}/KERAFound

```python
from kerafound.models import build_encoder
encoder = build_encoder("{name}")      # downloads this file; returns (GAP, CLS) features
```

Preprocessing: greyscale, bicubic resize to 224x224 without cropping, replicated to 3 channels, ImageNet mean/std.

## Licence
{licence_text}

Research use only. The model is not a medical device.
"""
ORIGIN = {"dinov3": "Continued DINOv3 self-supervised pretraining of Meta's DINOv3 ViT-L/16 (LVD-1689M). Built with DINOv3.",
          "mae": "Continued masked-autoencoder pretraining of the ImageNet MAE ViT-L/16."}
MODALITY = {"OCT": "anterior segment OCT B-scans", "Placido": "Placido-disc corneal topography",
            "Photo": "anterior segment / corneal photographs"}

ap = argparse.ArgumentParser(); ap.add_argument("release"); ap.add_argument("--org", default="R2Squared")
args = ap.parse_args(); api = HfApi()
for d in sorted(Path(args.release).iterdir()):
    cfg = json.load(open(d / "config.json")); name, kind = cfg["name"], cfg["kind"]
    repo = f"{args.org}/{name}"
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(repo, gated="manual")
    if kind == "dinov3":
        lic = dict(license="other", license_extra="license_name: dinov3-license\nlicense_link: LICENSE.md\n",
                   licence_text="Derivative of DINOv3, distributed under the DINOv3 License (LICENSE.md in this repository).")
        api.upload_file(path_or_fileobj=str(HERE / "LICENSE_DINOv3.md"), path_in_repo="LICENSE.md", repo_id=repo)
    else:
        lic = dict(license="cc-by-nc-4.0", license_extra="", licence_text="CC BY-NC 4.0.")
    card = CARD.format(name=name, kind=kind, org=args.org, modality=MODALITY[name.split("-")[-1]], origin=ORIGIN[kind], **lic)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo)
    api.upload_folder(folder_path=str(d), repo_id=repo, allow_patterns=["model.safetensors", "config.json"])
    print("uploaded", repo, flush=True)
