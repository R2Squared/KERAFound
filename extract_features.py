#!/usr/bin/env python
"""Extract frozen KERAFound GAP (and CLS) features for a folder of images.

  python extract_features.py --model KERAFound-DINOv3-OCT --images my_bscans/ --out features.npz
Writes paths (n,), gap (n, 1024) and cls (n, 1024).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from kerafound.data import load_image, to_input, EXTS
from kerafound.models import build_encoder


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--weights", help="local model.safetensors (default: HuggingFace)")
    ap.add_argument("--images", required=True); ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = build_encoder(args.model, args.weights).to(dev).eval()
    paths = sorted(str(p) for p in Path(args.images).rglob("*") if p.suffix.lower() in EXTS)
    gap, cls = [], []
    for i in range(0, len(paths), args.batch):
        x = torch.stack([load_image(p) for p in paths[i:i + args.batch]]).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            g, c = enc(to_input(x))
        gap.append(g.float().cpu().numpy()); cls.append(c.float().cpu().numpy())
    np.savez(args.out, paths=np.array(paths), gap=np.concatenate(gap), cls=np.concatenate(cls))
    print(f"{len(paths)} images -> {args.out}")


if __name__ == "__main__":
    main()
