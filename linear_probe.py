#!/usr/bin/env python
"""Linear probe on frozen KERAFound features, as in the KERAFound benchmark.

GAP features of every image are extracted once (frozen encoder), standardised on the training set, and an L2-regularised
logistic regression (softmax for more than two classes) is fitted full-batch with LBFGS. Each training unit gets equal
total weight regardless of its number of images. The L2 strength is chosen on validation unit-level AUROC.

  python linear_probe.py --model KERAFound-DINOv3-OCT --data data/keratoconus --out runs/kc_lp
  python linear_probe.py --model KERAFound-MAE-OCT --csv data/keratoconus.csv --out runs/kc_lp_mae
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from kerafound.data import load_table, units, ImageSet, to_input
from kerafound.metrics import unit_probs, score, report
from kerafound.models import build_encoder

L2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]


@torch.no_grad()
def extract(encoder, u, dev, batch, workers):
    """GAP features of every image of the units in u -> (features (n_images, 1024), unit index per image)."""
    F, idx = [], []
    for x, k in DataLoader(ImageSet(u), batch_size=batch, num_workers=workers, pin_memory=dev == "cuda"):
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            F.append(encoder(to_input(x.to(dev)))[0].float().cpu()); idx.append(k)
    return torch.cat(F), torch.cat(idx).numpy()


def fit(X, y, w, l2, K, iters=150):
    W = torch.zeros(X.shape[1], K, device=X.device, requires_grad=True)
    b = torch.zeros(K, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.cross_entropy(X @ W + b, y, reduction="none") * w).sum() / w.sum() + l2 * (W ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    return W.detach(), b.detach()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--weights", help="local model.safetensors (default: HuggingFace)")
    ap.add_argument("--data", help="folder layout root"); ap.add_argument("--csv", help="CSV layout file")
    ap.add_argument("--out", required=True); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df, classes = load_table(args.data, args.csv); K = len(classes)
    U = {s: units(df[df.split == s]) for s in ("train", "val", "test") if (df.split == s).any()}
    enc = build_encoder(args.model, args.weights).to(dev).eval()
    feats = {s: extract(enc, u, dev, args.batch, args.workers) for s, u in U.items()}
    Xtr, itr = feats["train"]; mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda X: ((X - mu) / sd).to(dev)
    y = torch.tensor(U["train"].y.to_numpy()[itr], device=dev)
    w = torch.tensor(1.0 / np.bincount(itr)[itr], dtype=torch.float32, device=dev)
    probs = lambda s, W, b: unit_probs(torch.softmax((z(feats[s][0]) @ W + b).double(), 1).cpu().numpy(), feats[s][1], len(U[s]))
    best = None
    for l2 in L2:
        W, b = fit(z(Xtr), y, w, l2, K); a = score(U["val"].y, probs("val", W, b))
        print(f"L2 {l2:g}: val AUROC {a:.4f}", flush=True)
        if best is None or a > best[0]: best = (a, l2, W, b)
    _, l2, W, b = best
    res = dict(model=args.model, classes=classes, l2=l2)
    for s in U:
        P = probs(s, W, b); res[s] = report(U[s].y, P)
        pd.DataFrame({"unit_id": U[s].unit_id, "patient_id": U[s].patient_id, "y": U[s].y,
                      **{f"p_{c}": P[:, k] for k, c in enumerate(classes)}}).to_csv(out / f"predictions_{s}.csv", index=False)
    torch.save(dict(W=W.cpu(), b=b.cpu(), mu=mu, sd=sd, classes=classes, model=args.model), out / "linear_head.pt")
    json.dump(res, open(out / "metrics.json", "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
