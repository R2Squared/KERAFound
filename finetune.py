#!/usr/bin/env python
"""Fine-tune a KERAFound encoder end-to-end, with the recipe used in the KERAFound benchmark (RETFound-style).

AdamW, peak lr 5e-4 with layer-wise decay 0.65, weight decay 0.05, drop path 0.2, label smoothing 0.1, batch 64, bf16,
10% warm-up then cosine. Each epoch samples one random image per training unit. Training lasts 30 epochs (at least
600 steps); validation (<= 4 evenly spaced images per unit) runs 30 times, and the checkpoint with the best validation
unit-level AUROC is evaluated on every image of the test units. Early stop after 10 evaluations without improvement.

  python finetune.py --model KERAFound-DINOv3-OCT --data data/keratoconus --out runs/kc_ft --seed 0
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from kerafound.data import load_table, units, TrainSet, ImageSet, to_input
from kerafound.metrics import unit_probs, score, report
from kerafound.models import build_encoder, Classifier, param_groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--weights", help="local model.safetensors (default: HuggingFace)")
    ap.add_argument("--data", help="folder layout root"); ap.add_argument("--csv", help="CSV layout file")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--min_steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--layer_decay", type=float, default=0.65); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.2); ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--n_evals", type=int, default=30); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--val_images", type=int, default=4); ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df, classes = load_table(args.data, args.csv); K = len(classes)
    U = {s: units(df[df.split == s]) for s in ("train", "val", "test") if (df.split == s).any()}
    model = Classifier(build_encoder(args.model, args.weights, drop_path=args.drop_path), n_classes=K).to(dev)
    bs = min(args.batch, len(U["train"]))
    g = torch.Generator(); g.manual_seed(args.seed)
    dl = DataLoader(TrainSet(U["train"]), batch_size=bs, shuffle=True, drop_last=True, num_workers=args.workers,
                    persistent_workers=args.workers > 0, pin_memory=dev == "cuda", generator=g,
                    worker_init_fn=lambda w: np.random.seed((torch.initial_seed() + w) % 2**32))
    total = max(args.epochs * len(dl), args.min_steps); every = max(1, total // args.n_evals); warm = max(1, int(0.1 * total))
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.wd, args.layer_decay), lr=args.lr)
    lr_at = lambda s: args.lr * (s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    amp = dict(device_type=dev, dtype=torch.bfloat16, enabled=dev == "cuda")

    @torch.no_grad()
    def predict(u, max_images=None):
        model.eval(); P, idx = [], []
        for x, k in DataLoader(ImageSet(u, max_images), batch_size=256, num_workers=args.workers, pin_memory=dev == "cuda"):
            with torch.autocast(**amp):
                P.append(model(to_input(x.to(dev))).float().softmax(1).cpu().numpy()); idx.append(k.numpy())
        model.train(); return unit_probs(np.concatenate(P), np.concatenate(idx), len(u))

    step, best, best_state, since, history, done = 0, -1.0, None, 0, [], False
    model.train()
    while not done:
        for x, y in dl:
            for gr in opt.param_groups: gr["lr"] = lr_at(step) * gr["lr_scale"]
            with torch.autocast(**amp):
                loss = crit(model(to_input(x.to(dev))).float(), y.to(dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            if step % every == 0 or step == total:
                a = score(U["val"].y, predict(U["val"], args.val_images)); history.append(dict(step=step, loss=float(loss), val_auroc=a))
                print(f"step {step}/{total} loss {float(loss):.4f} val AUROC {a:.4f}", flush=True)
                if a > best: best, since, best_state = a, 0, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                else: since += 1
                if since >= args.patience: done = True
            if step >= total or done: done = True; break

    model.load_state_dict(best_state)
    torch.save(dict(state_dict=best_state, classes=classes, model=args.model), out / "checkpoint_best.pt")
    res = dict(model=args.model, classes=classes, seed=args.seed, steps=step, best_val_auroc_subsampled=best,
               hparams=vars(args), seconds=round(time.time() - t0), history=history)
    for s in U:
        P = predict(U[s]); res[s] = report(U[s].y, P)
        pd.DataFrame({"unit_id": U[s].unit_id, "patient_id": U[s].patient_id, "y": U[s].y,
                      **{f"p_{c}": P[:, k] for k, c in enumerate(classes)}}).to_csv(out / f"predictions_{s}.csv", index=False)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    print(json.dumps({s: res[s] for s in U}, indent=1))


if __name__ == "__main__":
    main()
