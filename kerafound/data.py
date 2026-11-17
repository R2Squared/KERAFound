"""Datasets. Images are read as greyscale, resized (squashed, bicubic) to 224x224 without cropping, replicated to three
channels and ImageNet-normalised, exactly as in the KERAFound benchmark.

Two layouts are accepted:
  folder  <data>/{train,val,test}/<class name>/<image>      (each image is its own unit)
  csv     a CSV with columns path, label, split (train/val/test) and optionally unit_id (e.g. an eye: all B-scans of
          one eye share a unit_id) and patient_id. Predictions are averaged over images of the same unit, and training
          samples one random image per unit per epoch.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_table(data=None, csv=None):
    """Return (frame with path, label, split, unit_id, patient_id, y), class names."""
    if csv:
        df = pd.read_csv(csv)
        root = Path(csv).parent
        df["path"] = [str(p if Path(p).is_absolute() else root / p) for p in df.path]
    else:
        root = Path(data)
        df = pd.DataFrame([dict(path=str(p), label=p.parent.name, split=s) for s in ("train", "val", "test")
                           for p in sorted((root / s).rglob("*")) if p.suffix.lower() in EXTS])
        if df.empty: raise FileNotFoundError(f"no images under {root}/{{train,val,test}}/<class>/")
    df["label"] = df.label.astype(str)
    if "unit_id" not in df: df["unit_id"] = df.path
    if "patient_id" not in df: df["patient_id"] = df.unit_id
    classes = sorted(df.label.unique())
    df["y"] = df.label.map({c: i for i, c in enumerate(classes)})
    return df.reset_index(drop=True), classes


def load_image(path):
    a = np.asarray(Image.open(path).convert("L").resize((224, 224), Image.BICUBIC), dtype=np.uint8)
    return torch.from_numpy(a.copy())


def to_input(x_u8):
    """(B,224,224) uint8 -> (B,3,224,224) ImageNet-normalised float."""
    x = x_u8.float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def units(df):
    """One row per unit with the list of its image paths."""
    return df.groupby("unit_id", sort=True).agg(patient_id=("patient_id", "first"), y=("y", "first"),
                                                 paths=("path", list)).reset_index()


class TrainSet(Dataset):
    """One random image per unit per epoch, with light augmentation (crop/rescale, flip, brightness/contrast)."""
    def __init__(self, u):
        from torchvision.transforms import v2
        self.paths, self.y = list(u.paths), u.y.to_numpy()
        self.aug = v2.Compose([v2.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.333), antialias=True),
                               v2.RandomHorizontalFlip(), v2.ColorJitter(brightness=0.2, contrast=0.2)])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i][np.random.randint(len(self.paths[i]))]
        return self.aug(load_image(p).unsqueeze(0))[0], int(self.y[i])


class ImageSet(Dataset):
    """Every image of the given units, returning (image, unit index)."""
    def __init__(self, u, max_per_unit=None):
        self.items = []
        for k, ps in enumerate(u.paths):
            if max_per_unit and len(ps) > max_per_unit:
                ps = [ps[j] for j in np.linspace(0, len(ps) - 1, max_per_unit).round().astype(int)]
            self.items += [(p, k) for p in ps]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        p, k = self.items[i]
        return load_image(p), k
