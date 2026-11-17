# KERAFound: foundation models for the external eye and anterior segment

KERAFound is a family of ViT-L/16 vision foundation models for corneal and anterior segment imaging. Each model is
continued self-supervised pretraining of a general-purpose encoder on unlabelled images of one modality:

- **KERAFound-DINOv3** continues from Meta's DINOv3 ViT-L/16 (LVD-1689M), using DINOv3 self-distillation.
- **KERAFound-MAE** continues from the ImageNet MAE ViT-L/16, using masked autoencoding.

This repository has the code to load the models and adapt them to your own task by **linear probing** (frozen
features) or **fine-tuning**. It uses the same recipes as the KERAFound benchmark.

## Models

All models are ViT-L/16 (about 303M parameters) with 224×224 input and 1,024-dimensional features. The weights are
hosted on HuggingFace with gated access; see [Request access](#1-request-access-to-the-weights) below.

| Model | Pretraining method | Imaging modality | Weights |
|---|---|---|---|
| KERAFound-DINOv3-OCT | DINOv3 (continued from LVD-1689M) | Anterior segment OCT B-scans | [🤗 R2Squared/KERAFound-DINOv3-OCT](https://huggingface.co/R2Squared/KERAFound-DINOv3-OCT) |
| KERAFound-DINOv3-Placido | DINOv3 (continued from LVD-1689M) | Placido-disc corneal topography | [🤗 R2Squared/KERAFound-DINOv3-Placido](https://huggingface.co/R2Squared/KERAFound-DINOv3-Placido) |
| KERAFound-DINOv3-Photo | DINOv3 (continued from LVD-1689M) | Anterior segment / corneal photographs | [🤗 R2Squared/KERAFound-DINOv3-Photo](https://huggingface.co/R2Squared/KERAFound-DINOv3-Photo) |
| KERAFound-MAE-OCT | MAE (continued from ImageNet MAE) | Anterior segment OCT B-scans | [🤗 R2Squared/KERAFound-MAE-OCT](https://huggingface.co/R2Squared/KERAFound-MAE-OCT) |
| KERAFound-MAE-Placido | MAE (continued from ImageNet MAE) | Placido-disc corneal topography | [🤗 R2Squared/KERAFound-MAE-Placido](https://huggingface.co/R2Squared/KERAFound-MAE-Placido) |

Each HuggingFace repository has `model.safetensors`, which holds the encoder only (no DINO/iBOT heads, MAE decoder or
optimiser state), and `config.json`.

**Which model?** Use the model that matches your imaging modality. For OCT and Placido, both a DINOv3 and an MAE
model are available, and the best one depends on the task. Trying both is cheap with a linear probe.

## Installation

```bash
git clone https://github.com/R2Squared/KERAFound.git
cd KERAFound
conda create -n kerafound python=3.11 -y && conda activate kerafound
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -r requirements.txt
```

`requirements.txt` installs the DINOv3 model code from Meta's repository, pinned to the commit used for KERAFound.

## Quick start

### 1. Request access to the weights
1. Sign in to HuggingFace and open the model page, for example
   [R2Squared/KERAFound-DINOv3-OCT](https://huggingface.co/R2Squared/KERAFound-DINOv3-OCT).
2. Accept the terms to request access.
3. Once access is granted, log in on the machine you will train on:

```bash
huggingface-cli login     # paste a HuggingFace access token with read permission
```

The scripts download the weights automatically on first use. To use weights you have already downloaded, pass
`--weights /path/to/model.safetensors`, or the folder that contains it.

### 2. Prepare your data

Images can be PNG, JPEG, BMP or TIFF. Every image is converted to greyscale and resized to 224×224 (bicubic, no
cropping), which is exactly the preprocessing used in the KERAFound benchmark. Two layouts are supported.

**Folder layout** (each image is one sample):

```
data/keratoconus/
├── train/
│   ├── 0_normal/       *.png
│   └── 1_keratoconus/  *.png
├── val/
│   ├── 0_normal/
│   └── 1_keratoconus/
└── test/
    ├── 0_normal/
    └── 1_keratoconus/
```

**CSV layout:** recommended for OCT volumes, where one eye has many B-scans.

```csv
path,label,split,unit_id,patient_id
images/P001_OD_b00.png,keratoconus,train,P001_OD,P001
images/P001_OD_b01.png,keratoconus,train,P001_OD,P001
images/P002_OS_b00.png,normal,val,P002_OS,P002
```

- `path` is relative to the CSV, or absolute.
- `unit_id` groups the images of one eye (or visit). During training, each epoch samples one random image per unit.
  At evaluation, predictions are averaged over all images of the unit, and metrics are computed per unit.
- `unit_id` and `patient_id` are optional. Split your data **by patient** so that no patient appears in more than one
  split.

**Class labels:** classes are ordered alphabetically. For binary tasks, the second class is treated as positive when
computing AUROC. Prefixes such as `0_normal` and `1_keratoconus` make the order explicit. More than two classes are
supported; the script then reports macro one-vs-rest AUROC.

### 3. Linear probe

```bash
python linear_probe.py --model KERAFound-DINOv3-OCT --data data/keratoconus --out runs/kc_lp
# or, with the CSV layout:
python linear_probe.py --model KERAFound-DINOv3-OCT --csv data/keratoconus.csv --out runs/kc_lp
```

How it works:
- GAP features (the mean of the final-layer patch tokens) are extracted once from the frozen encoder and standardised.
- A logistic regression with an L2 penalty is fitted full-batch with LBFGS. Each unit gets equal weight, however many
  images it has.
- The L2 strength is chosen from {1e-6 … 10} by validation AUROC.

**Outputs:** `metrics.json`, `predictions_{train,val,test}.csv` (probabilities per unit) and `linear_head.pt`.

### 4. Fine-tuning

```bash
python finetune.py --model KERAFound-DINOv3-OCT --data data/keratoconus --out runs/kc_ft_seed0 --seed 0
```

The default recipe is the one used for every encoder in the KERAFound benchmark (RETFound-style):

| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 0.05 |
| Learning rate | 5e-4 peak, layer-wise decay 0.65, 10% linear warm-up then cosine |
| Regularisation | drop path 0.2, label smoothing 0.1 |
| Batch size | 64 (bf16 mixed precision) |
| Length | 30 epochs, minimum 600 steps |
| Augmentation | random resized crop (scale 0.6–1.0), horizontal flip, brightness/contrast ±0.2 |
| Model selection | validation AUROC checked 30 times on ≤4 images per unit; best checkpoint kept; early stop after 10 checks without improvement |
| Head | GAP → LayerNorm → Linear |

Every setting can be changed from the command line (`python finetune.py -h`).
- **Memory:** one ViT-L fine-tuning run at batch 64 needs about 24 GB of GPU memory. For smaller GPUs, use
  `--batch 32`.
- **Seeds:** in the paper, results are 5-seed ensembles (seeds 0–4, mean probability per unit).

**Outputs:** `metrics.json`, `predictions_{train,val,test}.csv` and `checkpoint_best.pt`.

### 5. Feature extraction only

```bash
python extract_features.py --model KERAFound-DINOv3-Placido --images my_topographies/ --out features.npz
```

This writes `paths`, `gap` (n×1024) and `cls` (n×1024) arrays.

### Use in your own code

```python
import torch
from kerafound.models import build_encoder, Classifier
from kerafound.data import load_image, to_input

encoder = build_encoder("KERAFound-DINOv3-OCT").eval()      # downloads from HuggingFace (or weights="path")
x = to_input(load_image("bscan.png").unsqueeze(0))          # (1, 3, 224, 224), ImageNet-normalised greyscale
with torch.no_grad():
    gap, cls = encoder(x)                                   # (1, 1024) each

model = Classifier(build_encoder("KERAFound-MAE-OCT", drop_path=0.2), n_classes=2)   # ready to fine-tune
```

## Repository layout

```
kerafound/models.py    encoders, weight loading, classifier, layer-wise lr decay
kerafound/data.py      preprocessing, folder/CSV datasets
kerafound/metrics.py   unit-level AUROC / AUPRC / accuracy
linear_probe.py        frozen-feature logistic regression
finetune.py            end-to-end fine-tuning
extract_features.py    GAP / CLS feature export
tools/                 weight export and HuggingFace upload (maintainers)
```

## Citation

Citation details will be added on publication. If you use KERAFound before then, please cite this repository.

KERAFound-DINOv3 models are built with DINOv3. Please also cite DINOv3 (Siméoni et al., 2025, arXiv:2508.10104). For
KERAFound-MAE, also cite MAE (He et al., CVPR 2022).

## Licence

- **Code:** CC BY-NC 4.0.
- **KERAFound-MAE weights:** CC BY-NC 4.0.
- **KERAFound-DINOv3 weights:** the [DINOv3 License](LICENSE_DINOv3.md).

See [LICENSE](LICENSE). The models are research tools. They are not medical devices and must not be used for
clinical decision-making.

## Acknowledgements

The recipes and repository structure follow [RETFound](https://github.com/rmaphoh/RETFound). The model code builds
on [DINOv3](https://github.com/facebookresearch/dinov3), [MAE](https://github.com/facebookresearch/mae) and
[timm](https://github.com/huggingface/pytorch-image-models).
