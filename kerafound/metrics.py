"""Unit-level metrics: AUROC (binary) or macro one-vs-rest AUROC (multi-class), plus accuracy."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def unit_probs(probs, unit_idx, n_units):
    """Average image probabilities (n_images, K) to units."""
    c = np.maximum(np.bincount(unit_idx, minlength=n_units), 1)[:, None]
    return np.stack([np.bincount(unit_idx, weights=probs[:, k], minlength=n_units) for k in range(probs.shape[1])], 1) / c


def score(y, P):
    """Model-selection score: AUROC for two classes, macro one-vs-rest AUROC otherwise."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2: return float("nan")
    if P.shape[1] == 2: return float(roc_auc_score(y, P[:, 1]))
    return float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(P.shape[1]))))


def report(y, P):
    m = dict(auroc=score(y, P), accuracy=float(accuracy_score(y, P.argmax(1))), n_units=int(len(y)))
    if P.shape[1] == 2 and len(np.unique(y)) == 2: m["auprc"] = float(average_precision_score(y, P[:, 1]))
    return m
