"""common.py — shared data, training, metrics and plotting utilities for the
paper-faithful D-ResNet replication (Theerthagiri et al., MTA 2024).

Conventions (spec sections 5-8):
- Images: plain resize to 224x224 RGB, pixels scaled to [0,1] then standardised
  per channel with the TRAINING split mean/std (the paper only says
  "normalised" -> DEVIATIONS.md row 10).
- Training full-batch mode (paper: "batch size = 1028 = training set size"):
  micro-batches of 32 are accumulated over the whole training set and the
  optimiser steps ONCE per epoch. BatchNorm statistics are therefore
  per-micro-batch (DEVIATIONS.md row 4).
- Class order everywhere: Ilmenite, Rutile, Zircon, Garnet, Monazite, Sillimanite.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models import CLASS_ORDER

IMG_SIZE = 224
SEED = 42


# ---------------------------------------------------------------------------
# Environment / paths
# ---------------------------------------------------------------------------
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_root() -> Path:
    """Persistent root: Drive on Colab, local folder otherwise."""
    drive = Path("/content/drive/MyDrive/bsm_paper_replication")
    if drive.exists():
        return drive
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    return here / "bsm_paper_replication_local"


SUBDIRS = ["data", "augmented", "arrays", "checkpoints", "results", "figures", "versions"]


def ensure_dirs(root: Path | None = None) -> Path:
    root = root or project_root()
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class ManifestDataset(Dataset):
    """Loads images listed in a manifest CSV (columns: filepath, class_id).
    Plain resize to 224x224 (no crop — paper is silent, DEVIATIONS row 10)."""

    def __init__(self, manifest: pd.DataFrame, mean, std):
        self.items = list(zip(manifest["filepath"].tolist(), manifest["class_id"].astype(int).tolist()))
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            x = torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std, label


class ArrayDataset(Dataset):
    """Training-compatible arrays from notebook 03/04: uint8 (N,224,224,3)
    + int labels. Normalisation applied per item (train-split mean/std)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mean, std):
        assert x.dtype == np.uint8 and x.shape[1:] == (IMG_SIZE, IMG_SIZE, 3)
        self.x, self.y = x, y.astype(np.int64)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        t = torch.from_numpy(self.x[i]).permute(2, 0, 1).float() / 255.0
        return (t - self.mean) / self.std, int(self.y[i])


def channel_stats(x_uint8: np.ndarray) -> tuple:
    """Per-channel mean/std of an (N,224,224,3) uint8 array, over [0,1] pixels."""
    x = x_uint8.astype(np.float32) / 255.0
    return tuple(x.mean(axis=(0, 1, 2)).tolist()), tuple(x.std(axis=(0, 1, 2)).tolist())


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    # num_workers 0 on Windows avoids DataLoader deadlocks; 2 on Colab.
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )


# ---------------------------------------------------------------------------
# Training (spec section 7)
# ---------------------------------------------------------------------------
def make_optimizer(model, lr: float = 1e-3, wd: float = 0.0) -> torch.optim.Optimizer:
    """Unstated by the paper -> Adam, lr 1e-3, no weight decay (DEVIATIONS row 10)."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def _finish_full_batch(optimizer, scaler):
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, loader, optimizer, device, scaler, full_batch: bool) -> tuple:
    """One epoch. full_batch=True: accumulate over ALL training samples and
    perform ONE optimiser step (the paper's batch-size rule). Returns
    (mean CE loss, accuracy) computed over raw per-sample losses."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    use_amp = device.type == "cuda"
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
        if full_batch:
            scaler.scale(loss / len(loader)).backward()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    if full_batch:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device) -> tuple:
    """Returns (mean CE loss, accuracy, logits np, targets np)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    logits_all, targets_all = [], []
    use_amp = device.type == "cuda"
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
        logits_all.append(logits.float().cpu())
        targets_all.append(yb.cpu())
    return total_loss / n, correct / n, torch.cat(logits_all).numpy(), torch.cat(targets_all).numpy()


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    full_batch: bool,
    ckpt_dir: Path,
    history_path: Path,
    tag: str,
    lr: float = 1e-3,
    wd: float = 0.0,
    resume: bool = True,
    epoch_log_fn=print,
) -> dict:
    """Full training loop: validation after every epoch, BEST-VALIDATION-ACC
    checkpointing (paper silent on tested epoch -> DEVIATIONS row 11), history
    JSON after every epoch, resume checkpoint every 10 epochs."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer = make_optimizer(model, lr=lr, wd=wd)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_time_s": []}
    start_epoch, best_val = 0, -1.0
    resume_path = ckpt_dir / f"{tag}_resume.pt"

    if resume and resume_path.exists() and history_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        history = load_json(history_path)
        best_val = state.get("best_val", -1.0)
        epoch_log_fn(f"resumed from epoch {start_epoch}")

    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device, scaler, full_batch)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, device)
        dt = time.time() - t0
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["epoch_time_s"].append(dt)
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), ckpt_dir / f"{tag}_best.pt")
        save_json(history, history_path)
        if epoch % 10 == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch, "best_val": best_val},
                resume_path,
            )
        epoch_log_fn(
            f"{tag} epoch {epoch:02d}/{epochs}  train_loss {tr_loss:.4f} acc {tr_acc:.4f}"
            f"  val_loss {va_loss:.4f} acc {va_acc:.4f}  ({dt:.1f}s)"
        )

    best_epoch = int(np.argmax(history["val_acc"])) + 1
    history["best_val_acc"] = float(np.max(history["val_acc"]))
    history["best_epoch"] = best_epoch
    history["convergent_epoch"] = best_epoch  # paper's term: epoch of best validation accuracy
    save_json(history, history_path)
    model.load_state_dict(torch.load(ckpt_dir / f"{tag}_best.pt", map_location=device, weights_only=True))
    return history


# ---------------------------------------------------------------------------
# Metrics (spec section 8; paper Tables 3/6/7/8 layout)
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred, n_classes: int = 6) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm(cm: np.ndarray, class_names=CLASS_ORDER) -> pd.DataFrame:
    """Per-class precision TP/(TP+FP) [the paper's Eq. 3 is a typo, DEVIATIONS
    row 9], recall/sensitivity TP/(TP+FN), F1, specificity TN/(TN+FP);
    Weighted Average and Overall Accuracy rows."""
    n = cm.shape[0]
    rows = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {"Metrix/Mineral": class_names[i], "Precision": prec, "Recall": rec,
             "F1-Score": f1, "Specificity": spec, "Support": int(cm[i, :].sum())}
        )
    supports = cm.sum(axis=1).astype(float)
    total = supports.sum()
    w = supports / total
    df = pd.DataFrame(rows)
    weighted = {
        "Metrix/Mineral": "Weighted Average",
        **{c: float((df[c] * w).sum()) for c in ["Precision", "Recall", "F1-Score", "Specificity"]},
        "Support": int(total),
    }
    overall_acc = float(np.trace(cm) / total)
    overall = {
        "Metrix/Mineral": "Overall Accuracy", "Precision": overall_acc, "Recall": overall_acc,
        "F1-Score": overall_acc, "Specificity": np.nan, "Support": int(total),
    }
    return pd.concat([df, pd.DataFrame([weighted, overall])], ignore_index=True)


def full_metrics(y_true, y_pred, class_names=CLASS_ORDER) -> dict:
    cm = confusion(y_true, y_pred, len(class_names))
    table = metrics_from_cm(cm, class_names)
    return {"cm": cm, "table": table, "accuracy": float(np.trace(cm) / cm.sum())}


# ---------------------------------------------------------------------------
# Plots — paper figure styles (spec section 8)
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_loss_curves(history: dict, save_to: Path | None = None):
    """Paper Fig. 6: lines Train_loss / val_loss, title 'Loss per epoch'."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], label="Train_loss")
    ax.plot(history["epoch"], history["val_loss"], label="val_loss")
    ax.set_title("Loss per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_accuracy_curves(history: dict, save_to: Path | None = None):
    """Paper Fig. 7: lines Train_accuracy / val_accuracy, title 'accuracy per epoch'."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_acc"], label="Train_accuracy")
    ax.plot(history["epoch"], history["val_acc"], label="val_accuracy")
    ax.set_title("accuracy per epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.legend()
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str, class_names=CLASS_ORDER, save_to: Path | None = None):
    """Paper Figs. 8/10/11/12: annotated heatmap, 'True label' / 'Predicted
    Label', dark-to-light map with colour bar."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")  # dark-to-light (paper silent on the exact map)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_prediction_grid(images, pred_ids, true_ids, class_names=CLASS_ORDER,
                         save_to: Path | None = None, cols: int = 3):
    """Paper Figs. 13/14: grid of the 18 test images, each titled
    'Predicted (Actual)', green if correct, red if wrong."""
    plt = _plt()
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis("off")
            continue
        ax.imshow(images[i])
        correct = int(pred_ids[i]) == int(true_ids[i])
        ax.set_title(f"{class_names[pred_ids[i]]} ({class_names[true_ids[i]]})",
                     color="green" if correct else "red")
        ax.axis("off")
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# Model versioning / deployment (spec section 7)
# ---------------------------------------------------------------------------
def save_model_version(model, name: str, tag: str, hyper: dict, history: dict, out_dir: Path) -> dict:
    """Save weights + hyperparameters + history + SHA-256 checksum."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{name}_{tag}.pt"
    torch.save(model.state_dict(), weights_path)
    record = {
        "model": name,
        "version_tag": tag,
        "weights_file": weights_path.name,
        "sha256": sha256_of(weights_path),
        "hyperparameters": hyper,
        "history": history,
    }
    save_json(record, out_dir / f"{name}_{tag}.json")
    return record


def predict_image(model, img_path, mean, std, class_names=CLASS_ORDER, device=None) -> dict:
    """Deployment demo: classify one image file, return label + probabilities."""
    device = device or get_device()
    model.eval()
    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        x = torch.from_numpy(np.array(im, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
    x = ((x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return {"predicted": class_names[pred], "class_id": pred,
            "probs": {c: float(p) for c, p in zip(class_names, probs)}}
