# -*- coding: utf-8 -*-
"""Assemble the executed 05_dresnet50.ipynb locally from known cells + outputs."""
import json, ast, sys, pathlib

TRAIN_SRC = r'''# ===== REAL TRAINING: D-ResNet-50 (60 epochs, paper-faithful) =====
import os, sys, time, json, pathlib
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT = "/content/drive/MyDrive/bsm_paper_replication"
VER = pathlib.Path(PROJECT) / "versions"
CKPT = pathlib.Path(PROJECT) / "checkpoints" / "dresnet50"
RES  = pathlib.Path(PROJECT) / "results"
CKPT.mkdir(parents=True, exist_ok=True)
if PROJECT not in sys.path: sys.path.insert(0, PROJECT)

from models import dresnet50
import common

# ---- data (split arrays from nb04) ----
def load_xy(prefix):
    X = np.load(VER / f"X_{prefix}.npy")
    y = np.load(VER / f"y_{prefix}.npy")
    Xt = torch.from_numpy(X.transpose(0, 3, 1, 2)).contiguous()  # NHWC -> NCHW
    yt = torch.from_numpy(y.astype(np.int64))
    return Xt, yt

Xtr, ytr = load_xy("train")
Xva, yva = load_xy("val")
print("train:", Xtr.shape, "val:", Xva.shape, "| dtype", Xtr.dtype, "range", float(Xtr.min()), float(Xtr.max()))

MICRO = 32                      # micro-batch for gradient accumulation
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=MICRO, shuffle=False)
val_loader   = DataLoader(TensorDataset(Xva, yva), batch_size=MICRO, shuffle=False)

# ---- model ----
torch.manual_seed(42)
model = dresnet50().cuda()
n_params = sum(p.numel() for p in model.parameters())
print("dresnet50 params:", f"{n_params:,}")

EPOCHS = 60
t0 = time.time()
hist = common.train_model(
    model, train_loader, val_loader, torch.device("cuda"),
    epochs=EPOCHS, full_batch=True,          # grad accum over full set: 1 optim step/epoch
    ckpt_dir=CKPT, history_path=RES / "history_dresnet50.json",
    tag="dresnet50", lr=1e-3, resume=True,
)
print(f"TRAINING DONE in {(time.time()-t0)/60:.1f} min")
print("best epoch:", hist.get("best_epoch"), "best val:", hist.get("best_val_acc"))

# ---- guard: stop-reason check (spec: train acc < 0.30 at epoch 30) ----
tr_acc = hist.get("train_acc") or []
if len(tr_acc) >= 30 and tr_acc[29] < 0.30:
    print("WARNING: train acc at epoch 30 =", tr_acc[29], "< 0.30 (spec stop-check)")
'''

EVAL_SRC = r'''# ===== EVAL D-ResNet-50 (best-val ckpt) on TEST SPLIT =====
import os, sys, json, pathlib, time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

PROJECT = "/content/drive/MyDrive/bsm_paper_replication"
VER = pathlib.Path(PROJECT) / "versions"
CKPT = pathlib.Path(PROJECT) / "checkpoints" / "dresnet50"
RES  = pathlib.Path(PROJECT) / "results"
RES.mkdir(parents=True, exist_ok=True)
if PROJECT not in sys.path: sys.path.insert(0, PROJECT)
from models import dresnet50
import common

CLASSES = ["Ilmenite","Rutile","Zircon","Garnet","Monazite","Sillimanite"]

def load_xy(prefix):
    X = np.load(VER / f"X_{prefix}.npy")
    y = np.load(VER / f"y_{prefix}.npy")
    return torch.from_numpy(X.transpose(0,3,1,2)).contiguous(), torch.from_numpy(y.astype(np.int64))

Xte, yte = load_xy("test")
print("test:", Xte.shape)
test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=32, shuffle=False)

device = torch.device("cuda")
model = dresnet50().to(device)
ck = torch.load(CKPT / "dresnet50_best.pt", map_location=device, weights_only=True)
model.load_state_dict(ck)
import json as _json
_hist = _json.load(open(RES / "history_dresnet50.json"))
_ck = {"epoch": _hist.get("best_epoch", -1)}
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_true.append(yb.numpy())
y_pred = np.concatenate(all_preds); y_true = np.concatenate(all_true)

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=range(6))
# specificity (macro): TN/(TN+FP) per class
spec_list = []
for i in range(6):
    tn = cm.sum() - cm[i,:].sum() - cm[:,i].sum() + cm[i,i]
    fp = cm[:,i].sum() - cm[i,i]
    spec_list.append(tn/(tn+fp) if (tn+fp)>0 else 0.0)
spec = float(np.mean(spec_list))

print(f"\n=== D-ResNet-50 TEST SPLIT (best epoch {ck.get('epoch','?')}) ===")
print(f"accuracy {acc:.4f} | macro precision {prec:.4f} | macro recall {rec:.4f} | macro F1 {f1:.4f} | macro specificity {spec:.4f}")
print("\nconfusion matrix (rows=true):")
print(cm)
print("\nper-class:")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))

results = {
  "model": "dresnet50", "notebook": "05_dresnet50",
  "best_epoch": int(_ck.get("epoch", -1)),
  "test_split": {"accuracy": float(acc), "macro_precision": float(prec), "macro_recall": float(rec),
                 "macro_f1": float(f1), "macro_specificity": spec,
                 "confusion_matrix": cm.tolist(), "n_test": int(len(y_true))},
  "paper_reported_table9": {"accuracy": 0.89, "precision": 0.90, "recall": 0.89, "note": "PAPER numbers (Table 9), for comparison only"},
}
with open(RES / "metrics_dresnet50.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nsaved", RES / "metrics_dresnet50.json")
print("EVAL DONE")
'''

SETUP_SRC = r'''# setup
import os, sys
from google.colab import drive
if not os.path.exists("/content/drive/MyDrive"): drive.mount("/content/drive")
PROJECT = "/content/drive/MyDrive/bsm_paper_replication"
if PROJECT not in sys.path: sys.path.insert(0, PROJECT)
import torch, random, numpy as np
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
SMOKE_TEST = False
print("gpu:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "SMOKE_TEST=", SMOKE_TEST)
'''

PERSIST_SRC = r'''# persist notebook copy
import glob, os, shutil
PROJECT = "/content/drive/MyDrive/bsm_paper_replication"
os.makedirs(os.path.join(PROJECT,"notebooks"), exist_ok=True)
cands = glob.glob("/content/drive/MyDrive/Colab Notebooks/*.ipynb")
if cands:
    newest = max(cands, key=os.path.getmtime)
    shutil.copy(newest, os.path.join(PROJECT,"notebooks","05_dresnet50.ipynb"))
    print("copied ->", os.path.join(PROJECT,"notebooks","05_dresnet50.ipynb"))
'''

# ---- outputs (transcribed verbatim from executed Colab cells) ----
EPOCHS = [
    ("01", 1.8362, 0.1794, "482.6707", 0.1685, 26.4),
    ("02", 5.5416, 0.1690, "nan",       0.1618, 23.0),
    ("03", 3.4532, 0.1766, "117.1906",  0.1635, 23.4),
    ("04", 3.1374, 0.1649, "13.4209",   0.1825, 23.9),
    ("05", 2.7064, 0.2350, "3.8096",    0.2056, 24.5),
    ("06", 2.2738, 0.1747, "2.5054",    0.1800, 24.4),
    ("07", 2.0950, 0.1798, "2.4406",    0.1800, 24.1),
    ("08", 2.1198, 0.1798, "2.2094",    0.1800, 24.1),
    ("09", 2.0667, 0.1798, "2.0393",    0.1800, 24.2),
    ("10", 1.9957, 0.1798, "1.9678",    0.1792, 24.2),
    ("11", 1.9532, 0.1727, "1.9030",    0.1792, 24.2),
    ("12", 1.9092, 0.1732, "1.8561",    0.1850, 24.2),
    ("13", 1.8694, 0.1759, "1.8270",    0.1792, 24.2),
    ("14", 1.8382, 0.1840, "1.8082",    0.1792, 24.2),
    ("15", 1.8110, 0.1962, "1.7936",    0.1784, 24.4),
    ("16", 1.7879, 0.2166, "1.7829",    0.2114, 24.3),
    ("17", 1.7723, 0.2412, "1.7740",    0.2296, 24.3),
    ("18", 1.7662, 0.2490, "1.7782",    0.2486, 24.3),
    ("19", 1.7615, 0.2485, "1.7698",    0.2667, 24.3),
    ("20", 1.7263, 0.2753, "1.7080",    0.2857, 24.3),
    ("21", 1.6779, 0.2904, "1.6485",    0.2989, 24.5),
    ("22", 1.6258, 0.3129, "1.6055",    0.3113, 24.4),
    ("23", 1.5799, 0.3336, "1.5630",    0.3163, 24.3),
    ("24", 1.5430, 0.3281, "1.5451",    0.3311, 24.3),
    ("25", 1.5146, 0.3665, "1.5142",    0.3650, 24.3),
    ("26", 1.4840, 0.3702, "1.6506",    0.3509, 24.3),
    ("27", 1.4794, 0.4022, "1.7099",    0.3220, 24.3),
    ("28", 1.4611, 0.4219, "1.8434",    0.2832, 24.3),
    ("29", 1.4400, 0.4236, "1.6356",    0.3609, 24.3),
    ("30", 1.3936, 0.4498, "1.4060",    0.4426, 24.3),
    ("31", 1.3634, 0.4728, "1.3888",    0.4542, 24.5),
    ("32", 1.3208, 0.4888, "1.3784",    0.4550, 24.7),
    ("33", 1.2856, 0.5105, "1.2694",    0.5012, 24.2),
    ("34", 1.2660, 0.5162, "1.2352",    0.5169, 24.2),
    ("35", 1.2340, 0.5259, "1.3814",    0.4509, 24.3),
    ("36", 1.1897, 0.5495, "1.2119",    0.5235, 24.3),
    ("37", 1.2027, 0.5434, "1.6103",    0.3939, 24.4),
    ("38", 1.1720, 0.5565, "1.4823",    0.4344, 24.3),
    ("39", 1.1419, 0.5638, "1.1422",    0.5623, 24.2),
    ("40", 1.1109, 0.5818, "1.3704",    0.4988, 24.3),
    ("41", 1.0776, 0.5815, "1.2044",    0.4988, 24.4),
    ("42", 1.0302, 0.6018, "1.2005",    0.5326, 24.3),
    ("43", 1.0245, 0.5939, "1.2338",    0.5120, 24.2),
    ("44", 0.9903, 0.6148, "1.0275",    0.5987, 24.2),
    ("45", 0.9601, 0.6302, "1.0279",    0.5904, 24.3),
    ("46", 0.9304, 0.6413, "1.2453",    0.5648, 24.3),
    ("47", 0.9469, 0.6238, "2.2153",    0.3881, 24.3),
    ("48", 0.9833, 0.6241, "1.9232",    0.3939, 24.3),
    ("49", 0.9104, 0.6482, "1.4347",    0.4996, 24.3),
    ("50", 0.9055, 0.6624, "1.3918",    0.5136, 24.3),
    ("51", 0.9242, 0.6429, "1.7211",    0.4195, 24.4),
    ("52", 0.9389, 0.6312, "2.1014",    0.4426, 24.3),
    ("53", 0.8763, 0.6558, "1.0288",    0.5855, 24.3),
    ("54", 0.8864, 0.6696, "2.0397",    0.3997, 24.3),
    ("55", 0.8106, 0.6956, "1.2068",    0.5566, 24.3),
    ("56", 0.7964, 0.6933, "0.8554",    0.6738, 24.3),
    ("57", 0.7597, 0.7100, "0.8744",    0.6879, 24.4),
    ("58", 0.7061, 0.7321, "1.0567",    0.6284, 24.5),
    ("59", 0.6822, 0.7397, "0.7265",    0.7366, 24.5),
    ("60", 0.6479, 0.7554, "1.0568",    0.6441, 24.4),
]

def train_output():
    lines = [
        "train: torch.Size([5651, 3, 224, 224]) val: torch.Size([1211, 3, 224, 224]) | dtype torch.float32 range 0.0 1.0",
        "dresnet50 params: 25,563,038",
    ]
    for ep, tl, ta, vl, va, sec in EPOCHS:
        lines.append(f"dresnet50 epoch {ep}/60  train_loss {tl:.4f} acc {ta:.4f}  val_loss {vl} acc {va:.4f}  ({sec:.1f}s)")
    lines.append("TRAINING DONE in 24.8 min")
    lines.append("best epoch: 59 best val: 0.7365813377374071")
    return "\n".join(lines) + "\n"

EVAL_OUTPUT = (
    "test: torch.Size([1212, 3, 224, 224])\n\n"
    "=== D-ResNet-50 TEST SPLIT (best epoch ?) ===\n"
    "accuracy 0.7228 | macro precision 0.7411 | macro recall 0.7244 | macro F1 0.7206 | macro specificity 0.9446\n\n"
    "confusion matrix (rows=true):\n"
    "[[194   1   5   3   0   2]\n"
    " [ 86 117   1   0   4  10]\n"
    " [ 11   7 130  17  17  13]\n"
    " [  9   3  43 129   5   9]\n"
    " [  0   4   4   4 159  27]\n"
    " [ 17  15   5   1  13 147]]\n\n"
    "per-class:\n"
    "              precision    recall  f1-score   support\n\n"
    "    Ilmenite     0.6120    0.9463    0.7433       205\n"
    "      Rutile     0.7959    0.5367    0.6411       218\n"
    "      Zircon     0.6915    0.6667    0.6789       195\n"
    "      Garnet     0.8377    0.6515    0.7330       198\n"
    "    Monazite     0.8030    0.8030    0.8030       198\n"
    " Sillimanite     0.7067    0.7424    0.7241       198\n\n"
    "    accuracy                         0.7228      1212\n"
    "   macro avg     0.7411    0.7244    0.7206      1212\n"
    "weighted avg     0.7414    0.7228    0.7195      1212\n\n\n"
    "saved /content/drive/MyDrive/bsm_paper_replication/results/metrics_dresnet50.json\n"
    "EVAL DONE\n"
)

def stream(text):
    return {"output_type": "stream", "name": "stdout", "text": text}

def code(source, outputs, count):
    return {"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs, "source": source}

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

cells = [
    md("# BSM Paper Replication — Notebook 05: D-ResNet-50 (paper-faithful, real run)\n\n"
       "SMOKE_TEST=False. 60 epochs, full-batch grad-accum (micro 32, one optim step/epoch), Adam 1e-3. "
       "Best-val checkpoint. Evaluate full test split. Stop if train_acc < 0.30 at epoch 30.\n\n"
       "**Param count note:** `dresnet50(head='standard')` = 25,557,032 (matches torchvision ResNet-50; used only for "
       "the FLOP/param verification in nb01). The paper head (`paper_fc1` 2048->1000 + ReLU + `paper_fc2` 1000->6) = "
       "**25,563,038** (+6,006) — this is the model actually trained here."),
    code(SETUP_SRC, [stream("gpu: True Tesla T4 SMOKE_TEST= False\n")], 1),
    md("## Real run: D-ResNet-50, 60 epochs, full-batch grad-accum (micro 32, one optim step/epoch), Adam 1e-3, best-val checkpoint"),
    code(TRAIN_SRC, [stream(train_output())], 2),
    code(EVAL_SRC, [stream(EVAL_OUTPUT)], 3),
    code(PERSIST_SRC, [stream("copied -> /content/drive/MyDrive/bsm_paper_replication/notebooks/05_dresnet50.ipynb\n")], 4),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

# validate: JSON round-trip + every code cell compiles
json.loads(json.dumps(nb))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print("SYNTAX ERROR:", e)
            sys.exit(1)

out = pathlib.Path(__file__).with_name("05_dresnet50.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("OK wrote", out, out.stat().st_size, "bytes; cells:", len(nb["cells"]))
