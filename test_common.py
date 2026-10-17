"""Local functional test of common.py + models.py on the real local dataset:
tiny subset, 1 epoch full-batch, metrics + plots. Exits 0 on success."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"C:\BSM Project 2")
import common as C
from models import CLASS_ORDER, resnet18

seed = C.seed_everything(42)
device = C.get_device()
print("device:", device)

# Real images -> tiny uint8 arrays (30 images, 5 per class).
from PIL import Image  # noqa: E402

root = Path(r"C:\BSM Project 2\data\raw\BSM")
rows = []
for cid, name in enumerate(CLASS_ORDER):
    files = sorted((root / name).glob("*"))[:5]
    for f in files:
        with Image.open(f) as im:
            arr = np.asarray(im.convert("RGB").resize((224, 224)), dtype=np.uint8)
        rows.append((arr, cid))
x = np.stack([r[0] for r in rows])
y = np.array([r[1] for r in rows])
print("arrays:", x.shape, y.dtype, x.dtype)

mean, std = C.channel_stats(x)
print("mean/std:", [round(m, 4) for m in mean], [round(s, 4) for s in std])

ds = C.ArrayDataset(x, y, mean, std)
tr, va = torch.utils.data.random_split(ds, [20, 10], generator=torch.Generator().manual_seed(42))
tr_loader = C.make_loader(tr, batch_size=8, shuffle=True)
va_loader = C.make_loader(va, batch_size=8, shuffle=False)

model = resnet18(num_classes=6).to(device)
out_dir = Path(r"C:\BSM Project 2\_local_test")
hist = C.train_model(model, tr_loader, va_loader, device, epochs=1, full_batch=True,
                     ckpt_dir=out_dir, history_path=out_dir / "h.json", tag="t1")
print("history keys ok:", sorted(hist.keys())[:3], "best_val:", hist["best_val_acc"])

loss, acc, logits, targets = C.evaluate(model, va_loader, device)
preds = logits.argmax(1)
m = C.full_metrics(targets.tolist(), preds.tolist())
print("acc:", round(m["accuracy"], 3))
print(m["table"][["Metrix/Mineral", "Precision", "Recall", "F1-Score", "Specificity"]].round(3).to_string(index=False))

C.plot_loss_curves(hist, out_dir / "loss.png")
C.plot_accuracy_curves(hist, out_dir / "acc.png")
C.plot_confusion_matrix(m["cm"], "Test", save_to=out_dir / "cm.png")
imgs = [x[i] for i in range(18)]
C.plot_prediction_grid(imgs, list(preds[:18]) if len(preds) >= 18 else list(preds) + [0] * (18 - len(preds)),
                       list(targets[:18]) if len(targets) >= 18 else list(targets) + [0] * (18 - len(targets)),
                       save_to=out_dir / "grid.png")
rec = C.save_model_version(model, "resnet18", "test", {"lr": 1e-3}, hist, out_dir)
print("version record sha256:", rec["sha256"][:16])
pred = C.predict_image(model, str(next((root / "Garnet").glob("*"))), mean, std)
print("predict_image:", pred["predicted"], round(max(pred["probs"].values()), 3))
print("COMMON FUNCTIONAL TEST: PASSED")
