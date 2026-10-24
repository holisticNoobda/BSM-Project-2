"""Merge a finished Kaggle kernel's artefacts back into the repo.

    python kaggle/collect_results.py faithful
    python kaggle/collect_results.py corrected

Copies results / results_corr / figures / executed notebooks out of
kaggle/output/<name>/ and prints a markdown summary table of every metrics JSON
it finds (new run values only - paper numbers are never mixed in).

Nothing is overwritten silently: every file that differs from the repo copy is
reported, and the old value is kept in the printed diff (results/*.json backup
is written next to the new file as *.prev.json when the content differs).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(r"C:\BSM Project 2")
OUT = ROOT / "kaggle" / "output"


def merge(name):
    src = OUT / name
    if not src.exists():
        raise SystemExit(f"no output at {src} - run 'python kaggle/run_kaggle.py fetch {name}' first")

    copied, changed = [], []
    pairs = [("artifacts/results", ROOT / "results"),
             ("artifacts/results_corr", ROOT / "results_corr"),
             ("artifacts/figures", ROOT / "figures")]
    # each kernel only owns its own result folder; the other one is a *seed*
    # (faithful JSONs copied into the corrected kernel at startup and vice versa),
    # so merging it would overwrite the fresh run with the old numbers
    if name == "corrected":
        pairs = [p for p in pairs if p[0] != "artifacts/results"]
    elif name == "faithful":
        pairs = [p for p in pairs if p[0] != "artifacts/results_corr"]
    for sub, dest in pairs:
        s = src / sub
        if not s.exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(s.rglob("*")):
            if not f.is_file():
                continue
            target = dest / f.relative_to(s)
            if target.exists() and target.read_bytes() != f.read_bytes():
                changed.append(target.relative_to(ROOT).as_posix())
                shutil.copy2(target, target.with_suffix(target.suffix + ".prev"))
            shutil.copy2(f, target)
            copied.append(target.relative_to(ROOT).as_posix())

    ex = src / "executed"
    if ex.exists():
        dest = ROOT / "executed"
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(ex.glob("*_executed.ipynb")):
            target = dest / f.name.replace("_executed", "")
            shutil.copy2(f, target)
            copied.append(target.relative_to(ROOT).as_posix())

    print(f"{name}: merged {len(copied)} file(s) into the repo")
    for c in copied:
        print("   ", c)
    if changed:
        print(f"   replaced {len(changed)} existing file(s) (previous copies kept as *.prev):")
        for c in changed:
            print("     ", c)


def summary():
    rows = []
    for folder in (ROOT / "results", ROOT / "results_corr"):
        for f in sorted(folder.glob("metrics_*.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            ts = m.get("test_split", m)
            re_ = m.get("re_engineering")
            if isinstance(re_, list):
                reeng = "round1" if len(re_) > 1 else "no"
            elif isinstance(re_, dict):
                reeng = "yes" if re_.get("triggered") else "no"
            else:
                reeng = ""
            rows.append({
                "json": f"{folder.name}/{f.name}",
                "best_ep": m.get("best_epoch", ""),
                "acc": ts.get("accuracy"),
                "P": ts.get("macro_precision"),
                "R": ts.get("macro_recall"),
                "F1": ts.get("macro_f1"),
                "spec": ts.get("macro_specificity"),
                "n": ts.get("n_test"),
                "min": m.get("train_minutes"),
                "reeng": reeng,
            })
    if not rows:
        print("no metrics JSONs found")
        return
    hdr = ["json", "best_ep", "acc", "P", "R", "F1", "spec", "n", "min", "reeng"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        cells = []
        for h in hdr:
            v = r[h]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("faithful", "corrected"):
        merge(arg)
    elif arg == "summary":
        summary()
    else:
        print(__doc__)
        print("known kernels: faithful, corrected")
