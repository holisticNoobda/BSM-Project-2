"""Build the Kaggle bundle from the Colab notebooks.

Outputs (all generated, safe to delete and rebuild):

    kaggle/dataset/            -> uploaded as a private Kaggle dataset
        dataset-metadata.json
        code/                  bsm_env.py, run_queue.py, models.py, common.py
        repo/                  data/originals/**, data/raw/BSM/**
        notebooks/             Kaggle variants of the paper notebooks
        seed/                  committed results/results_corr JSONs
    kaggle/kernels/faithful/   -> kernel that runs 01..09
    kaggle/kernels/corrected/  -> kernel that runs the corrected notebooks

The only edits applied to the notebooks are environment adaptations:

 1. the Colab Drive mount block is removed (bsm_env recreates the folder tree);
 2. a bootstrap cell is prepended (imports bsm_env, sets BSM_RAW_SRC);
 3. nb01's dataset-acquisition cell reads /kaggle/input instead of kagglehub,
    and its inventory cell points CACHE at that mount;
 4. nb01's module-download cell falls back to the local copy if GitHub is
    unreachable.

No pipeline logic, hyper-parameter, split, seed or metric changes.
"""

import json
import re
import shutil
import uuid
from pathlib import Path

ROOT = Path(r"C:\BSM Project 2")
KAGGLE = ROOT / "kaggle"
DATASET = KAGGLE / "dataset"
KERNELS = KAGGLE / "kernels"
SRC_NB = ROOT / "notebooks"

USERNAME = "kiramada"
DATASET_SLUG = "bsm-paper-2-kaggle"
DATASET_ID = f"{USERNAME}/{DATASET_SLUG}"

FAITHFUL = [
    "01_eda",
    "02_data_augmentation",
    "03_imputation_wrangling_versioning",
    "04_split_70_15_15",
    "05_dresnet50",
    "06_resnet18",
    "07_resnet34",
    "08_vgg16",
    "09_comparison",
]
CORRECTED = [
    "06_corr_resnet18",
    "07_corr_resnet34",
    "08_corr_vgg16",
    "09_corr_comparison",
]
CRITICAL = ["01_eda", "02_data_augmentation", "03_imputation_wrangling_versioning", "04_split_70_15_15"]

BOOTSTRAP = """# ===== [KAGGLE] bootstrap: recreate the Colab layout =====
# The paper notebooks expect /content/drive/MyDrive/bsm_paper_replication and
# /content/BSM_Project_2. bsm_env builds both from the attached dataset; the
# pipeline itself (images, augmentation, splits, training) is unchanged.
import pathlib, sys
_code = pathlib.Path("/kaggle/input/bsm-paper-2-kaggle/code")
if not _code.is_dir():
    _hits = list(pathlib.Path("/kaggle/input").rglob("bsm_env.py"))
    _code = _hits[0].parent if _hits else _code
if str(_code) not in sys.path:
    sys.path.insert(0, str(_code))
import bsm_env
BSM_RAW_SRC = bsm_env.setup()
"""

ACQUISITION_CELL = """# ===== [KAGGLE] dataset acquisition =====
# On Colab this cell downloaded the public dataset [15] with kagglehub using Colab
# Secrets. On Kaggle the same 367 originals are attached read-only, so the
# acquisition step reduces to inspecting the mount.
import pathlib
from collections import Counter

CACHE = pathlib.Path(BSM_RAW_SRC)
files = [f for f in CACHE.rglob("*") if f.is_file()]
dirs = sorted({f.parent.name for f in files})
print("dataset mount:", CACHE)
print("top-level folders:", dirs)
print("total files:", len(files))
print("files per folder:", dict(Counter(f.parent.name for f in files)))
"""

DRIVE_LINE = re.compile(
    r"^\s*(from google\.colab import drive|drive\.mount\(|"
    r"if not os\.path\.exists\(\"/content/drive/MyDrive\"\)\s*:\s*$|"
    r"if not os\.path\.exists\(\"/content/drive/MyDrive\"\)\s*:\s*drive\.mount)",
)


def strip_drive(lines):
    out = []
    for line in lines:
        if DRIVE_LINE.match(line):
            out.append("# [kaggle] Colab Drive mount removed - bsm_env recreates the folder tree\n")
            continue
        out.append(line)
    # collapse duplicated marker comments
    dedup = []
    for line in out:
        if line.startswith("# [kaggle] Colab Drive mount removed") and dedup and dedup[-1].startswith("# [kaggle] Colab Drive"):
            continue
        dedup.append(line)
    return dedup


def patch_notebook(name):
    nb = json.loads((SRC_NB / f"{name}.ipynb").read_text(encoding="utf-8"))
    cells = nb["cells"]

    # nbformat allows cell["source"] to be a string or a list of lines
    for cell in cells:
        if isinstance(cell["source"], str):
            cell["source"] = cell["source"].splitlines(keepends=True)
        cell.setdefault("id", uuid.uuid4().hex[:8])
        cell.setdefault("metadata", {})
        if cell["cell_type"] == "code":
            # stale outputs from the Colab runs make the file invalid for Kaggle's
            # stricter nbformat validator, and they are not needed: the executed
            # copy produced by this run carries the fresh outputs.
            cell["outputs"] = []
            cell["execution_count"] = None

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        cell["source"] = strip_drive(cell["source"])

    if name == "01_eda":
        # replace the kagglehub acquisition cell
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if "kagglehub.dataset_download" in src and "userdata" in src:
                cell["source"] = ACQUISITION_CELL.splitlines(keepends=True)

        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            # point the inventory scan at the Kaggle mount
            if 'CACHE = pathlib.Path("/root/.cache/kagglehub' in src:
                cell["source"] = [
                    re.sub(r'CACHE = pathlib\.Path\("/root/\.cache/kagglehub[^"]*"\)',
                           "CACHE = pathlib.Path(BSM_RAW_SRC)", line)
                    for line in cell["source"]
                ]
            # tolerate a missing network for the module download
            if "urllib.request.urlretrieve" in src:
                new = []
                for line in cell["source"]:
                    if "urllib.request.urlretrieve(" in line:
                        indent = line[: len(line) - len(line.lstrip())]
                        new.append(f"{indent}try:\n")
                        new.append("    " + line)
                        new.append(f"{indent}except Exception as _dl_err:\n")
                        new.append(f"{indent}    print(\"  [kaggle] download failed, using local copy:\", _dl_err)\n")
                        new.append(f"{indent}    assert os.path.exists(dst), f\"{{fname}} missing and no network\"\n")
                    else:
                        new.append(line)
                cell["source"] = new

    boot = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": BOOTSTRAP.splitlines(keepends=True),
    }
    nb["cells"] = [boot] + cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    return nb


def write_notebook_kernel(path, title_slug, queue, critical):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {title_slug}\n",
                "\n",
                "Batch run on Kaggle. Cells below build the Colab folder layout from the attached\n",
                "dataset, then execute the paper notebooks in order with nbconvert.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, pathlib\n",
                "IN = pathlib.Path('/kaggle/input')\n",
                "print('input dirs:', sorted(p.name for p in IN.iterdir()) if IN.exists() else 'MISSING')\n",
                "for top in sorted(IN.iterdir()):\n",
                "    print(' ', top.name, '->', sorted(p.name for p in top.iterdir())[:12])\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                'import sys\n',
                'CODE = "/kaggle/input/bsm-paper-2-kaggle/code"\n',
                'if not pathlib.Path(CODE).is_dir():\n',
                '    hits = list(pathlib.Path("/kaggle/input").rglob("bsm_env.py"))\n',
                '    print("bsm_env candidates:", hits)\n',
                '    CODE = str(hits[0].parent) if hits else CODE\n',
                'sys.path.insert(0, CODE)\n',
                'import bsm_env\n',
                'BSM_RAW_SRC = bsm_env.setup()\n',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                'import run_queue\n',
                f'QUEUE = {queue!r}\n',
                f'CRITICAL = {critical!r}\n',
                "\n",
                "run_queue.main(QUEUE + [f\"--critical={','.join(CRITICAL)}\"])\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json, pathlib\n",
                "summary = json.loads(pathlib.Path('/kaggle/working/run_summary.json').read_text())\n",
                "print('total minutes:', summary['minutes'])\n",
                "for r in summary['runs']:\n",
                "    print(f\"{r['notebook']:35s} rc={r['returncode']:>3} {r['minutes']:6.1f} min  errors={len(r['errors'])}\")\n",
                "print()\n",
                "print('artefacts:', sorted(p.name for p in pathlib.Path('/kaggle/working/artifacts').iterdir()))\n",
            ],
        },
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for cell in nb["cells"]:
        cell.setdefault("id", uuid.uuid4().hex[:8])
        cell.setdefault("metadata", {})
    d = KERNELS / path
    d.mkdir(parents=True, exist_ok=True)
    (d / "kernel.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{USERNAME}/{title_slug}",
        "title": title_slug,
        "code_file": "kernel.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    if DATASET.exists():
        shutil.rmtree(DATASET)
    (DATASET / "code").mkdir(parents=True)

    # 1. code
    for f in ["models.py", "common.py"]:
        shutil.copy2(ROOT / f, DATASET / "code" / f)
    for f in ["bsm_env.py", "run_queue.py"]:
        shutil.copy2(KAGGLE / "src" / f, DATASET / "code" / f)

    # 2. images (originals + raw class folders)
    for sub, dst in [("data/originals", "repo/data/originals"), ("data/raw/BSM", "repo/data/raw/BSM")]:
        shutil.copytree(ROOT / sub, DATASET / dst)

    # 3. notebooks
    made = []
    for name in FAITHFUL + CORRECTED:
        nb = patch_notebook(name)
        (DATASET / "notebooks").mkdir(parents=True, exist_ok=True)
        (DATASET / "notebooks" / f"{name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
        made.append(name)
    print(f"patched {len(made)} notebooks -> {DATASET / 'notebooks'}")

    # 4. seed results (context for the comparison notebooks)
    for sub in ["results", "results_corr"]:
        (DATASET / "seed" / sub).mkdir(parents=True, exist_ok=True)
        for f in sorted((ROOT / sub).glob("*.json")):
            shutil.copy2(f, DATASET / "seed" / sub / f.name)

    # 5. dataset metadata
    meta = {
        "title": "BSM Project 2 Kaggle bundle",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (DATASET / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 6. kernels
    write_notebook_kernel("bsm-faithful-remaining", "bsm-faithful-remaining", FAITHFUL, CRITICAL)
    write_notebook_kernel("bsm-corrected-remaining", "bsm-corrected-remaining", CORRECTED, [])
    write_notebook_kernel("bsm-smoke-check", "bsm-smoke-check", ["01_eda"], ["01_eda"])

    total = sum(f.stat().st_size for f in DATASET.rglob("*") if f.is_file())
    print(f"dataset files: {sum(1 for f in DATASET.rglob('*') if f.is_file())}, {total / 1e6:.1f} MB")
    print("bundle ready:", DATASET)


if __name__ == "__main__":
    main()
