"""Kaggle execution shim for the BSM paper replication (Project 2).

The paper notebooks were written for Colab and expect this layout:

    /content/drive/MyDrive/bsm_paper_replication   <- project dir (arrays, ckpt, results)
    /content/BSM_Project_2                         <- repo checkout (models.py, common.py, data/)

This module recreates that layout inside a Kaggle kernel, where the read-only
inputs live under /kaggle/input and only /kaggle/working is persisted.

Nothing about the pipeline changes: same images, same augmentation table, same
splits, same training protocol. Only filesystem paths are adapted.

Dataset layout (attached as /kaggle/input/bsm-paper-2-kaggle):

    code/       bsm_env.py, run_queue.py, models.py, common.py
    repo/       data/originals/<Class>/*.jpg, data/raw/BSM/<Class>/*.jpg
    notebooks/  Kaggle variants of the paper notebooks
    seed/       committed results/results_corr JSONs (context for the comparison notebooks)
"""

import os
import pathlib
import shutil
import sys
import zipfile

DATASET = pathlib.Path("/kaggle/input/bsm-paper-2-kaggle")
PROJECT = pathlib.Path("/content/drive/MyDrive/bsm_paper_replication")
REPO = pathlib.Path("/content/BSM_Project_2")
MARKER = pathlib.Path("/content/.bsm_env_ready")
STAGE = pathlib.Path("/content/.bsm_dataset")


def _resolve_dataset():
    """Find the attached dataset root (Kaggle has used several mount layouts)."""
    candidates = [
        pathlib.Path("/kaggle/input/bsm-paper-2-kaggle"),
        pathlib.Path("/kaggle/input/datasets/kiramada/bsm-paper-2-kaggle"),
    ]
    for cand in candidates:
        if (cand / "code").is_dir() or (cand / "code.zip").exists():
            return cand
    root = pathlib.Path("/kaggle/input")
    if root.exists():
        for hit in root.rglob("bsm_env.py"):
            return hit.parent.parent
    raise FileNotFoundError(
        "dataset 'bsm-paper-2-kaggle' not found under /kaggle/input - attach it to this kernel"
    )


def _dataset():
    global DATASET
    if not (DATASET / "code").is_dir() and not (DATASET / "code.zip").exists():
        DATASET = _resolve_dataset()
    return DATASET

PROJECT_SUBDIRS = [
    "data", "augmented", "arrays", "arrays_corr", "checkpoints", "checkpoints_corr",
    "results", "results_corr", "figures", "versions", "notebooks",
]

COMPONENTS = ("code", "repo", "notebooks", "seed")


def _component(name):
    """Locate a dataset component, transparently extracting it from a zip bundle.

    Kaggle's CLI uploads sub-folders as <name>.zip, so both layouts are supported.
    """
    plain = DATASET / name
    if plain.is_dir():
        return plain
    bundle = DATASET / f"{name}.zip"
    if bundle.exists():
        target = STAGE / name
        if not (target / ".extracted").exists():
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(bundle) as zf:
                zf.extractall(target)
            (target / ".extracted").write_text("ok", encoding="utf-8")
        return target
    return None


def _ensure_content_root(verbose=True):
    """Return /content as a writable directory, symlinking it if it cannot be created."""
    root = pathlib.Path("/content")
    if root.exists() and os.access(root, os.W_OK):
        if verbose:
            print(f"[bsm_env] using existing {root}")
        return "existing"

    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        if verbose:
            print(f"[bsm_env] created {root}")
        return "created"
    except OSError as exc:
        target = pathlib.Path("/kaggle/working/content")
        target.mkdir(parents=True, exist_ok=True)
        if not root.exists():
            os.symlink(target, root)
        if verbose:
            print(f"[bsm_env] {root} not creatable ({exc}); using symlink -> {target}")
        return "symlink"


def setup(force=False, verbose=True):
    """Create the Colab folder layout. Idempotent and safe to call in every notebook.

    Returns the path to the raw BSM image folders (the class dirs).
    """
    mode = _ensure_content_root(verbose)

    _dataset()  # resolve the mount layout first
    if not DATASET.exists():
        raise FileNotFoundError(
            f"dataset not found at {DATASET} - attach the 'bsm-paper-2-kaggle' dataset to this kernel"
        )

    marker_ok = MARKER.exists() and not force
    if not marker_ok:
        comp = {name: _component(name) for name in COMPONENTS}
        missing = [k for k, v in comp.items() if v is None or not v.exists()]
        if missing:
            raise FileNotFoundError(f"dataset components missing from {DATASET}: {missing}")

        # repo checkout + shared modules
        REPO.mkdir(parents=True, exist_ok=True)
        shutil.copytree(comp["repo"], REPO, dirs_exist_ok=True)
        shutil.copytree(comp["code"], REPO, dirs_exist_ok=True)
        if comp["notebooks"] is not None:
            shutil.copytree(comp["notebooks"], REPO / "notebooks_kaggle", dirs_exist_ok=True)

        # project dir
        PROJECT.mkdir(parents=True, exist_ok=True)
        shutil.copytree(comp["code"], PROJECT, dirs_exist_ok=True)
        if comp["seed"] is not None:
            shutil.copytree(comp["seed"], PROJECT, dirs_exist_ok=True)

        MARKER.write_text(f"mode={mode}\n", encoding="utf-8")

    for sub in PROJECT_SUBDIRS:
        (PROJECT / sub).mkdir(parents=True, exist_ok=True)

    for p in (str(REPO), str(PROJECT)):
        if p not in sys.path:
            sys.path.insert(0, p)

    raw = REPO / "data" / "raw" / "BSM"
    if not raw.exists():
        raise FileNotFoundError(f"raw image folders missing at {raw}")

    if verbose:
        n_raw = sum(1 for c in raw.iterdir() if c.is_dir() for _ in c.iterdir())
        n_orig = REPO / "data" / "originals"
        n_orig_count = sum(1 for c in n_orig.iterdir() if c.is_dir() for _ in c.iterdir()) if n_orig.exists() else 0
        print(f"[bsm_env] mode={mode}  project={PROJECT}")
        print(f"[bsm_env] raw images={n_raw}  originals={n_orig_count}")
        for mod in ("models.py", "common.py"):
            print(f"[bsm_env] {mod}: {'ok' if (PROJECT / mod).exists() and (REPO / mod).exists() else 'MISSING'}")
        try:
            import torch
            print(f"[bsm_env] torch {torch.__version__} cuda={torch.cuda.is_available()} "
                  f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[bsm_env] torch check failed: {exc}")
        usage = shutil.disk_usage("/kaggle/working")
        print(f"[bsm_env] disk /kaggle/working free={usage.free / 1e9:.1f} GB")
    return str(raw)


if __name__ == "__main__":
    setup(force=True)
