"""Create the Kaggle dataset, push the kernels, poll them, fetch their outputs.

Commands:
    python kaggle/run_kaggle.py dataset            # create/version the dataset
    python kaggle/run_kaggle.py push smoke|faithful|corrected|all
    python kaggle/run_kaggle.py status [ref ...]
    python kaggle/run_kaggle.py fetch smoke|faithful|corrected|all
    python kaggle/run_kaggle.py logs <ref>

Requires the new Kaggle auth (kaggle auth login). The legacy kaggle.json key is
rejected by the current API, so it must not shadow the OAuth credential.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\BSM Project 2")
K = ROOT / "kaggle"
DATASET = K / "dataset"
KERNELS = K / "kernels"
OUT = K / "output"

USER = "kiramada"
DS_ID = f"{USER}/bsm-paper-2-kaggle"
ACCELERATOR = "NvidiaTeslaP100"
KERNEL_SLUGS = {
    "smoke": f"{USER}/bsm-smoke-check",
    "faithful": f"{USER}/bsm-faithful-remaining",
    "corrected": f"{USER}/bsm-corrected-remaining",
}
KERNEL_FOLDERS = {
    "smoke": "bsm-smoke-check",
    "faithful": "bsm-faithful-remaining",
    "corrected": "bsm-corrected-remaining",
}


def cli(*args, check=True):
    cmd = ["kaggle", *args]
    print("$", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout[:4000], flush=True)
    if res.returncode != 0:
        print(res.stderr[:4000], flush=True)
        if check:
            raise SystemExit(f"command failed: {' '.join(cmd)}")
    return res


def dataset():
    ds_json = DATASET / "dataset-metadata.json"
    meta = json.loads(ds_json.read_text(encoding="utf-8"))
    meta["id"] = DS_ID
    ds_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    res = cli("datasets", "create", "-p", str(DATASET), "--dir-mode", "zip", check=False)
    combined = ((res.stdout or "") + (res.stderr or "")).lower()
    if res.returncode != 0 or "already" in combined or "error" in combined:
        print("dataset exists -> creating a new version", flush=True)
        cli("datasets", "version", "-p", str(DATASET), "-m", "refresh bundle", "--dir-mode", "zip")


def push(target):
    names = ["smoke", "faithful", "corrected"] if target == "all" else [target]
    for name in names:
        folder = KERNELS / KERNEL_FOLDERS[name]
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        meta["id"] = KERNEL_SLUGS[name]
        meta["dataset_sources"] = [DS_ID]
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        cli("kernels", "push", "-p", str(folder), "--accelerator", ACCELERATOR)


def status(refs):
    refs = refs or list(KERNEL_SLUGS.values())
    for ref in refs:
        res = cli("kernels", "status", ref, check=False)
        line = (res.stdout or res.stderr or "").strip().splitlines()[-1:] or [""]
        print(f"{ref}: {line[0]}", flush=True)


def fetch(target):
    names = ["smoke", "faithful", "corrected"] if target == "all" else [target]
    for name in names:
        ref = KERNEL_SLUGS[name]
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        cli("kernels", "output", ref, "-p", str(dest), check=False)


def logs(ref):
    cli("kernels", "logs", ref)


def main(argv):
    if not argv:
        print(__doc__)
        return
    cmd, *rest = argv
    if cmd == "dataset":
        dataset()
    elif cmd == "push":
        push(rest[0] if rest else "all")
    elif cmd == "status":
        status(rest)
    elif cmd == "fetch":
        fetch(rest[0] if rest else "all")
    elif cmd == "logs":
        logs(rest[0])
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
