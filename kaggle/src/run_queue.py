"""Run the BSM notebooks in sequence inside a Kaggle kernel, then collect artefacts.

Each notebook is executed with jupyter nbconvert in its own kernel process, in the
same container, so files written by an earlier stage (augmented PNGs, arrays,
versions, checkpoints) are visible to the next one - exactly like the Colab runs.

Usage:
    python run_queue.py 01_eda 02_data_augmentation ... [--critical 01_eda,02_data_augmentation]
"""

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/bsm_paper_replication")
REPO = Path("/content/BSM_Project_2")
NB_DIR = REPO / "notebooks_kaggle"
WORK = Path("/kaggle/working")
EXECUTED = WORK / "executed"
LOGS = WORK / "logs"
ARTIFACTS = WORK / "artifacts"

DEFAULT_CRITICAL = {"01_eda", "02_data_augmentation", "03_imputation_wrangling_versioning", "04_split_70_15_15"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_errors(path):
    """Return a list of (cell index, ename, evalue) for error outputs in an executed notebook."""
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    bad = []
    for i, cell in enumerate(nb.get("cells", [])):
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "error":
                bad.append((i, out.get("ename", "?"), out.get("evalue", "")))
    return bad


def run_one(name, timeout_s):
    src = NB_DIR / f"{name}.ipynb"
    if not src.exists():
        raise FileNotFoundError(src)
    dst = EXECUTED / f"{name}_executed.ipynb"
    log_path = LOGS / f"{name}.log"
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--allow-errors",
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
        "--output-dir", str(EXECUTED), "--output", dst.name, str(src),
    ]
    log(f"START {name}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        try:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\nTIMEOUT after {timeout_s}s\n")
            rc = -9
    dt = (time.time() - t0) / 60
    errs = scan_errors(dst)
    rec = {"notebook": name, "returncode": rc, "minutes": round(dt, 1),
           "executed": str(dst) if dst.exists() else None,
           "log": str(log_path), "errors": [{"cell": c, "name": n, "value": v[:300]} for c, n, v in errs]}
    log(f"DONE  {name} rc={rc} {dt:.1f} min errors={len(errs)}")
    for c, n, v in errs[:5]:
        log(f"  cell {c} {n}: {v[:160]}")
    return rec


def collect():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    copied = 0
    for sub in ("results", "results_corr", "figures"):
        src = PROJECT / sub
        if not src.exists():
            continue
        dst = ARTIFACTS / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*"):
            if f.is_file():
                target = dst / f.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                copied += 1
    # best checkpoints only (resume checkpoints are disposable)
    for sub in ("checkpoints", "checkpoints_corr"):
        src = PROJECT / sub
        if not src.exists():
            continue
        for f in src.rglob("*_best.pt"):
            target = ARTIFACTS / sub / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            copied += 1
    log(f"collected {copied} artefact files into {ARTIFACTS}")


def main(argv):
    names = [a for a in argv if not a.startswith("--")]
    critical = set(DEFAULT_CRITICAL)
    for a in argv:
        if a.startswith("--critical="):
            critical = {x for x in a.split("=", 1)[1].split(",") if x}
    for d in (EXECUTED, LOGS, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)

    results = []
    t_start = time.time()
    try:
        for name in names:
            rec = run_one(name, timeout_s=int(7.5 * 3600))
            results.append(rec)
            if rec["returncode"] != 0 and name in critical:
                log(f"CRITICAL notebook {name} failed - stopping the queue")
                break
    except Exception:  # noqa: BLE001
        log("queue crashed:\n" + traceback.format_exc())
    finally:
        try:
            collect()
        except Exception:  # noqa: BLE001
            log("collect failed:\n" + traceback.format_exc())
        summary = {"started": t_start, "minutes": round((time.time() - t_start) / 60, 1), "runs": results}
        (WORK / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log(f"queue finished in {summary['minutes']} min")
        for r in results:
            log(f"  {r['notebook']}: rc={r['returncode']} {r['minutes']} min errors={len(r['errors'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
