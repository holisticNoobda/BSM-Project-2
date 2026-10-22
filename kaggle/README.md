# Kaggle execution bundle

The paper notebooks were written for Colab. Colab's GPU quota is exhausted, so the
remaining runs execute on Kaggle instead. **Nothing about the pipeline changes** —
same images, same augmentation table, same splits, same seeds, same protocol. Only
filesystem paths are adapted, because Kaggle mounts inputs read-only and persists
only `/kaggle/working`.

## What is in here

```
kaggle/
  src/bsm_env.py        shim: rebuilds /content/drive/MyDrive/bsm_paper_replication
                        and /content/BSM_Project_2 from the attached dataset
  src/run_queue.py      driver: executes notebooks in order with nbconvert,
                        writes per-notebook logs, collects artefacts
  build_bundle.py       regenerates dataset/ + kernels/ from notebooks/ (idempotent)
  run_kaggle.py         local orchestrator: dataset create/version, kernel push,
                        status, output fetch
  dataset/              GENERATED - uploaded as the private dataset
                        kiramada/bsm-paper-2-kaggle
  kernels/              GENERATED - three kernels
  output/               downloaded kernel outputs (artefacts, executed notebooks)
```

## How the shim works

`bsm_env.setup()` runs as the first cell of every Kaggle notebook and:

1. resolves the dataset mount (Kaggle currently uses
   `/kaggle/input/datasets/<owner>/<slug>`, older layouts put it directly under
   `/kaggle/input/`);
2. copies `repo/` to `/content/BSM_Project_2` and `code/` (models.py, common.py)
   plus `seed/` to `/content/drive/MyDrive/bsm_paper_replication`;
3. creates the project sub-folders the notebooks expect;
4. prints diagnostics (image counts, torch/CUDA, free disk).

Because `/content` is writable in a Kaggle container (root), the notebooks run
**unmodified** apart from three mechanical edits, all applied by `build_bundle.py`:

| edit | reason |
| --- | --- |
| Colab `drive.mount` block removed | no Drive on Kaggle; `bsm_env` pre-creates the tree |
| bootstrap cell prepended | imports `bsm_env`, exposes `BSM_RAW_SRC` |
| nb01 acquisition cell rewritten | reads the attached dataset instead of `kagglehub`; module download falls back to the local copy |

Stored cell outputs are also stripped: the Colab copies carry outputs that Kaggle's
stricter nbformat validator rejects before execution. The executed notebooks
produced by the run carry the fresh outputs.

## Kernels

| kernel | notebooks | est. time |
| --- | --- | --- |
| `bsm-smoke-check` | 01_eda | ~2 min |
| `bsm-faithful-remaining` | 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 | ~2 h 05 |
| `bsm-corrected-remaining` | 06_corr → 07_corr → 08_corr → 09_corr | ~1 h 30 |

`01_eda` is re-run so the inventory CSV points at real Kaggle paths; `05_dresnet50`
is re-run so `09_comparison` has a checkpoint for every Table 9 row. Prep notebooks
are marked *critical*: if one fails the queue stops instead of training on nothing.

## Commands

```powershell
# requires the new Kaggle auth (legacy kaggle.json keys are rejected)
kaggle auth login

python kaggle\build_bundle.py          # regenerate dataset/ + kernels/
python kaggle\run_kaggle.py dataset    # create or version the dataset
python kaggle\run_kaggle.py push smoke # or: faithful / corrected / all
python kaggle\run_kaggle.py status
python kaggle\run_kaggle.py fetch all  # download artefacts into kaggle/output/
python kaggle\run_kaggle.py logs kiramada/bsm-smoke-check
```

`push` always passes `--accelerator NvidiaTeslaP100`.

## Requirements on the Kaggle side

- account phone-verified (**GPU is not granted otherwise** — a kernel pushed with a
  GPU shape then silently runs on CPU, `torch.cuda.is_available()` is `False`);
- GPU allowance: 6 h/week (`quota_view()` reports 21600 s), so the two queues use
  ~60 % of one week's GPU budget;
- kernels run with internet enabled (nb01 re-downloads models.py/common.py with a
  SHA-256 print; it falls back to the bundled copy if GitHub is unreachable).

## Outputs

`/kaggle/working` after a run:

```
executed/    <notebook>_executed.ipynb  (with fresh outputs)
logs/        <notebook>.log             (nbconvert stdout/stderr)
artifacts/   results/, results_corr/, figures/, checkpoints/*_best.pt
run_summary.json                        (per-notebook rc, minutes, error list)
```

Fetch with `python kaggle\run_kaggle.py fetch <name>`; the artefacts land in
`kaggle/output/<name>/artifacts/` and are then copied into the repo's `results/`,
`results_corr/` and `executed/` folders.
