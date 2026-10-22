# BSM Project 2 — paper-faithful D-ResNet replication

Faithful, step-by-step replication of

> **"D-ResNet: deep residual neural network for exploration, identification, and
> classification of beach sand minerals"** — Theerthagiri et al., *Multimedia
> Tools and Applications* (2024) 83:14539–14563.

as Google Colab notebooks, executed on a Colab GPU. Every methodological choice
follows the paper **literally** — including the ones that are suboptimal
(augment-before-split, augmented test data, full-batch training). Deviations
exist only where the paper is silent or states something impossible on the
hardware; each one is logged in [DEVIATIONS.md](DEVIATIONS.md).

> This project deliberately does **not** fix the leakage of the paper's
> protocol. Scores are therefore not directly comparable to the earlier
> corrected-protocol study (`BSM-Project` repo).

## Pipeline (paper Fig. 1 / Fig. 5, executed in this order)

| # | Notebook | Paper stage |
|---|----------|-------------|
| 01 | `01_eda.ipynb` | Data pre-processing — EDA; writes `common.py` + `models.py` to Drive |
| 02 | `02_data_augmentation.ipynb` | Data augmentation (12 paper variants + 10 requested variants, deterministic, whole dataset BEFORE the split) |
| 03 | `03_imputation_wrangling_versioning.ipynb` | Median imputation, wrangling, versioning (v1/v2/v3) |
| 04 | `04_split_70_15_15.ipynb` | 70/15/15 train/test/validation split + normalisation + paper's 18-image set |
| 05 | `05_dresnet50.ipynb` | D-ResNet-50 (Table 2) training + evaluation |
| 06 | `06_resnet18.ipynb` | ResNet-18 (Table 5) |
| 07 | `07_resnet34.ipynb` | ResNet-34 (Table 5) |
| 08 | `08_vgg16.ipynb` | VGG16 (Table 4) |
| 09 | `09_comparison.ipynb` | Table 9 / Fig. 9 / Figs. 13-14 comparison (reads results, never retrains) |

Notebooks 05–08 contain a `SMOKE_TEST` flag (60 images, 1 epoch, all four
graph types) that must pass before any full 60-epoch run.

## How to run

1. Open a blank Colab notebook (GPU runtime), keep it open.
2. Colab Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY` (notebook access ON) — needed
   only by notebook 01, which reads them with `google.colab.userdata` and
   deletes them from the environment right after the download. The dataset is
   public, so `kagglehub` also works anonymously (that is the credential-free
   path the `CORRECTED` notebooks use).
3. Run the notebooks **in numeric order**. Each starts with the same setup
   cell (mount Drive, `sys.path` += the project folder on Drive, seed 42,
   print versions + GPU).
4. All persistent data lives under
   `/content/drive/MyDrive/bsm_paper_replication/` (augmented PNGs, arrays,
   checkpoints, results, figures).
5. Training is resumable: history JSON every epoch, resume checkpoint every
   10 epochs; completed steps are skipped on re-run.

## Shared modules

- `models.py` — D-ResNet-50, ResNet-18/34, VGG16 built **layer-by-layer from
  the paper's tables** (no torchvision, no pretrained weights) + the
  architecture verification (stage sizes 112/56/28/14/7, exact reference
  parameter counts, MACs vs the paper's FLOPs column).
- `common.py` — data (manifest + array datasets, normalisation), training
  (full-batch via gradient accumulation, best-validation checkpointing,
  history/resume), metrics (per-class precision/recall/F1/specificity,
  weighted average, overall accuracy), paper-styled plots.

Notebook 01 writes both modules to the Drive project folder with
`%%writefile`; the copies in this repo are the canonical source.

## Key dataset facts (verified from the real Kaggle download)

- 367 images in 6 class folders. Class order used everywhere (paper Fig. 2):
  Ilmenite, Rutile, Zircon, Garnet, Monazite, Sillimanite.
- The paper's text says **357** while its own Fig. 2 sums to **367** — both
  numbers are reported next to the real count (DEVIATIONS row 1).
- Real downloaded counts (verified, `01_eda.ipynb` inventory):
  Ilmenite 62, Rutile 66, Zircon 59, Garnet 60, Monazite 60, Sillimanite 60.
  Paper Fig. 2 order is Ilmenite 60, Rutile 62, Zircon 60, Garnet 66,
  Monazite 60, Sillimanite 59 — the two lists differ per class, the totals agree.

## Preprocessing conventions (both pipelines)

- Every image is resized to 224x224x3 and **scaled to [0,1]**. The faithful run
  feeds those pixels to the model unchanged (DEVIATIONS row 10 — the paper only
  says "224x224x3 normalised" and never defines it).
- The **corrected** run takes one extra step: per-channel standardisation with
  the corrected train-fold mean/std (DEVIATIONS row 17).

## Paper results for comparison (Table 9 — PAPER numbers, never ours)

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| VGG16 | 0.83 | 0.89 | 0.83 |
| ResNet-18 | 0.78 | 0.87 | 0.78 |
| ResNet-34 | 0.83 | 0.89 | 0.83 |
| D-ResNet | 0.89 | 0.90 | 0.89 |

## Measured results (MEASURED by us — never paper numbers)

### Paper-faithful protocol (test split contains augmented copies of training photos — 1 212 test images)

| Model | Notebook | Test accuracy | Macro precision | Macro recall | Macro F1 | Macro specificity | Best epoch | Train min |
|---|---|---|---|---|---|---|---|---|
| D-ResNet-50 | `05_dresnet50` | **0.7228** | 0.7411 | 0.7244 | 0.7206 | 0.9446 | 59 | 24.3 |

Artefacts: `notebooks/05_dresnet50.ipynb` (executed, with outputs),
`results/metrics_dresnet50.json`, `results/history_dresnet50.json`.

ResNet-18 and ResNet-34 were measured on this protocol earlier
(test accuracy 0.5817 and 0.7698) but their metrics files and checkpoints were
never written to the shared Drive folder and were lost when the Colab runtime
was reset — those numbers are **not retained** here and the runs are pending
again. VGG16 (`08_vgg16.ipynb`) is built and syntax-checked but has never been
executed.

### Corrected protocol (group split, original test images only — 56 images)

| Model | Notebook | Test accuracy | Macro precision | Macro recall | Macro F1 | Macro specificity | Best epoch | Train min |
|---|---|---|---|---|---|---|---|---|
| D-ResNet-50 | `05_corr_dresnet50` | **0.5179** (29/56) | 0.5752 | 0.5148 | 0.4974 | 0.9039 | 53 | 27.8 |
| ResNet-18 | `06_corr_resnet18` | **0.5179** (29/56) | 0.5369 | 0.5167 | 0.4970 | 0.9037 | 59 | 13.7 |
| ResNet-34 | `07_corr_resnet34` | **0.4464** (25/56) | 0.4722 | 0.4407 | 0.4249 | 0.8889 | 60 | 19.6 |

Artefacts: `executed/05_corr_dresnet50.ipynb`, `executed/06_corr_resnet18.ipynb`,
`executed/07_corr_resnet34.ipynb` (executed copies with outputs),
`results_corr/metrics_*_corrected.json`, `results_corr/history_*_corrected.json`.

**How to read the two tables.** The drop from 0.7228 (faithful) to 0.5179
(corrected) for D-ResNet-50 is **not** a regression of the model — it is what
the paper's leaky protocol (augmented copies of training photographs sitting in
the test split) was worth. On the corrected 56-image test set a single image is
1.79 % of the score, so those figures are noisy. ResNet-18 and ResNet-34 on the
corrected protocol still owe their re-engineering round (measured train−val gap
+0.2507 and +0.2270, both above the 0.15 trigger — see
[DEVIATIONS.md](DEVIATIONS.md) "RUN STATUS").

## CORRECTED RUN (separate, non-paper-faithful variant)

A second, **user-requested** pipeline that removes the leakage of the paper's
protocol while keeping the optimisation protocol identical. It lives in its own
notebooks so the faithful ones stay untouched.

| # | Notebook | What it does |
|---|----------|--------------|
| 05 | `05_corr_dresnet50.ipynb` | D-ResNet-50, corrected split |
| 06 | `06_corr_resnet18.ipynb` | ResNet-18, corrected split |
| 07 | `07_corr_resnet34.ipynb` | ResNet-34, corrected split |
| 08 | `08_corr_vgg16.ipynb` | VGG16, corrected split |
| 09 | `09_corr_comparison.ipynb` | comparison table of the four corrected runs |

Protocol: split by original-image identity **before** augmenting
(stratified 70/15/15 → 256 train / 55 val / 56 test originals); the 22
augmented variants are produced **only** from the train originals; the
training set is `256 originals + 5 632 augmented = 5 888` images; validation
and test are **original images only**. Training is unchanged (60 epochs,
full-batch gradient accumulation with micro-batch 32, Adam 1e-3, best-val
checkpoint). Results are written to
`results_corr/metrics_*_corrected.json` and always carry `"type": "CORRECTED"`.
All parameter values are listed in DEVIATIONS.md ("CORRECTED RUN" section).

## Not reproduced

Paper **Table 10 / Fig. 15** (RMSE of D-ResNet vs SVM, CART, KNN, RF, SGB,
Cubist) — those numbers come from a different task (ilmenite-content
regression, Lv et al. [19]) and the paper never defines an RMSE for D-ResNet.
See DEVIATIONS row 15.

## Run status (2026-10-23)

Colab's GPU quota for this account ran out on 2026-10-22, so the remaining runs
were moved to **Kaggle** (deviation 18). The pipeline is unchanged — same images,
augmentation table, splits, seeds, epochs, checkpoint rule; a shim
(`kaggle/src/bsm_env.py`) recreates the Colab folder layout inside the Kaggle
container so the notebook code itself runs untouched. See [kaggle/README.md](kaggle/README.md).

| Kernel | Queue | Estimated |
|---|---|---|
| `kiramada/bsm-faithful-remaining` | 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 | ~2 h 05 |
| `kiramada/bsm-corrected-remaining` | 06_corr → 07_corr → 08_corr → 09_corr | ~1 h 30 |

Estimates use the measured per-epoch times from the committed history JSONs
(R18 13.6 s, R34 19.5 s, D-R50 24.3 s, VGG16 51.3 s) plus ~12 min of data
preparation; run in parallel the wall clock is ~2 h 15. Kaggle gives 6 GPU-hours
per week (`quota_view()` → 21 600 s), so the queue uses ~60 % of one week.

Everything that was measured on Colab and retained is listed above; the full
ledger of what ran, what was lost and what is pending (with reasons) is in
[DEVIATIONS.md](DEVIATIONS.md) → "RUN STATUS".

Every training notebook (05–08 and 05_corr–08_corr) now contains the
re-engineering decision as an explicit, printed step, so the pending re-runs
will log it themselves.

## Repo contents

Notebooks, shared modules, docs. No credentials, no datasets, no checkpoints
(gitignored).

| Path | What |
|---|---|
| `notebooks/` | the nine paper-faithful notebooks (01–09) + the five `CORRECTED` ones; 01–07 carry their executed outputs |
| `executed/` | executed copies of the corrected notebooks (with outputs) |
| `results/` | paper-faithful metrics + training history JSON (and the EDA CSVs) |
| `results_corr/` | corrected metrics + training history JSON (`"type": "CORRECTED"`) |
| `models.py`, `common.py` | shared architectures / training + evaluation code |
| `DEVIATIONS.md` | every ambiguity, what was done, and why — plus the run ledger |
