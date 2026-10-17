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
2. Colab Secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY` (notebook access ON).
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
  Ilmenite 60, Rutile 62, Zircon 60, Garnet 66, Monazite 60, Sillimanite 59.
- The paper's text says **357** while its own Fig. 2 sums to **367** —
  both numbers are reported next to the real count (DEVIATIONS row 1).

## Paper results for comparison (Table 9 — PAPER numbers, never ours)

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| VGG16 | 0.83 | 0.89 | 0.83 |
| ResNet-18 | 0.78 | 0.87 | 0.78 |
| ResNet-34 | 0.83 | 0.89 | 0.83 |
| D-ResNet | 0.89 | 0.90 | 0.89 |

## Not reproduced

Paper **Table 10 / Fig. 15** (RMSE of D-ResNet vs SVM, CART, KNN, RF, SGB,
Cubist) — those numbers come from a different task (ilmenite-content
regression, Lv et al. [19]) and the paper never defines an RMSE for D-ResNet.
See DEVIATIONS row 15.

## Repo contents

Notebooks, shared modules, docs. No credentials, no datasets, no checkpoints
(gitignored).
