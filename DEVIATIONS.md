# DEVIATIONS.md — ambiguities and how each is resolved

One row per item: what the paper says, what we did, why. Rows 1–16 are the
pre-agreed decisions (spec section 9); row 17 was added during execution; the
"RUN STATUS" section at the end is the execution ledger.

| # | Paper statement | What we did | Why |
|---|-----------------|-------------|-----|
| 1 | Text says "357 mineral images"; Fig. 2 counts sum to 367 (60, 62, 60, 66, 60, 59) | Use the real downloaded count (367) and report all three numbers (paper text 357 / paper Fig. 2 sum 367 / real 367) | The real data is verifiable; the text is a typo we must not copy silently |
| 2 | Augmentation yields "four times larger" = 1468 images | Apply the requested larger augmentation set (22 deterministic variants per image) and report the ACTUAL count next to 1468 | User-requested addition (the only intended difference, R5); the number must never be forced |
| 3 | Augment the whole dataset BEFORE imputation/wrangling/split; validation and test therefore contain augmented copies of training photos | Implemented exactly as stated. README warns this lets near-identical copies of one photo cross splits and can inflate test scores. `source_image_id` recorded only so this can be measured later | R2: implement flawed protocols literally; do not fix leakage |
| 4 | "Batch size = 1028 = training set size" (one weight update per epoch) | Gradient accumulation over micro-batches of 32 with ONE optimiser step per epoch (`BATCH_SIZE_MODE="paper_full_batch"`, default). BatchNorm statistics are then per-micro-batch, not per-full-batch | R6: full-batch cannot fit in the GPU memory; accumulation is the equivalent implementation. Configurable to `"minibatch"` (32) if requested |
| 5 | Text: "various epochs from 1 to 60"; all figures stop at 30 | Train and plot 60 epochs | Follow the more specific stated range (R4); plots being shorter is not a statement of the limit |
| 6 | "tested on untrained 18 mineral images, 3 per class"; reported split sizes imply a 220-image test set | Evaluate BOTH: the full 15% test split AND the paper's 18 unseen images (3 per class sampled from the test split with seed 42). Both reported | The paper is silent on where the 18 come from; sampling from the test split is the simplest reading |
| 7 | Stage-1 prose: "kernels 64, 64, 128 bits"; Table 2: [1x1 64, 3x3 64, 1x1 256] | Follow Table 2 | Tables beat prose (R4) |
| 8 | D-ResNet head: FC layer with 1000 neurons; but the task has 6 classes | avgpool -> FC(1000)+ReLU -> FC(6) | The 1000-neuron layer is stated by the paper; the 6-way layer is required by Fig. 1's six classes |
| 9 | Eq. 1 prints the residual block wrongly; Eq. 3 prints precision as TN/(TN+FP) | Standard residual block y = F(x,W)+x (+ReLU); standard precision TP/(TP+FP) | Obvious typographic errors; reproducing them literally would not be replicating the method |
| 10 | Optimiser, learning rate, loss, initialisation, pretrained weights: not stated. Input "224x224x3 normalised"; resize method not stated | Adam, lr 1e-3, weight decay 0, no scheduler, cross-entropy, Kaiming init for convolutions, NO pretrained weights, plain resize to 224x224, **pixels scaled to [0,1] only (no per-channel standardisation)**, mixed precision on GPU, seed 42 | R3: simplest standard option for every silent point; identical across all four models. The paper does not define "normalised", so the minimal reading (scale to [0,1]) is used; the corrected run adds train-fold standardisation — see row 17 |
| 11 | Which epoch's weights are tested: not stated | Best validation accuracy epoch | Simplest selection rule consistent with having a validation split; test never used for selection |
| 12 | Table 4 (VGG16) is incomplete (no filter counts, one FC row, "33 convolutional layers" typo) | Standard VGG16 matching the paper's text: 13 conv (blocks 2,2,3,3,3; 64,128,256,512,512) + 3 FC (4096,4096,6), dropout 0.5, NO batch norm | The text describes standard VGG16; the table is incomplete rather than contradictory |
| 13 | "Data imputation (median)" and "data versioning" applied to image data — the paper's tabular definitions do not map onto image files | nb03: metadata-level median imputation (missing/null + MAD-z outliers in width/height/channels/mean intensity) AND pixel-level impulse-outlier replacement by the 3x3 local median; versioning = uint8 (N,224,224,3) arrays + labels + manifests + checksums, tags v1-v4, never overwritten | Literal implementation of the paper's wording at both levels, documented in nb03 |
| 14 | "Model re-engineering" (Fig. 5 loop) and "deployment with the support of cross-validation" are described but not specified | Re-engineering = if train_acc − best_val_acc > 0.15 (measured at the best-validation epoch), run ONE extra round from a fresh model with weight decay 1e-4 and lr 5e-4 (MAX_REENGINEERING_ROUNDS=1); the round with the higher validation accuracy is the one evaluated on the test split. Implemented in every training notebook (05–08 and 05_corr–08_corr) as an explicit, printed decision, with the gap and both rounds recorded in the metrics JSON (`"re_engineering"`, `"selected_round"`). 5-fold stratified CV cell exists but is guarded by RUN_CROSS_VALIDATION=False | Minimal faithful reading; both configurable and logged per round |
| 15 | Table 10 / Fig. 15: RMSE of D-ResNet vs SVM, CART, KNN, RF, SGB, Cubist | NOT reproduced | The numbers come from a different task (ilmenite-content prediction, Lv et al. [19]); the paper never defines an RMSE for D-ResNet, so there is nothing to replicate |
| 16 | Paper reports accuracy/precision/recall only in Table 9; abstract mentions specificity | Per-class specificity = TN/(TN+FP) computed and reported as an extra column beyond the paper's layout | Spec section 8 requires it; it extends (not contradicts) the paper's tables |
| 17 | "224x224x3 normalised" is undefined in the paper, and the two pipelines normalise differently | **Faithful run**: uint8/float32 arrays scaled to [0,1] and fed to the model as-is (no per-channel standardisation) — verified on the executed runs: `versions/X_train.npy` is float32 in [0,1], mean 0.537. **Corrected run**: uint8 → [0,1] → per-channel standardisation with the corrected train-fold mean/std (0.0568 / 0.2383) | Both conventions are applied consistently *within* their own pipeline and identically to all four models. The corrected train fold is a different set of images, so its statistics must be recomputed. Changing the faithful convention now would invalidate the already-measured faithful results, so the difference is documented rather than retro-fitted |
| 18 | Execution environment: the paper uses "NVIDIA V100, 32 GB RAM, 12th Gen Intel i5" (Section 4); the replication ran on Colab, whose GPU quota is now exhausted | Remaining runs (faithful 05–09, corrected 06–09) execute on **Kaggle** (P100 16 GB requested), driven by `kaggle/build_bundle.py` + `kaggle/src/bsm_env.py`. The shim recreates the Colab folder layout (`/content/drive/MyDrive/bsm_paper_replication`, `/content/BSM_Project_2`) inside the Kaggle container, so the notebook code itself is unchanged. Only three mechanical edits are applied: Drive mount removed, bootstrap cell prepended, and nb01's dataset-acquisition cell reads the attached Kaggle dataset instead of `kagglehub` | Hardware/environment swap only — no change to data, augmentation, splits, seeds, hyper-parameters, epoch budget, checkpoint rule, or metrics. Different GPU ⇒ per-epoch wall-clock changes (recorded per epoch in `epoch_time_s`), which does not affect any reported metric |
| 19 | Notebook housekeeping for the Kaggle runs | Stored cell outputs are stripped from the Kaggle variants, and each cell receives an `id` | Kaggle's nbformat validator rejects the Colab notebooks before execution (`'metadata' is a required property` on an old `execute_result`). The executed notebooks produced by the run carry fresh outputs; no analysis code or output used in the report is removed |

## Standing notes

- **Softmax**: the paper's tables end in softmax; our models return logits and
  the cross-entropy loss applies softmax internally (numerically identical).
- **FLOPs convention**: the paper's FLOPs column (1.8e9 / 3.6e9 / 3.8e9) is
  compared against our multiply-accumulate counts; we report both, never
  silently converting.
- **Paper numbers are always labelled "paper"** and never mixed into our
  result tables as ours (R7).

## CORRECTED RUN (user-requested, non-paper-faithful for better accuracy)

| Change | Rationale | Notes |
|---|---|---|
| Split by original image identity (group-stratified) | Prevents leakage between train/val/test | All augmented variants from the same source image stay in the same split. |
| Split-before-augment | Avoids augment-before-split leakage | Only original images in the training fold are augmented. Val/test use original images only. |
| Include original unaugmented images in training | Adds real data the paper excluded | Train = original_train + augmented_from_original_train. |
| Augmentation applied only to training fold | Standard correct practice | Reduces overfitting to augmented artifacts. |
| Separate notebooks/metrics labeled `CORRECTED` | Keep faithful runs untouched | 05_corr_*, 06_corr_*, 07_corr_*, 08_corr_*, 09_corr_comparison. All metrics saved as `metrics_*_corrected.json`. |

### CORRECTED RUN — finalised parameter values (all logged, none silent)

| Item | Value |
|---|---|
| Source of the 367 originals | KaggleHub `prasannavenkatesant/beach-sand-mineral-bsm` (public dataset, downloaded anonymously — no credentials are stored in code, notebooks or outputs). On Colab the download happens at run time; `data/` is git-ignored. |
| Group id | `class__filename_stem` — every original file is its own group (367 unique groups; md5 check found 0 byte-identical images, so no hidden duplicates). |
| Split algorithm | `StratifiedShuffleSplit` (sklearn) on the class label: 70 % train / 15 % val / 15 % test, seed 42 → **256 / 55 / 56** originals. Class balance: train 43/46/41/42/42/42, val 9/10/9/9/9/9, test 10/10/9/9/9/9 (Ilmenite, Rutile, Zircon, Garnet, Monazite, Sillimanite). Group-disjointness is asserted in the notebook. |
| Augmentation | Same 22-variant recipe as the faithful nb02, applied **only** to the 256 train originals (5 632 variants). |
| Training set | 256 originals + 5 632 augmented variants = **5 888** images (uint8, 224x224x3, ~887 MB). |
| Normalisation | Pixels to [0,1] then per-channel standardisation with the **corrected train-fold** mean/std (0.0568 / 0.2383; identical for R, G, B because the source images are effectively grayscale). This is **one step more** than the faithful run, which scales to [0,1] only (DEVIATIONS rows 10 and 17). |
| Optimisation | Unchanged from the faithful run: 60 epochs, full-batch gradient accumulation (micro-batch 32, one optimiser step per epoch), Adam lr 1e-3, wd 0, no scheduler, best-val checkpoint. |
| Evaluation | **Original test images only** (56). The paper's "18 unseen images" subset is not reproduced in the corrected run (that subset belongs to the faithful protocol). |
| Reported metrics | Accuracy, macro precision/recall/F1, macro specificity, confusion matrix, best epoch, n_test → `results_corr/metrics_*_corrected.json`. |

## RUN STATUS (2026-10-22)

The Colab GPU quota for this account was exhausted on 2026-10-22, which stopped
the run queue. This is the ledger of what ran, what was retained, what was lost
and what is still pending. Nothing below is estimated — every number comes from
a metrics/history JSON that is committed in this repo.

### Completed and retained

| Run | Protocol | Test accuracy | Artefacts in this repo |
|---|---|---|---|
| `05_dresnet50` | paper-faithful | 0.7228 (1 212 test images) | `notebooks/05_dresnet50.ipynb`, `results/metrics_dresnet50.json`, `results/history_dresnet50.json` |
| `05_corr_dresnet50` | corrected | 0.5179 (29/56) | `executed/05_corr_dresnet50.ipynb`, `results_corr/metrics_dresnet50_corrected.json`, `results_corr/history_dresnet50_corrected.json` |
| `06_corr_resnet18` | corrected | 0.5179 (29/56) | `executed/06_corr_resnet18.ipynb`, `results_corr/metrics_resnet18_corrected.json`, `results_corr/history_resnet18_corrected.json` |
| `07_corr_resnet34` | corrected | 0.4464 (25/56) | `executed/07_corr_resnet34.ipynb`, `results_corr/metrics_resnet34_corrected.json`, `results_corr/history_resnet34_corrected.json` |

### Interrupted or lost

| Run | What happened |
|---|---|
| `08_corr_vgg16` | Stopped at epoch 38/60 when the Colab runtime was reset. The history is retained (`results_corr/history_vgg16_corrected.json`: best val 0.2364 at epoch 26, train acc 0.1703 at epoch 38) and the run resumes from the epoch-30 resume checkpoint, so at most epochs 31–38 are redone. No metrics file exists yet. |
| `06_resnet18`, `07_resnet34` (paper-faithful) | Measured earlier on 2026-10-21 (test accuracy 0.5817 and 0.7698) but their metrics JSONs and checkpoints were never written to the shared Drive folder and did not survive the runtime reset. The numbers are therefore **not** recorded as results here; both runs are pending again. |
| `05_corr_dresnet50` metrics | The executed notebook was re-run to refresh it after the staging-file fix; that re-run resumed in 46 s and overwrote `train_minutes` with the resumed wall clock (0.07 min). The committed metrics JSON was restored from a backup and reports the real first-run time (27.98 min). The history's `epoch_time_s` sum (27.77 min) agrees, so the notebook now derives `train_minutes` from that sum instead of the wall clock. |

### Pending — moved to Kaggle (2026-10-23)

The Colab GPU quota is exhausted, so the queue below runs on Kaggle instead
(deviation 18). Nothing about the protocol changes; only the execution
environment and file paths.

**Faithful kernel** (`kiramada/bsm-faithful-remaining`, ~2 h 05 on a P100):
`01_eda` → `02_data_augmentation` → `03_imputation_wrangling_versioning` →
`04_split_70_15_15` → `05_dresnet50` → `06_resnet18` → `07_resnet34` →
`08_vgg16` → `09_comparison`.

- `01`/`02`/`03`/`04` are re-run because the augmented PNGs and the
  `versions/*.npy` arrays were never committed (too large) and the inventory CSV
  must point at real Kaggle paths. The augmentation table is deterministic, so
  the arrays are reproducible; the split is `StratifiedShuffleSplit(seed=42)`.
- `05_dresnet50` is re-run so `09_comparison` has a checkpoint for every Table 9
  row; the earlier Colab measurement (0.7228) stays in the ledger above, and the
  Kaggle re-run is reported alongside it rather than replacing it silently.

**Corrected kernel** (`kiramada/bsm-corrected-remaining`, ~1 h 30):
`06_corr_resnet18` → `07_corr_resnet34` → `08_corr_vgg16` → `09_corr_comparison`.

1. `08_vgg16` (paper-faithful VGG16) — built and syntax-checked, never executed.
2. `06_resnet18`, `07_resnet34` (paper-faithful) — re-run to regenerate the
   metrics JSONs and checkpoints.
3. `09_comparison` — Table 9 / Fig. 9 comparison plus the paper's 18-image
   subset evaluation; needs the four paper-faithful checkpoints.
4. `08_corr_vgg16` — the epoch-30 resume checkpoint did **not** survive the Colab
   runtime reset (only the history did), so round 0 restarts from epoch 1.
5. `06_corr_resnet18`, `07_corr_resnet34` — re-engineering round 1 (see below).
6. `09_corr_comparison` — corrected comparison table over the four corrected runs.

### Re-engineering decisions measured so far (trigger: gap = train_acc(best-val epoch) − best_val_acc > 0.15)

| Run | Gap | Decision |
|---|---|---|
| `05_dresnet50` (paper-faithful) | +0.0031 | no extra round |
| `05_corr_dresnet50` | +0.0152 | no extra round |
| `06_corr_resnet18` | +0.2507 | **extra round required** (lr 5e-4, wd 1e-4) — pending |
| `07_corr_resnet34` | +0.2270 | **extra round required** (lr 5e-4, wd 1e-4) — pending |
| `08_corr_vgg16` (epoch 38) | −0.0480 (still underfitting) | no extra round |

The paper-faithful ResNet-18/34 gaps are unknown until those runs are repeated;
their notebooks now print and log the decision automatically.

