# DEVIATIONS.md — ambiguities and how each is resolved

One row per item: what the paper says, what we did, why. Rows 1–16 are the
pre-agreed decisions (spec section 9); further rows are appended as new
ambiguities are met during execution.

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
| 10 | Optimiser, learning rate, loss, initialisation, pretrained weights: not stated. Input "224x224x3 normalised"; resize method not stated | Adam, lr 1e-3, weight decay 0, no scheduler, cross-entropy, Kaiming init for convolutions, NO pretrained weights, plain resize to 224x224, pixels to [0,1] then per-channel standardisation with training-split mean/std, mixed precision on GPU, seed 42 | R3: simplest standard option for every silent point; identical across all four models |
| 11 | Which epoch's weights are tested: not stated | Best validation accuracy epoch | Simplest selection rule consistent with having a validation split; test never used for selection |
| 12 | Table 4 (VGG16) is incomplete (no filter counts, one FC row, "33 convolutional layers" typo) | Standard VGG16 matching the paper's text: 13 conv (blocks 2,2,3,3,3; 64,128,256,512,512) + 3 FC (4096,4096,6), dropout 0.5, NO batch norm | The text describes standard VGG16; the table is incomplete rather than contradictory |
| 13 | "Data imputation (median)" and "data versioning" applied to image data — the paper's tabular definitions do not map onto image files | nb03: metadata-level median imputation (missing/null + MAD-z outliers in width/height/channels/mean intensity) AND pixel-level impulse-outlier replacement by the 3x3 local median; versioning = uint8 (N,224,224,3) arrays + labels + manifests + checksums, tags v1-v4, never overwritten | Literal implementation of the paper's wording at both levels, documented in nb03 |
| 14 | "Model re-engineering" (Fig. 5 loop) and "deployment with the support of cross-validation" are described but not specified | Re-engineering = if train_acc − best_val_acc > 0.15 after round 0, run ONE round with weight decay 1e-4 and lr 5e-4 (MAX_REENGINEERING_ROUNDS=1). 5-fold stratified CV cell exists but is guarded by RUN_CROSS_VALIDATION=False | Minimal faithful reading; both configurable and logged per round |
| 15 | Table 10 / Fig. 15: RMSE of D-ResNet vs SVM, CART, KNN, RF, SGB, Cubist | NOT reproduced | The numbers come from a different task (ilmenite-content prediction, Lv et al. [19]); the paper never defines an RMSE for D-ResNet, so there is nothing to replicate |
| 16 | Paper reports accuracy/precision/recall only in Table 9; abstract mentions specificity | Per-class specificity = TN/(TN+FP) computed and reported as an extra column beyond the paper's layout | Spec section 8 requires it; it extends (not contradicts) the paper's tables |

## Standing notes

- **Softmax**: the paper's tables end in softmax; our models return logits and
  the cross-entropy loss applies softmax internally (numerically identical).
- **FLOPs convention**: the paper's FLOPs column (1.8e9 / 3.6e9 / 3.8e9) is
  compared against our multiply-accumulate counts; we report both, never
  silently converting.
- **Paper numbers are always labelled "paper"** and never mixed into our
  result tables as ours (R7).
