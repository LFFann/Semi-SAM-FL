# A3-PASS-KnowSAM Multiclass 260514

This dated variant adapts `A3_PASS_KnowSAM` to 3-class segmentation:

```text
0 = background
1 = foreground class 1
2 = foreground class 2
```

It preserves the A3-PASS design:

1. **AASP: Acoustic-Anatomical State Posterior**  
   Learns geometry and dense state targets derived automatically from labeled masks.

2. **SCMD: State-Conditioned Mask Decoder**  
   Generates the final segmentation mask from image evidence and the predicted state posterior.

3. **PGUL: Posterior-Guided Unlabeled Learning**  
   Uses state consistency and posterior reliability to gate unlabeled pseudo supervision.

The final task is semi-supervised multiclass segmentation. The prediction
script reports Dice, IoU, and HD95 for each foreground class plus macro-average
metrics over classes 1 and 2.

## Files

```text
variants/A3_PASS_KnowSAM_Multiclass_260514/
  train_semi_SAM_a3_pass.py   # training entry
  trainer_a3_pass.py          # A3-PASS training logic
  prediction_a3_pass.py       # evaluation and visualization
  state_modules.py            # AASP and SCMD modules
  state_targets.py            # mask-derived state targets and perturbations
  train_v100_a3_pass.sh       # V100-32G training launcher
  test_v100_a3_pass.sh        # V100-32G evaluation launcher
  utils/losses_a3_pass.py     # local loss copy
```

## Design Boundary

This implementation uses `Model.model.KnowSAM` as the segmentation backbone and does not put the SAM decoder into the main training loop. This is intentional for the first runnable version: it keeps the method focused on posterior state-space learning rather than SAM prompt engineering. SAM or A3-RCP prompts can be reintroduced later as an auxiliary proposal branch.

## V100-32G Defaults

The launcher uses the current V100 profile:

```text
IMAGE_SIZE=256
BATCH_SIZE=32
LABELED_BS=16
PASS_STATE_SIZE=64
PASS_STATE_DIM=64
PASS_BASE_CHANNELS=32
UNET_LR=0.003
PASS_STATE_LR=0.001
MAX_ITERATIONS=10000
VAL_INTERVAL=50
```

Keep `BATCH_SIZE - LABELED_BS >= LABELED_BS`; the semi-supervised sampler
needs enough unlabeled samples. If CUDA OOM occurs, fall back to
`BATCH_SIZE=24 LABELED_BS=12`. If memory is still underused, try
`BATCH_SIZE=40 LABELED_BS=20`.

## Train

From the repository root:

```bash
bash ./variants/A3_PASS_KnowSAM_Multiclass_260514/train_v100_a3_pass.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=./SampleData \
DATASET=/260513_data_multiclass \
SNAPSHOT_PATH=./Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514 \
BATCH_SIZE=32 \
LABELED_BS=16 \
bash ./variants/A3_PASS_KnowSAM_Multiclass_260514/train_v100_a3_pass.sh
```

Main outputs:

```text
Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514/fold_0/PASS_best_model.pth
Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514/fold_0/SGDL_best_model.pth
Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514/fold_0/log.txt
Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514/fold_0/monitor/
```

## Test

```bash
bash ./variants/A3_PASS_KnowSAM_Multiclass_260514/test_v100_a3_pass.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
SNAPSHOT_PATH=./Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514 \
SPLIT=test \
bash ./variants/A3_PASS_KnowSAM_Multiclass_260514/test_v100_a3_pass.sh
```

The default output is:

```text
Results/A3_PASS_KnowSAM_Multiclass_V100_bs32_10k_106_117_13_13_260514/fold_0/prediction_test/
```

It contains case-level CSV metrics, summary JSON, original images, raw class
label masks, colorized GT/PASS/SGDL masks, and overlays.

Metrics are computed as follows:

```text
class_k Dice/IoU/HD95 = metric(pred == k, gt == k), for k in {1, 2}
avg Dice/IoU/HD95 = macro-average over foreground classes 1 and 2
```

Validation logs include per-class and average metrics for PASS, SGDL, UNet, and
VNet. Test logs include per-case per-class metrics and final per-class summary
metrics for PASS and SGDL.

## Recommended Ablations

Run the full method first, then ablate one mechanism at a time:

```bash
# weaker posterior-guided unlabeled learning
PASS_PSEUDO_WEIGHT=0.0 bash ./variants/A3_PASS_KnowSAM_Multiclass_260514/train_v100_a3_pass.sh

# weaker state supervision
PASS_STATE_WEIGHT=0.0 bash ./variants/A3_PASS_KnowSAM_Multiclass_260514/train_v100_a3_pass.sh

# weaker state consistency
PASS_STATE_CONSISTENCY_WEIGHT=0.0 bash ./variants/A3_PASS_KnowSAM_Multiclass_260514/train_v100_a3_pass.sh
```

For paper tables, compare against `A3_RCP_KnowSAM` with the same data split, image size, label ratio, validation interval, and max iterations.
