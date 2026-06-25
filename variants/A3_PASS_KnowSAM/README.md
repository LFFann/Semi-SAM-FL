# A3-PASS-KnowSAM

This variant implements the compact three-innovation A3-PASS design:

1. **AASP: Acoustic-Anatomical State Posterior**  
   Learns geometry and dense state targets derived automatically from labeled masks.

2. **SCMD: State-Conditioned Mask Decoder**  
   Generates the final segmentation mask from image evidence and the predicted state posterior.

3. **PGUL: Posterior-Guided Unlabeled Learning**  
   Uses state consistency and posterior reliability to gate unlabeled pseudo supervision.

The final task is still semi-supervised segmentation. The prediction script reports the standard metrics used by this repo: Dice, IoU, and 95HD.

## Files

```text
variants/A3_PASS_KnowSAM/
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

The launcher uses conservative defaults for a V100-32G GPU:

```text
IMAGE_SIZE=256
BATCH_SIZE=16
LABELED_BS=8
PASS_STATE_SIZE=64
PASS_STATE_DIM=64
PASS_BASE_CHANNELS=32
UNET_LR=0.0025
PASS_STATE_LR=0.001
MAX_ITERATIONS=50000
VAL_INTERVAL=200
```

If memory is still tight, first reduce `BATCH_SIZE=12` and `LABELED_BS=6`. If memory is clearly underused, try `BATCH_SIZE=20` and `LABELED_BS=10`.

## Train

From the repository root:

```bash
bash ./variants/A3_PASS_KnowSAM/train_v100_a3_pass.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=./SampleData \
DATASET=/260513_data_label1 \
SNAPSHOT_PATH=./Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13 \
BATCH_SIZE=16 \
LABELED_BS=8 \
bash ./variants/A3_PASS_KnowSAM/train_v100_a3_pass.sh
```

Main outputs:

```text
Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13/fold_0/PASS_best_model.pth
Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13/fold_0/SGDL_best_model.pth
Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13/fold_0/log.txt
Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13/fold_0/monitor/
```

## Test

```bash
bash ./variants/A3_PASS_KnowSAM/test_v100_a3_pass.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
SNAPSHOT_PATH=./Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13 \
SPLIT=test \
bash ./variants/A3_PASS_KnowSAM/test_v100_a3_pass.sh
```

The default output is:

```text
Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13/fold_0/prediction_test/
```

It contains case-level CSV metrics, summary JSON, original images, GT masks, predicted PASS masks, SGDL masks, and overlays.

## Recommended Ablations

Run the full method first, then ablate one mechanism at a time:

```bash
# weaker posterior-guided unlabeled learning
PASS_PSEUDO_WEIGHT=0.0 bash ./variants/A3_PASS_KnowSAM/train_v100_a3_pass.sh

# weaker state supervision
PASS_STATE_WEIGHT=0.0 bash ./variants/A3_PASS_KnowSAM/train_v100_a3_pass.sh

# weaker state consistency
PASS_STATE_CONSISTENCY_WEIGHT=0.0 bash ./variants/A3_PASS_KnowSAM/train_v100_a3_pass.sh
```

For paper tables, compare against `A3_RCP_KnowSAM` with the same data split, image size, label ratio, validation interval, and max iterations.
