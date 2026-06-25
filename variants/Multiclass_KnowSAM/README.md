# SRG-SAM Multiclass Training

This copied variant now trains SRG-SAM, not the original KnowSAM pseudo-label
or SAM-distillation path.

SRG-SAM uses:

```text
SAM -> frozen structure reliability estimator only
SSRF -> R_b, R_a, R_u from prompt-response stability
SRPC -> R_b * (1 - R_u) weighted posterior pseudo-label learning
DPG -> dynamic class prototype graph from features and R_a
```

It explicitly does not use SAM masks as pseudo labels, confidence-threshold
pseudo-label filtering, or pure teacher-student consistency loss.

## Dataset Layout

The loader supports BUSI-style or local ultrasound splits with this structure:

```text
DATA_PATH/DATASET/
  labeled/image/*.png
  labeled/mask/*.png
  unlabeled/image/*.png
  val/image/*.png
  val/mask/*.png
```

Masks may be binary or integer multiclass labels. Set `--num_classes` to `2`
for binary segmentation or `3+` for multiclass segmentation.

## Train

From the SRG-SAM root:

```bash
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=./SampleData \
DATASET=/260513_data_multiclass \
BATCH_SIZE=24 \
LABELED_BS=12 \
SRG_PROMPT_COUNT=4 \
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
```

For BUSI, point `DATA_PATH` and `DATASET` to a folder with the same
labeled/unlabeled/val layout:

```bash
DATA_PATH=./SampleData DATASET=/BUSI_semi NUM_CLASSES=2 \
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
```

The required runtime diagnostics are printed every iteration:

```text
R_b mean:
R_u mean:
pseudo weight mean:
graph adjacency norm:
```

