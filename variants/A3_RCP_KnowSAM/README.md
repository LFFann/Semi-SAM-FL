# A3-RCP-KnowSAM

This directory keeps the A3-RCP experimental variant separate from the original
KnowSAM baseline. The root-level baseline files remain available for direct
comparison, while this variant owns its training entry, trainer, prediction
script, and local loss definitions.

## Files

```text
variants/A3_RCP_KnowSAM/
  train_semi_SAM_a3_rcp.py   # training entry
  trainer_a3_rcp.py          # A3-RCP training logic
  prediction_a3_rcp.py       # evaluation and visualization
  train_v100_a3_rcp.sh       # V100-32G training launcher
  test_v100_a3_rcp.sh        # V100-32G evaluation launcher
  utils/losses_a3_rcp.py     # local loss copy
  README.md
```

## Relationship to the Baseline

This variant reuses shared repository modules such as:

- `Model/`
- `dataloader/`
- `utils/utils.py`
- `utils/training_monitor.py`

It isolates the method-specific training loop and loss design so that the
original KnowSAM implementation remains intact.

## Method Components

A3-RCP is used to test three semi-supervised segmentation ideas:

- Reliability-Calibrated Consensus Prompt
- Reliability-Weighted SAM Distillation
- Anatomy-Aware Quality Control for pseudo-label learning

The goal is to improve unlabeled-sample supervision while keeping the baseline
segmentation backbone comparable to KnowSAM.

## Train

From the repository root:

```bash
bash ./variants/A3_RCP_KnowSAM/train_v100_a3_rcp.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=./SampleData \
DATASET=/260513_data_label1 \
SNAPSHOT_PATH=./Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13 \
BATCH_SIZE=24 \
LABELED_BS=12 \
bash ./variants/A3_RCP_KnowSAM/train_v100_a3_rcp.sh
```

Main outputs:

```text
Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13/fold_0/SGDL_best_model.pth
Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13/fold_0/sam_best_model.pth
Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13/fold_0/log.txt
Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13/fold_0/monitor/
```

## Test

```bash
bash ./variants/A3_RCP_KnowSAM/test_v100_a3_rcp.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
SNAPSHOT_PATH=./Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13 \
SPLIT=test \
bash ./variants/A3_RCP_KnowSAM/test_v100_a3_rcp.sh
```

The default prediction output contains case-level CSV metrics, summary JSON,
original images, ground-truth masks, predicted masks, overlays, and monitor
figures.

## Local Smoke Test

For a short local run:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128'
python .\variants\A3_RCP_KnowSAM\train_semi_SAM_a3_rcp.py `
  --batch_size 2 `
  --labeled_bs 1 `
  --num_workers 0 `
  --val_num_workers 0 `
  --max_iterations 2 `
  --val_interval 0 `
  --mixed_iterations 0 `
  --snapshot_path .\Results\A3_RCP_KnowSAM_smoke
```
