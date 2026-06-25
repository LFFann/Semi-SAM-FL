# A3-SAM / KnowSAM

This repository is built on KnowSAM, the official implementation of
"Learnable Prompting SAM-induced Knowledge Distillation for Semi-supervised
Medical Image Segmentation". The current working version keeps the original
KnowSAM baseline and adds A3-SAM variants for semi-supervised medical image
segmentation experiments.

## What Is Included

- Original KnowSAM training, validation, and prediction code.
- Dataset handling updates for `train_semi`, `val`, and `test` image/mask splits.
- Prediction utilities that save case-level metrics, masks, overlays, logs, and JSON summaries.
- Training/evaluation monitor utilities in `utils/training_monitor.py`.
- Dataset preparation and split-management scripts in `scripts/`.
- A3-RCP and A3-PASS experimental variants under `variants/`.
- A 3-class `background + two foreground classes` KnowSAM variant under `variants/Multiclass_KnowSAM/`.

Large datasets, training outputs, model weights, and Python cache files are intentionally excluded from Git.

## Repository Layout

```text
Model/                         KnowSAM, SAM, UNet, VNet, and prompt modules
dataloader/                    Dataset loaders, transforms, and batch sampler
utils/                         Metrics, losses, and training monitor utilities
scripts/                       Dataset preparation and experiment helper scripts
variants/A3_RCP_KnowSAM/       A3-RCP variant
variants/A3_PASS_KnowSAM/      A3-PASS variant
variants/Multiclass_KnowSAM/   3-class KnowSAM variant
train_semi_SAM.py              Original KnowSAM training entry
prediction.py                  Updated prediction/evaluation entry
requirements.txt               Python dependencies
```

## Environment

Create a Python environment with PyTorch, then install the project dependencies:

```bash
pip install -r requirements.txt
```

The scripts are written for CUDA training. CPU execution is not the target path for full experiments.

## Data Preparation

The expected 2D dataset layout is:

```text
SampleData/<dataset_name>/
  labeled/
    image/
    mask/
  unlabeled/
    image/
  val/
    image/
    mask/
  test/
    image/
    mask/
```

Use the helper scripts when preparing local data:

```bash
python scripts/prepare_260513_dataset.py --target-label 1 --output-root ./SampleData/260513_data_label1
python scripts/prepare_260513_dataset.py --target-label 2 --output-root ./SampleData/260513_data_label2
python scripts/prepare_260513_dataset.py --multi-class --output-root ./SampleData/260513_data_multiclass
python scripts/prepare_knowsam_dataset.py
python scripts/prepare_tumor_2_dataset.py
python scripts/repartition_annotated_splits.py
```

Local data directories are ignored by Git. Keep medical images, archives, and model checkpoints outside commits.
The current default binary dataset is `SampleData/260513_data_label1`, generated from label value 1 in `data/260513_data`. `SampleData/260513_data_label2` contains the same split with label value 2 as foreground. `SampleData/260513_data_multiclass` preserves mask labels `0/1/2` for 3-class training. Each split has 106 labeled training images, 117 unlabeled training images, 13 validation images, and 13 test images.

## Baseline Training

Run the original KnowSAM training entry from the repository root:

```bash
python train_semi_SAM.py
```

For ACDC-style data:

```bash
python train_semi_SAM_ACDC.py
```

## Baseline Prediction

The updated prediction script evaluates a selected split and writes visualizations plus metrics:

```bash
python prediction.py \
  --data_path ./SampleData \
  --dataset /260513_data_label1 \
  --split test \
  --SGDL_model_path ./Results/<experiment>/SGDL_best_model.pth \
  --save_dir ./Results/<experiment>/prediction_test
```

Outputs include:

```text
prediction.log
case_metrics.csv
summary.json
original/
gt_mask/
pred_mask/
overlay/
monitor/
```

## A3-RCP Variant

`variants/A3_RCP_KnowSAM/` contains an independent training/evaluation path for:

- Reliability-Calibrated Consensus Prompt
- Reliability-Weighted SAM Distillation
- Anatomy-Aware Quality Control for pseudo-label learning

Run training:

```bash
bash ./variants/A3_RCP_KnowSAM/train_v100_a3_rcp.sh
```

Run evaluation:

```bash
bash ./variants/A3_RCP_KnowSAM/test_v100_a3_rcp.sh
```

See `variants/A3_RCP_KnowSAM/README.md` for variant-specific details.

## A3-PASS Variant

`variants/A3_PASS_KnowSAM/` contains the compact A3-PASS design:

- Acoustic-Anatomical State Posterior
- State-Conditioned Mask Decoder
- Posterior-Guided Unlabeled Learning

Run training:

```bash
bash ./variants/A3_PASS_KnowSAM/train_v100_a3_pass.sh
```

Run evaluation:

```bash
bash ./variants/A3_PASS_KnowSAM/test_v100_a3_pass.sh
```

See `variants/A3_PASS_KnowSAM/README.md` for variant-specific details.

## Multiclass KnowSAM Variant

`variants/Multiclass_KnowSAM/` trains KnowSAM with `num_classes=3` for background plus two foreground classes:

```bash
bash ./variants/Multiclass_KnowSAM/train_v100_multiclass.sh
bash ./variants/Multiclass_KnowSAM/test_v100_multiclass.sh
```

The multiclass prediction path reports macro-average metrics over foreground classes 1 and 2, plus per-class metrics.

## Notes

- `sam_vit_b_01ec64.pth` and experiment checkpoints are not committed. Download or place them locally when needed.
- `Results/`, `SampleData/`, `data/`, and generated comparison repositories are ignored.
- Keep each new experimental method under `variants/<method_name>/` to preserve the original baseline.

## Acknowledgements

This project builds on KnowSAM and SSL4MIS, and uses SAM-related modules for prompt-based segmentation research.
