# PERD-KnowSAM

This variant implements **Prompt-Evidence Response Disentanglement (PERD)** as an isolated KnowSAM experimental version. It keeps the root KnowSAM baseline unchanged and places all method-specific code under `variants/PERD_KnowSAM/`.

## Method

PERD audits SAM pseudo-boundaries before using them for unlabeled distillation:

1. Build signed-distance mask prompts from the student/fusion prediction.
2. Query SAM with a prompt-dose family, by default `delta={-4,-2,0,2,4}`.
3. Build a boundary tube from the zero-dose SAM response.
4. Mildly attenuate local ultrasound evidence only inside the selected tube.
5. Re-query SAM and compute prompt compliance, evidence dependence, jump instability, and probe validity.
6. Keep ordinary interior KD, but gate boundary KD by the PERD trust map.

The implementation does not add a segmentation head, artifact detector, topology loss, or handcrafted edge gate.

## Files

```text
variants/PERD_KnowSAM/
  train_semi_SAM_perd.py   # training entry
  trainer_perd.py          # KnowSAM trainer with PERD-gated KD
  perd_modules.py          # SDF prompts, evidence attenuation, PC/ED/JI/V/trust
  prediction_perd.py       # SGDL/SAM evaluation and visual outputs
  diagnose_perd.py         # response diagnostics against labeled boundaries
  train_v100_perd.sh       # V100 training launcher
  test_v100_perd.sh        # evaluation launcher
  diagnose_v100_perd.sh    # PERD diagnostic launcher
  ablate_v100_perd.sh      # core ablation launcher
  utils/losses_perd.py     # local loss definitions
```

## Train

From the repository root:

```bash
bash ./variants/PERD_KnowSAM/train_v100_perd.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=./SampleData \
DATASET=/260513_data_label1 \
SNAPSHOT_PATH=./Results/PERD_KnowSAM_V100_label1_106_117_13_13 \
BATCH_SIZE=24 \
LABELED_BS=12 \
bash ./variants/PERD_KnowSAM/train_v100_perd.sh
```

Main outputs:

```text
Results/PERD_KnowSAM_V100_label1_106_117_13_13/SGDL_best_model.pth
Results/PERD_KnowSAM_V100_label1_106_117_13_13/sam_best_model.pth
Results/PERD_KnowSAM_V100_label1_106_117_13_13/log.txt
Results/PERD_KnowSAM_V100_label1_106_117_13_13/monitor/
```

## Test

```bash
SNAPSHOT_PATH=./Results/PERD_KnowSAM_V100_label1_106_117_13_13 \
bash ./variants/PERD_KnowSAM/test_v100_perd.sh
```

The prediction folder contains case-level CSV metrics, summary JSON, original images, GT masks, SGDL masks, SAM masks, and overlays.

## Diagnose

```bash
SNAPSHOT_PATH=./Results/PERD_KnowSAM_V100_label1_106_117_13_13 \
SPLIT=val \
MAX_CASES=0 \
bash ./variants/PERD_KnowSAM/diagnose_v100_perd.sh
```

Outputs include:

```text
diagnosis_val/case_diagnosis.csv
diagnosis_val/summary.json
diagnosis_val/heatmaps/*_{trust,pc,ed,ji,valid,boundary_error}.png
diagnosis_val/response_curves/*_response_curve.png
```

## Ablations

```bash
BASE_SNAPSHOT_PATH=./Results/PERD_KnowSAM_Ablations \
bash ./variants/PERD_KnowSAM/ablate_v100_perd.sh
```

The launcher runs:

- `PERD_ENABLED=0`: original KnowSAM-style KD.
- `PERD_BASELINE=prompt_ensemble`: equal-query prompt ensemble teacher.
- `PERD_DISABLE_ED=1`: prompt compliance only.
- `PERD_DISABLE_PC=1`: evidence dependence only.
- `PERD_NO_ATTENUATION=1`: prompt-dose only.
- `PERD_ATTENUATION_MODE=random`: random tube attenuation.
- `PERD_ATTENUATION_MODE=interior`: interior attenuation.

## Local Smoke Test

```powershell
python .\variants\PERD_KnowSAM\train_semi_SAM_perd.py `
  --batch_size 2 `
  --labeled_bs 1 `
  --num_workers 0 `
  --val_num_workers 0 `
  --max_iterations 2 `
  --val_interval 0 `
  --mixed_iterations 0 `
  --snapshot_path .\Results\PERD_KnowSAM_smoke
```

For a low-cost diagnostic check:

```powershell
python .\variants\PERD_KnowSAM\diagnose_perd.py `
  --split val `
  --max_cases 2 `
  --SGDL_model_path .\Results\PERD_KnowSAM_smoke\SGDL_best_model.pth `
  --sam_model_path .\Results\PERD_KnowSAM_smoke\sam_best_model.pth `
  --save_dir .\Results\PERD_KnowSAM_smoke\diagnosis_val
```
