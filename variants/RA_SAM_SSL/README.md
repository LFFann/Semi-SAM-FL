# RA-SAM-SSL

`RA-SAM-SSL` is an isolated implementation of **Response-Audited SAM for Semi-Supervised Ultrasound Segmentation**.

It keeps the root KnowSAM code unchanged and places all method-specific code under `variants/RA_SAM_SSL/`.

## Method

The method audits SAM pseudo-boundaries before using them for unlabeled supervision:

1. Treat the student/fusion prediction as a hypothesis.
2. Build an SDF prompt family from the hypothesis.
3. Query SAM with a prompt-dose curve.
4. Apply a validity-checked local evidence intervention inside the boundary tube.
5. Re-query SAM and compute prompt response, ESR, jump instability, saturation, and intervention validity.
6. Attribute boundary pixels to prompt-dominated, prior-locked, evidence-sensitive, or unidentifiable regimes.
7. Admit only audited boundary regions for strong SAM boundary KD while keeping ordinary interior KD.

## Files

```text
variants/RA_SAM_SSL/
  IDEA_ANALYSIS.md        # idea-to-code mapping and paper-facing interpretation
  README.md               # usage
  train_semi_SAM_ra.py    # training entry
  trainer_ra.py           # KnowSAM trainer with RA-gated boundary KD
  ra_modules.py           # response audit, intervention validity, attribution
  prediction_ra.py        # SGDL/SAM evaluation and visual outputs
  diagnose_ra.py          # response diagnostics against labeled boundaries
  train_v100_ra.sh        # V100 training launcher
  test_v100_ra.sh         # evaluation launcher
  diagnose_v100_ra.sh     # diagnostic launcher
  ablate_v100_ra.sh       # core ablation launcher
  train_v100_ra.ps1       # PowerShell training launcher
  test_v100_ra.ps1        # PowerShell evaluation launcher
  diagnose_v100_ra.ps1    # PowerShell diagnostic launcher
  utils/losses_ra.py      # local loss definitions
```

## Train

From the repository root:

```bash
bash ./variants/RA_SAM_SSL/train_v100_ra.sh
```

On Windows PowerShell:

```powershell
.\variants\RA_SAM_SSL\train_v100_ra.ps1
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=./SampleData \
DATASET=/260513_data_label1 \
SNAPSHOT_PATH=./Results/RA_SAM_SSL_V100_label1 \
BATCH_SIZE=24 \
LABELED_BS=12 \
bash ./variants/RA_SAM_SSL/train_v100_ra.sh
```

Main outputs:

```text
Results/RA_SAM_SSL_V100_label1/SGDL_best_model.pth
Results/RA_SAM_SSL_V100_label1/sam_best_model.pth
Results/RA_SAM_SSL_V100_label1/log.txt
Results/RA_SAM_SSL_V100_label1/monitor/
```

## Test

```bash
SNAPSHOT_PATH=./Results/RA_SAM_SSL_V100_label1 \
bash ./variants/RA_SAM_SSL/test_v100_ra.sh
```

PowerShell:

```powershell
.\variants\RA_SAM_SSL\test_v100_ra.ps1 -SnapshotPath .\Results\RA_SAM_SSL_V100_label1
```

The prediction folder contains case-level CSV metrics, summary JSON, original images, GT masks, SGDL masks, SAM masks, and overlays.

## Diagnose

```bash
SNAPSHOT_PATH=./Results/RA_SAM_SSL_V100_label1 \
SPLIT=val \
MAX_CASES=0 \
bash ./variants/RA_SAM_SSL/diagnose_v100_ra.sh
```

PowerShell:

```powershell
.\variants\RA_SAM_SSL\diagnose_v100_ra.ps1 -SnapshotPath .\Results\RA_SAM_SSL_V100_label1 -Split val -MaxCases 0
```

Outputs include:

```text
diagnosis_val/case_diagnosis.csv
diagnosis_val/summary.json
diagnosis_val/heatmaps/*_{admission,pc,esr,ji,valid,prompt_dominated,prior_locked,evidence_sensitive,unidentifiable,boundary_error}.png
diagnosis_val/response_curves/*_response_curve.png
```

## Ablations

```bash
BASE_SNAPSHOT_PATH=./Results/RA_SAM_SSL_Ablations \
bash ./variants/RA_SAM_SSL/ablate_v100_ra.sh
```

The launcher runs:

- `RA_ENABLED=0`: original KnowSAM-style KD.
- `RA_BASELINE=prompt_ensemble`: equal-query prompt ensemble teacher.
- `RA_DISABLE_ESR=1`: prompt-response only.
- `RA_DISABLE_PROMPT=1`: evidence-response only.
- `RA_NO_INTERVENTION=1`: prompt-dose only.
- `RA_INTERVENTION_MODE=random`: random-tube intervention.
- `RA_INTERVENTION_MODE=interior`: interior intervention.
- full RA-SAM-SSL: prompt response + valid boundary evidence intervention + attribution.

## Local Smoke Test

```powershell
python .\variants\RA_SAM_SSL\train_semi_SAM_ra.py `
  --batch_size 2 `
  --labeled_bs 1 `
  --num_workers 0 `
  --val_num_workers 0 `
  --max_iterations 2 `
  --val_interval 0 `
  --mixed_iterations 0 `
  --snapshot_path .\Results\RA_SAM_SSL_smoke
```

For a low-cost diagnostic check after checkpoints exist:

```powershell
python .\variants\RA_SAM_SSL\diagnose_ra.py `
  --split val `
  --max_cases 2 `
  --SGDL_model_path .\Results\RA_SAM_SSL_smoke\SGDL_best_model.pth `
  --sam_model_path .\Results\RA_SAM_SSL_smoke\sam_best_model.pth `
  --save_dir .\Results\RA_SAM_SSL_smoke\diagnosis_val
```
