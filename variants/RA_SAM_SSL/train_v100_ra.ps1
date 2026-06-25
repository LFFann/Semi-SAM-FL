param(
  [string]$PythonBin = "python",
  [string]$DataPath = ".\SampleData",
  [string]$Dataset = "/260513_data_label1",
  [int]$ImageSize = 256,
  [int]$BatchSize = 24,
  [int]$LabeledBs = 12,
  [int]$NumWorkers = 4,
  [int]$ValNumWorkers = 2,
  [int]$MaxIterations = 50000,
  [int]$MixedIterations = 12000,
  [int]$ValInterval = 200,
  [string]$SnapshotPath = ".\Results\RA_SAM_SSL_V100_label1_106_117_13_13",
  [string]$SamCheckpoint = ".\sam_vit_b_01ec64.pth",
  [int]$RaEnabled = 1,
  [string]$RaDeltaLevels = "-4,-2,0,2,4",
  [string]$RaInterventionMode = "boundary",
  [string]$RaBaseline = "response_audit"
)

$ErrorActionPreference = "Stop"
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RootDir
New-Item -ItemType Directory -Force -Path $SnapshotPath | Out-Null

& $PythonBin .\variants\RA_SAM_SSL\train_semi_SAM_ra.py `
  --data_path $DataPath `
  --dataset $Dataset `
  --image_size $ImageSize `
  --batch_size $BatchSize `
  --labeled_bs $LabeledBs `
  --num_workers $NumWorkers `
  --val_num_workers $ValNumWorkers `
  --max_iterations $MaxIterations `
  --mixed_iterations $MixedIterations `
  --val_interval $ValInterval `
  --snapshot_path $SnapshotPath `
  --sam_checkpoint $SamCheckpoint `
  --ra_enabled $RaEnabled `
  --ra_delta_levels $RaDeltaLevels `
  --ra_intervention_mode $RaInterventionMode `
  --ra_baseline $RaBaseline
