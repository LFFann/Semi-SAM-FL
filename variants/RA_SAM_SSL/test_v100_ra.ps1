param(
  [string]$PythonBin = "python",
  [string]$DataPath = ".\SampleData",
  [string]$Dataset = "/260513_data_label1",
  [string]$Split = "test",
  [int]$ImageSize = 256,
  [int]$NumWorkers = 4,
  [string]$SnapshotPath = ".\Results\RA_SAM_SSL_V100_label1_106_117_13_13",
  [string]$SamCheckpoint = ".\sam_vit_b_01ec64.pth",
  [string]$SgdlModelPath = "",
  [string]$SamModelPath = "",
  [string]$SaveDir = ""
)

$ErrorActionPreference = "Stop"
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RootDir

if (-not $SgdlModelPath) { $SgdlModelPath = Join-Path $SnapshotPath "SGDL_best_model.pth" }
if (-not $SamModelPath) { $SamModelPath = Join-Path $SnapshotPath "sam_best_model.pth" }
if (-not (Test-Path $SgdlModelPath)) { $SgdlModelPath = Join-Path $SnapshotPath "fold_0\SGDL_best_model.pth" }
if (-not (Test-Path $SamModelPath)) { $SamModelPath = Join-Path $SnapshotPath "fold_0\sam_best_model.pth" }
if (-not (Test-Path $SgdlModelPath)) { throw "Missing SGDL checkpoint: $SgdlModelPath" }
if (-not (Test-Path $SamModelPath)) { throw "Missing SAM checkpoint: $SamModelPath" }
if (-not $SaveDir) { $SaveDir = Join-Path $SnapshotPath "prediction_$Split" }
New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null

& $PythonBin .\variants\RA_SAM_SSL\prediction_ra.py `
  --data_path $DataPath `
  --dataset $Dataset `
  --split $Split `
  --image_size $ImageSize `
  --sam_checkpoint $SamCheckpoint `
  --SGDL_model_path $SgdlModelPath `
  --sam_model_path $SamModelPath `
  --save_dir $SaveDir `
  --num_workers $NumWorkers
