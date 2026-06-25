#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Semi-supervised defaults for the current split:
#   106 labeled / 117 unlabeled / 13 val / 13 test
# Design goals:
#   1. keep image_size at 256 for stability
#   2. keep labeled/unlabeled balanced inside each batch for mixup
#   3. keep enough iterations per epoch for smoother validation curves

export DATA_PATH="${DATA_PATH:-./SampleData}"
export DATASET="${DATASET:-/260513_data_label1}"
export IMAGE_SIZE="${IMAGE_SIZE:-256}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export LABELED_BS="${LABELED_BS:-8}"
export LR="${LR:-1e-4}"
export UNET_LR="${UNET_LR:-0.0025}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
export MIXED_ITERATIONS="${MIXED_ITERATIONS:-240}"
export VAL_INTERVAL="${VAL_INTERVAL:-10}"
export CONSISTENCY="${CONSISTENCY:-0.1}"
export CONSISTENCY_RAMPUP="${CONSISTENCY_RAMPUP:-200}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
export SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/train_260513_data_label1_v100_semi_106_117_13_13}"

exec bash "${ROOT_DIR}/train_v100.sh" "$@"
