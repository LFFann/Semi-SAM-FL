#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_label1}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/train_260513_data_label1_v100_semi_106_117_13_13}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"

BATCH_SIZE="${BATCH_SIZE:-16}"
LABELED_BS="${LABELED_BS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
LR="${LR:-1e-4}"
UNET_LR="${UNET_LR:-0.0025}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3000}"
MIXED_ITERATIONS="${MIXED_ITERATIONS:-240}"
VAL_INTERVAL="${VAL_INTERVAL:-10}"
CONSISTENCY="${CONSISTENCY:-0.1}"
CONSISTENCY_RAMPUP="${CONSISTENCY_RAMPUP:-200}"
LABELED_NUM="${LABELED_NUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
  echo "Missing SAM checkpoint: ${SAM_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${SNAPSHOT_PATH}"

echo "Starting V100 training with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SNAPSHOT_PATH=${SNAPSHOT_PATH}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LABELED_BS=${LABELED_BS}"
echo "  IMAGE_SIZE=${IMAGE_SIZE}"
echo "  LR=${LR}"
echo "  UNET_LR=${UNET_LR}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  MIXED_ITERATIONS=${MIXED_ITERATIONS}"
echo "  VAL_INTERVAL=${VAL_INTERVAL}"
echo "  CONSISTENCY=${CONSISTENCY}"
echo "  CONSISTENCY_RAMPUP=${CONSISTENCY_RAMPUP}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  VAL_NUM_WORKERS=${VAL_NUM_WORKERS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

EXTRA_ARGS=()
TRAIN_HELP="$("${PYTHON_BIN}" train_semi_SAM.py -h 2>&1 || true)"
if grep -q -- "--num_workers" <<< "${TRAIN_HELP}"; then
  EXTRA_ARGS+=(--num_workers "${NUM_WORKERS}")
fi
if grep -q -- "--val_num_workers" <<< "${TRAIN_HELP}"; then
  EXTRA_ARGS+=(--val_num_workers "${VAL_NUM_WORKERS}")
fi

"${PYTHON_BIN}" train_semi_SAM.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --labeled_num "${LABELED_NUM}" \
  --batch_size "${BATCH_SIZE}" \
  --labeled_bs "${LABELED_BS}" \
  --image_size "${IMAGE_SIZE}" \
  -lr "${LR}" \
  -UNet_lr "${UNET_LR}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --mixed_iterations "${MIXED_ITERATIONS}" \
  --val_interval "${VAL_INTERVAL}" \
  --consistency "${CONSISTENCY}" \
  --consistency_rampup "${CONSISTENCY_RAMPUP}" \
  --sam_checkpoint "${SAM_CHECKPOINT}" \
  --snapshot_path "${SNAPSHOT_PATH}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "Training finished. Outputs:"
echo "  ${SNAPSHOT_PATH}"
echo "  ${SNAPSHOT_PATH}/log.txt"
echo "  ${SNAPSHOT_PATH}/monitor"
