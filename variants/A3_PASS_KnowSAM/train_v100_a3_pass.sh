#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_label1}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/A3_PASS_KnowSAM_V100_label1_106_117_13_13}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LABELED_BS="${LABELED_BS:-8}"
UNET_LR="${UNET_LR:-0.0025}"
PASS_STATE_LR="${PASS_STATE_LR:-0.001}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"
CONSISTENCY="${CONSISTENCY:-0.1}"
CONSISTENCY_RAMPUP="${CONSISTENCY_RAMPUP:-200}"
LABELED_NUM="${LABELED_NUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"

PASS_STATE_SIZE="${PASS_STATE_SIZE:-64}"
PASS_STATE_DIM="${PASS_STATE_DIM:-64}"
PASS_BASE_CHANNELS="${PASS_BASE_CHANNELS:-32}"
PASS_STATE_WEIGHT="${PASS_STATE_WEIGHT:-0.20}"
PASS_STATE_CONSISTENCY_WEIGHT="${PASS_STATE_CONSISTENCY_WEIGHT:-0.20}"
PASS_PSEUDO_WEIGHT="${PASS_PSEUDO_WEIGHT:-0.35}"
PASS_DECODE_CONSISTENCY_WEIGHT="${PASS_DECODE_CONSISTENCY_WEIGHT:-0.10}"
PASS_RELIABILITY_ALPHA="${PASS_RELIABILITY_ALPHA:-4.0}"
PASS_MIN_RELIABILITY="${PASS_MIN_RELIABILITY:-0.05}"
PASS_NOISE_STD="${PASS_NOISE_STD:-0.03}"
PASS_GAIN_RANGE="${PASS_GAIN_RANGE:-0.12}"
SAP_BOUNDARY_WEIGHT="${SAP_BOUNDARY_WEIGHT:-0.05}"
SAP_SHAPE_WEIGHT="${SAP_SHAPE_WEIGHT:-0.03}"
SAP_AREA_LOWER="${SAP_AREA_LOWER:-0.001}"
SAP_AREA_UPPER="${SAP_AREA_UPPER:-0.08}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

mkdir -p "${SNAPSHOT_PATH}"

echo "Starting A3-PASS-KnowSAM V100-32G training with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SNAPSHOT_PATH=${SNAPSHOT_PATH}"
echo "  IMAGE_SIZE=${IMAGE_SIZE}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LABELED_BS=${LABELED_BS}"
echo "  UNET_LR=${UNET_LR}"
echo "  PASS_STATE_LR=${PASS_STATE_LR}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  VAL_INTERVAL=${VAL_INTERVAL}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" ./variants/A3_PASS_KnowSAM/train_semi_SAM_a3_pass.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --labeled_num "${LABELED_NUM}" \
  --batch_size "${BATCH_SIZE}" \
  --labeled_bs "${LABELED_BS}" \
  --image_size "${IMAGE_SIZE}" \
  -UNet_lr "${UNET_LR}" \
  --pass_state_lr "${PASS_STATE_LR}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --val_interval "${VAL_INTERVAL}" \
  --consistency "${CONSISTENCY}" \
  --consistency_rampup "${CONSISTENCY_RAMPUP}" \
  --snapshot_path "${SNAPSHOT_PATH}" \
  --num_workers "${NUM_WORKERS}" \
  --val_num_workers "${VAL_NUM_WORKERS}" \
  --pass_state_size "${PASS_STATE_SIZE}" \
  --pass_state_dim "${PASS_STATE_DIM}" \
  --pass_base_channels "${PASS_BASE_CHANNELS}" \
  --pass_state_weight "${PASS_STATE_WEIGHT}" \
  --pass_state_consistency_weight "${PASS_STATE_CONSISTENCY_WEIGHT}" \
  --pass_pseudo_weight "${PASS_PSEUDO_WEIGHT}" \
  --pass_decode_consistency_weight "${PASS_DECODE_CONSISTENCY_WEIGHT}" \
  --pass_reliability_alpha "${PASS_RELIABILITY_ALPHA}" \
  --pass_min_reliability "${PASS_MIN_RELIABILITY}" \
  --pass_noise_std "${PASS_NOISE_STD}" \
  --pass_gain_range "${PASS_GAIN_RANGE}" \
  --sap_boundary_weight "${SAP_BOUNDARY_WEIGHT}" \
  --sap_shape_weight "${SAP_SHAPE_WEIGHT}" \
  --sap_area_lower "${SAP_AREA_LOWER}" \
  --sap_area_upper "${SAP_AREA_UPPER}" \
  "$@"

echo "Training finished. Outputs:"
echo "  ${SNAPSHOT_PATH}"
echo "  ${SNAPSHOT_PATH}/fold_0/log.txt"
echo "  ${SNAPSHOT_PATH}/fold_0/PASS_best_model.pth"
echo "  ${SNAPSHOT_PATH}/fold_0/monitor"
