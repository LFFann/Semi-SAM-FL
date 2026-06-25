#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_label1}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/A3_RCP_KnowSAM_V100_label1_106_117_13_13}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"

IMAGE_SIZE="${IMAGE_SIZE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LABELED_BS="${LABELED_BS:-8}"
LR="${LR:-1e-4}"
UNET_LR="${UNET_LR:-0.0025}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"
MIXED_ITERATIONS="${MIXED_ITERATIONS:-12000}"
VAL_INTERVAL="${VAL_INTERVAL:-200}"
CONSISTENCY="${CONSISTENCY:-0.1}"
CONSISTENCY_RAMPUP="${CONSISTENCY_RAMPUP:-200}"
LABELED_NUM="${LABELED_NUM:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"

UCKD_ALPHA="${UCKD_ALPHA:-2.0}"
UCKD_MIN_WEIGHT="${UCKD_MIN_WEIGHT:-0.15}"
QAPL_MIN_WEIGHT="${QAPL_MIN_WEIGHT:-0.20}"
RCP_ALPHA="${RCP_ALPHA:-2.0}"
RCP_MIN_WEIGHT="${RCP_MIN_WEIGHT:-0.10}"
RCP_SHARPEN="${RCP_SHARPEN:-1.5}"
SAP_BOUNDARY_WEIGHT="${SAP_BOUNDARY_WEIGHT:-0.10}"
SAP_SHAPE_WEIGHT="${SAP_SHAPE_WEIGHT:-0.05}"
SAP_AREA_LOWER="${SAP_AREA_LOWER:-0.001}"
SAP_AREA_UPPER="${SAP_AREA_UPPER:-0.08}"

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=8
fi

if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
  echo "Missing SAM checkpoint: ${SAM_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${SNAPSHOT_PATH}"

echo "Starting A3-RCP-KnowSAM V100 training with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SNAPSHOT_PATH=${SNAPSHOT_PATH}"
echo "  IMAGE_SIZE=${IMAGE_SIZE}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LABELED_BS=${LABELED_BS}"
echo "  LR=${LR}"
echo "  UNET_LR=${UNET_LR}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  MIXED_ITERATIONS=${MIXED_ITERATIONS}"
echo "  VAL_INTERVAL=${VAL_INTERVAL}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  VAL_NUM_WORKERS=${VAL_NUM_WORKERS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" ./variants/A3_RCP_KnowSAM/train_semi_SAM_a3_rcp.py \
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
  --num_workers "${NUM_WORKERS}" \
  --val_num_workers "${VAL_NUM_WORKERS}" \
  --uckd_alpha "${UCKD_ALPHA}" \
  --uckd_min_weight "${UCKD_MIN_WEIGHT}" \
  --qapl_min_weight "${QAPL_MIN_WEIGHT}" \
  --rcp_alpha "${RCP_ALPHA}" \
  --rcp_min_weight "${RCP_MIN_WEIGHT}" \
  --rcp_sharpen "${RCP_SHARPEN}" \
  --sap_boundary_weight "${SAP_BOUNDARY_WEIGHT}" \
  --sap_shape_weight "${SAP_SHAPE_WEIGHT}" \
  --sap_area_lower "${SAP_AREA_LOWER}" \
  --sap_area_upper "${SAP_AREA_UPPER}" \
  "$@"

echo "Training finished. Outputs:"
echo "  ${SNAPSHOT_PATH}"
echo "  ${SNAPSHOT_PATH}/fold_0/log.txt"
echo "  ${SNAPSHOT_PATH}/fold_0/monitor"
