#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_labeled30pct}"
NUM_CLASSES="${NUM_CLASSES:-3}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/CA_SRG_SAMPP_labeled30pct_${TRAIN_STAGE:-student}}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"

TRAIN_STAGE="${TRAIN_STAGE:-student}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LABELED_BS="${LABELED_BS:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
UNET_LR="${UNET_LR:-0.003}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
VAL_INTERVAL="${VAL_INTERVAL:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"

USE_TRAIN_ONLY_SAM="${USE_TRAIN_ONLY_SAM:-false}"
USE_ADAPTER_SAM_SEMANTIC="${USE_ADAPTER_SAM_SEMANTIC:-false}"
USE_FROZEN_SAM_STRUCT="${USE_FROZEN_SAM_STRUCT:-false}"
SGDL_INIT_CHECKPOINT="${SGDL_INIT_CHECKPOINT:-}"
GRAPH_INIT_CHECKPOINT="${GRAPH_INIT_CHECKPOINT:-}"

EXTRA_ARGS=()
if [[ -n "${SGDL_INIT_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--sgdl_init_checkpoint "${SGDL_INIT_CHECKPOINT}")
fi
if [[ -n "${GRAPH_INIT_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--graph_init_checkpoint "${GRAPH_INIT_CHECKPOINT}")
fi

mkdir -p "${SNAPSHOT_PATH}"

echo "Starting CA_SRG_SAMPP with:"
echo "  TRAIN_STAGE=${TRAIN_STAGE}"
echo "  DATASET=${DATASET}"
echo "  SNAPSHOT_PATH=${SNAPSHOT_PATH}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LABELED_BS=${LABELED_BS}"
echo "  UNET_LR=${UNET_LR}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  VAL_INTERVAL=${VAL_INTERVAL}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" train_ssl.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --num_classes "${NUM_CLASSES}" \
  --batch_size "${BATCH_SIZE}" \
  --labeled_bs "${LABELED_BS}" \
  --image_size "${IMAGE_SIZE}" \
  -UNet_lr "${UNET_LR}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --val_interval "${VAL_INTERVAL}" \
  --sam_checkpoint "${SAM_CHECKPOINT}" \
  --snapshot_path "${SNAPSHOT_PATH}" \
  --num_workers "${NUM_WORKERS}" \
  --val_num_workers "${VAL_NUM_WORKERS}" \
  --use_ca_srg_sampp true \
  --train_stage "${TRAIN_STAGE}" \
  --graph_type region_boundary_prototype \
  --reliability_type complementarity \
  --reliability_target_type delta_utility \
  --deploy_fusion_type gated_residual \
  --use_train_only_sam "${USE_TRAIN_ONLY_SAM}" \
  --use_adapter_sam_semantic "${USE_ADAPTER_SAM_SEMANTIC}" \
  --use_frozen_sam_struct "${USE_FROZEN_SAM_STRUCT}" \
  --lambda_sam 0 \
  --lambda_R 0 \
  --lambda_graph 0 \
  --lambda_graph_reg 0 \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "CA_SRG_SAMPP finished. Outputs:"
echo "  ${SNAPSHOT_PATH}"
echo "  ${SNAPSHOT_PATH}/best_student_model.pth"
echo "  ${SNAPSHOT_PATH}/best_deploy_model.pth"
