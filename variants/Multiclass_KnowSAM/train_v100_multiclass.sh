#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-./SampleData}"
DATASET="${DATASET:-/260513_data_labeled30pct}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/SRG-SAM_Lite_V100_260513_data_labeled30pct_from_scratch}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-./sam_vit_b_01ec64.pth}"
NUM_CLASSES="${NUM_CLASSES:-3}"

BATCH_SIZE="${BATCH_SIZE:-16}"
LABELED_BS="${LABELED_BS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
LR="${LR:-1e-4}"
UNET_LR="${UNET_LR:-0.002}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
MIXED_ITERATIONS="${MIXED_ITERATIONS:-0}"
VAL_INTERVAL="${VAL_INTERVAL:-50}"
LAMBDA_U="${LAMBDA_U:-0.6}"
LAMBDA_BRANCH="${LAMBDA_BRANCH:-0.0}"
LAMBDA_U_WARMUP="${LAMBDA_U_WARMUP:-500}"
LAMBDA_G="${LAMBDA_G:-0.08}"
LAMBDA_B="${LAMBDA_B:-0.08}"
LAMBDA_SAM="${LAMBDA_SAM:-0.35}"
LAMBDA_FUSE="${LAMBDA_FUSE:-1.0}"
LAMBDA_DEPLOY="${LAMBDA_DEPLOY:-1.0}"
LAMBDA_R="${LAMBDA_R:-0.25}"
LAMBDA_GRAPH_REG="${LAMBDA_GRAPH_REG:-0.01}"
LAMBDA_PG="${LAMBDA_PG:-0.5}"
LAMBDA_PA="${LAMBDA_PA:-0.5}"
ALPHA_GRAPH="${ALPHA_GRAPH:-0.25}"
BETA_SAM="${BETA_SAM:-0.25}"
USE_SRG_SAM_LITE="${USE_SRG_SAM_LITE:-true}"
USE_TRAIN_ONLY_SAM="${USE_TRAIN_ONLY_SAM:-true}"
INFERENCE_WITHOUT_SAM="${INFERENCE_WITHOUT_SAM:-true}"
USE_DPG_DEPLOY="${USE_DPG_DEPLOY:-true}"
USE_RELIABILITY_HEAD="${USE_RELIABILITY_HEAD:-true}"
USE_ADAPTER_SAM_SEMANTIC="${USE_ADAPTER_SAM_SEMANTIC:-true}"
USE_DPG_LOGITS_FUSION="${USE_DPG_LOGITS_FUSION:-true}"
USE_FUSION="${USE_FUSION:-true}"
USE_RELIABILITY_GATE="${USE_RELIABILITY_GATE:-true}"
EVAL_WITH_SSRF="${EVAL_WITH_SSRF:-false}"
EVAL_FULL_SAM_ASSISTED="${EVAL_FULL_SAM_ASSISTED:-false}"
CLASS_WEIGHTS="${CLASS_WEIGHTS:-}"
EXCLUDE_BG_DICE="${EXCLUDE_BG_DICE:-0}"
SRG_GRAPH_LAMBDA="${SRG_GRAPH_LAMBDA:-0.5}"
SRG_GRAPH_ALPHA="${SRG_GRAPH_ALPHA:-0.75}"
SRG_PROMPT_COUNT="${SRG_PROMPT_COUNT:-4}"
SRG_AFFINITY_SIZE="${SRG_AFFINITY_SIZE:-16}"
SRG_EMA_DECAY="${SRG_EMA_DECAY:-0.995}"
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

echo "Starting SRG-SAM multiclass training with:"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATASET=${DATASET}"
echo "  SNAPSHOT_PATH=${SNAPSHOT_PATH}"
echo "  NUM_CLASSES=${NUM_CLASSES}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  LABELED_BS=${LABELED_BS}"
echo "  IMAGE_SIZE=${IMAGE_SIZE}"
echo "  LR=${LR}"
echo "  UNET_LR=${UNET_LR}"
echo "  MAX_ITERATIONS=${MAX_ITERATIONS}"
echo "  VAL_INTERVAL=${VAL_INTERVAL}"
echo "  LAMBDA_U=${LAMBDA_U}"
echo "  LAMBDA_BRANCH=${LAMBDA_BRANCH}"
echo "  LAMBDA_U_WARMUP=${LAMBDA_U_WARMUP}"
echo "  LAMBDA_G=${LAMBDA_G}"
echo "  LAMBDA_B=${LAMBDA_B}"
echo "  LAMBDA_SAM=${LAMBDA_SAM}"
echo "  LAMBDA_FUSE=${LAMBDA_FUSE}"
echo "  LAMBDA_DEPLOY=${LAMBDA_DEPLOY}"
echo "  LAMBDA_R=${LAMBDA_R}"
echo "  LAMBDA_GRAPH_REG=${LAMBDA_GRAPH_REG}"
echo "  LAMBDA_PG=${LAMBDA_PG}"
echo "  LAMBDA_PA=${LAMBDA_PA}"
echo "  ALPHA_GRAPH=${ALPHA_GRAPH}"
echo "  BETA_SAM=${BETA_SAM}"
echo "  USE_SRG_SAM_LITE=${USE_SRG_SAM_LITE}"
echo "  USE_TRAIN_ONLY_SAM=${USE_TRAIN_ONLY_SAM}"
echo "  INFERENCE_WITHOUT_SAM=${INFERENCE_WITHOUT_SAM}"
echo "  USE_DPG_DEPLOY=${USE_DPG_DEPLOY}"
echo "  USE_RELIABILITY_HEAD=${USE_RELIABILITY_HEAD}"
echo "  USE_ADAPTER_SAM_SEMANTIC=${USE_ADAPTER_SAM_SEMANTIC}"
echo "  USE_DPG_LOGITS_FUSION=${USE_DPG_LOGITS_FUSION}"
echo "  USE_FUSION=${USE_FUSION}"
echo "  USE_RELIABILITY_GATE=${USE_RELIABILITY_GATE}"
echo "  EVAL_WITH_SSRF=${EVAL_WITH_SSRF}"
echo "  EVAL_FULL_SAM_ASSISTED=${EVAL_FULL_SAM_ASSISTED}"
echo "  CLASS_WEIGHTS=${CLASS_WEIGHTS:-<none>}"
echo "  EXCLUDE_BG_DICE=${EXCLUDE_BG_DICE}"
echo "  SRG_GRAPH_LAMBDA=${SRG_GRAPH_LAMBDA}"
echo "  SRG_GRAPH_ALPHA=${SRG_GRAPH_ALPHA}"
echo "  SRG_PROMPT_COUNT=${SRG_PROMPT_COUNT}"
echo "  SRG_AFFINITY_SIZE=${SRG_AFFINITY_SIZE}"
echo "  SRG_EMA_DECAY=${SRG_EMA_DECAY}"
echo "  NUM_WORKERS=${NUM_WORKERS}"
echo "  VAL_NUM_WORKERS=${VAL_NUM_WORKERS}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

EXTRA_ARGS=()
if [[ -n "${CLASS_WEIGHTS}" ]]; then
  EXTRA_ARGS+=(--class_weights "${CLASS_WEIGHTS}")
fi
if [[ "${EXCLUDE_BG_DICE}" == "1" ]]; then
  EXTRA_ARGS+=(--exclude_bg_dice)
fi

"${PYTHON_BIN}" train_ssl.py \
  --data_path "${DATA_PATH}" \
  --dataset "${DATASET}" \
  --num_classes "${NUM_CLASSES}" \
  --labeled_num 1 \
  --batch_size "${BATCH_SIZE}" \
  --labeled_bs "${LABELED_BS}" \
  --image_size "${IMAGE_SIZE}" \
  -lr "${LR}" \
  -UNet_lr "${UNET_LR}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --mixed_iterations "${MIXED_ITERATIONS}" \
  --val_interval "${VAL_INTERVAL}" \
  --lambda_u "${LAMBDA_U}" \
  --lambda_branch "${LAMBDA_BRANCH}" \
  --lambda_u_warmup "${LAMBDA_U_WARMUP}" \
  --lambda_g "${LAMBDA_G}" \
  --lambda_b "${LAMBDA_B}" \
  --lambda_sam "${LAMBDA_SAM}" \
  --lambda_fuse "${LAMBDA_FUSE}" \
  --lambda_deploy "${LAMBDA_DEPLOY}" \
  --lambda_R "${LAMBDA_R}" \
  --lambda_graph_reg "${LAMBDA_GRAPH_REG}" \
  --lambda_pg "${LAMBDA_PG}" \
  --lambda_pa "${LAMBDA_PA}" \
  --alpha_graph "${ALPHA_GRAPH}" \
  --beta_sam "${BETA_SAM}" \
  --use_srg_sam_lite "${USE_SRG_SAM_LITE}" \
  --use_train_only_sam "${USE_TRAIN_ONLY_SAM}" \
  --inference_without_sam "${INFERENCE_WITHOUT_SAM}" \
  --use_dpg_deploy "${USE_DPG_DEPLOY}" \
  --use_reliability_head "${USE_RELIABILITY_HEAD}" \
  --use_adapter_sam_semantic "${USE_ADAPTER_SAM_SEMANTIC}" \
  --use_dpg_logits_fusion "${USE_DPG_LOGITS_FUSION}" \
  --use_fusion "${USE_FUSION}" \
  --use_reliability_gate "${USE_RELIABILITY_GATE}" \
  --eval_with_ssrf "${EVAL_WITH_SSRF}" \
  --eval_full_sam_assisted "${EVAL_FULL_SAM_ASSISTED}" \
  "${EXTRA_ARGS[@]}" \
  --srg_graph_lambda "${SRG_GRAPH_LAMBDA}" \
  --srg_graph_alpha "${SRG_GRAPH_ALPHA}" \
  --srg_prompt_count "${SRG_PROMPT_COUNT}" \
  --srg_affinity_size "${SRG_AFFINITY_SIZE}" \
  --srg_ema_decay "${SRG_EMA_DECAY}" \
  --sam_checkpoint "${SAM_CHECKPOINT}" \
  --snapshot_path "${SNAPSHOT_PATH}" \
  --num_workers "${NUM_WORKERS}" \
  --val_num_workers "${VAL_NUM_WORKERS}" \
  "$@"

echo "Training finished. Outputs:"
echo "  ${SNAPSHOT_PATH}"
echo "  ${SNAPSHOT_PATH}/SGDL_best_model.pth"
echo "  ${SNAPSHOT_PATH}/DPG_best_model.pth"
echo "  ${SNAPSHOT_PATH}/SRG_SAM_Lite_deploy_SGDL_best_model.pth"
echo "  ${SNAPSHOT_PATH}/SRG_SAM_Lite_deploy_DPG_best_model.pth"
echo "  ${SNAPSHOT_PATH}/SRG_SAM_Lite_deploy_reliability_best_model.pth"
