#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export SOURCE_SNAPSHOT="${SOURCE_SNAPSHOT:-./Results/SRG-SAM_V100_260513_data_labeled30pct_finetune_lr12e4}"
export SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/SRG-SAM_V100_260513_data_labeled30pct_round2_branch_fg}"

export DATA_PATH="${DATA_PATH:-./SampleData}"
export DATASET="${DATASET:-/260513_data_labeled30pct}"
export NUM_CLASSES="${NUM_CLASSES:-3}"

export BATCH_SIZE="${BATCH_SIZE:-32}"
export LABELED_BS="${LABELED_BS:-16}"
export UNET_LR="${UNET_LR:-0.0008}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-1800}"
export VAL_INTERVAL="${VAL_INTERVAL:-25}"

export LAMBDA_U="${LAMBDA_U:-0.45}"
export LAMBDA_BRANCH="${LAMBDA_BRANCH:-0.8}"
export LAMBDA_U_WARMUP="${LAMBDA_U_WARMUP:-1}"
export LAMBDA_G="${LAMBDA_G:-0.03}"
export LAMBDA_B="${LAMBDA_B:-0.12}"
export CLASS_WEIGHTS="${CLASS_WEIGHTS:-0.2,1.7,1.2}"
export EXCLUDE_BG_DICE="${EXCLUDE_BG_DICE:-1}"
export SRG_GRAPH_LAMBDA="${SRG_GRAPH_LAMBDA:-0.25}"
export SRG_GRAPH_ALPHA="${SRG_GRAPH_ALPHA:-0.8}"
export SRG_EMA_DECAY="${SRG_EMA_DECAY:-0.997}"

bash ./variants/Multiclass_KnowSAM/finetune_v100_labeled30pct.sh "$@"

