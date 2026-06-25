#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Start again from the better round1 fine-tune checkpoint. Round2 improved class 1
# only briefly and then reduced class 2 / VNet stability.
export SOURCE_SNAPSHOT="${SOURCE_SNAPSHOT:-./Results/SRG-SAM_V100_260513_data_labeled30pct_finetune_lr12e4}"
export SNAPSHOT_PATH="${SNAPSHOT_PATH:-./Results/SRG-SAM_V100_260513_data_labeled30pct_round3_vnet_balance}"

export DATA_PATH="${DATA_PATH:-./SampleData}"
export DATASET="${DATASET:-/260513_data_labeled30pct}"
export NUM_CLASSES="${NUM_CLASSES:-3}"

export BATCH_SIZE="${BATCH_SIZE:-32}"
export LABELED_BS="${LABELED_BS:-16}"
export UNET_LR="${UNET_LR:-0.0005}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-1400}"
export VAL_INTERVAL="${VAL_INTERVAL:-25}"

export LAMBDA_U="${LAMBDA_U:-0.28}"
export LAMBDA_BRANCH="${LAMBDA_BRANCH:-0.0}"
export LAMBDA_BRANCH_UNET="${LAMBDA_BRANCH_UNET:-0.05}"
export LAMBDA_BRANCH_VNET="${LAMBDA_BRANCH_VNET:-0.65}"
export LAMBDA_U_WARMUP="${LAMBDA_U_WARMUP:-1}"
export LAMBDA_G="${LAMBDA_G:-0.01}"
export LAMBDA_B="${LAMBDA_B:-0.08}"
export CLASS_WEIGHTS="${CLASS_WEIGHTS:-0.45,1.25,1.10}"
export EXCLUDE_BG_DICE="${EXCLUDE_BG_DICE:-0}"
export SRG_GRAPH_LAMBDA="${SRG_GRAPH_LAMBDA:-0.18}"
export SRG_GRAPH_ALPHA="${SRG_GRAPH_ALPHA:-0.88}"
export SRG_EMA_DECAY="${SRG_EMA_DECAY:-0.998}"

bash ./variants/Multiclass_KnowSAM/finetune_v100_labeled30pct.sh "$@"

