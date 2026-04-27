#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_v2_parallel_fast_common_20260416.sh"

KEY_ENV_VAR="${1:?missing api key env var}"
exec > >(tee -a "$LOG_DIR/full_detect.lane.log") 2>&1

printf '[%s] lane=full_detect key=%s start\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$KEY_ENV_VAR"

FULL_DETECT_SFT_LOG="$LOG_DIR/full_detect_sft.fast.log"
run_and_log \
  "$FULL_DETECT_SFT_LOG" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_detect \
  --config "$CONFIG_DIR/train_inspector_detect_full_sft_v2_20260415.json" \
  --api-key-env-var "$KEY_ENV_VAR" \
  --finetune-name "inspector-detect-full-v2-sft-fastparallel-20260416" \
  --wandb-run-name "inspector-detect-full-v2-sft-fastparallel-20260416"
FULL_DETECT_FINETUNE_ID="$(require_finetune_id "$FULL_DETECT_SFT_LOG")"

run_and_log \
  "$LOG_DIR/full_detect_rl.fast.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_detect \
  --config "$CONFIG_DIR/train_inspector_detect_full_rl_v2_20260415.json" \
  --api-key-env-var "$KEY_ENV_VAR" \
  --finetune-id "$FULL_DETECT_FINETUNE_ID" \
  --wandb-run-name "inspector-detect-full-v2-rl-fastparallel-20260416"

printf '[%s] lane=full_detect key=%s done\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$KEY_ENV_VAR"
