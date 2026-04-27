#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_v2_query_salvage_common_20260416.sh"

KEY_ENV_VAR="${1:?missing api key env var}"
exec > >(tee -a "$LOG_DIR/subset_query_combined_salvage.lane.log") 2>&1

printf '[%s] lane=subset_query_combined_salvage key=%s start\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$KEY_ENV_VAR"

SUBSET_QUERY_COMBINED_SFT_LOG="$LOG_DIR/subset_query_combined_sft.salvage.log"
run_and_log \
  "$SUBSET_QUERY_COMBINED_SFT_LOG" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_subset_balanced_1000_v2_best_20260415.json" \
  --api-key-env-var "$KEY_ENV_VAR" \
  --mode sft \
  --rl-steps 0 \
  --eval-every 15 \
  --save-every 15 \
  --no-async-checkpoint-eval \
  --finetune-name "inspector-query-subset1k-v2-combined-sft-sync-eval-20260416" \
  --wandb-run-name "inspector-query-subset1k-v2-combined-sft-sync-eval-20260416"
SUBSET_QUERY_COMBINED_FINETUNE_ID="$(require_finetune_id "$SUBSET_QUERY_COMBINED_SFT_LOG")"

run_and_log \
  "$LOG_DIR/subset_query_combined_rl.salvage.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_subset_balanced_1000_v2_best_20260415.json" \
  --api-key-env-var "$KEY_ENV_VAR" \
  --mode rl \
  --finetune-id "$SUBSET_QUERY_COMBINED_FINETUNE_ID" \
  --sft-steps 0 \
  --rl-steps 120 \
  --resume-step-offset 60 \
  --eval-every 15 \
  --save-every 15 \
  --no-async-checkpoint-eval \
  --wandb-run-name "inspector-query-subset1k-v2-combined-rl-sync-eval-20260416"

printf '[%s] lane=subset_query_combined_salvage key=%s done\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$KEY_ENV_VAR"
