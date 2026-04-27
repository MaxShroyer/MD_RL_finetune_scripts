#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_v2_parallel_fast_common_20260416.sh"

KEY_ENV_VAR="${1:?missing api key env var}"
exec > >(tee -a "$LOG_DIR/subset_detect_and_sft.lane.log") 2>&1

printf '[%s] lane=subset_detect_and_sft key=%s start\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$KEY_ENV_VAR"

run_and_log \
  "$LOG_DIR/subset_detect_hybrid.fast.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_detect \
  --config "$CONFIG_DIR/train_inspector_detect_subset_balanced_1000_hybrid_v2_20260415.json" \
  --api-key-env-var "$KEY_ENV_VAR" \
  --finetune-name "inspector-detect-subset1k-v2-hybrid-fastparallel-20260416" \
  --wandb-run-name "inspector-detect-subset1k-v2-hybrid-fastparallel-20260416"

run_and_log \
  "$LOG_DIR/subset_query_sft_only.fast.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_subset_balanced_1000_v2_best_20260415.json" \
  --api-key-env-var "$KEY_ENV_VAR" \
  --mode sft \
  --rl-steps 0 \
  --finetune-name "inspector-query-subset1k-v2-sft-only-fastparallel-20260416" \
  --wandb-run-name "inspector-query-subset1k-v2-sft-only-fastparallel-20260416"

printf '[%s] lane=subset_detect_and_sft key=%s done\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$KEY_ENV_VAR"
