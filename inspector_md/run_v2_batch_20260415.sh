#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PYTHON_BIN_DEFAULT="$REPO_ROOT/.venv/bin/python"
if [[ -x "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$PYTHON_BIN_DEFAULT" ]]; then
  PYTHON_BIN="$PYTHON_BIN_DEFAULT"
else
  PYTHON_BIN="python3"
fi

RUN_ROOT="$REPO_ROOT/inspector_md/outputs/v2_launch_20260415"
LOG_DIR="$RUN_ROOT/logs"
CONFIG_DIR="$REPO_ROOT/inspector_md/configs/runs_20260415"
mkdir -p "$LOG_DIR"

run_and_log() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname "$log_path")"
  {
    printf '[%s] CMD ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%q ' "$@"
    printf '\n'
  } | tee "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

extract_finetune_id() {
  "$PYTHON_BIN" - "$1" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"resolved_finetune_id=([A-Za-z0-9_-]+)", text)
print(matches[-1] if matches else "")
PY
}

require_finetune_id() {
  local log_path="$1"
  local finetune_id
  finetune_id="$(extract_finetune_id "$log_path")"
  if [[ -z "$finetune_id" ]]; then
    echo "failed to extract finetune id from $log_path" >&2
    exit 1
  fi
  printf '%s\n' "$finetune_id"
}

echo "using python: $PYTHON_BIN"
echo "logs: $LOG_DIR"

# Build fresh dataset-v2 outputs with OpenRouter refresh before any query training.
run_and_log \
  "$LOG_DIR/build_full_dataset_v2.log" \
  "$PYTHON_BIN" -m inspector_md.build_inspector_dataset \
  --config "$CONFIG_DIR/build_inspector_dataset_full_v2_20260415.json"

run_and_log \
  "$LOG_DIR/build_subset_dataset_v2.log" \
  "$PYTHON_BIN" -m inspector_md.build_inspector_dataset \
  --config "$CONFIG_DIR/build_inspector_dataset_subset_balanced_1000_v2_20260415.json"

# Full detect: use the established two-stage SFT -> RL path.
FULL_DETECT_SFT_LOG="$LOG_DIR/full_detect_sft.log"
run_and_log \
  "$FULL_DETECT_SFT_LOG" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_detect \
  --config "$CONFIG_DIR/train_inspector_detect_full_sft_v2_20260415.json" \
  --finetune-name "inspector-detect-full-v2-sft-20260415" \
  --wandb-run-name "inspector-detect-full-v2-sft-20260415"
FULL_DETECT_FINETUNE_ID="$(require_finetune_id "$FULL_DETECT_SFT_LOG")"

run_and_log \
  "$LOG_DIR/full_detect_rl.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_detect \
  --config "$CONFIG_DIR/train_inspector_detect_full_rl_v2_20260415.json" \
  --finetune-id "$FULL_DETECT_FINETUNE_ID" \
  --wandb-run-name "inspector-detect-full-v2-rl-after-sft-20260415"

# Full query combined: keep the backend handoff explicit as separate SFT then RL runs.
FULL_QUERY_SFT_LOG="$LOG_DIR/full_query_combined_sft.log"
run_and_log \
  "$FULL_QUERY_SFT_LOG" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_full_v2_best_20260415.json" \
  --mode sft \
  --rl-steps 0 \
  --finetune-name "inspector-query-full-v2-sft-20260415" \
  --wandb-run-name "inspector-query-full-v2-sft-20260415"
FULL_QUERY_FINETUNE_ID="$(require_finetune_id "$FULL_QUERY_SFT_LOG")"

run_and_log \
  "$LOG_DIR/full_query_combined_rl.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_full_v2_best_20260415.json" \
  --mode rl \
  --sft-steps 0 \
  --finetune-id "$FULL_QUERY_FINETUNE_ID" \
  --wandb-run-name "inspector-query-full-v2-rl-after-sft-20260415"

# Subset detect: use the existing best hybrid recipe.
run_and_log \
  "$LOG_DIR/subset_detect_hybrid.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_detect \
  --config "$CONFIG_DIR/train_inspector_detect_subset_balanced_1000_hybrid_v2_20260415.json" \
  --finetune-name "inspector-detect-subset1k-v2-hybrid-20260415" \
  --wandb-run-name "inspector-detect-subset1k-v2-hybrid-20260415"

# Subset query SFT-only.
run_and_log \
  "$LOG_DIR/subset_query_sft_only.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_subset_balanced_1000_v2_best_20260415.json" \
  --mode sft \
  --rl-steps 0 \
  --finetune-name "inspector-query-subset1k-v2-sft-only-20260415" \
  --wandb-run-name "inspector-query-subset1k-v2-sft-only-20260415"

# Subset query RL-only.
run_and_log \
  "$LOG_DIR/subset_query_rl_only.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_subset_balanced_1000_v2_best_20260415.json" \
  --mode rl \
  --sft-steps 0 \
  --finetune-name "inspector-query-subset1k-v2-rl-only-20260415" \
  --wandb-run-name "inspector-query-subset1k-v2-rl-only-20260415"

# Subset query combined, again with an explicit finetune-id handoff.
SUBSET_QUERY_SFT_LOG="$LOG_DIR/subset_query_combined_sft.log"
run_and_log \
  "$SUBSET_QUERY_SFT_LOG" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_subset_balanced_1000_v2_best_20260415.json" \
  --mode sft \
  --rl-steps 0 \
  --finetune-name "inspector-query-subset1k-v2-combined-sft-20260415" \
  --wandb-run-name "inspector-query-subset1k-v2-combined-sft-20260415"
SUBSET_QUERY_FINETUNE_ID="$(require_finetune_id "$SUBSET_QUERY_SFT_LOG")"

run_and_log \
  "$LOG_DIR/subset_query_combined_rl.log" \
  "$PYTHON_BIN" -m inspector_md.train_inspector_query \
  --config "$CONFIG_DIR/train_inspector_query_subset_balanced_1000_v2_best_20260415.json" \
  --mode rl \
  --sft-steps 0 \
  --finetune-id "$SUBSET_QUERY_FINETUNE_ID" \
  --wandb-run-name "inspector-query-subset1k-v2-combined-rl-after-sft-20260415"

echo "batch run completed successfully"
