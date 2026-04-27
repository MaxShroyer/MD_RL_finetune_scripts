#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_v2_parallel_fast_common_20260416.sh"

MASTER_LOG="$RUN_ROOT/relaunch_parallel_fast_20260416.nohup.log"
PID_LOG="$RUN_ROOT/relaunch_parallel_fast_20260416.pids.txt"
KEY_1="CICID_GPUB_MOONDREAM_API_KEY_1"
KEY_2="CICID_GPUB_MOONDREAM_API_KEY_2"
KEY_3="CICID_GPUB_MOONDREAM_API_KEY_3"
KEY_4="CICID_GPUB_MOONDREAM_API_KEY_4"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "using python: $PYTHON_BIN"
echo "master log: $MASTER_LOG"
echo "lane logs dir: $LOG_DIR"
echo "reusing dataset outputs under: $RUN_ROOT"

ensure_dataset_ready "$RUN_ROOT/full_dataset/inspector_detect_v1"
ensure_dataset_ready "$RUN_ROOT/full_dataset/inspector_query_issues_v2"
ensure_dataset_ready "$RUN_ROOT/subset_balanced_1000/inspector_detect_v1"
ensure_dataset_ready "$RUN_ROOT/subset_balanced_1000/inspector_query_issues_v2"

launch_detached_lane() {
  local lane_name="$1"
  local key_env_var="$2"
  local lane_script="$3"
  nohup bash "$lane_script" "$key_env_var" >/dev/null 2>&1 &
  local lane_pid="$!"
  printf '%s\t%s\t%s\t%s\n' "$lane_name" "$lane_pid" "$key_env_var" "$lane_script" | tee -a "$PID_LOG"
}

: > "$PID_LOG"
launch_detached_lane "full_detect" "$KEY_1" "$SCRIPT_DIR/run_v2_lane_full_detect_fast_20260416.sh"
launch_detached_lane "full_query" "$KEY_2" "$SCRIPT_DIR/run_v2_lane_full_query_fast_20260416.sh"
launch_detached_lane "subset_detect_and_sft" "$KEY_3" "$SCRIPT_DIR/run_v2_lane_subset_detect_and_sft_fast_20260416.sh"
launch_detached_lane "subset_query_rl_and_combined" "$KEY_4" "$SCRIPT_DIR/run_v2_lane_subset_query_rl_and_combined_fast_20260416.sh"

echo "launched all lanes"
echo "lane pid log: $PID_LOG"
