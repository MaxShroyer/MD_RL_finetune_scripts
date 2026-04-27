#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_v2_query_salvage_common_20260416.sh"

MASTER_LOG="$RUN_ROOT/relaunch_query_salvage_20260416.nohup.log"
PID_LOG="$RUN_ROOT/relaunch_query_salvage_20260416.pids.txt"
KEY_1="CICID_GPUB_MOONDREAM_API_KEY_1"
KEY_2="CICID_GPUB_MOONDREAM_API_KEY_2"
KEY_3="CICID_GPUB_MOONDREAM_API_KEY_3"
KEY_4="CICID_GPUB_MOONDREAM_API_KEY_4"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "using python: $PYTHON_BIN"
echo "master log: $MASTER_LOG"
echo "lane logs dir: $LOG_DIR"
echo "reusing query dataset outputs under: $RUN_ROOT"

ensure_dataset_ready "$RUN_ROOT/full_dataset/inspector_query_issues_v2"
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
launch_detached_lane "full_query_salvage" "$KEY_1" "$SCRIPT_DIR/run_v2_lane_full_query_salvage_20260416.sh"
launch_detached_lane "subset_query_rl_salvage" "$KEY_2" "$SCRIPT_DIR/run_v2_lane_subset_query_rl_salvage_20260416.sh"
launch_detached_lane "subset_query_sft_only_salvage" "$KEY_3" "$SCRIPT_DIR/run_v2_lane_subset_query_sft_only_salvage_20260416.sh"
launch_detached_lane "subset_query_combined_salvage" "$KEY_4" "$SCRIPT_DIR/run_v2_lane_subset_query_combined_salvage_20260416.sh"

echo "launched all query salvage lanes"
echo "lane pid log: $PID_LOG"
