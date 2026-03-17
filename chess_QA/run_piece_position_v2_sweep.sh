#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
LOG_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$SCRIPT_DIR/outputs/sweeps/piece_position_v2_$LOG_STAMP"
PID_FILE="$LOG_DIR/pids.tsv"
STARTUP_STAGGER_S="${STARTUP_STAGGER_S:-0}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
fi
if [[ -z "$PYTHON_BIN" && -x "$PARENT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PARENT_ROOT/.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" && -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "error: no python interpreter found" >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
printf "run_name\tpid\tlog_path\n" >"$PID_FILE"
echo "using python: $PYTHON_BIN"

CONFIGS=(
  "query_rl_chess_piece_position_v2_baseline_reasoning.json"
  "query_rl_chess_piece_position_v2_offpolicy_reasoning.json"
  "query_rl_chess_piece_position_v2_offpolicy_reasoning_rank24_lowlr.json"
  "query_rl_chess_piece_position_v2_offpolicy_no_reasoning.json"
)

declare -a PIDS=()
declare -a NAMES=()

for config_name in "${CONFIGS[@]}"; do
  run_name="${config_name%.json}"
  config_path="$SCRIPT_DIR/configs/$config_name"
  log_path="$LOG_DIR/$run_name.log"

  "$PYTHON_BIN" "$SCRIPT_DIR/train_chess_query_rl.py" --config "$config_path" "$@" >"$log_path" 2>&1 &
  pid="$!"
  PIDS+=("$pid")
  NAMES+=("$run_name")

  printf "%s\t%s\t%s\n" "$run_name" "$pid" "$log_path" >>"$PID_FILE"
  echo "started $run_name pid=$pid log=$log_path"
  if [[ "$STARTUP_STAGGER_S" != "0" ]]; then
    sleep "$STARTUP_STAGGER_S"
  fi
done

status=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  run_name="${NAMES[$idx]}"
  if wait "$pid"; then
    echo "completed $run_name"
  else
    echo "failed $run_name"
    status=1
  fi
done

echo "sweep logs: $LOG_DIR"
exit "$status"
