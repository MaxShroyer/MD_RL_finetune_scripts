#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
LOG_STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/outputs/sweeps/cicd_$LOG_STAMP}"
PID_FILE="$LOG_DIR/pids.tsv"
CONFIG_DIR="$SCRIPT_DIR/configs/cicd"
STARTUP_STAGGER_S="${STARTUP_STAGGER_S:-2}"

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

DEFAULT_CONFIGS=(
  "cicd_query_rl_chess_all_baseline_no_reasoning_rewardfix.json"
  "cicd_query_rl_chess_all_baseline_reasoning_rewardfix_group4.json"
  "cicd_query_rl_chess_all_offpolicy_no_reasoning_safe_mix.json"
  "cicd_query_rl_chess_all_offpolicy_reasoning_rank24_safe_mix.json"
)

if [[ -n "${CICD_CONFIGS:-}" ]]; then
  read -r -a CONFIGS <<<"$CICD_CONFIGS"
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "error: no configs selected" >&2
  exit 1
fi

for config_name in "${CONFIGS[@]}"; do
  if [[ ! -f "$CONFIG_DIR/$config_name" ]]; then
    echo "error: missing config $CONFIG_DIR/$config_name" >&2
    exit 1
  fi
done

declare -a PIDS=()
declare -a NAMES=()

cleanup() {
  local pid
  trap - INT TERM
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
}

trap cleanup INT TERM

echo "log dir: $LOG_DIR"
echo "configs: ${CONFIGS[*]}"

for config_name in "${CONFIGS[@]}"; do
  run_name="${config_name%.json}"
  config_path="$CONFIG_DIR/$config_name"
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

trap - INT TERM
echo "sweep logs: $LOG_DIR"
exit "$status"
