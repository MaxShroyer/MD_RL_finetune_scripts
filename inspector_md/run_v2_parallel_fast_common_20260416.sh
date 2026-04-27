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
LOG_DIR="$RUN_ROOT/logs_parallel_fast_20260416"
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

ensure_dataset_ready() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "missing dataset directory: $path" >&2
    exit 1
  fi
}
