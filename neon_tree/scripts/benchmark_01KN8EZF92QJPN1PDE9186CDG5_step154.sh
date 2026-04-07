#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f ".env.staging" ]]; then
  set -a
  source ".env.staging"
  set +a
fi

FINETUNE_ID="${FINETUNE_ID:-01KN8EZF92QJPN1PDE9186CDG5}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-154}"
API_KEY_ENV_VAR="${API_KEY_ENV_VAR:-CICID_GPUB_MOONDREAM_API_KEY_4}"
BENCH_CONFIG="${BENCH_CONFIG:-neon_tree/configs/support_configs/benchmark_neon_tree_detect_default.json}"
FLYOVER_RUN_DIR="${FLYOVER_RUN_DIR:-neon_tree/outputs/flyovers/synthetic_flyover_20260402_180842}"
FLYOVER_MAX_SAMPLES="${FLYOVER_MAX_SAMPLES:-64}"
TRACKING_FRAME_RATE="${TRACKING_FRAME_RATE:-30}"
PRIMARY_VIZ_CLIP="${PRIMARY_VIZ_CLIP:-2019_YELL_2_541000_4977000_image_crop}"

RUN_TAG="${FINETUNE_ID}_step${CHECKPOINT_STEP}"
OUT_DIR="neon_tree/outputs/benchmarks/${RUN_TAG}"
FLYOVER_OUT_DIR="${OUT_DIR}/flyovers"

mkdir -p "$OUT_DIR" "$FLYOVER_OUT_DIR"

echo "benchmarking validation split for ${RUN_TAG}"
./.venv/bin/python -m neon_tree.benchmark_neon_tree_detect \
  --config "$BENCH_CONFIG" \
  --api-key-env-var "$API_KEY_ENV_VAR" \
  --finetune-id "$FINETUNE_ID" \
  --checkpoint-step "$CHECKPOINT_STEP" \
  --checkpoint-fallback-policy exact \
  --checkpoint-ready-max-wait-s 300 \
  --checkpoint-ready-poll-interval-s 5 \
  --output-json "${OUT_DIR}/validation.json" \
  --predictions-jsonl "${OUT_DIR}/validation_predictions.jsonl"

echo "benchmarking synthetic flyovers from ${FLYOVER_RUN_DIR}"
for manifest in "${FLYOVER_RUN_DIR}"/*/clip_manifest.json; do
  clip_dir="$(dirname "$manifest")"
  clip_id="$(basename "$clip_dir")"
  echo "  flyover ${clip_id}"
  ./.venv/bin/python -m neon_tree.benchmark_neon_tree_detect \
    --config "$BENCH_CONFIG" \
    --api-key-env-var "$API_KEY_ENV_VAR" \
    --dataset-source synthetic_flyover \
    --clip-manifest "$manifest" \
    --finetune-id "$FINETUNE_ID" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --checkpoint-fallback-policy exact \
    --checkpoint-ready-max-wait-s 300 \
    --checkpoint-ready-poll-interval-s 5 \
    --max-samples "$FLYOVER_MAX_SAMPLES" \
    --output-json "${FLYOVER_OUT_DIR}/${clip_id}_${FLYOVER_MAX_SAMPLES}.json"
done

primary_manifest="${FLYOVER_RUN_DIR}/${PRIMARY_VIZ_CLIP}/clip_manifest.json"
if [[ -f "$primary_manifest" ]]; then
  echo "rendering primary output video viz for ${PRIMARY_VIZ_CLIP}"
  ./.venv/bin/python -m neon_tree.benchmark_neon_tree_detect \
    --config "$BENCH_CONFIG" \
    --api-key-env-var "$API_KEY_ENV_VAR" \
    --dataset-source synthetic_flyover \
    --clip-manifest "$primary_manifest" \
    --finetune-id "$FINETUNE_ID" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --checkpoint-fallback-policy exact \
    --checkpoint-ready-max-wait-s 300 \
    --checkpoint-ready-poll-interval-s 5 \
    --max-samples "$FLYOVER_MAX_SAMPLES" \
    --tracking-output-jsonl "${OUT_DIR}/output_viz_tracks.jsonl" \
    --tracking-render-output "${OUT_DIR}/output_viz.mp4" \
    --tracking-frame-rate "$TRACKING_FRAME_RATE" \
    --output-json "${OUT_DIR}/output_viz.json"
else
  echo "warning: primary viz clip ${PRIMARY_VIZ_CLIP} not found under ${FLYOVER_RUN_DIR}" >&2
fi

for clip_id in \
  "2019_YELL_2_541000_4977000_image_crop" \
  "2018_NIWO_2_450000_4426000_image_crop"
do
  manifest="${FLYOVER_RUN_DIR}/${clip_id}/clip_manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "skipping tracked render for missing clip ${clip_id}" >&2
    continue
  fi
  echo "rendering tracked flyover ${clip_id}"
  ./.venv/bin/python -m neon_tree.benchmark_neon_tree_detect \
    --config "$BENCH_CONFIG" \
    --api-key-env-var "$API_KEY_ENV_VAR" \
    --dataset-source synthetic_flyover \
    --clip-manifest "$manifest" \
    --finetune-id "$FINETUNE_ID" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --checkpoint-fallback-policy exact \
    --checkpoint-ready-max-wait-s 300 \
    --checkpoint-ready-poll-interval-s 5 \
    --max-samples "$FLYOVER_MAX_SAMPLES" \
    --tracking-output-jsonl "${FLYOVER_OUT_DIR}/${clip_id}_${FLYOVER_MAX_SAMPLES}_tracks.jsonl" \
    --tracking-render-output "${FLYOVER_OUT_DIR}/${clip_id}_${FLYOVER_MAX_SAMPLES}_tracks.mp4" \
    --tracking-frame-rate "$TRACKING_FRAME_RATE" \
    --output-json "${FLYOVER_OUT_DIR}/${clip_id}_${FLYOVER_MAX_SAMPLES}_tracked.json"
done

echo "done"
echo "validation: ${OUT_DIR}/validation.json"
echo "flyovers:   ${FLYOVER_OUT_DIR}"
echo "video viz:  ${OUT_DIR}/output_viz.mp4"
