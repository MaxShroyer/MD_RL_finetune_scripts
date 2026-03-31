#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-construction_site/.env.staging}"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [[ $# -ge 1 ]]; then
  ENV_FILE="$1"
fi
if [[ $# -ge 2 ]]; then
  SKIP_BUILD="$2"
fi

if [[ "$SKIP_BUILD" != "1" ]]; then
  python construction_site/build_construction_site_hf_dataset.py \
    --config construction_site/configs/build_construction_site_hf_dataset_default.json \
    --env-file "$ENV_FILE" \
    --skip-detect
fi

python construction_site/train_construction_site_detect.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_detect_control.json \
  --env-file "$ENV_FILE"

python construction_site/train_construction_site_detect.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_detect_offpolicy.json \
  --env-file "$ENV_FILE"

python construction_site/train_construction_site_query_caption.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_query_caption_control.json \
  --env-file "$ENV_FILE"

python construction_site/train_construction_site_query_caption.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_query_caption_offpolicy.json \
  --env-file "$ENV_FILE"

python construction_site/train_construction_site_query_rule_vqa.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_query_rule_vqa_control.json \
  --env-file "$ENV_FILE"

python construction_site/train_construction_site_query_rule_vqa.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_query_rule_vqa_offpolicy.json \
  --env-file "$ENV_FILE"
