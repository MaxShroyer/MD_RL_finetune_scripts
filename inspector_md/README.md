# inspector-md

`inspector_md/` is a repo-local staged Moondream pipeline for visible exterior and site inspection findings. It follows the same config-first/script-first layout used by the other task directories in this monorepo.

Runtime, benchmark, and query-training inference default to the staging API at `https://api-staging.moondream.ai/v1` and rotate requests across these four env vars:

- `CICID_GPUB_MOONDREAM_API_KEY_1`
- `CICID_GPUB_MOONDREAM_API_KEY_2`
- `CICID_GPUB_MOONDREAM_API_KEY_3`
- `CICID_GPUB_MOONDREAM_API_KEY_4`

## Build Datasets

Merge `MBDD2025`, `CUBIT-Det`, and `CODEBRIM` from `inspector_md/raw_datasets/` into a local normalized dataset tree:

```bash
python inspector_md/build_merged_synth_dataset.py \
  --config inspector_md/configs/build_merged_synth_dataset_default.json \
  --stage normalize
```

Write the builder-ready merged manifest with deterministic template synthesis only:

```bash
python inspector_md/build_merged_synth_dataset.py \
  --config inspector_md/configs/build_merged_synth_dataset_default.json \
  --stage all
```

Use OpenRouter for text synthesis by providing a teacher model id:

```bash
python inspector_md/build_merged_synth_dataset.py \
  --config inspector_md/configs/build_merged_synth_dataset_default.json \
  --stage all \
  --teacher-model-id openrouter/model-id
```

The merged dataset is written under `inspector_md/dataset/merged_synth_v1/` with:

- `assets/<source>/<split>/...`
- `base_records.jsonl`
- `synthetic_manifest.json`
- `provenance.jsonl`
- `openrouter_cache.jsonl`
- `stats.json`
- `mapping_summary.json`
- `build_summary.json`

Build the local detect, point, proposal-query, finding-query, and reasoning-hard datasets. The default config now points at `inspector_md/dataset/merged_synth_v1/synthetic_manifest.json` and expects OpenRouter-backed query-text refresh:

```bash
python inspector_md/build_inspector_dataset.py \
  --config inspector_md/configs/build_inspector_dataset_default.json
```

If you want to materialize the task datasets without OpenRouter refresh first, override the refresh mode explicitly:

```bash
python inspector_md/build_inspector_dataset.py \
  --config inspector_md/configs/build_inspector_dataset_default.json \
  --query-text-refresh-mode template_only
```

Build from a custom manifest instead of the built-in seed records:

```bash
python inspector_md/build_inspector_dataset.py \
  --config inspector_md/configs/build_inspector_dataset_default.json \
  --source-manifest /path/to/inspection_manifest.json
```

Use a real JSON-array manifest as the dataset source of truth. Each row must include:

```json
[
  {
    "row_id": "sample-001",
    "image_path": "/abs/path/to/image.jpg",
    "split": "train",
    "inspection_request": "Inspect this exterior image for visible issues.",
    "asset_context": "Two-story exterior elevation",
    "hard_example": false,
    "expected_proposals": [
      {
        "issue_code": "roof_cover_damage",
        "evidence": "Missing shingles are visible near the roof edge."
      }
    ],
    "expected_findings": [
      {
        "issue_code": "roof_cover_damage",
        "title": "Roof Cover Damage",
        "box": {
          "x_min": 0.08,
          "y_min": 0.05,
          "x_max": 0.72,
          "y_max": 0.34
        },
        "evidence": ["Missing shingles are visible near the roof edge."],
        "recommended_action": "Repair the damaged roof covering.",
        "cost_band": "high",
        "possible_compliance_issue": true,
        "insufficient_evidence": false,
        "compliance_note": "Possible compliance issue related to water_intrusion_risk.",
        "source_detect_labels": ["missing shingle"],
        "spatial_ref_index": 0
      }
    ]
  }
]
```

The builder validates issue codes, detect labels, boxes, and required fields before writing outputs. Start with a small manually reviewed manifest, then inspect:

- `inspector_md/outputs/source_manifest.normalized.json`
- `inspector_md/outputs/build_summary.json`
- `metadata.json` in each output directory
- `jsonl/` contents under the query issues output

Build training artifacts directly from the merged synthetic manifest:

```bash
python inspector_md/build_inspector_dataset.py \
  --config inspector_md/configs/build_inspector_dataset_default.json \
  --source-manifest inspector_md/dataset/merged_synth_v1/synthetic_manifest.json
```

Check finetuning readiness before launching any train or sweep run:

```bash
python inspector_md/check_inspector_finetune_readiness.py \
  --config inspector_md/configs/check_inspector_finetune_readiness_default.json
```

The readiness summary verifies:

- detect / point / query datasets exist and are non-empty
- query rows include `question`, `target_text`, and `final_answer_json`
- query refresh cache exists when `query_text_refresh_mode=openrouter`
- OpenRouter judge config resolves
- reasoning-hard rows are actually hard rows unless fallback was required

## Run The Pipeline

```bash
python inspector_md/run_inspector_pipeline.py \
  --config inspector_md/configs/run_inspector_pipeline_default.json \
  --image-path /path/to/image.jpg \
  --inspection-request "Inspect this building image for visible exterior and site issues." \
  --detect-finetune-id md_detect_ft \
  --query-finetune-id md_inspector_query_ft
```

Outputs:

- `inspector_md/outputs/reports/last_report.json`
- `inspector_md/outputs/reports/last_punch_list.txt`
- `inspector_md/outputs/reports/last_trace.json`

## Detect Training

SFT base:

```bash
python inspector_md/train_inspector_detect.py \
  --config inspector_md/configs/train_inspector_detect_sft_default.json
```

RL refine:

```bash
python inspector_md/train_inspector_detect.py \
  --config inspector_md/configs/train_inspector_detect_default.json
```

## Query Training

Finding-query default:

```bash
python inspector_md/train_inspector_query.py \
  --config inspector_md/configs/train_inspector_query_default.json
```

Whole-image proposal query:

```bash
python inspector_md/train_inspector_query.py \
  --config inspector_md/configs/train_inspector_query_proposal_default.json
```

Reasoning-only hard subset:

```bash
python inspector_md/train_inspector_query.py \
  --config inspector_md/configs/train_inspector_query_reasoning_hard.json
```

## Point Training

SFT base:

```bash
python inspector_md/train_inspector_point.py \
  --config inspector_md/configs/train_inspector_point_sft_default.json
```

RL refine:

```bash
python inspector_md/train_inspector_point.py \
  --config inspector_md/configs/train_inspector_point_default.json
```

## Sweep Runner

Dry-run the approved rank/lr/groups-per-step sweep manifest:

```bash
python inspector_md/run_inspector_sweep.py --dry-run
```

Write the sweep manifest only:

```bash
python inspector_md/run_inspector_sweep.py
```

Each generated run is assigned one staging key slot so the launched jobs spread evenly across the four configured keys. `run_inspector_sweep.py --launch` now enforces the readiness check before it starts any staged SFT -> RL runs.

## Benchmark

Benchmark the end-to-end pipeline against the normalized source manifest created by the builder:

```bash
python inspector_md/benchmark_inspector_pipeline.py \
  --config inspector_md/configs/benchmark_inspector_pipeline_default.json \
  --detect-finetune-id md_detect_ft \
  --query-finetune-id md_inspector_query_ft
```

## Report Builder

Aggregate benchmark JSON into JSON and Markdown reports:

```bash
python inspector_md/build_inspector_report.py \
  --input-jsons inspector_md/outputs/benchmarks/inspector_pipeline.metrics.json
```
