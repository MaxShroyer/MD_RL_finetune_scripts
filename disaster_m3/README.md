# DisasterM3 Mixed-Skill Moondream Pipeline

This module builds a repo-local Moondream dataset under `disaster_m3/dataset/` from the local raw release under `disaster_m3/raw_dataset/`. Each built sample is a single composite RGB image with `left = pre-disaster` and `right = post-disaster`.

## Design choices

- Source of truth: the local `Kingdrone-Junjue/DisasterM3` release already downloaded into `disaster_m3/raw_dataset/`.
- Built dataset layout:
  - full dataset: `disaster_m3/dataset/full`
  - subset dataset: `disaster_m3/dataset/subset_1000`
- Prompt format: direct composite-image instruction only. The builder no longer inserts any `"If one panel is blank"` wording.
- Skill mapping:
  - `query`: recognition, reasoning, counting, descriptions, restoration advice
  - `point`: centroid-like localization
  - `detect`: box localization
- Unsupported coverage:
  - segmentation is skipped and counted in the dataset report
  - SAR and other non-RGB variants are skipped and counted
- Current local `DisasterM3_Instruct` release is effectively query-only after segmentation is excluded. The trainer and benchmark still keep mixed-skill support for future releases with localization rows.

## Dataset outputs

The builder writes:

- `jsonl/train.jsonl`, `jsonl/val.jsonl`, `jsonl/test.jsonl`
- `metadata.json`
- `build_stats.json`
- `dataset_report.json`
- `dataset_report.md`
- `task_samples.json`

`dataset_report.*` includes raw row totals, usable row totals, skip counts, counts and percentages by task/task family/skill, and representative GT-backed samples from each supported task.

## Training outputs

Each training stage writes:

- `eval_history.jsonl`
- `eval_predictions/*.jsonl`
- `eval_samples/*.jsonl`
- `<stage>_summary.json`

Eval history records include `overall`, `by_task`, and `by_skill` metrics. Saved eval samples include GT, model response, parsed prediction, and grading details.

## Commands

Build the full dataset:

```bash
python disaster_m3/build_disaster_m3_dataset.py \
  --output-dir disaster_m3/dataset/full \
  --max-samples 0 \
  --no-download
```

Build the 1000-sample subset:

```bash
python disaster_m3/build_disaster_m3_dataset.py \
  --config disaster_m3/configs/build_disaster_m3_dataset_subset_1000.json
```

Run one direct RL-only baseline:

```bash
python disaster_m3/train_disaster_m3_mixed.py \
  --config disaster_m3/configs/train_disaster_m3_mixed_rl_only.json
```

Run the full RL-only sweep manifest for both off-policy and reasoning sweeps:

```bash
python disaster_m3/run_disaster_m3_rl_sweep.py \
  --dataset-dir disaster_m3/dataset/full \
  --sweep both
```

Run the additional warmup-200 sweep family:

```bash
python disaster_m3/run_disaster_m3_rl_sweep.py \
  --dataset-dir disaster_m3/dataset/full \
  --sweep both \
  --training-regime warmup_200
```

Run both RL-only and warmup-200 families together:

```bash
python disaster_m3/run_disaster_m3_rl_sweep.py \
  --dataset-dir disaster_m3/dataset/full \
  --sweep both \
  --training-regime both
```

Dry-run the sweep and print commands without launching:

```bash
python disaster_m3/run_disaster_m3_rl_sweep.py \
  --dataset-dir disaster_m3/dataset/full \
  --sweep both \
  --dry-run
```

Benchmark one saved checkpoint:

```bash
python disaster_m3/benchmark_disaster_m3_mixed.py \
  --dataset-dir disaster_m3/dataset/full \
  --finetune-id <finetune_id> \
  --checkpoint-step <step> \
  --split test
```

## Sweep layout

The sweep launcher materializes 24 RL-only runs:

- Off-policy sweep: ranks `32, 24, 16` × learning rates `2e-4, 5e-4, 5e-5, 1e-5`
- Reasoning sweep: ranks `32, 24, 16` × learning rates `2e-4, 5e-4, 5e-5, 1e-5`

The warmup family adds another 24 runs with the same rank/lr grid, but each uses `mode=bootstrap_then_rl` with `bootstrap_steps=200` before the RL stage.

Off-policy and reasoning are mutually exclusive. Every sweep run uses:

- `batch_size=32`
- `group_size=8`
- `max_workers=4`
- `eval_every=10`
- `save_every=10`
- async checkpoint eval enabled
