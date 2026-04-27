# md_train_framework

## Overview

`md_train_framework` is a standalone training layer for Moondream finetunes. It gives you one JSON config format for `detect`, `point`, and `query`, plus local artifacts, compact WandB logging, baselines, leaderboards, comparisons, checkpoint-time eval capture, and sweep orchestration.

The quickstart below uses a tiny local detect dataset and the built-in mock backend. It works without an API key. To switch to a live Moondream finetune, replace `backend.id` with `generic_detect`, `generic_point`, or `generic_query` and set your API key env var in the config overrides.

## What You Need

- Python 3.10+
- Optional: `datasets`, `wandb`, `tuna-sdk`
- Optional for live query judge presets: `OPENROUTER_API_KEY`

## Quickstart

Inspect the starter dataset:

```bash
python -m md_train_framework inspect-dataset \
  --config md_train_framework/configs/detect_quickstart.json
```

Validate the config, reward preset, dataset, backend mapping, and API slots:

```bash
python -m md_train_framework dry-run \
  --config md_train_framework/configs/detect_quickstart.json
```

Run a baseline:

```bash
python -m md_train_framework baseline \
  --config md_train_framework/configs/detect_quickstart.json
```

Run training:

```bash
python -m md_train_framework train \
  --config md_train_framework/configs/detect_quickstart.json
```

Show the leaderboard:

```bash
python -m md_train_framework leaderboard \
  --config md_train_framework/configs/detect_quickstart.json
```

## What Gets Saved

- `run_config.json`
- `eval_history.jsonl`
- `train_summary.json`
- async checkpoint eval job outputs
- `benchmark_metrics.json`
- `predictions.jsonl`
- `failure_log.jsonl`
- local SQLite registry at `md_train_framework/outputs/runs.db`

## Next Steps

- Use [datasets.md](docs/datasets.md) to plug in local JSONL, HF hub, HF disk, or repo datasets.
- Use [rewards-and-metrics.md](docs/rewards-and-metrics.md) to pick reward presets and see which metrics log for each task family.
- Use [sweeps-and-recovery.md](docs/sweeps-and-recovery.md) for staged sweeps, staggered parallel runs, quarantine, and resume.
- For a query `sft_then_rl` shape, start from `md_train_framework/configs/query_sft_then_rl_example.json`.

## Showcase Examples

- `md_train_framework/examples/ballholder`: detect showcase with the BallHolder best-known defaults.
- `md_train_framework/examples/statefarm`: detect showcase for the Statefarm logo task.
- `md_train_framework/examples/pandid`: point-only SFT-to-RL showcase for class-conditional icon localization.
- `md_train_framework/examples/ttt_qa`: weighted multi-task query showcase for tic-tac-toe QA.
