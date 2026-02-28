# TicTacToe VLM QA Synth Dataset

This folder builds a synthetic tic-tac-toe QA dataset with:
- deduplicated PNG images (`768x768`)
- exact main split size (`train/val/test = 40k/5k/5k`)
- top-50 benchmark tracks (`canonical` + `paraphrase`, all 4 colorways)
- JSONL and HF `DatasetDict` exports, plus HF Hub upload

Benchmark note:
- `best_move`, `turn_player`, and available-move tasks use Cloudwalk top-50 states directly.
- `winner`, `is_game_over`, and `has_winning_move` inject probe states (non-top50) so labels are not degenerate.

## Install

```bash
cd /Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo
python -m venv .venv
source .venv/bin/activate
pip install -r tictaktoe_QA/synth_dataset/requirements.txt
```

## Build (full v2)

```bash
python tictaktoe_QA/synth_dataset/build_ttt_qa_dataset.py \
  --output-dir tictaktoe_QA/synth_dataset/outputs/v2 \
  --cache-dir tictaktoe_QA/synth_dataset/cache/cloudwalk \
  --hf-repo-id maxs-m87/tictactoe-qa-v2 \
  --seed 42 \
  --target-states 600 \
  --target-rows 10000
```

By default, the builder loads `tictaktoe_QA/.env` (`--env-file` can override).
For HF export (default), it writes local HF artifacts and uploads to Hub.
Set either:
- `--hf-repo-id ...` on CLI, or
- `HF_DATASET_REPO_ID=...` in `.env`

`--target-rows` is now configurable (default `50000`).
Example: `--target-rows 10000` yields an 80/10/10 split target (`8000/1000/1000`) with per-task quotas scaled proportionally.

Optional HF auth vars in `.env`:
- `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`)
- `OPENAI_API_KEY` (only used for optional rationale paraphrasing)

Optional rationale LLM paraphrase (train split only, ~20%) uses `OPENAI_API_KEY` and `gpt-4o-mini` by default.
Disable with `--no-llm`.

Use `--hf-private` to push to a private dataset repo.

If disk is tight, you can export JSONL only:

```bash
python tictaktoe_QA/synth_dataset/build_ttt_qa_dataset.py --skip-hf-export
```

## Smoke Build

```bash
python tictaktoe_QA/synth_dataset/build_ttt_qa_dataset.py \
  --no-llm \
  --no-network \
  --skip-hf-export \
  --max-main-rows 500 \
  --max-benchmark-rows 200
```

## Validate

```bash
python tictaktoe_QA/synth_dataset/validate_ttt_qa_dataset.py \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v2 \
  --cache-dir tictaktoe_QA/synth_dataset/cache/cloudwalk
```

Prediction scoring (optional):

```bash
python tictaktoe_QA/synth_dataset/validate_ttt_qa_dataset.py \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v2 \
  --predictions-jsonl path/to/predictions.jsonl
```

## Analyze

```bash
python tictaktoe_QA/synth_dataset/analyze_ttt_qa_dataset.py \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v2
```

Writes `analysis_report.json` in the dataset output directory and prints key warnings:
- split/task counts
- label coverage gaps
- symmetry-group overlap across train/val/test
- image/state reuse and parse-health checks

## Preview

```bash
python tictaktoe_QA/synth_dataset/preview_ttt_qa_dataset.py \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v2
```

Outputs:
- `preview/preview.md`
- `preview/summary.json`
- `preview/grid_*.png`

## Row Fields

Required row fields follow the public contract:
- `image`, `image_path`
- `split`, `task_type`, `question`, `answer_text`, `final_answer_json`, `messages_json`
- `state_key`, `symmetry_group`, `player_to_move`
- `winner_label`, `is_terminal`
- `legal_moves_json`
- `best_move_canonical_json`, `best_move_optimal_set_json`
- `depth_complexity`, `choice_complexity_num`, `choice_complexity_den`
- `colorway`, `augmentation_profile`, `prompt_variant_id`, `source_name`

Compatibility note:
- canonical task name for game-over classification is `is_game_over`.
- canonical task names are `available_moves_count` and `available_moves_list`.
- canonical answer key for game-over classification is `is_game_over`.
- canonical answer keys are `available_move_count` and `available_moves`.
- legacy names/keys (`is_terminal`, `legal_*`) are still accepted by train/benchmark/validation tooling.
