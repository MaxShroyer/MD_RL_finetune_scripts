# TicTacToe VLM QA Synth Dataset

This folder builds a synthetic tic-tac-toe QA dataset with:
- deduplicated PNG images (`768x768`)
- exact main split size (`train/val/test = 40k/5k/5k`)
- top-50 benchmark tracks (`canonical` + `paraphrase`, all 4 colorways)
- JSONL and HF `DatasetDict` exports

Benchmark note:
- `best_move`, `turn_player`, and legal-move tasks use Cloudwalk top-50 states directly.
- `winner`, `is_terminal`, and `has_winning_move` inject probe states (non-top50) so labels are not degenerate.

## Install

```bash
cd /Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo
python -m venv .venv
source .venv/bin/activate
pip install -r tictaktoe_QA/synth_dataset/requirements.txt
```

## Build (full v1)

```bash
python tictaktoe_QA/synth_dataset/build_ttt_qa_dataset.py \
  --output-dir tictaktoe_QA/synth_dataset/outputs/v1 \
  --cache-dir tictaktoe_QA/synth_dataset/cache/cloudwalk \
  --seed 42 \
  --target-states 3000 \
  --target-rows 50000
```

Optional rationale LLM paraphrase (train split only, ~20%) uses `OPENAI_API_KEY` and `gpt-4o-mini` by default.
Disable with `--no-llm`.

If disk is tight, you can export JSONL only:

```bash
python tictaktoe_QA/synth_dataset/build_ttt_qa_dataset.py --skip-hf-export
```

## Smoke Build

```bash
python tictaktoe_QA/synth_dataset/build_ttt_qa_dataset.py \
  --no-llm \
  --no-network \
  --max-main-rows 500 \
  --max-benchmark-rows 200
```

## Validate

```bash
python tictaktoe_QA/synth_dataset/validate_ttt_qa_dataset.py \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v1 \
  --cache-dir tictaktoe_QA/synth_dataset/cache/cloudwalk
```

Prediction scoring (optional):

```bash
python tictaktoe_QA/synth_dataset/validate_ttt_qa_dataset.py \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v1 \
  --predictions-jsonl path/to/predictions.jsonl
```

## Analyze

```bash
python tictaktoe_QA/synth_dataset/analyze_ttt_qa_dataset.py \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v1
```

Writes `analysis_report.json` in the dataset output directory and prints key warnings:
- split/task counts
- label coverage gaps
- symmetry-group overlap across train/val/test
- image/state reuse and parse-health checks

## Preview

```bash
python tictaktoe_QA/synth_dataset/preview_ttt_qa_dataset.py \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v1
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
