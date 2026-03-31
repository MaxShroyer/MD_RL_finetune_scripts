# train_ttt_query_rl.py

Lean Tuna query-RL example for the hard public tic-tac-toe QA tasks:

- `best_move`
- `available_moves_count`
- `available_moves_list`

## Dataset

- Hugging Face dataset: https://huggingface.co/datasets/maxs-m87/tictactoe-qa-v1
- This repo does not include a matching local stats file for the public v1 dataset.
- The sibling synthetic builder README documents a v2 build target with `train/val/test = 40k/5k/5k`, but this example intentionally keeps the public v1 default so the script stays easy to run.

## Quick Start

The script reads `MOONDREAM_API_KEY`, `HF_TOKEN`, and `TUNA_BASE_URL` from the environment when available.

```bash
pip install -e .
python examples/train_ttt_query_rl.py \
  --finetune-name "ttt-query-rl" \
  --num-steps 200
```

## Best Train Settings

```bash
--dataset-name maxs-m87/tictactoe-qa-v1 \
--rank 16 \
--num-steps 200 \
--batch-size 16 \
--group-size 4 \
--lr 0.0005 \
--max-tokens 256 \
--reasoning \
--off-policy \
--off-policy-mix-ratio 0.5 \
--off-policy-buffer-size 4096 \
--off-policy-warmup-steps 10 \
--off-policy-min-buffer-groups 64 \
--eval-every 10 \
--eval-max-samples 200 \
--save-every 10 \
```

These settings come from `tictaktoe_QA/configs/query_rl_off_policy.json`.

## Benchmark Notes

- HF baseline snapshot in `tictaktoe_QA/outputs/benchmark_baseline_hf.json`:
  - Eval reward mean: 0.2548
  - JSON parse rate: 0.8910
  - Best-move set accuracy: 0.1091
  - Best-move canonical accuracy: 0.0909
  - Non-best exact accuracy: 0.2974
  - Evaluated rows: 1,000
- Local test snapshot in `tictaktoe_QA/outputs/benchmark_test_metrics.json`:
  - Eval reward mean: 0.4899
  - JSON parse rate: 0.9976
  - Best-move set accuracy: 0.0718
  - Best-move canonical accuracy: 0.0718
  - Non-best exact accuracy: 0.6080
  - Evaluated rows: 4,995
- These are not apples-to-apples. The HF baseline config points at `maxs-m87/tictactoe-qa-v2`, while this example defaults to `maxs-m87/tictactoe-qa-v1`.
- TODO: run matching v1 baseline and fine-tuned benchmarks with the same task mix and evaluation split.
