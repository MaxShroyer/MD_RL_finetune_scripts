# train_chess_query_rl.py

Lean Tuna query-RL example for the public chess QA dataset, with a narrow default scope:

- dataset variant: `piece_position_v2_dataset2`
- task focus: `list_all_pieces`

## Dataset

- Hugging Face dataset: https://huggingface.co/datasets/maxs-m87/chess-qa-synth-v1
- The underlying builder can merge multiple source families into `piece_position_v2_dataset2`, so total row counts depend on which sources were present when the dataset was exported.
- A local piece-position training run for this variant recorded 4,540 train rows and 164 validation rows after filtering to the active task setup used by the run summary below.

## Quick Start

The script reads `MOONDREAM_API_KEY`, `HF_TOKEN`, and `TUNA_BASE_URL` from the environment when available.

```bash
pip install -e .
python examples/train_chess_query_rl.py \
  --dataset-variant-tag piece_position_v2_dataset2 \
  --finetune-name "chess-query-rl"
```

## Best Train Settings

```bash
--dataset-name maxs-m87/chess-qa-synth-v1 \
--dataset-variant-tag piece_position_v2_dataset2 \
--rank 32 \
--num-steps 300 \
--batch-size 4 \
--group-size 2 \
--lr 0.00005 \
--max-tokens 512 \
--off-policy \
--off-policy-mix-ratio 0.5 \
--off-policy-buffer-size 4096 \
--off-policy-warmup-steps 10 \
--off-policy-min-buffer-groups 64 \
--eval-every 10 \
--eval-max-samples 1000 \
--eval-batch-size 16 \
--save-every 10 \
```

These settings come from `chess_QA/configs/cicd/cicd_query_rl_chess_all_offpolicy_no_reasoning.json`.

## Benchmark Notes

- Local W&B summary snapshot for the piece-position experiment:
  - Baseline eval reward mean: 0.0695
  - Best average eval reward: 0.2654
  - Checkpoint val eval reward mean: 0.2047
  - Checkpoint test eval reward mean: 0.2479
  - Baseline JSON parse rate: 0.8125
  - Checkpoint val JSON parse rate: 1.0000
  - Checkpoint test JSON parse rate: 1.0000
- A second run of the same family reached best average eval reward 0.2459 at step 98.
- These values come from local W&B summaries rather than a standalone benchmark compare JSON, so treat them as experiment snapshots rather than a final public benchmark card.
- TODO: export a clean benchmark compare artifact for `piece_position_v2_dataset2` with the same task filter used by this example.
