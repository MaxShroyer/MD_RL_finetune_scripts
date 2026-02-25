# TicTacToe Query RL Training

This folder contains query-skill RL finetuning and benchmarking for the synthetic TicTacToe QA dataset.

## Files
- `train_ttt_query_rl.py`: query RL training loop using `QueryRequest` rollouts + `train_step`.
- `benchmark_ttt_query.py`: split benchmark runner against `/query`.
- `configs/query_rl_default.json`: default training config.
- `configs/query_rl_quick.json`: quick training config.
- `configs/benchmark_default.json`: default benchmark config.
- `.env.example`: required environment variable template.

## Setup
```bash
cd /Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo
cp tictaktoe_QA/.env.example tictaktoe_QA/.env
# edit tictaktoe_QA/.env with your keys
# optional for private HF access:
# HF_TOKEN=...
```

## Train
```bash
python tictaktoe_QA/train_ttt_query_rl.py \
  --config tictaktoe_QA/configs/query_rl_default.json
```

HF-only override example:
```bash
python tictaktoe_QA/train_ttt_query_rl.py \
  --config tictaktoe_QA/configs/query_rl_default.json \
  --dataset-source hf_hub \
  --hf-dataset-repo-id maxs-m87/tictactoe-qa-v1
```

Reasoning and sampling overrides:
```bash
python tictaktoe_QA/train_ttt_query_rl.py \
  --config tictaktoe_QA/configs/query_rl_default.json \
  --reasoning \
  --task-sampling-weights-json '{"best_move":3.5,"legal_moves_count":3.0,"legal_moves_list":4.0}' \
  --max-tokens-by-task-json '{"legal_moves_list":384}'
```

Local JSONL fallback:
```bash
python tictaktoe_QA/train_ttt_query_rl.py \
  --config tictaktoe_QA/configs/query_rl_default.json \
  --dataset-source local_jsonl \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v1
```

## Training config keys
- `dataset_source` (`hf_hub` or `local_jsonl`, default `hf_hub`).
- `hf_dataset_repo_id` / `hf_dataset_revision` / `hf_token` / `hf_cache_dir`: HF loader controls.
- `reasoning` (bool): toggles query reasoning mode (default `false`).
- `task_sampling_weights` (object): `task_type -> weight`, missing tasks default to `1.0`.
- `max_tokens_by_task` (object): optional `task_type -> max_tokens`; fallback is global `max_tokens`.
- `checkpoint_avg_splits` (list): splits to average during periodic eval/checkpoint ranking.
- `checkpoint_ranking_output` (path): JSON artifact with all ranked eval checkpoints.

Config precedence:
- CLI overrides config JSON.
- Config JSON overrides hardcoded defaults.

## Reward and parse semantics
- `best_move`:
  - reward `1.0` for canonical move
  - reward `best_move_optimal_reward` (default `0.7`) for any optimal-set non-canonical move
  - reward `0.0` otherwise
- Other tasks: exact semantic match on normalized JSON (`winner`, `is_terminal`, `has_winning_move`, `turn_player`, `legal_moves_count`, `legal_moves_list`).
- `eval_json_parse_rate` is schema-valid parse rate for the active task.
- `eval_json_object_rate` is raw JSON-object extraction rate (dict parsed, regardless of schema validity).

## Dataset source
Default is HF Hub (`maxs-m87/tictactoe-qa-v1`), so local JSONL is not required.

If you choose `--dataset-source local_jsonl`, `--dataset-dir` should point to a synth dataset output containing:
- `jsonl/train.jsonl`
- `jsonl/val.jsonl`
- optional eval splits in `jsonl/*.jsonl`
- `images/` for fallback image path resolution

## Checkpoint ranking output
At each periodic eval, the trainer evaluates `checkpoint_avg_splits`, computes average `eval_reward_mean`, and records it.
At train end, it writes the full ranking JSON (default `tictaktoe_QA/outputs/checkpoint_ranking.json`) and prints the highest-average checkpoint step.

## Benchmark
Use config-driven benchmark defaults:
```bash
python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json
```

Benchmark config/CLI supports:
- `dataset_source` (`hf_hub` default) plus HF args (`hf_dataset_repo_id`, `hf_dataset_revision`, etc.)
- `reasoning` via config or `--reasoning` / `--no-reasoning`
- `task_types` via config or `--task-types`
- existing model flags (`--finetune-id`, `--checkpoint-step`, `--max-tokens`, etc.)

Hard-task-only benchmark:
```bash
python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <STEP> \
  --task-types best_move,legal_moves_count,legal_moves_list
```

## Checkpoint comparison (`@129` vs `@148`)
```bash
python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step 129 \
  --output-json tictaktoe_QA/outputs/benchmark_test_metrics_step129.json \
  --predictions-jsonl tictaktoe_QA/outputs/benchmark_test_predictions_step129.jsonl

python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step 148 \
  --output-json tictaktoe_QA/outputs/benchmark_test_metrics_step148.json \
  --predictions-jsonl tictaktoe_QA/outputs/benchmark_test_predictions_step148.jsonl
```

## Reasoning A/B on hard tasks
Quick pass:
```bash
python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <STEP> \
  --task-types best_move,legal_moves_count,legal_moves_list \
  --max-samples 300 \
  --no-reasoning \
  --output-json tictaktoe_QA/outputs/benchmark_hardtasks_reasoning_false_quick.json

python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <STEP> \
  --task-types best_move,legal_moves_count,legal_moves_list \
  --max-samples 300 \
  --reasoning \
  --output-json tictaktoe_QA/outputs/benchmark_hardtasks_reasoning_true_quick.json
```

Full hard-task pass:
```bash
python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <STEP> \
  --task-types best_move,legal_moves_count,legal_moves_list \
  --no-reasoning \
  --output-json tictaktoe_QA/outputs/benchmark_hardtasks_reasoning_false_full.json

python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <STEP> \
  --task-types best_move,legal_moves_count,legal_moves_list \
  --reasoning \
  --output-json tictaktoe_QA/outputs/benchmark_hardtasks_reasoning_true_full.json
```
