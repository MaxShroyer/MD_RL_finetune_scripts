# TicTacToe Query RL Training

This folder contains query-skill RL finetuning and benchmarking for the synthetic TicTacToe QA dataset.

## Files
- `train_ttt_query_rl.py`: query RL training loop using `QueryRequest` rollouts + `train_step`.
- `benchmark_ttt_query.py`: split benchmark runner against `/query`.
- `configs/query_rl_default.json`: default training config.
- `configs/query_rl_quick.json`: quick training config.
- `configs/query_rl_off_policy.json`: off-policy replay config.
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
  --task-sampling-weights-json '{"best_move":3.5,"available_moves_count":3.0,"available_moves_list":4.0}' \
  --max-tokens-by-task-json '{"available_moves_list":384}'
```

Off-policy replay config:
```bash
python tictaktoe_QA/train_ttt_query_rl.py \
  --config tictaktoe_QA/configs/query_rl_off_policy.json
```

Hyperparameter sweep (staged Optuna TPE):
```bash
pip install optuna
```

Run paths in this section assume you are in repo root:
`/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo`

If you run from inside `tictaktoe_QA/`, drop the `tictaktoe_QA/` prefix from script/config paths.

Dry-run (generates planned trial configs/artifacts only):
```bash
python tictaktoe_QA/sweep_ttt_query_optuna.py \
  --base-config tictaktoe_QA/configs/query_rl_off_policy.json \
  --sweep-config tictaktoe_QA/configs/query_rl_sweep_optuna.json \
  --study-name ttt_query_sweep_v1 \
  --dry-run
```

Normal sweep run:
```bash
python tictaktoe_QA/sweep_ttt_query_optuna.py \
  --base-config tictaktoe_QA/configs/query_rl_off_policy.json \
  --sweep-config tictaktoe_QA/configs/query_rl_sweep_optuna.json \
  --study-name ttt_query_sweep_v1
```

Resume an existing sweep:
```bash
python tictaktoe_QA/sweep_ttt_query_optuna.py \
  --base-config tictaktoe_QA/configs/query_rl_off_policy.json \
  --sweep-config tictaktoe_QA/configs/query_rl_sweep_optuna.json \
  --study-name ttt_query_sweep_v1 \
  --resume
```

Sweep artifacts are written under:
`tictaktoe_QA/outputs/sweeps/<study_name>/`

Key outputs:
- `optuna.db`: persistent Optuna study storage.
- `resolved_sweep_config.json`: fully resolved runtime sweep settings.
- `stage_a_results.json`, `stage_b_results.json`, `stage_c_results.json`: trial outcomes per stage.
- `final_confirmation_results.json`: multi-seed confirmation runs + aggregate ranking.
- `best_config.json`: selected winning config payload.
- `summary.json`: top-level run summary and artifact pointers.

Local JSONL fallback:
```bash
python tictaktoe_QA/train_ttt_query_rl.py \
  --config tictaktoe_QA/configs/query_rl_default.json \
  --dataset-source local_jsonl \
  --dataset-dir tictaktoe_QA/synth_dataset/outputs/v2
```

## Training config keys
- `dataset_source` (`hf_hub` or `local_jsonl`, default `hf_hub`).
- `hf_dataset_repo_id` / `hf_dataset_revision` / `hf_token` / `hf_cache_dir`: HF loader controls.
- `reasoning` (bool): toggles query reasoning mode (default `false`).
- `task_sampling_weights` (object): `task_type -> weight`, missing tasks default to `1.0`. Weights can be `0.0` to disable a task, but effective train-task weights must include at least one positive value.
- Eval/final-test also respect `task_sampling_weights`: rows whose `task_type` has weight `0.0` are excluded from eval scoring.
- `max_tokens_by_task` (object): optional `task_type -> max_tokens`; fallback is global `max_tokens`.
- `off_policy` (bool): enable replay-buffer off-policy group mixing during train updates.
- Do not combine `off_policy=true` and `reasoning=true`; trainer emits a warning because this setup is unstable.
- `off_policy_mix_ratio` (float in `[0,1]`): target fraction of train groups sampled from replay.
- `off_policy_buffer_size` / `off_policy_warmup_steps` / `off_policy_min_buffer_groups`: replay capacity and activation thresholds.
- `checkpoint_avg_splits` (list): splits to average during periodic eval/checkpoint ranking.
- `checkpoint_ranking_output` (path): JSON artifact with all ranked eval checkpoints.
- `auto_benchmark_best_checkpoint` (bool, default `true`): after training, run benchmark on the best ranked checkpoint.
- `auto_benchmark_config` (path): benchmark config used for automatic post-training benchmark launch.
- `auto_benchmark_output_json` / `auto_benchmark_predictions_jsonl` (paths): optional output overrides for the automatic benchmark artifacts.

Config precedence:
- CLI overrides config JSON.
- Config JSON overrides hardcoded defaults.

## Reward and parse semantics
- `best_move`:
  - if `scores_by_move_json` is present, reward is ranked: `1.0 - (# available moves strictly better than predicted)/(n-1)` (top-tier move gets `1.0`)
  - if `scores_by_move_json` is missing/invalid, fallback to legacy behavior:
    - reward `1.0` for canonical move
    - reward `best_move_optimal_reward` (default `0.7`) for any optimal-set non-canonical move
    - reward `0.0` otherwise
- Other tasks: exact semantic match on normalized JSON (`winner`, `is_game_over`, `has_winning_move`, `turn_player`, `available_moves_count`, `available_moves_list`).
- Compatibility: legacy task names (`is_terminal`, `legal_moves_count`, `legal_moves_list`) and legacy answer keys (`is_terminal`, `legal_move_count`, `legal_moves`) are still accepted and normalized.
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
  --task-types best_move,available_moves_count,available_moves_list
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
  --task-types best_move,available_moves_count,available_moves_list \
  --max-samples 300 \
  --no-reasoning \
  --output-json tictaktoe_QA/outputs/benchmark_hardtasks_reasoning_false_quick.json

python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <STEP> \
  --task-types best_move,available_moves_count,available_moves_list \
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
  --task-types best_move,available_moves_count,available_moves_list \
  --no-reasoning \
  --output-json tictaktoe_QA/outputs/benchmark_hardtasks_reasoning_false_full.json

python tictaktoe_QA/benchmark_ttt_query.py \
  --config tictaktoe_QA/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <STEP> \
  --task-types best_move,available_moves_count,available_moves_list \
  --reasoning \
  --output-json tictaktoe_QA/outputs/benchmark_hardtasks_reasoning_true_full.json
```
