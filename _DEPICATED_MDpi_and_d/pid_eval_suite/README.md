# P&ID OpenRouter Eval Suite

Config-driven multi-model benchmarks for P&ID icons using OpenRouter chat-completions.

## Defaults
- Dataset: `maxs-m87/pandid_dataset_v2`
- Split: `post_val`
- Skills: `detect` + `point`
- Models:
  - `anthropic/claude-3.7-sonnet`
  - `openai/gpt-5.1`
  - `qwen/qwen-2.5-vl-7b-instruct`

## Full Run (default config)
```bash
python pid_eval_suite/benchmark_openrouter_pid_icons.py \
  --config pid_eval_suite/configs/openrouter_pid_full.json
```

## Quick Run (capped samples)
```bash
python pid_eval_suite/benchmark_openrouter_pid_icons.py \
  --config pid_eval_suite/configs/openrouter_pid_quick.json
```

## Override Models From CLI
```bash
python pid_eval_suite/benchmark_openrouter_pid_icons.py \
  --config pid_eval_suite/configs/openrouter_pid_quick.json \
  --models-json '[
    {"label":"claude_frontier","model_id":"anthropic/claude-3.7-sonnet"},
    {"label":"chatgpt_frontier","model_id":"openai/gpt-5.1"},
    {"label":"qwen_vl_frontier","model_id":"qwen/qwen-2.5-vl-7b-instruct"}
  ]'
```

## Outputs
Run outputs are written under:
`outputs/openrouter_pid/openrouter_pid_<timestamp>/`

Each run writes:
- `run_manifest.json`
- `metrics_<skill>_<model_label_slug>.json`
- `predictions_<skill>_<model_label_slug>.jsonl`

## Notes On Runtime/Cost
A full run evaluates 3 models x 2 skills across full `post_val`, which is materially higher in latency and cost than quick runs. Use the quick config first to validate setup and output shape.
