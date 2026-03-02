# V2 Eval Suite

V2-only benchmark analysis and OpenRouter benchmarking tools for TicTacToe QA.

## Scripts
- `analyze_v2_benchmarks.py`: scans benchmark prediction artifacts, applies V2 filtering, computes best-move buckets, and writes plots/tables/report.
- `benchmark_openrouter_v2.py`: runs hard-task V2 benchmarks on OpenRouter chat-completions models.

## Analysis Usage
```bash
python tictaktoe_QA/v2_eval_suite/analyze_v2_benchmarks.py \
  --config tictaktoe_QA/v2_eval_suite/configs/analysis_v2_default.json
```

Optional include filter example:
```bash
python tictaktoe_QA/v2_eval_suite/analyze_v2_benchmarks.py \
  --config tictaktoe_QA/v2_eval_suite/configs/analysis_v2_default.json \
  --include-runs "benchmark_auto_*"
```

Outputs are written under:
`/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/tictaktoe_QA/v2_eval_suite/outputs/analysis/<timestamp>/`

## OpenRouter Benchmark Usage
Quick config:
```bash
python tictaktoe_QA/v2_eval_suite/benchmark_openrouter_v2.py \
  --config tictaktoe_QA/v2_eval_suite/configs/openrouter_v2_quick.json
```

Full config:
```bash
python tictaktoe_QA/v2_eval_suite/benchmark_openrouter_v2.py \
  --config tictaktoe_QA/v2_eval_suite/configs/openrouter_v2_full.json
```

Auth note:
- Set `OPENROUTER_API_KEY` (or pass `--api-key`) for OpenRouter runs.
- `OPENAI_API_KEY` is not used for OpenRouter auth.
- Model IDs are validated against OpenRouter `/models` before benchmarking starts.
- Invalid model IDs are skipped with a warning when at least one configured model is valid.

Optional model override from CLI:
```bash
python tictaktoe_QA/v2_eval_suite/benchmark_openrouter_v2.py \
  --config tictaktoe_QA/v2_eval_suite/configs/openrouter_v2_quick.json \
  --models-json '[{"label":"claude","model_id":"anthropic/claude-opus-latest"}]'
```

Extended-thinking run (for models that support it):
```bash
python tictaktoe_QA/v2_eval_suite/benchmark_openrouter_v2.py \
  --config tictaktoe_QA/v2_eval_suite/configs/openrouter_v2_quick.json \
  --models-json '[{"label":"claude_thinking","model_id":"anthropic/claude-3.7-sonnet:thinking"},{"label":"chatgpt_reasoning","model_id":"openai/gpt-5.1","request_overrides":{"reasoning":{"effort":"high"}}}]'
```

OpenRouter benchmark outputs are written under:
`/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/tictaktoe_QA/v2_eval_suite/outputs/openrouter/openrouter_v2_<timestamp>/`

## Buckets
Best-move bucket categories:
- `best_move`
- `second_best`
- `third_best`
- `fourth_plus`
- `invalid_move`
- `improper_response_format`
- `request_error`

Dense rank semantics match training score ordering (`value` primary, `depth` tie-break with faster wins preferred).
