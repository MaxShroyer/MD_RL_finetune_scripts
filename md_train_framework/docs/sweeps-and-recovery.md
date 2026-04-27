# Sweeps And Recovery

Sweeps are staged by default.

## Default Policy

1. Generate candidates from the task family defaults.
2. Run a short stage-1 screen.
3. Continue only the top configs.
4. Rank the winners from saved checkpoints and compare them to the baseline.

## Defaults

- Query sweeps focus on `lr`, `rank`, `num_rollouts`, `batch_size`, `off_policy`, and `reasoning`
- Detect and point sweeps focus on `lr`, `rank`, `batch_size`, `group_size`, `reward_metric`, and `off_policy`
- Eval stays deterministic with `eval_temperature=0.0`
- Parallel launches are staggered

## Run A Sweep Plan

```bash
python -m md_train_framework sweep \
  --config md_train_framework/configs/detect_quickstart.json \
  --plan-only
```

## Failure Handling

- failed runs are appended to the quarantine log
- resume candidates are appended to the resume queue
- one failed run does not stop the rest of the sweep
