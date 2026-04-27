# Rewards And Metrics

Reward choice is config-driven.

## Detect And Point Presets

- `detect_f1`
- `detect_miou`
- `point_f1`
- `point_recall_first`

## Query Presets

- `query_exact`
- `query_token_f1`
- `query_soft_hybrid`
- `query_judge_hybrid`
- `query_ranked_reward`

## Legacy Compatibility

Older task-specific preset ids are still accepted as aliases, but `list-presets` now shows only the canonical generic ids.

## Metric Policy

The framework only logs metrics that make sense for the current skill and reward family.

- Detect and point log F1, precision, recall, mIoU, and `tp/fp/fn`
- Query logs reward, parse rate, accuracy-style metrics, and judge metrics when the preset uses a judge
- `tn` is only emitted when the task defines negatives explicitly

List the registered presets:

```bash
python -m md_train_framework list-presets
```
