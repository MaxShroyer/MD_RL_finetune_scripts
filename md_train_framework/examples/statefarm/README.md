# Statefarm

Client-facing detect example for the Statefarm logo task.

Defaults:
- dataset: `maxs-m87/NBA_StateFarm_Splits_01`
- backend profile: `statefarm_detect`
- reward preset: `detect_miou`
- selection metric: `eval_miou`
- staging API: `https://api-staging.moondream.ai/v1`
- env file: repo-root `.env.staging`

Commands:

```bash
python -m md_train_framework.examples.statefarm.dataset_loader
python -m md_train_framework.examples.statefarm.train
python -m md_train_framework.examples.statefarm.eval --finetune-id 01KFYJ3T93RST3147ANRCJ8VA2 --checkpoint-step 139
python -m md_train_framework.examples.statefarm.sweep --plan-only
python -m md_train_framework.examples.statefarm.leaderboard
```

The example defaults are pinned to the public, reproducible checkpoint family rather than stronger but unanchored W&B-only runs.
The default config also mirrors the legacy training path more closely now by using the old concurrency, IoU threshold, and eval cadence, plus an uncapped validation pass instead of the earlier tiny fixed subset.
