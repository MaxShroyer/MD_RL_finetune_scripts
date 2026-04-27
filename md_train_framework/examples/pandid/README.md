# PandID

Client-facing point-only example for class-conditional icon localization.

Defaults:
- dataset: `maxs-m87/pandid_dataset_v2`
- backend profile: `pandid_point`
- reward preset: `point_recall_first`
- mode: `rl`
- prompt style: `class_name`
- staging API: `https://api-staging.moondream.ai/v1`
- env file: repo-root `.env.staging`

Commands:

```bash
python -m md_train_framework.examples.pandid.dataset_loader
python -m md_train_framework.examples.pandid.train
python -m md_train_framework.examples.pandid.eval --finetune-id 01KHWBR92JZZ2F4PG090PZMWKX --checkpoint-step 140
python -m md_train_framework.examples.pandid.sweep --plan-only
python -m md_train_framework.examples.pandid.leaderboard
```

The default config is now the apples-to-apples legacy parity path: RL-only, recall-first reward shaping, class-name prompts, and the higher-throughput eval settings from `train_pid_icons_best_point_recall_primary.json`. The profile module still expands each image row into class-conditional positive and negative point tasks so the wrapper scripts stay minimal.

If you still want the old bootstrap showcase, use `md_train_framework/examples/pandid/configs/sft_then_rl.json`. That config keeps the SFT box-target warm start followed by RL continuation:

```bash
python -m md_train_framework.examples.pandid.train --config md_train_framework/examples/pandid/configs/sft_then_rl.json
```

For multi-phase continuation runs, set `phases[1].finetune_id`, `phases[1].finetune_name`, or `phases[1].finetune_name_prefix` in the bootstrap config before the RL stage. The framework records both the initial and final finetune IDs in the run summary.
