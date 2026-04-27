# BallHolder

Client-facing detect example for the BallHolder finetune path.

Defaults:
- dataset: `maxs-m87/Ball-Holder-splits-v1`
- backend profile: `ballholder_detect`
- reward preset: `detect_miou`
- selection metric: `eval_miou`
- staging API: `https://api-staging.moondream.ai/v1`
- env file: repo-root `.env.staging`

Commands:

```bash
python -m md_train_framework.examples.ballholder.dataset_loader
python -m md_train_framework.examples.ballholder.train
python -m md_train_framework.examples.ballholder.eval --finetune-id 01KHQ1H5BBP3G74PKFTER45F5Q --checkpoint-step 200
python -m md_train_framework.examples.ballholder.sweep --plan-only
python -m md_train_framework.examples.ballholder.leaderboard
```

This example keeps the surface thin. The client-facing eval prompt, legacy-style reward shaping, empty-frame sampling, and detect augmentation live in the reusable profile/config path, not in the wrapper scripts. The default config now also matches the legacy BallHolder ranking and eval cadence more closely by using `eval_miou`, `group_size=4`, `rank=8`, and the larger validation slice.
