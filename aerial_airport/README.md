# Aerial Airport Point RL Pipeline

This module reads the local COCO export at `aerial_airport/raw_dataset/Aerial Airport.coco`, normalizes it into the repo's image-level point-training schema, optionally pushes that dataset to Hugging Face, and trains or benchmarks Moondream point-skill RL runs against it.

## Dataset Build

Default config: [build_aerial_airport_hf_dataset_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/build_aerial_airport_hf_dataset_default.json)

```bash
python aerial_airport/build_aerial_airport_hf_dataset.py \
  --config aerial_airport/configs/build_aerial_airport_hf_dataset_default.json
```

Behavior:

- Reads the local COCO export from `aerial_airport/raw_dataset/Aerial Airport.coco`.
- If the raw dataset only contains `train/`, auto-splits it deterministically into `train`, `validation`, and `test`.
- Preserves image-level rows, including empty/unlabeled images as `answer_boxes=[]`.
- Tops up each split toward the configured `target_empty_fraction` using deterministic synthetic background-negative crops.
- Writes a local HF `DatasetDict` plus `metadata.json` and `stats.json`.
- Optionally pushes `train`, `validation`, and `test` splits to HF.

Optional HF push:

```bash
python aerial_airport/build_aerial_airport_hf_dataset.py \
  --config aerial_airport/configs/build_aerial_airport_hf_dataset_default.json \
  --push-to-hub maxs-m87/aerial_airport_point_v2
```

## Training

Default config: [train_aerial_airport_point_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/train_aerial_airport_point_default.json)

```bash
python aerial_airport/train_aerial_airport_point.py \
  --config aerial_airport/configs/train_aerial_airport_point_default.json
```

Key behavior:

- Uses the shared point RL trainer with airport-specific defaults.
- Defaults to the local normalized dataset at `aerial_airport/outputs/maxs-m87_aerial_airport_point_v2`.
- Defaults to `skill=point`, `point_prompt_style=class_name`, and `reward_metric=f1`.
- Uses recall-gated point defaults tuned for single-class point supervision.
- Uses W&B project `moondream-aerial-airport-point-rl`.
- Uses `CICID_GPUB_MOONDREAM_API_KEY_1` by default; the round-2 recall grid rotates across keys `1` through `4`.

## Benchmark

Default config: [benchmark_aerial_airport_point_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/benchmark_aerial_airport_point_default.json)

```bash
python aerial_airport/benchmark_aerial_airport_point.py \
  --config aerial_airport/configs/benchmark_aerial_airport_point_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <CHECKPOINT_STEP>
```

This runs held-out evaluation on the dataset `test` split rather than adding trainer-level final-test logic.
The benchmark default config uses the local normalized dataset output and `CICID_GPUB_MOONDREAM_API_KEY_4`.

## Round-2 Recall Grid

- `cicd_train_aerial_airport_point_recall_lr1e4_r8.json`
- `cicd_train_aerial_airport_point_recall_lr5e4_r8.json`
- `cicd_train_aerial_airport_point_recall_lr1e4_r16.json`
- `cicd_train_aerial_airport_point_recall_lr5e4_r16.json`
