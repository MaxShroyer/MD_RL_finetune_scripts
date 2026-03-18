# Football Detect RL Pipeline

This module builds and trains a class-conditional football detection finetune from the raw HF dataset snapshot `maxs-m87/football_detect_no_split` at revision `6544f3946353a780683f2243b2a648c72ea5de17`.

The dataset is treated as a multi-class detect problem. Labels come from every `attributes[].value` where `attributes[].key == "element"` inside `answer_boxes`.

## Workflow

1. Generate deterministic stratified splits from the raw HF `train` split.
2. Train against the already split dataset from disk or from a pushed HF repo.

## Split Generation

Default config: [generate_football_splits_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/football_detect/configs/generate_football_splits_default.json)

Corrected v2 config: [generate_football_splits_v2.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/football_detect/configs/generate_football_splits_v2.json)

Run:

```bash
python football_detect/generate_football_splits.py \
  --config football_detect/configs/generate_football_splits_default.json
```

Behavior:

- Reads the raw HF dataset with `streaming=True`.
- Validates `notes` is one of `close`, `mid`, or `far` for every row.
- Preserves every source row.
- Allocates `val` rows with largest-remainder rounding inside the note buckets.
- Allocates `post_val` rows from the bucket-local `val` pool with largest-remainder rounding.
- Writes the split `DatasetDict` plus `metadata.json` and `stats.json`.

The v2 config additionally:

- Flattens each split into one positive row per `(image, box, element)` annotation.
- Preserves split assignment at the original source-row level before flattening.
- Stores `class_name`, `prompt`, `source_row_id`, `source_box_index`, `source_element_index`, and `task_schema`.
- Rewrites `line of scrimmage` into `offensive line / defensive line`.
- Adds one synthetic `offensive line / defensive line` union box when both `offensive line` and `defensive line` appear in the same source row.

Default output:

`football_detect/outputs/maxs-m87_football_detect_no_split_splits`

Corrected v2 output:

`football_detect/outputs/maxs-m87_football_detect_v2_splits`

Optional hub push:

```bash
python football_detect/generate_football_splits.py \
  --config football_detect/configs/generate_football_splits_default.json \
  --push-to-hub your-org/football-detect-splits
```

This pushes `train`, `validation`, and `test` by default.

## Training

Default config: [train_football_detect_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/football_detect/configs/train_football_detect_default.json)

Quick config: [train_football_detect_quick.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/football_detect/configs/train_football_detect_quick.json)

Run:

```bash
python football_detect/train_football_detect.py \
  --config football_detect/configs/train_football_detect_default.json
```

Key behavior:

- Requires a pre-split dataset with `train` plus a validation split.
- Discovers the class catalog from the `train` split by default.
- Uses default football prompts unless `prompt_overrides_json` changes them.
- Generates one positive task per labeled box and negative tasks from absent classes.
- Uses football defaults `neg_prompts_per_nonempty=1` and `neg_prompts_per_empty=0`.
- Applies State Farm-style augmentation per task sample, not per raw row.
- Uses W&B project `moondream-football-detect-rl` by default.

Useful overrides:

- `--dataset-path` or `--dataset-name`
- `--split`
- `--val-split`
- `--class-names-file`
- `--include-classes`
- `--exclude-classes`
- `--augment-prob`

Example class filtering:

```bash
python football_detect/train_football_detect.py \
  --config football_detect/configs/train_football_detect_default.json \
  --include-classes "ball holder" tackle
```

Example prompt overrides from CLI:

```bash
python football_detect/train_football_detect.py \
  --config football_detect/configs/train_football_detect_default.json \
  --prompt-overrides-json '{"ball holder":"player holding the football"}'
```

Default football prompts:

- `ball holder` -> `ball carrier`
- `area of focus` -> `main action area`
- `offensive line / defensive line` -> `offensive and defensive lines engaged after the snap`

The current default train-split catalog is expected to contain:

- `area of focus`
- `ball holder`
- `defensive line`
- `offensive line`
- `offensive line / defensive line`
- `players on the field`
- `tackle`
