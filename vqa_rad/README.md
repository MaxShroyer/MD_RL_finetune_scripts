# VQA-RAD Config-First Suite

This module builds a local normalized dataset from `flaviagiammarino/vqa-rad` and provides:

- a streaming dataset builder
- a config-driven Moondream query trainer
- a benchmark entrypoint for base and finetuned checkpoints

## Dataset Build

```bash
python vqa_rad/build_vqa_rad_hf_dataset.py \
  --config vqa_rad/configs/build_vqa_rad_hf_dataset_default.json
```

Outputs:

- Query dataset: `vqa_rad/outputs/vqa_rad_query_v1`
- Shared images: `vqa_rad/outputs/vqa_rad_query_images`

## Training

```bash
python vqa_rad/train_vqa_rad_query.py \
  --config vqa_rad/configs/train_vqa_rad_query_default.json
```

Comparison configs:

```bash
python vqa_rad/train_vqa_rad_query.py \
  --config vqa_rad/configs/cicd/cicd_train_vqa_rad_query_control.json

python vqa_rad/train_vqa_rad_query.py \
  --config vqa_rad/configs/cicd/cicd_train_vqa_rad_query_offpolicy_balanced.json

python vqa_rad/train_vqa_rad_query.py \
  --config vqa_rad/configs/cicd/cicd_train_vqa_rad_query_open_heavy.json
```

## Benchmark

```bash
python vqa_rad/benchmark_vqa_rad_query.py \
  --config vqa_rad/configs/benchmark_vqa_rad_query_default.json
```

Suggested run order:

1. Build the local dataset.
2. Benchmark the base model on the dataset.
3. Run the default trainer.
4. Compare the control, off-policy, and open-heavy configs.
5. Benchmark the selected checkpoint and inspect the predictions JSONL.

