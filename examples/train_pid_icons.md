# train_pid_icons.py

Lean Tuna example for class-conditional PI&D icon training. The same script supports both `detect` and `point` skills.

## Dataset

- Hugging Face dataset: https://huggingface.co/datasets/maxs-m87/pandid_dataset_v2
- Total samples: 30,000
- Total classes: 32
- Split sizes: train 24,000, val 5,400, post_val 600
- The script auto-discovers classes when `--class-names` is omitted and builds positive and negative class-conditional prompts from each image.

## Quick Start

The script reads `MOONDREAM_API_KEY`, `HF_TOKEN`, and `TUNA_BASE_URL` from the environment when available.

```bash
pip install -e .
python examples/train_pid_icons.py \
  --skill detect \
  --finetune-name "pid-icons-detect"
```

Point mode uses the same entrypoint:

```bash
python examples/train_pid_icons.py \
  --skill point \
  --point-prompt-style class_name \
  --finetune-name "pid-icons-point"
```

## Best Train Settings

Detect configuration from `MDpi_and_d/configs/train_pid_icons_default.json`:

```bash
--dataset-name maxs-m87/pandid_dataset_v2 \
--skill detect \
--point-prompt-style detect_phrase \
--rank 16 \
--num-steps 100 \
--batch-size 32 \
--group-size 6 \
--lr 0.002 \
--max-objects 50 \
--reward-metric f1 \
--augment-prob 0.5 \
--no-off-policy \
--fn-penalty-exponent 1.0 \
--fp-penalty-exponent 1.0 \
--neg-reward-weight 0.5 \
--pos-task-prob 0.95 \
--eval-every 50 \
--eval-max-samples 200 \
--save-every 20 \
```

Point recall / off-policy settings from the local recall stabilization runs:

```bash
--dataset-name maxs-m87/pandid_dataset_v2 \
--skill point \
--point-prompt-style detect_phrase \
--rank 16 \
--num-steps 200 \
--batch-size 16 \
--group-size 8 \
--lr 0.0005 \
--augment-prob 0.5 \
--off-policy \
--off-policy-std-thresh 0.02 \
--off-policy-max-reward 0.15 \
--off-policy-min-reward 0.15 \
--off-policy-reward-scale 2.0 \
--fn-penalty-exponent 2.0 \
--fp-penalty-exponent 1.0 \
--neg-reward-weight 0.15 \
--pos-task-prob 0.995 \
--eval-every 10 \
--eval-max-samples 400 \
--save-every 10 \
```

## Benchmark Notes

- Stored detect snapshot on `post_val`:
  - F1: 0.2434
  - Macro F1: 0.3347
  - mIoU: 0.2665
- Stored point snapshot on `post_val`:
  - F1: 0.4077
  - Macro F1: 0.4538
- An older artifact at `MDpi_and_d/outputs/bench_1/benchmark_pid_icons.json` reports F1 0.0478, macro F1 0.1483, and mIoU 0.1156, but that file does not label its skill. Treat it as a rough historical reference rather than a clean baseline.
- TODO: save paired baseline/final benchmark artifacts for both `detect` and `point`.
