# train_aerial_airport_point.py

Lean Tuna example for point-style airplane localization in overhead airport imagery.

## Dataset

- Hugging Face dataset: https://huggingface.co/datasets/maxs-m87/aerial_airport_point_v2
- Local stats for the exported dataset used by the benchmark: 371 images total.
- Split sizes: train 296, validation 38, test 37.
- Airplane box counts: train 4,018, validation 546, test 520.
- Empty-row fraction is about 10% on every split, so the example keeps negative images in the stream.

## Quick Start

The script reads `MOONDREAM_API_KEY`, `HF_TOKEN`, and `TUNA_BASE_URL` from the environment when available.

```bash
pip install -e .
python examples/train_aerial_airport_point.py \
  --finetune-name "aerial-airport-point" \
  --num-steps 200
```

## Best Train Settings

```bash
--dataset-name maxs-m87/aerial_airport_point_v2 \
--rank 16 \
--num-steps 120 \
--batch-size 16 \
--group-size 8 \
--lr 0.0005 \
--augment-prob 0.5 \
--off-policy \
--off-policy-std-thresh 0.01 \
--off-policy-max-reward 0.08 \
--off-policy-min-reward 0.08 \
--off-policy-reward-scale 1.25 \
--fn-penalty-exponent 2.0 \
--fp-penalty-exponent 2.0 \
--neg-reward-weight 0.5 \
--eval-every 5 \
--eval-max-samples 200 \
--save-every 5 \
```

These settings come from `aerial_airport/configs/cicd/cicd_train_aerial_airport_point_v2_recall_precision_offpolicy_light.json`.

## Benchmark Snapshot

- Baseline test F1: 0.3856
- Baseline test macro F1: 0.6494
- Fine-tuned test F1: 0.5574
- Fine-tuned test macro F1: 0.6726

These numbers come from `aerial_airport/outputs/benchmarks/benchmark_aerial_airport_point_best.json` and were evaluated on the same local export of `maxs-m87/aerial_airport_point_v2`.
