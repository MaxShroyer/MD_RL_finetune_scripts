# train_bone_fracture_point.py

Lean Tuna example for point-style localization of bone fractures or other visible abnormalities.

## Dataset

- Hugging Face dataset: https://huggingface.co/datasets/maxs-m87/bone_fracture_point_v1
- Local stats for the public default dataset: 458 images total.
- Split sizes: train 369, validation 45, test 44.
- Target box counts: train 490, validation 59, test 52.
- The example uses a single target phrase: `bone fracture or abnormality`.

## Quick Start

The script reads `MOONDREAM_API_KEY`, `HF_TOKEN`, and `TUNA_BASE_URL` from the environment when available.

```bash
pip install -e .
python examples/train_bone_fracture_point.py \
  --finetune-name "bone-fracture-point" \
  --num-steps 200
```

## Best Train Settings

```bash
--dataset-name maxs-m87/bone_fracture_point_v1 \
--rank 16 \
--num-steps 200 \
--batch-size 16 \
--group-size 8 \
--lr 0.00005 \
--augment-prob 0.5 \
--off-policy \
--off-policy-std-thresh 0.02 \
--off-policy-max-reward 0.15 \
--off-policy-min-reward 0.15 \
--off-policy-reward-scale 2.0 \
--fn-penalty-exponent 2.0 \
--fp-penalty-exponent 1.0 \
--neg-reward-weight 0.15 \
--eval-every 10 \
--eval-max-samples 200 \
--save-every 10 \
```

These settings come from `bone_fracture/configs/cicd/cicd_train_bone_fracture_point_recall_offpolicy.json`.

## Benchmark Notes

- The stored compare file in this repo targets `bone_fracture_point_angle_only_break_point_v1`, not the public default dataset used by this example.
- Current snapshot on that alternate dataset:
  - Baseline F1: 0.6957
  - Baseline macro F1: 0.7000
  - Fine-tuned F1: 0.6957
  - Fine-tuned macro F1: 0.7000
- TODO: run the same benchmark flow on `maxs-m87/bone_fracture_point_v1` so this example has an apples-to-apples baseline/final pair.
