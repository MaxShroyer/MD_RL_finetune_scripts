# train_bone_fracture_detect.py

Lean Tuna example for fracture-only detection on bone X-ray imagery.

## Dataset

- Hugging Face dataset: https://huggingface.co/datasets/maxs-m87/bone_fracture_detect_v1
- Local stats for the exported dataset: 601 images total.
- Split sizes: train 490, validation 60, test 51.
- Test-set label counts: fracture 29, line 13, messed_up_angle 8, angle 1.
- This example narrows rewards and prompts to the `fracture` class and uses the object name `fracture line`.

## Quick Start

The script reads `MOONDREAM_API_KEY`, `HF_TOKEN`, and `TUNA_BASE_URL` from the environment when available.

```bash
pip install -e .
python examples/train_bone_fracture_detect.py \
  --finetune-name "bone-fracture-detect" \
  --reward-metric miou
```

## Best Train Settings

```bash
--dataset-name maxs-m87/bone_fracture_detect_v1 \
--object-name "bone fracture" \
--rank 16 \
--num-steps 120 \
--batch-size 8 \
--group-size 4 \
--lr 0.0001 \
--max-objects 50 \
--reward-metric miou \
--augment-prob 0.5 \
--off-policy \
--off-policy-std-thresh 0.02 \
--off-policy-max-reward 0.15 \
--off-policy-min-reward 0.15 \
--off-policy-reward-scale 2.0 \
--fn-penalty-exponent 1.0 \
--fp-penalty-exponent 1.0 \
--neg-reward-weight 0.5 \
--eval-every 5 \
--eval-max-samples 200 \
--save-every 5 \
```

These settings come from `bone_fracture/configs/cicd/cicd_train_bone_fracture_detect_fracture_promptmix_offpolicy_anchor_miou.json`.

## Benchmark Snapshot

- Baseline test F1: 0.0000
- Baseline test macro F1: 0.0392
- Baseline test mIoU: 0.0508
- Fine-tuned test F1: 0.0000
- Fine-tuned test macro F1: 0.3529
- Fine-tuned test mIoU: 0.3529

These numbers come from `bone_fracture/outputs/benchmarks/benchmark_bone_fracture_detect_best.json` on the fracture-only test benchmark. In the stored candidate snapshot, true positives stayed at 0, but false positives dropped from 166 to 15, which is why the visible gain is in macro F1 and mIoU rather than micro F1.
