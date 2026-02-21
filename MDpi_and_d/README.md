# PI&D Pipeline

## What this includes
- `build_pid_hf_dataset.py`: merges Dataset1 + PID2Graph (Complete + Patched), tags source metadata, and creates leakage-safe `train/val/post_val` splits.
- `train_pid_icons.py`: class-conditional RL training (`detect the "<class>"`) with per-class positives and random negative prompts.
- `benchmark_pid_icons.py`: class-conditional benchmark for baseline/candidate models.
- `notebooks/pid_icons_pipeline.ipynb`: end-to-end runnable notebook.

## Split leakage protection
Splits are assigned by `split_group_id` (group-level split), not by raw sample.
This prevents related variants (especially patched data) from crossing train/val/post_val.

## Runtime config
- Most scripts load `.env` from `--env-file` and then resolve `MOONDREAM_API_KEY` / `HF_TOKEN`.
- `--base-url` / `--api-base` can be set explicitly, otherwise they fall back to `TUNA_BASE_URL`, `MOONDREAM_BASE_URL`, then the production default.

## Quick start
```bash
cd MDpi_and_d

python build_pid_hf_dataset.py \
  --dataset1-dir datasets/raw/Dataset1 \
  --pid2graph-dir datasets/raw/PID2Graph \
  --label-key datasets/raw/Dataset1/lable_key.txt \
  --output-dir outputs/pid_icons_merged

python train_pid_icons.py \
  --env-file .env \
  --dataset-path outputs/pid_icons_merged

python benchmark_pid_icons.py \
  --env-file .env \
  --dataset-path outputs/pid_icons_merged \
  --split post_val \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <CHECKPOINT_STEP>
```

## Point recall stabilization runs
Use `--skill point` plus the recall-first preset to reduce conservative collapse:

```bash
python train_pid_icons.py \
  --env-file .env \
  --dataset-path outputs/pandid_dataset_v2 \
  --split train \
  --val-split val \
  --skill point \
  --use-recall-first-preset \
  --num-steps 200 \
  --batch-size 16 \
  --group-size 8 \
  --eval-every 10 \
  --save-every 10 \
  --eval-max-samples 400
```

Prompt A/B for point mode:
- `--point-prompt-style detect_phrase` (default): uses `"<class> icon or icons"`.
- `--point-prompt-style class_name`: uses raw class name only.

Acceptance gates now logged in `wandb-summary.json`:
- `recall_gate_pass`: fails if TP drops more than `--recall-drop-threshold` at/after `--recall-gate-step`.
- `f1_target_pass`: passes when best `eval_f1 >= baseline_eval_f1 + --f1-improvement-target`.

Run the full control/recall/off-policy suite (plus optional prompt A/B):

```bash
python run_point_recall_suite.py --include-prompt-ab
```
