# PI&D Pipeline

## What this includes
- `build_pid_hf_dataset.py`: merges Dataset1 + PID2Graph (Complete + Patched), tags source metadata, and creates leakage-safe `train/val/post_val` splits.
- `train_pid_icons.py`: class-conditional RL training (`detect the "<class>"`) with per-class positives and random negative prompts.
- `benchmark_pid_icons.py`: class-conditional benchmark for baseline/candidate models.
- `notebooks/pid_icons_pipeline.ipynb`: end-to-end runnable notebook.

## Split leakage protection
Splits are assigned by `split_group_id` (group-level split), not by raw sample.
This prevents related variants (especially patched data) from crossing train/val/post_val.

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
  --split post_val
```
