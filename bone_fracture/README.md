# Bone Fracture Point RL Pipeline

This module reads the local COCO export at `bone_fracture/raw_dataset/bone fracture.coco`, collapses every annotated abnormality into one point-training target named `bone fracture or abnormality`, optionally pushes the normalized dataset to Hugging Face, and trains Moondream point-skill RL runs against it.

The raw source labels `fracture`, `line`, `angle`, and `messed_up_angle` are preserved inside each box as `source_class_name`, but the training target is intentionally simplified to a single generic fracture/abnormality prompt.

## Dataset Build

Default config: [build_bone_fracture_hf_dataset_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/bone_fracture/configs/build_bone_fracture_hf_dataset_default.json)

Run:

```bash
python bone_fracture/build_bone_fracture_hf_dataset.py \
  --config bone_fracture/configs/build_bone_fracture_hf_dataset_default.json \
  --push-to-hub ""
```

Behavior:

- Uses the local COCO export by default and only falls back to Roboflow download logic if no local export is found.
- Builds one normalized row per image, not one row per box.
- Keeps empty images as negative rows with `answer_boxes=[]`.
- Collapses every positive box into the single class `bone fracture or abnormality`.
- Preserves the original Roboflow label in each box as `source_class_name`.
- Writes a local HF `DatasetDict` plus `metadata.json` and `stats.json`.

Default output:

- Local path: `bone_fracture/outputs/maxs-m87_bone_fracture_point_v1`
- HF dataset name: `maxs-m87/bone_fracture_point_v1`

## Point Training

Default config: [train_bone_fracture_point_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/bone_fracture/configs/train_bone_fracture_point_default.json)

Run:

```bash
python bone_fracture/train_bone_fracture_point.py \
  --config bone_fracture/configs/train_bone_fracture_point_default.json
```

Key behavior:

- Uses the shared point RL trainer with `skill=point`.
- Uses `point_prompt_style=class_name`, so the model is asked for `bone fracture or abnormality`.
- Uses staging API defaults and `.env.staging`.
- Uses W&B project `moondream-bone-fracture-point-rl`.
- Follows the recall-first point-tuning pattern used in the airport pipeline.

## Starter CICD Experiments

- `cicd_train_bone_fracture_point_control.json`
- `cicd_train_bone_fracture_point_recall_primary.json`
- `cicd_train_bone_fracture_point_recall_offpolicy.json`

API key rotation:

- `control` -> `MOONDREAM_API_KEY_1`
- `recall_primary` -> `MOONDREAM_API_KEY_2`
- `recall_offpolicy` -> `MOONDREAM_API_KEY_3`

Legacy detect assets are still present in this module, but the default workflow, docs, and notebook now target the point pipeline.
