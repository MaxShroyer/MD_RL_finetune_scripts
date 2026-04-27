# Aerial Airport / VisDrone RL Pipeline

`aerial_airport/` now defaults to frame-level VisDrone2019-VID detect and point training. The code stays in the same module, and the old airport COCO workflow is preserved under `aerial_airport/configs/legacy/`.

## Dataset Build

Default config: [build_aerial_airport_hf_dataset_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/build_aerial_airport_hf_dataset_default.json)

```bash
python aerial_airport/build_aerial_airport_hf_dataset.py \
  --config aerial_airport/configs/build_aerial_airport_hf_dataset_default.json
```

Default VisDrone build behavior:

- Reads the local raw tree at `aerial_airport/raw_dataset/VisDrone2019-VID-{train,val,test-dev}`.
- Builds one frame-level HF `DatasetDict` at `aerial_airport/outputs/maxs-m87_visdrone_vid_frames_car_van_merged_v2`.
- Maps splits directly to `train`, `validation`, and `test`.
- Keeps empty frames.
- Uses `frame_stride_train=5`, `frame_stride_validation=1`, and `frame_stride_test=1`.
- Uses a merged 9-class VisDrone taxonomy where raw `van` annotations are normalized into `car`.
- Does not synthesize background negatives by default.

Legacy airport build configs live under [configs/legacy](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/legacy) and explicitly use `source_format=airport_coco`.

## Training

Default point config: [train_aerial_airport_point_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/train_aerial_airport_point_default.json)

```bash
python aerial_airport/train_aerial_airport_point.py \
  --config aerial_airport/configs/train_aerial_airport_point_default.json
```

Default detect config: [train_aerial_airport_detect_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/train_aerial_airport_detect_default.json)

```bash
python aerial_airport/train_aerial_airport_detect.py \
  --config aerial_airport/configs/train_aerial_airport_detect_default.json
```

VisDrone-specific configs live under [configs/visdrone](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/visdrone):

- `visdrone_class_catalog.json` for the all-class general run.
- `train_aerial_airport_point_<class>.json` and `train_aerial_airport_detect_<class>.json` for isolated per-class runs.
- `train_aerial_airport_detect_control.json` for the non-tiling detect control recipe.

The local detect wrappers use plain class-name prompts for VisDrone categories rather than the shared icon-style phrasing.

## Benchmark

Default point benchmark: [benchmark_aerial_airport_point_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/benchmark_aerial_airport_point_default.json)

```bash
python aerial_airport/benchmark_aerial_airport_point.py \
  --config aerial_airport/configs/benchmark_aerial_airport_point_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <CHECKPOINT_STEP>
```

Default detect benchmark: [benchmark_aerial_airport_detect_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/benchmark_aerial_airport_detect_default.json)

```bash
python aerial_airport/benchmark_aerial_airport_detect.py \
  --config aerial_airport/configs/benchmark_aerial_airport_detect_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <CHECKPOINT_STEP>
```

The default benchmark configs process all classes from the merged VisDrone catalog and prefer the rebuilt local dataset path.
Per-class isolated benchmark configs remain in `aerial_airport/configs/visdrone/` for debugging.

## Legacy Airport

Airport-specific build, train, and benchmark configs are preserved in [configs/legacy](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/aerial_airport/configs/legacy).
