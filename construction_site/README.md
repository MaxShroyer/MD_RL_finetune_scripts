# ConstructionSite Config-First Suite

This module builds local normalized datasets from `LouisChen15/ConstructionSite` and provides three config-driven training paths:

- detect grounding for 7 box targets
- query RL for dense captions
- query RL for safety-rule selection plus reasons

## Dataset Build

Default config:
[build_construction_site_hf_dataset_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/construction_site/configs/build_construction_site_hf_dataset_default.json)

```bash
python construction_site/build_construction_site_hf_dataset.py \
  --config construction_site/configs/build_construction_site_hf_dataset_default.json
```

Outputs:

- Detect dataset: `construction_site/outputs/construction_site_detect_v1`
- Caption query dataset: `construction_site/outputs/construction_site_query_caption_v1`
- Rule-VQA query dataset: `construction_site/outputs/construction_site_query_rule_vqa_v1`
- Shared query images: `construction_site/outputs/construction_site_query_images`

## Detect Training

Default config:
[train_construction_site_detect_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/construction_site/configs/train_construction_site_detect_default.json)

```bash
python construction_site/train_construction_site_detect.py \
  --config construction_site/configs/train_construction_site_detect_default.json
```

Recommended CICD config:

```bash
python construction_site/train_construction_site_detect.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_detect_offpolicy.json
```

## Caption Query Training

Default config:
[train_construction_site_query_caption_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/construction_site/configs/train_construction_site_query_caption_default.json)

```bash
python construction_site/train_construction_site_query_caption.py \
  --config construction_site/configs/train_construction_site_query_caption_default.json
```

## Rule-VQA Query Training

Default config:
[train_construction_site_query_rule_vqa_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/construction_site/configs/train_construction_site_query_rule_vqa_default.json)

```bash
python construction_site/train_construction_site_query_rule_vqa.py \
  --config construction_site/configs/train_construction_site_query_rule_vqa_default.json
```

## Benchmarks

Each benchmark entrypoint also defaults to a local JSON config:

- [benchmark_construction_site_detect_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/construction_site/configs/benchmark_construction_site_detect_default.json)
- [benchmark_construction_site_query_caption_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/construction_site/configs/benchmark_construction_site_query_caption_default.json)
- [benchmark_construction_site_query_rule_vqa_default.json](/Users/maxs/Documents/Repos/MD/MD_RL_Pipe/RL_amazon_logo/construction_site/configs/benchmark_construction_site_query_rule_vqa_default.json)
