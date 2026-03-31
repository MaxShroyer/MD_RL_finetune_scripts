# train_construction_site_query_caption.py

Config-first ConstructionSite query RL for dense captions.

Output schema:

```json
{"caption":"..."}
```

## Quick Start

Build the local datasets first:

```bash
python construction_site/build_construction_site_hf_dataset.py \
  --config construction_site/configs/build_construction_site_hf_dataset_default.json
```

Train with the default config:

```bash
python examples/train_construction_site_query_caption.py \
  --config construction_site/configs/train_construction_site_query_caption_default.json
```

Recommended CICD config:

```bash
python examples/train_construction_site_query_caption.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_query_caption_control.json
```

