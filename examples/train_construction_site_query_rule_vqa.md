# train_construction_site_query_rule_vqa.py

Config-first ConstructionSite query RL for safety-rule VQA.

Output schema:

```json
{"violated_rules":[1,4],"reasons":{"1":"...","4":"..."}}
```

## Quick Start

Build the local datasets first:

```bash
python construction_site/build_construction_site_hf_dataset.py \
  --config construction_site/configs/build_construction_site_hf_dataset_default.json
```

Train with the default config:

```bash
python examples/train_construction_site_query_rule_vqa.py \
  --config construction_site/configs/train_construction_site_query_rule_vqa_default.json
```

Recommended CICD config:

```bash
python examples/train_construction_site_query_rule_vqa.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_query_rule_vqa_offpolicy.json
```

