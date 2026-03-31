# train_construction_site_detect.py

Config-first ConstructionSite detect training for 7 grounding classes:

- `excavator`
- `rebar`
- `worker_with_white_hard_hat`
- `rule_1_violation`
- `rule_2_violation`
- `rule_3_violation`
- `rule_4_violation`

## Quick Start

Build the local datasets first:

```bash
python construction_site/build_construction_site_hf_dataset.py \
  --config construction_site/configs/build_construction_site_hf_dataset_default.json
```

Train with the default config:

```bash
python examples/train_construction_site_detect.py \
  --config construction_site/configs/train_construction_site_detect_default.json
```

Recommended CICD config:

```bash
python examples/train_construction_site_detect.py \
  --config construction_site/configs/cicd/cicd_train_construction_site_detect_offpolicy.json
```

