# Datasets

`md_train_framework` accepts one dataset section across all skills.

## Sources

- `local_jsonl`: directory or file-backed JSONL splits
- `hf_hub`: a Hugging Face dataset name
- `hf_disk`: a Hugging Face dataset saved with `load_from_disk`
- `repo_dir`: a repo-local dataset directory

## Minimal Detect Example

```json
{
  "dataset": {
    "source": "local_jsonl",
    "path": "my_dataset",
    "image_root": "my_dataset/images",
    "train_split": "train",
    "val_split": "validation",
    "test_split": "test"
  }
}
```

## Common Fields

- `image_path`
- `class_name`
- `prompt`
- `boxes` for detect
- `point` or equivalent target for point
- `question` and target text fields for query

## Inspect Before Training

```bash
python -m md_train_framework inspect-dataset \
  --config md_train_framework/configs/detect_quickstart.json
```

This checks split counts, fields, sample previews, label counts, and missing image paths.
