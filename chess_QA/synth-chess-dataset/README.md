# Synth Chess QA Dataset Builder

Build local QA datasets from:
- `chess-dataset/labeled_originals` (FEN filename labels)
- `Chess Pieces.v23-raw.coco` (COCO bboxes)
- `osfstorage-archive` (JSON piece square + bbox labels)

The main v2 outputs now merge whichever v2-capable sources you provide:
- `dataset2_coco`
- `osfstorage_archive`
- dedupe exact `(piece, square)` duplicates
- drop impossible multi-piece-per-square boards
- use simple canonical prompts
- expand mixed-task data to one image -> four task rows

The builder can also create separate OSF-native v2 variants when explicitly requested that:
- use `osfstorage_archive` only
- preserve the source `train/val/test` split folders
- use the same simple canonical prompts and one-to-many mixed-task expansion

## Outputs

The builder creates:
- `piece_position_v2_dataset2`
- `mixed_tasks_v2_dataset2`

Optional outputs when explicitly enabled:
- `piece_position_v1`
- `mixed_tasks_v1`
- `piece_position_v2_osfstorage`
- `mixed_tasks_v2_osfstorage`

The main v2 row counts depend on which v2-capable input sources you provide to the builder.
If you pass both `dataset2_coco` and `osfstorage_archive`, the two main v2 datasets contain both.

`mixed_tasks_v1` uses:
- `list_all_pieces`
- `count_by_color`
- `list_color_pieces`
- `color_presence_check`

`list_all_pieces` and `list_color_pieces` output keyed square maps, for example:
`{"pieces":[{"black_bishop":"c8","black_king":"g8","black_pawn":["a7","b7"]}]}`

For COCO rows, square is approximated from bbox center.

For the main v2 outputs:
- `dataset2_coco` rows are split dynamically at `80/10/10` after cleaning
- `osfstorage_archive` rows keep native `train/val/test` splits
- the final `piece_position_v2_dataset2` and `mixed_tasks_v2_dataset2` outputs merge the
  available input sources by split

For OSF v2 outputs, split counts are taken directly from the archive's native
`train/val/test` folders.

## Run

From inside `chess_QA/`:

```bash
pip install -r synth-chess-dataset/requirements.txt
python synth-chess-dataset/build_chess_qa_dataset.py --seed 42
```

If you see `ModuleNotFoundError: No module named 'PIL'`, install `Pillow`.

Images are copied by default to:
- `synth-chess-dataset/outputs/imges`

If `chess-dataset/labeled_originals` is unavailable, the builder skips regenerating
the legacy v1 datasets and still writes any buildable v2 outputs.

If `osfstorage-archive` is unavailable, the builder skips the OSF variants and
still writes any other buildable datasets.

If you only want mixed-task outputs, pass `--mixed-task-only true` to skip writing
the piece-position datasets for each enabled build family.

## CLI arguments

- `--dataset1-dir` default `chess_QA/synth-chess-dataset/rawdatasets/chess-dataset/labeled_originals`
- `--dataset2-dir` default `chess_QA/synth-chess-dataset/rawdatasets/Chess Pieces.v23-raw.coco`
- `--osf-dir` default `chess_QA/synth-chess-dataset/rawdatasets/osfstorage-archive`
- `--output-dir` default `chess_QA/synth-chess-dataset/outputs`
- `--seed` default `42`
- `--copy-images` default `true`
- `--build-legacy-v1` default `false`
- `--build-source-specific-v2` default `false`
- `--mixed-task-only` default `false`
- `--piece-position-name` default `piece_position_v1`
- `--mixed-name` default `mixed_tasks_v1`
- `--piece-position-v2-name` default `piece_position_v2_dataset2`
- `--mixed-v2-name` default `mixed_tasks_v2_dataset2`
- `--piece-position-v2-osf-name` default `piece_position_v2_osfstorage`
- `--mixed-v2-osf-name` default `mixed_tasks_v2_osfstorage`
- `--export-hf-dataset` default `false`
- `--push-to-hub` default `false`
- `--hf-repo-id` default `""` (required when pushing)
- `--hf-token` default `""` (falls back to `HF_TOKEN`)
- `--hf-private` default `false`

## Hugging Face export/push

```bash
python synth-chess-dataset/build_chess_qa_dataset.py \
  --mixed-task-only true \
  --export-hf-dataset true \
  --push-to-hub true \
  --hf-repo-id your-user-or-org/chess-qa-synth \
  --hf-private true
```

## Output layout

Under `synth-chess-dataset/outputs/`:
- `imges/*`
- `piece_position_v2_dataset2/jsonl/train.jsonl`
- `piece_position_v2_dataset2/jsonl/val.jsonl`
- `piece_position_v2_dataset2/jsonl/test.jsonl`
- `piece_position_v2_dataset2/stats.json`
- `piece_position_v2_dataset2/metadata.json`
- `mixed_tasks_v2_dataset2/jsonl/train.jsonl`
- `mixed_tasks_v2_dataset2/jsonl/val.jsonl`
- `mixed_tasks_v2_dataset2/jsonl/test.jsonl`
- `mixed_tasks_v2_dataset2/stats.json`
- `mixed_tasks_v2_dataset2/metadata.json`

Optional when enabled:
- `piece_position_v1/jsonl/train.jsonl`
- `piece_position_v1/jsonl/val.jsonl`
- `piece_position_v1/jsonl/test.jsonl`
- `piece_position_v1/stats.json`
- `piece_position_v1/metadata.json`
- `mixed_tasks_v1/jsonl/train.jsonl`
- `mixed_tasks_v1/jsonl/val.jsonl`
- `mixed_tasks_v1/jsonl/test.jsonl`
- `mixed_tasks_v1/stats.json`
- `mixed_tasks_v1/metadata.json`
- `piece_position_v2_osfstorage/jsonl/train.jsonl`
- `piece_position_v2_osfstorage/jsonl/val.jsonl`
- `piece_position_v2_osfstorage/jsonl/test.jsonl`
- `piece_position_v2_osfstorage/stats.json`
- `piece_position_v2_osfstorage/metadata.json`
- `mixed_tasks_v2_osfstorage/jsonl/train.jsonl`
- `mixed_tasks_v2_osfstorage/jsonl/val.jsonl`
- `mixed_tasks_v2_osfstorage/jsonl/test.jsonl`
- `mixed_tasks_v2_osfstorage/stats.json`
- `mixed_tasks_v2_osfstorage/metadata.json`

Required JSONL fields:
- `row_id`
- `split`
- `task_type`
- `source_dataset`
- `image` (filename only)
- `image_path` (`/imges/<name>`)
- `question`
- `answer_text`
- `final_answer_json`
- `prompt_variant_id`
- `source_image_id`
- `source_label_format`

## Tests

```bash
pytest chess_QA/synth-chess-dataset/tests
```
