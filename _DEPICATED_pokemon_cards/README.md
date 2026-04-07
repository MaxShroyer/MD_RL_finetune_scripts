# PokemonCards Reasoning Distillation

This suite builds a structured PokemonCards dataset, generates teacher rationale labels with OpenRouter, trains Moondream query finetunes in two stages, and benchmarks checkpoints on the derived splits.

## Run Order

```bash
python pokemon_cards/build_pokemon_cards_dataset.py \
  --output-dir pokemon_cards/outputs/thefusion21_pokemoncards_v1
```

```bash
python pokemon_cards/generate_pokemon_teacher_reasoning_openrouter.py \
  --config pokemon_cards/configs/teacher_openrouter_default.json \
  --teacher-model-id <OPENROUTER_MODEL_ID>
```

```bash
python pokemon_cards/train_pokemon_query_rl.py \
  --config pokemon_cards/configs/stage1_no_reasoning_cicd.json
```

```bash
python pokemon_cards/train_pokemon_query_rl.py \
  --config pokemon_cards/configs/stage2_reasoning_distill_cicd.json \
  --finetune-id <STAGE1_FINETUNE_ID>
```

```bash
python pokemon_cards/benchmark_pokemon_query.py \
  --config pokemon_cards/configs/benchmark_default.json \
  --finetune-id <FINETUNE_ID> \
  --checkpoint-step <STEP>
```
