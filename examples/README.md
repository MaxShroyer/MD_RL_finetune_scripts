# Example Trainers

This directory contains lean fine-tuning examples prepared to match the standalone style used by `tuna-sdk-tmp/examples`.

- `train_aerial_airport_point.py`: Point-style airplane localization in overhead airport imagery.
  Doc: `train_aerial_airport_point.md`
- `train_bone_fracture_point.py`: Point-style bone fracture or abnormality localization.
  Doc: `train_bone_fracture_point.md`
- `train_bone_fracture_detect.py`: Fracture-only detection training on bone X-rays.
  Doc: `train_bone_fracture_detect.md`
- `train_pid_icons.py`: Class-conditional PI&D icon training in `detect` or `point` mode.
  Doc: `train_pid_icons.md`
- `train_ttt_query_rl.py`: Tic-tac-toe query RL on the hard QA tasks.
  Doc: `train_ttt_query_rl.md`
- `train_chess_query_rl.py`: Chess query RL for the public piece-position variant.
  Doc: `train_chess_query_rl.md`
- `train_construction_site_detect.py`: ConstructionSite detect training with config-first defaults.
  Doc: `train_construction_site_detect.md`
- `train_construction_site_query_caption.py`: ConstructionSite dense-caption query RL.
  Doc: `train_construction_site_query_caption.md`
- `train_construction_site_query_rule_vqa.py`: ConstructionSite safety-rule query RL.
  Doc: `train_construction_site_query_rule_vqa.md`

These files live in the local fallback `examples/` directory because the writable sandbox for this session does not include `/Users/maxs/Documents/Repos/MD/tuna-sdk-tmp`.
