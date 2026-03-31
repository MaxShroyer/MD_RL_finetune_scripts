from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bone_fracture import common
from bone_fracture import train_bone_fracture_detect as mod


def _sample_row(answer_boxes: object) -> dict[str, object]:
    return {
        "image": Image.new("RGB", (100, 100), color=(255, 255, 255)),
        "answer_boxes": answer_boxes,
        "source_collection": "bone-fracture-unit-test",
    }


class BoneFractureConfigTests(unittest.TestCase):
    def test_cicd_configs_parse_with_expected_overrides(self) -> None:
        config_root = REPO_ROOT / "bone_fracture" / "configs" / "cicd"
        expectations = {
            "cicd_train_bone_fracture_detect_repaired_offpolicy_lr5e4_r8_g2_f1.json": (
                "f1",
                "f1",
                8,
                0.0005,
                [],
                {},
                True,
                4,
                2,
                300,
                10,
                10,
                0.0,
                0.0,
                1,
                "CICID_GPUB_MOONDREAM_API_KEY_1",
                "bone-fracture-staging-repaired-offpolicy-lr5e4-r8-g2-f1",
            ),
            "cicd_train_bone_fracture_detect_repaired_offpolicy_lr5e4_r8_g2_miou.json": (
                "miou",
                "miou",
                8,
                0.0005,
                [],
                {},
                True,
                4,
                2,
                300,
                10,
                10,
                0.0,
                0.0,
                1,
                "CICID_GPUB_MOONDREAM_API_KEY_2",
                "bone-fracture-staging-repaired-offpolicy-lr5e4-r8-g2-miou",
            ),
            "cicd_train_bone_fracture_detect_repaired_offpolicy_lr1e4_r16_g2_f1.json": (
                "f1",
                "f1",
                16,
                0.0001,
                [],
                {},
                True,
                4,
                2,
                300,
                10,
                10,
                0.0,
                0.0,
                1,
                "CICID_GPUB_MOONDREAM_API_KEY_3",
                "bone-fracture-staging-repaired-offpolicy-lr1e4-r16-g2-f1",
            ),
            "cicd_train_bone_fracture_detect_repaired_offpolicy_lr1e4_r16_g2_miou.json": (
                "miou",
                "miou",
                16,
                0.0001,
                [],
                {},
                True,
                4,
                2,
                300,
                10,
                10,
                0.0,
                0.0,
                1,
                "CICID_GPUB_MOONDREAM_API_KEY_4",
                "bone-fracture-staging-repaired-offpolicy-lr1e4-r16-g2-miou",
            ),
            "cicd_train_bone_fracture_detect_fracture_only_f1_klguard.json": (
                "f1",
                "f1",
                16,
                0.0001,
                ["fracture"],
                {"fracture": "bone fracture"},
                False,
                8,
                4,
                120,
                10,
                10,
                0.0,
                0.0,
                1,
                "CICID_GPUB_MOONDREAM_API_KEY_4",
                "bone-fracture-detect-fracture-only-f1-klguard",
            ),
        }

        for filename, (
            reward_metric,
            selection_metric,
            rank,
            lr,
            include_classes,
            prompt_overrides,
            off_policy,
            batch_size,
            group_size,
            num_steps,
            eval_every,
            save_every,
            kl_warning_threshold,
            kl_stop_threshold,
            kl_stop_consecutive,
            api_key_env_var,
            wandb_run_name,
        ) in expectations.items():
            with self.subTest(config=filename):
                args = mod.parse_args(["--config", str(config_root / filename)])

                self.assertEqual(args.env_file, ".env.staging")
                self.assertEqual(args.base_url, "https://api-staging.moondream.ai/v1")
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.dataset_name, "maxs-m87/bone_fracture_detect_v1")
                self.assertEqual(args.off_policy, off_policy)
                self.assertEqual(args.reward_metric, reward_metric)
                self.assertEqual(args.selection_metric, selection_metric)
                self.assertEqual(args.rank, rank)
                self.assertEqual(args.include_classes, include_classes)
                self.assertEqual(args.prompt_overrides, prompt_overrides)
                self.assertEqual(args.batch_size, batch_size)
                self.assertEqual(args.group_size, group_size)
                self.assertEqual(args.lr, lr)
                self.assertEqual(args.exclude_classes, [])
                self.assertFalse(args.reasoning)
                self.assertEqual(args.augment_prob, 0.5)
                self.assertEqual(args.num_steps, num_steps)
                self.assertEqual(args.max_workers, 2)
                self.assertEqual(args.eval_every, eval_every)
                self.assertEqual(args.save_every, save_every)
                self.assertEqual(args.eval_batch_size, 16)
                self.assertEqual(args.eval_max_samples, 200)
                self.assertEqual(args.kl_warning_threshold, kl_warning_threshold)
                self.assertEqual(args.kl_stop_threshold, kl_stop_threshold)
                self.assertEqual(args.kl_stop_consecutive, kl_stop_consecutive)
                self.assertTrue(args.run_final_test)
                self.assertEqual(args.wandb_run_name, wandb_run_name)

    def test_round2_detect_configs_parse_with_expected_overrides(self) -> None:
        config_root = REPO_ROOT / "bone_fracture" / "configs" / "cicd"
        expectations = {
            "cicd_train_bone_fracture_detect_fracture_promptmix_primary_anchor.json": (
                True,
                "f1",
                "f1",
                32,
                0.00001,
                64,
                4,
                500,
                5,
                5,
                1.0,
                1.0,
                0,
                1,
                0.5,
                "CICID_GPUB_MOONDREAM_API_KEY_1",
                "bone-fracture-detect-fracture-promptmix-primary-anchor",
            ),
            "cicd_train_bone_fracture_detect_fracture_promptmix_primary_anchor_fn_harsh.json": (
                True,
                "f1",
                "f1",
                32,
                0.00001,
                64,
                4,
                500,
                5,
                5,
                2.0,
                1.0,
                0,
                1,
                0.15,
                "CICID_GPUB_MOONDREAM_API_KEY_1",
                "bone-fracture-detect-fracture-promptmix-primary-anchor-fn-harsh",
            ),
            "cicd_train_bone_fracture_detect_fracture_promptmix_offpolicy_anchor_f1.json": (
                True,
                "f1",
                "f1",
                16,
                0.0001,
                8,
                4,
                120,
                5,
                5,
                1.0,
                1.0,
                0,
                1,
                0.5,
                "CICID_GPUB_MOONDREAM_API_KEY_2",
                "bone-fracture-detect-fracture-promptmix-offpolicy-anchor-f1",
            ),
            "cicd_train_bone_fracture_detect_fracture_promptmix_offpolicy_anchor_miou.json": (
                True,
                "miou",
                "miou",
                16,
                0.0001,
                8,
                4,
                120,
                5,
                5,
                1.0,
                1.0,
                0,
                1,
                0.5,
                "CICID_GPUB_MOONDREAM_API_KEY_3",
                "bone-fracture-detect-fracture-promptmix-offpolicy-anchor-miou",
            ),
            "cicd_train_bone_fracture_detect_fracture_promptmix_offpolicy_aggressive_f1.json": (
                True,
                "f1",
                "f1",
                8,
                0.0005,
                4,
                2,
                80,
                5,
                5,
                1.0,
                1.0,
                1,
                1,
                0.5,
                "CICID_GPUB_MOONDREAM_API_KEY_4",
                "bone-fracture-detect-fracture-promptmix-offpolicy-aggressive-f1",
            ),
        }

        for filename, (
            off_policy,
            reward_metric,
            selection_metric,
            rank,
            lr,
            batch_size,
            group_size,
            num_steps,
            eval_every,
            save_every,
            fn_penalty_exponent,
            fp_penalty_exponent,
            neg_prompts_per_empty,
            neg_prompts_per_nonempty,
            neg_reward_weight,
            api_key_env_var,
            wandb_run_name,
        ) in expectations.items():
            with self.subTest(config=filename):
                args = mod.parse_args(["--config", str(config_root / filename)])

                self.assertEqual(
                    args.dataset_path,
                    "bone_fracture/outputs/maxs-m87_bone_fracture_detect_v1_prompt_variants",
                )
                self.assertEqual(args.dataset_name, "maxs-m87/bone_fracture_detect_v1")
                self.assertEqual(args.include_classes, ["fracture"])
                self.assertEqual(args.prompt_overrides, {"fracture": "bone fracture"})
                self.assertEqual(args.off_policy, off_policy)
                self.assertEqual(args.reward_metric, reward_metric)
                self.assertEqual(args.selection_metric, selection_metric)
                self.assertEqual(args.rank, rank)
                self.assertEqual(args.lr, lr)
                self.assertEqual(args.batch_size, batch_size)
                self.assertEqual(args.group_size, group_size)
                self.assertEqual(args.num_steps, num_steps)
                self.assertEqual(args.eval_every, eval_every)
                self.assertEqual(args.save_every, save_every)
                self.assertEqual(args.fn_penalty_exponent, fn_penalty_exponent)
                self.assertEqual(args.fp_penalty_exponent, fp_penalty_exponent)
                self.assertEqual(args.neg_prompts_per_empty, neg_prompts_per_empty)
                self.assertEqual(args.neg_prompts_per_nonempty, neg_prompts_per_nonempty)
                self.assertEqual(args.neg_reward_weight, neg_reward_weight)
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.wandb_run_name, wandb_run_name)

    def test_default_config_uses_staging_env_and_first_staged_key(self) -> None:
        config_path = REPO_ROOT / "bone_fracture" / "configs" / "train_bone_fracture_detect_default.json"
        args = mod.parse_args(["--config", str(config_path)])
        self.assertEqual(args.env_file, ".env.staging")
        self.assertEqual(args.api_key_env_var, "CICID_GPUB_MOONDREAM_API_KEY_1")
        self.assertEqual(args.base_url, "https://api-staging.moondream.ai/v1")
        self.assertEqual(args.kl_warning_threshold, 0.0)
        self.assertEqual(args.kl_stop_threshold, 0.0)
        self.assertEqual(args.kl_stop_consecutive, 1)


class BoneFracturePromptTests(unittest.TestCase):
    def test_discover_class_names_reads_raw_annotation_labels(self) -> None:
        rows = [
            _sample_row(
                json.dumps(
                    [
                        {"attributes": [{"key": "element", "value": "messed_up_angle"}]},
                        {"attributes": [{"key": "element", "value": "fracture"}]},
                    ]
                )
            ),
            _sample_row(json.dumps([{"attributes": [{"key": "element", "value": "line"}]}])),
        ]

        self.assertEqual(common.discover_class_names(rows), ["fracture", "line", "messed_up_angle"])

    def test_task_generation_uses_bone_fracture_prompt_fallbacks(self) -> None:
        base = mod.BaseSample(
            image=Image.new("RGB", (64, 64), color=(255, 255, 255)),
            boxes=[mod.ClassBox(class_name="fracture", box=mod._box_from_normalized(0.1, 0.2, 0.3, 0.8))],
            source="bone-fracture-unit-test",
        )

        tasks = mod.tasks_from_base_sample(
            base,
            all_class_names=["fracture", "angle"],
            rng=random.Random(42),
            neg_prompts_per_empty=0,
            neg_prompts_per_nonempty=1,
            prompt_overrides={},
        )

        positive_prompts = {task.class_name: task.prompt for task in tasks if task.is_positive}
        negative_classes = [task.class_name for task in tasks if not task.is_positive]
        self.assertEqual(positive_prompts["fracture"], "fracture line")
        self.assertEqual(len(negative_classes), 1)
        self.assertEqual(negative_classes[0], "angle")
