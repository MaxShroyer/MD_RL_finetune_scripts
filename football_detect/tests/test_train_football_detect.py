from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_detect import common
from football_detect import train_football_detect as mod


FOOTBALL_CLASSES = [
    "area of focus",
    "ball holder",
    "defensive line",
    "offensive line",
    "offensive line / defensive line",
    "players on the field",
    "tackle",
]


def _box_tuple(box) -> tuple[float, float, float, float]:
    return (round(box.x_min, 4), round(box.y_min, 4), round(box.x_max, 4), round(box.y_max, 4))


def _sample_row(answer_boxes: object) -> dict[str, object]:
    return {
        "image": Image.new("RGB", (100, 100), color=(255, 255, 255)),
        "answer_boxes": answer_boxes,
        "source_collection": "football-unit-test",
    }


class ConfigPrecedenceTests(unittest.TestCase):
    def test_config_precedence_matches_repo_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "dataset_path": "football_detect/outputs/default_splits",
                        "val_split": "validation",
                        "group_size": 8,
                        "include_classes": ["ball holder"],
                        "prompt_overrides_json": {"ball holder": "player with ball"},
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--group-size",
                    "4",
                    "--include-classes",
                    "ball holder",
                    "tackle",
                    "--prompt-overrides-json",
                    '{"tackle":"tackle marker"}',
                ]
            )

        self.assertEqual(args.dataset_path, "football_detect/outputs/default_splits")
        self.assertEqual(args.val_split, "validation")
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.include_classes, ["ball holder", "tackle"])
        self.assertEqual(args.prompt_overrides, {"tackle": "tackle marker"})

    def test_api_key_env_var_is_accepted_from_config_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "env_file": ".env.staging",
                        "api_key_env_var": "MOONDREAM_API_KEY_1",
                        "base_url": "https://api-staging.moondream.ai/v1",
                        "off_policy": True,
                    }
                ),
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--config",
                    str(cfg_path),
                    "--api-key-env-var",
                    "MOONDREAM_API_KEY_4",
                ]
            )

        self.assertEqual(args.env_file, ".env.staging")
        self.assertEqual(args.base_url, "https://api-staging.moondream.ai/v1")
        self.assertEqual(args.api_key_env_var, "MOONDREAM_API_KEY_4")
        self.assertTrue(args.off_policy)

    def test_explicit_env_file_resolves_named_key_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"MOONDREAM_API_KEY": "generic_shell_key_should_not_win"},
            clear=False,
        ):
            env_path = Path(tmp) / ".env.staging"
            env_path.write_text(
                "MOONDREAM_API_KEY_3=file_key_from_named_env_var\n",
                encoding="utf-8",
            )

            args = mod.parse_args(
                [
                    "--env-file",
                    str(env_path),
                    "--api-key-env-var",
                    "MOONDREAM_API_KEY_3",
                    "--base-url",
                    "https://api-staging.moondream.ai/v1",
                ]
            )
            resolved = mod._resolve_runtime_env(args)

        self.assertEqual(resolved.api_key, "file_key_from_named_env_var")
        self.assertEqual(resolved.api_key_env_var, "MOONDREAM_API_KEY_3")
        self.assertEqual(resolved.base_url, "https://api-staging.moondream.ai/v1")

    def test_fallback_to_generic_api_key_when_named_key_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"MOONDREAM_API_KEY": "generic_shell_key_should_win"},
            clear=False,
        ):
            env_path = Path(tmp) / ".env.staging"
            env_path.write_text("UNRELATED_KEY=value\n", encoding="utf-8")

            args = mod.parse_args(
                [
                    "--env-file",
                    str(env_path),
                    "--api-key-env-var",
                    "MOONDREAM_API_KEY_4",
                ]
            )
            resolved = mod._resolve_runtime_env(args)

        self.assertEqual(resolved.api_key, "generic_shell_key_should_win")
        self.assertEqual(resolved.api_key_env_var, "MOONDREAM_API_KEY_4")

    def test_cicd_configs_parse_with_staging_settings(self) -> None:
        config_root = REPO_ROOT / "football_detect" / "configs" / "cicd"
        expectations = {
            "cicd_train_football_detect_onpolicy_lr5e5_r16.json": (False, 0.00005, 16, "MOONDREAM_API_KEY_1"),
            "cicd_train_football_detect_onpolicy_lr1e4_r16.json": (False, 0.0001, 16, "MOONDREAM_API_KEY_2"),
            "cicd_train_football_detect_offpolicy_lr5e5_r16.json": (True, 0.00005, 16, "MOONDREAM_API_KEY_3"),
            "cicd_train_football_detect_offpolicy_lr1e4_r16.json": (True, 0.0001, 16, "MOONDREAM_API_KEY_4"),
        }

        for filename, (off_policy, lr, rank, api_key_env_var) in expectations.items():
            with self.subTest(config=filename):
                args = mod.parse_args(["--config", str(config_root / filename)])

                self.assertEqual(args.env_file, ".env.staging")
                self.assertEqual(args.base_url, "https://api-staging.moondream.ai/v1")
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.dataset_name, "maxs-m87/football_detect_v2")
                self.assertEqual(args.off_policy, off_policy)
                self.assertEqual(args.lr, lr)
                self.assertEqual(args.rank, rank)
                self.assertFalse(args.reasoning)
                self.assertEqual(args.augment_prob, 0.5)
                self.assertEqual(args.num_steps, 300)
                self.assertEqual(args.batch_size, 4)
                self.assertEqual(args.group_size, 2)
                self.assertEqual(args.max_workers, 2)
                self.assertEqual(args.eval_every, 10)
                self.assertEqual(args.save_every, 10)
                self.assertEqual(args.eval_batch_size, 16)
                self.assertEqual(args.eval_max_samples, 200)


class ParsingAndCatalogTests(unittest.TestCase):
    def test_answer_boxes_parsing_extracts_normalized_boxes_and_classes(self) -> None:
        row = _sample_row(
            json.dumps(
                [
                    {
                        "x_min": 10,
                        "y_min": 20,
                        "x_max": 30,
                        "y_max": 60,
                        "attributes": [{"key": "element", "value": "ball holder"}],
                    },
                    {
                        "x": 50,
                        "y": 40,
                        "width": 20,
                        "height": 10,
                        "attributes": [{"key": "element", "value": ["tackle", "players on the field"]}],
                    },
                ]
            )
        )

        base = mod._to_base_sample(row)
        self.assertIsNotNone(base)
        assert base is not None
        parsed = sorted((item.class_name, _box_tuple(item.box)) for item in base.boxes)
        self.assertEqual(
            parsed,
            [
                ("ball holder", (0.1, 0.2, 0.3, 0.6)),
                ("players on the field", (0.4, 0.35, 0.6, 0.45)),
                ("tackle", (0.4, 0.35, 0.6, 0.45)),
            ],
        )

    def test_line_of_scrimmage_is_replaced_and_ol_dl_union_box_is_added(self) -> None:
        row = _sample_row(
            json.dumps(
                [
                    {
                        "x_min": 10,
                        "y_min": 10,
                        "x_max": 20,
                        "y_max": 20,
                        "attributes": [{"key": "element", "value": "offensive line"}],
                    },
                    {
                        "x_min": 30,
                        "y_min": 10,
                        "x_max": 40,
                        "y_max": 20,
                        "attributes": [{"key": "element", "value": "defensive line"}],
                    },
                    {
                        "x_min": 12,
                        "y_min": 30,
                        "x_max": 38,
                        "y_max": 40,
                        "attributes": [{"key": "element", "value": "line of scrimmage"}],
                    },
                ]
            )
        )

        base = mod._to_base_sample(row)
        self.assertIsNotNone(base)
        assert base is not None
        parsed = sorted((item.class_name, _box_tuple(item.box)) for item in base.boxes)
        self.assertEqual(
            parsed,
            [
                ("defensive line", (0.3, 0.1, 0.4, 0.2)),
                ("offensive line", (0.1, 0.1, 0.2, 0.2)),
                ("offensive line / defensive line", (0.1, 0.1, 0.4, 0.2)),
                ("offensive line / defensive line", (0.12, 0.3, 0.38, 0.4)),
            ],
        )

    def test_train_split_class_catalog_discovery_is_stable(self) -> None:
        raw_labels = [
            "area of focus",
            "ball holder",
            "defensive line",
            "line of scrimmage",
            "offensive line",
            "players on the field",
            "tackle",
        ]
        rows = [
            {"answer_boxes": json.dumps([{"attributes": [{"key": "element", "value": name}]}])}
            for name in reversed(raw_labels)
        ]
        rows.append(
            {
                "answer_boxes": json.dumps(
                    [
                        {"attributes": [{"key": "element", "value": "offensive line"}]},
                        {"attributes": [{"key": "element", "value": "defensive line"}]},
                    ]
                )
            }
        )
        rows.append({"answer_boxes": json.dumps([{"attributes": [{"key": "element", "value": "ball holder"}]}])})
        self.assertEqual(common.discover_class_names(rows), FOOTBALL_CLASSES)


class TaskGenerationTests(unittest.TestCase):
    def test_task_generation_produces_positive_and_negative_tasks(self) -> None:
        base = mod.BaseSample(
            image=Image.new("RGB", (64, 64), color=(255, 255, 255)),
            boxes=[
                mod.ClassBox(class_name="ball holder", box=mod._box_from_normalized(0.1, 0.1, 0.2, 0.2)),
                mod.ClassBox(class_name="tackle", box=mod._box_from_normalized(0.5, 0.5, 0.7, 0.8)),
            ],
            source="football-unit-test",
        )

        tasks = mod.tasks_from_base_sample(
            base,
            all_class_names=FOOTBALL_CLASSES,
            rng=random.Random(42),
            neg_prompts_per_empty=0,
            neg_prompts_per_nonempty=1,
            prompt_overrides={},
        )

        positive_classes = sorted(task.class_name for task in tasks if task.is_positive)
        negative_classes = [task.class_name for task in tasks if not task.is_positive]
        self.assertEqual(positive_classes, ["ball holder", "tackle"])
        self.assertEqual(len(negative_classes), 1)
        self.assertNotIn(negative_classes[0], {"ball holder", "tackle"})
        prompt_by_class = {task.class_name: task.prompt for task in tasks if task.is_positive}
        self.assertEqual(prompt_by_class["ball holder"], "ball carrier")
        self.assertEqual(prompt_by_class["tackle"], "tackle")
        self.assertTrue(all(len(task.gt_boxes) == (1 if task.is_positive else 0) for task in tasks))

    def test_task_generation_keeps_one_positive_task_per_box(self) -> None:
        base = mod.BaseSample(
            image=Image.new("RGB", (64, 64), color=(255, 255, 255)),
            boxes=[
                mod.ClassBox(class_name="ball holder", box=mod._box_from_normalized(0.1, 0.1, 0.2, 0.2)),
                mod.ClassBox(class_name="ball holder", box=mod._box_from_normalized(0.3, 0.3, 0.4, 0.4)),
                mod.ClassBox(class_name="tackle", box=mod._box_from_normalized(0.5, 0.5, 0.7, 0.8)),
            ],
            source="football-unit-test",
        )

        tasks = mod.tasks_from_base_sample(
            base,
            all_class_names=FOOTBALL_CLASSES,
            rng=random.Random(42),
            neg_prompts_per_empty=0,
            neg_prompts_per_nonempty=1,
            prompt_overrides={},
        )

        positives = [task for task in tasks if task.is_positive]
        self.assertEqual([task.class_name for task in positives], ["ball holder", "ball holder", "tackle"])
        self.assertTrue(all(len(task.gt_boxes) == 1 for task in positives))

    def test_flattened_row_uses_row_level_prompt(self) -> None:
        row = {
            "image": Image.new("RGB", (100, 100), color=(255, 255, 255)),
            "answer_boxes": json.dumps(
                [
                    {
                        "x_min": 20,
                        "y_min": 20,
                        "x_max": 40,
                        "y_max": 40,
                        "attributes": [{"key": "element", "value": "offensive line / defensive line"}],
                    }
                ]
            ),
            "class_name": "offensive line / defensive line",
            "prompt": "offensive and defensive lines engaged after the snap",
            "task_schema": "per_box_element",
            "source_collection": "football-unit-test",
        }

        base = mod._to_base_sample(row)
        self.assertIsNotNone(base)
        assert base is not None
        tasks = mod.tasks_from_base_sample(
            base,
            all_class_names=FOOTBALL_CLASSES,
            rng=random.Random(42),
            neg_prompts_per_empty=0,
            neg_prompts_per_nonempty=0,
            prompt_overrides={},
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].prompt, "offensive and defensive lines engaged after the snap")
        self.assertEqual(tasks[0].class_name, "offensive line / defensive line")
        self.assertEqual([_box_tuple(box) for box in tasks[0].gt_boxes], [(0.2, 0.2, 0.4, 0.4)])


class AugmentationTests(unittest.TestCase):
    def _augment_config(self) -> mod.AugmentConfig:
        return mod.AugmentConfig(
            flip_p=0.0,
            crop_p=1.0,
            crop_scale_min=1.0,
            crop_scale_max=1.0,
            resize_min=1.0,
            resize_max=1.0,
            stretch_p=0.0,
            stretch_min=1.0,
            stretch_max=1.0,
            color_p=0.0,
            brightness_min=1.0,
            brightness_max=1.0,
            contrast_min=1.0,
            contrast_max=1.0,
            saturation_min=1.0,
            saturation_max=1.0,
            hue_p=0.0,
            hue_delta_min=0.0,
            hue_delta_max=0.0,
            noise_p=0.0,
            noise_std_min=0.0,
            noise_std_max=0.0,
        )

    def test_flip_remaps_boxes_correctly(self) -> None:
        image = Image.new("RGB", (80, 60), color=(255, 255, 255))
        boxes = [mod._box_from_normalized(0.2, 0.1, 0.4, 0.3)]
        _, flipped = mod._horizontal_flip(image, boxes)
        self.assertEqual(_box_tuple(flipped[0]), (0.6, 0.1, 0.8, 0.3))

    def test_positive_crop_safeguard_restores_pre_crop_boxes(self) -> None:
        sample = mod.TaskSample(
            image=Image.new("RGB", (80, 60), color=(255, 255, 255)),
            prompt="ball holder",
            gt_boxes=[mod._box_from_normalized(0.2, 0.2, 0.4, 0.4)],
            class_name="ball holder",
            is_positive=True,
            source="football-unit-test",
        )
        with patch.object(
            mod,
            "_random_crop",
            return_value=(Image.new("RGB", (30, 30), color=(0, 0, 0)), []),
        ):
            augmented = mod.augment_task_sample(
                sample,
                random.Random(7),
                np.random.default_rng(7),
                self._augment_config(),
                augment_prob=1.0,
            )
        self.assertEqual([_box_tuple(box) for box in augmented.gt_boxes], [(0.2, 0.2, 0.4, 0.4)])

    def test_negative_tasks_can_remain_empty_after_crop(self) -> None:
        sample = mod.TaskSample(
            image=Image.new("RGB", (80, 60), color=(255, 255, 255)),
            prompt="defensive line",
            gt_boxes=[],
            class_name="defensive line",
            is_positive=False,
            source="football-unit-test",
        )
        with patch.object(
            mod,
            "_random_crop",
            return_value=(Image.new("RGB", (30, 30), color=(0, 0, 0)), []),
        ):
            augmented = mod.augment_task_sample(
                sample,
                random.Random(7),
                np.random.default_rng(7),
                self._augment_config(),
                augment_prob=1.0,
            )
        self.assertEqual(augmented.gt_boxes, [])
