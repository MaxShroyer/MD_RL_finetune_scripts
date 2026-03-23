from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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


def _sample_row_for_split(class_name: str, *, split_name: str) -> dict[str, object]:
    return {
        "image": Image.new("RGB", (100, 100), color=(255, 255, 255)),
        "answer_boxes": json.dumps(
            [
                {
                    "x_min": 10,
                    "y_min": 10,
                    "x_max": 30,
                    "y_max": 30,
                    "attributes": [{"key": "element", "value": class_name}],
                }
            ]
        ),
        "source_collection": split_name,
    }


def _eval_metrics(*, f1: float, miou: float, tasks: int = 1) -> dict[str, object]:
    return {
        "eval_tasks": tasks,
        "eval_f1": f1,
        "eval_f1_macro": f1,
        "eval_miou": miou,
        "eval_tp": 1 if tasks else 0,
        "eval_fp": 0,
        "eval_fn": 0,
        "eval_positive_tasks": tasks,
        "eval_positive_f1": f1,
        "eval_positive_f1_macro": f1,
        "eval_positive_miou": miou,
        "eval_positive_tp": 1 if tasks else 0,
        "eval_positive_fp": 0,
        "eval_positive_fn": 0,
        "eval_negative_tasks": 0,
        "eval_negative_f1": 0.0,
        "eval_negative_f1_macro": 0.0,
        "eval_negative_miou": 0.0,
        "eval_negative_tp": 0,
        "eval_negative_fp": 0,
        "eval_negative_fn": 0,
        "eval_class_task_counts": {"ball holder": tasks} if tasks else {},
        "eval_class_positive_task_counts": {"ball holder": tasks} if tasks else {},
        "eval_class_negative_task_counts": {},
        "eval_class_tp": {"ball holder": 1} if tasks else {},
        "eval_class_fp": {"ball holder": 0} if tasks else {},
        "eval_class_fn": {"ball holder": 0} if tasks else {},
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
            "cicd_train_football_detect_onpolicy_lr5e5_r16.json": (
                False,
                0.00005,
                16,
                "MOONDREAM_API_KEY_1",
                [],
                "football-staging-onpolicy-lr5e5-r16",
            ),
            "cicd_train_football_detect_onpolicy_lr1e4_r16.json": (
                False,
                0.0001,
                16,
                "MOONDREAM_API_KEY_2",
                [],
                "football-staging-onpolicy-lr1e4-r16",
            ),
            "cicd_train_football_detect_offpolicy_lr5e5_r16.json": (
                True,
                0.00005,
                16,
                "MOONDREAM_API_KEY_3",
                [],
                "football-staging-offpolicy-lr5e5-r16",
            ),
            "cicd_train_football_detect_offpolicy_lr1e4_r16.json": (
                True,
                0.0001,
                16,
                "MOONDREAM_API_KEY_4",
                [],
                "football-staging-offpolicy-lr1e4-r16",
            ),
            "cicd_train_football_detect_offpolicy_lr1e4_r16_area_of_focus.json": (
                True,
                0.0001,
                16,
                "MOONDREAM_API_KEY_4",
                ["area of focus"],
                "football-staging-offpolicy-lr1e4-r16-area-of-focus",
            ),
            "cicd_train_football_detect_offpolicy_lr1e4_r16_ball_holder.json": (
                True,
                0.0001,
                16,
                "MOONDREAM_API_KEY_4",
                ["ball holder"],
                "football-staging-offpolicy-lr1e4-r16-ball-holder",
            ),
            "cicd_train_football_detect_offpolicy_lr1e4_r16_defensive_line.json": (
                True,
                0.0001,
                16,
                "MOONDREAM_API_KEY_3",
                ["defensive line"],
                "football-staging-offpolicy-lr1e4-r16-defensive-line",
            ),
            "cicd_train_football_detect_offpolicy_lr1e4_r16_offensive_line.json": (
                True,
                0.0001,
                16,
                "MOONDREAM_API_KEY_1",
                ["offensive line"],
                "football-staging-offpolicy-lr1e4-r16-offensive-line",
            ),
            "cicd_train_football_detect_offpolicy_lr1e4_r16_offensive_line_defensive_line.json": (
                True,
                0.0001,
                16,
                "MOONDREAM_API_KEY_1",
                ["offensive line / defensive line"],
                "football-staging-offpolicy-lr1e4-r16-offensive-line-defensive-line",
            ),
            "cicd_train_football_detect_offpolicy_lr1e4_r16_players_on_the_field.json": (
                True,
                0.0001,
                16,
                "MOONDREAM_API_KEY_2",
                ["players on the field"],
                "football-staging-offpolicy-lr1e4-r16-players-on-the-field",
            ),
            "cicd_train_football_detect_offpolicy_lr1e4_r16_tackle.json": (
                True,
                0.0001,
                16,
                "MOONDREAM_API_KEY_3",
                ["tackle"],
                "football-staging-offpolicy-lr1e4-r16-tackle",
            ),
        }

        for filename, (off_policy, lr, rank, api_key_env_var, include_classes, wandb_run_name) in expectations.items():
            with self.subTest(config=filename):
                args = mod.parse_args(["--config", str(config_root / filename)])

                self.assertEqual(args.env_file, ".env.staging")
                self.assertEqual(args.base_url, "https://api-staging.moondream.ai/v1")
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.dataset_name, "maxs-m87/football_detect_v2")
                self.assertEqual(args.off_policy, off_policy)
                self.assertEqual(args.lr, lr)
                self.assertEqual(args.rank, rank)
                self.assertEqual(args.include_classes, include_classes)
                self.assertEqual(args.exclude_classes, [])
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
                self.assertEqual(args.wandb_run_name, wandb_run_name)

    def test_repaired_cicd_configs_parse_with_expected_overrides(self) -> None:
        config_root = REPO_ROOT / "football_detect" / "configs" / "cicd"
        expectations = {
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r16_g2_f1.json": ("f1", "f1", 16, 2, [], False),
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r16_g2_miou.json": ("miou", "miou", 16, 2, [], False),
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r32_g2_f1.json": ("f1", "f1", 32, 2, [], False),
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r16_g4_f1.json": ("f1", "f1", 16, 4, [], False),
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r16_players_on_the_field.json": (
                "f1",
                "f1",
                16,
                2,
                ["players on the field"],
                True,
            ),
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r16_tackle.json": (
                "f1",
                "f1",
                16,
                2,
                ["tackle"],
                True,
            ),
        }

        for filename, (reward_metric, selection_metric, rank, group_size, include_classes, tightened_isolated) in expectations.items():
            with self.subTest(config=filename):
                args = mod.parse_args(["--config", str(config_root / filename)])

                self.assertEqual(args.dataset_name, "maxs-m87/football_detect_v2")
                if filename == "cicd_train_football_detect_repaired_offpolicy_lr1e4_r32_g2_f1.json":
                    self.assertEqual(args.val_split, "val")
                    self.assertEqual(args.test_split, "post_val")
                else:
                    self.assertEqual(args.val_split, "validation")
                    self.assertEqual(args.test_split, "test")
                self.assertTrue(args.run_final_test)
                self.assertEqual(args.reward_metric, reward_metric)
                self.assertEqual(args.selection_metric, selection_metric)
                self.assertEqual(args.rank, rank)
                self.assertEqual(args.group_size, group_size)
                self.assertEqual(args.include_classes, include_classes)
                if tightened_isolated:
                    self.assertEqual(args.neg_prompts_per_empty, 1)
                    self.assertEqual(args.pos_task_prob, 0.7)
                    self.assertEqual(args.neg_reward_weight, 1.0)
                else:
                    self.assertEqual(args.neg_prompts_per_empty, 0)
                    self.assertEqual(args.pos_task_prob, 0.95)
                    self.assertEqual(args.neg_reward_weight, 0.5)

    def test_new_repaired_cicd_configs_parse_with_expected_overrides(self) -> None:
        config_root = REPO_ROOT / "football_detect" / "configs" / "cicd"
        expectations = {
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r32_g2_miou.json": {
                "dataset_path": "football_detect/outputs/maxs-m87_football_detect_v2_splits",
                "dataset_name": "maxs-m87/football_detect_v2",
                "val_split": "val",
                "test_split": "post_val",
                "reward_metric": "miou",
                "selection_metric": "miou",
                "rank": 32,
                "group_size": 2,
                "lr": 0.0001,
                "wandb_run_name": "football-staging-repaired-offpolicy-lr1e4-r32-g2-miou",
            },
            "cicd_train_football_detect_repaired_offpolicy_lr5e5_r32_g2_f1.json": {
                "dataset_path": "football_detect/outputs/maxs-m87_football_detect_v2_splits",
                "dataset_name": "maxs-m87/football_detect_v2",
                "val_split": "val",
                "test_split": "post_val",
                "reward_metric": "f1",
                "selection_metric": "f1",
                "rank": 32,
                "group_size": 2,
                "lr": 0.00005,
                "wandb_run_name": "football-staging-repaired-offpolicy-lr5e5-r32-g2-f1",
            },
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r32_g2_f1_promptfix.json": {
                "dataset_path": "football_detect/outputs/maxs-m87_football_detect_v2_splits",
                "dataset_name": "maxs-m87/football_detect_v2",
                "val_split": "val",
                "test_split": "post_val",
                "reward_metric": "f1",
                "selection_metric": "f1",
                "rank": 32,
                "group_size": 2,
                "lr": 0.0001,
                "prompt_overrides": {
                    "ball holder": "player holding the football",
                    "players on the field": "outline of all players on the field",
                    "tackle": "player tackling another player",
                    "offensive line": "offensive line players",
                    "defensive line": "defensive line players",
                    "offensive line / defensive line": "line of scrimmage",
                },
                "wandb_run_name": "football-staging-repaired-offpolicy-lr1e4-r32-g2-f1-promptfix",
            },
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r32_g2_f1_playfocus_groupbox.json": {
                "dataset_path": "football_detect/outputs/maxs-m87_football_detect_v2_splits",
                "dataset_name": "maxs-m87/football_detect_v2",
                "val_split": "val",
                "test_split": "post_val",
                "reward_metric": "f1",
                "selection_metric": "f1",
                "rank": 32,
                "group_size": 2,
                "lr": 0.0001,
                "prompt_overrides": {
                    "ball holder": "player holding the football",
                    "area of focus": "single tight bounding box around the ball carrier and the immediate nearby players involved in the play, such as likely tacklers or pass targets; do not outline the whole field",
                    "players on the field": "single bounding box enclosing all players visible on the field; do not return separate boxes for individual players",
                    "offensive line": "single bounding box enclosing the offensive line players; do not return separate boxes for individual players",
                    "defensive line": "single bounding box enclosing the defensive line players; do not return separate boxes for individual players",
                    "offensive line / defensive line": "single bounding box enclosing both the offensive and defensive lines engaged after the snap; do not return separate boxes for individual players",
                    "tackle": "player tackling another player",
                },
                "wandb_run_name": "football-staging-repaired-offpolicy-lr1e4-r32-g2-f1-playfocus-groupbox",
            },
            "cicd_train_football_detect_repaired_offpolicy_lr1e4_r32_g2_f1_negmix.json": {
                "dataset_path": "football_detect/outputs/maxs-m87_football_detect_v2_splits",
                "dataset_name": "maxs-m87/football_detect_v2",
                "val_split": "val",
                "test_split": "post_val",
                "reward_metric": "f1",
                "selection_metric": "f1",
                "rank": 32,
                "group_size": 2,
                "lr": 0.0001,
                "neg_prompts_per_empty": 1,
                "neg_prompts_per_nonempty": 1,
                "pos_task_prob": 0.85,
                "neg_reward_weight": 0.75,
                "wandb_run_name": "football-staging-repaired-offpolicy-lr1e4-r32-g2-f1-negmix",
            },
        }

        for filename, expected in expectations.items():
            with self.subTest(config=filename):
                args = mod.parse_args(["--config", str(config_root / filename)])

                self.assertEqual(args.dataset_path, expected["dataset_path"])
                self.assertEqual(args.dataset_name, expected["dataset_name"])
                self.assertEqual(args.val_split, expected["val_split"])
                self.assertEqual(args.test_split, expected["test_split"])
                self.assertTrue(args.run_final_test)
                self.assertEqual(args.reward_metric, expected["reward_metric"])
                self.assertEqual(args.selection_metric, expected["selection_metric"])
                self.assertEqual(args.rank, expected["rank"])
                self.assertEqual(args.group_size, expected["group_size"])
                self.assertEqual(args.lr, expected["lr"])
                self.assertEqual(args.wandb_run_name, expected["wandb_run_name"])
                if "prompt_overrides" in expected:
                    self.assertEqual(args.prompt_overrides, expected["prompt_overrides"])
                if "neg_prompts_per_empty" in expected:
                    self.assertEqual(args.neg_prompts_per_empty, expected["neg_prompts_per_empty"])
                    self.assertEqual(args.neg_prompts_per_nonempty, expected["neg_prompts_per_nonempty"])
                    self.assertEqual(args.pos_task_prob, expected["pos_task_prob"])
                    self.assertEqual(args.neg_reward_weight, expected["neg_reward_weight"])


class AuthHeaderTests(unittest.TestCase):
    def test_build_auth_headers_adds_accept_and_browser_user_agent_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            headers = mod._build_auth_headers("test-key")

        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Accept"], "application/json")
        self.assertEqual(headers["X-Moondream-Auth"], "test-key")
        self.assertIn("Mozilla/5.0", headers["User-Agent"])

    def test_build_auth_headers_respects_authorization_override(self) -> None:
        with patch.dict(os.environ, {"MOONDREAM_AUTH_HEADER": "Authorization"}, clear=True):
            headers = mod._build_auth_headers("test-key")

        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertNotIn("X-Moondream-Auth", headers)


class PromptTests(unittest.TestCase):
    def test_players_on_the_field_default_prompt_requests_outline(self) -> None:
        self.assertEqual(mod.default_prompt_for_class("players on the field"), "outline of all players on the field")


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

    def test_task_generation_filters_positive_tasks_to_active_classes(self) -> None:
        base = mod.BaseSample(
            image=Image.new("RGB", (64, 64), color=(255, 255, 255)),
            boxes=[
                mod.ClassBox(class_name="ball holder", box=mod._box_from_normalized(0.1, 0.1, 0.2, 0.2)),
                mod.ClassBox(class_name="players on the field", box=mod._box_from_normalized(0.5, 0.5, 0.7, 0.8)),
            ],
            source="football-unit-test",
        )

        tasks = mod.tasks_from_base_sample(
            base,
            all_class_names=["players on the field"],
            rng=random.Random(42),
            neg_prompts_per_empty=1,
            neg_prompts_per_nonempty=1,
            prompt_overrides={},
        )

        self.assertEqual([(task.class_name, task.is_positive) for task in tasks], [("players on the field", True)])

    def test_task_generation_turns_other_class_rows_into_negative_opportunities(self) -> None:
        base = mod.BaseSample(
            image=Image.new("RGB", (64, 64), color=(255, 255, 255)),
            boxes=[mod.ClassBox(class_name="ball holder", box=mod._box_from_normalized(0.1, 0.1, 0.2, 0.2))],
            source="football-unit-test",
        )

        tasks = mod.tasks_from_base_sample(
            base,
            all_class_names=["players on the field"],
            rng=random.Random(42),
            neg_prompts_per_empty=1,
            neg_prompts_per_nonempty=1,
            prompt_overrides={},
        )

        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0].is_positive)
        self.assertEqual(tasks[0].class_name, "players on the field")
        self.assertEqual(tasks[0].gt_boxes, [])

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


class ApiEvalTests(unittest.TestCase):
    def test_evaluate_api_raises_when_all_detect_calls_fail(self) -> None:
        with patch.object(mod, "_call_detect_api", side_effect=RuntimeError("HTTP 403 from /detect")):
            with self.assertRaises(mod._DetectEvalError) as ctx:
                mod._evaluate_api(
                    model="moondream3-preview/ft_test@10",
                    eval_rows=[_sample_row_for_split("ball holder", split_name="test")],
                    all_class_names=["ball holder"],
                    prompt_overrides={},
                    rng=random.Random(42),
                    neg_prompts_per_empty=0,
                    neg_prompts_per_nonempty=0,
                    max_samples=4,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=128,
                    max_objects=10,
                    api_base="https://api-staging.moondream.ai/v1",
                    api_key="test-key",
                )

        self.assertEqual(ctx.exception.failure_count, 1)
        self.assertIn("all /detect eval calls failed", str(ctx.exception))


class MainFlowTests(unittest.TestCase):
    class _MockRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}

        def finish(self) -> None:
            return

    class _MockWandb:
        def __init__(self) -> None:
            self.run = MainFlowTests._MockRun()
            self.logged: list[dict[str, object]] = []

        def init(self, *args: object, **kwargs: object) -> "MainFlowTests._MockRun":
            return self.run

        def log(self, payload: dict[str, object], step: object = None) -> None:
            self.logged.append({"step": step, "payload": dict(payload)})

    class _MockFinetune:
        def __init__(self) -> None:
            self.finetune_id = "ft_test"
            self.saved_steps: list[int] = []

        def train_step(self, *, groups: object, lr: float) -> SimpleNamespace:
            return SimpleNamespace(kl=0.0, router_kl=0.0, grad_norm=0.0)

        def save_checkpoint(self) -> SimpleNamespace:
            step = len(self.saved_steps) + 10
            self.saved_steps.append(step)
            return SimpleNamespace(checkpoint=SimpleNamespace(step=step))

    class _MockClient:
        def __init__(self, finetune: "MainFlowTests._MockFinetune") -> None:
            self._finetune = finetune

        def create_finetune(self, *, name: str, rank: int) -> "MainFlowTests._MockFinetune":
            return self._finetune

        def get_finetune(self, finetune_id: str) -> "MainFlowTests._MockFinetune":
            return self._finetune

    def _run_main_with_mocks(
        self,
        *,
        selection_metric: str,
        eval_metrics: list[dict[str, object]],
        run_final_test: bool,
        test_eval_side_effect: object = None,
    ) -> tuple[dict[str, object], list[dict[str, object]], list[list[str]], list[list[str]], "MainFlowTests._MockFinetune"]:
        dataset = {
            "train": [_sample_row_for_split("ball holder", split_name="train")],
            "validation": [_sample_row_for_split("ball holder", split_name="validation")],
            "test": [_sample_row_for_split("ball holder", split_name="test")],
        }
        mock_wandb = self._MockWandb()
        mock_finetune = self._MockFinetune()
        mock_client = self._MockClient(mock_finetune)
        val_split_calls: list[list[str]] = []
        test_split_calls: list[list[str]] = []
        eval_queue = list(eval_metrics)

        def _fake_iter_dataset_rows(ds: list[dict[str, object]], seed: int) -> object:
            def _iterator() -> object:
                while True:
                    for row in ds:
                        yield row

            return _iterator()

        def _fake_rollouts_batch_with_retry(**kwargs: object) -> list[SimpleNamespace]:
            requests = list(kwargs["requests"])
            num_rollouts = int(kwargs["num_rollouts"])
            out: list[SimpleNamespace] = []
            for req in requests:
                out.append(
                    SimpleNamespace(
                        rollouts=[
                            SimpleNamespace(
                                output=mod.DetectOutput(objects=[]),
                                answer_tokens=[],
                                thinking_tokens=[],
                                coords=[],
                                sizes=[],
                                skill="detect",
                                finish_reason="stop",
                            )
                        ],
                        request=mod.RolloutsRequest(
                            finetune_id="ft_test",
                            num_rollouts=num_rollouts,
                            request=req,
                            ground_truth=None,
                            org_id=None,
                        ),
                    )
                )
            return out

        def _fake_evaluate(*, eval_rows: object, **kwargs: object) -> dict[str, object]:
            rows = list(eval_rows)
            val_split_calls.append([str(row.get("source_collection")) for row in rows])
            return eval_queue.pop(0)

        def _fake_evaluate_api(*, eval_rows: object, **kwargs: object) -> dict[str, object]:
            rows = list(eval_rows)
            test_split_calls.append([str(row.get("source_collection")) for row in rows])
            if isinstance(test_eval_side_effect, Exception):
                raise test_eval_side_effect
            return _eval_metrics(f1=0.4, miou=0.3)

        argv = [
            "--api-key",
            "test-key",
            "--dataset-path",
            str(REPO_ROOT / "football_detect"),
            "--num-steps",
            "2" if not run_final_test else "1",
            "--batch-size",
            "1",
            "--group-size",
            "1",
            "--eval-every",
            "1",
            "--eval-batch-size",
            "1",
            "--eval-max-samples",
            "4",
            "--save-every",
            "0",
            "--augment-prob",
            "0.0",
            "--selection-metric",
            selection_metric,
            "--val-split",
            "validation",
            "--test-split",
            "test",
        ]
        if run_final_test:
            argv.append("--run-final-test")

        with patch.object(mod, "wandb", mock_wandb), patch.object(mod, "TunaClient", return_value=mock_client), patch.object(
            mod, "_load_local_dataset_dict", return_value=dataset
        ), patch.object(mod, "_iter_dataset_rows", side_effect=_fake_iter_dataset_rows), patch.object(
            mod, "_rollouts_batch_with_retry", side_effect=_fake_rollouts_batch_with_retry
        ), patch.object(mod, "_evaluate", side_effect=_fake_evaluate), patch.object(
            mod, "_evaluate_api", side_effect=_fake_evaluate_api
        ):
            mod.main(argv)

        return mock_wandb.run.summary, mock_wandb.logged, val_split_calls, test_split_calls, mock_finetune

    def _run_main_with_hf_mocks(
        self,
        *,
        selection_metric: str,
        eval_metrics: list[dict[str, object]],
        run_final_test: bool,
        dataset_path: str = "",
        dataset_name: str = "fake/football",
    ) -> tuple[dict[str, object], list[list[str]], list[list[str]], list[str]]:
        dataset_by_split = {
            "train": [_sample_row_for_split("ball holder", split_name="train")],
            "validation": [_sample_row_for_split("ball holder", split_name="validation")],
            "test": [_sample_row_for_split("ball holder", split_name="test")],
        }
        mock_wandb = self._MockWandb()
        mock_finetune = self._MockFinetune()
        mock_client = self._MockClient(mock_finetune)
        eval_queue = list(eval_metrics)
        val_split_calls: list[list[str]] = []
        test_split_calls: list[list[str]] = []
        hf_once_calls: list[str] = []

        def _fake_iter_hf_rows(dataset_name: str, split: str, token: object, seed: int, buffer_size: int) -> object:
            rows = list(dataset_by_split[split])

            def _iterator() -> object:
                while True:
                    for row in rows:
                        yield row

            return _iterator()

        def _fake_iter_hf_rows_once(dataset_name: str, split: str, token: object) -> object:
            hf_once_calls.append(split)
            return iter(list(dataset_by_split[split]))

        def _fake_rollouts_batch_with_retry(**kwargs: object) -> list[SimpleNamespace]:
            requests = list(kwargs["requests"])
            num_rollouts = int(kwargs["num_rollouts"])
            out: list[SimpleNamespace] = []
            for req in requests:
                out.append(
                    SimpleNamespace(
                        rollouts=[
                            SimpleNamespace(
                                output=mod.DetectOutput(objects=[]),
                                answer_tokens=[],
                                thinking_tokens=[],
                                coords=[],
                                sizes=[],
                                skill="detect",
                                finish_reason="stop",
                            )
                        ],
                        request=mod.RolloutsRequest(
                            finetune_id="ft_test",
                            num_rollouts=num_rollouts,
                            request=req,
                            ground_truth=None,
                            org_id=None,
                        ),
                    )
                )
            return out

        def _fake_evaluate(*, eval_rows: object, **kwargs: object) -> dict[str, object]:
            rows = list(eval_rows)
            val_split_calls.append([str(row.get("source_collection")) for row in rows])
            return eval_queue.pop(0)

        def _fake_evaluate_api(*, eval_rows: object, **kwargs: object) -> dict[str, object]:
            rows = list(eval_rows)
            test_split_calls.append([str(row.get("source_collection")) for row in rows])
            return _eval_metrics(f1=0.4, miou=0.3)

        argv = [
            "--api-key",
            "test-key",
            "--dataset-path",
            dataset_path,
            "--dataset-name",
            dataset_name,
            "--num-steps",
            "2" if not run_final_test else "1",
            "--batch-size",
            "1",
            "--group-size",
            "1",
            "--eval-every",
            "1",
            "--eval-batch-size",
            "1",
            "--eval-max-samples",
            "4",
            "--save-every",
            "0",
            "--augment-prob",
            "0.0",
            "--selection-metric",
            selection_metric,
            "--val-split",
            "validation",
            "--test-split",
            "test",
        ]
        if run_final_test:
            argv.append("--run-final-test")

        with patch.object(mod, "wandb", mock_wandb), patch.object(mod, "TunaClient", return_value=mock_client), patch.object(
            mod, "get_dataset_split_names", return_value=["train", "validation", "test"]
        ), patch.object(mod, "_iter_hf_rows", side_effect=_fake_iter_hf_rows), patch.object(
            mod, "_iter_hf_rows_once", side_effect=_fake_iter_hf_rows_once
        ), patch.object(mod, "_rollouts_batch_with_retry", side_effect=_fake_rollouts_batch_with_retry), patch.object(
            mod, "_evaluate", side_effect=_fake_evaluate
        ), patch.object(mod, "_evaluate_api", side_effect=_fake_evaluate_api):
            mod.main(argv)

        return mock_wandb.run.summary, val_split_calls, test_split_calls, hf_once_calls

    def test_selection_metric_f1_saves_best_checkpoint_by_eval_f1(self) -> None:
        summary, _, _, _, finetune = self._run_main_with_mocks(
            selection_metric="f1",
            eval_metrics=[
                _eval_metrics(f1=0.1, miou=0.1),
                _eval_metrics(f1=0.7, miou=0.2),
                _eval_metrics(f1=0.4, miou=0.9),
            ],
            run_final_test=False,
        )

        self.assertEqual(summary["best_step"], 0)
        self.assertEqual(summary["best_selection_metric_name"], "f1")
        self.assertEqual(summary["best_selection_metric"], 0.7)
        self.assertGreaterEqual(len(finetune.saved_steps), 2)

    def test_selection_metric_miou_saves_best_checkpoint_by_eval_miou(self) -> None:
        summary, _, _, _, finetune = self._run_main_with_mocks(
            selection_metric="miou",
            eval_metrics=[
                _eval_metrics(f1=0.1, miou=0.1),
                _eval_metrics(f1=0.7, miou=0.2),
                _eval_metrics(f1=0.4, miou=0.9),
            ],
            run_final_test=False,
        )

        self.assertEqual(summary["best_step"], 1)
        self.assertEqual(summary["best_selection_metric_name"], "miou")
        self.assertEqual(summary["best_selection_metric"], 0.9)
        self.assertGreaterEqual(len(finetune.saved_steps), 2)

    def test_periodic_eval_uses_validation_and_final_test_uses_test_split(self) -> None:
        summary, logged, val_split_calls, test_split_calls, _ = self._run_main_with_mocks(
            selection_metric="f1",
            eval_metrics=[
                _eval_metrics(f1=0.1, miou=0.1),
                _eval_metrics(f1=0.4, miou=0.2),
            ],
            run_final_test=True,
        )

        self.assertEqual(val_split_calls, [["validation"], ["validation"]])
        self.assertEqual(test_split_calls, [["test"]])
        self.assertEqual(summary["test_tasks"], 1)
        self.assertEqual(summary["test_f1"], 0.4)
        self.assertTrue(any("test_f1" in entry["payload"] for entry in logged))
        self.assertEqual(summary["test_eval_failures"], 0)

    def test_final_test_failure_records_error_without_zero_metrics(self) -> None:
        summary, logged, _, test_split_calls, _ = self._run_main_with_mocks(
            selection_metric="f1",
            eval_metrics=[
                _eval_metrics(f1=0.1, miou=0.1),
                _eval_metrics(f1=0.4, miou=0.2),
            ],
            run_final_test=True,
            test_eval_side_effect=mod._DetectEvalError(
                "all /detect eval calls failed (4 failures); last_error=RuntimeError: HTTP 403 from /detect",
                failure_count=4,
                last_error="RuntimeError: HTTP 403 from /detect",
            ),
        )

        self.assertEqual(test_split_calls, [["test"]])
        self.assertEqual(summary["test_eval_failures"], 4)
        self.assertIn("test_eval_error", summary)
        self.assertNotIn("test_tasks", summary)
        self.assertTrue(any("test_eval_error" in entry["payload"] for entry in logged))
        self.assertFalse(any("test_tasks" in entry["payload"] for entry in logged if "test_eval_error" in entry["payload"]))

    def test_hf_eval_and_test_rows_are_materialized_once(self) -> None:
        summary, val_split_calls, test_split_calls, hf_once_calls = self._run_main_with_hf_mocks(
            selection_metric="f1",
            eval_metrics=[
                _eval_metrics(f1=0.1, miou=0.1),
                _eval_metrics(f1=0.4, miou=0.2),
            ],
            run_final_test=True,
        )

        self.assertEqual(val_split_calls, [["validation"], ["validation"]])
        self.assertEqual(test_split_calls, [["test"]])
        self.assertEqual(hf_once_calls.count("train"), 1)
        self.assertEqual(hf_once_calls.count("validation"), 1)
        self.assertEqual(hf_once_calls.count("test"), 1)
        self.assertEqual(summary["test_eval_failures"], 0)

    def test_missing_local_dataset_path_falls_back_to_dataset_name(self) -> None:
        summary, val_split_calls, test_split_calls, hf_once_calls = self._run_main_with_hf_mocks(
            selection_metric="f1",
            eval_metrics=[
                _eval_metrics(f1=0.1, miou=0.1),
                _eval_metrics(f1=0.4, miou=0.2),
            ],
            run_final_test=True,
            dataset_path="football_detect/outputs/does_not_exist_for_fallback_test",
            dataset_name="fake/football",
        )

        self.assertEqual(val_split_calls, [["validation"], ["validation"]])
        self.assertEqual(test_split_calls, [["test"]])
        self.assertEqual(hf_once_calls.count("train"), 1)
        self.assertEqual(hf_once_calls.count("validation"), 1)
        self.assertEqual(hf_once_calls.count("test"), 1)
        self.assertEqual(summary["test_eval_failures"], 0)
