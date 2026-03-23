from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MDpi_and_d import train_pid_icons as shared_train

from bone_fracture import build_bone_fracture_hf_dataset as build_mod
from bone_fracture import train_bone_fracture_point as train_mod
from bone_fracture.common import (
    DEFAULT_POINT_CLASS_NAME,
    DEFAULT_POINT_HF_DATASET_NAME,
    DEFAULT_POINT_WANDB_PROJECT,
    DEFAULT_STAGING_API_BASE,
)


def _build_coco_fixture(root: Path) -> Path:
    export_root = root / "bone fracture.coco"
    split_defs = {
        "train": [
            {"id": 1, "file_name": "train_101_jpg.rf.aaa111.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "train_102_jpg.rf.bbb222.jpg", "width": 100, "height": 80},
        ],
        "valid": [
            {"id": 11, "file_name": "valid_201_jpg.rf.ccc333.jpg", "width": 120, "height": 90},
        ],
        "test": [
            {"id": 21, "file_name": "test_301_jpg.rf.ddd444.jpg", "width": 90, "height": 90},
        ],
    }
    annotation_defs = {
        "train": [
            {"id": 101, "image_id": 1, "category_id": 2, "bbox": [10, 20, 20, 40]},
        ],
        "valid": [
            {"id": 201, "image_id": 11, "category_id": 3, "bbox": [12, 18, 48, 20]},
        ],
        "test": [],
    }
    categories = [
        {"id": 0, "name": "bone-fracture", "supercategory": "none"},
        {"id": 1, "name": "angle", "supercategory": "bone-fracture"},
        {"id": 2, "name": "fracture", "supercategory": "bone-fracture"},
        {"id": 3, "name": "line", "supercategory": "bone-fracture"},
        {"id": 4, "name": "messed_up_angle", "supercategory": "bone-fracture"},
    ]
    for split_name, images in split_defs.items():
        split_dir = export_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for image in images:
            Image.new("RGB", (image["width"], image["height"]), color=(255, 255, 255)).save(split_dir / image["file_name"])
        payload = {
            "images": images,
            "annotations": annotation_defs[split_name],
            "categories": categories,
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    return export_root


class BoneFracturePointConfigTests(unittest.TestCase):
    def test_default_config_uses_point_settings(self) -> None:
        config_path = REPO_ROOT / "bone_fracture" / "configs" / "train_bone_fracture_point_default.json"
        args = train_mod.parse_args(["--config", str(config_path)])
        self.assertEqual(args.dataset_name, DEFAULT_POINT_HF_DATASET_NAME)
        self.assertEqual(args.base_url, DEFAULT_STAGING_API_BASE)
        self.assertTrue(str(args.env_file).endswith("/bone_fracture/.env.staging"))
        self.assertTrue(str(args.dataset_path).endswith("/bone_fracture/outputs/maxs-m87_bone_fracture_point_v1"))
        self.assertEqual(args.skill, "point")
        self.assertEqual(args.point_prompt_style, "class_name")
        self.assertEqual(args.reward_metric, "f1")
        self.assertEqual(args.api_key_env_var, "MOONDREAM_API_KEY_1")
        self.assertEqual(args.wandb_project, DEFAULT_POINT_WANDB_PROJECT)

    def test_cicd_configs_parse_with_expected_overrides(self) -> None:
        config_root = REPO_ROOT / "bone_fracture" / "configs" / "cicd"
        expectations = {
            "cicd_train_bone_fracture_point_control.json": (
                False,
                False,
                1.0,
                1.0,
                1,
                0.95,
                0.5,
                200,
                10,
                10,
                0.0,
                0.0,
                1,
                "MOONDREAM_API_KEY_1",
                "bone-fracture-point-control",
            ),
            "cicd_train_bone_fracture_point_recall_primary.json": (
                False,
                True,
                2.0,
                1.0,
                0,
                0.995,
                0.15,
                200,
                10,
                10,
                0.0,
                0.0,
                1,
                "MOONDREAM_API_KEY_2",
                "bone-fracture-point-recall-primary",
            ),
            "cicd_train_bone_fracture_point_recall_offpolicy.json": (
                True,
                True,
                2.0,
                1.0,
                0,
                0.995,
                0.15,
                200,
                10,
                10,
                0.0,
                0.0,
                1,
                "MOONDREAM_API_KEY_3",
                "bone-fracture-point-recall-offpolicy",
            ),
            "cicd_train_bone_fracture_point_control_short_klguard.json": (
                False,
                False,
                1.0,
                1.0,
                1,
                0.95,
                0.5,
                60,
                5,
                5,
                0.0,
                0.0,
                1,
                "MOONDREAM_API_KEY_1",
                "bone-fracture-point-control-short-klguard",
            ),
            "cicd_train_bone_fracture_point_recall_primary_short_klguard.json": (
                False,
                True,
                2.0,
                1.0,
                0,
                0.995,
                0.15,
                60,
                5,
                5,
                0.0,
                0.0,
                1,
                "MOONDREAM_API_KEY_2",
                "bone-fracture-point-angle-only-break-point-recall-primary-short-klguard",
            ),
            "cicd_train_bone_fracture_point_recall_offpolicy_lite_klguard.json": (
                True,
                True,
                2.0,
                1.0,
                0,
                0.995,
                0.15,
                60,
                5,
                5,
                0.0,
                0.0,
                1,
                "MOONDREAM_API_KEY_3",
                "bone-fracture-point-recall-offpolicy-lite-klguard",
            ),
        }
        for filename, (
            off_policy,
            recall_preset,
            fn_exp,
            fp_exp,
            neg_nonempty,
            pos_task_prob,
            neg_reward_weight,
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
                args = train_mod.parse_args(["--config", str(config_root / filename)])
                expected_dataset_name = DEFAULT_POINT_HF_DATASET_NAME
                expected_dataset_path_suffix = "/bone_fracture/outputs/maxs-m87_bone_fracture_point_v1"
                if filename == "cicd_train_bone_fracture_point_recall_primary_short_klguard.json":
                    expected_dataset_name = "maxs-m87/bone_fracture_point_angle_only_break_point_v1"
                    expected_dataset_path_suffix = (
                        "/bone_fracture/outputs/maxs-m87_bone_fracture_point_angle_only_break_point_v1"
                    )
                self.assertEqual(args.dataset_name, expected_dataset_name)
                self.assertEqual(args.base_url, DEFAULT_STAGING_API_BASE)
                self.assertTrue(str(args.dataset_path).endswith(expected_dataset_path_suffix))
                self.assertEqual(args.skill, "point")
                self.assertEqual(args.point_prompt_style, "class_name")
                self.assertEqual(args.reward_metric, "f1")
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.off_policy, off_policy)
                self.assertEqual(args.use_recall_first_preset, recall_preset)
                self.assertEqual(args.fn_penalty_exponent, fn_exp)
                self.assertEqual(args.fp_penalty_exponent, fp_exp)
                self.assertEqual(args.neg_prompts_per_nonempty, neg_nonempty)
                self.assertEqual(args.pos_task_prob, pos_task_prob)
                self.assertEqual(args.neg_reward_weight, neg_reward_weight)
                self.assertEqual(args.num_steps, num_steps)
                self.assertEqual(args.eval_every, eval_every)
                self.assertEqual(args.save_every, save_every)
                self.assertEqual(args.kl_warning_threshold, kl_warning_threshold)
                self.assertEqual(args.kl_stop_threshold, kl_stop_threshold)
                self.assertEqual(args.kl_stop_consecutive, kl_stop_consecutive)
                self.assertEqual(args.wandb_run_name, wandb_run_name)

    def test_round2_configs_parse_with_expected_overrides(self) -> None:
        config_root = REPO_ROOT / "bone_fracture" / "configs" / "cicd"
        expectations = {
            "cicd_train_bone_fracture_point_angle_only_recall_primary_anchor.json": (
                False,
                "maxs-m87/bone_fracture_point_angle_only_break_point_v1",
                "/bone_fracture/outputs/maxs-m87_bone_fracture_point_angle_only_break_point_v1",
                True,
                0.0005,
                60,
                "MOONDREAM_API_KEY_1",
                "bone-fracture-point-angle-only-recall-primary-anchor",
            ),
            "cicd_train_bone_fracture_point_angle_only_recall_offpolicy_anchor.json": (
                True,
                "maxs-m87/bone_fracture_point_angle_only_break_point_v1",
                "/bone_fracture/outputs/maxs-m87_bone_fracture_point_angle_only_break_point_v1",
                True,
                0.0005,
                60,
                "MOONDREAM_API_KEY_2",
                "bone-fracture-point-angle-only-recall-offpolicy-anchor",
            ),
            "cicd_train_bone_fracture_point_angle_only_recall_offpolicy_lite.json": (
                True,
                "maxs-m87/bone_fracture_point_angle_only_break_point_v1",
                "/bone_fracture/outputs/maxs-m87_bone_fracture_point_angle_only_break_point_v1",
                False,
                0.00025,
                60,
                "MOONDREAM_API_KEY_3",
                "bone-fracture-point-angle-only-recall-offpolicy-lite",
            ),
            "cicd_train_bone_fracture_point_full_recall_offpolicy_lite.json": (
                True,
                DEFAULT_POINT_HF_DATASET_NAME,
                "/bone_fracture/outputs/maxs-m87_bone_fracture_point_v1",
                False,
                0.00025,
                60,
                "MOONDREAM_API_KEY_4",
                "bone-fracture-point-full-recall-offpolicy-lite",
            ),
        }

        for filename, (
            off_policy,
            dataset_name,
            dataset_path_suffix,
            use_recall_first_preset,
            lr,
            num_steps,
            api_key_env_var,
            wandb_run_name,
        ) in expectations.items():
            with self.subTest(config=filename):
                args = train_mod.parse_args(["--config", str(config_root / filename)])
                self.assertEqual(args.dataset_name, dataset_name)
                self.assertTrue(str(args.dataset_path).endswith(dataset_path_suffix))
                self.assertEqual(args.off_policy, off_policy)
                self.assertEqual(args.use_recall_first_preset, use_recall_first_preset)
                self.assertEqual(args.lr, lr)
                self.assertEqual(args.num_steps, num_steps)
                self.assertEqual(args.eval_every, 5)
                self.assertEqual(args.save_every, 5)
                self.assertEqual(args.api_key_env_var, api_key_env_var)
                self.assertEqual(args.wandb_run_name, wandb_run_name)

                if "lite" in filename:
                    self.assertEqual(args.off_policy_std_thresh, 0.005)
                    self.assertEqual(args.off_policy_max_reward, 0.05)
                    self.assertEqual(args.off_policy_min_reward, 0.05)
                    self.assertEqual(args.off_policy_reward_scale, 1.0)
                else:
                    self.assertEqual(args.off_policy_std_thresh, 0.02)
                    self.assertEqual(args.off_policy_max_reward, 0.15)
                    self.assertEqual(args.off_policy_min_reward, 0.15)
                    self.assertEqual(args.off_policy_reward_scale, 2.0)


class BoneFracturePointSmokeTests(unittest.TestCase):
    def test_built_rows_support_shared_point_task_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = _build_coco_fixture(Path(tmp))
            dataset_dict, _, class_names = build_mod.build_dataset_dict_from_coco_export(
                raw_root,
                workspace="roboflow-100",
                project="bone-fracture-7fylg",
                version=2,
                seed=42,
                single_split_val_fraction=0.2,
                single_split_test_fraction=0.5,
            )

            train_rows = [dataset_dict["train"][idx] for idx in range(len(dataset_dict["train"]))]
            positive_row = next(row for row in train_rows if int(row["class_count"]) > 0)
            negative_row = next(row for row in dataset_dict["test"] if int(row["class_count"]) == 0)

            self.assertEqual(class_names, [DEFAULT_POINT_CLASS_NAME])

            positive_base = shared_train._to_base_sample(positive_row)
            negative_base = shared_train._to_base_sample(negative_row)
            self.assertIsNotNone(positive_base)
            self.assertIsNotNone(negative_base)
            assert positive_base is not None
            assert negative_base is not None

            positive_tasks = shared_train._tasks_from_base_sample(
                positive_base,
                all_class_names=[DEFAULT_POINT_CLASS_NAME],
                rng=shared_train.random.Random(42),
                neg_prompts_per_empty=1,
                neg_prompts_per_nonempty=0,
                prompt_style="class_name",
            )
            negative_tasks = shared_train._tasks_from_base_sample(
                negative_base,
                all_class_names=[DEFAULT_POINT_CLASS_NAME],
                rng=shared_train.random.Random(42),
                neg_prompts_per_empty=1,
                neg_prompts_per_nonempty=0,
                prompt_style="class_name",
            )

            self.assertTrue(any(task.is_positive for task in positive_tasks))
            self.assertTrue(any(not task.is_positive for task in negative_tasks))
            self.assertEqual({task.class_name for task in positive_tasks}, {DEFAULT_POINT_CLASS_NAME})
