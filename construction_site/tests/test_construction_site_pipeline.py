from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from construction_site import build_construction_site_hf_dataset as build_mod
from construction_site import benchmark_construction_site_detect as bench_detect_mod
from construction_site import benchmark_construction_site_query_caption as bench_caption_mod
from construction_site import benchmark_construction_site_query_rule_vqa as bench_rule_mod
from construction_site import common
from construction_site import query_common
from construction_site import train_construction_site_detect as train_detect_mod
from construction_site import train_construction_site_query_caption as train_caption_mod
from construction_site import train_construction_site_query_rule_vqa as train_rule_mod


class ConstructionSiteCommonTests(unittest.TestCase):
    def test_build_detect_boxes_includes_grounding_and_rule_boxes(self) -> None:
        row = {
            "excavator": [[0.1, 0.2, 0.3, 0.4]],
            "rebar": [],
            "worker_with_white_hard_hat": [[0.5, 0.5, 0.6, 0.8]],
            "rule_1_violation": {"bounding_box": [[0.11, 0.22, 0.33, 0.44]], "reason": "missing hard hat"},
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": {"bounding_box": [[0.7, 0.2, 0.9, 0.7]], "reason": "too close to excavator"},
        }
        boxes = common.build_detect_boxes(row)
        names = [box["class_name"] for box in boxes]
        self.assertEqual(
            names,
            ["excavator", "worker_with_white_hard_hat", "rule_1_violation", "rule_4_violation"],
        )
        self.assertTrue(all(0.0 <= float(box["x_min"]) <= 1.0 for box in boxes))

    def test_extract_rule_reasons_only_returns_present_rules(self) -> None:
        row = {
            "rule_1_violation": {"bounding_box": [[0.1, 0.1, 0.2, 0.2]], "reason": "missing hard hat"},
            "rule_2_violation": None,
            "rule_3_violation": {"bounding_box": [[0.2, 0.2, 0.3, 0.4]], "reason": ""},
            "rule_4_violation": {"bounding_box": [[0.2, 0.2, 0.3, 0.4]], "reason": "inside radius"},
        }
        self.assertEqual(common.extract_rule_reasons(row), {1: "missing hard hat", 4: "inside radius"})

    def test_query_common_save_checkpoint_returns_saved_step(self) -> None:
        finetune = mock.Mock()
        finetune.save_checkpoint.return_value = mock.Mock(checkpoint=mock.Mock(step=42))
        self.assertEqual(query_common.save_checkpoint(finetune=finetune, context="test"), 42)


class ConstructionSiteBuilderTests(unittest.TestCase):
    def test_split_train_rows_is_deterministic_and_keeps_all_rows(self) -> None:
        rows = [{"image_id": str(idx)} for idx in range(10)]
        train_rows, val_rows = build_mod._split_train_rows(rows, seed=42, val_fraction=0.2)
        self.assertEqual(len(train_rows) + len(val_rows), len(rows))
        self.assertEqual(len(val_rows), 2)
        second_train_rows, second_val_rows = build_mod._split_train_rows(rows, seed=42, val_fraction=0.2)
        self.assertEqual(train_rows, second_train_rows)
        self.assertEqual(val_rows, second_val_rows)

    def test_query_jsonl_builders_emit_expected_schema(self) -> None:
        row = {
            "image_id": "abc123",
            "image_caption": "An excavator and a worker are in the scene.",
            "illumination": "normal lighting",
            "camera_distance": "mid distance",
            "view": "elevation view",
            "quality_of_info": "rich info",
            "excavator": [[0.1, 0.2, 0.3, 0.4]],
            "rebar": [],
            "worker_with_white_hard_hat": [[0.5, 0.5, 0.6, 0.8]],
            "rule_1_violation": {"bounding_box": [[0.11, 0.22, 0.33, 0.44]], "reason": "missing hard hat"},
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "caption"
            image_path = Path(tmpdir) / "images" / "train" / "abc123.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"fake")
            caption_rows = build_mod._caption_jsonl_rows(
                rows=[row],
                split_name="train",
                dataset_dir=dataset_dir,
                image_paths={"abc123": image_path},
            )
            rule_rows = build_mod._rule_vqa_jsonl_rows(
                rows=[row],
                split_name="train",
                dataset_dir=dataset_dir,
                image_paths={"abc123": image_path},
            )
            self.assertEqual(json.loads(caption_rows[0]["final_answer_json"]), {"caption": row["image_caption"]})
            self.assertEqual(
                json.loads(rule_rows[0]["final_answer_json"]),
                {"violated_rules": [1], "reasons": {"1": "missing hard hat"}},
            )


class ConstructionSiteConfigTests(unittest.TestCase):
    def test_all_configs_parse(self) -> None:
        config_root = REPO_ROOT / "construction_site" / "configs"
        for config_path in sorted(config_root.rglob("*.json")):
            with self.subTest(config=str(config_path.relative_to(REPO_ROOT))):
                name = config_path.name
                if name == "construction_site_detect_class_catalog.json":
                    payload = json.loads(config_path.read_text(encoding="utf-8"))
                    self.assertIn("class_catalog", payload)
                    continue
                if name.startswith("build_"):
                    args = build_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(Path(args.detect_output_dir).name, "construction_site_detect_v1")
                elif "detect" in name and name.startswith("benchmark_"):
                    args = bench_detect_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.skill, "detect")
                    self.assertEqual(args.api_base, common.DEFAULT_STAGING_API_BASE)
                elif "detect" in name:
                    args = train_detect_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.skill, "detect")
                elif "query_caption" in name and name.startswith("benchmark_"):
                    args = bench_caption_mod.parse_args(["--config", str(config_path)])
                    self.assertTrue(str(args.dataset_dir).endswith("construction_site_query_caption_v1"))
                elif "query_caption" in name:
                    args = train_caption_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.rank, 32)
                    self.assertGreater(args.lr, 0.0)
                elif "query_rule_vqa" in name and name.startswith("benchmark_"):
                    args = bench_rule_mod.parse_args(["--config", str(config_path)])
                    self.assertTrue(str(args.dataset_dir).endswith("construction_site_query_rule_vqa_v1"))
                elif "query_rule_vqa" in name:
                    args = train_rule_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.rank, 32)
                    self.assertGreater(args.lr, 0.0)
                else:
                    self.fail(f"unhandled config path: {config_path}")

    def test_query_benchmarks_resolve_named_api_key_env_var(self) -> None:
        caption_args = bench_caption_mod.parse_args(["--api-key-env-var", "CICID_GPUB_MOONDREAM_API_KEY_1"])
        rule_args = bench_rule_mod.parse_args(["--api-key-env-var", "CICID_GPUB_MOONDREAM_API_KEY_2"])
        with mock.patch.dict(
            os.environ,
            {
                "CICID_GPUB_MOONDREAM_API_KEY_1": "caption-secret",
                "CICID_GPUB_MOONDREAM_API_KEY_2": "rule-secret",
            },
            clear=False,
        ):
            self.assertEqual(bench_caption_mod._resolve_api_key(caption_args), "caption-secret")
            self.assertEqual(bench_rule_mod._resolve_api_key(rule_args), "rule-secret")

    def test_query_benchmark_configs_default_to_staging(self) -> None:
        caption_args = bench_caption_mod.parse_args(
            ["--config", str(REPO_ROOT / "construction_site/configs/benchmark_construction_site_query_caption_default.json")]
        )
        rule_args = bench_rule_mod.parse_args(
            ["--config", str(REPO_ROOT / "construction_site/configs/benchmark_construction_site_query_rule_vqa_default.json")]
        )
        self.assertEqual(caption_args.base_url, common.DEFAULT_STAGING_API_BASE)
        self.assertEqual(rule_args.base_url, common.DEFAULT_STAGING_API_BASE)
        self.assertTrue(str(caption_args.env_file).endswith("construction_site/.env.staging"))
        self.assertTrue(str(rule_args.env_file).endswith("construction_site/.env.staging"))
        self.assertEqual(caption_args.api_key_env_var, "CICID_GPUB_MOONDREAM_API_KEY_1")
        self.assertEqual(rule_args.api_key_env_var, "CICID_GPUB_MOONDREAM_API_KEY_2")

    def test_cicd_detect_configs_use_hf_dataset_source(self) -> None:
        control_args = train_detect_mod.parse_args(
            ["--config", str(REPO_ROOT / "construction_site/configs/cicd/cicd_train_construction_site_detect_reasoning.json")]
        )
        offpolicy_args = train_detect_mod.parse_args(
            ["--config", str(REPO_ROOT / "construction_site/configs/cicd/cicd_train_construction_site_detect_offpolicy.json")]
        )
        self.assertEqual(control_args.dataset_path, "")
        self.assertEqual(offpolicy_args.dataset_path, "")
        self.assertEqual(control_args.dataset_name, common.DEFAULT_DATASET_NAME)
        self.assertEqual(offpolicy_args.dataset_name, common.DEFAULT_DATASET_NAME)
        self.assertEqual(control_args.val_split, "")
        self.assertEqual(offpolicy_args.val_split, "")
        self.assertEqual(control_args.test_split, "")
        self.assertEqual(offpolicy_args.test_split, "")
        self.assertTrue(str(control_args.class_names_file).endswith("construction_site_detect_class_catalog.json"))
        self.assertTrue(str(offpolicy_args.class_names_file).endswith("construction_site_detect_class_catalog.json"))
        self.assertFalse(control_args.run_final_test)
        self.assertFalse(offpolicy_args.run_final_test)

    def test_cicd_rule_vqa_configs_select_rule_set_f1_metric(self) -> None:
        control_args = train_rule_mod.parse_args(
            ["--config", str(REPO_ROOT / "construction_site/configs/cicd/cicd_train_construction_site_query_rule_vqa_reasoning.json")]
        )
        offpolicy_args = train_rule_mod.parse_args(
            ["--config", str(REPO_ROOT / "construction_site/configs/cicd/cicd_train_construction_site_query_rule_vqa_offpolicy.json")]
        )
        self.assertEqual(control_args.best_metric, "eval_rule_set_f1")
        self.assertEqual(offpolicy_args.best_metric, "eval_rule_set_f1")


class ConstructionSiteCaptionRewardTests(unittest.TestCase):
    def test_caption_reward_perfect_prediction_scores_high(self) -> None:
        example = train_caption_mod.CaptionExample(
            row_id="x",
            split="validation",
            task_type=common.CAPTION_TASK_TYPE,
            question="q",
            image_path=Path("/tmp/example.png"),
            reference_caption="An excavator works near a worker in normal lighting.",
            attribute_tags=("normal lighting",),
            object_tags=("excavator", "worker"),
        )
        outcome = train_caption_mod._score_payload_for_example(
            example,
            {"caption": "An excavator works near a worker in normal lighting."},
        )
        self.assertTrue(outcome.parse_success)
        self.assertGreaterEqual(outcome.caption_token_f1, 0.99)
        self.assertGreaterEqual(outcome.reward, 0.95)

    def test_caption_reward_bad_payload_fails_parse(self) -> None:
        example = train_caption_mod.CaptionExample(
            row_id="x",
            split="validation",
            task_type=common.CAPTION_TASK_TYPE,
            question="q",
            image_path=Path("/tmp/example.png"),
            reference_caption="An excavator works near a worker in normal lighting.",
            attribute_tags=("normal lighting",),
            object_tags=("excavator", "worker"),
        )
        outcome = train_caption_mod._score_payload_for_example(example, {"wrong": "schema"})
        self.assertFalse(outcome.parse_success)
        self.assertEqual(outcome.reward, 0.0)


class ConstructionSiteDetectBenchmarkTests(unittest.TestCase):
    def test_detect_wrapper_converts_raw_hf_row_to_base_sample(self) -> None:
        row = {
            "image_id": "abc123",
            "image": Image.new("RGB", (8, 8), "white"),
            "excavator": [[0.1, 0.2, 0.3, 0.4]],
            "rebar": [],
            "worker_with_white_hard_hat": [[0.5, 0.5, 0.6, 0.8]],
            "rule_1_violation": {"bounding_box": [[0.11, 0.22, 0.33, 0.44]], "reason": "missing hard hat"},
            "rule_2_violation": None,
            "rule_3_violation": None,
            "rule_4_violation": None,
        }
        sample = bench_detect_mod._to_base_sample(row, fallback_id=0)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.sample_id, "abc123")
        self.assertEqual(
            [item.class_name for item in sample.boxes],
            ["excavator", "worker_with_white_hard_hat", "rule_1_violation"],
        )

    def test_detect_train_wrapper_filters_to_reduced_catalog(self) -> None:
        row = {
            "image_id": "abc123",
            "image": Image.new("RGB", (8, 8), "white"),
            "excavator": [[0.1, 0.2, 0.3, 0.4]],
            "rebar": [],
            "worker_with_white_hard_hat": [[0.5, 0.5, 0.6, 0.8]],
            "rule_1_violation": {"bounding_box": [[0.11, 0.22, 0.33, 0.44]], "reason": "missing hard hat"},
            "rule_2_violation": {"bounding_box": [[0.12, 0.22, 0.34, 0.45]], "reason": "missing harness"},
            "rule_3_violation": None,
            "rule_4_violation": {"bounding_box": [[0.61, 0.22, 0.73, 0.45]], "reason": "inside excavator radius"},
        }
        original_allowed = set(train_detect_mod._ALLOWED_DETECT_CLASS_NAMES)
        try:
            train_detect_mod._ALLOWED_DETECT_CLASS_NAMES = {
                "excavator",
                "rebar",
                "worker_with_white_hard_hat",
                "rule_1_violation",
            }
            sample = train_detect_mod._to_base_sample(row)
        finally:
            train_detect_mod._ALLOWED_DETECT_CLASS_NAMES = original_allowed
        self.assertIsNotNone(sample)
        self.assertEqual(
            [item.class_name for item in sample.boxes],
            ["excavator", "worker_with_white_hard_hat", "rule_1_violation"],
        )


class ConstructionSiteRuleRewardTests(unittest.TestCase):
    def test_rule_reward_perfect_prediction_scores_high(self) -> None:
        example = train_rule_mod.RuleVQAExample(
            row_id="x",
            split="validation",
            task_type=common.RULE_VQA_TASK_TYPE,
            question="q",
            image_path=Path("/tmp/example.png"),
            violated_rules=(1, 4),
            reasons={1: "worker is missing a hard hat", 4: "worker is inside the excavator radius"},
        )
        outcome = train_rule_mod._score_payload_for_example(
            example,
            {
                "violated_rules": [1, 4],
                "reasons": {
                    "1": "worker is missing a hard hat",
                    "4": "worker is inside the excavator radius",
                },
            },
        )
        self.assertTrue(outcome.parse_success)
        self.assertTrue(outcome.strict_parse_success)
        self.assertEqual(outcome.rule_set_accuracy, 1.0)
        self.assertEqual(outcome.strict_rule_set_accuracy, 1.0)
        self.assertGreaterEqual(outcome.reason_token_f1, 0.99)
        self.assertGreaterEqual(outcome.strict_reason_token_f1, 0.99)
        self.assertGreaterEqual(outcome.reward, 0.95)
        self.assertGreaterEqual(outcome.strict_reward, 0.95)

    def test_rule_reward_penalizes_hallucinated_rules(self) -> None:
        example = train_rule_mod.RuleVQAExample(
            row_id="x",
            split="validation",
            task_type=common.RULE_VQA_TASK_TYPE,
            question="q",
            image_path=Path("/tmp/example.png"),
            violated_rules=(1,),
            reasons={1: "worker is missing a hard hat"},
        )
        outcome = train_rule_mod._score_payload_for_example(
            example,
            {
                "violated_rules": [1, 4],
                "reasons": {
                    "1": "worker is missing a hard hat",
                    "4": "worker is inside the excavator radius",
                },
            },
        )
        self.assertTrue(outcome.parse_success)
        self.assertTrue(outcome.strict_parse_success)
        self.assertEqual(outcome.hallucinated_rule_count, 1)
        self.assertEqual(outcome.strict_hallucinated_rule_count, 1)
        self.assertLess(outcome.reward, 1.0)
        self.assertLess(outcome.strict_reward, 1.0)

    def test_rule_reward_softens_common_schema_variants(self) -> None:
        example = train_rule_mod.RuleVQAExample(
            row_id="x",
            split="validation",
            task_type=common.RULE_VQA_TASK_TYPE,
            question="q",
            image_path=Path("/tmp/example.png"),
            violated_rules=(1,),
            reasons={1: "worker is missing a hard hat"},
        )
        outcome = train_rule_mod._score_payload_for_example(
            example,
            {
                "violated_rules": [
                    "Workers on foot must use required PPE such as hard hats and proper protective clothing."
                ],
                "reasons": [],
            },
        )
        self.assertTrue(outcome.parse_success)
        self.assertFalse(outcome.strict_parse_success)
        self.assertEqual(outcome.rule_set_f1, 1.0)
        self.assertEqual(outcome.strict_rule_set_f1, 0.0)
        self.assertGreater(outcome.reward, 0.7)
        self.assertEqual(outcome.strict_reward, 0.0)

    def test_rule_reward_softens_empty_reason_list_for_no_violation_case(self) -> None:
        example = train_rule_mod.RuleVQAExample(
            row_id="x",
            split="validation",
            task_type=common.RULE_VQA_TASK_TYPE,
            question="q",
            image_path=Path("/tmp/example.png"),
            violated_rules=(),
            reasons={},
        )
        outcome = train_rule_mod._score_payload_for_example(
            example,
            {
                "violated_rules": [],
                "reasons": [],
            },
        )
        self.assertTrue(outcome.parse_success)
        self.assertTrue(outcome.no_violation_correct)
        self.assertFalse(outcome.strict_parse_success)
        self.assertFalse(outcome.strict_no_violation_correct)
        self.assertEqual(outcome.reward, 0.85)
        self.assertEqual(outcome.strict_reward, 0.0)

    def test_rule_reward_complete_positive_miss_scores_zero(self) -> None:
        example = train_rule_mod.RuleVQAExample(
            row_id="x",
            split="validation",
            task_type=common.RULE_VQA_TASK_TYPE,
            question="q",
            image_path=Path("/tmp/example.png"),
            violated_rules=(1,),
            reasons={1: "worker is missing a hard hat"},
        )
        outcome = train_rule_mod._score_payload_for_example(
            example,
            {
                "violated_rules": [],
                "reasons": {},
            },
        )
        self.assertTrue(outcome.parse_success)
        self.assertEqual(outcome.rule_set_f1, 0.0)
        self.assertEqual(outcome.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
