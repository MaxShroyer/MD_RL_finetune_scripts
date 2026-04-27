from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from collections import Counter, deque
from pathlib import Path
from unittest import mock

from PIL import Image
from tuna_sdk import DetectRequest, PointRequest, QueryRequest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from disaster_m3 import benchmark_disaster_m3_mixed as bench_mod
from disaster_m3 import build_disaster_m3_dataset as build_mod
from disaster_m3 import common
from disaster_m3 import run_disaster_m3_rl_sweep as sweep_mod
from disaster_m3 import train_disaster_m3_mixed as train_mod


def _write_image(path: Path, color: str, size: tuple[int, int] = (32, 24)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _record(
    *,
    image_path: Path,
    skill: str = "query",
    task_name: str = "disaster_type",
    final_answer: dict[str, object] | None = None,
    boxes: str = "",
    points: str = "",
    metadata: dict[str, object] | None = None,
    question: str = "Question?",
    object_name: str = "building in the right post-disaster panel",
    row_id: str = "row-1",
) -> common.MixedTaskRecord:
    metadata = metadata or {}
    final_answer = final_answer or {}
    return common.MixedTaskRecord(
        row_id=row_id,
        split="test",
        task_name=task_name,
        task_family=common.infer_task_family(task_name, skill=skill),
        skill=skill,
        image_path=image_path,
        question=question,
        object_name=object_name if skill != "query" else "",
        final_answer_json=json.dumps(final_answer, ensure_ascii=False),
        answer_boxes_json=boxes,
        answer_points_json=points,
        source_pre_image_path="",
        source_post_image_path="",
        source_image_path="",
        metadata_json=json.dumps(metadata, ensure_ascii=False),
        metadata=dict(metadata),
    )


class DisasterM3CompositeTests(unittest.TestCase):
    def test_build_composite_image_and_remap_uses_correct_side_offsets(self) -> None:
        pre_image = Image.new("RGB", (20, 10), "red")
        post_image = Image.new("RGB", (10, 20), "blue")
        composite, layout = common.build_composite_image(
            pre_image=pre_image,
            post_image=post_image,
            panel_size=100,
            divider_px=4,
        )
        self.assertEqual(composite.size, (204, 100))
        self.assertFalse(layout.left.blank)
        self.assertFalse(layout.right.blank)

        pre_box = common.remap_box(
            common.LabeledBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0, source_side="pre"),
            layout=layout,
        )
        post_point = common.remap_point(
            common.LabeledPoint(x=0.5, y=0.5, source_side="post"),
            layout=layout,
        )
        self.assertLess(pre_box.x_max, 0.5)
        self.assertGreater(post_point.x, 0.5)

    def test_build_composite_image_supports_blank_side_fallback(self) -> None:
        pre_image = Image.new("RGB", (24, 24), "green")
        _, layout = common.build_composite_image(pre_image=pre_image, post_image=None, panel_size=48, divider_px=4)
        self.assertFalse(layout.left.blank)
        self.assertTrue(layout.right.blank)
        point = common.remap_point(common.LabeledPoint(x=0.5, y=0.5, source_side="pre"), layout=layout)
        self.assertLess(point.x, 0.5)


class DisasterM3BuilderTests(unittest.TestCase):
    def test_query_prompt_is_direct_and_has_no_blank_panel_clause(self) -> None:
        prompt = build_mod._query_prompt(
            task_kind="multi_choice_single_answer",
            original_question="Which disaster is shown?",
            options_by_letter={"A": "Flood", "B": "Wildfire"},
        )
        self.assertIn("left is pre-disaster and right is post-disaster", prompt)
        self.assertNotIn("If one panel is blank", prompt)

    def test_adapt_row_skips_segmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = _write_image(tmp / "raw" / "images" / "sample.png", "white")
            image_index = common.load_image_index(tmp / "raw")
            stats = Counter()
            rows = build_mod._adapt_row(
                row={
                    "question": "Segment the damaged buildings.",
                    "answer": "ignored",
                    "pre_image_path": str(image_path),
                    "mask_path": "sample_mask.png",
                },
                task_name="building_segmentation",
                split="train",
                source_path=tmp / "raw" / "DisasterM3_Instruct" / "train" / "building_segmentation.json",
                source_row_index=0,
                image_index=image_index,
                output_dir=tmp / "out",
                raw_root=tmp / "raw",
                panel_size=64,
                divider_px=4,
                jpeg_quality=92,
                stats=stats,
            )
            self.assertEqual(rows, [])
            self.assertEqual(stats["skipped_segmentation"], 1)

    def test_build_dataset_preserves_splits_and_uses_bench_as_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            instruct_root = raw_dir / "DisasterM3_Instruct"
            bench_root = raw_dir / "DisasterM3_Bench"

            _write_image(raw_dir / "images" / "train_pre.png", "red")
            _write_image(raw_dir / "images" / "train_post.png", "blue")
            _write_image(raw_dir / "images" / "val_post.png", "green")
            _write_image(raw_dir / "images" / "test_pre.png", "yellow")
            _write_image(raw_dir / "images" / "test_post.png", "purple")

            _write_json(
                instruct_root / "train" / "disaster_type.json",
                [
                    {
                        "id": "train-1",
                        "question": "Which disaster affected the area?",
                        "answer": "A",
                        "options": {"A": "Flood", "B": "Wildfire"},
                        "pre_image_path": "images/train_pre.png",
                        "post_image_path": "images/train_post.png",
                    }
                ],
            )
            _write_json(
                instruct_root / "val" / "road_damage_point.json",
                [
                    {
                        "id": "val-1",
                        "post_image_path": "images/val_post.png",
                        "points": {"road": [16, 10]},
                        "boxes": {"road": [4, 4, 24, 16]},
                    }
                ],
            )
            _write_json(
                bench_root / "building_localization.json",
                [
                    {
                        "id": "test-1",
                        "pre_image_path": "images/test_pre.png",
                        "post_image_path": "images/test_post.png",
                        "boxes": {"building": [4, 3, 18, 18]},
                    }
                ],
            )

            args = build_mod.parse_args(
                [
                    "--dataset-root",
                    str(root),
                    "--raw-dir",
                    str(raw_dir),
                    "--output-dir",
                    str(root / "dataset"),
                    "--no-download",
                    "--panel-max-side",
                    "64",
                ]
            )
            summary = build_mod.build_dataset(args)
            self.assertEqual(summary["metadata"]["split_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(summary["dataset_report"]["totals"]["raw_total_rows"], 3)
            self.assertEqual(summary["dataset_report"]["totals"]["built_manifest_rows"], 3)

            train_records = common.load_mixed_records(dataset_dir=Path(args.output_dir), split_name="train")
            val_records = common.load_mixed_records(dataset_dir=Path(args.output_dir), split_name="val")
            test_records = common.load_mixed_records(dataset_dir=Path(args.output_dir), split_name="test")
            self.assertEqual(train_records[0].skill, "query")
            self.assertEqual(val_records[0].skill, "point")
            self.assertTrue(val_records[0].answer_boxes_json)
            self.assertEqual(test_records[0].skill, "detect")
            self.assertFalse(Path(json.loads((Path(args.output_dir) / "jsonl" / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])["image_path"]).is_absolute())
            self.assertTrue((Path(args.output_dir) / "dataset_report.json").exists())
            self.assertTrue((Path(args.output_dir) / "dataset_report.md").exists())
            task_samples = json.loads((Path(args.output_dir) / "task_samples.json").read_text(encoding="utf-8"))
            self.assertIn("disaster_type", task_samples)
            self.assertEqual(task_samples["disaster_type"][0]["ground_truth"], {"answer": "A"})


class DisasterM3RewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.image_path = _write_image(self.tmp / "sample.png", "white")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_multi_choice_single_and_multi_answer_rewards(self) -> None:
        single = _record(
            image_path=self.image_path,
            task_name="disaster_type",
            final_answer={"answer": "A"},
            metadata={"query_kind": "multi_choice_single_answer", "options_by_letter": {"A": "Flood", "B": "Fire"}},
            row_id="single",
        )
        multi = _record(
            image_path=self.image_path,
            task_name="bearing_body",
            final_answer={"answers": ["A", "C"]},
            metadata={"query_kind": "multi_choice_multi_answer", "options_by_letter": {"A": "x", "B": "y", "C": "z"}},
            row_id="multi",
        )
        single_outcome = train_mod.score_prediction_for_record(
            single,
            query_answer='{"answer":"A"}',
            strict_query_rewards=True,
        )
        multi_outcome = train_mod.score_prediction_for_record(
            multi,
            query_answer='{"answers":["A","C"]}',
            strict_query_rewards=False,
        )
        self.assertTrue(single_outcome.task_correct)
        self.assertGreaterEqual(single_outcome.reward, 0.99)
        self.assertGreaterEqual(multi_outcome.answer_set_f1, 0.99)

    def test_count_description_and_recovery_rewards(self) -> None:
        count_record = _record(
            image_path=self.image_path,
            task_name="building_damage_counting",
            final_answer={"count": 3},
            metadata={"query_kind": "count"},
            row_id="count",
        )
        description_target = {
            "disaster": "Flood",
            "building": "Several buildings are inundated.",
            "road": "Roads are partially submerged.",
            "vegetation": "Vegetation is waterlogged.",
            "water_body": "Water covers the central area.",
            "agriculture": "Fields are flooded.",
            "conclusion": "The post-disaster image shows severe flood damage.",
        }
        description_record = _record(
            image_path=self.image_path,
            task_name="caption",
            final_answer=description_target,
            metadata={"query_kind": "description"},
            row_id="description",
        )
        recovery_record = _record(
            image_path=self.image_path,
            task_name="recovery",
            final_answer={
                "needs_recovery": True,
                "immediate_recovery": "Clear debris and restore access roads.",
                "long_term_recovery": "Rebuild damaged homes and improve drainage.",
            },
            metadata={"query_kind": "recovery"},
            row_id="recovery",
        )
        count_outcome = train_mod.score_prediction_for_record(
            count_record,
            query_answer='{"count":3}',
            strict_query_rewards=False,
        )
        description_outcome = train_mod.score_prediction_for_record(
            description_record,
            query_answer=json.dumps(description_target),
            strict_query_rewards=False,
        )
        recovery_outcome = train_mod.score_prediction_for_record(
            recovery_record,
            query_answer='{"needs_recovery":true,"immediate_recovery":"Clear debris and restore access roads.","long_term_recovery":"Rebuild damaged homes and improve drainage."}',
            strict_query_rewards=False,
        )
        self.assertGreaterEqual(count_outcome.count_score, 0.99)
        self.assertGreater(description_outcome.description_field_coverage, 0.95)
        self.assertGreater(description_outcome.change_awareness_score, 0.9)
        self.assertGreater(recovery_outcome.recovery_action_score, 0.95)
        self.assertGreater(recovery_outcome.needs_recovery_accuracy, 0.99)

    def test_detect_and_point_rewards(self) -> None:
        boxes = common.serialize_boxes(
            [
                common.DetectAnnotation(x_min=0.60, y_min=0.10, x_max=0.90, y_max=0.40),
            ]
        )
        points = common.serialize_points([common.PointAnnotation(x=0.75, y=0.25)])
        detect_record = _record(
            image_path=self.image_path,
            skill="detect",
            task_name="building_localization",
            boxes=boxes,
            final_answer={},
            metadata={},
            question="",
            row_id="detect",
        )
        point_record = _record(
            image_path=self.image_path,
            skill="point",
            task_name="road_damage_point",
            boxes=boxes,
            points=points,
            final_answer={},
            metadata={},
            question="",
            row_id="point",
        )
        detect_outcome = train_mod.score_prediction_for_record(
            detect_record,
            detect_boxes=[{"x_min": 0.60, "y_min": 0.10, "x_max": 0.90, "y_max": 0.40}],
            strict_query_rewards=False,
        )
        point_outcome = train_mod.score_prediction_for_record(
            point_record,
            point_coords=[{"x": 0.75, "y": 0.25}],
            strict_query_rewards=False,
        )
        self.assertGreaterEqual(detect_outcome.detect_f1, 0.99)
        self.assertGreaterEqual(detect_outcome.detect_miou, 0.99)
        self.assertGreaterEqual(point_outcome.point_f1, 0.99)


class DisasterM3RequestTests(unittest.TestCase):
    def test_prepare_request_for_record_emits_skill_specific_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _write_image(Path(tmpdir) / "sample.png", "orange")
            query_record = _record(
                image_path=image_path,
                skill="query",
                task_name="disaster_type",
                final_answer={"answer": "A"},
                metadata={"query_kind": "multi_choice_single_answer"},
                row_id="query",
            )
            point_record = _record(
                image_path=image_path,
                skill="point",
                task_name="road_damage_point",
                points=common.serialize_points([common.PointAnnotation(x=0.25, y=0.25)]),
                question="",
                row_id="point",
            )
            detect_record = _record(
                image_path=image_path,
                skill="detect",
                task_name="building_localization",
                boxes=common.serialize_boxes([common.DetectAnnotation(x_min=0.1, y_min=0.2, x_max=0.4, y_max=0.5)]),
                question="",
                row_id="detect",
            )

            query_request, query_gt = common.prepare_request_for_record(query_record, temperature=0.0, top_p=1.0, max_tokens=64, reasoning=False)
            point_request, point_gt = common.prepare_request_for_record(point_record, temperature=0.0, top_p=1.0, max_tokens=64, reasoning=False)
            detect_request, detect_gt = common.prepare_request_for_record(detect_record, temperature=0.0, top_p=1.0, max_tokens=64, reasoning=False)

            self.assertIsInstance(query_request, QueryRequest)
            self.assertIsNone(query_gt)
            self.assertIsInstance(point_request, PointRequest)
            self.assertEqual(len(point_gt.points), 1)
            self.assertIsInstance(detect_request, DetectRequest)
            self.assertEqual(len(detect_gt.boxes), 1)


class DisasterM3TrainerUtilityTests(unittest.TestCase):
    def test_compose_train_groups_uses_replay_after_warmup(self) -> None:
        on_policy = ["a", "b", "c", "d"]
        replay = deque(["r1", "r2", "r3", "r4", "r5", "r6"])
        mixed, off_policy_count = train_mod.compose_train_groups(
            on_policy_groups=on_policy,
            replay_groups=replay,
            off_policy=True,
            off_policy_mix_ratio=0.5,
            off_policy_warmup_steps=2,
            off_policy_min_buffer_groups=4,
            global_step=3,
            rng=random.Random(7),
        )
        self.assertEqual(len(mixed), 4)
        self.assertEqual(off_policy_count, 2)
        self.assertGreaterEqual(sum(1 for item in mixed if str(item).startswith("r")), 1)

    def test_validate_args_rejects_off_policy_with_reasoning(self) -> None:
        args = train_mod.parse_args(
            [
                "--config",
                str(REPO_ROOT / "disaster_m3" / "configs" / "train_disaster_m3_mixed_default.json"),
                "--off-policy",
                "--rl-reasoning",
            ]
        )
        with self.assertRaises(ValueError):
            train_mod._validate_args(args)

    def test_append_eval_history_writes_overall_and_by_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eval_history.jsonl"
            train_mod._append_eval_history(
                path=path,
                stage_name="rl",
                split_name="val",
                step=10,
                checkpoint_step=20,
                payload={
                    "overall": {"reward_mean": 0.5},
                    "by_task": {"disaster_type": {"reward_mean": 0.75}},
                    "by_skill": {"query": {"reward_mean": 0.5}},
                },
                source="sync",
                predictions_jsonl="predictions.jsonl",
            )
            record = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(record["stage"], "rl")
            self.assertIn("overall", record)
            self.assertIn("by_task", record)
            self.assertIn("by_skill", record)


class DisasterM3BenchmarkTests(unittest.TestCase):
    def test_benchmark_main_writes_summary_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_path = _write_image(root / "images" / "test_query.jpg", "white")
            dataset_dir = root / "dataset"
            (dataset_dir / "jsonl").mkdir(parents=True, exist_ok=True)
            row = {
                "row_id": "bench-query",
                "split": "test",
                "task_name": "disaster_type",
                "task_family": "recognition",
                "skill": "query",
                "image_path": str(image_path),
                "question": 'Respond in JSON only with the format {"answer":"A"}.',
                "object_name": "",
                "final_answer_json": json.dumps({"answer": "A"}),
                "answer_boxes_json": "",
                "answer_points_json": "",
                "source_pre_image_path": "",
                "source_post_image_path": "",
                "source_image_path": "",
                "metadata_json": json.dumps({"query_kind": "multi_choice_single_answer", "options_by_letter": {"A": "Flood"}}),
            }
            (dataset_dir / "jsonl" / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            summary_path = root / "summary.json"
            predictions_path = root / "predictions.jsonl"

            with mock.patch.object(
                bench_mod.common,
                "resolve_inference_model",
                return_value=common.InferenceModelResolution(
                    model="moondream3-preview",
                    finetune_id="",
                    requested_checkpoint_step=None,
                    resolved_checkpoint_step=None,
                ),
            ), mock.patch.object(
                bench_mod.common,
                "call_inference_api",
                return_value=('{"answer":"A"}', {"answer": '{"answer":"A"}'}, 12.0),
            ):
                bench_mod.main(
                    [
                        "--api-key",
                        "test-key",
                        "--dataset-dir",
                        str(dataset_dir),
                        "--split",
                        "test",
                        "--output-json",
                        str(summary_path),
                        "--predictions-jsonl",
                        str(predictions_path),
                        "--no-progress",
                    ]
                )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            prediction_row = json.loads(predictions_path.read_text(encoding="utf-8").strip())
            self.assertEqual(summary["evaluated_samples"], 1)
            self.assertGreater(summary["metrics"]["overall"]["reward_mean"], 0.99)
            self.assertGreater(summary["overall_reward_mean"], 0.99)
            self.assertEqual(prediction_row["ground_truth"], {"answer": "A"})
            self.assertIn("grading", prediction_row)
            self.assertTrue(predictions_path.exists())


class DisasterM3ConfigTests(unittest.TestCase):
    def test_all_configs_parse(self) -> None:
        config_root = REPO_ROOT / "disaster_m3" / "configs"
        for config_path in sorted(config_root.rglob("*.json")):
            with self.subTest(config=str(config_path.relative_to(REPO_ROOT))):
                name = config_path.name
                if name.startswith("build_"):
                    args = build_mod.parse_args(["--config", str(config_path)])
                    self.assertIn(Path(args.output_dir).name, {"full", "subset_1000"})
                elif name.startswith("benchmark_"):
                    args = bench_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.split, "test")
                    self.assertTrue(str(args.dataset_dir).endswith("disaster_m3/dataset/full"))
                else:
                    args = train_mod.parse_args(["--config", str(config_path)])
                    self.assertEqual(args.rank, 32)
                    self.assertTrue(str(args.dataset_dir).endswith("disaster_m3/dataset/full"))
                    if "offpolicy" in name:
                        self.assertTrue(args.off_policy)
                        self.assertFalse(args.rl_reasoning)
                    elif "reasoning" in name:
                        self.assertFalse(args.off_policy)
                        self.assertTrue(args.rl_reasoning)
                    else:
                        self.assertFalse(args.off_policy)
                    if "warmup200" in name:
                        self.assertEqual(args.mode, "bootstrap_then_rl")
                        self.assertEqual(args.bootstrap_steps, 200)
                    elif "rl_only" in name or name == "train_disaster_m3_mixed_default.json":
                        self.assertEqual(args.mode, "rl")
                    self.assertEqual(args.eval_every, 10)
                    self.assertEqual(args.save_every, 10)
                    self.assertEqual(args.rl_batch_size, 32)
                    self.assertEqual(args.rl_group_size, 8)


class DisasterM3SweepTests(unittest.TestCase):
    def test_build_sweep_runs_has_expected_grid(self) -> None:
        args = sweep_mod.parse_args(
            [
                "--dataset-dir",
                "disaster_m3/dataset/full",
                "--manifest-path",
                "disaster_m3/outputs/test_manifest.json",
                "--dry-run",
            ]
        )
        runs = sweep_mod.build_sweep_runs(args)
        self.assertEqual(len(runs), 24)
        off_policy_runs = [run for run in runs if run["sweep_mode"] == "off_policy"]
        reasoning_runs = [run for run in runs if run["sweep_mode"] == "reasoning"]
        self.assertEqual(len(off_policy_runs), 12)
        self.assertEqual(len(reasoning_runs), 12)
        self.assertEqual({run["rank"] for run in runs}, {16, 24, 32})
        self.assertEqual({run["rl_lr"] for run in runs}, {2e-4, 5e-4, 5e-5, 1e-5})
        self.assertEqual(args.max_parallel, 8)
        self.assertEqual(len(args.key_slot_env_vars), 8)
        self.assertEqual(args.key_slot_env_vars.count("CICID_GPUB_MOONDREAM_API_KEY_1"), 2)
        self.assertIn("__KEY_SLOT_ENV_VAR__", runs[0]["command"])

    def test_build_sweep_runs_includes_warmup_family_when_requested(self) -> None:
        args = sweep_mod.parse_args(
            [
                "--dataset-dir",
                "disaster_m3/dataset/full",
                "--training-regime",
                "both",
                "--dry-run",
            ]
        )
        runs = sweep_mod.build_sweep_runs(args)
        self.assertEqual(len(runs), 48)
        warmup_runs = [run for run in runs if run["training_regime"] == "warmup_200"]
        rl_only_runs = [run for run in runs if run["training_regime"] == "rl_only"]
        self.assertEqual(len(warmup_runs), 24)
        self.assertEqual(len(rl_only_runs), 24)
        self.assertTrue(any("warmup200" in run["finetune_name"] for run in warmup_runs))


if __name__ == "__main__":
    unittest.main()
