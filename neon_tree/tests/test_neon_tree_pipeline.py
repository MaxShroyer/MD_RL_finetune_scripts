from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from async_checkpoint_eval import CheckpointEvalResult
from neon_tree import benchmark_neon_tree_detect as benchmark_mod
from neon_tree import build_neon_tree_hf_dataset as build_mod
from neon_tree import common
from neon_tree import generate_synthetic_flyover as flyover_mod
from neon_tree import tracking_utils
from neon_tree import track_neon_tree_video as track_mod
from neon_tree import train_neon_tree_detect as train_mod


def _write_image(path: Path, *, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_xml(path: Path, *, filename: str, boxes: list[tuple[int, int, int, int]]) -> None:
    objects = "\n".join(
        [
            f"""
  <object>
    <name>Tree</name>
    <bndbox>
      <xmin>{x1}</xmin>
      <ymin>{y1}</ymin>
      <xmax>{x2}</xmax>
      <ymax>{y2}</ymax>
    </bndbox>
  </object>"""
            for x1, y1, x2, y2 in boxes
        ]
    )
    payload = f"""<annotation>
  <filename>{filename}</filename>
  <size>
    <width>100</width>
    <height>100</height>
  </size>
{objects}
</annotation>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> common.DetectAnnotation:
    return common.DetectAnnotation(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _payload_to_box(payload: dict[str, float]) -> common.DetectAnnotation:
    return _box(
        float(payload["x_min"]),
        float(payload["y_min"]),
        float(payload["x_max"]),
        float(payload["y_max"]),
    )


class DatasetBuilderTests(unittest.TestCase):
    def test_build_dataset_dict_from_raw_root_uses_official_split_and_cites_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw_root = Path(tmp) / "raw"
            train_image = raw_root / "training" / "RGB" / "ABCD_001_image.tif"
            eval_image = raw_root / "evaluation" / "RGB" / "EFGH_001_image.tif"
            extra_eval_image = raw_root / "evaluation" / "RGB" / "IJKL_001_image.tif"
            _write_image(train_image)
            _write_image(eval_image)
            _write_image(extra_eval_image)
            _write_xml(raw_root / "annotations" / "ABCD_001_image.xml", filename="ABCD_001_image.tif", boxes=[(10, 10, 30, 30)])
            _write_xml(raw_root / "annotations" / "EFGH_001_image.xml", filename="EFGH_001_image.tif", boxes=[(20, 20, 60, 60)])
            _write_xml(raw_root / "annotations" / "MISSING_001_image.xml", filename="MISSING_001_image.tif", boxes=[(5, 5, 15, 15)])

            dataset_dict, metadata, stats, readme = build_mod.build_dataset_dict_from_raw_root(raw_root)
            self.assertEqual(len(dataset_dict["train"]), 1)
            self.assertEqual(len(dataset_dict["validation"]), 1)
            self.assertEqual(dataset_dict["train"][0]["source_split"], "training")
            self.assertEqual(dataset_dict["validation"][0]["source_split"], "evaluation")
            self.assertEqual(stats["skipped_unmatched_annotations"], 1)
            self.assertEqual(metadata["label_name"], "tree")
            self.assertIn("10.5281/zenodo.5914554", readme)
            self.assertIn("weecology/NeonTreeEvaluation", readme)
            train_boxes = common.parse_answer_boxes(dataset_dict["train"][0]["answer_boxes"])
            self.assertEqual(len(train_boxes), 1)


class BenchmarkTests(unittest.TestCase):
    def test_evaluate_rows_computes_perfect_metrics_with_fake_detector(self) -> None:
        row = {
            "image": Image.new("RGB", (100, 100), color=(255, 255, 255)),
            "answer_boxes": json.dumps(
                [
                    {
                        "class_name": "tree",
                        "x_min": 0.1,
                        "y_min": 0.1,
                        "x_max": 0.3,
                        "y_max": 0.3,
                    }
                ]
            ),
            "source_image_id": "sample",
        }
        fake_box = common.DetectAnnotation(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3)
        metrics, predictions = benchmark_mod.evaluate_rows(
            rows=[row],
            model="moondream3-preview",
            api_base="https://example.com/v1",
            api_key="test",
            prompt="tree",
            tiling=common.TilingConfig(enabled=False),
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
            max_objects=8,
            timeout=5.0,
            detector=lambda _row: [fake_box],
        )
        self.assertEqual(metrics["eval_tasks"], 1)
        self.assertAlmostEqual(metrics["eval_f1"], 1.0)
        self.assertAlmostEqual(metrics["eval_f1_macro"], 1.0)
        self.assertAlmostEqual(metrics["eval_miou"], 1.0)
        self.assertEqual(predictions[0]["source_image_id"], "sample")

    def test_build_flyover_windows_uses_serpentine_order(self) -> None:
        windows = common.build_flyover_windows(
            width=500,
            height=500,
            window_width=400,
            window_height=400,
            step_x=32,
            step_y=32,
            path_style="serpentine",
        )
        self.assertEqual([(win.left, win.top) for win in windows[:8]], [(0, 0), (32, 0), (64, 0), (96, 0), (100, 0), (100, 32), (96, 32), (64, 32)])

    def test_synthetic_rows_load_from_clip_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = {
                "image": Image.new("RGB", (450, 450), color=(255, 255, 255)),
                "answer_boxes": common.serialize_answer_boxes(
                    [
                        common.DetectAnnotation(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3),
                        common.DetectAnnotation(x_min=0.6, y_min=0.6, x_max=0.9, y_max=0.9),
                    ]
                ),
                "source_image_id": "dense_source",
            }
            manifest = flyover_mod.generate_clip_from_row(
                row=row,
                source_image_id="dense_source",
                output_dir=Path(tmp),
                window_width=400,
                window_height=400,
                step_x=32,
                step_y=32,
                fps=12,
                path_style="serpentine",
            )
            rows = common.load_synthetic_rows(
                clip_manifest=str(manifest["manifest_path"]),
                synthetic_gt_jsonl="",
                synthetic_video="",
            )
            self.assertGreater(len(rows), 0)
            self.assertIn("synthetic_track_ids", rows[0])


class FlyoverTests(unittest.TestCase):
    def test_select_source_rows_prefers_most_boxes_then_source_id(self) -> None:
        rows = [
            {
                "image": Image.new("RGB", (450, 450), color=(255, 255, 255)),
                "answer_boxes": common.serialize_answer_boxes([common.DetectAnnotation(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)]),
                "source_image_id": "b_source",
            },
            {
                "image": Image.new("RGB", (450, 450), color=(255, 255, 255)),
                "answer_boxes": common.serialize_answer_boxes(
                    [
                        common.DetectAnnotation(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2),
                        common.DetectAnnotation(x_min=0.3, y_min=0.3, x_max=0.4, y_max=0.4),
                    ]
                ),
                "source_image_id": "a_source",
            },
        ]
        selected = flyover_mod.select_source_rows(
            rows,
            window_width=400,
            window_height=400,
            max_source_rasters=1,
            source_selection="most_boxes",
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["source_image_id"], "a_source")

    def test_select_source_rows_zero_limit_returns_all_candidates(self) -> None:
        rows = [
            {
                "image": Image.new("RGB", (450, 450), color=(255, 255, 255)),
                "answer_boxes": common.serialize_answer_boxes([common.DetectAnnotation(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)]),
                "source_image_id": "b_source",
            },
            {
                "image": Image.new("RGB", (450, 450), color=(255, 255, 255)),
                "answer_boxes": common.serialize_answer_boxes(
                    [
                        common.DetectAnnotation(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2),
                        common.DetectAnnotation(x_min=0.3, y_min=0.3, x_max=0.4, y_max=0.4),
                    ]
                ),
                "source_image_id": "a_source",
            },
        ]
        selected = flyover_mod.select_source_rows(
            rows,
            window_width=400,
            window_height=400,
            max_source_rasters=0,
            source_selection="most_boxes",
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual([item["source_image_id"] for item in selected], ["a_source", "b_source"])


class TrainingTests(unittest.TestCase):
    def test_compose_train_groups_mixes_replay_when_ready(self) -> None:
        rng = __import__("random").Random(7)
        mixed, off_policy_count = train_mod.compose_train_groups(
            on_policy_groups=["a", "b", "c", "d"],  # type: ignore[arg-type]
            replay_groups=__import__("collections").deque(["x", "y", "z"], maxlen=16),  # type: ignore[arg-type]
            off_policy=True,
            off_policy_mix_ratio=0.5,
            off_policy_warmup_steps=2,
            off_policy_min_buffer_groups=2,
            global_step=3,
            rng=rng,
        )
        self.assertEqual(len(mixed), 4)
        self.assertEqual(off_policy_count, 2)
        self.assertTrue(any(item in {"x", "y", "z"} for item in mixed))

    def test_parse_args_reads_current_rank32_recall_config(self) -> None:
        args = train_mod.parse_args(
            [
                "--config",
                str(common.repo_relative("configs", "current", "train_neon_tree_detect_exp04_hybrid_rank32_recall.json")),
            ]
        )
        self.assertFalse(args.off_policy)
        self.assertEqual(args.api_key_env_var, "CICID_GPUB_MOONDREAM_API_KEY_4")
        self.assertEqual(args.rank, 32)
        self.assertEqual(args.group_size, 8)
        self.assertEqual(args.selection_metric, "f1")
        self.assertEqual(args.reward_metric, "hybrid")
        self.assertAlmostEqual(args.reward_fn_beta, 2.5, places=8)

    def test_ingest_async_results_logs_arrival_step_and_updates_best_summary(self) -> None:
        logged: list[tuple[dict[str, float], int]] = []

        class _WandbStub:
            @staticmethod
            def log(payload: dict[str, float], step: int) -> None:
                logged.append((payload, step))

        run = SimpleNamespace(summary={})
        result = CheckpointEvalResult(
            trainer="neon_tree_detect",
            finetune_id="ft-test",
            checkpoint_step=42,
            selection_metric="miou",
            status="succeeded",
            returncode=0,
            job_dir=Path("/tmp/job"),
            job_json_path=Path("/tmp/job/job.json"),
            metrics_json_path=Path("/tmp/job/metrics.json"),
            predictions_jsonl_path=Path("/tmp/job/preds.jsonl"),
            stdout_log_path=Path("/tmp/job/stdout.log"),
            command=["python"],
            started_at=0.0,
            completed_at=1.0,
            metrics_payload={"eval_f1": 0.8, "eval_f1_macro": 0.75, "eval_miou": 0.7},
            metadata={"step_for_log": 7},
        )

        with patch.object(train_mod, "wandb", _WandbStub):
            best_metric, best_step, best_checkpoint_step, success_count = train_mod.ingest_async_results(
                results=[result],
                run=run,
                selection_metric="miou",
                baseline_metrics={"eval_f1": 0.5, "eval_f1_macro": 0.5, "eval_miou": 0.4},
                log_step=22,
                current_best_metric=None,
                current_best_step=None,
                current_best_checkpoint_step=None,
            )

        self.assertEqual(success_count, 1)
        self.assertEqual(best_step, 7)
        self.assertEqual(best_checkpoint_step, 42)
        self.assertAlmostEqual(best_metric or 0.0, 0.7)
        self.assertEqual(len(logged), 1)
        payload, step = logged[0]
        self.assertEqual(step, 22)
        self.assertEqual(payload["async_eval_checkpoint_step"], 42)
        self.assertAlmostEqual(payload["eval_selection_metric_delta_vs_baseline"], 0.3)
        self.assertEqual(run.summary["best_checkpoint_step"], 42)


class TrackingTests(unittest.TestCase):
    def test_require_supervision_raises_clean_runtime_error(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "supervision":
                raise ModuleNotFoundError("missing supervision")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_fake_import):
            with self.assertRaisesRegex(RuntimeError, "supervision is required"):
                tracking_utils.require_supervision()

    def test_track_predictions_writes_records_with_fake_tracker(self) -> None:
        row = {
            "image": Image.new("RGB", (100, 100), color=(255, 255, 255)),
            "source_image_id": "clip:000000",
        }
        prediction = {
            "source_image_id": "clip:000000",
            "pred_boxes": [
                {
                    "x_min": 0.1,
                    "y_min": 0.1,
                    "x_max": 0.3,
                    "y_max": 0.3,
                }
            ],
        }

        class _Detections:
            def __init__(self, *, xyxy, confidence, class_id):
                self.xyxy = xyxy
                self.confidence = confidence
                self.class_id = class_id
                self.tracker_id = None

        class _Tracker:
            def update_with_detections(self, detections):
                detections.tracker_id = np.array([7], dtype=np.int32) if len(detections.xyxy) else np.empty((0,), dtype=np.int32)
                return detections

        class _SV:
            Detections = _Detections

        with tempfile.TemporaryDirectory() as tmp:
            out_jsonl = Path(tmp) / "tracks.jsonl"
            with patch.object(tracking_utils, "require_supervision", return_value=_SV), patch.object(
                tracking_utils,
                "make_tracker",
                return_value=_Tracker(),
            ):
                summary = tracking_utils.track_predictions(
                    rows=[row],
                    predictions=[prediction],
                    prompt="tree",
                    output_jsonl=str(out_jsonl),
                    render_output="",
                    frame_rate=30,
                    track_activation_threshold=0.25,
                    lost_track_buffer=30,
                    minimum_matching_threshold=0.8,
                    minimum_consecutive_frames=1,
                )
            self.assertEqual(summary["tracking_records"], 1)
            self.assertEqual(summary["tracking_tracks"], 1)
            lines = out_jsonl.read_text().splitlines()
            self.assertEqual(len(lines), 1)


class RewardTests(unittest.TestCase):
    def test_reward_hybrid_is_one_for_perfect_prediction(self) -> None:
        gt_boxes = [_box(0.1, 0.1, 0.2, 0.2), _box(0.4, 0.4, 0.5, 0.5)]
        reward = common.reward_hybrid(gt_boxes, gt_boxes)
        self.assertAlmostEqual(reward, 1.0, places=6)

    def test_reward_hybrid_prefers_partial_tight_prediction_over_missing_all_gt(self) -> None:
        gt_boxes = [_box(0.1, 0.1, 0.2, 0.2), _box(0.4, 0.4, 0.5, 0.5)]
        missing_reward = common.reward_hybrid([], gt_boxes)
        partial_reward = common.reward_hybrid([gt_boxes[0]], gt_boxes)
        self.assertLess(missing_reward, partial_reward)

    def test_reward_hybrid_heavily_penalizes_full_image_collapse_box(self) -> None:
        gt_boxes = [
            _box(0.05, 0.05, 0.10, 0.10),
            _box(0.25, 0.05, 0.30, 0.10),
            _box(0.45, 0.05, 0.50, 0.10),
            _box(0.65, 0.05, 0.70, 0.10),
        ]
        collapse_reward = common.reward_hybrid([_box(0.0, 0.0, 1.0, 1.0)], gt_boxes)
        partial_reward = common.reward_hybrid([gt_boxes[0], gt_boxes[1]], gt_boxes)
        self.assertLess(collapse_reward, 0.05)
        self.assertLess(collapse_reward, partial_reward)

    def test_reward_hybrid_penalizes_oversized_match_more_than_tight_match(self) -> None:
        gt_boxes = [_box(0.40, 0.40, 0.50, 0.50)]
        tight_reward = common.reward_hybrid([_box(0.40, 0.40, 0.50, 0.50)], gt_boxes)
        oversized_breakdown = common.hybrid_reward_breakdown([_box(0.30, 0.30, 0.70, 0.70)], gt_boxes)
        self.assertGreater(oversized_breakdown.matched_oversize_penalty, 0.0)
        self.assertLess(oversized_breakdown.reward, tight_reward)

    def test_reward_hybrid_penalizes_box_covering_multiple_gt_centers(self) -> None:
        gt_boxes = [_box(0.10, 0.10, 0.18, 0.18), _box(0.22, 0.10, 0.30, 0.18)]
        one_to_one_breakdown = common.hybrid_reward_breakdown([_box(0.00, 0.10, 0.20, 0.22)], gt_boxes)
        collapse_breakdown = common.hybrid_reward_breakdown([_box(0.10, 0.10, 0.30, 0.22)], gt_boxes)
        self.assertGreater(collapse_breakdown.collapse_penalty, 0.0)
        self.assertLess(collapse_breakdown.reward, one_to_one_breakdown.reward)

    def test_reward_hybrid_empty_tile_no_predictions_is_one(self) -> None:
        breakdown = common.hybrid_reward_breakdown([], [])
        self.assertAlmostEqual(breakdown.reward, 1.0, places=6)

    def test_saved_checkpoint_predictions_rank_collapse_rows_below_partial_rows(self) -> None:
        saved_prediction_paths = {
            132: common.repo_relative(
                "outputs",
                "async_checkpoint_eval",
                "key4_prompt-v2",
                "neon_tree_detect",
                "01KN8EZF92QJPN1PDE9186CDG5",
                "step000132_20260403_035010",
                "predictions.jsonl",
            ),
            144: common.repo_relative(
                "outputs",
                "async_checkpoint_eval",
                "key4_prompt-v2",
                "neon_tree_detect",
                "01KN8EZF92QJPN1PDE9186CDG5",
                "step000144_20260403_043250",
                "predictions.jsonl",
            ),
        }
        missing = [path for path in saved_prediction_paths.values() if not path.exists()]
        if missing:
            self.skipTest(f"saved predictions not available: {missing[0]}")

        for checkpoint_step, predictions_path in saved_prediction_paths.items():
            collapse_rewards: list[float] = []
            partial_rewards: list[float] = []
            for line in predictions_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                pred_boxes = [_payload_to_box(payload) for payload in row.get("pred_boxes", [])]
                gt_boxes = [_payload_to_box(payload) for payload in row.get("gt_boxes", [])]
                reward = common.reward_hybrid(pred_boxes, gt_boxes)
                max_pred_area = max(
                    (
                        max(0.0, pred_box.x_max - pred_box.x_min) * max(0.0, pred_box.y_max - pred_box.y_min)
                        for pred_box in pred_boxes
                    ),
                    default=0.0,
                )
                if len(pred_boxes) == 1 and len(gt_boxes) >= 20 and max_pred_area > 0.9:
                    collapse_rewards.append(reward)
                if len(pred_boxes) >= 10 and len(gt_boxes) >= 20:
                    partial_rewards.append(reward)

            self.assertTrue(collapse_rewards, f"missing collapse rewards for checkpoint {checkpoint_step}")
            self.assertTrue(partial_rewards, f"missing partial rewards for checkpoint {checkpoint_step}")
            self.assertLess(max(collapse_rewards), 0.05)
            self.assertLess(max(collapse_rewards), max(partial_rewards))


if __name__ == "__main__":
    unittest.main()
