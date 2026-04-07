from __future__ import annotations

import io
import json
import sys
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmark_openrouter_task_packets as mod


class _FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class OpenRouterTaskPacketParsingTests(unittest.TestCase):
    def test_parse_detect_accepts_normalized_coords(self) -> None:
        boxes, points, ok = mod.parse_openrouter_prediction(
            skill="detect",
            answer_text='{"objects":[{"x_min":0.1,"y_min":0.2,"x_max":0.4,"y_max":0.5}]}',
            raw_response={},
            image_width=1000,
            image_height=500,
        )
        self.assertTrue(ok)
        self.assertEqual(points, [])


class OpenRouterTaskPacketRequestFallbackTests(unittest.TestCase):
    def test_build_request_variants_relaxes_constraints(self) -> None:
        variants = mod.build_request_variants("original")
        self.assertEqual(variants[0].name, "strict_schema")
        self.assertTrue(variants[0].require_parameters)
        self.assertTrue(variants[0].use_response_format)
        self.assertTrue(variants[0].use_response_healing)
        self.assertEqual(variants[-1].name, "prompt_json_fallback")
        self.assertFalse(variants[-1].require_parameters)
        self.assertFalse(variants[-1].use_response_format)
        self.assertFalse(variants[-1].use_response_healing)

    def test_call_openrouter_chat_api_falls_back_after_parameter_routing_404(self) -> None:
        strict_variant = mod.RequestVariant(
            name="strict_schema",
            detail="original",
            use_response_format=True,
            require_parameters=True,
            use_response_healing=True,
        )
        relaxed_variant = mod.RequestVariant(
            name="schema_without_require_parameters",
            detail="original",
            use_response_format=True,
            require_parameters=False,
            use_response_healing=True,
        )
        first_error = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(
                b'{"error":{"message":"No endpoints found that can handle the requested parameters","code":404}}'
            ),
        )
        success_body = '{"choices":[{"message":{"content":"{\\"objects\\":[]}"}}]}'
        with patch("urllib.request.urlopen", side_effect=[first_error, _FakeHTTPResponse(success_body)]):
            answer_text, raw_response, latency_ms, variant_name = mod.call_openrouter_chat_api(
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                payload_variants=[
                    (strict_variant, {"model": "openai/gpt-5.4"}),
                    (relaxed_variant, {"model": "openai/gpt-5.4"}),
                ],
                timeout=5.0,
                retry_429_max_retries=0,
                retry_429_backoff_s=0.0,
                retry_429_max_backoff_s=0.0,
            )
        self.assertEqual(variant_name, "schema_without_require_parameters")
        self.assertEqual(answer_text, '{"objects":[]}')
        self.assertIsInstance(raw_response, dict)
        self.assertGreaterEqual(latency_ms, 0.0)


class OpenRouterTaskPacketEnvTests(unittest.TestCase):
    def test_resolve_env_file_path_falls_back_to_env_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / ".env.staging").write_text("OPENROUTER_API_KEY=test\n", encoding="utf-8")
            resolved = mod._resolve_env_file_path(".env", repo_root=root)
            self.assertEqual(resolved, str((root / ".env.staging").resolve()))

    def test_resolve_openrouter_api_key_uses_named_env_var_then_openrouter_default(self) -> None:
        with patch.dict("os.environ", {"CUSTOM_OR_KEY": "custom-key", "OPENROUTER_API_KEY": "default-key"}, clear=True):
            self.assertEqual(mod._resolve_openrouter_api_key("", "CUSTOM_OR_KEY"), "custom-key")
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "default-key"}, clear=True):
            self.assertEqual(mod._resolve_openrouter_api_key("", "CUSTOM_OR_KEY"), "default-key")

    def test_parse_detect_accepts_pixel_coords(self) -> None:
        boxes, _, ok = mod.parse_openrouter_prediction(
            skill="detect",
            answer_text='{"objects":[{"x_min":100,"y_min":50,"x_max":400,"y_max":250}]}',
            raw_response={},
            image_width=1000,
            image_height=500,
        )
        self.assertTrue(ok)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].x_min, 0.1)
        self.assertAlmostEqual(boxes[0].y_min, 0.1)
        self.assertAlmostEqual(boxes[0].x_max, 0.4)
        self.assertAlmostEqual(boxes[0].y_max, 0.5)

    def test_parse_detect_accepts_empty_array(self) -> None:
        boxes, _, ok = mod.parse_openrouter_prediction(
            skill="detect",
            answer_text='{"objects":[]}',
            raw_response={},
            image_width=100,
            image_height=100,
        )
        self.assertTrue(ok)
        self.assertEqual(boxes, [])

    def test_parse_detect_extracts_json_from_extra_prose(self) -> None:
        boxes, _, ok = mod.parse_openrouter_prediction(
            skill="detect",
            answer_text='Here is the result: {"objects":[{"x_min":0.2,"y_min":0.3,"x_max":0.6,"y_max":0.8}]}',
            raw_response={},
            image_width=100,
            image_height=100,
        )
        self.assertTrue(ok)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].x_max, 0.6)

    def test_parse_detect_uses_healed_raw_response_when_primary_text_is_malformed(self) -> None:
        raw_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"objects":[{"x_min":0.1,"y_min":0.2,"x_max":0.3,"y_max":0.4}]}'
                    }
                }
            ]
        }
        boxes, _, ok = mod.parse_openrouter_prediction(
            skill="detect",
            answer_text='{"objects":[{"x_min":0.1,"y_min":0.2,"x_max":0.3,"y_max":0.4}',
            raw_response=raw_response,
            image_width=100,
            image_height=100,
        )
        self.assertTrue(ok)
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].y_max, 0.4)

    def test_parse_point_accepts_normalized_coords(self) -> None:
        _, points, ok = mod.parse_openrouter_prediction(
            skill="point",
            answer_text='{"points":[{"x":0.25,"y":0.75},{"x":0.5,"y":0.5}]}',
            raw_response={},
            image_width=100,
            image_height=100,
        )
        self.assertTrue(ok)
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0].x, 0.25)
        self.assertAlmostEqual(points[1].y, 0.5)

    def test_parse_point_accepts_pixel_coords(self) -> None:
        _, points, ok = mod.parse_openrouter_prediction(
            skill="point",
            answer_text='{"points":[{"x":250,"y":75}]}',
            raw_response={},
            image_width=1000,
            image_height=100,
        )
        self.assertTrue(ok)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].x, 0.25)
        self.assertAlmostEqual(points[0].y, 0.75)

    def test_parse_point_accepts_empty_array(self) -> None:
        _, points, ok = mod.parse_openrouter_prediction(
            skill="point",
            answer_text='{"points":[]}',
            raw_response={},
            image_width=100,
            image_height=100,
        )
        self.assertTrue(ok)
        self.assertEqual(points, [])


class OpenRouterTaskPacketRegistryTests(unittest.TestCase):
    def test_build_task_registry_resolves_expected_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            state_ref = root / "outputs" / "task_sample_packets" / "state_farm" / "state_baseline_vs_ft"
            state_ref.mkdir(parents=True)
            (state_ref / "metrics.json").write_text("{}", encoding="utf-8")
            (state_ref / "sample_before_after.json").write_text("[]", encoding="utf-8")
            (root / "outputs" / "task_sample_packets" / "state_farm" / "README.md").write_text("# State Farm\n", encoding="utf-8")

            player_ref = root / "outputs" / "task_sample_packets" / "player_with_ball" / "player_baseline_vs_ft"
            player_ref.mkdir(parents=True)
            (player_ref / "player_compare.json").write_text("{}", encoding="utf-8")
            (root / "outputs" / "task_sample_packets" / "player_with_ball" / "README.md").write_text("# Player\n", encoding="utf-8")

            aerial_dir = root / "outputs" / "task_sample_packets" / "aerial"
            aerial_dir.mkdir(parents=True)
            (aerial_dir / "samples.before_after.json").write_text("[]", encoding="utf-8")
            (aerial_dir / "README.md").write_text("# Aerial\n", encoding="utf-8")
            (aerial_dir / "benchmark_packet.json").write_text("{}", encoding="utf-8")

            subset_dir = root / "outputs" / "advertising_subsets" / "aerial_manual_top100" / "subset" / "dataset"
            subset_dir.mkdir(parents=True)

            registry = mod.build_task_registry(root)

            self.assertEqual(registry["state_farm"].skill, "detect")
            self.assertEqual(registry["state_farm"].dataset_id, "maxs-m87/NBA_StateFarm_Splits_01")
            self.assertEqual(registry["state_farm"].split, "validation")
            self.assertEqual(registry["state_farm"].reference_metrics_path, state_ref / "metrics.json")

            self.assertEqual(registry["player_with_ball"].skill, "detect")
            self.assertEqual(registry["player_with_ball"].dataset_id, "maxs-m87/Ball-Holder-splits-v1")
            self.assertEqual(registry["player_with_ball"].split, "test")
            self.assertEqual(registry["player_with_ball"].reference_compare_path, player_ref / "player_compare.json")
            self.assertEqual(registry["player_with_ball"].reference_readme_path, root / "outputs" / "task_sample_packets" / "player_with_ball" / "README.md")

            self.assertEqual(registry["aerial"].skill, "point")
            self.assertEqual(registry["aerial"].prompt, "airplane")
            self.assertEqual(registry["aerial"].aerial_packet_path, aerial_dir / "samples.before_after.json")
            self.assertEqual(registry["aerial"].aerial_benchmark_path, aerial_dir / "benchmark_packet.json")
            self.assertEqual(registry["aerial"].dataset_id, "maxs-m87/aerial_airport_point_v2")
            self.assertEqual(registry["aerial"].split, "test")


class OpenRouterTaskPacketReferenceFallbackTests(unittest.TestCase):
    def test_load_reference_metrics_uses_player_readme_when_compare_json_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            readme = root / "outputs" / "task_sample_packets" / "player_with_ball" / "README.md"
            readme.parent.mkdir(parents=True)
            readme.write_text(
                "\n".join(
                    [
                        "# Player With Ball",
                        "",
                        "- sample count: `20`",
                        "",
                        "| Metric | Baseline | Finetuned `@200` | Delta |",
                        "| --- | ---: | ---: | ---: |",
                        "| F1 | 0.4889 | 0.7368 | +0.2480 |",
                        "| Macro F1 | 0.5091 | 0.7000 | +0.1909 |",
                        "| mIoU | 0.4129 | 0.5137 | +0.1008 |",
                        "| True Positives | 11 | 14 | +3 |",
                        "| False Positives | 16 | 6 | -10 |",
                        "| False Negatives | 7 | 4 | -3 |",
                    ]
                ),
                encoding="utf-8",
            )
            spec = mod.TaskSpec(
                name="player_with_ball",
                skill="detect",
                prompt="Player with the ball",
                task_dir=readme.parent,
                source_kind="cached_hf_parquet",
                dataset_id="maxs-m87/Ball-Holder-splits-v1",
                split="test",
                iou_threshold=0.4,
                reference_metrics_path=None,
                reference_samples_path=None,
                reference_compare_path=None,
                reference_readme_path=readme,
                aerial_subset_dataset_path=None,
                aerial_packet_path=None,
                aerial_benchmark_path=None,
            )
            with patch.object(mod, "REPO_ROOT", root):
                before, after, source = mod.load_reference_metrics(spec)
            self.assertEqual(source, "outputs/task_sample_packets/player_with_ball/README.md")
            self.assertEqual(before["samples"], 20)
            self.assertAlmostEqual(before["eval_f1"], 0.4889)
            self.assertEqual(after["tp"], 14)
            self.assertEqual(after["fp"], 6)

    def test_load_reference_metrics_uses_aerial_benchmark_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_path = root / "outputs" / "task_sample_packets" / "aerial" / "benchmark_packet.json"
            benchmark_path.parent.mkdir(parents=True)
            benchmark_path.write_text(
                json.dumps(
                    {
                        "baseline": {"eval_f1": 0.3, "eval_f1_macro": 0.4, "eval_miou": 0.0, "tp": 3, "fp": 7, "fn": 2, "tasks": 5},
                        "candidate": {"eval_f1": 0.7, "eval_f1_macro": 0.75, "eval_miou": 0.0, "tp": 7, "fp": 1, "fn": 1, "tasks": 5},
                    }
                ),
                encoding="utf-8",
            )
            spec = mod.TaskSpec(
                name="aerial",
                skill="point",
                prompt="airplane",
                task_dir=benchmark_path.parent,
                source_kind="cached_hf_parquet_packet_benchmark",
                dataset_id="maxs-m87/aerial_airport_point_v2",
                split="test",
                iou_threshold=0.0,
                reference_metrics_path=None,
                reference_samples_path=None,
                reference_compare_path=None,
                reference_readme_path=benchmark_path.parent / "README.md",
                aerial_subset_dataset_path=None,
                aerial_packet_path=None,
                aerial_benchmark_path=benchmark_path,
            )
            with patch.object(mod, "REPO_ROOT", root):
                before, after, source = mod.load_reference_metrics(spec)
            self.assertEqual(source, "outputs/task_sample_packets/aerial/benchmark_packet.json")
            self.assertEqual(before["samples"], 5)
            self.assertEqual(after["tp"], 7)

    def test_canonical_aerial_sample_key_matches_viz_and_dataset_names(self) -> None:
        viz_name = "0000_airport_175_jpg_rf_vgds5p9vjdfdilkszv8g_airplane.jpg"
        dataset_name = "airport_175_jpg.rf.vGDS5p9VJDFdILkSZv8G.jpg"
        self.assertEqual(mod._canonical_aerial_sample_key(viz_name), mod._canonical_aerial_sample_key(dataset_name))


class OpenRouterTaskPacketMappingTests(unittest.TestCase):
    def test_merge_packet_samples_with_predictions_raises_on_missing_sample_id(self) -> None:
        with self.assertRaises(KeyError):
            mod.merge_packet_samples_with_predictions(
                base_records=[
                    {"sample_id": "a"},
                    {"sample_id": "b"},
                ],
                prediction_records=[
                    {"sample_id": "a", "skill": "detect", "model": "openai/gpt-5.4", "prompt": "x", "task_f1": 1.0, "task_miou": 1.0, "tp": 1, "fp": 0, "fn": 0, "pred_count": 1, "gt_count": 1, "latency_ms": 10.0, "parse_success": True, "failed": False, "error": None, "pred_boxes": [], "pred_points": []}
                ],
            )

    def test_merge_packet_samples_with_predictions_preserves_order(self) -> None:
        merged = mod.merge_packet_samples_with_predictions(
            base_records=[
                {"sample_id": "b", "value": 2},
                {"sample_id": "a", "value": 1},
            ],
            prediction_records=[
                {"sample_id": "a", "skill": "detect", "model": "openai/gpt-5.4", "prompt": "x", "task_f1": 0.0, "task_miou": 0.0, "tp": 0, "fp": 0, "fn": 1, "pred_count": 0, "gt_count": 1, "latency_ms": 10.0, "parse_success": True, "failed": False, "error": None, "pred_boxes": [], "pred_points": []},
                {"sample_id": "b", "skill": "detect", "model": "openai/gpt-5.4", "prompt": "x", "task_f1": 1.0, "task_miou": 1.0, "tp": 1, "fp": 0, "fn": 0, "pred_count": 1, "gt_count": 1, "latency_ms": 20.0, "parse_success": True, "failed": False, "error": None, "pred_boxes": [], "pred_points": []},
            ],
        )
        self.assertEqual([item["sample_id"] for item in merged], ["b", "a"])
        self.assertIn("gpt_5_4", merged[0])
        self.assertEqual(merged[0]["value"], 2)


class OpenRouterTaskPacketMetricTests(unittest.TestCase):
    def test_aggregate_detection_metrics_handles_empty_and_present_cases(self) -> None:
        records = [
            {
                "sample_id": "empty_ok",
                "ground_truth_boxes": [],
                "pred_boxes": [],
                "pred_points": [],
                "latency_ms": 10.0,
                "parse_success": True,
                "failed": False,
            },
            {
                "sample_id": "miss",
                "ground_truth_boxes": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.3, "y_max": 0.3}],
                "pred_boxes": [],
                "pred_points": [],
                "latency_ms": 20.0,
                "parse_success": True,
                "failed": False,
            },
            {
                "sample_id": "hit",
                "ground_truth_boxes": [{"x_min": 0.4, "y_min": 0.4, "x_max": 0.6, "y_max": 0.6}],
                "pred_boxes": [{"x_min": 0.4, "y_min": 0.4, "x_max": 0.6, "y_max": 0.6}],
                "pred_points": [],
                "latency_ms": 30.0,
                "parse_success": False,
                "failed": False,
            },
        ]
        metrics = mod.aggregate_prediction_metrics(records, skill="detect", iou_threshold=0.4)
        self.assertEqual(metrics["samples"], 3)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["eval_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["parse_success_rate"], 2.0 / 3.0)

    def test_aggregate_point_metrics_handles_multi_object_case(self) -> None:
        records = [
            {
                "sample_id": "multi",
                "ground_truth_boxes": [
                    {"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2},
                    {"x_min": 0.7, "y_min": 0.7, "x_max": 0.8, "y_max": 0.8},
                ],
                "pred_boxes": [],
                "pred_points": [{"x": 0.15, "y": 0.15}, {"x": 0.75, "y": 0.75}],
                "latency_ms": 15.0,
                "parse_success": True,
                "failed": False,
            }
        ]
        metrics = mod.aggregate_prediction_metrics(records, skill="point", iou_threshold=0.0)
        self.assertEqual(metrics["samples"], 1)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertAlmostEqual(metrics["eval_f1"], 1.0)
        self.assertAlmostEqual(metrics["eval_miou"], 0.0)


class OpenRouterTaskPacketSummaryTests(unittest.TestCase):
    def test_build_task_summary_includes_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "outputs" / "task_sample_packets" / "state_farm"
            ref_dir = task_dir / "ref"
            ref_dir.mkdir(parents=True)
            (ref_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "baseline": {"eval_f1": 0.2, "eval_f1_macro": 0.3, "eval_miou": 0.1, "eval_true_pos": 1, "eval_false_pos": 4, "eval_false_neg": 3, "eval_samples": 5},
                        "checkpoint": {"eval_f1": 0.8, "eval_f1_macro": 0.8, "eval_miou": 0.7, "eval_true_pos": 4, "eval_false_pos": 0, "eval_false_neg": 1, "eval_samples": 5},
                    }
                ),
                encoding="utf-8",
            )
            spec = mod.TaskSpec(
                name="state_farm",
                skill="detect",
                prompt="State Farm logo",
                task_dir=task_dir,
                source_kind="cached_hf_parquet",
                dataset_id="maxs-m87/NBA_StateFarm_Splits_01",
                split="validation",
                iou_threshold=0.4,
                reference_metrics_path=ref_dir / "metrics.json",
                reference_samples_path=ref_dir / "sample_before_after.json",
                reference_compare_path=None,
                reference_readme_path=task_dir / "README.md",
                aerial_subset_dataset_path=None,
                aerial_packet_path=None,
                aerial_benchmark_path=None,
            )
            gpt_metrics = {
                "eval_f1": 0.5,
                "eval_f1_macro": 0.55,
                "eval_miou": 0.45,
                "tp": 3,
                "fp": 1,
                "fn": 2,
                "samples": 5,
                "failed_samples": 0,
                "parse_success_rate": 1.0,
                "avg_latency_ms": 100.0,
            }
            with patch.object(mod, "REPO_ROOT", root):
                before, after, _ = mod.load_reference_metrics(spec)
                summary = mod.build_task_summary(
                    spec=spec,
                    reference_before=before,
                    reference_after=after,
                    gpt_metrics=gpt_metrics,
                )
            self.assertEqual(summary["reference_source"], "outputs/task_sample_packets/state_farm/ref/metrics.json")
            self.assertAlmostEqual(summary["delta_vs_reference"]["before"]["eval_f1"], 0.3)
            self.assertAlmostEqual(summary["delta_vs_reference"]["after"]["eval_miou"], -0.25)


if __name__ == "__main__":
    unittest.main()
