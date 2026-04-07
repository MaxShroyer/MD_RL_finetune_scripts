from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rerun_task_packet_finetune_exports as rerun_mod
import task_packet_benchmark_common as common


class _FakeHTTPResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _image_payload(path: str) -> dict[str, object]:
    image = Image.new("RGB", (16, 12), color=(128, 64, 32))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return {"path": path, "bytes": buf.getvalue()}


class TaskPacketLoaderTests(unittest.TestCase):
    def test_load_player_with_ball_samples_maps_packet_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "outputs" / "task_sample_packets" / "player_with_ball"
            imgs_dir = task_dir / "imgs"
            imgs_dir.mkdir(parents=True)
            (imgs_dir / "001_alpha.jpg").write_bytes(b"packet")
            (imgs_dir / "002_beta.jpg").write_bytes(b"packet")

            rows = [
                {"image": _image_payload("alpha.jpg"), "answer_boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.6}], "prompt": "", "type": "region"},
                {"image": _image_payload("beta.jpg"), "answer_boxes": [{"x_min": 0.2, "y_min": 0.3, "x_max": 0.5, "y_max": 0.7}], "prompt": "custom beta prompt", "type": "region"},
            ]
            spec = common.TaskSpec(
                name="player_with_ball",
                skill="detect",
                prompt="Player with the ball",
                task_dir=task_dir,
                source_kind="cached_hf_parquet",
                dataset_id="dataset/player",
                split="test",
                iou_threshold=0.4,
                reference_metrics_path=None,
                reference_samples_path=None,
                reference_compare_path=None,
                reference_readme_path=task_dir / "README.md",
                aerial_subset_dataset_path=None,
                aerial_packet_path=None,
                aerial_benchmark_path=None,
            )
            with patch.object(common, "_find_cached_parquet_path", return_value=root / "player.parquet"), patch.object(
                common,
                "_read_parquet_rows",
                return_value=rows,
            ):
                samples = common.load_player_with_ball_samples(spec, repo_root=root)

            self.assertEqual([sample.sample_id for sample in samples], ["alpha", "beta"])
            self.assertEqual(samples[0].packet_image_path, "outputs/task_sample_packets/player_with_ball/imgs/001_alpha.jpg")
            self.assertEqual(samples[0].prompt, "Player with the ball")
            self.assertEqual(samples[1].prompt, "custom beta prompt")

    def test_load_state_farm_samples_maps_packet_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "outputs" / "task_sample_packets" / "state_farm"
            imgs_dir = task_dir / "imgs"
            imgs_dir.mkdir(parents=True)
            (imgs_dir / "001_demo-id.jpg").write_bytes(b"packet")

            rows = [
                {
                    "image": _image_payload("demo-id.jpg"),
                    "answer_boxes": [{"x_min": 0.15, "y_min": 0.25, "x_max": 0.45, "y_max": 0.55}],
                    "prompt": "State Farm logo",
                    "type": "region",
                }
            ]
            spec = common.TaskSpec(
                name="state_farm",
                skill="detect",
                prompt="State Farm logo",
                task_dir=task_dir,
                source_kind="cached_hf_parquet",
                dataset_id="dataset/state",
                split="validation",
                iou_threshold=0.4,
                reference_metrics_path=None,
                reference_samples_path=None,
                reference_compare_path=None,
                reference_readme_path=task_dir / "README.md",
                aerial_subset_dataset_path=None,
                aerial_packet_path=None,
                aerial_benchmark_path=None,
            )
            with patch.object(common, "_find_cached_parquet_path", return_value=root / "state.parquet"), patch.object(
                common,
                "_read_parquet_rows",
                return_value=rows,
            ):
                samples = common.load_state_farm_samples(spec, repo_root=root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "demo-id")
            self.assertEqual(samples[0].packet_image_path, "outputs/task_sample_packets/state_farm/imgs/001_demo-id.jpg")

    def test_load_aerial_samples_maps_packet_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "outputs" / "task_sample_packets" / "aerial"
            imgs_dir = task_dir / "imgs"
            imgs_dir.mkdir(parents=True)
            packet_name = "001_airport_303_jpg.rf.oFfWpDSMSMJJUvsn2qWI.jpg"
            (imgs_dir / packet_name).write_bytes(b"packet")

            row = {
                "image": _image_payload("airport_303_jpg.rf.oFfWpDSMSMJJUvsn2qWI.jpg"),
                "source_image_id": "airport_303_jpg.rf.oFfWpDSMSMJJUvsn2qWI.jpg",
                "answer_boxes": [{"x_min": 0.05, "y_min": 0.15, "x_max": 0.25, "y_max": 0.35}],
                "prompt": "airplanes",
                "source_split": "test",
            }
            spec = common.TaskSpec(
                name="aerial",
                skill="point",
                prompt="airplane",
                task_dir=task_dir,
                source_kind="cached_hf_parquet_packet_benchmark",
                dataset_id="dataset/aerial",
                split="test",
                iou_threshold=0.0,
                reference_metrics_path=None,
                reference_samples_path=None,
                reference_compare_path=None,
                reference_readme_path=task_dir / "README.md",
                aerial_subset_dataset_path=None,
                aerial_packet_path=None,
                aerial_benchmark_path=None,
            )

            def _fake_find_cached_parquet_path(dataset_id: str, split: str) -> Path:
                if split != "test":
                    raise FileNotFoundError(split)
                return root / "aerial.parquet"

            with patch.object(common, "_find_cached_parquet_path", side_effect=_fake_find_cached_parquet_path), patch.object(
                common,
                "_read_parquet_rows",
                return_value=[row],
            ):
                samples = common.load_aerial_samples(spec, repo_root=root)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "airport_303_jpg.rf.oFfWpDSMSMJJUvsn2qWI.jpg")
            self.assertEqual(samples[0].packet_image_path, f"outputs/task_sample_packets/aerial/imgs/{packet_name}")
            self.assertEqual(samples[0].prompt, "airplanes")


class FineuneRerunApiTests(unittest.TestCase):
    def test_call_detect_api_raw_preserves_raw_response(self) -> None:
        body = json.dumps(
            {
                "objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}],
                "metrics": {"output_tokens": 4},
            }
        )
        image = Image.new("RGB", (100, 50), color=(255, 255, 255))
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            boxes, raw_response, latency_ms = rerun_mod.call_detect_api_raw(
                api_base="https://api-staging.moondream.ai/v1",
                api_key="test-key",
                model="moondream3-preview",
                image=image,
                prompt="State Farm logo",
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
                max_objects=50,
                timeout=5.0,
                retry_429_max_retries=0,
                retry_429_backoff_s=0.0,
                retry_429_max_backoff_s=0.0,
                retry_timeout_max_retries=0,
                retry_timeout_backoff_s=0.0,
                retry_timeout_max_backoff_s=0.0,
            )
        self.assertEqual(len(boxes), 1)
        self.assertEqual(raw_response["metrics"]["output_tokens"], 4)
        self.assertGreaterEqual(latency_ms, 0.0)

    def test_call_point_api_raw_preserves_raw_response(self) -> None:
        body = json.dumps(
            {
                "points": [{"x": 0.25, "y": 0.75}],
                "metrics": {"output_tokens": 2},
            }
        )
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(body)):
            points, raw_response, latency_ms = rerun_mod.call_point_api_raw(
                api_base="https://api-staging.moondream.ai/v1",
                api_key="test-key",
                model="moondream3-preview",
                image=image,
                prompt="airplane",
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
                timeout=5.0,
                retry_429_max_retries=0,
                retry_429_backoff_s=0.0,
                retry_429_max_backoff_s=0.0,
                retry_timeout_max_retries=0,
                retry_timeout_backoff_s=0.0,
                retry_timeout_max_backoff_s=0.0,
            )
        self.assertEqual(len(points), 1)
        self.assertEqual(raw_response["metrics"]["output_tokens"], 2)
        self.assertGreaterEqual(latency_ms, 0.0)


class FineuneRerunRecordTests(unittest.TestCase):
    def test_evaluate_packet_sample_failure_records_error(self) -> None:
        sample = common.Sample(
            sample_index=1,
            sample_id="alpha",
            source_image_path="alpha.jpg",
            packet_image_path="outputs/task_sample_packets/player_with_ball/imgs/001_alpha.jpg",
            prompt="Player with the ball",
            task_type="region",
            notes="",
            timestamp="",
            image=Image.new("RGB", (32, 32), color=(0, 0, 0)),
            ground_truth_boxes=[common.Box(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.4)],
            base_record={},
        )
        spec = common.TaskSpec(
            name="player_with_ball",
            skill="detect",
            prompt="Player with the ball",
            task_dir=REPO_ROOT / "outputs" / "task_sample_packets" / "player_with_ball",
            source_kind="cached_hf_parquet",
            dataset_id="dataset/player",
            split="test",
            iou_threshold=0.4,
            reference_metrics_path=None,
            reference_samples_path=None,
            reference_compare_path=None,
            reference_readme_path=None,
            aerial_subset_dataset_path=None,
            aerial_packet_path=None,
            aerial_benchmark_path=None,
        )
        args = argparse.Namespace(
            api_base="https://api-staging.moondream.ai/v1",
            api_key="test-key",
            temperature=0.0,
            top_p=1.0,
            max_tokens_detect=256,
            max_tokens_point=256,
            max_objects=50,
            timeout=5.0,
            retry_429_max_retries=0,
            retry_429_backoff_s=0.0,
            retry_429_max_backoff_s=0.0,
            retry_timeout_max_retries=0,
            retry_timeout_backoff_s=0.0,
            retry_timeout_max_backoff_s=0.0,
        )
        with patch.object(rerun_mod, "call_detect_api_raw", side_effect=RuntimeError("boom")):
            record = rerun_mod.evaluate_packet_sample(
                sample=sample,
                spec=spec,
                run_label="baseline",
                model="moondream3-preview",
                finetune_id=None,
                checkpoint_step=None,
                args=args,
            )

        self.assertTrue(record["failed"])
        self.assertIn("boom", record["error"])
        self.assertEqual(record["pred_boxes"], [])
        self.assertIsNone(record["raw_response"])


class SimpleTaskPacketExportTests(unittest.TestCase):
    def test_build_simple_task_samples_detect(self) -> None:
        merged = common.build_simple_task_samples(
            task_dir=REPO_ROOT / "outputs" / "task_sample_packets" / "player_with_ball",
            repo_root=REPO_ROOT,
            skill="detect",
            gpt_packet_samples=[
                {
                    "sample_id": "alpha",
                    "packet_image_path": "outputs/task_sample_packets/player_with_ball/imgs/001_alpha.jpg",
                    "prompt": "Player with the ball",
                    "ground_truth_boxes": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}],
                    "gpt_5_4": {
                        "pred_boxes": [{"x_min": 0.11, "y_min": 0.11, "x_max": 0.39, "y_max": 0.39}],
                    },
                }
            ],
            baseline_records=[
                {
                    "sample_id": "alpha",
                    "pred_boxes": [{"x_min": 0.12, "y_min": 0.12, "x_max": 0.38, "y_max": 0.38}],
                }
            ],
            checkpoint_records=[
                {
                    "sample_id": "alpha",
                    "pred_boxes": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}],
                }
            ],
        )

        self.assertEqual(
            merged,
            [
                {
                    "image": "imgs/001_alpha.jpg",
                    "prompt": "Player with the ball",
                    "gt": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}],
                    "baseline": [{"x_min": 0.12, "y_min": 0.12, "x_max": 0.38, "y_max": 0.38}],
                    "finetune": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}],
                    "gpt_5.4": [{"x_min": 0.11, "y_min": 0.11, "x_max": 0.39, "y_max": 0.39}],
                }
            ],
        )

    def test_refresh_simple_task_samples_reads_latest_rerun_and_writes_point_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "outputs" / "task_sample_packets" / "aerial"
            rerun_dir = task_dir / "reruns" / "20260331_123456"
            rerun_dir.mkdir(parents=True)

            gpt_packet_path = task_dir / "openrouter_gpt_5_4_point.packet_samples.json"
            gpt_packet_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "airport_alpha",
                            "packet_image_path": "outputs/task_sample_packets/aerial/imgs/001_airport_alpha.jpg",
                            "prompt": "airplane",
                            "ground_truth_boxes": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}],
                            "gpt_5_4": {
                                "pred_points": [{"x": 0.15, "y": 0.15}],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (rerun_dir / "baseline.records.jsonl").write_text(
                json.dumps({"sample_id": "airport_alpha", "pred_points": [{"x": 0.3, "y": 0.3}]}) + "\n",
                encoding="utf-8",
            )
            (rerun_dir / "checkpoint.records.jsonl").write_text(
                json.dumps({"sample_id": "airport_alpha", "pred_points": [{"x": 0.12, "y": 0.12}]}) + "\n",
                encoding="utf-8",
            )
            (rerun_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "generated_utc": "2026-04-01T00:00:00+00:00",
                        "baseline": {
                            "records_jsonl": "outputs/task_sample_packets/aerial/reruns/20260331_123456/baseline.records.jsonl"
                        },
                        "checkpoint": {
                            "records_jsonl": "outputs/task_sample_packets/aerial/reruns/20260331_123456/checkpoint.records.jsonl"
                        },
                    }
                ),
                encoding="utf-8",
            )

            output_path = common.refresh_simple_task_samples(
                task_dir=task_dir,
                skill="point",
                repo_root=root,
            )

            self.assertEqual(output_path, task_dir / "samples.json")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload,
                [
                    {
                        "image": "imgs/001_airport_alpha.jpg",
                        "prompt": "airplane",
                        "gt": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.2, "y_max": 0.2}],
                        "baseline": [{"x": 0.3, "y": 0.3}],
                        "finetune": [{"x": 0.12, "y": 0.12}],
                        "gpt_5.4": [{"x": 0.15, "y": 0.15}],
                    }
                ],
            )

    def test_refresh_simple_task_samples_uses_manual_checkpoint_bbox_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "outputs" / "task_sample_packets" / "state_farm"
            raw_dir = task_dir / "raw_json"
            raw_dir.mkdir(parents=True)

            (raw_dir / "openrouter_gpt_5_4_detect.packet_samples.json").write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "alpha",
                            "packet_image_path": "outputs/task_sample_packets/state_farm/imgs/001_alpha.jpg",
                            "prompt": "State Farm logo",
                            "ground_truth_boxes": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}],
                            "gpt_5_4": {
                                "pred_boxes": [{"x_min": 0.11, "y_min": 0.21, "x_max": 0.31, "y_max": 0.41}],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (raw_dir / "baseline.records.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "alpha",
                        "failed": False,
                        "pred_boxes": [{"x_min": 0.12, "y_min": 0.22, "x_max": 0.32, "y_max": 0.42}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (raw_dir / "checkpoint.records.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "alpha",
                        "failed": True,
                        "pred_boxes": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (raw_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "generated_utc": "2026-04-01T00:00:00+00:00",
                        "baseline": {"records_jsonl": "missing/baseline.records.jsonl"},
                        "checkpoint": {"records_jsonl": "missing/checkpoint.records.jsonl"},
                    }
                ),
                encoding="utf-8",
            )
            (raw_dir / "use this for checkpoint bboxs.json").write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "alpha",
                            "baseline": {
                                "pred_boxes": [{"x_min": 0.13, "y_min": 0.23, "x_max": 0.33, "y_max": 0.43}]
                            },
                            "after": {
                                "pred_boxes": [{"x_min": 0.14, "y_min": 0.24, "x_max": 0.34, "y_max": 0.44}]
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            output_path = common.refresh_simple_task_samples(
                task_dir=task_dir,
                skill="detect",
                repo_root=root,
            )

            self.assertEqual(output_path, task_dir / "samples.json")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload,
                [
                    {
                        "image": "imgs/001_alpha.jpg",
                        "prompt": "State Farm logo",
                        "gt": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}],
                        "baseline": [{"x_min": 0.12, "y_min": 0.22, "x_max": 0.32, "y_max": 0.42}],
                        "finetune": [{"x_min": 0.14, "y_min": 0.24, "x_max": 0.34, "y_max": 0.44}],
                        "gpt_5.4": [{"x_min": 0.11, "y_min": 0.21, "x_max": 0.31, "y_max": 0.41}],
                    }
                ],
            )


class FineuneRerunOutputTests(unittest.TestCase):
    def test_run_task_writes_timestamped_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_dir = root / "outputs" / "task_sample_packets" / "player_with_ball"
            task_dir.mkdir(parents=True)
            sample = common.Sample(
                sample_index=1,
                sample_id="alpha",
                source_image_path="alpha.jpg",
                packet_image_path="outputs/task_sample_packets/player_with_ball/imgs/001_alpha.jpg",
                prompt="Player with the ball",
                task_type="region",
                notes="",
                timestamp="",
                image=Image.new("RGB", (32, 32), color=(0, 0, 0)),
                ground_truth_boxes=[common.Box(x_min=0.1, y_min=0.1, x_max=0.4, y_max=0.4)],
                base_record={},
            )
            spec = common.TaskSpec(
                name="player_with_ball",
                skill="detect",
                prompt="Player with the ball",
                task_dir=task_dir,
                source_kind="cached_hf_parquet",
                dataset_id="dataset/player",
                split="test",
                iou_threshold=0.4,
                reference_metrics_path=None,
                reference_samples_path=None,
                reference_compare_path=None,
                reference_readme_path=None,
                aerial_subset_dataset_path=None,
                aerial_packet_path=None,
                aerial_benchmark_path=None,
                finetune_id="ft-player",
                checkpoint_step=60,
            )
            args = argparse.Namespace(
                baseline_model="moondream3-preview",
                base_model="moondream3-preview",
                api_base="https://api-staging.moondream.ai/v1",
            )

            def _fake_eval(*, run_label: str, model: str, finetune_id: str | None, checkpoint_step: int | None, **_: object) -> dict[str, object]:
                return {
                    "task": "player_with_ball",
                    "skill": "detect",
                    "run_label": run_label,
                    "model": model,
                    "finetune_id": finetune_id,
                    "checkpoint_step": checkpoint_step,
                    "sample_index": 1,
                    "sample_id": "alpha",
                    "prompt": "Player with the ball",
                    "source_image_path": "alpha.jpg",
                    "packet_image_path": "outputs/task_sample_packets/player_with_ball/imgs/001_alpha.jpg",
                    "packet_image_abspath": str((root / "outputs" / "task_sample_packets" / "player_with_ball" / "imgs" / "001_alpha.jpg").resolve()),
                    "ground_truth_boxes": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}],
                    "pred_boxes": [{"x_min": 0.12, "y_min": 0.12, "x_max": 0.39, "y_max": 0.39}],
                    "pred_points": [],
                    "task_f1": 1.0,
                    "task_miou": 0.8,
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "latency_ms": 10.0,
                    "failed": False,
                    "error": None,
                    "raw_response": {"objects": []},
                    "task_type": "region",
                    "notes": "",
                    "timestamp": "",
                }

            with patch.object(rerun_mod, "REPO_ROOT", root), patch.object(
                rerun_mod.task_packet_common,
                "load_task_samples",
                return_value=[sample],
            ), patch.object(rerun_mod, "evaluate_packet_sample", side_effect=_fake_eval):
                manifest = rerun_mod.run_task(spec, args, run_stamp="20260331_123456")

            output_dir = task_dir / "reruns" / "20260331_123456"
            self.assertTrue((output_dir / "baseline.records.jsonl").exists())
            self.assertTrue((output_dir / "checkpoint.records.jsonl").exists())
            self.assertTrue((output_dir / "baseline.metrics.json").exists())
            self.assertTrue((output_dir / "checkpoint.metrics.json").exists())
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertEqual(manifest["output_dir"], "outputs/task_sample_packets/player_with_ball/reruns/20260331_123456")


class TaskPacketMetricTests(unittest.TestCase):
    def test_detect_aggregate_metrics(self) -> None:
        records = [
            {
                "failed": False,
                "ground_truth_boxes": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}],
                "pred_boxes": [{"x_min": 0.1, "y_min": 0.1, "x_max": 0.4, "y_max": 0.4}],
                "pred_points": [],
                "latency_ms": 12.0,
            },
            {
                "failed": False,
                "ground_truth_boxes": [{"x_min": 0.5, "y_min": 0.5, "x_max": 0.8, "y_max": 0.8}],
                "pred_boxes": [],
                "pred_points": [],
                "latency_ms": 18.0,
            },
        ]
        metrics = common.aggregate_prediction_metrics(records, skill="detect", iou_threshold=0.4)
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["eval_f1"], 2.0 / 3.0)

    def test_point_aggregate_metrics(self) -> None:
        records = [
            {
                "failed": False,
                "ground_truth_boxes": [{"x_min": 0.2, "y_min": 0.2, "x_max": 0.4, "y_max": 0.4}],
                "pred_boxes": [],
                "pred_points": [{"x": 0.3, "y": 0.3}],
                "latency_ms": 7.0,
            }
        ]
        metrics = common.aggregate_prediction_metrics(records, skill="point", iou_threshold=0.0)
        self.assertEqual(metrics["samples"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertAlmostEqual(metrics["eval_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
