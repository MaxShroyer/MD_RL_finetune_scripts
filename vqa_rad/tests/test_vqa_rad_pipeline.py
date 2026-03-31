from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vqa_rad import benchmark_vqa_rad_query as bench_mod
from vqa_rad import build_vqa_rad_hf_dataset as build_mod
from vqa_rad import common
from vqa_rad import train_vqa_rad_query as train_mod


def _solid_image(color: str) -> Image.Image:
    return Image.new("RGB", (8, 8), color=color)


def _write_example_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _solid_image(color).save(path, format="PNG")


def _write_jsonl_split(dataset_dir: Path, split_name: str, rows: list[dict[str, object]]) -> None:
    jsonl_dir = dataset_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    (jsonl_dir / f"{split_name}.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset_row(
    *,
    dataset_dir: Path,
    split_name: str,
    image_name: str,
    color: str,
    question: str,
    answer_text: str,
    question_family: str,
    prompt_style: str = common.PROMPT_STYLE_LEGACY_JSON,
) -> dict[str, object]:
    image_path = dataset_dir / "images" / image_name
    _write_example_image(image_path, color)
    image_group_id = image_name.removesuffix(".png")
    return {
        "row_id": f"{split_name}_{image_group_id}",
        "split": split_name,
        "source_split": split_name,
        "task_type": common.QUERY_TASK_TYPE,
        "question": common.make_prompt(question, prompt_style=prompt_style),
        "question_family": question_family,
        "answer_type": common.infer_answer_type(answer_text),
        "answer_text": answer_text,
        "final_answer_json": json.dumps({"answer": answer_text}),
        "image_path": common.relative_path(dataset_dir, image_path),
        "image_group_id": image_group_id,
    }


class VQARadBuilderTests(unittest.TestCase):
    def test_assign_local_split_is_sticky_per_image_group(self) -> None:
        first = build_mod._assign_local_split(seed=42, image_group_id="abc", val_fraction=0.15)
        second = build_mod._assign_local_split(seed=42, image_group_id="abc", val_fraction=0.15)
        self.assertEqual(first, second)

    def test_streaming_build_dedupes_and_blocks_test_overlap(self) -> None:
        shared_image = _solid_image("red")
        unique_image = _solid_image("blue")
        rows_by_split = {
            "test": [
                {"image": shared_image.copy(), "question": "are the lungs normal appearing?", "answer": "no"},
                {"image": _solid_image("green"), "question": "what is the location of the mass?", "answer": "pineal region"},
            ],
            "train": [
                {"image": shared_image.copy(), "question": "are the lungs normal appearing?", "answer": "no"},
                {"image": unique_image.copy(), "question": "are there pulmonary findings?", "answer": "no"},
                {"image": unique_image.copy(), "question": "how was this image taken?", "answer": "mri"},
                {"image": unique_image.copy(), "question": "how was this image taken?", "answer": "mri"},
            ],
        }

        def fake_iter_source_rows(*, dataset_name: str, split_name: str, hf_token: str, hf_cache_dir: str):
            del dataset_name, hf_token, hf_cache_dir
            for row in rows_by_split[split_name]:
                yield dict(row)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(build_mod, "_iter_source_rows", side_effect=fake_iter_source_rows):
            output_dir = Path(tmpdir) / "dataset"
            shared_dir = Path(tmpdir) / "shared_images"
            build_mod._build_query_jsonl_dataset_streaming(
                dataset_name=common.DEFAULT_DATASET_NAME,
                hf_token="",
                hf_cache_dir="",
                output_dir=output_dir,
                shared_image_dir=shared_dir,
                val_fraction=0.5,
                seed=7,
                prompt_style=common.PROMPT_STYLE_LEGACY_JSON,
            )

            train_rows = [
                json.loads(line)
                for line in (output_dir / "jsonl" / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            val_rows = [
                json.loads(line)
                for line in (output_dir / "jsonl" / "validation.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            test_rows = [
                json.loads(line)
                for line in (output_dir / "jsonl" / "test.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(test_rows), 2)
            self.assertEqual(len(train_rows) + len(val_rows), 2)
            combined_train_rows = train_rows + val_rows
            self.assertEqual(len({row["image_group_id"] for row in combined_train_rows}), 1)
            self.assertEqual(
                {row["question"] for row in combined_train_rows},
                {
                    common.make_prompt("are there pulmonary findings?"),
                    common.make_prompt("how was this image taken?"),
                },
            )
            self.assertTrue(all(row["task_type"] == common.QUERY_TASK_TYPE for row in combined_train_rows))
            self.assertTrue(all(json.loads(row["final_answer_json"]).get("answer") for row in combined_train_rows))

            stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(stats["skipped"]["train_test_image_overlap"], 1)
            duplicate_keys = [key for key in stats["skipped"] if key.startswith("duplicate_")]
            self.assertTrue(duplicate_keys)

    def test_streaming_build_raw_question_prompt_style(self) -> None:
        rows_by_split = {
            "test": [
                {"image": _solid_image("white"), "question": "is this an mri or a ct scan?", "answer": "mri"},
            ],
            "train": [],
        }

        def fake_iter_source_rows(*, dataset_name: str, split_name: str, hf_token: str, hf_cache_dir: str):
            del dataset_name, hf_token, hf_cache_dir
            for row in rows_by_split[split_name]:
                yield dict(row)

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(build_mod, "_iter_source_rows", side_effect=fake_iter_source_rows):
            output_dir = Path(tmpdir) / "dataset"
            shared_dir = Path(tmpdir) / "shared_images"
            build_mod._build_query_jsonl_dataset_streaming(
                dataset_name=common.DEFAULT_DATASET_NAME,
                hf_token="",
                hf_cache_dir="",
                output_dir=output_dir,
                shared_image_dir=shared_dir,
                val_fraction=0.15,
                seed=42,
                prompt_style=common.PROMPT_STYLE_RAW_QUESTION,
            )
            test_rows = [
                json.loads(line)
                for line in (output_dir / "jsonl" / "test.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(test_rows[0]["question"], "is this an mri or a ct scan?")
            metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["prompt_style"], common.PROMPT_STYLE_RAW_QUESTION)
            self.assertEqual(metadata["prediction_formats"], ["plain_text", "json_object"])
            self.assertEqual(metadata["question_template"], "{question}")


class VQARadRewardTests(unittest.TestCase):
    def _example(self, *, answer_text: str, answer_type: str, question_family: str = "other") -> train_mod.VQARadExample:
        return train_mod.VQARadExample(
            row_id="row1",
            split="validation",
            source_split="train",
            task_type=common.QUERY_TASK_TYPE,
            question="q",
            image_path=Path("/tmp/example.png"),
            question_family=question_family,
            answer_type=answer_type,
            answer_text=answer_text,
            normalized_answer=common.normalize_close_answer(answer_text)
            if answer_type == common.ANSWER_TYPE_CLOSE
            else common.normalize_open_answer(answer_text),
            image_group_id="img1",
        )

    def test_binary_reward_exact_match(self) -> None:
        example = self._example(answer_text="yes", answer_type=common.ANSWER_TYPE_CLOSE, question_family="yes_no")
        outcome = train_mod._score_answer_text(
            example,
            '{"answer":"yes"}',
            open_exact_weight=0.55,
            open_token_f1_weight=0.25,
            open_numeric_weight=0.10,
            open_brevity_weight=0.05,
        )
        self.assertTrue(outcome.parse_success)
        self.assertTrue(outcome.task_correct)
        self.assertEqual(outcome.reward, 1.0)
        self.assertEqual(outcome.close_accuracy, 1.0)

    def test_binary_wrong_answer_gets_parse_bonus_only(self) -> None:
        example = self._example(answer_text="no", answer_type=common.ANSWER_TYPE_CLOSE, question_family="yes_no")
        outcome = train_mod._score_answer_text(
            example,
            '{"answer":"yes"}',
            open_exact_weight=0.55,
            open_token_f1_weight=0.25,
            open_numeric_weight=0.10,
            open_brevity_weight=0.05,
        )
        self.assertTrue(outcome.parse_success)
        self.assertFalse(outcome.task_correct)
        self.assertAlmostEqual(outcome.reward, 0.05, places=6)

    def test_open_alias_match_counts_as_exact(self) -> None:
        example = self._example(answer_text="mri", answer_type=common.ANSWER_TYPE_OPEN, question_family="modality")
        outcome = train_mod._score_answer_text(
            example,
            '{"answer":"magnetic resonance imaging"}',
            open_exact_weight=0.55,
            open_token_f1_weight=0.25,
            open_numeric_weight=0.10,
            open_brevity_weight=0.05,
        )
        self.assertTrue(outcome.task_correct)
        self.assertEqual(outcome.open_accuracy, 1.0)
        self.assertGreaterEqual(outcome.reward, 0.95)

    def test_numeric_match_and_brevity_are_tracked(self) -> None:
        example = self._example(answer_text="3 cm", answer_type=common.ANSWER_TYPE_OPEN, question_family="count")
        outcome = train_mod._score_answer_text(
            example,
            '{"answer":"3 cm lesion in the left lung with additional descriptive text"}',
            open_exact_weight=0.55,
            open_token_f1_weight=0.25,
            open_numeric_weight=0.10,
            open_brevity_weight=0.05,
        )
        self.assertEqual(outcome.numeric_match, 1.0)
        self.assertLess(outcome.brevity_score, 1.0)

    def test_soft_json_fallback_parses_embedded_object(self) -> None:
        example = self._example(answer_text="yes", answer_type=common.ANSWER_TYPE_CLOSE, question_family="yes_no")
        outcome = train_mod._score_answer_text(
            example,
            'answer: {"answer":"yes"}',
            open_exact_weight=0.55,
            open_token_f1_weight=0.25,
            open_numeric_weight=0.10,
            open_brevity_weight=0.05,
        )
        self.assertTrue(outcome.parse_success)
        self.assertFalse(outcome.strict_parse_success)
        self.assertTrue(outcome.json_object_parsed)

    def test_plain_text_yes_no_answer_is_accepted(self) -> None:
        example = self._example(answer_text="yes", answer_type=common.ANSWER_TYPE_CLOSE, question_family="yes_no")
        outcome = train_mod._score_answer_text(
            example,
            "yes, there is an abnormality",
            open_exact_weight=0.55,
            open_token_f1_weight=0.25,
            open_numeric_weight=0.10,
            open_brevity_weight=0.05,
        )
        self.assertTrue(outcome.parse_success)
        self.assertFalse(outcome.json_object_parsed)
        self.assertFalse(outcome.strict_parse_success)
        self.assertEqual(outcome.prediction_format, common.PREDICTION_FORMAT_PLAIN_TEXT)
        self.assertEqual(outcome.normalized_prediction, "yes")
        self.assertEqual(outcome.reward, 1.0)

    def test_plain_text_label_cleanup_is_accepted(self) -> None:
        example = self._example(answer_text="mri", answer_type=common.ANSWER_TYPE_OPEN, question_family="modality")
        outcome = train_mod._score_answer_text(
            example,
            "Final answer: magnetic resonance imaging",
            open_exact_weight=0.55,
            open_token_f1_weight=0.25,
            open_numeric_weight=0.10,
            open_brevity_weight=0.05,
        )
        self.assertTrue(outcome.parse_success)
        self.assertFalse(outcome.json_object_parsed)
        self.assertEqual(outcome.prediction_format, common.PREDICTION_FORMAT_PLAIN_TEXT)
        self.assertEqual(outcome.open_accuracy, 1.0)

    def test_empty_plain_text_still_scores_zero(self) -> None:
        example = self._example(answer_text="mri", answer_type=common.ANSWER_TYPE_OPEN, question_family="modality")
        outcome = train_mod._score_answer_text(
            example,
            "   \n  ",
            open_exact_weight=0.55,
            open_token_f1_weight=0.25,
            open_numeric_weight=0.10,
            open_brevity_weight=0.05,
        )
        self.assertFalse(outcome.parse_success)
        self.assertEqual(outcome.prediction_format, common.PREDICTION_FORMAT_NONE)
        self.assertEqual(outcome.reward, 0.0)


class VQARadConfigTests(unittest.TestCase):
    def test_all_configs_parse(self) -> None:
        config_root = REPO_ROOT / "vqa_rad" / "configs"
        for config_path in sorted(config_root.rglob("*.json")):
            with self.subTest(config=str(config_path.relative_to(REPO_ROOT))):
                name = config_path.name
                if name.startswith("build_"):
                    args = build_mod.parse_args(["--config", str(config_path)])
                    if "_v2" in name:
                        self.assertTrue(str(args.output_dir).endswith("vqa_rad/outputs/vqa_rad_query_v2"))
                        self.assertEqual(args.prompt_style, common.PROMPT_STYLE_RAW_QUESTION)
                    else:
                        self.assertTrue(str(args.output_dir).endswith("vqa_rad/outputs/vqa_rad_query_v1"))
                        self.assertEqual(args.prompt_style, common.PROMPT_STYLE_LEGACY_JSON)
                elif name.startswith("benchmark_"):
                    args = bench_mod.parse_args(["--config", str(config_path)])
                    if "_v2" in name:
                        self.assertTrue(str(args.dataset_dir).endswith("vqa_rad/outputs/vqa_rad_query_v2"))
                    else:
                        self.assertTrue(str(args.dataset_dir).endswith("vqa_rad/outputs/vqa_rad_query_v1"))
                    self.assertEqual(args.max_tokens, 64)
                else:
                    args = train_mod.parse_args(["--config", str(config_path)])
                    self.assertIn(args.rank, {16, 32})
                    self.assertAlmostEqual(args.answer_type_sampling_weights["close_ended"], 1.0)
                    self.assertGreaterEqual(args.answer_type_sampling_weights["open_ended"], 1.0)
                    if "_v2_" in name or "_v2." in name:
                        self.assertTrue(str(args.dataset_dir).endswith("vqa_rad/outputs/vqa_rad_query_v2"))


class VQARadMetricsTests(unittest.TestCase):
    def test_finalize_metrics_reports_balanced_accuracy_and_family_breakdown(self) -> None:
        close_example = train_mod.VQARadExample(
            row_id="close",
            split="test",
            source_split="test",
            task_type=common.QUERY_TASK_TYPE,
            question="q1",
            image_path=Path("/tmp/a.png"),
            question_family="yes_no",
            answer_type=common.ANSWER_TYPE_CLOSE,
            answer_text="yes",
            normalized_answer="yes",
            image_group_id="img-close",
        )
        open_example = train_mod.VQARadExample(
            row_id="open",
            split="test",
            source_split="test",
            task_type=common.QUERY_TASK_TYPE,
            question="q2",
            image_path=Path("/tmp/b.png"),
            question_family="modality",
            answer_type=common.ANSWER_TYPE_OPEN,
            answer_text="mri",
            normalized_answer="mri",
            image_group_id="img-open",
        )
        state = train_mod._new_metric_state()
        for example, answer in ((close_example, "yes"), (open_example, '{"answer":"mri"}')):
            outcome = train_mod._score_answer_text(
                example,
                answer,
                open_exact_weight=0.55,
                open_token_f1_weight=0.25,
                open_numeric_weight=0.10,
                open_brevity_weight=0.05,
            )
            train_mod._record_outcome(state, example, outcome)

        metrics = train_mod._finalize_metrics(state, prefix="eval_", include_family_breakdown=True)
        self.assertEqual(metrics["eval_close_accuracy"], 1.0)
        self.assertEqual(metrics["eval_open_accuracy"], 1.0)
        self.assertEqual(metrics["eval_overall_accuracy"], 1.0)
        self.assertEqual(metrics["eval_balanced_accuracy"], 1.0)
        self.assertEqual(metrics["eval_plain_text_rate"], 0.5)
        self.assertIn("yes_no", metrics["question_family_breakdown"])
        self.assertIn("modality", metrics["question_family_breakdown"])


class VQARadModelResolutionTests(unittest.TestCase):
    class _FakeFinetune:
        def __init__(self, checkpoints: list[int]) -> None:
            self._checkpoints = list(checkpoints)

        def list_checkpoints(self, *, limit: int = 50, cursor: str | None = None):
            del limit, cursor
            return SimpleNamespace(
                checkpoints=[SimpleNamespace(step=step) for step in self._checkpoints],
                next_cursor=None,
                has_more=False,
            )

    class _FakeClient:
        def __init__(self, checkpoints: list[int]) -> None:
            self._finetune = VQARadModelResolutionTests._FakeFinetune(checkpoints)

        def get_finetune(self, finetune_id: str):
            del finetune_id
            return self._finetune

        def close(self) -> None:
            return

    def test_resolve_query_inference_model_uses_nearest_saved_checkpoint(self) -> None:
        with mock.patch.object(
            bench_mod.query_common,
            "TunaClient",
            return_value=self._FakeClient([10, 100, 180, 200]),
        ):
            resolved = bench_mod.query_common.resolve_query_inference_model(
                api_base="https://api-staging.moondream.ai/v1",
                api_key="test-key",
                model="",
                finetune_id="01KMPQW01YHZRFDJPGVGMKZ3PA",
                checkpoint_step=189,
                timeout=30.0,
            )
        self.assertEqual(resolved.model, "moondream3-preview/01KMPQW01YHZRFDJPGVGMKZ3PA@180")
        self.assertEqual(resolved.requested_checkpoint_step, 189)
        self.assertEqual(resolved.resolved_checkpoint_step, 180)
        self.assertTrue(resolved.used_checkpoint_fallback)

    def test_resolve_query_inference_model_requires_checkpoint_step_for_finetune(self) -> None:
        with self.assertRaisesRegex(ValueError, "--finetune-id requires --checkpoint-step"):
            bench_mod.query_common.resolve_query_inference_model(
                api_base="https://api-staging.moondream.ai/v1",
                api_key="test-key",
                model="",
                finetune_id="01KMPQW01YHZRFDJPGVGMKZ3PA",
                checkpoint_step=None,
                timeout=30.0,
            )


class VQARadBenchmarkTests(unittest.TestCase):
    def test_benchmark_main_writes_metrics_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            rows = [
                _dataset_row(
                    dataset_dir=dataset_dir,
                    split_name="test",
                    image_name="close.png",
                    color="white",
                    question="are the lungs normal appearing?",
                    answer_text="no",
                    question_family="yes_no",
                    prompt_style=common.PROMPT_STYLE_RAW_QUESTION,
                ),
                _dataset_row(
                    dataset_dir=dataset_dir,
                    split_name="test",
                    image_name="open.png",
                    color="black",
                    question="how was this image taken?",
                    answer_text="mri",
                    question_family="modality",
                    prompt_style=common.PROMPT_STYLE_RAW_QUESTION,
                ),
            ]
            _write_jsonl_split(dataset_dir, "test", rows)
            output_json = Path(tmpdir) / "benchmark.json"
            predictions_jsonl = Path(tmpdir) / "benchmark_predictions.jsonl"

            def fake_call_query_api(**kwargs):
                question = str(kwargs["question"]).lower()
                if "how was this image taken" in question:
                    return '{"answer":"mri"}', {"answer": '{"answer":"mri"}'}, 25.0
                return "no, there are findings", {"answer": "no, there are findings"}, 15.0

            with mock.patch.object(bench_mod.query_common, "call_query_api", side_effect=fake_call_query_api):
                bench_mod.main(
                    [
                        "--dataset-dir",
                        str(dataset_dir),
                        "--api-key",
                        "test-key",
                        "--output-json",
                        str(output_json),
                        "--predictions-jsonl",
                        str(predictions_jsonl),
                        "--max-samples",
                        "2",
                        "--no-progress",
                    ]
                )

            metrics = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(metrics["eval_balanced_accuracy"], 1.0)
            self.assertEqual(metrics["eval_overall_accuracy"], 1.0)
            self.assertEqual(metrics["eval_plain_text_rate"], 0.5)
            self.assertEqual(metrics["requested_rows"], 2)
            self.assertEqual(metrics["requested_checkpoint_step"], -1)
            self.assertEqual(metrics["resolved_checkpoint_step"], -1)
            prediction_lines = predictions_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(prediction_lines), 2)
            records = [json.loads(line) for line in prediction_lines]
            self.assertEqual({record["prediction_format"] for record in records}, {"plain_text", "strict_json"})


class VQARadTrainSmokeTests(unittest.TestCase):
    class _FakeRolloutResult:
        def __init__(self, answer: str, num_rollouts: int) -> None:
            self.rollouts = [
                SimpleNamespace(output=SimpleNamespace(answer=answer))
                for _ in range(num_rollouts)
            ]

        def to_group(self, *, rewards: list[float]) -> dict[str, object]:
            return {"rewards": list(rewards)}

    class _FakeFinetune:
        def __init__(self) -> None:
            self.finetune_id = "ft_vqa_rad_test"
            self.name = "ft_vqa_rad_test"
            self.saved = 0

        def rollouts_batch(self, *, requests, num_rollouts: int, max_workers: int):
            del max_workers
            results = []
            for request in requests:
                question = str(request.question).lower()
                if "how was this image taken" in question:
                    answer = '{"answer":"mri"}'
                elif "location of the mass" in question:
                    answer = '{"answer":"pineal region"}'
                else:
                    answer = '{"answer":"no"}'
                results.append(VQARadTrainSmokeTests._FakeRolloutResult(answer, num_rollouts))
            return results

        def train_step(self, *, groups, lr: float):
            del groups, lr
            return SimpleNamespace(kl=0.1, router_kl=0.0, grad_norm=0.2)

        def save_checkpoint(self):
            self.saved += 1
            return SimpleNamespace(checkpoint=SimpleNamespace(step=self.saved * 10))

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.finetune = VQARadTrainSmokeTests._FakeFinetune()

        def create_finetune(self, *, name: str, rank: int):
            del name, rank
            return self.finetune

        def get_finetune(self, finetune_id: str):
            del finetune_id
            return self.finetune

        def close(self) -> None:
            return

    class _FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}

        def finish(self) -> None:
            return

    def test_train_main_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            train_rows = [
                _dataset_row(
                    dataset_dir=dataset_dir,
                    split_name="train",
                    image_name="a.png",
                    color="white",
                    question="are the lungs normal appearing?",
                    answer_text="no",
                    question_family="yes_no",
                ),
                _dataset_row(
                    dataset_dir=dataset_dir,
                    split_name="train",
                    image_name="b.png",
                    color="black",
                    question="how was this image taken?",
                    answer_text="mri",
                    question_family="modality",
                ),
            ]
            val_rows = [
                _dataset_row(
                    dataset_dir=dataset_dir,
                    split_name="validation",
                    image_name="c.png",
                    color="green",
                    question="what is the location of the mass?",
                    answer_text="pineal region",
                    question_family="location",
                ),
            ]
            test_rows = [
                _dataset_row(
                    dataset_dir=dataset_dir,
                    split_name="test",
                    image_name="d.png",
                    color="blue",
                    question="are there pulmonary findings?",
                    answer_text="no",
                    question_family="yes_no",
                ),
            ]
            _write_jsonl_split(dataset_dir, "train", train_rows)
            _write_jsonl_split(dataset_dir, "validation", val_rows)
            _write_jsonl_split(dataset_dir, "test", test_rows)

            fake_run = self._FakeRun()
            fake_wandb = SimpleNamespace(init=lambda *args, **kwargs: fake_run, log=lambda *args, **kwargs: None)
            fake_client = self._FakeClient()

            with (
                mock.patch.object(train_mod, "TunaClient", return_value=fake_client),
                mock.patch.object(train_mod.query_common, "wandb", fake_wandb),
            ):
                train_mod.main(
                    [
                        "--dataset-dir",
                        str(dataset_dir),
                        "--api-key",
                        "test-key",
                        "--num-steps",
                        "1",
                        "--eval-every",
                        "1",
                        "--save-every",
                        "0",
                        "--eval-max-samples",
                        "1",
                        "--eval-predictions-output-dir",
                        str(Path(tmpdir) / "eval_preds"),
                        "--no-progress",
                    ]
                )

            self.assertGreaterEqual(fake_client.finetune.saved, 1)
            self.assertEqual(fake_run.summary["finetune_id"], "ft_vqa_rad_test")
            self.assertEqual(fake_run.summary["best_checkpoint_step"], 10)
            self.assertEqual(fake_run.summary["latest_checkpoint_step"], 20)


if __name__ == "__main__":
    unittest.main()
