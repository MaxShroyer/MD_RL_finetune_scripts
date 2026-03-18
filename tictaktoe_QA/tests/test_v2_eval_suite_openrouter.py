from __future__ import annotations

import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import patch

from PIL import Image

from tictaktoe_QA import train_ttt_query_rl as train_utils
from tictaktoe_QA.v2_eval_suite import benchmark_openrouter_v2 as mod
from tictaktoe_QA.v2_eval_suite import common


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False


class ResponseExtractionTests(unittest.TestCase):
    def test_extract_openrouter_answer_string(self) -> None:
        payload = {"choices": [{"message": {"content": "{\"row\":1,\"col\":1}"}}]}
        self.assertEqual(common.extract_openrouter_answer_text(payload), '{"row":1,"col":1}')

    def test_extract_openrouter_answer_parts_array(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "line one"},
                            {"type": "text", "text": "line two"},
                        ]
                    }
                }
            ]
        }
        self.assertEqual(common.extract_openrouter_answer_text(payload), "line one\nline two")


class RetryBehaviorTests(unittest.TestCase):
    def test_retry_429_then_success(self) -> None:
        headers = Message()
        headers["Retry-After"] = "0"
        first_error = HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=headers,
            fp=io.BytesIO(b'{"error":"rate_limited"}'),
        )

        calls = {"count": 0}

        def _fake_urlopen(req, timeout):  # type: ignore[no-untyped-def]
            calls["count"] += 1
            if calls["count"] == 1:
                raise first_error
            return _FakeHTTPResponse({"choices": [{"message": {"content": "ok"}}]})

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen), patch("time.sleep") as sleep_mock:
            answer, payload, _latency = mod._call_openrouter_chat_api(
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                model_id="openai/gpt-5.1",
                question="q",
                image_url="data:image/png;base64,abc",
                temperature=0.0,
                top_p=1.0,
                max_tokens=32,
                timeout=30.0,
                retry_429_max_retries=1,
                retry_429_backoff_s=0.01,
                retry_429_max_backoff_s=0.05,
            )

        self.assertEqual(answer, "ok")
        self.assertIsInstance(payload, dict)
        self.assertEqual(calls["count"], 2)
        self.assertTrue(sleep_mock.called)


class AuthTests(unittest.TestCase):
    def test_resolve_openrouter_api_key_prefers_cli(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-or", "OPENAI_API_KEY": "env-openai"}, clear=True):
            self.assertEqual(mod._resolve_openrouter_api_key("cli-or"), "cli-or")

    def test_resolve_openrouter_api_key_uses_openrouter_env(self) -> None:
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-or"}, clear=True):
            self.assertEqual(mod._resolve_openrouter_api_key(""), "env-or")

    def test_resolve_openrouter_api_key_rejects_openai_only_env(self) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-openai"}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY is required"):
                mod._resolve_openrouter_api_key("")

    def test_preflight_openrouter_auth_success(self) -> None:
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse({"data": []})) as urlopen_mock:
            mod._preflight_openrouter_auth(
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                timeout=10.0,
            )
        self.assertTrue(urlopen_mock.called)

    def test_preflight_openrouter_auth_raises_query_api_error_on_401(self) -> None:
        err = HTTPError(
            url="https://openrouter.ai/api/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=Message(),
            fp=io.BytesIO(b'{"error":{"message":"User not found.","code":401}}'),
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(mod.QueryAPIError) as ctx:
                mod._preflight_openrouter_auth(
                    api_base="https://openrouter.ai/api/v1",
                    api_key="bad-key",
                    timeout=10.0,
                )
        self.assertEqual(ctx.exception.status_code, 401)

    def test_fetch_openrouter_model_ids(self) -> None:
        payload = {"data": [{"id": "anthropic/claude-opus-latest"}, {"id": "openai/gpt-5.1"}]}
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)):
            model_ids = mod._fetch_openrouter_model_ids(
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                timeout=10.0,
            )
        self.assertEqual(model_ids, {"anthropic/claude-opus-latest", "openai/gpt-5.1"})

    def test_fetch_openrouter_model_catalog(self) -> None:
        payload = {
            "data": [
                {"id": "anthropic/claude-3.7-sonnet", "architecture": {"modality": "text+image->text"}},
                {"id": "qwen/qwen-max", "architecture": {"modality": "text->text"}},
            ]
        }
        with patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(payload)):
            catalog = mod._fetch_openrouter_model_catalog(
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                timeout=10.0,
            )
        self.assertIn("anthropic/claude-3.7-sonnet", catalog)
        self.assertIn("qwen/qwen-max", catalog)

    def test_validate_requested_model_ids_returns_only_valid_models(self) -> None:
        models = [
            {"label": "claude", "model_id": "anthropic/claude-opus-latest"},
            {"label": "chatgpt", "model_id": "openai/gpt-latest"},
        ]
        available = {"anthropic/claude-opus-latest", "openai/gpt-5.1", "openai/gpt-5-mini"}
        with patch("builtins.print") as print_mock:
            valid = mod._validate_requested_model_ids(
                models=models,
                available_model_ids=available,
            )
        self.assertEqual(valid, [{"label": "claude", "model_id": "anthropic/claude-opus-latest"}])
        self.assertTrue(print_mock.called)

    def test_validate_requested_model_ids_raises_when_all_models_invalid(self) -> None:
        models = [
            {"label": "claude", "model_id": "anthropic/claude-opus-latest"},
            {"label": "chatgpt", "model_id": "openai/gpt-latest"},
        ]
        available = {"anthropic/claude-3.7-sonnet", "openai/gpt-5.1", "openai/gpt-5-mini"}
        with self.assertRaisesRegex(ValueError, "invalid OpenRouter model_id"):
            mod._validate_requested_model_ids(
                models=models,
                available_model_ids=available,
            )

    def test_filter_models_for_image_input_skips_text_only_models(self) -> None:
        models = [
            {"label": "claude", "model_id": "anthropic/claude-3.7-sonnet"},
            {"label": "qwen_text", "model_id": "qwen/qwen-max"},
        ]
        catalog = {
            "anthropic/claude-3.7-sonnet": {"id": "anthropic/claude-3.7-sonnet", "architecture": {"modality": "text+image->text"}},
            "qwen/qwen-max": {"id": "qwen/qwen-max", "architecture": {"modality": "text->text"}},
        }
        with patch("builtins.print") as print_mock:
            kept = mod._filter_models_for_image_input(
                models=models,
                model_catalog=catalog,
            )
        self.assertEqual(kept, [{"label": "claude", "model_id": "anthropic/claude-3.7-sonnet"}])
        self.assertTrue(print_mock.called)

    def test_build_chat_payload_applies_request_overrides(self) -> None:
        payload = mod._build_chat_payload(
            model_id="anthropic/claude-3.7-sonnet:thinking",
            question="q",
            image_url="data:image/png;base64,abc",
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
            request_overrides={"reasoning": {"effort": "high"}},
        )
        self.assertEqual(payload["reasoning"], {"effort": "high"})


class SamplingTests(unittest.TestCase):
    def _qa_example(
        self,
        *,
        row_id: str,
        task_type: str,
        expected_answer: dict[str, object],
    ) -> train_utils.QAExample:
        return train_utils.QAExample(
            row_id=row_id,
            split="test",
            task_type=task_type,
            question=f"q_{row_id}",
            image_path=Path("/tmp/unused.png"),
            expected_answer=expected_answer,
            best_move_canonical=1 if task_type == "best_move" else None,
            best_move_optimal_set=frozenset({1}) if task_type == "best_move" else frozenset(),
            best_move_scores=((1, 1, 2), (2, 0, 4)) if task_type == "best_move" else tuple(),
        )

    def test_sample_examples_by_task_quick_and_full(self) -> None:
        examples = []
        for idx in range(6):
            examples.append(SimpleNamespace(task_type="best_move", row_id=f"b{idx}"))
            examples.append(SimpleNamespace(task_type="available_moves_count", row_id=f"c{idx}"))
            examples.append(SimpleNamespace(task_type="available_moves_list", row_id=f"l{idx}"))
            examples.append(SimpleNamespace(task_type="winner", row_id=f"w{idx}"))

        quick = mod._sample_examples_by_task(
            examples,  # type: ignore[arg-type]
            task_types=["best_move", "available_moves_count", "available_moves_list"],
            max_samples_per_task=2,
            seed=7,
        )
        quick_counts = mod._count_examples_by_task(quick)  # type: ignore[arg-type]
        self.assertEqual(quick_counts.get("best_move", 0), 2)
        self.assertEqual(quick_counts.get("available_moves_count", 0), 2)
        self.assertEqual(quick_counts.get("available_moves_list", 0), 2)
        self.assertNotIn("winner", quick_counts)

        full = mod._sample_examples_by_task(
            examples,  # type: ignore[arg-type]
            task_types=["best_move", "available_moves_count", "available_moves_list"],
            max_samples_per_task=0,
            seed=7,
        )
        full_counts = mod._count_examples_by_task(full)  # type: ignore[arg-type]
        self.assertEqual(full_counts.get("best_move", 0), 6)
        self.assertEqual(full_counts.get("available_moves_count", 0), 6)
        self.assertEqual(full_counts.get("available_moves_list", 0), 6)
        self.assertNotIn("winner", full_counts)

    def test_sample_examples_by_task_stratified_turn_player_balances_classes(self) -> None:
        examples: list[train_utils.QAExample] = []
        for idx in range(8):
            examples.append(
                self._qa_example(
                    row_id=f"tp_x_{idx}",
                    task_type="turn_player",
                    expected_answer={"player": "X"},
                )
            )
        for idx in range(2):
            examples.append(
                self._qa_example(
                    row_id=f"tp_o_{idx}",
                    task_type="turn_player",
                    expected_answer={"player": "O"},
                )
            )
        selected = mod._sample_examples_by_task(
            examples,
            task_types=["turn_player"],
            max_samples_per_task=4,
            seed=3,
            sample_strategy="stratified",
            stratify_tasks=["turn_player"],
        )
        self.assertEqual(len(selected), 4)
        selected_players = {
            ex.expected_answer.get("player")  # type: ignore[union-attr]
            for ex in selected
        }
        self.assertIn("X", selected_players)
        self.assertIn("O", selected_players)

    def test_sample_examples_by_task_stratified_count_covers_buckets_and_is_deterministic(self) -> None:
        examples: list[train_utils.QAExample] = []
        for count in (0, 1, 2):
            for idx in range(3):
                examples.append(
                    self._qa_example(
                        row_id=f"cnt_{count}_{idx}",
                        task_type="available_moves_count",
                        expected_answer={"available_move_count": count},
                    )
                )
        selected_a = mod._sample_examples_by_task(
            examples,
            task_types=["available_moves_count"],
            max_samples_per_task=6,
            seed=11,
            sample_strategy="stratified",
            stratify_tasks=["available_moves_count"],
        )
        selected_b = mod._sample_examples_by_task(
            examples,
            task_types=["available_moves_count"],
            max_samples_per_task=6,
            seed=11,
            sample_strategy="stratified",
            stratify_tasks=["available_moves_count"],
        )
        self.assertEqual([ex.row_id for ex in selected_a], [ex.row_id for ex in selected_b])
        selected_counts = {
            ex.expected_answer.get("available_move_count")  # type: ignore[union-attr]
            for ex in selected_a
        }
        self.assertEqual(selected_counts, {0, 1, 2})


class DryRunBenchmarkTests(unittest.TestCase):
    def _build_image(self, path: Path) -> None:
        Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)

    def test_benchmark_model_with_mock_api_has_stable_metrics_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            img_a = tmp_path / "a.png"
            img_b = tmp_path / "b.png"
            img_c = tmp_path / "c.png"
            self._build_image(img_a)
            self._build_image(img_b)
            self._build_image(img_c)

            examples = [
                train_utils.QAExample(
                    row_id="r_best",
                    split="test",
                    task_type="best_move",
                    question="q_best",
                    image_path=img_a,
                    expected_answer={"row": 1, "col": 1},
                    best_move_canonical=1,
                    best_move_optimal_set=frozenset({1}),
                    best_move_scores=((1, 1, 2), (2, 0, 4), (3, -1, 5)),
                ),
                train_utils.QAExample(
                    row_id="r_count",
                    split="test",
                    task_type="available_moves_count",
                    question="q_count",
                    image_path=img_b,
                    expected_answer={"available_move_count": 2},
                    best_move_canonical=None,
                    best_move_optimal_set=frozenset(),
                ),
                train_utils.QAExample(
                    row_id="r_list",
                    split="test",
                    task_type="available_moves_list",
                    question="q_list",
                    image_path=img_c,
                    expected_answer={"available_moves": [{"row": 1, "col": 2}]},
                    best_move_canonical=None,
                    best_move_optimal_set=frozenset(),
                ),
            ]

            def _fake_api(**kwargs):  # type: ignore[no-untyped-def]
                question = kwargs["question"]
                if question == "q_best":
                    return '{"row":1,"col":1}', {"answer": "ok"}, 5.0
                if question == "q_count":
                    return '{"available_move_count":2}', {"answer": "ok"}, 6.0
                return '{"available_moves":[{"row":1,"col":2}]}', {"answer": "ok"}, 7.0

            predictions_path = tmp_path / "predictions.jsonl"
            metrics = mod._benchmark_model(
                model_label="unit",
                model_id="unit/model",
                examples=examples,
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                temperature=0.0,
                top_p=1.0,
                max_tokens=64,
                timeout=10.0,
                retry_429_max_retries=0,
                retry_429_backoff_s=0.0,
                retry_429_max_backoff_s=0.0,
                request_overrides=None,
                best_move_optimal_reward=0.7,
                predictions_path=predictions_path,
                show_progress=False,
                call_api_fn=_fake_api,
            )

            self.assertEqual(metrics["evaluated_rows"], 3)
            self.assertIn("eval_reward_mean", metrics)
            self.assertIn("eval_json_parse_rate", metrics)
            self.assertIn("by_task", metrics)
            self.assertIn("best_move", metrics["by_task"])
            self.assertTrue(predictions_path.exists())
            lines = predictions_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)

    def test_benchmark_model_emits_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_a = tmp_path / "a.png"
            img_b = tmp_path / "b.png"
            img_c = tmp_path / "c.png"
            self._build_image(img_a)
            self._build_image(img_b)
            self._build_image(img_c)

            examples = [
                train_utils.QAExample(
                    row_id="r_best_wrong",
                    split="test",
                    task_type="best_move",
                    question="q_best_wrong",
                    image_path=img_a,
                    expected_answer={"row": 1, "col": 1},
                    best_move_canonical=1,
                    best_move_optimal_set=frozenset({1}),
                    best_move_scores=((1, 1, 2), (5, 1, 4), (9, 0, 6)),
                ),
                train_utils.QAExample(
                    row_id="r_turn",
                    split="test",
                    task_type="turn_player",
                    question="q_turn",
                    image_path=img_b,
                    expected_answer={"player": "X"},
                    best_move_canonical=None,
                    best_move_optimal_set=frozenset(),
                ),
                train_utils.QAExample(
                    row_id="r_count",
                    split="test",
                    task_type="available_moves_count",
                    question="q_count",
                    image_path=img_c,
                    expected_answer={"available_move_count": 2},
                    best_move_canonical=None,
                    best_move_optimal_set=frozenset(),
                ),
            ]

            def _fake_api(**kwargs):  # type: ignore[no-untyped-def]
                question = kwargs["question"]
                if question == "q_best_wrong":
                    return '{"row":2,"col":2}', {"answer": "ok"}, 5.0
                if question == "q_turn":
                    return '{"player":"O"}', {"answer": "ok"}, 6.0
                return '{"available_move_count":1}', {"answer": "ok"}, 7.0

            predictions_path = tmp_path / "predictions_diag.jsonl"
            metrics = mod._benchmark_model(
                model_label="unit",
                model_id="unit/model",
                examples=examples,
                api_base="https://openrouter.ai/api/v1",
                api_key="test-key",
                temperature=0.0,
                top_p=1.0,
                max_tokens=64,
                timeout=10.0,
                retry_429_max_retries=0,
                retry_429_backoff_s=0.0,
                retry_429_max_backoff_s=0.0,
                request_overrides=None,
                best_move_optimal_reward=0.7,
                predictions_path=predictions_path,
                show_progress=False,
                call_api_fn=_fake_api,
            )
            self.assertIn("diagnostics", metrics)
            self.assertIn("best_move_wrong_reward_mean", metrics["diagnostics"])
            self.assertIn("turn_player_confusion", metrics["diagnostics"])
            self.assertEqual(metrics["diagnostics"]["turn_player_confusion"]["X->O"], 1)
            self.assertIn("available_moves_count_delta_hist", metrics["diagnostics"])
            self.assertEqual(metrics["diagnostics"]["available_moves_count_delta_hist"]["-1"], 1)


class CompletionStatusTests(unittest.TestCase):
    def test_model_status_and_line_count_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
            self.assertEqual(mod._count_jsonl_lines(path), 2)
            self.assertEqual(
                mod._model_status_from_counts(requested_rows=2, prediction_lines=2),
                "completed",
            )
            self.assertEqual(
                mod._model_status_from_counts(requested_rows=3, prediction_lines=2),
                "partial",
            )


if __name__ == "__main__":
    unittest.main()
