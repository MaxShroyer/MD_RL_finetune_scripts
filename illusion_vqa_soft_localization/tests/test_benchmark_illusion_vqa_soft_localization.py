from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from illusion_vqa_soft_localization import benchmark_illusion_vqa_soft_localization as mod


OPTIONS = [
    "The left object",
    "The right object",
    "Both objects",
    "Neither object",
]


def test_construct_mcq_marks_correct_letter() -> None:
    mcq, gold_letter = mod.construct_mcq(OPTIONS, "Both objects")
    assert "a. The left object" in mcq
    assert "d. Neither object" in mcq
    assert gold_letter == "c"


def test_construct_mcq_accepts_letter_answer() -> None:
    _, gold_letter = mod.construct_mcq(OPTIONS, "c")
    assert gold_letter == "c"


def test_normalize_prediction_letter_accepts_letter_variants() -> None:
    assert mod.normalize_prediction_letter("b", OPTIONS) == "b"
    assert mod.normalize_prediction_letter("Answer: c", OPTIONS) == "c"
    assert mod.normalize_prediction_letter("The answer is (d).", OPTIONS) == "d"
    assert mod.normalize_prediction_letter('{"answer":"a"}', OPTIONS) == "a"


def test_normalize_prediction_letter_accepts_option_text() -> None:
    assert mod.normalize_prediction_letter("The right object", OPTIONS) == "b"
    assert mod.normalize_prediction_letter("I choose both objects.", OPTIONS) == "c"


def test_normalize_prediction_letter_rejects_ambiguous_or_empty() -> None:
    assert mod.normalize_prediction_letter("", OPTIONS) is None
    assert mod.normalize_prediction_letter("both the left object and the right object", OPTIONS) is None


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
        self._finetune = _FakeFinetune(checkpoints)

    def get_finetune(self, finetune_id: str):
        del finetune_id
        return self._finetune

    def close(self) -> None:
        return


def test_shared_resolution_uses_nearest_saved_checkpoint() -> None:
    with mock.patch.object(
        mod.shared_query_common,
        "TunaClient",
        return_value=_FakeClient([50, 120, 180]),
    ):
        resolved = mod.shared_query_common.resolve_query_inference_model(
            api_base="https://api-staging.moondream.ai/v1",
            api_key="test-key",
            model="",
            finetune_id="ft-illusion",
            checkpoint_step=189,
            timeout=30.0,
        )
    assert resolved.model == "moondream3-preview/ft-illusion@180"
    assert resolved.resolved_checkpoint_step == 180


def test_unsuffixed_finetuned_model_string_is_rejected() -> None:
    with mock.patch.object(
        mod.shared_query_common,
        "TunaClient",
        return_value=_FakeClient([50, 120, 180]),
    ):
        try:
            mod.shared_query_common.resolve_query_inference_model(
                api_base="https://api-staging.moondream.ai/v1",
                api_key="test-key",
                model="moondream3-preview/ft-illusion",
                finetune_id="",
                checkpoint_step=None,
                timeout=30.0,
            )
        except ValueError as exc:
            assert "checkpoint step" in str(exc)
        else:
            raise AssertionError("expected ValueError")
