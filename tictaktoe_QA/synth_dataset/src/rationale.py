"""Rationale and answer text generation."""

from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .label_engine import BoardRecord, move_to_row_col
from tictaktoe_QA.task_schema import normalize_task_type


@dataclass(frozen=True)
class AnswerPayload:
    final_answer_json: str
    answer_text: str
    rationale_source: str


class OpenAIParaphraser:
    """Small wrapper around Chat Completions for rationale paraphrasing."""

    def __init__(self, *, model: str = "gpt-4o-mini", timeout_sec: float = 30.0) -> None:
        self.model = model
        self.timeout_sec = timeout_sec
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def paraphrase(self, text: str) -> str:
        if not self.enabled:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        payload = {
            "model": self.model,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "system",
                    "content": "Paraphrase the sentence into one concise sentence. Keep meaning identical.",
                },
                {"role": "user", "content": text},
            ],
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty paraphrase")
            return " ".join(content.split())
        except (KeyError, ValueError, OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise RuntimeError(f"paraphrase failed: {exc}") from exc


def _json_str(obj: dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


def _state_counts(record: BoardRecord) -> tuple[int, int]:
    x_count = sum(1 for cell in record.state if cell == "X")
    o_count = sum(1 for cell in record.state if cell == "O")
    return x_count, o_count


def _base_rationale(task_type: str, final_obj: dict[str, Any], record: BoardRecord) -> str:
    task_type = normalize_task_type(task_type, allow_unknown=True)
    x_count, o_count = _state_counts(record)

    if task_type == "best_move":
        move = row_col = (None, None)
        if record.best_move_canonical is not None:
            row_col = move_to_row_col(record.best_move_canonical)
            move = record.best_move_canonical
        return (
            "The selected move maximizes minimax value, applies depth tie-breaking, "
            f"and resolves remaining ties lexicographically; canonical move is {move} at row {row_col[0]}, col {row_col[1]}."
        )

    if task_type == "has_winning_move":
        has_win = bool(final_obj.get("has_winning_move", False))
        if has_win:
            return f"At least one available move completes three-in-a-row immediately for {record.player_to_move}."
        return f"No available move produces an immediate three-in-a-row for {record.player_to_move}."

    if task_type == "turn_player":
        return f"X count is {x_count} and O count is {o_count}, so the next player is {record.player_to_move}."

    if task_type == "winner":
        winner = str(final_obj.get("winner", "in_progress"))
        if winner == "in_progress":
            return "No winning line exists and open cells remain, so the game is still in progress."
        if winner == "draw":
            return "No winning line exists and no available moves remain, so the result is a draw."
        return f"A complete three-in-a-row exists for {winner}."

    if task_type == "is_game_over":
        return "The game is over when a win is present or no available moves remain."

    if task_type == "available_moves_count":
        return f"Available moves correspond to empty cells; there are {len(record.legal_moves)} available positions."

    if task_type == "available_moves_list":
        return "Available moves are the empty cells listed in top-left to bottom-right order."

    raise ValueError(f"unknown task_type: {task_type}")


def build_final_answer(task_type: str, record: BoardRecord) -> dict[str, Any]:
    task_type = normalize_task_type(task_type, allow_unknown=True)

    if task_type == "best_move":
        if record.best_move_canonical is None:
            raise ValueError("best_move requested for terminal board")
        row, col = move_to_row_col(record.best_move_canonical)
        return {"row": row, "col": col}

    if task_type == "has_winning_move":
        return {"has_winning_move": bool(record.immediate_winning_moves)}

    if task_type == "turn_player":
        return {"player": record.player_to_move}

    if task_type == "winner":
        return {"winner": record.winner_label}

    if task_type == "is_game_over":
        return {"is_game_over": bool(record.is_terminal)}

    if task_type == "available_moves_count":
        return {"available_move_count": len(record.legal_moves)}

    if task_type == "available_moves_list":
        moves = [{"row": move_to_row_col(m)[0], "col": move_to_row_col(m)[1]} for m in record.legal_moves]
        return {"available_moves": moves}

    raise ValueError(f"unknown task_type: {task_type}")


def build_answer_payload(
    *,
    task_type: str,
    record: BoardRecord,
    rng: random.Random,
    split: str,
    llm_ratio: float,
    paraphraser: OpenAIParaphraser | None,
    llm_enabled: bool,
) -> AnswerPayload:
    final_obj = build_final_answer(task_type, record)
    final_json = _json_str(final_obj)

    reason = _base_rationale(task_type, final_obj, record)
    source = "template"

    use_llm = (
        llm_enabled
        and split == "train"
        and paraphraser is not None
        and paraphraser.enabled
        and rng.random() < llm_ratio
    )
    if use_llm:
        try:
            reason = paraphraser.paraphrase(reason)
            source = "llm"
        except RuntimeError:
            source = "template_fallback"

    answer_text = f"Reason: {reason}\nFinal: {final_json}"
    return AnswerPayload(final_answer_json=final_json, answer_text=answer_text, rationale_source=source)
