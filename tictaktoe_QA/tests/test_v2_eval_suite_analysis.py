from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tictaktoe_QA.v2_eval_suite import common


def _move_to_row_col(move: int) -> tuple[int, int]:
    row = ((move - 1) // 3) + 1
    col = ((move - 1) % 3) + 1
    return row, col


class BestMoveBucketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gt_row = {
            "scores_by_move_json": json.dumps(
                {
                    "1": {"value": 1, "depth": 2},
                    "2": {"value": 1, "depth": 5},
                    "3": {"value": 0, "depth": 1},
                    "4": {"value": -1, "depth": 3},
                }
            )
        }

    def _record_for_move(self, move: int) -> dict[str, object]:
        row, col = _move_to_row_col(move)
        return {
            "status": "",
            "json_object_parsed": True,
            "parse_success": True,
            "prediction_json": {"row": row, "col": col},
        }

    def test_dense_rank_buckets(self) -> None:
        self.assertEqual(
            common.classify_best_move_prediction(self._record_for_move(1), ground_truth_row=self.gt_row),
            "best_move",
        )
        self.assertEqual(
            common.classify_best_move_prediction(self._record_for_move(2), ground_truth_row=self.gt_row),
            "second_best",
        )
        self.assertEqual(
            common.classify_best_move_prediction(self._record_for_move(3), ground_truth_row=self.gt_row),
            "third_best",
        )
        self.assertEqual(
            common.classify_best_move_prediction(self._record_for_move(4), ground_truth_row=self.gt_row),
            "fourth_plus",
        )

    def test_invalid_move_bucket(self) -> None:
        record = {
            "status": "",
            "json_object_parsed": True,
            "parse_success": True,
            "prediction_json": {"row": 3, "col": 3},
        }
        self.assertEqual(
            common.classify_best_move_prediction(record, ground_truth_row={"scores_by_move_json": "{}"}),
            "invalid_move",
        )

    def test_improper_format_bucket(self) -> None:
        record_non_json = {
            "status": "",
            "json_object_parsed": False,
            "parse_success": False,
            "prediction_json": None,
        }
        self.assertEqual(
            common.classify_best_move_prediction(record_non_json, ground_truth_row=self.gt_row),
            "improper_response_format",
        )

        record_bad_schema = {
            "status": "",
            "json_object_parsed": True,
            "parse_success": True,
            "prediction_json": {"winner": "X"},
        }
        self.assertEqual(
            common.classify_best_move_prediction(record_bad_schema, ground_truth_row=self.gt_row),
            "improper_response_format",
        )

    def test_request_error_bucket(self) -> None:
        record = {
            "status": "request_error",
            "json_object_parsed": False,
            "parse_success": False,
            "prediction_json": None,
        }
        self.assertEqual(
            common.classify_best_move_prediction(record, ground_truth_row=self.gt_row),
            "request_error",
        )


class V2FilterTests(unittest.TestCase):
    def test_is_v2_metrics_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp) / "synth_dataset" / "outputs" / "v2"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            self.assertTrue(
                common.is_v2_metrics_payload(
                    {"hf_dataset_repo_id": "maxs-m87/tictactoe-qa-v2"},
                    dataset_dir=dataset_dir,
                )
            )
            self.assertTrue(
                common.is_v2_metrics_payload(
                    {"dataset_dir": str(dataset_dir)},
                    dataset_dir=dataset_dir,
                )
            )
            self.assertFalse(
                common.is_v2_metrics_payload(
                    {"dataset_dir": str(dataset_dir.parent / "v1")},
                    dataset_dir=dataset_dir,
                )
            )
            self.assertFalse(common.is_v2_metrics_payload({}, dataset_dir=dataset_dir))


if __name__ == "__main__":
    unittest.main()
