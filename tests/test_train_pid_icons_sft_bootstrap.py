from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _DEPICATED_MDpi_and_d as pid_pkg

sys.modules.setdefault("MDpi_and_d", pid_pkg)

from _DEPICATED_MDpi_and_d import train_pid_icons as mod
from tuna_sdk import DetectAnnotation, DetectRequest, DetectSFTTarget, PointRequest, PointSFTTarget
from tuna_sdk.errors import TunaAPIError


class PIDIconsSFTBootstrapTests(unittest.TestCase):
    def _task(self, *, is_positive: bool) -> mod.TaskSample:
        gt_boxes = [DetectAnnotation(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4)] if is_positive else []
        return mod.TaskSample(
            image=Image.new("RGB", (8, 8), color=(255, 255, 255)),
            prompt="gate valve",
            gt_boxes=gt_boxes,
            class_name="gate valve",
            is_positive=is_positive,
            source="unit",
        )

    def test_detect_bootstrap_uses_gt_boxes(self) -> None:
        task = self._task(is_positive=True)
        request = DetectRequest(object_name=task.prompt, image_url="data:image/jpeg;base64,abc")

        group = mod._build_sft_group_for_task(task, request=request, skill="detect")

        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group.mode, "sft")
        self.assertIsInstance(group.targets[0], DetectSFTTarget)
        self.assertEqual(group.targets[0].boxes, task.gt_boxes)

    def test_point_bootstrap_uses_boxes_not_center_points(self) -> None:
        task = self._task(is_positive=True)
        request = PointRequest(object_name=task.prompt, image_url="data:image/jpeg;base64,abc")

        group = mod._build_sft_group_for_task(task, request=request, skill="point")

        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group.mode, "sft")
        self.assertIsInstance(group.targets[0], PointSFTTarget)
        self.assertEqual(group.targets[0].boxes, task.gt_boxes)
        self.assertIsNone(group.targets[0].points)

    def test_negative_task_is_excluded_from_bootstrap(self) -> None:
        task = self._task(is_positive=False)
        request = DetectRequest(object_name=task.prompt, image_url="data:image/jpeg;base64,abc")

        group = mod._build_sft_group_for_task(task, request=request, skill="detect")

        self.assertIsNone(group)

    def test_detects_backend_rl_only_schema_rejection_for_sft(self) -> None:
        exc = TunaAPIError(
            "Request failed",
            status_code=422,
            response_body={
                "detail": [
                    {"msg": "Input should be 'rl'", "loc": ["body", "groups", 0, "mode"]},
                    {"msg": "Field required", "loc": ["body", "groups", 0, "request", "finetune_id"]},
                ]
            },
            request_id="req_test",
        )
        self.assertTrue(mod._is_sft_bootstrap_unsupported_error(exc))


if __name__ == "__main__":
    unittest.main()
