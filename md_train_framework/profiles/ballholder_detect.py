from __future__ import annotations

from md_train_framework.runtime import DetectTrainer

from .detect_common import build_constant_object_detect_samples, profile_object_name


class BallHolderDetectTrainer(DetectTrainer):
    DEFAULT_OBJECT_NAME = "ball holder"
    DEFAULT_PROMPT_VARIANTS = [
        "Player with ball in hand",
        "ball holder",
        "player with the ball",
        "person with the ball",
        "ballhandler",
        "ball handler",
        "player in possession",
        "player holding the basketball",
        "offensive player with the ball",
    ]

    def _load_samples(self):
        object_name = profile_object_name(
            default_name=self.DEFAULT_OBJECT_NAME,
            override_value=self._override("object_name", self.config.task.extra.get("object_name", "")),
        )
        max_boxes = self._override("max_boxes", 1)
        return (
            build_constant_object_detect_samples(
                self.config,
                split_name=self.config.dataset.train_split,
                object_name=object_name,
                max_boxes=max_boxes,
            ),
            build_constant_object_detect_samples(
                self.config,
                split_name=self.config.dataset.val_split,
                object_name=object_name,
                max_boxes=max_boxes,
            ),
            build_constant_object_detect_samples(
                self.config,
                split_name=self.config.dataset.test_split,
                object_name=object_name,
                max_boxes=max_boxes,
            ),
        )

    def _training_prompt_variants(self, sample):
        del sample
        raw = self._override("prompt_variants", self.DEFAULT_PROMPT_VARIANTS)
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        return list(self.DEFAULT_PROMPT_VARIANTS)
