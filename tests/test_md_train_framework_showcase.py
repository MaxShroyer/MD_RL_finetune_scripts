from __future__ import annotations

import importlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from md_train_framework import cli
from md_train_framework.artifacts import create_run_paths
from md_train_framework.config import load_framework_config, save_framework_config
from md_train_framework.metrics import build_metric_policy
from md_train_framework.profiles.ballholder_detect import BallHolderDetectTrainer
from md_train_framework.profiles.pandid_point import PandidPointTrainer
from md_train_framework.profiles.statefarm_detect import StatefarmDetectTrainer
from md_train_framework.profiles.ttt_query import CANONICAL_TASK_TYPES, TTTQueryTrainer
from md_train_framework.registry import RunRegistry
from md_train_framework.rewards import get_reward_preset
from md_train_framework.runtime import make_trainer
from md_train_framework.samples import DetectSample, PointSample, normalize_row
from md_train_framework.scoring import score_detect, score_point
from md_train_framework.wandb_logger import WandbLogger
from tuna_sdk import DetectAnnotation, DetectOutput, PointAnnotation, PointOutput, QueryOutput, Rollout, RolloutsRequest, TrainStepGroup


EXAMPLE_MODULES = {
    "ballholder": {
        "compare": "md_train_framework.examples.ballholder.compare",
        "dataset_loader": "md_train_framework.examples.ballholder.dataset_loader",
        "eval": "md_train_framework.examples.ballholder.eval",
        "leaderboard": "md_train_framework.examples.ballholder.leaderboard",
        "resume": "md_train_framework.examples.ballholder.resume",
        "sweep": "md_train_framework.examples.ballholder.sweep",
        "train": "md_train_framework.examples.ballholder.train",
    },
    "statefarm": {
        "compare": "md_train_framework.examples.statefarm.compare",
        "dataset_loader": "md_train_framework.examples.statefarm.dataset_loader",
        "eval": "md_train_framework.examples.statefarm.eval",
        "leaderboard": "md_train_framework.examples.statefarm.leaderboard",
        "resume": "md_train_framework.examples.statefarm.resume",
        "sweep": "md_train_framework.examples.statefarm.sweep",
        "train": "md_train_framework.examples.statefarm.train",
    },
    "pandid": {
        "compare": "md_train_framework.examples.pandid.compare",
        "dataset_loader": "md_train_framework.examples.pandid.dataset_loader",
        "eval": "md_train_framework.examples.pandid.eval",
        "leaderboard": "md_train_framework.examples.pandid.leaderboard",
        "resume": "md_train_framework.examples.pandid.resume",
        "sweep": "md_train_framework.examples.pandid.sweep",
        "train": "md_train_framework.examples.pandid.train",
    },
    "ttt_qa": {
        "compare": "md_train_framework.examples.ttt_qa.compare",
        "dataset_loader": "md_train_framework.examples.ttt_qa.dataset_loader",
        "eval": "md_train_framework.examples.ttt_qa.eval",
        "leaderboard": "md_train_framework.examples.ttt_qa.leaderboard",
        "resume": "md_train_framework.examples.ttt_qa.resume",
        "sweep": "md_train_framework.examples.ttt_qa.sweep",
        "train": "md_train_framework.examples.ttt_qa.train",
    },
}

API_KEY_SLOTS = [
    "CICID_GPUB_MOONDREAM_API_KEY_1",
    "CICID_GPUB_MOONDREAM_API_KEY_2",
    "CICID_GPUB_MOONDREAM_API_KEY_3",
    "CICID_GPUB_MOONDREAM_API_KEY_4",
]


class MdTrainFrameworkShowcaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.fixture_root = self.tmp_path / "fixtures"

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_profile_dispatch_for_showcase_backends(self) -> None:
        cases = (
            ("ballholder", "ballholder_detect", BallHolderDetectTrainer),
            ("statefarm", "statefarm_detect", StatefarmDetectTrainer),
            ("pandid", "pandid_point", PandidPointTrainer),
            ("ttt_qa", "ttt_query", TTTQueryTrainer),
        )
        for example_name, backend_id, expected_type in cases:
            config_path = self._write_local_config(example_name, backend_id=backend_id)
            trainer = self._make_trainer(config_path)
            self.assertIsInstance(trainer, expected_type)

    def test_detect_iou_override_is_applied_by_profile_trainer(self) -> None:
        config_path = self._write_local_config("ballholder", backend_id="ballholder_detect")
        trainer = self._make_trainer(config_path)
        sample = DetectSample(
            sample_id="detect-override",
            split="validation",
            object_name="Player with ball in hand",
            image_url="data:image/png;base64,AA==",
            boxes=(DetectAnnotation(x_min=0.10, y_min=0.10, x_max=0.40, y_max=0.40),),
        )
        shifted = DetectAnnotation(
            x_min=0.16,
            y_min=0.16,
            x_max=0.46,
            y_max=0.46,
        )
        strict = score_detect(
            ground_truth=sample.boxes,
            output=DetectOutput(objects=[shifted]),
            reward_preset_id="detect_miou",
            iou_threshold=0.5,
        )
        relaxed = trainer._score_rollout(sample, DetectOutput(objects=[shifted]))
        self.assertEqual(strict.tp, 0)
        self.assertEqual(relaxed.tp, 1)

    def test_detect_eval_object_name_override_is_applied(self) -> None:
        config_path = self._write_local_config(
            "ballholder",
            backend_id="ballholder_detect",
            backend_overrides={"eval_object_name": "Player with ball in hand"},
        )
        trainer = self._make_trainer(config_path)
        sample = DetectSample(
            sample_id="detect-eval-prompt",
            split="validation",
            object_name="ball holder",
            image_url="data:image/png;base64,AA==",
            boxes=(),
        )
        self.assertEqual(trainer._request_object_name(sample, for_eval=True), "Player with ball in hand")
        self.assertIn(trainer._request_object_name(sample, for_eval=False), trainer._training_prompt_variants(sample))

    def test_detect_miou_penalizes_false_positive_spam(self) -> None:
        ground_truth = [DetectAnnotation(x_min=0.10, y_min=0.10, x_max=0.40, y_max=0.40)]
        single = score_detect(
            ground_truth=ground_truth,
            output=DetectOutput(objects=[DetectAnnotation(x_min=0.10, y_min=0.10, x_max=0.40, y_max=0.40)]),
            reward_preset_id="detect_miou",
            iou_threshold=0.5,
        )
        spam = score_detect(
            ground_truth=ground_truth,
            output=DetectOutput(
                objects=[
                    DetectAnnotation(x_min=0.10, y_min=0.10, x_max=0.40, y_max=0.40),
                    DetectAnnotation(x_min=0.05, y_min=0.05, x_max=0.20, y_max=0.20),
                    DetectAnnotation(x_min=0.60, y_min=0.60, x_max=0.90, y_max=0.90),
                ]
            ),
            reward_preset_id="detect_miou",
            iou_threshold=0.5,
        )
        self.assertAlmostEqual(single.miou, 1.0, places=6)
        self.assertLess(spam.miou, single.miou)

    def test_detect_refusal_penalty_punishes_empty_positive_predictions(self) -> None:
        score = score_detect(
            ground_truth=[DetectAnnotation(x_min=0.10, y_min=0.10, x_max=0.40, y_max=0.40)],
            output=DetectOutput(objects=[]),
            reward_preset_id="detect_f1",
            fn_penalty_weight=2.5,
            fn_penalty_exponent=4.0,
            empty_refusal_penalty=1.5,
            allow_negative_reward=True,
        )
        self.assertEqual(score.fn, 1)
        self.assertTrue(score.positive_empty_prediction)
        self.assertLess(score.reward, -3.0)

    def test_pandid_distance_override_and_recall_first_reward(self) -> None:
        config_path = self._write_local_config(
            "pandid",
            backend_id="pandid_point",
            backend_overrides={"point_distance_threshold": 0.12},
        )
        trainer = self._make_trainer(config_path)
        sample = PointSample(
            sample_id="point-override",
            split="validation",
            object_name="amazon",
            image_url="data:image/png;base64,AA==",
            points=(PointAnnotation(x=0.20, y=0.20, width=None, height=None),),
            boxes=(),
        )
        shifted = PointAnnotation(x=0.31, y=0.20, width=None, height=None)
        strict = score_point(
            ground_truth_points=sample.points,
            ground_truth_boxes=sample.boxes,
            output=PointOutput(points=[shifted]),
            reward_preset_id="point_recall_first",
            distance_threshold=0.08,
        )
        relaxed = trainer._score_rollout(sample, PointOutput(points=[shifted]))
        self.assertEqual(strict.tp, 0)
        self.assertEqual(relaxed.tp, 1)

        recall_first = score_point(
            ground_truth_points=[
                PointAnnotation(x=0.20, y=0.20, width=None, height=None),
                PointAnnotation(x=0.80, y=0.80, width=None, height=None),
            ],
            ground_truth_boxes=[],
            output=PointOutput(points=[PointAnnotation(x=0.20, y=0.20, width=None, height=None)]),
            reward_preset_id="point_recall_first",
        )
        self.assertAlmostEqual(recall_first.recall, 0.5, places=6)
        self.assertAlmostEqual(recall_first.f1, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(recall_first.reward, 0.55, places=2)

    def test_point_sft_defaults_to_boxes_when_available(self) -> None:
        config_path = self._write_local_config(
            "pandid",
            backend_id="pandid_point",
            config_name="sft_then_rl.json",
        )
        trainer = self._make_trainer(config_path)
        phase = next(item for item in trainer.config.phases if item.mode == "sft")
        sample = PointSample(
            sample_id="point-sft-boxes",
            split="train",
            object_name="valve",
            image_url="data:image/png;base64,AA==",
            points=(PointAnnotation(x=0.5, y=0.5, width=None, height=None),),
            boxes=(DetectAnnotation(x_min=0.4, y_min=0.4, x_max=0.6, y_max=0.6),),
        )
        group = trainer._sample_to_sft_group(sample, phase=phase)
        payload = group.to_payload()
        self.assertIn("target", payload)
        self.assertEqual(payload["target"]["boxes"][0]["x_min"], 0.4)
        self.assertNotIn("points", payload["target"])

    def test_string_encoded_annotations_are_parsed_for_detect_and_point(self) -> None:
        detect_config = load_framework_config(self._write_local_config("ballholder", backend_id="ballholder_detect"))
        detect_sample = normalize_row(
            detect_config,
            "validation",
            {
                "sample_id": "detect-string",
                "image_url": "data:image/png;base64,AA==",
                "answer_boxes": '[{"x_min":0.1,"y_min":0.2,"x_max":0.3,"y_max":0.4}]',
            },
            index=1,
        )
        self.assertEqual(len(detect_sample.boxes), 1)
        self.assertAlmostEqual(detect_sample.boxes[0].x_min, 0.1, places=6)

        point_config = load_framework_config(self._write_local_config("pandid", backend_id="pandid_point"))
        point_sample = normalize_row(
            point_config,
            "validation",
            {
                "sample_id": "point-string",
                "image_url": "data:image/png;base64,AA==",
                "answer_points": '[{"x":0.4,"y":0.6,"width":0.1,"height":0.2}]',
            },
            index=1,
        )
        self.assertEqual(len(point_sample.points), 1)
        self.assertAlmostEqual(point_sample.points[0].x, 0.4, places=6)

    def test_legacy_reward_preset_aliases_resolve_to_generic_ids(self) -> None:
        self.assertEqual(get_reward_preset("football_detect_f1").id, "detect_f1")
        self.assertEqual(get_reward_preset("ballholder_detect_miou").id, "detect_miou")
        self.assertEqual(get_reward_preset("pandid_point_recall_first").id, "point_recall_first")
        self.assertEqual(get_reward_preset("inspector_local_judge_hybrid").id, "query_judge_hybrid")
        self.assertEqual(get_reward_preset("ttt_ranked_reward").id, "query_ranked_reward")

    def test_ttt_weighted_sampling_and_task_token_caps(self) -> None:
        task_weights = {task: 0.0 for task in CANONICAL_TASK_TYPES}
        task_weights["available_moves_list"] = 1.0
        config_path = self._write_local_config(
            "ttt_qa",
            backend_id="ttt_query",
            backend_overrides={
                "task_sampling_weights": task_weights,
                "max_tokens_by_task": {"available_moves_list": 128},
            },
            phase_overrides={"rl": {"max_tokens": 64}},
        )
        trainer = self._make_trainer(config_path)
        sampled = trainer._sample_train_batch(12)
        self.assertTrue(sampled)
        self.assertEqual({str(item.meta.get("task_type")) for item in sampled}, {"available_moves_list"})
        phase = next(item for item in trainer.config.phases if item.mode == "rl")
        request = trainer._sample_to_request(sampled[0], phase=phase, for_eval=False)
        self.assertEqual(int(request.settings.max_tokens), 128)

    def test_ttt_best_move_ranking_is_prioritized_and_tracked(self) -> None:
        trainer = self._make_trainer(self._write_local_config("ttt_qa", backend_id="ttt_query"))
        self.assertGreater(trainer.task_sampling_weights["best_move"], 0.0)
        self.assertEqual(
            trainer.task_sampling_weights["best_move"],
            trainer.task_sampling_weights["available_moves_count"],
        )
        self.assertGreater(
            trainer.task_sampling_weights["available_moves_list"],
            trainer.task_sampling_weights["best_move"],
        )

        sample = trainer.train_samples_by_task["best_move"][0]
        optimal = trainer._score_rollout(sample, QueryOutput(answer='{"move": 5}'))
        second_best = trainer._score_rollout(sample, QueryOutput(answer='{"move": 1}'))
        losing = trainer._score_rollout(sample, QueryOutput(answer='{"move": 9}'))

        self.assertTrue(optimal.task_correct)
        self.assertAlmostEqual(optimal.reward, 1.0, places=6)
        self.assertGreater(second_best.best_move_rank_reward, losing.best_move_rank_reward)
        self.assertGreater(second_best.reward, losing.reward)

        metrics = trainer._aggregate_scores([optimal, second_best, losing])
        self.assertIn("eval_best_move_reward_mean", metrics)
        self.assertIn("eval_best_move_rank_reward_mean", metrics)
        self.assertAlmostEqual(metrics["eval_best_move_accuracy"], 1.0 / 3.0, places=6)

    def test_fixed_eval_subset_is_stable_for_repeated_calls(self) -> None:
        config_path = self._write_local_config(
            "ttt_qa",
            backend_id="ttt_query",
            phase_overrides={"rl": {"steps": 1}},
        )
        payload = load_framework_config(config_path).to_dict(include_meta=False)
        payload["eval"]["max_samples"] = 2
        payload["eval"]["fixed_subset_size"] = 2
        payload["eval"]["fixed_subset_seed"] = 19
        save_framework_config(config_path, payload)
        trainer = self._make_trainer(config_path)
        first = [sample.sample_id for sample in trainer._sample_eval_subset("train", trainer.train_samples)]
        second = [sample.sample_id for sample in trainer._sample_eval_subset("train", trainer.train_samples)]
        self.assertEqual(len(first), 2)
        self.assertEqual(first, second)

    def test_ttt_exactness_guard_blocks_reward_only_improvements(self) -> None:
        config_path = self._write_local_config("ttt_qa", backend_id="ttt_query")
        trainer = self._make_trainer(config_path)
        guard = trainer._checkpoint_guard_decision(
            phase=next(item for item in trainer.config.phases if item.mode == "rl"),
            global_step=10,
            checkpoint_event=SimpleNamespace(
                metrics={
                    "eval_accuracy": 0.12,
                    "eval_best_move_accuracy": 0.15,
                    "eval_best_move_canonical_accuracy": 0.12,
                    "eval_reward_mean": 0.31,
                },
                step=10,
                checkpoint_step=10,
                split_name="val",
                metrics_path=self.tmp_path / "metrics.json",
                predictions_path=self.tmp_path / "predictions.jsonl",
                job_json_path=self.tmp_path / "job.json",
            ),
            baseline_metrics={
                "eval_accuracy": 0.19,
                "eval_best_move_accuracy": 0.33,
                "eval_best_move_canonical_accuracy": 0.33,
                "eval_reward_mean": 0.19,
            },
            best_metrics={},
        )
        self.assertEqual(guard["guard"], "ttt_exactness_tradeoff")

    def test_off_policy_replay_only_uses_rl_groups(self) -> None:
        config_path = self._write_local_config(
            "ttt_qa",
            backend_id="ttt_query",
            backend_overrides={
                "off_policy": True,
                "off_policy_min_buffer_groups": 1,
                "off_policy_mix_ratio": 1.0,
                "off_policy_warmup_steps": 0,
            },
        )
        trainer = self._make_trainer(config_path)
        sample = trainer._sample_train_batch(1)[0]
        phase = next(item for item in trainer.config.phases if item.mode == "rl")
        request = trainer._sample_to_request(sample, phase=phase, for_eval=False)
        sft_group = trainer._sample_to_sft_group(sample, phase=phase)
        rl_group = TrainStepGroup.from_rl(
            request=RolloutsRequest(finetune_id="ft-test", num_rollouts=1, request=request),
            rollouts=[Rollout(skill="query", finish_reason="stop", output=QueryOutput(answer=sample.answer))],
            rewards=[1.0],
        )
        trainer.off_policy_buffer.extend([sft_group, rl_group])
        groups = trainer._off_policy_groups(phase=phase, rl_group_count=1)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].mode, "rl")

    def test_phase_finetune_handoff_tracks_final_id(self) -> None:
        class _FakeClient:
            def get_finetune(self, finetune_id: str):
                return SimpleNamespace(finetune_id=str(finetune_id), _client=self)

            def iter_finetunes(self, page_size: int = 100):
                return iter(())

        fake_client = _FakeClient()
        initial = SimpleNamespace(finetune_id="ft-sft", _client=fake_client)
        config_path = self._write_local_config(
            "pandid",
            backend_id="pandid_point",
            config_name="sft_then_rl.json",
            phase_overrides={"rl": {"finetune_id": "ft-rl"}},
        )
        trainer = self._make_trainer(config_path, finetune=initial)
        sft_phase = next(item for item in trainer.config.phases if item.mode == "sft")
        rl_phase = next(item for item in trainer.config.phases if item.mode == "rl")
        trainer._activate_phase_finetune(sft_phase)
        trainer._activate_phase_finetune(rl_phase)
        self.assertEqual(trainer.initial_finetune_id, "ft-sft")
        self.assertEqual(trainer.finetune.finetune_id, "ft-rl")
        self.assertEqual(trainer.phase_finetune_ids[-1]["finetune_id"], "ft-rl")

    def test_phase_finetune_handoff_resolves_latest_name_or_prefix(self) -> None:
        class _FakeClient:
            def __init__(self) -> None:
                self._items = [
                    SimpleNamespace(
                        finetune_id="ft-older",
                        name="pandid-rl-candidate",
                        created_at_ms=10,
                        updated_at_ms=10,
                        _client=self,
                    ),
                    SimpleNamespace(
                        finetune_id="ft-newer",
                        name="pandid-rl-candidate",
                        created_at_ms=20,
                        updated_at_ms=30,
                        _client=self,
                    ),
                    SimpleNamespace(
                        finetune_id="ft-prefix",
                        name="pandid-rl-prefix-20260415",
                        created_at_ms=25,
                        updated_at_ms=25,
                        _client=self,
                    ),
                ]

            def get_finetune(self, finetune_id: str):
                return SimpleNamespace(finetune_id=str(finetune_id), _client=self)

            def iter_finetunes(self, page_size: int = 100):
                return iter(self._items)

        fake_client = _FakeClient()
        initial = SimpleNamespace(finetune_id="ft-sft", _client=fake_client)

        exact_config_path = self._write_local_config(
            "pandid",
            backend_id="pandid_point",
            config_name="sft_then_rl.json",
            phase_overrides={"rl": {"finetune_name": "pandid-rl-candidate"}},
        )
        exact_trainer = self._make_trainer(exact_config_path, finetune=initial)
        exact_rl_phase = next(item for item in exact_trainer.config.phases if item.mode == "rl")
        exact_trainer._activate_phase_finetune(exact_rl_phase)
        self.assertEqual(exact_trainer.finetune.finetune_id, "ft-newer")

        prefix_config_path = self._write_local_config(
            "pandid",
            backend_id="pandid_point",
            config_name="sft_then_rl.json",
            phase_overrides={"rl": {"finetune_name_prefix": "pandid-rl-prefix-"}},
        )
        prefix_trainer = self._make_trainer(prefix_config_path, finetune=initial)
        prefix_rl_phase = next(item for item in prefix_trainer.config.phases if item.mode == "rl")
        prefix_trainer._activate_phase_finetune(prefix_rl_phase)
        self.assertEqual(prefix_trainer.finetune.finetune_id, "ft-prefix")

    def test_profile_sweeps_support_nested_overrides(self) -> None:
        from md_train_framework.sweep import generate_sweep_candidates

        pandid_config = load_framework_config(REPO_ROOT / "md_train_framework" / "examples" / "pandid" / "configs" / "default.json")
        pandid_candidates = generate_sweep_candidates(
            pandid_config,
            output_dir=self.tmp_path / "pandid_sweep",
            max_candidates=1,
        )
        self.assertEqual(len(pandid_candidates), 1)
        pandid_candidate = pandid_candidates[0].config
        pandid_rl = next(item for item in pandid_candidate.phases if item.mode == "rl")
        self.assertEqual([item.mode for item in pandid_candidate.phases], ["rl"])
        self.assertEqual(pandid_rl.lr, 3e-05)
        self.assertEqual(pandid_rl.batch_size, 32)
        self.assertEqual(pandid_rl.group_size, 8)
        self.assertEqual(pandid_candidate.backend.train_overrides["off_policy_mix_ratio"], 0.1)

        pandid_bootstrap_config = load_framework_config(
            REPO_ROOT / "md_train_framework" / "examples" / "pandid" / "configs" / "sft_then_rl.json"
        )
        pandid_bootstrap_candidates = generate_sweep_candidates(
            pandid_bootstrap_config,
            output_dir=self.tmp_path / "pandid_sweep_bootstrap",
            max_candidates=1,
        )
        self.assertEqual(len(pandid_bootstrap_candidates), 1)
        pandid_bootstrap_candidate = pandid_bootstrap_candidates[0].config
        pandid_bootstrap_sft = next(item for item in pandid_bootstrap_candidate.phases if item.mode == "sft")
        pandid_bootstrap_rl = next(item for item in pandid_bootstrap_candidate.phases if item.mode == "rl")
        self.assertEqual(pandid_bootstrap_sft.steps, 20)
        self.assertEqual(pandid_bootstrap_sft.lr, 0.0002)
        self.assertEqual(pandid_bootstrap_rl.lr, 3e-05)
        self.assertEqual(pandid_bootstrap_rl.batch_size, 16)
        self.assertEqual(pandid_bootstrap_candidate.backend.train_overrides["off_policy_mix_ratio"], 0.1)

        ttt_config = load_framework_config(REPO_ROOT / "md_train_framework" / "examples" / "ttt_qa" / "configs" / "default.json")
        ttt_candidates = generate_sweep_candidates(
            ttt_config,
            output_dir=self.tmp_path / "ttt_sweep",
            max_candidates=1,
        )
        self.assertEqual(len(ttt_candidates), 1)
        ttt_candidate = ttt_candidates[0].config
        ttt_rl = next(item for item in ttt_candidate.phases if item.mode == "rl")
        self.assertEqual(ttt_rl.lr, 0.001)
        self.assertEqual(ttt_rl.group_size, 8)
        self.assertTrue(bool(ttt_rl.reasoning))
        self.assertEqual(ttt_candidate.backend.train_overrides["off_policy_mix_ratio"], 0.25)
        self.assertEqual(ttt_candidate.backend.train_overrides["off_policy_warmup_steps"], 10)
        self.assertEqual(ttt_candidate.backend.train_overrides["off_policy_min_buffer_groups"], 64)

    def test_showcase_config_defaults_use_staging_and_framework_outputs(self) -> None:
        expected_eval = {
            "ballholder": (25, 25, 2000),
            "statefarm": (20, 20, None),
            "pandid": (10, 10, 400),
            "ttt_qa": (10, 10, 200),
        }
        for example_name in ("ballholder", "statefarm", "pandid", "ttt_qa"):
            config = load_framework_config(
                REPO_ROOT / "md_train_framework" / "examples" / example_name / "configs" / "default.json"
            )
            self.assertEqual(config.extra["env_file"], "../../../../.env.staging")
            self.assertEqual(config.backend.train_overrides["base_url"], "https://api-staging.moondream.ai/v1")
            self.assertEqual(config.backend.train_overrides["api_key_env_vars"], API_KEY_SLOTS)
            self.assertEqual(config.eval.eval_every, expected_eval[example_name][0])
            self.assertEqual(config.eval.save_every, expected_eval[example_name][1])
            self.assertEqual(config.eval.max_samples, expected_eval[example_name][2])
            if example_name == "pandid":
                self.assertEqual(config.eval.fixed_subset_size, 400)
                self.assertEqual(config.eval.fixed_subset_seed, 1337)
                self.assertEqual(config.backend.train_overrides["off_policy_mix_ratio"], 0.1)
                self.assertEqual(config.backend.train_overrides["recall_gate_step"], 10)
            if example_name == "ttt_qa":
                self.assertEqual(config.eval.fixed_subset_size, 200)
                self.assertEqual(config.eval.fixed_subset_seed, 1337)
                self.assertEqual(config.backend.train_overrides["exactness_guard_min_reward_gain"], 0.02)
            self.assertEqual(
                config.resolved_path(config.logging.run_root),
                REPO_ROOT / "md_train_framework" / "outputs" / "runs",
            )
            self.assertEqual(
                config.resolved_path(config.logging.registry_path),
                REPO_ROOT / "md_train_framework" / "outputs" / "runs.db",
            )

    def test_mock_train_writes_best_and_latest_checkpoint_pointers(self) -> None:
        config_path = self._write_local_config("ballholder", backend_id="mock_detect")
        train_output = self._run_cli(["train", "--config", str(config_path)])
        self.assertIn("run_id", train_output)
        config = load_framework_config(config_path)
        registry = RunRegistry(config.resolved_path(config.logging.registry_path))
        run_record = next(record for record in registry.list_records() if record.record_type == "run")
        run_dir = Path(run_record.artifact_paths["run_dir"])
        best_checkpoint = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
        latest_checkpoint = json.loads((run_dir / "latest_checkpoint.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(best_checkpoint["status"], "best")
        self.assertEqual(latest_checkpoint["status"], "latest")
        self.assertEqual(summary["status"], "succeeded")

    def test_detect_and_json_parse_guards_trigger(self) -> None:
        detect_config_path = self._write_local_config("ballholder", backend_id="ballholder_detect")
        detect_trainer = self._make_trainer(detect_config_path)
        detect_guard = detect_trainer._checkpoint_guard_decision(
            phase=next(item for item in detect_trainer.config.phases if item.mode == "rl"),
            global_step=10,
            checkpoint_event=SimpleNamespace(
                metrics={"eval_f1": 0.05, "eval_miou": 0.70},
                step=10,
                checkpoint_step=10,
                split_name="validation",
                metrics_path=self.tmp_path / "metrics.json",
                predictions_path=self.tmp_path / "predictions.jsonl",
                job_json_path=self.tmp_path / "job.json",
            ),
            baseline_metrics={"eval_f1": 0.60, "eval_miou": 0.50},
            best_metrics={},
        )
        self.assertEqual(detect_guard["guard"], "detect_f1_collapse")

        ttt_config_path = self._write_local_config("ttt_qa", backend_id="ttt_query")
        ttt_trainer = self._make_trainer(ttt_config_path)
        parse_guard = ttt_trainer._checkpoint_guard_decision(
            phase=next(item for item in ttt_trainer.config.phases if item.mode == "rl"),
            global_step=30,
            checkpoint_event=SimpleNamespace(
                metrics={"eval_json_parse_rate": 0.10},
                step=30,
                checkpoint_step=30,
                split_name="val",
                metrics_path=self.tmp_path / "metrics.json",
                predictions_path=self.tmp_path / "predictions.jsonl",
                job_json_path=self.tmp_path / "job.json",
            ),
            baseline_metrics={},
            best_metrics={},
        )
        self.assertEqual(parse_guard["guard"], "json_parse_collapse")

    def test_example_entrypoints_run_with_local_mock_configs(self) -> None:
        cases = (
            ("ballholder", "mock_detect"),
            ("statefarm", "mock_detect"),
            ("pandid", "mock_point"),
            ("ttt_qa", "mock_query"),
        )
        for example_name, backend_id in cases:
            config_path = self._write_local_config(example_name, backend_id=backend_id)
            modules = {
                name: importlib.import_module(path)
                for name, path in EXAMPLE_MODULES[example_name].items()
            }

            inspect_output = self._run_module(modules["dataset_loader"], ["--config", str(config_path)])
            dry_run_output = self._run_cli(["dry-run", "--config", str(config_path)])
            baseline_output = self._run_module(modules["eval"], ["--config", str(config_path)])
            train_output = self._run_module(modules["train"], ["--config", str(config_path)])

            registry = RunRegistry(self._registry_path(example_name))
            records = registry.list_records(task=load_framework_config(config_path).task.name)
            baseline_id = next(record.record_id for record in records if record.record_type == "baseline")
            run_id = next(record.run_id for record in records if record.record_type == "run")

            replay_output = self._run_module(
                modules["eval"],
                ["--config", str(config_path), "--finetune-id", "mock-replay", "--checkpoint-step", "2"],
            )
            leaderboard_output = self._run_module(modules["leaderboard"], ["--config", str(config_path)])
            compare_output = self._run_module(
                modules["compare"],
                ["--config", str(config_path), "--left", run_id, "--right", baseline_id],
            )
            sweep_output = self._run_module(
                modules["sweep"],
                ["--config", str(config_path), "--plan-only", "--max-candidates", "2"],
            )
            resume_output = self._run_module(modules["resume"], ["--config", str(config_path)])

            self.assertIn('"splits"', inspect_output)
            self.assertIn('"backend_id"', dry_run_output)
            self.assertIn('"record_id"', baseline_output)
            self.assertIn('"run_id"', train_output)
            self.assertIn('"record_id"', replay_output)
            self.assertIn('"rows"', leaderboard_output)
            self.assertIn('"selection_metric_delta"', compare_output)
            self.assertIn('"generated_candidates"', sweep_output)
            self.assertIn('"status"', resume_output)

    def _make_trainer(self, config_path: Path, *, finetune: object | None = None):
        config = load_framework_config(config_path)
        reward_preset = get_reward_preset(config.reward.preset)
        metric_policy = build_metric_policy(
            config.skill.id,
            reward_preset,
            requested_selection_metric=config.reward.selection_metric,
        )
        paths = create_run_paths(config, suffix="tests")
        logger = WandbLogger(
            enabled=False,
            project="md-train-framework-tests",
            run_name="",
            config_payload={},
        )
        return make_trainer(
            config=config,
            reward_preset=reward_preset,
            metric_policy=metric_policy,
            paths=paths,
            finetune=finetune or SimpleNamespace(finetune_id="ft-test"),
            logger=logger,
        )

    def _registry_path(self, example_name: str) -> Path:
        return self.tmp_path / example_name / "runs.db"

    def _write_local_config(
        self,
        example_name: str,
        *,
        backend_id: str,
        config_name: str = "default.json",
        backend_overrides: dict[str, object] | None = None,
        phase_overrides: dict[str, dict[str, object]] | None = None,
    ) -> Path:
        base_config_path = REPO_ROOT / "md_train_framework" / "examples" / example_name / "configs" / config_name
        config = load_framework_config(base_config_path)
        payload = config.to_dict(include_meta=False)
        payload["backend"]["id"] = backend_id
        payload["env_file"] = ""
        payload["logging"]["enable_wandb"] = False
        example_root = self.tmp_path / example_name
        payload["logging"]["run_root"] = str(example_root / "runs")
        payload["logging"]["registry_path"] = str(example_root / "runs.db")
        payload["eval"]["async_checkpoint_eval_dir"] = str(example_root / "async")
        payload["recovery"]["quarantine_path"] = str(example_root / "recovery" / "quarantine.jsonl")
        payload["recovery"]["resume_queue_path"] = str(example_root / "recovery" / "resume_queue.jsonl")
        payload["eval"]["eval_every"] = 1
        payload["eval"]["save_every"] = 1
        payload["eval"]["max_samples"] = 8
        payload["sweep"]["max_parallel"] = 1
        payload["sweep"]["stage1_scale"] = 0.5
        for phase in payload["phases"]:
            phase["steps"] = min(int(phase.get("steps", 1)), 3)
        if phase_overrides:
            for phase in payload["phases"]:
                override = phase_overrides.get(str(phase.get("mode") or phase.get("name") or ""))
                if override:
                    phase.update(override)
        if backend_overrides:
            payload["backend"]["train_overrides"].update(dict(backend_overrides))
        payload["dataset"] = self._local_dataset_payload(example_name)
        config_path = example_root / f"{example_name}.json"
        save_framework_config(config_path, payload)
        return config_path

    def _local_dataset_payload(self, example_name: str) -> dict[str, object]:
        images_dir = self.fixture_root / "images"
        self._write_ppm(images_dir / "frame_train_1.ppm", color=(255, 0, 0))
        self._write_ppm(images_dir / "frame_val_1.ppm", color=(0, 255, 0))
        if example_name in {"ballholder", "statefarm"}:
            dataset_dir = self.fixture_root / example_name
            self._write_jsonl(
                dataset_dir / "train.jsonl",
                [
                    {
                        "image_path": "frame_train_1.ppm",
                        "row_id": f"{example_name}-train-1",
                        "boxes": [{"x_min": 0.10, "y_min": 0.10, "x_max": 0.40, "y_max": 0.40}],
                    },
                    {
                        "image_path": "frame_train_1.ppm",
                        "row_id": f"{example_name}-train-2",
                        "boxes": [{"x_min": 0.50, "y_min": 0.50, "x_max": 0.80, "y_max": 0.80}],
                    },
                ],
            )
            self._write_jsonl(
                dataset_dir / "validation.jsonl",
                [
                    {
                        "image_path": "frame_val_1.ppm",
                        "row_id": f"{example_name}-val-1",
                        "boxes": [{"x_min": 0.20, "y_min": 0.20, "x_max": 0.45, "y_max": 0.45}],
                    }
                ],
            )
            self._write_jsonl(
                dataset_dir / "test.jsonl",
                [
                    {
                        "image_path": "frame_val_1.ppm",
                        "row_id": f"{example_name}-test-1",
                        "boxes": [{"x_min": 0.25, "y_min": 0.25, "x_max": 0.50, "y_max": 0.50}],
                    }
                ],
            )
            return {
                "source": "local_jsonl",
                "path": str(dataset_dir),
                "image_root": str(images_dir),
                "split_files": {
                    "train": str(dataset_dir / "train.jsonl"),
                    "validation": str(dataset_dir / "validation.jsonl"),
                    "test": str(dataset_dir / "test.jsonl"),
                },
                "train_split": "train",
                "val_split": "validation",
                "test_split": "test",
            }
        if example_name == "pandid":
            dataset_dir = self.fixture_root / example_name
            self._write_jsonl(
                dataset_dir / "train.jsonl",
                [
                    {
                        "answer_boxes": [
                            {"class_name": "amazon", "x_min": 0.15, "y_min": 0.20, "x_max": 0.35, "y_max": 0.45},
                            {"class_name": "visa", "x_min": 0.60, "y_min": 0.16, "x_max": 0.78, "y_max": 0.38},
                        ],
                        "image_path": "frame_train_1.ppm",
                        "row_id": "pandid-train-1",
                    },
                    {
                        "answer_boxes": [{"class_name": "amazon", "x_min": 0.42, "y_min": 0.50, "x_max": 0.62, "y_max": 0.72}],
                        "image_path": "frame_train_1.ppm",
                        "row_id": "pandid-train-2",
                    },
                ],
            )
            self._write_jsonl(
                dataset_dir / "validation.jsonl",
                [
                    {
                        "answer_boxes": [{"class_name": "visa", "x_min": 0.33, "y_min": 0.31, "x_max": 0.58, "y_max": 0.62}],
                        "image_path": "frame_val_1.ppm",
                        "row_id": "pandid-val-1",
                    }
                ],
            )
            self._write_jsonl(
                dataset_dir / "post_val.jsonl",
                [{"answer_boxes": [], "image_path": "frame_val_1.ppm", "row_id": "pandid-test-1"}],
            )
            return {
                "source": "local_jsonl",
                "path": str(dataset_dir),
                "image_root": str(images_dir),
                "split_files": {
                    "train": str(dataset_dir / "train.jsonl"),
                    "val": str(dataset_dir / "validation.jsonl"),
                    "post_val": str(dataset_dir / "post_val.jsonl"),
                },
                "train_split": "train",
                "val_split": "val",
                "test_split": "post_val",
            }
        dataset_dir = self.fixture_root / example_name
        self._write_jsonl(
            dataset_dir / "train.jsonl",
            [
                {
                    "answer_text": "Reason: Move 5 creates the immediate threat and is the only optimal play.\nFinal: {\"move\": 5}",
                    "best_move_canonical_json": "{\"move\": 5}",
                    "best_move_optimal_set_json": "[{\"move\": 5}]",
                    "final_answer_json": "{\"move\": 5}",
                    "image_path": "frame_train_1.ppm",
                    "question": "What is the best move for X? Return JSON with a move field.",
                    "row_id": "ttt-train-best-move",
                    "scores_by_move_json": "{\"5\": {\"value\": 1, \"depth\": 1}, \"1\": {\"value\": 0, \"depth\": 2}, \"9\": {\"value\": -1, \"depth\": 1}}",
                    "task_type": "best_move",
                },
                {
                    "answer_text": "Reason: There are three legal moves left.\nFinal: {\"available_move_count\": 3}",
                    "final_answer_json": "{\"available_move_count\": 3}",
                    "image_path": "frame_train_1.ppm",
                    "question": "How many legal moves remain? Return JSON.",
                    "row_id": "ttt-train-count",
                    "task_type": "available_moves_count",
                },
                {
                    "answer_text": "Reason: The open squares are the center and two corners.\nFinal: {\"available_moves\": [{\"row\": 1, \"col\": 1}, {\"row\": 2, \"col\": 2}, {\"row\": 3, \"col\": 3}]}",
                    "final_answer_json": "{\"available_moves\": [{\"row\": 1, \"col\": 1}, {\"row\": 2, \"col\": 2}, {\"row\": 3, \"col\": 3}]}",
                    "image_path": "frame_train_1.ppm",
                    "question": "List the legal moves as JSON.",
                    "row_id": "ttt-train-list",
                    "task_type": "available_moves_list",
                },
                {
                    "answer_text": "Reason: X has one more mark than O, so it is O's turn.\nFinal: {\"player\": \"O\"}",
                    "final_answer_json": "{\"player\": \"O\"}",
                    "image_path": "frame_train_1.ppm",
                    "question": "Whose turn is it? Return JSON.",
                    "row_id": "ttt-train-turn",
                    "task_type": "turn_player",
                },
            ],
        )
        self._write_jsonl(
            dataset_dir / "validation.jsonl",
            [
                {
                    "answer_text": "Reason: Move 3 wins on the next line.\nFinal: {\"move\": 3}",
                    "best_move_canonical_json": "{\"move\": 3}",
                    "best_move_optimal_set_json": "[{\"move\": 3}]",
                    "final_answer_json": "{\"move\": 3}",
                    "image_path": "frame_val_1.ppm",
                    "question": "What is the best move for O? Return JSON with a move field.",
                    "row_id": "ttt-val-best-move",
                    "scores_by_move_json": "{\"3\": {\"value\": 1, \"depth\": 1}, \"7\": {\"value\": 0, \"depth\": 2}}",
                    "task_type": "best_move",
                },
                {
                    "answer_text": "Reason: The board is still active.\nFinal: {\"is_game_over\": false}",
                    "final_answer_json": "{\"is_game_over\": false}",
                    "image_path": "frame_val_1.ppm",
                    "question": "Is the game over? Return JSON.",
                    "row_id": "ttt-val-terminal",
                    "task_type": "is_game_over",
                },
            ],
        )
        self._write_jsonl(
            dataset_dir / "test.jsonl",
            [
                {
                    "answer_text": "Reason: Two legal moves remain.\nFinal: {\"available_move_count\": 2}",
                    "final_answer_json": "{\"available_move_count\": 2}",
                    "image_path": "frame_val_1.ppm",
                    "question": "How many legal moves remain? Return JSON.",
                    "row_id": "ttt-test-count",
                    "task_type": "available_moves_count",
                }
            ],
        )
        return {
            "source": "local_jsonl",
            "path": str(dataset_dir),
            "image_root": str(images_dir),
            "split_files": {
                "train": str(dataset_dir / "train.jsonl"),
                "val": str(dataset_dir / "validation.jsonl"),
                "test": str(dataset_dir / "test.jsonl"),
            },
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
        }

    def _write_jsonl(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )

    def _write_ppm(self, path: Path, *, color: tuple[int, int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        r, g, b = color
        path.write_text(
            f"P3\n2 2\n255\n{r} {g} {b} {r} {g} {b}\n{r} {g} {b} {r} {g} {b}\n",
            encoding="utf-8",
        )

    def _run_cli(self, argv: list[str]) -> str:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = cli.main(argv)
        self.assertEqual(code, 0)
        return stdout.getvalue()

    def _run_module(self, module: object, argv: list[str]) -> str:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = module.main(argv)
        self.assertEqual(code, 0)
        return stdout.getvalue()


if __name__ == "__main__":
    unittest.main()
