from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tictaktoe_QA import train_ttt_query_rl_compact as mod


def _eval_metrics(reward: float) -> dict[str, float]:
    return {
        "eval_samples": 1.0,
        "eval_reward_mean": float(reward),
        "eval_json_object_rate": 1.0,
        "eval_json_parse_rate": 1.0,
        "eval_best_move_set_accuracy": 0.0,
        "eval_best_move_canonical_accuracy": 0.0,
        "eval_best_move_valid_prediction_count": 0.0,
        "eval_best_move_valid_prediction_rate": 0.0,
        "eval_best_move_center_prediction_rate": 0.0,
        "eval_best_move_invalid_prediction_rate": 0.0,
        "eval_exact_accuracy_non_best_move": 1.0,
        "eval_task_accuracy_available_moves_count": 1.0,
        "eval_task_count_available_moves_count": 1.0,
    }


class ParserTests(unittest.TestCase):
    def test_parse_args_uses_config_defaults_and_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "env_file": ".env",
                        "dataset_source": "local_jsonl",
                        "dataset_dir": "synth_dataset/outputs/v2",
                        "train_split": "train",
                        "val_split": "val",
                        "final_eval_splits": ["val", "test"],
                        "finetune_name": "compact-test",
                        "rank": 8,
                        "seed": 7,
                        "num_steps": 4,
                        "resume_step": 3,
                        "batch_size": 2,
                        "group_size": 3,
                        "lr": 0.001,
                        "max_workers": 2,
                        "rollout_retries": 1,
                        "rollout_retry_backoff_s": 0.5,
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "max_tokens": 320,
                        "eval_temperature": 0.1,
                        "eval_top_p": 1.0,
                        "reasoning": True,
                        "eval_reasoning": False,
                        "task_sampling_weights": {
                            "best_move": 2.0,
                            "available_moves_count": 1.5,
                        },
                        "max_tokens_by_task": {"available_moves_list": 512},
                        "off_policy": True,
                        "off_policy_mix_ratio": 0.25,
                        "off_policy_buffer_size": 256,
                        "off_policy_warmup_steps": 5,
                        "off_policy_min_buffer_groups": 32,
                        "eval_every": 8,
                        "save_every": 10,
                        "save_on_eval": False,
                        "eval_batch_size": 11,
                        "eval_max_samples": 22,
                        "eval_fixed_subset_size": 5,
                        "eval_fixed_subset_seed": 99,
                        "best_metric": "eval_json_parse_rate",
                        "best_move_optimal_reward": 0.6,
                        "best_move_reward_mode": "hybrid_strict",
                        "best_move_wrong_rank_scale": 0.3,
                        "best_move_center_not_optimal_ratio": 0.4,
                        "intra_task_sampling_json": {"best_move": "center_hard_negative"},
                        "skip_final_eval": True,
                        "no_progress": True,
                        "wandb_project": "proj",
                        "wandb_run_name": "run",
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "--config",
                str(cfg_path),
                "--num-steps",
                "9",
                "--batch-size",
                "5",
            ]
            args = mod.parse_args(argv)

            self.assertEqual(args.dataset_source, "local_jsonl")
            self.assertEqual(args.dataset_dir, "synth_dataset/outputs/v2")
            self.assertEqual(args.train_split, "train")
            self.assertEqual(args.val_split, "val")
            self.assertEqual(args.final_eval_splits, ["val", "test"])
            self.assertEqual(args.finetune_name, "compact-test")
            self.assertEqual(args.rank, 8)
            self.assertEqual(args.seed, 7)
            self.assertEqual(args.num_steps, 9)
            self.assertEqual(args.resume_step, 3)
            self.assertEqual(args.batch_size, 5)
            self.assertEqual(args.group_size, 3)
            self.assertAlmostEqual(args.lr, 0.001, places=8)
            self.assertEqual(args.max_workers, 2)
            self.assertEqual(args.rollout_retries, 1)
            self.assertAlmostEqual(args.rollout_retry_backoff_s, 0.5, places=8)
            self.assertAlmostEqual(args.temperature, 0.8, places=8)
            self.assertAlmostEqual(args.top_p, 0.95, places=8)
            self.assertEqual(args.max_tokens, 320)
            self.assertAlmostEqual(float(args.eval_temperature), 0.1, places=8)
            self.assertAlmostEqual(float(args.eval_top_p), 1.0, places=8)
            self.assertTrue(args.reasoning)
            self.assertFalse(args.eval_reasoning)
            self.assertTrue(args.off_policy)
            self.assertAlmostEqual(args.off_policy_mix_ratio, 0.25, places=8)
            self.assertEqual(args.off_policy_buffer_size, 256)
            self.assertEqual(args.off_policy_warmup_steps, 5)
            self.assertEqual(args.off_policy_min_buffer_groups, 32)
            self.assertEqual(args.eval_every, 8)
            self.assertEqual(args.save_every, 10)
            self.assertFalse(args.save_on_eval)
            self.assertEqual(args.eval_batch_size, 11)
            self.assertEqual(args.eval_max_samples, 22)
            self.assertEqual(args.eval_fixed_subset_size, 5)
            self.assertEqual(args.eval_fixed_subset_seed, 99)
            self.assertEqual(args.best_metric, "eval_json_parse_rate")
            self.assertAlmostEqual(args.best_move_optimal_reward, 0.6, places=8)
            self.assertEqual(args.best_move_reward_mode, "hybrid_strict")
            self.assertAlmostEqual(args.best_move_wrong_rank_scale, 0.3, places=8)
            self.assertAlmostEqual(args.best_move_center_not_optimal_ratio, 0.4, places=8)
            self.assertTrue(args.skip_final_eval)
            self.assertTrue(args.no_progress)
            self.assertEqual(args.wandb_project, "proj")
            self.assertEqual(args.wandb_run_name, "run")
            self.assertEqual(
                args.task_sampling_weights,
                {
                    "best_move": 2.0,
                    "has_winning_move": 1.0,
                    "turn_player": 1.0,
                    "winner": 1.0,
                    "is_game_over": 1.0,
                    "available_moves_count": 1.5,
                    "available_moves_list": 1.0,
                },
            )
            self.assertEqual(args.max_tokens_by_task, {"available_moves_list": 512})
            self.assertEqual(args.intra_task_sampling, {"best_move": "center_hard_negative"})

    def test_parse_args_warns_on_unknown_config_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "num_steps": 5,
                        "legacy_internal_knob": True,
                    }
                ),
                encoding="utf-8",
            )

            with patch("builtins.print") as mock_print:
                args = mod.parse_args(["--config", str(cfg_path)])

            self.assertEqual(args.num_steps, 5)
            self.assertTrue(
                any(
                    "ignoring unknown config keys" in str(call.args[0])
                    for call in mock_print.call_args_list
                    if call.args
                )
            )


class CompactHelperTests(unittest.TestCase):
    def test_compose_train_groups_uses_replay_after_warmup(self) -> None:
        rng = random.Random(7)
        train_groups, off_policy_count = mod._compose_train_groups(
            on_policy_groups=["a", "b", "c", "d"],
            replay_groups=["r1", "r2", "r3", "r4"],
            off_policy=True,
            off_policy_mix_ratio=0.5,
            off_policy_warmup_steps=2,
            off_policy_min_buffer_groups=2,
            global_step=5,
            rng=rng,
        )

        self.assertEqual(len(train_groups), 4)
        self.assertEqual(off_policy_count, 2)
        self.assertEqual(sum(1 for item in train_groups if str(item).startswith("r")), 2)


class CompactScoreOutcomeTests(unittest.TestCase):
    def _best_move_example(self) -> mod.QAExample:
        return mod.QAExample(
            row_id="r_best",
            split="train",
            task_type="best_move",
            question="q",
            image_path=Path("/tmp/unused.png"),
            expected_answer={"row": 1, "col": 1},
            best_move_canonical=1,
            best_move_optimal_set=frozenset({1}),
            best_move_scores=((1, 1, 2), (5, 0, 4), (9, -1, 6)),
            best_move_legal_moves=frozenset({1, 5, 9}),
        )

    def test_best_move_valid_prediction_true_for_legal_non_optimal_move(self) -> None:
        out = mod._score_payload_for_example(
            self._best_move_example(),
            {"row": 2, "col": 2},
            best_move_optimal_reward=0.7,
        )
        self.assertTrue(out.best_move_valid_prediction)
        self.assertFalse(out.best_move_set_correct)

    def test_best_move_valid_prediction_false_for_occupied_square(self) -> None:
        out = mod._score_payload_for_example(
            self._best_move_example(),
            {"row": 2, "col": 1},
            best_move_optimal_reward=0.7,
        )
        self.assertFalse(out.best_move_valid_prediction)
        self.assertFalse(out.best_move_set_correct)


class CompactMainFlowTests(unittest.TestCase):
    class _FakeRolloutResult:
        def __init__(self, answer: str, num_rollouts: int) -> None:
            self.rollouts = [
                SimpleNamespace(output=SimpleNamespace(answer=answer))
                for _ in range(num_rollouts)
            ]

        def to_group(self, *, rewards: list[float]) -> dict[str, object]:
            return {"rewards": list(rewards)}

    class _FakeFinetune:
        def __init__(self, answer: str) -> None:
            self.finetune_id = "ft_compact_test"
            self.name = "ft_compact_test"
            self.answer = answer
            self.train_steps = 0
            self.saved_checkpoints = 0

        def train_step(self, *, groups: list[object], lr: float) -> SimpleNamespace:
            _ = groups
            _ = lr
            self.train_steps += 1
            return SimpleNamespace(kl=0.01, router_kl=0.0, grad_norm=1.0)

        def save_checkpoint(self) -> SimpleNamespace:
            self.saved_checkpoints += 1
            return SimpleNamespace(ok=True)

    class _FakeClient:
        def __init__(self, finetune: "CompactMainFlowTests._FakeFinetune") -> None:
            self._finetune = finetune
            self.created_names: list[str] = []
            self.got_finetune_ids: list[str] = []
            self.closed = False

        def create_finetune(self, *, name: str, rank: int) -> "CompactMainFlowTests._FakeFinetune":
            _ = rank
            self.created_names.append(name)
            self._finetune.name = name
            return self._finetune

        def get_finetune(self, finetune_id: str) -> "CompactMainFlowTests._FakeFinetune":
            self.got_finetune_ids.append(finetune_id)
            return self._finetune

        def close(self) -> None:
            self.closed = True

    class _FakeRun:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}
            self.finished = False

        def finish(self) -> None:
            self.finished = True

    class _FakeWandb:
        def __init__(self) -> None:
            self.run: CompactMainFlowTests._FakeRun | None = None
            self.logs: list[tuple[dict[str, float], int | None]] = []

        def init(self, *args: object, **kwargs: object) -> "CompactMainFlowTests._FakeRun":
            _ = args
            _ = kwargs
            self.run = CompactMainFlowTests._FakeRun()
            return self.run

        def log(self, payload: dict[str, float], step: int | None = None) -> None:
            self.logs.append((dict(payload), step))

    def _example(self, image_path: Path, *, split_name: str) -> mod.QAExample:
        return mod.QAExample(
            row_id=f"row_{split_name}",
            split=split_name,
            task_type="available_moves_count",
            question="How many legal moves are available?",
            image_path=image_path,
            expected_answer={"available_move_count": 1},
            best_move_canonical=None,
            best_move_optimal_set=frozenset(),
        )

    def _run_main_with_mocks(
        self,
        *,
        argv: list[str],
        eval_metrics: list[dict[str, float]],
    ) -> tuple[
        "CompactMainFlowTests._FakeClient",
        "CompactMainFlowTests._FakeFinetune",
        "CompactMainFlowTests._FakeWandb",
        list[str],
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "board.png"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            split_map = {
                "train": [self._example(image_path, split_name="train")],
                "val": [self._example(image_path, split_name="val")],
                "test": [self._example(image_path, split_name="test")],
            }
            fake_finetune = self._FakeFinetune(answer='{"available_move_count": 1}')
            fake_client = self._FakeClient(fake_finetune)
            fake_wandb = self._FakeWandb()
            eval_calls: list[str] = []
            eval_queue = list(eval_metrics)

            def _fake_load_split_examples(*, split_name: str, **kwargs: object) -> list[mod.QAExample]:
                _ = kwargs
                return list(split_map[split_name])

            def _fake_rollouts_batch_with_retry(**kwargs: object) -> list[CompactMainFlowTests._FakeRolloutResult]:
                requests = list(kwargs["requests"])
                num_rollouts = int(kwargs["num_rollouts"])
                return [
                    CompactMainFlowTests._FakeRolloutResult(fake_finetune.answer, num_rollouts)
                    for _ in requests
                ]

            def _fake_evaluate_split(*, split_name: str, **kwargs: object) -> dict[str, float]:
                _ = kwargs
                eval_calls.append(split_name)
                return eval_queue.pop(0)

            with patch.object(mod, "wandb", fake_wandb), patch.object(
                mod, "TunaClient", return_value=fake_client
            ), patch.object(mod, "_load_split_examples", side_effect=_fake_load_split_examples), patch.object(
                mod, "_rollouts_batch_with_retry", side_effect=_fake_rollouts_batch_with_retry
            ), patch.object(
                mod, "_evaluate_split", side_effect=_fake_evaluate_split
            ):
                mod.main(argv)

            return fake_client, fake_finetune, fake_wandb, eval_calls

    def test_main_create_flow_updates_best_metric_and_runs_final_eval(self) -> None:
        client, finetune, fake_wandb, eval_calls = self._run_main_with_mocks(
            argv=[
                "--api-key",
                "test-key",
                "--base-url",
                "https://example.invalid/v1",
                "--num-steps",
                "2",
                "--batch-size",
                "1",
                "--group-size",
                "1",
                "--eval-every",
                "2",
                "--save-every",
                "0",
                "--no-save-on-eval",
                "--final-eval-splits",
                "val",
                "test",
                "--no-progress",
            ],
            eval_metrics=[
                _eval_metrics(0.2),
                _eval_metrics(0.7),
                _eval_metrics(0.6),
                _eval_metrics(0.5),
            ],
        )

        assert fake_wandb.run is not None
        self.assertEqual(eval_calls, ["val", "val", "val", "test"])
        self.assertEqual(len(client.created_names), 1)
        self.assertEqual(client.got_finetune_ids, [])
        self.assertTrue(client.closed)
        self.assertEqual(finetune.saved_checkpoints, 2)
        self.assertEqual(fake_wandb.run.summary["best_metric_step"], 1)
        self.assertAlmostEqual(float(fake_wandb.run.summary["best_metric_value"]), 0.7, places=6)
        self.assertNotIn("auto_benchmark_success", fake_wandb.run.summary)
        self.assertNotIn("checkpoint_ranking_output", fake_wandb.run.summary)
        self.assertTrue(
            any("final_test_eval_reward_mean" in payload for payload, _step in fake_wandb.logs)
        )

    def test_main_resume_flow_uses_get_finetune(self) -> None:
        client, finetune, fake_wandb, eval_calls = self._run_main_with_mocks(
            argv=[
                "--api-key",
                "test-key",
                "--base-url",
                "https://example.invalid/v1",
                "--finetune-id",
                "ft_resume",
                "--num-steps",
                "0",
                "--eval-every",
                "0",
                "--save-every",
                "0",
                "--skip-final-eval",
                "--no-progress",
            ],
            eval_metrics=[],
        )

        assert fake_wandb.run is not None
        self.assertEqual(eval_calls, [])
        self.assertEqual(client.created_names, [])
        self.assertEqual(client.got_finetune_ids, ["ft_resume"])
        self.assertEqual(finetune.saved_checkpoints, 1)
        self.assertTrue(fake_wandb.run.finished)

    def test_main_save_on_eval_and_save_every(self) -> None:
        client, finetune, fake_wandb, eval_calls = self._run_main_with_mocks(
            argv=[
                "--api-key",
                "test-key",
                "--base-url",
                "https://example.invalid/v1",
                "--num-steps",
                "2",
                "--batch-size",
                "1",
                "--group-size",
                "1",
                "--eval-every",
                "1",
                "--save-every",
                "1",
                "--save-on-eval",
                "--skip-final-eval",
                "--no-progress",
            ],
            eval_metrics=[
                _eval_metrics(0.2),
                _eval_metrics(0.3),
                _eval_metrics(0.4),
            ],
        )

        assert fake_wandb.run is not None
        self.assertEqual(eval_calls, ["val", "val", "val"])
        self.assertEqual(client.got_finetune_ids, [])
        self.assertEqual(finetune.saved_checkpoints, 6)
        self.assertTrue(fake_wandb.run.finished)


if __name__ == "__main__":
    unittest.main()
