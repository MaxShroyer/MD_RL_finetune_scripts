from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import _DEPICATED_MDpi_and_d as pid_pkg

sys.modules.setdefault("MDpi_and_d", pid_pkg)

from _DEPICATED_MDpi_and_d import train_pid_icons as pid_base

from aerial_airport import train_aerial_airport_detect as aerial_detect_mod
from aerial_airport import train_aerial_airport_point as aerial_point_mod
from bone_fracture import train_bone_fracture_detect as bone_detect_mod
from bone_fracture import train_bone_fracture_point as bone_point_mod
from construction_site import train_construction_site_detect as construction_detect_mod
from football_detect import train_football_detect as football_base


class WrapperBenchmarkScriptTests(unittest.TestCase):
    def test_wrapper_parse_args_sets_async_benchmark_scripts(self) -> None:
        expectations = [
            (
                construction_detect_mod.parse_args([]),
                "construction_site/benchmark_construction_site_detect.py",
            ),
            (
                aerial_detect_mod.parse_args([]),
                "aerial_airport/benchmark_aerial_airport_detect.py",
            ),
            (
                aerial_point_mod.parse_args([]),
                "aerial_airport/benchmark_aerial_airport_point.py",
            ),
            (
                bone_point_mod.parse_args([]),
                "bone_fracture/benchmark_bone_fracture_point.py",
            ),
            (
                bone_detect_mod.parse_args([]),
                "bone_fracture/benchmark_bone_fracture_detect.py",
            ),
        ]

        for args, expected_suffix in expectations:
            with self.subTest(expected_suffix=expected_suffix):
                resolved_path = Path(str(args.async_checkpoint_eval_benchmark_script)).resolve()
                self.assertTrue(str(resolved_path).endswith(expected_suffix))

    def test_football_wrapper_async_command_uses_wrapper_script(self) -> None:
        args = bone_detect_mod.parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cmd = football_base._build_async_checkpoint_eval_command(
                args=args,
                finetune_id="ft_123",
                split_name="validation",
                checkpoint_step=12,
                metrics_json_path=tmp_path / "metrics.json",
                predictions_jsonl_path=tmp_path / "predictions.jsonl",
            )
        self.assertEqual(Path(cmd[1]).resolve(), Path(args.async_checkpoint_eval_benchmark_script).resolve())

    def test_pid_wrapper_async_command_uses_wrapper_script(self) -> None:
        args = bone_point_mod.parse_args([])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cmd = pid_base._build_async_checkpoint_eval_command(
                args=args,
                split_name="validation",
                finetune_id="ft_123",
                checkpoint_step=9,
                effective_point_prompt_style=str(args.point_prompt_style),
                metrics_json_path=tmp_path / "metrics.json",
                records_jsonl_path=tmp_path / "records.jsonl",
            )
        self.assertEqual(Path(cmd[1]).resolve(), Path(args.async_checkpoint_eval_benchmark_script).resolve())


if __name__ == "__main__":
    unittest.main()
