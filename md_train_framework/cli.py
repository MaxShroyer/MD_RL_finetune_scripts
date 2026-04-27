from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from md_train_framework.artifacts import create_run_paths, record_failure
from md_train_framework.compat import (
    api_key_slots,
    backend_ids,
    baseline_backend,
    render_train_config,
    replay_eval_backend,
    resolve_backend,
    train_backend,
)
from md_train_framework.config import load_framework_config
from md_train_framework.datasets import inspect_dataset
from md_train_framework.legacy import default_legacy_roots, import_legacy_runs
from md_train_framework.metrics import build_metric_policy, metric_deltas
from md_train_framework.registry import RegistryRecord, RunRegistry, leaderboard_rows
from md_train_framework.rewards import get_reward_preset, list_reward_aliases, list_reward_presets
from md_train_framework.sweep import SweepOrchestrator, generate_sweep_candidates, stage1_candidates
from md_train_framework.utils import append_jsonl, now_utc_iso, slugify, write_json


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", ""):
        parser.print_help()
        return 1
    return int(args.func(args))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generic Moondream finetune framework.")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Inspect dataset splits and fields.")
    _add_config_arg(inspect_parser)
    inspect_parser.add_argument("--output-json", default="")
    inspect_parser.set_defaults(func=cmd_inspect_dataset)

    dry_run_parser = subparsers.add_parser("dry-run", help="Validate config, reward, dataset, backend, and API slots.")
    _add_config_arg(dry_run_parser)
    dry_run_parser.add_argument("--output-json", default="")
    dry_run_parser.set_defaults(func=cmd_dry_run)

    train_parser = subparsers.add_parser("train", help="Run training through the internal tuna-sdk runtime.")
    _add_config_arg(train_parser)
    train_parser.set_defaults(func=cmd_train)

    resume_parser = subparsers.add_parser("resume", help="Resume the latest queued run for this config or fall back to training.")
    _add_config_arg(resume_parser)
    resume_parser.add_argument("--output-json", default="")
    resume_parser.set_defaults(func=cmd_resume)

    baseline_parser = subparsers.add_parser("baseline", help="Run a baseline or checkpoint eval through the internal tuna-sdk runtime.")
    _add_config_arg(baseline_parser)
    baseline_parser.add_argument("--finetune-id", default="")
    baseline_parser.add_argument("--checkpoint-step", type=int, default=-1)
    baseline_parser.set_defaults(func=cmd_baseline)

    leaderboard_parser = subparsers.add_parser("leaderboard", help="Show the best runs or checkpoints in the local registry.")
    _add_optional_config_arg(leaderboard_parser)
    leaderboard_parser.add_argument("--registry-path", default="")
    leaderboard_parser.add_argument("--skill", default="")
    leaderboard_parser.add_argument("--task", default="")
    leaderboard_parser.add_argument("--limit", type=int, default=10)
    leaderboard_parser.add_argument("--output-json", default="")
    leaderboard_parser.set_defaults(func=cmd_leaderboard)

    compare_parser = subparsers.add_parser("compare", help="Compare two runs or records from the local registry.")
    _add_optional_config_arg(compare_parser)
    compare_parser.add_argument("--registry-path", default="")
    compare_parser.add_argument("--left", required=True)
    compare_parser.add_argument("--right", required=True)
    compare_parser.add_argument("--output-json", default="")
    compare_parser.set_defaults(func=cmd_compare)

    sweep_parser = subparsers.add_parser("sweep", help="Generate or execute a staged sweep.")
    _add_config_arg(sweep_parser)
    sweep_parser.add_argument("--max-candidates", type=int, default=12)
    sweep_parser.add_argument("--plan-only", action="store_true")
    sweep_parser.add_argument("--output-json", default="")
    sweep_parser.set_defaults(func=cmd_sweep)

    import_parser = subparsers.add_parser("import-legacy-runs", help="Scan existing run artifacts into the registry.")
    _add_optional_config_arg(import_parser)
    import_parser.add_argument("--registry-path", default="")
    import_parser.add_argument("--root", action="append", default=[])
    import_parser.add_argument("--output-json", default="")
    import_parser.set_defaults(func=cmd_import_legacy_runs)

    replay_parser = subparsers.add_parser("replay-eval", help="Benchmark a finetune or checkpoint again.")
    _add_config_arg(replay_parser)
    replay_parser.add_argument("--finetune-id", required=True)
    replay_parser.add_argument("--checkpoint-step", type=int, default=-1)
    replay_parser.set_defaults(func=cmd_replay_eval)

    presets_parser = subparsers.add_parser("list-presets", help="List registered reward presets and backend ids.")
    presets_parser.add_argument("--skill", default="")
    presets_parser.set_defaults(func=cmd_list_presets)

    return parser


def cmd_inspect_dataset(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config)
    from md_train_framework.runtime import _load_env_file

    _load_env_file(config)
    inspection = inspect_dataset(config)
    payload = inspection.to_dict()
    if str(args.output_json).strip():
        write_json(Path(args.output_json).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config)
    from md_train_framework.runtime import _load_env_file

    _load_env_file(config)
    reward_preset = get_reward_preset(config.reward.preset)
    metric_policy = build_metric_policy(
        config.skill.id,
        reward_preset,
        requested_selection_metric=config.reward.selection_metric,
    )
    inspection = inspect_dataset(config)
    backend = resolve_backend(config)
    rendered = render_train_config(config, create_run_paths(config, suffix="dry-run"), reward_preset=reward_preset)
    slots = api_key_slots(config, rendered)
    payload = {
        "config_hash": config.config_hash,
        "mode": config.mode,
        "skill": config.skill.id,
        "task": config.task.name,
        "backend_id": backend.id,
        "selection_metric": metric_policy.selection_metric,
        "dataset_fingerprint": inspection.fingerprint,
        "api_key_slots": [
            {"env_var": slot, "present": bool(os.environ.get(slot))}
            for slot in slots
        ],
        "dataset": inspection.to_dict(),
        "rendered_backend_config": rendered,
    }
    if str(args.output_json).strip():
        write_json(Path(args.output_json).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config)
    reward_preset = get_reward_preset(config.reward.preset)
    metric_policy = build_metric_policy(
        config.skill.id,
        reward_preset,
        requested_selection_metric=config.reward.selection_metric,
    )
    inspection = inspect_dataset(config)
    registry = RunRegistry(config.resolved_path(config.logging.registry_path))
    paths = create_run_paths(config)
    try:
        result = train_backend(config, paths, reward_preset, metric_policy)
    except Exception as exc:
        record_failure(paths, stage="train", message=f"{type(exc).__name__}: {exc}")
        _append_resume_queue(config, message=f"{type(exc).__name__}: {exc}")
        raise
    dataset_fingerprint = inspection.fingerprint
    baseline_record = registry.latest_baseline(
        skill=config.skill.id,
        task=config.task.name,
        dataset_fingerprint=dataset_fingerprint,
    )
    baseline_metrics = dict(baseline_record.metrics) if baseline_record else dict(result.baseline_metrics)
    delta = metric_deltas(result.metrics, baseline_metrics)
    run_record = RegistryRecord(
        record_type="run",
        run_id=paths.run_id,
        skill=config.skill.id,
        task=config.task.name,
        mode=config.mode,
        backend_id=resolve_backend(config).id,
        status=result.status,
        config_hash=config.config_hash,
        dataset_fingerprint=dataset_fingerprint,
        finetune_id=result.finetune_id,
        selection_metric_name=result.selection_metric_name,
        selection_metric_value=result.selection_metric_value,
        metrics=result.metrics,
        baseline_metrics=baseline_metrics,
        delta_metrics=delta,
        artifact_paths=result.artifact_paths,
        source_provenance={
            "source": "md_train_framework",
            "config_path": str(Path(args.config).expanduser().resolve()),
        },
        metadata={
            "reward_preset": reward_preset.id,
            "selection_metric": result.selection_metric_name,
            "logged_at": now_utc_iso(),
            "dataset_summary": inspection.to_dict(),
        },
    )
    registry.upsert(run_record)
    _ingest_local_checkpoints(
        registry,
        config=config,
        run_id=paths.run_id,
        finetune_id=result.finetune_id,
        dataset_fingerprint=dataset_fingerprint,
        selection_metric_name=result.selection_metric_name,
        baseline_metrics=baseline_metrics,
        eval_history_path=paths.eval_history_jsonl,
        async_eval_dir=paths.async_eval_dir,
    )
    payload = {
        "run_id": paths.run_id,
        "finetune_id": result.finetune_id,
        "selection_metric_name": result.selection_metric_name,
        "selection_metric_value": result.selection_metric_value,
        "artifact_paths": result.artifact_paths,
        "delta_vs_baseline": delta,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config)
    queue_path = config.resolved_path(config.recovery.resume_queue_path)
    entries = _read_resume_queue(queue_path)
    matching = [
        entry
        for entry in entries
        if str(entry.get("config_hash", "")).strip() == config.config_hash
        or str(entry.get("config_path", "")).strip() == str(config.config_path or "")
        or (
            str(entry.get("task", "")).strip() == config.task.name
            and str(entry.get("skill", "")).strip() == config.skill.id
        )
    ]
    selected = matching[-1] if matching else (entries[-1] if entries else None)
    payload = {
        "status": "queued_resume" if selected is not None else "fresh_train",
        "queue_path": str(queue_path),
        "resume_entry": selected,
    }
    if selected is not None:
        remaining = entries[:]
        remaining.remove(selected)
        _write_resume_queue(queue_path, remaining)
    if str(args.output_json).strip():
        write_json(Path(args.output_json).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return cmd_train(args)


def cmd_baseline(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config)
    reward_preset = get_reward_preset(config.reward.preset)
    metric_policy = build_metric_policy(
        config.skill.id,
        reward_preset,
        requested_selection_metric=config.reward.selection_metric,
    )
    inspection = inspect_dataset(config)
    registry = RunRegistry(config.resolved_path(config.logging.registry_path))
    paths = create_run_paths(config, suffix="baseline")
    result = baseline_backend(
        config,
        paths,
        reward_preset,
        metric_policy,
        finetune_id=str(args.finetune_id or "").strip(),
        checkpoint_step=None if int(args.checkpoint_step) < 0 else int(args.checkpoint_step),
    )
    record = RegistryRecord(
        record_type="baseline" if int(args.checkpoint_step) < 0 else "checkpoint",
        run_id=paths.run_id,
        parent_run_id=str(args.finetune_id or "").strip(),
        skill=config.skill.id,
        task=config.task.name,
        mode="eval",
        backend_id=resolve_backend(config).id,
        status=result.status,
        config_hash=config.config_hash,
        dataset_fingerprint=inspection.fingerprint,
        finetune_id=result.finetune_id,
        checkpoint_step=None if int(args.checkpoint_step) < 0 else int(args.checkpoint_step),
        selection_metric_name=result.selection_metric_name,
        selection_metric_value=result.selection_metric_value,
        metrics=result.metrics,
        artifact_paths=result.artifact_paths,
        source_provenance={"source": "md_train_framework", "config_path": str(Path(args.config).expanduser().resolve())},
        metadata={"reward_preset": reward_preset.id},
    )
    registry.upsert(record)
    print(
        json.dumps(
            {
                "record_id": record.with_defaults().record_id,
                "selection_metric_name": result.selection_metric_name,
                "selection_metric_value": result.selection_metric_value,
                "artifact_paths": result.artifact_paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config) if str(getattr(args, "config", "")).strip() else None
    registry = _resolve_registry(args, config=config)
    rows = leaderboard_rows(
        registry.leaderboard(
            skill=str(args.skill or (config.skill.id if config else "")).strip() or None,
            task=str(args.task or (config.task.name if config else "")).strip() or None,
            limit=int(args.limit),
        )
    )
    payload = {"rows": rows}
    if str(args.output_json).strip():
        write_json(Path(args.output_json).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config) if str(getattr(args, "config", "")).strip() else None
    registry = _resolve_registry(args, config=config)
    payload = registry.compare(left_id=str(args.left), right_id=str(args.right))
    if str(args.output_json).strip():
        write_json(Path(args.output_json).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config)
    registry = RunRegistry(config.resolved_path(config.logging.registry_path))
    candidates = generate_sweep_candidates(
        config,
        registry=registry,
        max_candidates=int(args.max_candidates),
    )
    stage1 = stage1_candidates(candidates, scale=config.sweep.stage1_scale)
    payload: dict[str, Any] = {
        "generated_candidates": [str(candidate.config_path) for candidate in candidates],
        "stage1_candidates": [str(candidate.config_path) for candidate in stage1],
    }
    if not args.plan_only:
        orchestrator = SweepOrchestrator(registry)
        sweep_root = config.resolved_path("md_train_framework/outputs/sweeps")
        quarantine_path = sweep_root / slugify(config.task.name) / "quarantine.jsonl"
        stage1_result = orchestrator.run(
            stage1,
            stagger_seconds=config.sweep.stagger_seconds,
            max_parallel=config.sweep.max_parallel,
            quarantine_path=quarantine_path,
        )
        top = orchestrator.select_top_candidates(stage1, top_k=config.sweep.continue_top_k)
        stage2_result = orchestrator.run(
            top,
            stagger_seconds=config.sweep.stagger_seconds,
            max_parallel=config.sweep.max_parallel,
            quarantine_path=quarantine_path,
        ) if top else {"completed": []}
        payload["stage1_result"] = stage1_result
        payload["stage2_candidates"] = [str(candidate.config_path) for candidate in top]
        payload["stage2_result"] = stage2_result
    if str(args.output_json).strip():
        write_json(Path(args.output_json).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_import_legacy_runs(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config) if str(getattr(args, "config", "")).strip() else None
    repo_root = Path(__file__).resolve().parents[1]
    registry = _resolve_registry(args, config=config)
    roots = list(args.root or [])
    if not roots:
        roots = [str(path) for path in default_legacy_roots(repo_root)]
    payload = import_legacy_runs(registry, roots=roots, repo_root=repo_root)
    if str(args.output_json).strip():
        write_json(Path(args.output_json).expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_replay_eval(args: argparse.Namespace) -> int:
    config = load_framework_config(args.config)
    reward_preset = get_reward_preset(config.reward.preset)
    metric_policy = build_metric_policy(
        config.skill.id,
        reward_preset,
        requested_selection_metric=config.reward.selection_metric,
    )
    inspection = inspect_dataset(config)
    registry = RunRegistry(config.resolved_path(config.logging.registry_path))
    paths = create_run_paths(config, suffix="replay-eval")
    result = replay_eval_backend(
        config,
        paths,
        reward_preset,
        metric_policy,
        finetune_id=str(args.finetune_id).strip(),
        checkpoint_step=None if int(args.checkpoint_step) < 0 else int(args.checkpoint_step),
    )
    record = RegistryRecord(
        record_type="checkpoint",
        run_id=paths.run_id,
        parent_run_id=str(args.finetune_id),
        skill=config.skill.id,
        task=config.task.name,
        mode="eval",
        backend_id=resolve_backend(config).id,
        status=result.status,
        config_hash=config.config_hash,
        dataset_fingerprint=inspection.fingerprint,
        finetune_id=str(args.finetune_id),
        checkpoint_step=None if int(args.checkpoint_step) < 0 else int(args.checkpoint_step),
        selection_metric_name=result.selection_metric_name,
        selection_metric_value=result.selection_metric_value,
        metrics=result.metrics,
        artifact_paths=result.artifact_paths,
        source_provenance={"source": "md_train_framework", "config_path": str(Path(args.config).expanduser().resolve())},
        metadata={"replay_eval": True},
    )
    registry.upsert(record)
    print(json.dumps({"record_id": record.with_defaults().record_id, "selection_metric_value": result.selection_metric_value}, indent=2, sort_keys=True))
    return 0


def cmd_list_presets(args: argparse.Namespace) -> int:
    skill = str(args.skill or "").strip() or None
    payload = {
        "reward_presets": [preset.__dict__ for preset in list_reward_presets(skill=skill)],
        "reward_aliases": list_reward_aliases(),
        "backend_ids": backend_ids(skill=skill),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _resolve_registry(args: argparse.Namespace, *, config: Optional[FrameworkConfig]) -> RunRegistry:
    raw = str(getattr(args, "registry_path", "") or "").strip()
    if raw:
        return RunRegistry(Path(raw).expanduser().resolve())
    if config is None:
        return RunRegistry(Path("md_train_framework/outputs/runs.db").resolve())
    return RunRegistry(config.resolved_path(config.logging.registry_path))


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)


def _add_optional_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="")


def _ingest_local_checkpoints(
    registry: RunRegistry,
    *,
    config: FrameworkConfig,
    run_id: str,
    finetune_id: str,
    dataset_fingerprint: str,
    selection_metric_name: str,
    baseline_metrics: dict[str, Any],
    eval_history_path: Path,
    async_eval_dir: Path,
) -> None:
    if eval_history_path.exists():
        with eval_history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    continue
                metrics = dict(payload.get("metrics", {}) or {})
                registry.upsert(
                    RegistryRecord(
                        record_type="checkpoint",
                        run_id=run_id,
                        parent_run_id=run_id,
                        skill=config.skill.id,
                        task=config.task.name,
                        mode=config.mode,
                        backend_id=resolve_backend(config).id,
                        status="succeeded",
                        config_hash=config.config_hash,
                        dataset_fingerprint=dataset_fingerprint,
                        finetune_id=finetune_id,
                        checkpoint_step=int(payload.get("step", 0) or 0),
                        selection_metric_name=selection_metric_name,
                        selection_metric_value=_lookup_metric(metrics, selection_metric_name),
                        metrics=metrics,
                        baseline_metrics=baseline_metrics,
                        delta_metrics=metric_deltas(metrics, baseline_metrics),
                        artifact_paths={"eval_history_jsonl": str(eval_history_path)},
                        source_provenance={"source": "md_train_framework", "path": str(eval_history_path)},
                        metadata={"stage": payload.get("stage", ""), "split": payload.get("split", "")},
                    )
                )
    if async_eval_dir.exists():
        for metrics_path in async_eval_dir.rglob("metrics.json"):
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint_step = _extract_step(metrics_path.parent.name)
            registry.upsert(
                RegistryRecord(
                    record_type="checkpoint",
                    run_id=run_id,
                    parent_run_id=run_id,
                    skill=config.skill.id,
                    task=config.task.name,
                    mode=config.mode,
                    backend_id=resolve_backend(config).id,
                    status="succeeded_async_eval",
                    config_hash=config.config_hash,
                    dataset_fingerprint=dataset_fingerprint,
                    finetune_id=finetune_id,
                    checkpoint_step=checkpoint_step,
                    selection_metric_name=selection_metric_name,
                    selection_metric_value=_lookup_metric(payload, selection_metric_name),
                    metrics=payload,
                    baseline_metrics=baseline_metrics,
                    delta_metrics=metric_deltas(payload, baseline_metrics),
                    artifact_paths={"metrics_json": str(metrics_path)},
                    source_provenance={"source": "md_train_framework", "path": str(metrics_path)},
                    metadata={"async_eval": True},
                )
            )


def _lookup_metric(metrics: dict[str, Any], name: str) -> Optional[float]:
    candidates = [name]
    if name.startswith("eval_"):
        candidates.append(name[5:])
    else:
        candidates.append(f"eval_{name}")
    for candidate in candidates:
        try:
            return float(metrics[candidate])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _extract_step(name: str) -> Optional[int]:
    text = str(name or "").strip()
    if not text:
        return None
    for pattern in (
        r"(?:^|[^A-Za-z0-9])step0*([0-9]+)(?:[^0-9]|$)",
        r"^checkpoint[_-]?0*([0-9]+)$",
        r"^0*([0-9]+)$",
    ):
        match = re.search(pattern, f" {text}" if not pattern.startswith("^") else text)
        if match:
            return int(match.group(1))
    return None


def _append_resume_queue(config: FrameworkConfig, *, message: str) -> None:
    payload = {
        "logged_at": now_utc_iso(),
        "config_hash": config.config_hash,
        "task": config.task.name,
        "skill": config.skill.id,
        "message": message,
        "config_path": str(config.config_path or ""),
    }
    append_jsonl(config.resolved_path(config.recovery.resume_queue_path), payload)


def _read_resume_queue(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_resume_queue(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
