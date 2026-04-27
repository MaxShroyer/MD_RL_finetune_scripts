from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from md_train_framework.metrics import metric_deltas
from md_train_framework.utils import now_utc_iso, stable_json_hash

_SQLITE_INT_MIN = -(2**63)
_SQLITE_INT_MAX = 2**63 - 1


@dataclass(frozen=True)
class RegistryRecord:
    record_type: str
    run_id: str
    skill: str
    task: str
    mode: str
    backend_id: str
    status: str
    config_hash: str
    dataset_fingerprint: str
    finetune_id: str = ""
    checkpoint_step: Optional[int] = None
    selection_metric_name: str = ""
    selection_metric_value: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    delta_metrics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    source_provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_run_id: str = ""
    record_id: str = ""
    created_at: str = ""

    def with_defaults(self) -> "RegistryRecord":
        record_id = self.record_id or stable_json_hash(
            {
                "record_type": self.record_type,
                "run_id": self.run_id,
                "checkpoint_step": self.checkpoint_step,
                "selection_metric_name": self.selection_metric_name,
                "selection_metric_value": self.selection_metric_value,
                "artifact_paths": self.artifact_paths,
            }
        )
        created_at = self.created_at or now_utc_iso()
        return RegistryRecord(
            **{
                **self.__dict__,
                "record_id": record_id,
                "created_at": created_at,
            }
        )

    def to_row(self) -> tuple[Any, ...]:
        record = self.with_defaults()
        return (
            record.record_id,
            record.record_type,
            record.run_id,
            record.parent_run_id,
            record.skill,
            record.task,
            record.mode,
            record.backend_id,
            record.status,
            record.config_hash,
            record.dataset_fingerprint,
            record.finetune_id,
            _sqlite_int(record.checkpoint_step),
            record.selection_metric_name,
            record.selection_metric_value,
            json.dumps(record.metrics, sort_keys=True),
            json.dumps(record.baseline_metrics, sort_keys=True),
            json.dumps(record.delta_metrics, sort_keys=True),
            json.dumps(record.artifact_paths, sort_keys=True),
            json.dumps(record.source_provenance, sort_keys=True),
            json.dumps(record.metadata, sort_keys=True),
            record.created_at,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RegistryRecord":
        return cls(
            record_id=row["record_id"],
            record_type=row["record_type"],
            run_id=row["run_id"],
            parent_run_id=row["parent_run_id"],
            skill=row["skill"],
            task=row["task"],
            mode=row["mode"],
            backend_id=row["backend_id"],
            status=row["status"],
            config_hash=row["config_hash"],
            dataset_fingerprint=row["dataset_fingerprint"],
            finetune_id=row["finetune_id"],
            checkpoint_step=_sqlite_int(row["checkpoint_step"]),
            selection_metric_name=row["selection_metric_name"],
            selection_metric_value=row["selection_metric_value"],
            metrics=json.loads(row["metrics_json"] or "{}"),
            baseline_metrics=json.loads(row["baseline_metrics_json"] or "{}"),
            delta_metrics=json.loads(row["delta_json"] or "{}"),
            artifact_paths=json.loads(row["artifact_paths_json"] or "{}"),
            source_provenance=json.loads(row["source_provenance_json"] or "{}"),
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"],
        )


def _sqlite_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return max(_SQLITE_INT_MIN, min(_SQLITE_INT_MAX, parsed))


class RunRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists records (
                    record_id text primary key,
                    record_type text not null,
                    run_id text not null,
                    parent_run_id text not null,
                    skill text not null,
                    task text not null,
                    mode text not null,
                    backend_id text not null,
                    status text not null,
                    config_hash text not null,
                    dataset_fingerprint text not null,
                    finetune_id text not null,
                    checkpoint_step integer,
                    selection_metric_name text not null,
                    selection_metric_value real,
                    metrics_json text not null,
                    baseline_metrics_json text not null,
                    delta_json text not null,
                    artifact_paths_json text not null,
                    source_provenance_json text not null,
                    metadata_json text not null,
                    created_at text not null
                )
                """
            )
            conn.execute(
                "create index if not exists idx_records_lookup on records "
                "(record_type, skill, task, dataset_fingerprint, selection_metric_value desc)"
            )

    def upsert(self, record: RegistryRecord) -> RegistryRecord:
        resolved = record.with_defaults()
        with self._connect() as conn:
            conn.execute(
                """
                insert into records (
                    record_id, record_type, run_id, parent_run_id, skill, task, mode, backend_id, status,
                    config_hash, dataset_fingerprint, finetune_id, checkpoint_step, selection_metric_name,
                    selection_metric_value, metrics_json, baseline_metrics_json, delta_json, artifact_paths_json,
                    source_provenance_json, metadata_json, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(record_id) do update set
                    record_type=excluded.record_type,
                    run_id=excluded.run_id,
                    parent_run_id=excluded.parent_run_id,
                    skill=excluded.skill,
                    task=excluded.task,
                    mode=excluded.mode,
                    backend_id=excluded.backend_id,
                    status=excluded.status,
                    config_hash=excluded.config_hash,
                    dataset_fingerprint=excluded.dataset_fingerprint,
                    finetune_id=excluded.finetune_id,
                    checkpoint_step=excluded.checkpoint_step,
                    selection_metric_name=excluded.selection_metric_name,
                    selection_metric_value=excluded.selection_metric_value,
                    metrics_json=excluded.metrics_json,
                    baseline_metrics_json=excluded.baseline_metrics_json,
                    delta_json=excluded.delta_json,
                    artifact_paths_json=excluded.artifact_paths_json,
                    source_provenance_json=excluded.source_provenance_json,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at
                """,
                resolved.to_row(),
            )
        return resolved

    def list_records(
        self,
        *,
        record_type: Optional[str] = None,
        skill: Optional[str] = None,
        task: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[RegistryRecord]:
        query = "select * from records where 1=1"
        params: list[Any] = []
        if record_type:
            query += " and record_type = ?"
            params.append(record_type)
        if skill:
            query += " and skill = ?"
            params.append(skill)
        if task:
            query += " and task = ?"
            params.append(task)
        if dataset_fingerprint:
            query += " and dataset_fingerprint = ?"
            params.append(dataset_fingerprint)
        if run_id:
            query += " and run_id = ?"
            params.append(run_id)
        query += " order by (selection_metric_value is null), selection_metric_value desc, created_at desc"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [RegistryRecord.from_row(row) for row in rows]

    def find_best_record(self, *, run_id: str) -> Optional[RegistryRecord]:
        records = self.list_records(run_id=run_id)
        if not records:
            return None
        return max(records, key=lambda item: float(item.selection_metric_value or float("-inf")))

    def latest_baseline(self, *, skill: str, task: str, dataset_fingerprint: str) -> Optional[RegistryRecord]:
        records = self.list_records(
            record_type="baseline",
            skill=skill,
            task=task,
            dataset_fingerprint=dataset_fingerprint,
        )
        return records[0] if records else None

    def leaderboard(
        self,
        *,
        skill: Optional[str] = None,
        task: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        limit: int = 10,
    ) -> list[RegistryRecord]:
        records = self.list_records(skill=skill, task=task, dataset_fingerprint=dataset_fingerprint)
        eligible = [record for record in records if record.record_type in {"run", "checkpoint"}]
        ranked = sorted(
            eligible,
            key=lambda item: (
                float(item.selection_metric_value or float("-inf")),
                item.checkpoint_step or -1,
                item.created_at,
            ),
            reverse=True,
        )
        return ranked[: max(1, int(limit))]

    def compare(self, *, left_id: str, right_id: str) -> dict[str, Any]:
        left = self._resolve_record_or_run(left_id)
        right = self._resolve_record_or_run(right_id)
        if left is None or right is None:
            raise ValueError("compare requires existing run_id or record_id values")
        return {
            "left": left.__dict__,
            "right": right.__dict__,
            "metric_delta": metric_deltas(left.metrics, right.metrics),
            "selection_metric_delta": (
                float(left.selection_metric_value or 0.0) - float(right.selection_metric_value or 0.0)
            ),
        }

    def export_jsonl(self, output_path: str | Path) -> Path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.list_records():
                handle.write(json.dumps(record.__dict__, ensure_ascii=True, sort_keys=True))
                handle.write("\n")
        return path

    def seen_config(self, *, config_hash: str, dataset_fingerprint: str) -> bool:
        query = "select 1 from records where config_hash = ? and dataset_fingerprint = ? limit 1"
        with self._connect() as conn:
            row = conn.execute(query, [config_hash, dataset_fingerprint]).fetchone()
        return row is not None

    def _resolve_record_or_run(self, identifier: str) -> Optional[RegistryRecord]:
        with self._connect() as conn:
            row = conn.execute("select * from records where record_id = ? limit 1", [identifier]).fetchone()
        if row is not None:
            return RegistryRecord.from_row(row)
        return self.find_best_record(run_id=identifier)


def leaderboard_rows(records: Iterable[RegistryRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rows.append(
            {
                "rank": index,
                "run_id": record.run_id,
                "task": record.task,
                "skill": record.skill,
                "selection_metric": record.selection_metric_name,
                "selection_metric_value": record.selection_metric_value,
                "checkpoint_step": record.checkpoint_step,
                "promotion": "champion" if index == 1 else "challenger",
                "artifact_paths": record.artifact_paths,
            }
        )
    return rows
