from __future__ import annotations

from typing import Any


class _WandbRunShim:
    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}

    def finish(self) -> None:
        return


class _WandbShim:
    def init(self, *args: Any, **kwargs: Any) -> _WandbRunShim:
        return _WandbRunShim()

    def log(self, *args: Any, **kwargs: Any) -> None:
        return


try:
    import wandb as _wandb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _wandb = _WandbShim()
else:  # pragma: no cover
    if not callable(getattr(_wandb, "init", None)) or not callable(getattr(_wandb, "log", None)):
        _wandb = _WandbShim()


class WandbLogger:
    def __init__(self, *, enabled: bool, project: str, run_name: str, config_payload: dict[str, Any]) -> None:
        self.enabled = bool(enabled)
        if self.enabled:
            try:
                self._run = _wandb.init(project=project, name=run_name or None, config=config_payload)
            except Exception:
                self.enabled = False
                self._run = _WandbRunShim()
        else:
            self._run = _WandbRunShim()

    @property
    def summary(self) -> dict[str, Any]:
        return self._run.summary

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        if not self.enabled:
            return
        try:
            _wandb.log(dict(payload), step=int(step))
        except Exception:
            self.enabled = False

    def finish(self) -> None:
        try:
            self._run.finish()
        except Exception:
            return
