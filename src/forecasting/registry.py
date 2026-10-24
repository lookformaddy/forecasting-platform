"""Champion registry + forecast versioning and audit.

A forecast is versioned the moment it is produced: a record carries the
series name, model, parameters, backtest scores, a SHA-256 fingerprint of
the training data, and a UTC timestamp. Records are appended to a JSON
file atomically (write-then-rename), so a crash never leaves a torn file.
The champion for a series is the record with the best (lowest) score for
its selection metric.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .errors import RegistryError
from .series import TimeSeries


@dataclass
class ForecastRecord:
    series_name: str
    model_name: str
    version: int
    horizon: int
    forecasts: list[float]
    params: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    data_fingerprint: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "series_name": self.series_name,
            "model_name": self.model_name,
            "version": self.version,
            "horizon": self.horizon,
            "forecasts": self.forecasts,
            "params": self.params,
            "scores": self.scores,
            "data_fingerprint": self.data_fingerprint,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ForecastRecord":
        return cls(
            series_name=data["series_name"],
            model_name=data["model_name"],
            version=int(data["version"]),
            horizon=int(data["horizon"]),
            forecasts=[float(v) for v in data["forecasts"]],
            params={k: float(v) for k, v in data.get("params", {}).items()},
            scores={k: float(v) for k, v in data.get("scores", {}).items()},
            data_fingerprint=data.get("data_fingerprint", ""),
            created_at=data.get("created_at", ""),
        )


def fingerprint(series: TimeSeries) -> str:
    """SHA-256 over the series values (reproducibility audit trail)."""
    payload = ",".join(f"{v:.6f}" for v in series.values).encode()
    return hashlib.sha256(payload).hexdigest()


class ChampionRegistry:
    """JSON-backed registry of forecast records, one file per deployment."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._records: list[ForecastRecord] = self._load()

    def _load(self) -> list[ForecastRecord]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
            return [ForecastRecord.from_dict(item) for item in raw]
        except (OSError, ValueError, KeyError) as exc:
            raise RegistryError(f"could not load registry at {self.path}: {exc}") from exc

    def _save(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump([r.to_dict() for r in self._records], fh, indent=2)
        os.replace(tmp, self.path)

    def register(
        self,
        series: TimeSeries,
        model_name: str,
        horizon: int,
        forecasts: list[float],
        params: dict[str, float] | None = None,
        scores: dict[str, float] | None = None,
    ) -> ForecastRecord:
        """Append a new versioned record and persist it."""
        same = [r for r in self._records if r.series_name == series.name]
        record = ForecastRecord(
            series_name=series.name,
            model_name=model_name,
            version=max((r.version for r in same), default=0) + 1,
            horizon=horizon,
            forecasts=list(forecasts),
            params=params or {},
            scores=scores or {},
            data_fingerprint=fingerprint(series),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._records.append(record)
        self._save()
        return record

    def history(self, series_name: str) -> list[ForecastRecord]:
        return [r for r in self._records if r.series_name == series_name]

    def champion(self, series_name: str, metric: str = "mae") -> ForecastRecord | None:
        """Best-scoring record for a series, or None if no record has that metric."""
        scored = [r for r in self.history(series_name) if metric in r.scores]
        if not scored:
            return None
        return min(scored, key=lambda r: r.scores[metric])
