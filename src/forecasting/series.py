"""The core data model: a single univariate time series.

Series are deliberately plain lists of floats plus ISO date labels. No
numpy/pandas dependency is introduced anywhere in the core, so the library
runs anywhere Python runs and every transformation is auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from .errors import ValidationError


@dataclass
class TimeSeries:
    """One univariate series.

    ``values`` and ``timestamps`` must be the same length. ``seasonal_period``
    is the number of observations per seasonal cycle (7 = weekly seasonality
    on daily data, 12 = yearly seasonality on monthly data).
    """

    name: str
    values: list[float]
    timestamps: list[str] = field(default_factory=list)
    seasonal_period: int = 7

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("series name must not be empty")
        if len(self.values) < 2:
            raise ValidationError(f"series '{self.name}' needs at least 2 observations")
        if any(v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) for v in self.values):
            raise ValidationError(f"series '{self.name}' contains missing/non-finite values")
        if self.timestamps and len(self.timestamps) != len(self.values):
            raise ValidationError("timestamps length must match values length")
        if self.seasonal_period < 1:
            raise ValidationError("seasonal_period must be >= 1")
        self.values = [float(v) for v in self.values]

    def __len__(self) -> int:
        return len(self.values)

    @classmethod
    def daily(cls, name: str, values: list[float], start: str, seasonal_period: int = 7) -> "TimeSeries":
        """Build a daily series with ISO timestamps generated from ``start``."""
        try:
            start_date = date.fromisoformat(start)
        except ValueError as exc:
            raise ValidationError(f"invalid start date {start!r}") from exc
        stamps = [(start_date + timedelta(days=i)).isoformat() for i in range(len(values))]
        return cls(name=name, values=values, timestamps=stamps, seasonal_period=seasonal_period)

    def slice(self, end: int) -> "TimeSeries":
        """Return the prefix ``values[:end]`` as a new series."""
        if end < 2 or end > len(self.values):
            raise ValidationError(f"slice end {end} out of range for length {len(self.values)}")
        return TimeSeries(
            name=self.name,
            values=self.values[:end],
            timestamps=self.timestamps[:end] if self.timestamps else [],
            seasonal_period=self.seasonal_period,
        )

    def extend_labels(self, horizon: int) -> list[str]:
        """Future timestamps for ``horizon`` steps past the last observation."""
        if not self.timestamps:
            return [f"t+{i + 1}" for i in range(horizon)]
        last = date.fromisoformat(self.timestamps[-1])
        return [(last + timedelta(days=i + 1)).isoformat() for i in range(horizon)]

    def mean(self) -> float:
        return sum(self.values) / len(self.values)
