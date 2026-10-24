"""Prediction intervals from in-sample residual quantiles.

Method (transparent and honest about its limits): take the forecaster's
in-sample residuals, read the ``(1-level)/2`` and ``(1+level)/2`` quantiles,
and widen them with ``sqrt(step)`` so distant horizons are more uncertain.
Residuals are assumed roughly stationary — the docstring and README say so,
and :func:`coverage` lets users check that assumption on held-out data.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ValidationError
from .metrics import quantile
import math


@dataclass
class PredictionInterval:
    lower: list[float]
    upper: list[float]
    level: float

    def contains(self, actual: list[float]) -> list[bool]:
        if len(actual) != len(self.lower):
            raise ValidationError("actual length must match interval length")
        return [lo <= a <= hi for a, lo, hi in zip(actual, self.lower, self.upper)]


def intervals_from_residuals(
    forecasts: list[float], residuals: list[float], level: float = 0.90
) -> PredictionInterval:
    """Build intervals around ``forecasts`` from residual quantiles."""
    if not forecasts:
        raise ValidationError("forecasts must be non-empty")
    if len(residuals) < 10:
        raise ValidationError("need at least 10 residuals to build intervals")
    if not 0.0 < level < 1.0:
        raise ValidationError("level must be in (0, 1)")
    ordered = sorted(residuals)
    lo_q = quantile(ordered, (1.0 - level) / 2.0)
    hi_q = quantile(ordered, (1.0 + level) / 2.0)
    lower, upper = [], []
    for h, point in enumerate(forecasts):
        scale = math.sqrt(h + 1)
        lower.append(point + lo_q * scale)
        upper.append(point + hi_q * scale)
    return PredictionInterval(lower=lower, upper=upper, level=level)


def coverage(actual: list[float], interval: PredictionInterval) -> float:
    """Fraction of actuals inside the interval (0..1)."""
    flags = interval.contains(actual)
    return sum(flags) / len(flags) if flags else 0.0
