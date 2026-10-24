"""Forecast accuracy metrics (pure stdlib, all defined on actual/forecast pairs)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ValidationError


@dataclass
class MetricScores:
    mae: float
    rmse: float
    mape: float  # percent; NaN-safe: zero actuals are skipped
    smape: float  # percent
    bias: float  # mean signed error (forecast - actual)

    def to_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "mape": self.mape, "smape": self.smape, "bias": self.bias}


def compute_metrics(actual: list[float], forecast: list[float]) -> MetricScores:
    """Compute MAE/RMSE/MAPE/sMAPE/bias; lengths must match and be non-empty."""
    if not actual or len(actual) != len(forecast):
        raise ValidationError("actual and forecast must be non-empty and the same length")
    n = len(actual)
    abs_err = [abs(a - f) for a, f in zip(actual, forecast)]
    mae = sum(abs_err) / n
    rmse = math.sqrt(sum(e * e for e in abs_err) / n)
    mape_terms = [abs(a - f) / abs(a) for a, f in zip(actual, forecast) if a != 0]
    mape = 100.0 * sum(mape_terms) / len(mape_terms) if mape_terms else 0.0
    smape = 100.0 * sum(
        (2.0 * abs(a - f) / (abs(a) + abs(f))) if (abs(a) + abs(f)) > 0 else 0.0
        for a, f in zip(actual, forecast)
    ) / n
    bias = sum(f - a for a, f in zip(actual, forecast)) / n
    return MetricScores(mae=mae, rmse=rmse, mape=mape, smape=smape, bias=bias)


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already-sorted list."""
    if not sorted_values:
        raise ValidationError("quantile requires non-empty values")
    if not 0.0 <= q <= 1.0:
        raise ValidationError("quantile q must be in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
