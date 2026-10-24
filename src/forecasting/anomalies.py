"""Anomaly flagging: actuals vs forecast, two complementary signals.

1. **Interval breach** — an actual outside its prediction interval.
2. **Residual z-score** — ``|residual| / residual_std`` above a threshold.

A point flagged by either signal is reported, with the signal(s) named, so
an on-call engineer can tell a genuine spike from a merely wide interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ValidationError


@dataclass
class AnomalyFlag:
    index: int
    timestamp: str | None
    actual: float
    forecast: float
    residual: float
    z_score: float
    signals: list[str]
    severity: str  # "warning" or "critical"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "actual": self.actual,
            "forecast": self.forecast,
            "residual": self.residual,
            "z_score": round(self.z_score, 3),
            "signals": self.signals,
            "severity": self.severity,
        }


def flag_anomalies(
    actual: list[float],
    forecast: list[float],
    lower: list[float] | None = None,
    upper: list[float] | None = None,
    timestamps: list[str] | None = None,
    z_threshold: float = 3.0,
) -> list[AnomalyFlag]:
    """Flag anomalous points; returns flags in chronological order."""
    if not actual or len(actual) != len(forecast):
        raise ValidationError("actual and forecast must be non-empty and the same length")
    if (lower is None) != (upper is None) or (lower is not None and len(lower) != len(actual)):
        raise ValidationError("lower/upper intervals must both be given with matching length")
    residuals = [a - f for a, f in zip(actual, forecast)]
    mean = sum(residuals) / len(residuals)
    std = math.sqrt(sum((r - mean) ** 2 for r in residuals) / max(1, len(residuals) - 1)) if len(residuals) > 1 else 0.0
    flags: list[AnomalyFlag] = []
    for i, (a, f, r) in enumerate(zip(actual, forecast, residuals)):
        signals: list[str] = []
        if lower is not None and upper is not None and not (lower[i] <= a <= upper[i]):
            signals.append("interval_breach")
        z = abs(r - mean) / std if std > 0 else 0.0
        if z >= z_threshold:
            signals.append("z_score")
        if signals:
            severity = "critical" if (len(signals) == 2 or z >= 2 * z_threshold) else "warning"
            flags.append(
                AnomalyFlag(
                    index=i,
                    timestamp=timestamps[i] if timestamps and i < len(timestamps) else None,
                    actual=a,
                    forecast=f,
                    residual=r,
                    z_score=z,
                    signals=signals,
                    severity=severity,
                )
            )
    return flags
