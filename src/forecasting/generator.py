"""Deterministic synthetic series generator (clean-room, no real data).

A series is trend + seasonality + event effects + Gaussian noise, with
optionally injected point anomalies. The same seed always yields the same
series, so demos, tests, and backtests are reproducible.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .errors import ValidationError
from .series import TimeSeries


@dataclass
class GeneratorConfig:
    """Parameters of one synthetic series."""

    length: int = 120
    base_level: float = 100.0
    trend_per_step: float = 0.25
    seasonal_amplitude: float = 12.0
    seasonal_period: int = 7
    noise_std: float = 3.0
    seed: int = 42
    # index -> additive effect, e.g. a promotion or holiday spike
    events: dict[int, float] = field(default_factory=dict)
    # indices where a large spike is injected (ground-truth anomalies)
    anomaly_indices: list[int] = field(default_factory=list)
    anomaly_magnitude: float = 45.0
    start_date: str = "2026-01-01"


def generate_series(name: str, cfg: GeneratorConfig) -> tuple[TimeSeries, list[int]]:
    """Generate a synthetic series; returns ``(series, anomaly_indices)``."""
    if cfg.length < 14:
        raise ValidationError("generator length must be >= 14")
    if cfg.seasonal_period < 1 or cfg.noise_std < 0:
        raise ValidationError("seasonal_period >= 1 and noise_std >= 0 required")
    rng = random.Random(cfg.seed)
    values: list[float] = []
    for t in range(cfg.length):
        seasonal = cfg.seasonal_amplitude * math.sin(2.0 * math.pi * t / cfg.seasonal_period)
        # A second, weaker harmonic keeps the seasonal shape non-trivial.
        seasonal += 0.3 * cfg.seasonal_amplitude * math.sin(4.0 * math.pi * t / cfg.seasonal_period)
        value = cfg.base_level + cfg.trend_per_step * t + seasonal
        value += cfg.events.get(t, 0.0)
        value += rng.gauss(0.0, cfg.noise_std)
        if t in cfg.anomaly_indices:
            value += cfg.anomaly_magnitude
        values.append(round(value, 4))
    series = TimeSeries.daily(name, values, cfg.start_date, seasonal_period=cfg.seasonal_period)
    return series, sorted(set(cfg.anomaly_indices))
