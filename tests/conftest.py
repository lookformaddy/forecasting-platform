"""Shared fixtures: deterministic synthetic series for every test module."""

import pytest

from forecasting.generator import GeneratorConfig, generate_series
from forecasting.series import TimeSeries


@pytest.fixture
def series():
    s, _ = generate_series("demand", GeneratorConfig(length=120, seed=7))
    return s


@pytest.fixture
def series_with_anomalies():
    s, truth = generate_series(
        "demand", GeneratorConfig(length=120, seed=7, anomaly_indices=[40, 80], anomaly_magnitude=60.0)
    )
    return s, truth


@pytest.fixture
def short_values():
    return [100.0, 102.0, 101.0, 105.0, 103.0, 107.0, 106.0, 110.0, 108.0, 112.0]
