"""Generator determinism, validation, and the series model."""

import pytest

from forecasting.errors import ValidationError
from forecasting.generator import GeneratorConfig, generate_series
from forecasting.series import TimeSeries


def test_generate_is_deterministic():
    cfg = GeneratorConfig(length=60, seed=123)
    a, _ = generate_series("x", cfg)
    b, _ = generate_series("x", cfg)
    assert a.values == b.values
    assert a.timestamps[0] == "2026-01-01"
    assert len(a.timestamps) == 60


def test_generate_reports_anomaly_indices():
    _, truth = generate_series("x", GeneratorConfig(length=60, seed=1, anomaly_indices=[5, 20]))
    assert truth == [5, 20]


def test_generate_rejects_short_length():
    with pytest.raises(ValidationError):
        generate_series("x", GeneratorConfig(length=5))


def test_series_rejects_mismatched_timestamps():
    with pytest.raises(ValidationError):
        TimeSeries(name="x", values=[1.0, 2.0], timestamps=["2026-01-01"])


def test_series_rejects_nonfinite():
    with pytest.raises(ValidationError):
        TimeSeries(name="x", values=[1.0, float("nan")])


def test_series_slice_and_labels(series):
    prefix = series.slice(50)
    assert len(prefix) == 50
    labels = series.extend_labels(3)
    assert len(labels) == 3 and labels[0] > series.timestamps[-1]
    with pytest.raises(ValidationError):
        series.slice(1)
