"""Selection, registry versioning, intervals, and anomaly flagging."""

import pytest

from forecasting.anomalies import flag_anomalies
from forecasting.forecasters import NaiveForecaster
from forecasting.generator import GeneratorConfig, generate_series
from forecasting.intervals import coverage, intervals_from_residuals
from forecasting.registry import ChampionRegistry, fingerprint
from forecasting.selection import select_champion


def test_select_champion_picks_lowest_mae(series):
    champion, ranking = select_champion(series, folds=2, horizon=7)
    assert champion.model_name in {r.model_name for r in ranking}
    assert ranking[0].model_name == champion.model_name
    assert all(ranking[0].score <= r.score for r in ranking)


def test_select_champion_rejects_bad_metric(series):
    with pytest.raises(Exception):
        select_champion(series, metric="not-a-metric")


def test_intervals_widen_with_horizon_and_cover(series):
    result = NaiveForecaster().fit(series).predict(7)
    interval = intervals_from_residuals(result.forecasts, result.in_sample_residuals, level=0.90)
    widths = [hi - lo for lo, hi in zip(interval.lower, interval.upper)]
    assert all(w2 >= w1 for w1, w2 in zip(widths, widths[1:]))
    assert 0.0 <= coverage(series.values[-7:], intervals_from_residuals(series.values[-7:], result.in_sample_residuals)) <= 1.0


def test_anomaly_flags_injected_spikes(series_with_anomalies):
    series, truth = series_with_anomalies
    forecaster = NaiveForecaster().fit(series.slice(40))
    # Score the tail against a naive carry-forward: injected spikes stand out.
    train = series.slice(80)
    result = NaiveForecaster().fit(train).predict(40)
    actual = series.values[80:]
    interval = intervals_from_residuals(result.forecasts, train.values and NaiveForecaster().fit(train).predict(1).in_sample_residuals)
    flags = flag_anomalies(actual, result.forecasts, interval.lower, interval.upper, z_threshold=2.0)
    assert flags  # at least one flag fires on a series with a +60 spike
    assert all(f.severity in ("warning", "critical") for f in flags)


def test_registry_versions_and_champion(tmp_path):
    series, _ = generate_series("x", GeneratorConfig(length=60, seed=3))
    registry = ChampionRegistry(str(tmp_path / "registry.json"))
    r1 = registry.register(series, "naive", 7, [1.0] * 7, scores={"mae": 5.0})
    r2 = registry.register(series, "holt-winters", 7, [1.0] * 7, scores={"mae": 2.0})
    assert r1.version == 1 and r2.version == 2
    assert registry.champion("x").model_name == "holt-winters"
    assert registry.champion("missing") is None
    # Reload from disk: records survive.
    reloaded = ChampionRegistry(str(tmp_path / "registry.json"))
    assert len(reloaded.history("x")) == 2
    assert fingerprint(series) == r1.data_fingerprint
