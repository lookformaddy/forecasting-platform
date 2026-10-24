"""Hierarchical reconciliation coherence + rolling-origin backtesting."""

import pytest

from forecasting.backtest import rolling_origin_backtest
from forecasting.forecasters import HoltWintersForecaster, NaiveForecaster
from forecasting.generator import GeneratorConfig, generate_series
from forecasting.hierarchy import HierarchyNode, bottom_up, reconciled, top_down
from forecasting.metrics import compute_metrics
from forecasting.series import TimeSeries


def _hierarchy():
    a, _ = generate_series("region-a", GeneratorConfig(length=90, seed=11, base_level=60.0))
    b, _ = generate_series("region-b", GeneratorConfig(length=90, seed=22, base_level=40.0))
    total_values = [x + y for x, y in zip(a.values, b.values)]
    total = TimeSeries(name="total", values=total_values, timestamps=a.timestamps, seasonal_period=7)
    return HierarchyNode(series=total, children=[HierarchyNode(a, []), HierarchyNode(b, [])])


def test_bottom_up_children_sum_to_parent():
    out = bottom_up(_hierarchy(), NaiveForecaster, 7)
    for h in range(7):
        assert out["total"][h] == pytest.approx(out["region-a"][h] + out["region-b"][h])


def test_top_down_children_sum_to_parent():
    out = top_down(_hierarchy(), NaiveForecaster, 7)
    for h in range(7):
        assert out["total"][h] == pytest.approx(out["region-a"][h] + out["region-b"][h])


def test_reconciled_children_sum_to_parent():
    out = reconciled(_hierarchy(), HoltWintersForecaster, 7)
    for h in range(7):
        assert out["total"][h] == pytest.approx(out["region-a"][h] + out["region-b"][h])


def test_backtest_returns_folds_and_mean(series):
    result = rolling_origin_backtest(series, NaiveForecaster, folds=3, horizon=7)
    assert len(result.folds) == 3
    assert result.mean_scores.mae > 0
    assert result.mean_scores.mae == pytest.approx(
        sum(f.scores.mae for f in result.folds) / 3
    )


def test_backtest_rejects_too_short_series():
    from forecasting.series import TimeSeries

    s = TimeSeries.daily("x", [1.0] * 20, "2026-01-01")
    with pytest.raises(Exception):
        rolling_origin_backtest(s, NaiveForecaster, folds=3, horizon=7, min_train=28)


def test_metrics_known_values():
    scores = compute_metrics([100.0, 100.0], [110.0, 90.0])
    assert scores.mae == pytest.approx(10.0)
    assert scores.bias == pytest.approx(0.0)
    assert scores.mape == pytest.approx(10.0)
