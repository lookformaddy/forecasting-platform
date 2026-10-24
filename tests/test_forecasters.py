"""Every forecaster: shape, sanity, and error behaviour."""

import pytest

from forecasting.errors import InsufficientDataError, ModelNotFittedError, ValidationError
from forecasting.forecasters import (
    HoltWintersForecaster,
    LinearTrendForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    all_forecasters,
    get_forecaster,
)


def test_all_forecasters_produce_horizon(series):
    for forecaster in all_forecasters():
        result = forecaster.fit(series).predict(14)
        assert len(result.forecasts) == 14
        assert all(isinstance(v, float) for v in result.forecasts)
        assert len(result.in_sample_residuals) == len(series)


def test_naive_repeats_last_value(short_values):
    from forecasting.series import TimeSeries

    s = TimeSeries.daily("x", short_values, "2026-01-01")
    result = NaiveForecaster().fit(s).predict(3)
    assert result.forecasts == [short_values[-1]] * 3


def test_seasonal_naive_repeats_last_cycle(series):
    result = SeasonalNaiveForecaster().fit(series).predict(7)
    assert result.forecasts == series.values[-7:]


def test_holt_winters_tracks_trending_seasonal_series(series):
    result = HoltWintersForecaster().fit(series).predict(7)
    # Series trends upward, so the forecast should sit above the series mean's tail.
    assert result.forecasts[-1] > series.values[-1] - 40
    assert set(result.fitted_params) >= {"alpha", "beta", "gamma"}


def test_linear_trend_has_positive_slope_on_trending_series(series):
    result = LinearTrendForecaster().fit(series).predict(7)
    # coef_1 is the fitted trend slope; weekly seasonality can make any
    # single 7-step window non-monotonic, so assert on the slope itself.
    assert result.fitted_params["coef_1"] > 0


def test_predict_before_fit_raises():
    with pytest.raises(ModelNotFittedError):
        NaiveForecaster().predict(3)


def test_seasonal_naive_needs_a_cycle():
    from forecasting.series import TimeSeries

    s = TimeSeries.daily("x", [1.0, 2.0, 3.0], "2026-01-01", seasonal_period=7)
    with pytest.raises(InsufficientDataError):
        SeasonalNaiveForecaster().fit(s).predict(2)


def test_get_forecaster_unknown():
    with pytest.raises(ValidationError):
        get_forecaster("no-such-model")
    assert get_forecaster("naive").name == "naive"
