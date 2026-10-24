"""Forecasters implemented from scratch in pure standard library.

Models (all share the ``BaseForecaster`` interface):

- :class:`NaiveForecaster` — last value carried forward.
- :class:`SeasonalNaiveForecaster` — value from one seasonal cycle ago.
- :class:`HoltWintersForecaster` — additive Holt-Winters exponential
  smoothing (level + trend + seasonal), with a small grid search over
  smoothing parameters scored by in-sample one-step MAE.
- :class:`LinearTrendForecaster` — ordinary least squares on
  ``[1, t, sin, cos]`` (seasonal Fourier terms included when the series has
  a seasonal period > 1), solved by Gaussian elimination — no numpy.

``ForecastResult`` carries point forecasts, in-sample residuals (the raw
material for prediction intervals), and the fitted parameters, so every
forecast is auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .errors import InsufficientDataError, ModelNotFittedError, ValidationError
from .series import TimeSeries


@dataclass
class ForecastResult:
    model_name: str
    forecasts: list[float]
    fitted_params: dict[str, float] = field(default_factory=dict)
    in_sample_residuals: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.forecasts:
            raise ValidationError("forecast horizon must be >= 1")


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValidationError("design matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))) / aug[i][i]
    return out


def _ols(design: list[list[float]], target: list[float]) -> list[float]:
    """Ordinary least squares via the normal equations."""
    n_cols = len(design[0])
    normal = [[0.0] * n_cols for _ in range(n_cols)]
    rhs = [0.0] * n_cols
    for row, y in zip(design, target):
        for i in range(n_cols):
            rhs[i] += row[i] * y
            for j in range(n_cols):
                normal[i][j] += row[i] * row[j]
    # Tiny ridge term keeps near-collinear seasonal designs solvable.
    for i in range(n_cols):
        normal[i][i] += 1e-8
    return _solve_linear_system(normal, rhs)


class BaseForecaster:
    """Shared fit/predict contract for every forecaster."""

    name: str = "base"

    def __init__(self) -> None:
        self._series: TimeSeries | None = None

    def fit(self, series: TimeSeries) -> "BaseForecaster":
        if len(series) < 2:
            raise InsufficientDataError(f"{self.name} needs at least 2 observations")
        self._series = series
        return self

    def predict(self, horizon: int) -> ForecastResult:
        if self._series is None:
            raise ModelNotFittedError(f"{self.name} must be fitted before predict()")
        if horizon < 1:
            raise ValidationError("horizon must be >= 1")
        return self._predict(horizon)

    def _predict(self, horizon: int) -> ForecastResult:  # pragma: no cover - subclasses
        raise NotImplementedError

    def _residuals(self, fitted: list[float]) -> list[float]:
        assert self._series is not None
        return [a - f for a, f in zip(self._series.values, fitted)]


class NaiveForecaster(BaseForecaster):
    name = "naive"

    def _predict(self, horizon: int) -> ForecastResult:
        assert self._series is not None
        last = self._series.values[-1]
        fitted = [self._series.values[0]] + self._series.values[:-1]
        return ForecastResult(self.name, [last] * horizon, {}, self._residuals(fitted))


class SeasonalNaiveForecaster(BaseForecaster):
    name = "seasonal-naive"

    def _predict(self, horizon: int) -> ForecastResult:
        assert self._series is not None
        period = self._series.seasonal_period
        if len(self._series) <= period:
            raise InsufficientDataError("seasonal-naive needs more than one seasonal cycle")
        values = self._series.values
        forecasts = [values[-period + (i % period)] if i < period else values[len(values) - period + (i % period)] for i in range(horizon)]
        # The expression above reduces to the last cycle repeated; keep it explicit:
        forecasts = [values[len(values) - period + (i % period)] for i in range(horizon)]
        fitted = [values[i - period] if i >= period else values[0] for i in range(len(values))]
        return ForecastResult(self.name, forecasts, {"seasonal_period": float(period)}, self._residuals(fitted))


def _holt_winters_smooth(values: list[float], period: int, alpha: float, beta: float, gamma: float) -> tuple[list[float], float, float, list[float]]:
    """One Holt-Winters pass; returns (one-step fitted, level, trend, seasonal)."""
    if len(values) < 2 * period:
        # Degenerate initialisation for short series: flat seasonal profile.
        level = values[0]
        trend = (values[-1] - values[0]) / max(1, len(values) - 1)
        seasonal = [0.0] * period
    else:
        level = sum(values[:period]) / period
        trend = (sum(values[period : 2 * period]) / period - level) / period
        seasonal = [values[i] - level for i in range(period)]
    fitted: list[float] = []
    for t, y in enumerate(values):
        if t == 0:
            fitted.append(level + seasonal[0])
            continue
        season_idx = t % period
        prev_level, prev_trend = level, trend
        forecast = prev_level + prev_trend + seasonal[season_idx]
        fitted.append(forecast)
        level = alpha * (y - seasonal[season_idx]) + (1 - alpha) * (prev_level + prev_trend)
        trend = beta * (level - prev_level) + (1 - beta) * prev_trend
        seasonal[season_idx] = gamma * (y - level) + (1 - gamma) * seasonal[season_idx]
    return fitted, level, trend, seasonal


class HoltWintersForecaster(BaseForecaster):
    name = "holt-winters"

    def __init__(self, grid: tuple[float, ...] = (0.1, 0.3, 0.6)) -> None:
        super().__init__()
        self.grid = grid

    def _predict(self, horizon: int) -> ForecastResult:
        assert self._series is not None
        values, period = self._series.values, self._series.seasonal_period
        if len(values) <= period:
            raise InsufficientDataError("holt-winters needs more than one seasonal cycle")
        best: tuple[float, tuple[float, float, float], list[float]] | None = None
        for alpha in self.grid:
            for beta in self.grid:
                for gamma in self.grid:
                    fitted, _, _, _ = _holt_winters_smooth(values, period, alpha, beta, gamma)
                    mae = sum(abs(a - f) for a, f in zip(values[1:], fitted[1:])) / max(1, len(values) - 1)
                    if best is None or mae < best[0]:
                        best = (mae, (alpha, beta, gamma), fitted)
        assert best is not None
        _, (alpha, beta, gamma), fitted = best
        _, level, trend, seasonal = _holt_winters_smooth(values, period, alpha, beta, gamma)
        n = len(values)
        forecasts = [level + (h + 1) * trend + seasonal[(n + h) % period] for h in range(horizon)]
        params = {"alpha": alpha, "beta": beta, "gamma": gamma, "in_sample_mae": best[0]}
        return ForecastResult(self.name, forecasts, params, self._residuals(fitted))


class LinearTrendForecaster(BaseForecaster):
    name = "linear-trend"

    def _design_row(self, t: int, period: int) -> list[float]:
        row = [1.0, float(t)]
        if period > 1:
            row += [math.sin(2.0 * math.pi * t / period), math.cos(2.0 * math.pi * t / period)]
        return row

    def _predict(self, horizon: int) -> ForecastResult:
        assert self._series is not None
        values, period = self._series.values, self._series.seasonal_period
        design = [self._design_row(t, period) for t in range(len(values))]
        coeffs = _ols(design, values)
        fitted = [sum(c * x for c, x in zip(coeffs, row)) for row in design]
        forecasts = [
            sum(c * x for c, x in zip(coeffs, self._design_row(len(values) + h, period)))
            for h in range(horizon)
        ]
        params = {f"coef_{i}": c for i, c in enumerate(coeffs)}
        return ForecastResult(self.name, forecasts, params, self._residuals(fitted))


def all_forecasters() -> list[BaseForecaster]:
    """Every shipped forecaster, in baseline-first order."""
    return [NaiveForecaster(), SeasonalNaiveForecaster(), HoltWintersForecaster(), LinearTrendForecaster()]


def get_forecaster(name: str) -> BaseForecaster:
    """Look a forecaster up by name; raises ValidationError if unknown."""
    for candidate in all_forecasters():
        if candidate.name == name:
            return candidate
    known = ", ".join(f.name for f in all_forecasters())
    raise ValidationError(f"unknown forecaster {name!r}; known: {known}")
