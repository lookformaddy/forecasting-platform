"""Rolling-origin backtesting: the honest way to compare forecasters.

For each fold, train on an expanding window and score the next ``horizon``
points; fold scores are averaged per model. Nothing here peeks at the test
window during fitting — each fold refits a fresh forecaster instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import ValidationError
from .forecasters import BaseForecaster
from .metrics import MetricScores, compute_metrics
from .series import TimeSeries


@dataclass
class FoldResult:
    fold: int
    train_end: int
    scores: MetricScores


@dataclass
class BacktestResult:
    model_name: str
    folds: list[FoldResult]
    mean_scores: MetricScores

    def mean_metric(self, metric: str = "mae") -> float:
        return getattr(self.mean_scores, metric)


def _mean_scores(scores: list[MetricScores]) -> MetricScores:
    n = len(scores)
    return MetricScores(
        mae=sum(s.mae for s in scores) / n,
        rmse=sum(s.rmse for s in scores) / n,
        mape=sum(s.mape for s in scores) / n,
        smape=sum(s.smape for s in scores) / n,
        bias=sum(s.bias for s in scores) / n,
    )


def rolling_origin_backtest(
    series: TimeSeries,
    forecaster_factory: "callable[[], BaseForecaster]",
    folds: int = 3,
    horizon: int = 7,
    min_train: int = 28,
) -> BacktestResult:
    """Run rolling-origin evaluation for one forecaster factory."""
    if folds < 1 or horizon < 1:
        raise ValidationError("folds and horizon must be >= 1")
    total = len(series)
    # Origins step backwards from the end so the last fold ends at the series end.
    last_origin = total - horizon
    if last_origin < min_train:
        raise ValidationError(
            f"series of length {total} too short for min_train={min_train} and horizon={horizon}"
        )
    origins = [last_origin - (folds - 1 - i) * horizon for i in range(folds)]
    origins = [o for o in origins if o >= min_train]
    if not origins:
        raise ValidationError("no valid backtest origins for the given folds/horizon/min_train")
    name = ""
    fold_results: list[FoldResult] = []
    for i, origin in enumerate(origins):
        forecaster = forecaster_factory()
        name = forecaster.name
        train = series.slice(origin)
        forecaster.fit(train)
        forecast = forecaster.predict(horizon).forecasts
        actual = series.values[origin : origin + horizon]
        fold_results.append(FoldResult(fold=i, train_end=origin, scores=compute_metrics(actual, forecast)))
    return BacktestResult(model_name=name, folds=fold_results, mean_scores=_mean_scores([f.scores for f in fold_results]))
