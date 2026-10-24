"""Model selection by backtest score, with an auditable ranking."""

from __future__ import annotations

from dataclasses import dataclass

from .backtest import BacktestResult, rolling_origin_backtest
from .errors import ValidationError
from .forecasters import BaseForecaster, all_forecasters
from .series import TimeSeries


@dataclass
class RankedModel:
    model_name: str
    metric: str
    score: float
    backtest: BacktestResult

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "metric": self.metric,
            "score": round(self.score, 4),
            "mean_scores": self.backtest.mean_scores.to_dict(),
            "n_folds": len(self.backtest.folds),
        }


def select_champion(
    series: TimeSeries,
    candidates: list[BaseForecaster] | None = None,
    folds: int = 3,
    horizon: int = 7,
    metric: str = "mae",
) -> tuple[RankedModel, list[RankedModel]]:
    """Backtest every candidate and return ``(champion, ranking)``.

    ``metric`` is one of mae/rmse/mape/smape; lower is always better. Models
    that fail a fold (e.g. too little data) are skipped, not silently ranked
    last — and if every model fails, that is an error, not an empty ranking.
    """
    if metric not in ("mae", "rmse", "mape", "smape"):
        raise ValidationError(f"unsupported selection metric {metric!r}")
    ranking: list[RankedModel] = []
    for candidate in candidates or all_forecasters():
        try:
            result = rolling_origin_backtest(
                series, lambda c=candidate: type(c)(), folds=folds, horizon=horizon
            )
        except Exception:
            continue
        ranking.append(RankedModel(candidate.name, metric, getattr(result.mean_scores, metric), result))
    if not ranking:
        raise ValidationError("no candidate forecaster produced a valid backtest")
    ranking.sort(key=lambda r: r.score)
    return ranking[0], ranking
