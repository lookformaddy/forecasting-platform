"""forecasting: a stdlib-only time-series forecasting platform.

Public surface: series + generator, forecasters, hierarchy, backtesting,
metrics, intervals, anomaly flagging, champion selection, and registry.
"""

from .anomalies import AnomalyFlag, flag_anomalies
from .backtest import BacktestResult, rolling_origin_backtest
from .config import Settings
from .errors import ForecastError, InsufficientDataError, ModelNotFittedError, RegistryError, ValidationError
from .forecasters import (
    BaseForecaster,
    ForecastResult,
    HoltWintersForecaster,
    LinearTrendForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    all_forecasters,
    get_forecaster,
)
from .generator import GeneratorConfig, generate_series
from .hierarchy import HierarchyNode, bottom_up, reconciled, top_down
from .intervals import PredictionInterval, coverage, intervals_from_residuals
from .metrics import MetricScores, compute_metrics
from .registry import ChampionRegistry, ForecastRecord, fingerprint
from .selection import RankedModel, select_champion
from .series import TimeSeries

__all__ = [
    "AnomalyFlag",
    "BacktestResult",
    "BaseForecaster",
    "ChampionRegistry",
    "ForecastRecord",
    "ForecastResult",
    "ForecastError",
    "GeneratorConfig",
    "HierarchyNode",
    "HoltWintersForecaster",
    "InsufficientDataError",
    "LinearTrendForecaster",
    "MetricScores",
    "ModelNotFittedError",
    "NaiveForecaster",
    "PredictionInterval",
    "RankedModel",
    "RegistryError",
    "SeasonalNaiveForecaster",
    "Settings",
    "TimeSeries",
    "ValidationError",
    "all_forecasters",
    "bottom_up",
    "compute_metrics",
    "coverage",
    "fingerprint",
    "flag_anomalies",
    "generate_series",
    "get_forecaster",
    "intervals_from_residuals",
    "reconciled",
    "rolling_origin_backtest",
    "select_champion",
    "top_down",
]

__version__ = "1.0.0"
