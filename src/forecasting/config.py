"""Environment-driven configuration (see ``.env.example``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime knobs; every field has a sane default."""

    default_horizon: int = field(default_factory=lambda: _int("FORECAST_HORIZON", 14))
    default_seasonal_period: int = field(default_factory=lambda: _int("FORECAST_SEASONAL_PERIOD", 7))
    backtest_folds: int = field(default_factory=lambda: _int("FORECAST_BACKTEST_FOLDS", 3))
    backtest_horizon: int = field(default_factory=lambda: _int("FORECAST_BACKTEST_HORIZON", 7))
    interval_level: float = field(default_factory=lambda: _float("FORECAST_INTERVAL_LEVEL", 0.90))
    anomaly_z_threshold: float = field(default_factory=lambda: _float("FORECAST_ANOMALY_Z", 3.0))
    data_dir: str = field(default_factory=lambda: os.getenv("FORECAST_DATA_DIR", "./data"))
    registry_path: str = field(default_factory=lambda: os.getenv("FORECAST_REGISTRY_PATH", ""))
    host: str = field(default_factory=lambda: os.getenv("FORECAST_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("FORECAST_PORT", 8000))
    log_level: str = field(default_factory=lambda: os.getenv("FORECAST_LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        if not self.registry_path:
            self.registry_path = os.path.join(self.data_dir, "registry.json")
        if not 0.0 < self.interval_level < 1.0:
            raise ValueError("FORECAST_INTERVAL_LEVEL must be in (0, 1)")
        if self.default_horizon < 1 or self.backtest_folds < 1:
            raise ValueError("horizon and backtest folds must be >= 1")
