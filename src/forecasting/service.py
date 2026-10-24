"""Optional FastAPI serving surface (import-guarded).

If FastAPI/pydantic are absent, importing this module still succeeds and
``create_app()`` returns ``None`` — the test suite then skips API tests
cleanly instead of failing at import time. Core logic never depends on
this module.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised only when FastAPI is installed
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    Field = lambda default, **kw: default  # type: ignore[assignment]
    _HAS_FASTAPI = False

from .errors import ForecastError
from .forecasters import get_forecaster
from .intervals import intervals_from_residuals
from .series import TimeSeries


class ForecastRequest(BaseModel if _HAS_FASTAPI else object):  # type: ignore[valid-type]
    """Request body for POST /forecast (pydantic when available)."""

    if _HAS_FASTAPI:
        series_name: str = Field(default="series")
        values: list[float] = Field(default_factory=list)
        start_date: str = "2026-01-01"
        seasonal_period: int = 7
        model: str = "holt-winters"
        horizon: int = 14
        interval_level: float = 0.90


def create_app() -> Any:
    """Build the FastAPI app, or ``None`` when FastAPI is unavailable."""
    if not _HAS_FASTAPI:
        return None
    app = FastAPI(title="forecasting-platform", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "forecasting-platform"}

    @app.post("/forecast")
    def forecast(payload: ForecastRequest) -> dict:
        try:
            series = TimeSeries.daily(
                payload.series_name, payload.values, payload.start_date, payload.seasonal_period
            )
            forecaster = get_forecaster(payload.model)
            result = forecaster.fit(series).predict(payload.horizon)
            response: dict[str, Any] = {
                "model": result.model_name,
                "forecasts": result.forecasts,
                "params": result.fitted_params,
                "timestamps": series.extend_labels(payload.horizon),
            }
            if len(result.in_sample_residuals) >= 10:
                interval = intervals_from_residuals(
                    result.forecasts, result.in_sample_residuals, payload.interval_level
                )
                response["interval"] = {"lower": interval.lower, "upper": interval.upper, "level": interval.level}
            return response
        except ForecastError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


app = create_app()
