# Architecture

## Component map

| Module | Responsibility |
| --- | --- |
| `series.py` | `TimeSeries`: validated values + ISO timestamps + seasonal period; slicing and future labels. |
| `generator.py` | Deterministic synthetic series: trend + seasonality + events + Gaussian noise + injected anomalies. |
| `forecasters.py` | `BaseForecaster` contract + naive, seasonal-naive, Holt-Winters (grid-searched), linear-trend OLS (Gaussian elimination). |
| `metrics.py` | MAE / RMSE / MAPE / sMAPE / bias + interpolation quantiles. |
| `backtest.py` | Rolling-origin expanding-window evaluation, per-fold and mean scores. |
| `selection.py` | Backtest every candidate, rank by one metric, return champion + full ranking. |
| `intervals.py` | Residual-quantile prediction intervals, widened by `sqrt(step)`; coverage check. |
| `anomalies.py` | Interval-breach + residual z-score flags with warning/critical severity. |
| `hierarchy.py` | Tree of series; bottom-up, top-down, and proportionally reconciled forecasts. |
| `registry.py` | Versioned forecast records (JSON, atomic writes, SHA-256 data fingerprint) + champion lookup. |
| `observability.py` | Structured logger + timed-run metrics. |
| `config.py` | Environment-driven settings with validated defaults. |
| `service.py` | Optional FastAPI app (import-guarded; `None` without FastAPI). |
| `cli.py` | `demo` / `forecast` / `backtest` / `serve` commands. |

## Data flow (demo path)

```
GeneratorConfig ──▶ generate_series ──▶ TimeSeries
                                            │
                              ┌─────────────┼──────────────┐
                              ▼             ▼              ▼
                        select_champion  hierarchy   get_forecaster.fit
                        (backtest all)   (reconcile)        │
                              │                              ▼
                              └──────▶ champion ──▶ predict ──▶ intervals
                                                              │
                                                              ▼
                                                    flag_anomalies(actuals)
                                                              │
                                                              ▼
                                              registry.register(versioned record)
```

## Design trade-offs

- **Stdlib-only core** trades raw speed for auditability and zero install
  friction; series of a few thousand points are fast, million-point series
  would want a vectorised backend behind the same interfaces.
- **Grid-searched Holt-Winters** (3×3×3 smoothing grid scored by in-sample
  one-step MAE) is simple and deterministic; a production system would use
  proper likelihood optimisation and cross-validated parameter choice.
- **Residual-quantile intervals** assume stationary residuals and widen
  only with `sqrt(horizon)`; they understate uncertainty under regime
  change, which is exactly what the anomaly flags are for.
- **Proportional reconciliation** preserves each child's relative share;
  it does not use covariance-aware (MinT-style) reconciliation, which needs
  a residual covariance matrix the stdlib core deliberately avoids.
- **JSON registry** is a single-file store with atomic writes — correct for
  one publisher, not for concurrent writers; a database slots in behind the
  `ChampionRegistry` interface.
- **File sessions, no auth**: the service is a demo surface. Production
  deployment adds authentication, rate limiting, and a real store.
