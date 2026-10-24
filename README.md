# forecasting-platform

An enterprise time-series forecasting platform, built from scratch in pure
Python standard library: deterministic synthetic series generation, four
forecasters implemented by hand (no stats library), hierarchical forecasting
with reconciliation, rolling-origin backtesting, prediction intervals,
anomaly flagging, backtest-driven model selection with a versioned champion
registry, and an optional FastAPI serving surface.

**Track:** ML Engineering · **Catalog #7** of the enterprise portfolio series.

## The enterprise problem

A forecast that nobody can audit is a liability. Production forecasting
teams need to answer, for every number they publish:

- **Which model produced it, and why that model?** — not the loudest model,
  the one that won a rolling-origin backtest on this series.
- **How uncertain is it?** — a point forecast without an interval hides the
  risk in the decision it drives.
- **Does it add up?** — a regional forecast that doesn't sum to the national
  forecast will be caught by finance, not by the model.
- **What changed, and when?** — every published forecast versioned, with a
  fingerprint of the data it was trained on.
- **Is something wrong right now?** — actuals drifting outside intervals
  should page someone, not sit in a dashboard.

This repo implements all of it, clean-room, with synthetic data only.

## Architecture overview

```
synthetic generator ──▶ TimeSeries ──▶ forecasters (naive / seasonal-naive /
  trend + seasonality + events +     │   Holt-Winters / linear-trend OLS)
  noise + injected anomalies         │
                                       ├─▶ backtest (rolling origin, MAE/RMSE/MAPE/sMAPE)
                                       ├─▶ selection ──▶ champion registry (versioned, JSON)
                                       ├─▶ intervals (residual quantiles, sqrt-horizon widening)
                                       ├─▶ anomalies (interval breach + residual z-score)
                                       └─▶ hierarchy (bottom-up / top-down / reconciled)
```

See [docs/architecture.md](docs/architecture.md) for the full component map,
data flow, and design trade-offs.

## How to run it

```bash
python -m venv /tmp/venv && source /tmp/venv/bin/activate
pip install -r requirements.txt
pip install -e .

# run the tests (API tests auto-skip without FastAPI)
pytest -q

# end-to-end demo: generate, backtest, pick a champion, forecast, flag anomalies
python -m forecasting demo
# or: make demo / python examples/run_demo.py

# forecast one series from the CLI
python -m forecasting forecast --values "100,102,101,105,103,107,106,110,108,112,111,115" --model holt-winters --horizon 7

# rolling-origin model selection from the CLI
python -m forecasting backtest --values "100,102,..." --folds 3 --horizon 7

# serve the HTTP API
python -m forecasting serve            # -> http://127.0.0.1:8000
# POST /forecast  {"series_name","values","start_date","seasonal_period","model","horizon","interval_level"}
# GET  /health
```

Docker:

```bash
docker build -t forecasting-platform .
docker run -p 8000:8000 forecasting-platform
```

Configuration is environment-driven; copy `.env.example` to `.env` to
override horizons, backtest folds, interval level, anomaly threshold, and
the registry path.

## Key design decisions and trade-offs

- **Pure stdlib core.** No numpy/pandas: the OLS solver is Gaussian
  elimination, Holt-Winters is a hand-rolled smoother, quantiles are
  interpolation. The library runs anywhere Python runs, and every number is
  traceable to code a reviewer can read in one sitting.
- **Baselines are first-class.** Naive and seasonal-naive ship alongside
  the real models and compete in every backtest — a fancy model that can't
  beat seasonal-naive on a seasonal series should not be champion.
- **Backtest, don't eyeball.** Model selection scores expanding-window
  rolling-origin folds only; in-sample fit is never a selection metric.
  Models that fail a fold are skipped, never silently ranked last.
- **Intervals are honest about their assumption.** Residual quantiles assume
  roughly stationary residuals; the `coverage` helper exists so users check
  that assumption on held-out data instead of trusting it.
- **Hierarchies are coherent by construction.** Bottom-up sums, top-down
  splits by historical share, and reconciled forecasts scale children to the
  parent — a reconciled forecast can never fail a sum check.
- **Registry writes are atomic** (write-then-rename) and every record carries
  a SHA-256 fingerprint of its training data, so a forecast can always be
  traced back to the exact data that produced it.
- **FastAPI is optional.** The service module import-guards FastAPI/pydantic;
  without them the library, CLI, and non-API tests all still work.

## What it demonstrates

For a Director-level interview, this repo is evidence of: building a
forecasting system end to end (generation → models → evaluation →
selection → serving → monitoring); implementing statistical methods from
first principles instead of wrapping a library; designing for auditability
(versioning, fingerprints, backtest-driven selection); hierarchical
forecasting with coherence guarantees; and operating a model in production
(intervals, anomaly detection, champion registry) rather than just training
one.

## CI

The CI definition ships at [docs/github-actions-ci.yml](docs/github-actions-ci.yml).
It activates by moving it to `.github/workflows/ci.yml` (kept out of
`.github/workflows/` here because the GitHub App used for publishing lacks
the Workflows permission).

## Clean-room note

All code is original and all data is synthetic (generated trend +
seasonality + events + noise). No employer code, data, or internal designs
are included.
