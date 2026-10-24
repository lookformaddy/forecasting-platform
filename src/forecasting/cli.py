"""Command-line interface: ``forecasting demo|forecast|backtest|serve``."""

from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .forecasters import get_forecaster
from .generator import GeneratorConfig, generate_series
from .observability import get_logger, timed
from .selection import select_champion
from .series import TimeSeries


def _demo(settings: Settings) -> int:
    logger = get_logger(level=settings.log_level)
    series, truth = generate_series(
        "demo-demand",
        GeneratorConfig(length=120, events={30: 25.0, 90: -18.0}, anomaly_indices=[50, 100]),
    )
    print(f"Generated {len(series)} points (true anomalies at {truth})")
    with timed("select_champion", logger) as metrics:
        champion, ranking = select_champion(series, folds=3, horizon=7)
        metrics.records["champion"] = champion.model_name
    for rank in ranking:
        print(f"  {rank.model_name:15s} MAE={rank.score:.3f}")
    print(f"Champion: {champion.model_name}")
    forecaster = get_forecaster(champion.model_name).fit(series)
    result = forecaster.predict(14)
    print(f"14-step forecast: {[round(v, 1) for v in result.forecasts]}")
    return 0


def _forecast(args: argparse.Namespace) -> int:
    values = [float(v) for v in args.values.split(",")]
    series = TimeSeries.daily(args.name, values, args.start, args.period)
    result = get_forecaster(args.model).fit(series).predict(args.horizon)
    print(json.dumps({"model": result.model_name, "forecasts": result.forecasts, "params": result.fitted_params}, indent=2))
    return 0


def _backtest(args: argparse.Namespace) -> int:
    values = [float(v) for v in args.values.split(",")]
    series = TimeSeries.daily(args.name, values, args.start, args.period)
    champion, ranking = select_champion(series, folds=args.folds, horizon=args.horizon, metric=args.metric)
    print(json.dumps({"champion": champion.to_dict(), "ranking": [r.to_dict() for r in ranking]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecasting", description="Time-series forecasting platform")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run the synthetic end-to-end demo")
    f = sub.add_parser("forecast", help="forecast one series from comma-separated values")
    f.add_argument("--name", default="series")
    f.add_argument("--values", required=True)
    f.add_argument("--start", default="2026-01-01")
    f.add_argument("--period", type=int, default=7)
    f.add_argument("--model", default="holt-winters")
    f.add_argument("--horizon", type=int, default=14)
    f.set_defaults(func=_forecast)
    b = sub.add_parser("backtest", help="rolling-origin model selection for one series")
    b.add_argument("--name", default="series")
    b.add_argument("--values", required=True)
    b.add_argument("--start", default="2026-01-01")
    b.add_argument("--period", type=int, default=7)
    b.add_argument("--folds", type=int, default=3)
    b.add_argument("--horizon", type=int, default=7)
    b.add_argument("--metric", default="mae")
    b.set_defaults(func=_backtest)
    s = sub.add_parser("serve", help="serve the FastAPI app")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return _demo(settings)
    if args.command == "serve":
        from .service import create_app

        app = create_app()
        if app is None:
            print("FastAPI is not installed; pip install -r requirements.txt first.", file=sys.stderr)
            return 1
        import uvicorn

        uvicorn.run(app, host=args.host or settings.host, port=args.port or settings.port)
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
