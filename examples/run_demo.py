"""End-to-end demo: generate, backtest, champion, forecast, reconcile, flag.

Run from the project root: ``python examples/run_demo.py`` (after
``pip install -e .``), or ``python -m forecasting demo``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.anomalies import flag_anomalies
from forecasting.forecasters import get_forecaster
from forecasting.generator import GeneratorConfig, generate_series
from forecasting.hierarchy import HierarchyNode, reconciled
from forecasting.intervals import coverage, intervals_from_residuals
from forecasting.registry import ChampionRegistry
from forecasting.selection import select_champion
from forecasting.series import TimeSeries


def main() -> int:
    series, truth = generate_series(
        "demo-demand",
        GeneratorConfig(length=120, events={30: 25.0, 90: -18.0}, anomaly_indices=[50, 100]),
    )
    print(f"Generated {len(series)} points; injected anomalies at {truth}")

    champion, ranking = select_champion(series, folds=3, horizon=7)
    print("\nBacktest ranking (MAE):")
    for rank in ranking:
        print(f"  {rank.model_name:15s} {rank.score:.3f}")
    print(f"Champion: {champion.model_name}")

    result = get_forecaster(champion.model_name).fit(series).predict(14)
    interval = intervals_from_residuals(result.forecasts, result.in_sample_residuals)
    print(f"\n14-step forecast: {[round(v, 1) for v in result.forecasts]}")
    print(f"90% interval, first step: [{interval.lower[0]:.1f}, {interval.upper[0]:.1f}]")

    # Hold out the last 14 points, forecast them, and flag anomalies there.
    train, actual = series.slice(len(series) - 14), series.values[-14:]
    held = get_forecaster(champion.model_name).fit(train).predict(14)
    held_interval = intervals_from_residuals(held.forecasts, held.in_sample_residuals)
    flags = flag_anomalies(actual, held.forecasts, held_interval.lower, held_interval.upper)
    print(f"\nHeld-out coverage: {coverage(actual, held_interval):.2f}")
    print(f"Anomaly flags on held-out window: {len(flags)}")
    for flag in flags:
        print(f"  {flag.to_dict()}")

    # Hierarchical demo: two regions summing to a total, reconciled.
    a, _ = generate_series("region-a", GeneratorConfig(length=90, seed=11, base_level=60.0))
    b, _ = generate_series("region-b", GeneratorConfig(length=90, seed=22, base_level=40.0))
    total = TimeSeries(
        name="total", values=[x + y for x, y in zip(a.values, b.values)],
        timestamps=a.timestamps, seasonal_period=7,
    )
    root = HierarchyNode(total, [HierarchyNode(a, []), HierarchyNode(b, [])])
    rec = reconciled(root, lambda: get_forecaster(champion.model_name), 7)
    print(f"\nReconciled total, first step: {rec['total'][0]:.1f} "
          f"= {rec['region-a'][0]:.1f} + {rec['region-b'][0]:.1f}")

    registry = ChampionRegistry("/tmp/forecasting-demo-registry.json")
    record = registry.register(series, champion.model_name, 14, result.forecasts,
                               params=result.fitted_params, scores={"mae": champion.score})
    print(f"\nRegistered forecast v{record.version} (fingerprint {record.data_fingerprint[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
