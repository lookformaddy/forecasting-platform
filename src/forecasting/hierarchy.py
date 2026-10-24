"""Hierarchical forecasting: aggregate + child series, reconciled.

A hierarchy is a tree: one parent series whose values equal the sum of its
children (e.g. total demand = region A + region B + region C). Three
strategies are provided:

- **bottom-up**: forecast every child, sum them for the parent.
- **top-down**: forecast the parent, split it by each child's historical
  share of the parent total.
- **reconciled**: forecast parent and children independently, then scale the
  child forecasts proportionally so they sum exactly to the parent forecast
  (the standard proportional reconciliation — coherent by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ValidationError
from .forecasters import BaseForecaster
from .series import TimeSeries


@dataclass
class HierarchyNode:
    series: TimeSeries
    children: list["HierarchyNode"]

    def is_leaf(self) -> bool:
        return not self.children


def _check_coherent(node: HierarchyNode) -> None:
    """Warn-free strict check: parent values must equal the child sum."""
    if node.is_leaf():
        return
    for child in node.children:
        _check_coherent(child)
    if any(len(c.series) != len(node.series) for c in node.children):
        raise ValidationError(f"hierarchy under '{node.series.name}' has mismatched lengths")
    for t, parent_value in enumerate(node.series.values):
        child_sum = sum(c.series.values[t] for c in node.children)
        if abs(parent_value - child_sum) > 1e-6:
            raise ValidationError(
                f"hierarchy incoherent at t={t}: parent {parent_value} != child sum {child_sum}"
            )


def _forecast_with(factory: "callable[[], BaseForecaster]", series: TimeSeries, horizon: int) -> list[float]:
    forecaster = factory()
    forecaster.fit(series)
    return forecaster.predict(horizon).forecasts


def bottom_up(node: HierarchyNode, factory: "callable[[], BaseForecaster]", horizon: int) -> dict[str, list[float]]:
    """Forecast leaves, sum up the tree; returns ``{series_name: forecast}``."""
    _check_coherent(node)
    out: dict[str, list[float]] = {}
    if node.is_leaf():
        out[node.series.name] = _forecast_with(factory, node.series, horizon)
        return out
    child_forecasts: list[list[float]] = []
    for child in node.children:
        child_out = bottom_up(child, factory, horizon)
        out.update(child_out)
        child_forecasts.append(child_out[child.series.name])
    out[node.series.name] = [sum(step) for step in zip(*child_forecasts)]
    return out


def top_down(node: HierarchyNode, factory: "callable[[], BaseForecaster]", horizon: int) -> dict[str, list[float]]:
    """Forecast the root, split it down by historical shares."""
    _check_coherent(node)
    out: dict[str, list[float]] = {}
    parent_forecast = _forecast_with(factory, node.series, horizon)
    out[node.series.name] = parent_forecast
    if node.is_leaf():
        return out
    totals = [sum(c.series.values) for c in node.children]
    grand = sum(totals) or 1.0
    shares = [t / grand for t in totals]
    for child, share in zip(node.children, shares):
        child_out = top_down(child, factory, horizon) if not child.is_leaf() else {}
        # Child forecast is its share of the parent forecast at this level;
        # deeper levels are then split recursively from that forecast.
        child_forecast = [share * v for v in parent_forecast]
        out[child.series.name] = child_forecast
        for key, value in child_out.items():
            if key != child.series.name:
                out[key] = value
        if not child.is_leaf():
            # Re-split the child's own forecast among its children by share.
            sub_totals = [sum(g.series.values) for g in child.children]
            sub_grand = sum(sub_totals) or 1.0
            for grandchild, sub_total in zip(child.children, sub_totals):
                out[grandchild.series.name] = [sub_total / sub_grand * v for v in child_forecast]
    return out


def reconciled(node: HierarchyNode, factory: "callable[[], BaseForecaster]", horizon: int) -> dict[str, list[float]]:
    """Independent forecasts, children scaled to sum to the parent forecast."""
    _check_coherent(node)
    out: dict[str, list[float]] = {}
    if node.is_leaf():
        out[node.series.name] = _forecast_with(factory, node.series, horizon)
        return out
    parent_forecast = _forecast_with(factory, node.series, horizon)
    out[node.series.name] = parent_forecast
    child_forecasts: dict[str, list[float]] = {}
    for child in node.children:
        child_out = reconciled(child, factory, horizon)
        # For internal children, the recursive call already reconciled below;
        # use the child's own (reconciled) forecast as its base here.
        child_forecasts[child.series.name] = child_out[child.series.name]
        out.update(child_out)
    for h in range(horizon):
        total = sum(child_forecasts[name][h] for name in child_forecasts)
        if total == 0:
            shares = {name: 1.0 / len(child_forecasts) for name in child_forecasts}
        else:
            shares = {name: child_forecasts[name][h] / total for name in child_forecasts}
        for name in child_forecasts:
            out[name][h] = shares[name] * parent_forecast[h]
    # Reconcile deeper levels to their (now adjusted) parent forecasts.
    for child in node.children:
        if not child.is_leaf():
            _rescale_subtree(child, out, horizon)
    return out


def _rescale_subtree(node: HierarchyNode, forecasts: dict[str, list[float]], horizon: int) -> None:
    """Scale a subtree's leaves so they sum to the node's forecast."""
    parent = forecasts[node.series.name]
    leaves = [c for c in node.children]
    for h in range(horizon):
        total = sum(forecasts[c.series.name][h] for c in leaves)
        if total == 0:
            continue
        for child in leaves:
            forecasts[child.series.name][h] = forecasts[child.series.name][h] / total * parent[h]
    for child in leaves:
        if not child.is_leaf():
            _rescale_subtree(child, forecasts, horizon)
