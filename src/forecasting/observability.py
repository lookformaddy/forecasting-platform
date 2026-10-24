"""Lightweight observability: structured logs and run timing.

No third-party logging framework is required; records are plain dicts so
they can be forwarded to any sink without a dependency.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


def get_logger(name: str = "forecasting", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@dataclass
class RunMetrics:
    """Timing + counts for one operation, for audit and debugging."""

    operation: str
    duration_s: float = 0.0
    records: dict[str, int | float | str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"operation": self.operation, "duration_s": round(self.duration_s, 4), **self.records}


@contextmanager
def timed(operation: str, logger: logging.Logger | None = None) -> Iterator[RunMetrics]:
    """Time a block and optionally log the resulting metrics."""
    metrics = RunMetrics(operation=operation)
    started = time.perf_counter()
    try:
        yield metrics
    finally:
        metrics.duration_s = time.perf_counter() - started
        if logger is not None:
            logger.info("run_metrics=%s", metrics.to_dict())
