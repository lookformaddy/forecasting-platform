"""Typed errors for the forecasting platform.

Every public entry point raises one of these (never a bare ``ValueError``
leaking out of a helper), so callers — CLI, service, or library users — can
handle failures by category instead of parsing messages.
"""

from __future__ import annotations


class ForecastError(Exception):
    """Base class for all forecasting-platform errors."""


class ValidationError(ForecastError):
    """Input data or configuration failed validation."""


class InsufficientDataError(ForecastError):
    """A series is too short for the requested model or horizon."""


class ModelNotFittedError(ForecastError):
    """A forecaster was asked to predict before :meth:`fit` ran."""


class RegistryError(ForecastError):
    """Champion-registry persistence or lookup failed."""
