"""Conftest for integration tests.

Defines pytest markers and fixtures for integration test suite.
"""

import pytest


def pytest_configure(config):
    """Register the integration marker."""
    config.addinivalue_line(
        "markers",
        "integration: integration test requiring external services (IB Gateway, OANDA, etc)",
    )
