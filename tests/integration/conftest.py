"""
Pytest configuration for integration tests.

Integration tests require a GPU and/or internet access to download models.
They are NOT run as part of the default ``pytest`` suite.

To run integration tests explicitly::

    pytest tests/integration/ -v
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """Mark all tests in this directory as integration tests."""
    for item in items:
        item.add_marker(pytest.mark.integration)
