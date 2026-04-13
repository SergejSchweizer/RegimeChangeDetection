"""
Shared pytest configuration and fixtures for all tests.

This file is automatically discovered by pytest and provides:
- Shared fixtures
- Common test configuration
- Mock setup/teardown
"""

import pytest
from datetime import datetime, timedelta, timezone
import pandas as pd
from unittest.mock import Mock, patch


@pytest.fixture
def sample_datetime_pair():
    """Provide a sample start and end datetime for testing."""
    end_dt = datetime(2024, 1, 31, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=7)
    return start_dt, end_dt


@pytest.fixture
def sample_ohlcv_data():
    """Provide sample OHLCV data as a DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1min", tz="UTC")
    return pd.DataFrame({
        "open": [100.0 + i * 0.1 for i in range(100)],
        "high": [101.0 + i * 0.1 for i in range(100)],
        "low": [99.0 + i * 0.1 for i in range(100)],
        "close": [100.5 + i * 0.1 for i in range(100)],
        "volume": [1000000.0 + i * 100 for i in range(100)],
        "cost": [100500000.0 + i * 50000 for i in range(100)],
    }, index=dates)


@pytest.fixture
def sample_funding_rates_data():
    """Provide sample funding rates data as a DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=50, freq="8h", tz="UTC")
    return pd.DataFrame({
        "funding_rate": [0.0001 + i * 0.00001 for i in range(50)],
        "index_price": [45000.0 + i * 10 for i in range(50)],
        "mark_price": [45010.0 + i * 10 for i in range(50)],
        "interest_8h": [0.0008 + i * 0.00001 for i in range(50)],
        "interest_1h": [0.0001 + i * 0.000001 for i in range(50)],
    }, index=dates)


@pytest.fixture
def mock_deribit_ohlcv_response():
    """Provide a mock Deribit OHLCV API response."""
    return {
        "jsonrpc": "2.0",
        "result": {
            "status": "ok",
            "ticks": [1704067200000 + i * 60000 for i in range(10)],
            "open": [100.0 + i * 0.1 for i in range(10)],
            "high": [101.0 + i * 0.1 for i in range(10)],
            "low": [99.0 + i * 0.1 for i in range(10)],
            "close": [100.5 + i * 0.1 for i in range(10)],
            "volume": [1000000.0 + i * 100 for i in range(10)],
            "cost": [100500000.0 + i * 50000 for i in range(10)],
        },
        "usIn": 1704067200000,
        "usOut": 1704067201000,
        "usDiff": 1000,
        "testnet": False,
    }


@pytest.fixture
def mock_deribit_funding_response():
    """Provide a mock Deribit funding rates API response."""
    return {
        "jsonrpc": "2.0",
        "result": [
            {
                "timestamp": 1704067200000 + i * 28800000,  # 8 hour intervals
                "funding_rate": 0.0001 + i * 0.00001,
                "index_price": 45000.0 + i * 10,
                "mark_price": 45010.0 + i * 10,
                "interest_8h": 0.0008 + i * 0.00001,
                "interest_1h": 0.0001 + i * 0.000001,
            }
            for i in range(10)
        ],
        "usIn": 1704067200000,
        "usOut": 1704067201000,
        "usDiff": 1000,
        "testnet": False,
    }


@pytest.fixture
def mock_requests_session():
    """Provide a mocked requests.Session object."""
    session = Mock()
    return session
