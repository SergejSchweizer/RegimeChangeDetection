from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pandas as pd

from regime_change.deribit_utils import fetch_deribit_funding_rates, fetch_deribit_ohlcv


class DummyProgressBar:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, amount=1):
        return None

    def close(self):
        return None


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_deribit_ohlcv_spot_returns_dataframe(monkeypatch):
    mock_payload = {
        "result": {
            "status": "ok",
            "ticks": [1704067200000 + i * 60000 for i in range(5)],
            "open": [100.0 + i * 1 for i in range(5)],
            "high": [101.0 + i * 1 for i in range(5)],
            "low": [99.0 + i * 1 for i in range(5)],
            "close": [100.5 + i * 1 for i in range(5)],
            "volume": [1000 + i * 10 for i in range(5)],
            "cost": [10000 + i * 100 for i in range(5)],
        }
    }

    session = Mock()
    session.get.return_value = DummyResponse(mock_payload)
    monkeypatch.setattr("regime_change.deribit_utils.requests.Session", lambda: session)
    monkeypatch.setattr("regime_change.deribit_utils.tqdm", DummyProgressBar)

    end_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(minutes=5)

    df = fetch_deribit_ohlcv("BTC", "spot", start_dt, end_dt, resolution="1", chunk_days=1, sleep_seconds=0)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "cost"]
    assert len(df) == 5
    assert df.index.tz.zone == "UTC"
    assert df.iloc[0]["open"] == 100.0


def test_fetch_deribit_ohlcv_invalid_market_type_raises():
    end_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=1)

    try:
        fetch_deribit_ohlcv("BTC", "invalid", start_dt, end_dt)
        assert False, "ValueError was not raised"
    except ValueError as exc:
        assert "market_type must be 'spot' or 'perpetual'" in str(exc)


def test_fetch_deribit_funding_rates_returns_dataframe(monkeypatch):
    mock_payload = {
        "result": [
            {
                "timestamp": 1704067200000 + i * 28800000,
                "funding_rate": 0.0001 + i * 0.00001,
                "index_price": 45000.0 + i * 10,
                "mark_price": 45010.0 + i * 10,
                "interest_8h": 0.0008 + i * 0.00001,
                "interest_1h": 0.0001 + i * 0.000001,
            }
            for i in range(4)
        ]
    }

    session = Mock()
    session.get.return_value = DummyResponse(mock_payload)
    monkeypatch.setattr("regime_change.deribit_utils.requests.Session", lambda: session)
    monkeypatch.setattr("regime_change.deribit_utils.tqdm", DummyProgressBar)

    end_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    start_dt = end_dt - timedelta(days=1)

    df = fetch_deribit_funding_rates("BTC", start_dt, end_dt, resolution="8h", chunk_days=1, sleep_seconds=0)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["funding_rate", "index_price", "mark_price", "interest_8h", "interest_1h"]
    assert len(df) == 4
    assert df.index.tz.zone == "UTC"
    assert df.iloc[0]["funding_rate"] == 0.0001
