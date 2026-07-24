from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from ai_stock_advisor.db.database import SessionLocal, WatchlistItem, UserSettings, init_db
from ai_stock_advisor.telegram.bot import (
    handle_add_watchlist,
    handle_remove_watchlist,
    handle_watchlist_list,
    handle_risk_calculator,
)


@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure database gets set up and wiped before each test."""
    init_db()
    db = SessionLocal()
    db.query(WatchlistItem).delete()
    db.query(UserSettings).delete()
    db.commit()
    db.close()


def test_add_watchlist_command() -> None:
    """Verify `/addwatch` successfully saves watched symbol to SQLite."""
    mock_message = MagicMock()
    mock_message.text = "/addwatch TCS.NS"
    mock_message.chat.id = 98765

    with patch("ai_stock_advisor.telegram.bot.bot.send_message") as mock_send:
        handle_add_watchlist(mock_message)
        
        mock_send.assert_called_once()
        reply_text = mock_send.call_args[0][1]
        assert "TCS.NS" in reply_text
        assert "added to watchlist" in reply_text

        # Assert database record was written
        db = SessionLocal()
        item = db.query(WatchlistItem).filter_by(telegram_chat_id="98765", ticker="TCS.NS").first()
        assert item is not None
        db.close()


def test_remove_watchlist_command() -> None:
    """Verify `/removewatch` removes watched symbol from SQLite."""
    # Seed symbol first
    db = SessionLocal()
    db.add(WatchlistItem(telegram_chat_id="98765", ticker="INFY.NS"))
    db.commit()
    db.close()

    mock_message = MagicMock()
    mock_message.text = "/removewatch INFY.NS"
    mock_message.chat.id = 98765

    with patch("ai_stock_advisor.telegram.bot.bot.send_message") as mock_send:
        handle_remove_watchlist(mock_message)
        
        mock_send.assert_called_once()
        reply_text = mock_send.call_args[0][1]
        assert "INFY.NS" in reply_text
        assert "removed from watchlist" in reply_text

        # Verify database record deleted
        db = SessionLocal()
        item = db.query(WatchlistItem).filter_by(telegram_chat_id="98765", ticker="INFY.NS").first()
        assert item is None
        db.close()


def test_watchlist_list_command() -> None:
    """Verify `/watchlist` list command lists symbols watched in SQLite."""
    # Seed symbols
    db = SessionLocal()
    db.add(WatchlistItem(telegram_chat_id="98765", ticker="RELIANCE.NS"))
    db.add(WatchlistItem(telegram_chat_id="98765", ticker="SBIN.NS"))
    db.commit()
    db.close()

    mock_message = MagicMock()
    mock_message.text = "/watchlist"
    mock_message.chat.id = 98765

    with patch("ai_stock_advisor.telegram.bot.bot.send_message") as mock_send:
        handle_watchlist_list(mock_message)
        
        mock_send.assert_called_once()
        reply_text = mock_send.call_args[0][1]
        assert "RELIANCE.NS" in reply_text
        assert "SBIN.NS" in reply_text


def test_risk_calculator_calculations() -> None:
    """Verify position size formulas inside `/risk` calculator handler."""
    mock_message = MagicMock()
    mock_message.text = "/risk 100000 2.0 TCS.NS"
    mock_message.chat.id = 98765

    mock_data = pd.DataFrame({
        "Open": [100.0, 100.0, 100.0],
        "High": [110.0, 110.0, 110.0],
        "Low": [90.0, 90.0, 90.0],
        "Close": [100.0, 100.0, 100.0],
        "Volume": [1000, 1000, 1000],
    })

    with patch("ai_stock_advisor.services.market_data.client.MarketDataClient.fetch_ohlcv", return_value=mock_data):
        with patch("ai_stock_advisor.telegram.bot.bot.send_message") as mock_send:
            with patch("ai_stock_advisor.telegram.bot.bot.send_chat_action") as mock_action:
                handle_risk_calculator(mock_message)
                
                mock_send.assert_called_once()
                reply_text = mock_send.call_args[0][1]
                assert "RISK & POSITION SIZING MATRIX" in reply_text
                assert "TCS.NS" in reply_text
                # capital 100k, risk 2% = 2000 max risk
                assert "Maximum Acceptable Loss" in reply_text
                assert "Recommended Quantity" in reply_text
