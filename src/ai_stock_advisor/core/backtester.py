"""
Backtesting Engine Module.
Simulates a trend-following option-adjusted technical strategy over 5 years of daily data.
Calculates key metrics: Win Rate, Profit Factor, Maximum Drawdown, and Average Return.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from config.settings import settings
from ai_stock_advisor.core.exceptions import MarketDataServiceError
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.services.market_data.client import MarketDataClient

logger = logging.getLogger("ai_stock_advisor.core.backtester")


class BacktestingEngine:
    """
    Backtester for technical indicators trading strategies.
    Runs trade executions over 5 years and logs return profiles.
    """

    def __init__(
        self,
        market_client: MarketDataClient,
        indicator_engine: TechnicalIndicatorEngine,
    ) -> None:
        """Initializes the backtester with core market data client stack."""
        self.market_client = market_client
        self.indicator_engine = indicator_engine

    def run_backtest(
        self,
        ticker: str,
        initial_capital: float = 100000.0,
        save_dir: Path | None = None,
    ) -> Dict[str, Any]:
        """
        Runs the backtest over 5 years of daily records.
        Simulates capital updates, Stop Loss/Targets triggers, and computes metrics.
        Exports trade logs in CSV and JSON formats.
        """
        logger.info("Executing 5-year backtest for symbol '%s'...", ticker)
        out_dir = save_dir or Path(settings.BASE_DIR) / "data"

        try:
            # Fetch 5 years of daily OHLCV
            hist_df = self.market_client.fetch_ohlcv(ticker, period="5y", interval="1d")
            if hist_df.empty or len(hist_df) < 50:
                raise ValueError(f"Insufficient history data retrieved for ticker: {ticker}")

            # Compute technical indicators
            df = self.indicator_engine.compute_all_indicators(hist_df)
        except Exception as exc:
            raise MarketDataServiceError(
                f"Backtest failed to load data for ticker '{ticker}'",
                details={"ticker": ticker, "error": str(exc)}
            ) from exc

        # Initialize Simulation Variables
        capital = initial_capital
        in_position = False
        qty = 0.0
        entry_price = 0.0
        entry_date = None
        target = 0.0
        stop_loss = 0.0

        trades: List[Dict[str, Any]] = []
        equity_curve: List[Dict[str, Any]] = []

        # Start sequence from index 1 to allow prev_row calculations
        for idx in range(1, len(df)):
            row = df.iloc[idx]
            dt = df.index[idx]
            
            # Helper timestamp conversion
            dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)

            if not in_position:
                # ----------------------------------------------------
                # Long Entry Condition:
                # Close above EMA20/50, EMA20 > EMA50, RSI > 50, MACD Hist > 0
                # ----------------------------------------------------
                macd_hist = float(row["MACD"] - row["MACD_Signal"])
                is_bullish_trend = (
                    row["Close"] > row["EMA_20"]
                    and row["Close"] > row["EMA_50"]
                    and row["EMA_20"] > row["EMA_50"]
                )
                is_momentum_strong = row["RSI_14"] > 50.0 and macd_hist > 0.0

                if is_bullish_trend and is_momentum_strong:
                    in_position = True
                    entry_price = float(row["Close"])
                    entry_date = dt_str
                    qty = capital / entry_price
                    
                    # Target/SL defined via ATR volatility metric
                    atr = float(row["ATR_14"])
                    target = entry_price + (1.5 * atr)
                    stop_loss = entry_price - (1.0 * atr)
            else:
                # ----------------------------------------------------
                # Position Monitoring & Exit Conditions:
                # Check target hit, stop loss hit, or indicator decay
                # ----------------------------------------------------
                exit_price = 0.0
                exit_reason = ""

                # Conservative SL check first
                if row["Low"] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "Stop Loss"
                elif row["High"] >= target:
                    exit_price = target
                    exit_reason = "Target"
                elif row["Close"] < row["EMA_50"] or row["RSI_14"] < 40.0:
                    exit_price = float(row["Close"])
                    exit_reason = "Technical Exit"

                if exit_price > 0.0:
                    exit_val = qty * exit_price
                    trade_profit = exit_val - (qty * entry_price)
                    trade_return = (exit_price / entry_price - 1.0) * 100.0

                    capital = exit_val
                    trades.append({
                        "Ticker": ticker,
                        "Buy_Date": entry_date,
                        "Buy_Price": round(entry_price, 2),
                        "Sell_Date": dt_str,
                        "Sell_Price": round(exit_price, 2),
                        "Return_Pct": round(trade_return, 2),
                        "Profit": round(trade_profit, 2),
                        "Exit_Reason": exit_reason
                    })
                    in_position = False
                    qty = 0.0

            # Daily Equity Tracking
            current_equity = qty * float(row["Close"]) if in_position else capital
            equity_curve.append({
                "Date": dt_str,
                "Equity": round(current_equity, 2)
            })

        # Force exit on last row if still holding
        if in_position:
            last_row = df.iloc[-1]
            last_date = df.index[-1]
            last_dt_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)
            
            exit_price = float(last_row["Close"])
            exit_val = qty * exit_price
            trade_profit = exit_val - (qty * entry_price)
            trade_return = (exit_price / entry_price - 1.0) * 100.0

            capital = exit_val
            trades.append({
                "Ticker": ticker,
                "Buy_Date": entry_date,
                "Buy_Price": round(entry_price, 2),
                "Sell_Date": last_dt_str,
                "Sell_Price": round(exit_price, 2),
                "Return_Pct": round(trade_return, 2),
                "Profit": round(trade_profit, 2),
                "Exit_Reason": "End of History"
            })
            equity_curve[-1]["Equity"] = round(capital, 2)

        # ----------------------------------------------------
        # Metrics Calculations
        # ----------------------------------------------------
        total_trades = len(trades)
        winning_trades = [t for t in trades if t["Profit"] > 0.0]
        losing_trades = [t for t in trades if t["Profit"] <= 0.0]
        
        win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0
        
        sum_wins = sum(t["Profit"] for t in winning_trades)
        sum_losses = abs(sum(t["Profit"] for t in losing_trades))
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0.0 else (99.9 if sum_wins > 0.0 else 1.0)
        
        avg_return = np.mean([t["Return_Pct"] for t in trades]) if total_trades > 0 else 0.0
        total_return = ((capital / initial_capital) - 1.0) * 100.0

        # Maximum Drawdown calculation
        equity_series = pd.DataFrame(equity_curve)["Equity"]
        peaks = equity_series.cummax()
        drawdowns = (peaks - equity_series) / peaks
        max_drawdown = float(drawdowns.max()) * 100.0 if not drawdowns.empty else 0.0

        results = {
            "Ticker": ticker,
            "Total_Return_Pct": round(total_return, 2),
            "Win_Rate_Pct": round(win_rate, 1),
            "Profit_Factor": round(profit_factor, 2),
            "Max_Drawdown_Pct": round(max_drawdown, 2),
            "Average_Return_Pct": round(avg_return, 2),
            "Total_Trades": total_trades,
            "Initial_Capital": initial_capital,
            "Final_Capital": round(capital, 2),
            "Trades": trades,
            "Equity_Curve": equity_curve
        }

        # ----------------------------------------------------
        # Export Reports
        # ----------------------------------------------------
        out_dir.mkdir(parents=True, exist_ok=True)
        trades_csv_path = out_dir / "backtest_trades.csv"
        metrics_json_path = out_dir / "backtest_metrics.json"

        try:
            # Save CSV of individual trades
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                trades_df.to_csv(trades_csv_path, index=False)
                logger.info("Saved backtest trades logs to %s", trades_csv_path)
            
            # Save JSON stats
            with open(metrics_json_path, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)
                logger.info("Saved backtest metrics summary to %s", metrics_json_path)
        except Exception as exc:
            logger.error("Failed exporting backtest report logs: %s", exc)

        return results
