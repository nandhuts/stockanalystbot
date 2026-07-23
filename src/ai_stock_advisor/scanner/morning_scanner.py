"""
Morning Opportunity Scanner Orchestrator.
Manages the daily pre-market F&O scan pipeline and database persistence.
"""
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List
import pandas as pd

from config.settings import settings
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.core.options import OptionAnalyzer
from ai_stock_advisor.services.market_data.client import MarketDataClient
from ai_stock_advisor.services.llm.news_analyzer import NewsAnalyzer
from ai_stock_advisor.services.scanner_service import MorningScannerService
from ai_stock_advisor.analysis.news_sentiment import MorningNewsSentimentAnalyzer
from ai_stock_advisor.analysis.option_chain_analyzer import MorningOptionChainAnalyzer
from ai_stock_advisor.analysis.risk_engine import RiskEngine
from ai_stock_advisor.analysis.opportunity_ranker import OpportunityRanker
from ai_stock_advisor.db.database import SessionLocal, DailyScan, init_db

logger = logging.getLogger("ai_stock_advisor.scanner.morning_scanner")


class MorningOpportunityScanner:
    """
    Morning pre-market F&O scanner pipeline.
    Runs scans, analyzes indicators, option chains, news sentiment, and risk profiles,
    ranks the top 10 opportunities, and saves results in SQLite.
    """

    def __init__(
        self,
        scanner_service: MorningScannerService | None = None,
        indicator_engine: TechnicalIndicatorEngine | None = None,
        news_analyzer: NewsAnalyzer | None = None,
        option_analyzer: OptionAnalyzer | None = None,
        db_session_factory: Any = None,
    ) -> None:
        """Initializes scanner pipeline clients and engines."""
        self.market_client = MarketDataClient()
        self.scanner_service = scanner_service or MorningScannerService(self.market_client)
        self.indicator_engine = indicator_engine or TechnicalIndicatorEngine()
        self.news_sentiment_analyzer = MorningNewsSentimentAnalyzer(news_analyzer or NewsAnalyzer())
        
        opt_analyzer = option_analyzer or OptionAnalyzer(self.market_client, self.indicator_engine)
        self.option_chain_analyzer = MorningOptionChainAnalyzer(opt_analyzer)
        
        self.risk_engine = RiskEngine()
        self.opportunity_ranker = OpportunityRanker()
        self.db_session_factory = db_session_factory or SessionLocal

        # Initialize database tables on startup
        init_db()

    def run_scan(self, force_refresh: bool = True) -> List[Dict[str, Any]]:
        """
        Executes the full morning scan pipeline.
        Fetches symbols, downloads timeframes, calculates scores, ranks Top 10,
        saves to SQLite database, and exports CSV/JSON files.
        """
        start_time = time.time()
        logger.info("🌅 Starting Morning F&O Opportunity Scan...")
        
        try:
            # 1. Fetch active F&O tickers
            tickers = self.scanner_service.fetch_fo_tickers()
            
            # Limit count for testing/rate-limiting safety if needed
            # In production, we run the full list. We will run the full list here.
            # 2. Bulk download daily, 15m, and 5m candles
            datasets = self.scanner_service.fetch_all_candles(tickers)
            
            candidates: List[Dict[str, Any]] = []
            scan_date = datetime.utcnow()
            
            # 3. Process each stock
            for idx, (ticker, data) in enumerate(datasets.items(), 1):
                df_daily = data.get("daily")
                df_15m = data.get("15m")
                df_5m = data.get("5m")
                
                if df_daily.empty or len(df_daily) < 20:
                    logger.warning("Skipping '%s': Insufficient daily candles.", ticker)
                    continue

                try:
                    # Calculate Technical Indicators
                    ind_df = self.indicator_engine.compute_all_indicators(df_daily)
                    latest_ind = ind_df.iloc[-1]
                    
                    # Compute Volume statistics
                    close = float(latest_ind["Close"])
                    volume = float(latest_ind["Volume"])
                    vol_5d = float(ind_df["Volume"].tail(5).mean())
                    vol_20d = float(ind_df["Volume"].tail(20).mean())
                    vol_ratio = float(volume / vol_20d) if vol_20d > 0 else 1.0

                    # Detect Technical Breakout Signals
                    price_above_ema20 = close > float(latest_ind["EMA_20"])
                    ema_cross = float(latest_ind["EMA_20"]) > float(latest_ind["EMA_50"])
                    bullish_macd = float(latest_ind["MACD_Hist"]) > 0
                    bullish_rsi = float(latest_ind["RSI_14"]) > 50
                    
                    # Higher High & Higher Low checks
                    higher_high = float(latest_ind["High"]) > float(ind_df["High"].iloc[-2])
                    higher_low = float(latest_ind["Low"]) > float(ind_df["Low"].iloc[-2])
                    
                    # Bollinger Band breakouts
                    resistance_breakout = close > float(latest_ind["BB_Upper"])
                    
                    # Strong Bullish Candle
                    body = abs(close - float(latest_ind["Open"]))
                    total_range = float(latest_ind["High"]) - float(latest_ind["Low"])
                    strong_bullish = (close > float(latest_ind["Open"])) and (body >= 0.6 * total_range) if total_range > 0 else False

                    # Momentum strength
                    momentum_strength = float(latest_ind["ADX_14"]) > 25

                    # Intraday VWAP checks
                    # VWAP is calculated on intraday 15m/5m frames
                    if not df_15m.empty:
                        df_15m_with_ind = self.indicator_engine.compute_all_indicators(df_15m)
                        latest_15m = df_15m_with_ind.iloc[-1]
                        vwap_val = float(latest_15m["VWAP"])
                        price_above_vwap = float(latest_15m["Close"]) > vwap_val
                    else:
                        vwap_val = close
                        price_above_vwap = True

                    indicator_metrics = {
                        "Close": close,
                        "EMA_20": float(latest_ind["EMA_20"]),
                        "EMA_50": float(latest_ind["EMA_50"]),
                        "EMA_200": float(latest_ind["EMA_200"]),
                        "VWAP": vwap_val,
                        "RSI_14": float(latest_ind["RSI_14"]),
                        "MACD": float(latest_ind["MACD"]),
                        "MACD_Signal": float(latest_ind["MACD_Signal"]),
                        "MACD_Hist": float(latest_ind["MACD_Hist"]),
                        "ATR_14": float(latest_ind["ATR_14"]),
                        "ADX_14": float(latest_ind["ADX_14"]),
                        "Supertrend": float(latest_ind["Supertrend"]),
                        "Supertrend_Direction": int(latest_ind["Supertrend_Direction"]),
                        "Volume_Ratio": vol_ratio,
                        "Signals": {
                            "price_above_ema20": bool(price_above_ema20),
                            "price_above_vwap": bool(price_above_vwap),
                            "ema_cross": bool(ema_cross),
                            "bullish_macd": bool(bullish_macd),
                            "bullish_rsi": bool(bullish_rsi),
                            "higher_high": bool(higher_high),
                            "higher_low": bool(higher_low),
                            "resistance_breakout": bool(resistance_breakout),
                            "strong_bullish": bool(strong_bullish),
                            "momentum_strength": bool(momentum_strength),
                        }
                    }

                    # Fetch Options details (OI, PCR, Max Pain, IV)
                    options_metrics = self.option_chain_analyzer.analyze_option_chain(ticker)

                    # Fetch overnight News details
                    news_metrics = self.news_sentiment_analyzer.analyze_overnight_news(ticker)

                    # Assess Risk level
                    risk_metrics = self.risk_engine.assess_risk(
                        df=df_daily,
                        atr=float(latest_ind["ATR_14"]),
                        iv=options_metrics["IV"],
                        news_sentiment=news_metrics["Sentiment"],
                        news_score=news_metrics["Score"]
                    )

                    # Compile Weighted score
                    scoring_profile = self.opportunity_ranker.calculate_scores(
                        ticker=ticker,
                        indicators=indicator_metrics,
                        options_data=options_metrics,
                        news_data=news_metrics,
                        risk_profile=risk_metrics
                    )

                    # Compile recommendation details
                    rec = self.opportunity_ranker.compile_recommendation(
                        ticker=ticker,
                        close=close,
                        direction=scoring_profile["Direction"],
                        atr=float(latest_ind["ATR_14"]),
                        final_score=scoring_profile["Final_Score"],
                        risk_profile=risk_metrics,
                        options_data=options_metrics,
                        news_data=news_metrics
                    )

                    candidates.append({
                        "ticker": ticker,
                        "indicators": indicator_metrics,
                        "options": options_metrics,
                        "news": news_metrics,
                        "risk": risk_metrics,
                        "score": scoring_profile["Final_Score"],
                        "recommendation": rec,
                        "scan_date": scan_date,
                    })

                except Exception as exc:
                    logger.error("Error processing opportunity evaluation for '%s': %s", ticker, str(exc), exc_info=True)

            # 4. Rank candidates and slice Top 10
            top10_candidates = self.opportunity_ranker.rank_opportunities(
                [c["recommendation"] for c in candidates]
            )

            # Find matching candidate details for database persistence
            top10_tickers = {c["Ticker"] for c in top10_candidates}
            top10_full = [c for c in candidates if c["ticker"] in top10_tickers]

            # 5. Persist to database
            db = self.db_session_factory()
            try:
                for item in top10_full:
                    scan_entry = DailyScan(
                        scan_date=item["scan_date"],
                        ticker=item["ticker"],
                        indicators=json.dumps(item["indicators"]),
                        news=json.dumps(item["news"]),
                        options=json.dumps(item["options"]),
                        score=item["score"],
                        recommendation=json.dumps(item["recommendation"])
                    )
                    db.add(scan_entry)
                db.commit()
                logger.info("Successfully persisted daily opportunity scans in database.")
            except Exception as exc:
                db.rollback()
                logger.error("Failed saving scans to database: %s", str(exc))
            finally:
                db.close()

            # 6. Save top 10 files for dashboard caching
            data_dir = Path(settings.BASE_DIR) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Export JSON
            json_file = data_dir / "morning_opportunities.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(top10_candidates, f, indent=4)
                
            # Export CSV
            csv_file = data_dir / "morning_opportunities.csv"
            pd.DataFrame(top10_candidates).to_csv(csv_file, index=False)

            duration = time.time() - start_time
            logger.info("🌅 Morning scan execution complete. Duration: %.2f seconds. Top 10 exported.", duration)
            return top10_candidates

        except Exception as exc:
            duration = time.time() - start_time
            logger.error("Morning scanner orchestration crashed after %.2f seconds: %s", duration, str(exc), exc_info=True)
            return []
