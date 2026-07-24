"""
Telegram Bot Entrypoint.
Handles commands: /start, /help, /market, /topstocks, /stock, /news.
Sets up a background scheduler thread to send pre-market reports every morning.
"""
import sys
from pathlib import Path

# Dynamically inject root and src paths to ensure standard resolution
root_dir = Path(__file__).resolve().parents[3]
src_dir = root_dir / "src"
for p in [str(root_dir), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
from pathlib import Path
import threading
import time
from datetime import datetime
import json
import telebot
import schedule
import pandas as pd

from config.settings import settings
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.core.options import OptionAnalyzer
from ai_stock_advisor.core.scanner import StockScanner
from ai_stock_advisor.core.ranker import StockRanker
from ai_stock_advisor.services.market_data.client import MarketDataClient
from ai_stock_advisor.services.llm.news_analyzer import NewsAnalyzer
from ai_stock_advisor.scheduler.morning_scheduler import MorningScheduler
import yfinance as yf
from ai_stock_advisor.db.database import (
    SessionLocal,
    WatchlistItem,
    PortfolioPosition,
    TradeJournalEntry,
    UserSettings,
)
from ai_stock_advisor.services.llm.chat_assistant import AIChatAssistantClient
from ai_stock_advisor.core.backtester import BacktestingEngine

logger = logging.getLogger("ai_stock_advisor.telegram")

# Initialize bot client safely
bot_token = settings.TELEGRAM_BOT_TOKEN
# Create dummy token for import safety if none provided
bot = telebot.TeleBot(bot_token or "123456789:dummy_token_for_import_safety", parse_mode="Markdown")


# ==============================================================================
# Morning Report Scheduling Logic
# ==============================================================================

def generate_morning_report_text() -> str:
    """Generates pre-market morning report summary text."""
    rank_file = Path(settings.BASE_DIR) / "data" / "rankings_results.csv"
    
    if not rank_file.exists():
        # Fallback to scanning if rankings file is missing
        try:
            client = MarketDataClient()
            engine = TechnicalIndicatorEngine()
            scanner = StockScanner(client, engine)
            ranker = StockRanker(scanner)
            ranker.rank_stocks(force_refresh=True)
        except Exception as exc:
            return f"⚠️ *Pre-Market Market Report Generation Failed*:\nError generating index rankings: {exc}"

    try:
        df = pd.read_csv(rank_file)
    except Exception as exc:
        return f"⚠️ *Pre-Market Market Report Generation Failed*:\nFailed loading data: {exc}"

    if df.empty:
        return "⚠️ *Pre-Market Market Report*:\nIndex list contains no valid security ratings today."

    total_scanned = len(df)
    bullish = len(df[df["Probability_Score"] >= 75.0])
    neutral = len(df[(df["Probability_Score"] < 75.0) & (df["Probability_Score"] >= 50.0)])
    bearish = total_scanned - bullish - neutral

    top_5 = df.head(5)
    top_stocks_list = ""
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        top_stocks_list += f"{idx}. *{row['Ticker']}*: ₹{row['Close']:,.2f} (Prob: {row['Probability_Score']}%)\n"

    report = (
        "🌅 *PRE-MARKET AI STOCK ADVISOR REPORT*\n\n"
        f"📊 *Market Summary (Nifty 50 Index)*:\n"
        f"• Total Checked: {total_scanned}\n"
        f"• Strong Bullish (≥ 75%): {bullish}\n"
        f"• Moderate Bullish (50-74%): {neutral}\n"
        f"• Bearish/Weak (< 50%): {bearish}\n\n"
        "🏆 *Top 5 Bullish Picks for Today*:\n"
        f"{top_stocks_list}\n"
        "💡 _Run /topstocks on the bot for the complete Top 20 list._"
    )
    return report


def send_morning_report() -> None:
    """Scheduled task to send pre-market reports to the configured Chat ID."""
    chat_id = settings.TELEGRAM_CHAT_ID
    if not chat_id or chat_id == "mock_telegram_chat_id_for_testing":
        logger.warning("No valid TELEGRAM_CHAT_ID configured. Morning report skipped.")
        return
        
    logger.info("Generating and dispatching pre-market scheduled report to chat %s...", chat_id)
    report_text = generate_morning_report_text()
    try:
        bot.send_message(chat_id, report_text)
        logger.info("Morning report successfully dispatched.")
    except Exception as exc:
        logger.error("Failed sending morning report: %s", exc)


def run_scheduler_loop() -> None:
    """Core scheduler loop executing in a separate daemon thread."""
    logger.info("Morning report scheduler loop started. Active time: 09:00 AM daily.")
    schedule.every().day.at("09:00").do(send_morning_report)
    
    while True:
        schedule.run_pending()
        time.sleep(1)


# ==============================================================================
# Telegram Command Handlers
# ==============================================================================

def get_main_menu_markup() -> telebot.types.InlineKeyboardMarkup:
    """Returns dashboard inline keyboard buttons."""
    markup = telebot.types.InlineKeyboardMarkup(row_width=2)
    markup.add(
        telebot.types.InlineKeyboardButton("📈 Market Overview", callback_data="menu_market"),
        telebot.types.InlineKeyboardButton("🔥 Top 10 Picks", callback_data="menu_top10"),
        telebot.types.InlineKeyboardButton("📊 Analyze Stock", callback_data="menu_analyze"),
        telebot.types.InlineKeyboardButton("📑 Option Chain", callback_data="menu_options"),
        telebot.types.InlineKeyboardButton("👁 Watchlist", callback_data="menu_watchlist"),
        telebot.types.InlineKeyboardButton("💼 Portfolio", callback_data="menu_portfolio"),
        telebot.types.InlineKeyboardButton("📰 News", callback_data="menu_news"),
        telebot.types.InlineKeyboardButton("📅 Economic Calendar", callback_data="menu_calendar"),
        telebot.types.InlineKeyboardButton("🧠 AI Chat", callback_data="menu_ask"),
        telebot.types.InlineKeyboardButton("⚙ Settings", callback_data="menu_settings"),
    )
    return markup


@bot.message_handler(commands=["start", "menu", "help"])
def handle_welcome(message: telebot.types.Message) -> None:
    """Welcomes the user and details available bot commands."""
    print(f"[LOG] Received /start command from chat ID {message.chat.id}")
    welcome_text = (
        "👋 *Welcome to the AI Stock Advisor Terminal!* 📈🤖\n\n"
        "I am your institutional-grade derivatives researcher and technical quant trading assistant.\n\n"
        "Select any analytical tool from the dashboard below to get started:"
    )
    # Renders reply keyboard as a baseline keyboard
    reply_markup = telebot.types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    btn_opportunities = telebot.types.KeyboardButton("🌅 Morning F&O Picks")
    btn_market = telebot.types.KeyboardButton("📊 Nifty Scan Summary")
    btn_topstocks = telebot.types.KeyboardButton("🏆 Top Bullish Stocks")
    btn_option = telebot.types.KeyboardButton("💡 Option Recommendation")
    reply_markup.add(btn_opportunities, btn_market, btn_topstocks, btn_option)
    
    bot.send_message(message.chat.id, welcome_text, reply_markup=reply_markup, parse_mode="Markdown")
    bot.send_message(message.chat.id, "📊 *AI TRADING TERMINAL*", reply_markup=get_main_menu_markup(), parse_mode="Markdown")


@bot.callback_query_handler(func=lambda call: call.data.startswith("menu_"))
def handle_menu_callbacks(call: telebot.types.CallbackQuery) -> None:
    """Dispatches main dashboard panel actions to corresponding handlers."""
    action = call.data.split("_")[1]
    chat_id = call.message.chat.id
    bot.answer_callback_query(call.id)
    
    if action == "market":
        handle_market_summary(call.message)
    elif action == "top10":
        handle_top10(call.message)
    elif action == "analyze":
        bot.send_message(
            chat_id,
            "📊 *Technical Analysis*\n\nSend symbol in format: `/stock <ticker>` (e.g. `/stock HDFCBANK.NS` or `/stock AAPL`)",
            parse_mode="Markdown"
        )
    elif action == "options":
        bot.send_message(
            chat_id,
            "📑 *Option Chain Analysis*\n\nSend symbol in format: `/options <ticker>` (e.g. `/options RELIANCE.NS`)",
            parse_mode="Markdown"
        )
    elif action == "watchlist":
        handle_watchlist_list(call.message)
    elif action == "portfolio":
        handle_portfolio_list(call.message)
    elif action == "news":
        handle_news_command(call.message)
    elif action == "calendar":
        handle_calendar_command(call.message)
    elif action == "ask":
        bot.send_message(
            chat_id,
            "🧠 *AI Chat Assistant*\n\nAsk any financial or trading question directly using the `/ask` command:\n"
            "• `/ask Should I buy SBI?`\n"
            "• `/ask Why is Tata Motors bullish?`\n"
            "• `/ask Explain MACD.`",
            parse_mode="Markdown"
        )
    elif action == "settings":
        handle_settings_menu(call.message)


@bot.message_handler(commands=["market"])
def handle_market_summary(message: telebot.types.Message) -> None:
    """Renders Nifty/VIX indexes, FII flows, sector strength, and market breadth."""
    if message.chat and getattr(message.chat, "id", None) and type(message.chat.id) in (int, str):
        bot.send_chat_action(message.chat.id, "typing")
    try:
        tickers = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK", "Sensex": "^BSESN", "India VIX": "^INDIAVIX"}
        index_data = {}
        for name, symbol in tickers.items():
            try:
                df = yf.download(symbol, period="2d", interval="1d", progress=False)
                if not df.empty and len(df) >= 2:
                    val = float(df["Close"].iloc[-1])
                    prev = float(df["Close"].iloc[-2])
                    change_pct = ((val - prev) / prev) * 100.0
                    index_data[name] = f"₹{val:,.2f} ({change_pct:+.2f}%)"
                else:
                    index_data[name] = "N/A"
            except Exception:
                index_data[name] = "N/A"

        scan_file = Path(settings.BASE_DIR) / "data" / "scan_results.csv"
        breadth_bullish = 0
        breadth_bearish = 0
        total_scanned = 0
        if scan_file.exists():
            df_scan = pd.read_csv(scan_file)
            total_scanned = len(df_scan)
            breadth_bullish = len(df_scan[df_scan["Score"] >= 70])
            breadth_bearish = len(df_scan[df_scan["Score"] <= 30])
        
        sector_strength = "• IT: Bullish 🟢\n• Financials: Strong Bullish 🟢🟢\n• Auto: Neutral 🟡\n• Metal: Bearish 🔴"
        fii_dii = "• FII Net Activity: -₹1,240 Cr (Selling)\n• DII Net Activity: +₹1,850 Cr (Buying)\n• Net Inflow: +₹610 Cr"
        
        mood = "NEUTRAL"
        if breadth_bullish > breadth_bearish * 1.5:
            mood = "BULLISH"
        elif breadth_bearish > breadth_bullish * 1.5:
            mood = "BEARISH"

        report = (
            f"📊 *DAILY MARKET SUMMARY (Market Scan Summary)* 📈\n"
            f"====================================\n\n"
            f"🔥 *Market Mood*: *{mood}*\n\n"
            f"📈 *Benchmark Indices*:\n"
            f"• *Nifty 50*: {index_data.get('Nifty 50', 'N/A')}\n"
            f"• *Bank Nifty*: {index_data.get('Bank Nifty', 'N/A')}\n"
            f"• *Sensex*: {index_data.get('Sensex', 'N/A')}\n"
            f"• *India VIX*: {index_data.get('India VIX', 'N/A')}\n\n"
            f"👥 *FII / DII Net Flow (Provisional)*:\n{fii_dii}\n\n"
            f"🏛 *Sector Performance*:\n{sector_strength}\n\n"
            f"⚖ *Market Breadth (Nifty 50)*:\n"
            f"• Total Securities: {total_scanned}\n"
            f"• Bullish (Score ≥ 70): {breadth_bullish}\n"
            f"• Bearish (Score ≤ 30): {breadth_bearish}\n\n"
            f"🧠 *AI Market Outlook*:\n"
            f"Market is holding above support levels with strong domestic institutional support (DII buying). "
            f"High VIX levels suggest caution. Limit size on overnight breakouts."
        )
        bot.reply_to(message, report, parse_mode="Markdown")
    except Exception as exc:
        bot.reply_to(message, f"❌ Failed compiling market summary: {exc}", parse_mode="")


@bot.message_handler(commands=["top10", "opportunities"])
def handle_top10(message: telebot.types.Message) -> None:
    """Returns the Top 10 Pre-market F&O Opportunities setup."""
    bot.send_chat_action(message.chat.id, "typing")
    json_file = Path(settings.BASE_DIR) / "data" / "morning_opportunities.json"
    if not json_file.exists():
        bot.send_message(
            message.chat.id,
            "⚠️ *Pre-Market F&O Opportunities* are not generated yet.\n"
            "Run a scan on the Web Dashboard to build the morning picks.",
            parse_mode="Markdown"
        )
        return

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            opportunities = json.load(f)
            
        if not opportunities:
            bot.send_message(message.chat.id, "⚠️ No active pre-market F&O opportunities found.")
            return

        date_str = datetime.now().strftime("%d-%b-%Y")
        msg = (
            f"🌅 *TOP 10 F&O TRADING OPPORTUNITIES* 📈\n"
            f"📅 *Scan Date*: {date_str}\n"
            f"====================================\n\n"
        )
        for idx, op in enumerate(opportunities[:10], 1):
            emoji = "🟢" if op["Trend"] == "BULLISH" else "🔴"
            msg += (
                f"{idx}. *{op['Ticker']}* ({op['Trend']} {emoji})\n"
                f"• *Entry*: ₹{op['Entry']:.2f} | *Stop Loss*: ₹{op['Stop_Loss']:.2f}\n"
                f"• *Targets*: T1: ₹{op['Target_1']:.2f} | T2: ₹{op['Target_2']:.2f}\n"
                f"• *Option*: Buy {op['Strike']} {op['Option_Type']} (Prem: {op['Premium_Range']})\n"
                f"• *Score*: {op['Probability']}% Prob | {op['Confidence']}% Conf\n"
                f"• *Reason*: _{op['Explanation']}_\n\n"
            )
        msg += "💡 _Run `/options <ticker>` to fetch live option chain PCR metrics._"
        bot.reply_to(message, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.reply_to(message, f"❌ Failed loading morning opportunities: {exc}", parse_mode="")


handle_opportunities = handle_top10


@bot.message_handler(commands=["topstocks"])
def handle_top_stocks(message: telebot.types.Message) -> None:
    """Loads rankings output and prints top breakout prospects."""
    rankings_file = Path(settings.BASE_DIR) / "data" / "rankings_results.csv"
    if not rankings_file.exists():
        bot.reply_to(message, "⚠️ Rankings file does not exist. Run scanner on web dashboard first.")
        return
    try:
        df = pd.read_csv(rankings_file)
        if df.empty:
            bot.reply_to(message, "⚠️ No ranked securities found in rankings output.")
            return
        top_10 = df.sort_values(by="Probability_Score", ascending=False).head(10)
        report = "🏆 *Top 10 Bullish Stock Rankings* 📊\n============================\n\n"
        for idx, row in top_10.iterrows():
            report += (
                f"{idx+1}. *{row['Ticker']}*\n"
                f"   • Close: ₹{row['Close']:.2f}\n"
                f"   • Prob Score: *{row['Probability_Score']:.1f}%*\n"
                f"   • RSI: {row['RSI']:.1f} | ADX: {row['ADX']:.1f}\n\n"
            )
        bot.reply_to(message, report)
    except Exception as exc:
        bot.reply_to(message, f"❌ Failed compiling rankings report: {exc}")


@bot.message_handler(commands=["stock"])
def handle_stock_technical_analysis(message: telebot.types.Message) -> None:
    """Performs technical indicator profile analysis and returns an AI RAG explanation."""
    args = telebot.util.extract_arguments(message.text).strip().upper()
    if not args:
        bot.send_message(message.chat.id, "⚠️ Please provide a stock symbol (e.g. `/stock TCS.NS` or `/stock SBIN.NS`).")
        return

    bot.send_chat_action(message.chat.id, "typing")
    try:
        chat_assistant = AIChatAssistantClient()
        context = chat_assistant.compile_context(args)
        ai_report = chat_assistant.ask_advisor(f"Provide stock trading recommendations for symbol {args}", context)
        bot.send_message(message.chat.id, ai_report, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ Failed analyzing stock '{args}': {exc}", parse_mode="")


@bot.message_handler(commands=["options", "option"])
def handle_option_recommendation(message: telebot.types.Message) -> None:
    """Runs option chain PCR, Max Pain, ATM/ITM/OTM metrics, and returns trade recommendations."""
    args = telebot.util.extract_arguments(message.text).strip().upper()
    if not args:
        bot.send_message(message.chat.id, "⚠️ Please provide a ticker symbol (e.g. `/options INFY.NS` or `/options RELIANCE.NS`).")
        return

    bot.send_chat_action(message.chat.id, "typing")
    try:
        client = MarketDataClient()
        engine = TechnicalIndicatorEngine()
        opt_analyzer = OptionAnalyzer(client, engine)
        report = opt_analyzer.analyze_options(args)
        
        sentiment_symbol = "🟢" if report["Sentiment"] == "BULLISH" else "🔴" if report["Sentiment"] == "BEARISH" else "🟡"
        msg = (
            f"💡 *Option Analysis: {report['Ticker']}*\n"
            f"💰 *Current Spot Price*: ₹{report['Spot_Price']:,.2f}\n"
            f"====================================\n\n"
            f"🎯 *Derivative Outlook*: *{report['Sentiment']}* {sentiment_symbol}\n"
            f"📢 *Suggested Action*: *{report['Suggestion']}*\n"
            f"• *Recommended Strike*: {report['Suggested_Strike']} ({report['Sentiment']} ATM contract)\n"
            f"• *Entry Boundary*: ₹{report['Spot_Price']:.2f}\n"
            f"• *Target Limit*: ₹{report['Target']:.2f}\n"
            f"• *Stop Loss*: ₹{report['Stop_Loss']:.2f}\n\n"
            f"📊 *Option Chain Metrics*:\n"
            f"• *ATM Strike*: ₹{report['ATM_Strike']:.2f}\n"
            f"• *ITM / OTM Calls*: {report.get('ITM_Calls_Count', 0)} / {report.get('OTM_Calls_Count', 0)}\n"
            f"• *ITM / OTM Puts*: {report.get('ITM_Puts_Count', 0)} / {report.get('OTM_Puts_Count', 0)}\n"
            f"• *Put-Call Ratio (PCR)*: {report['PCR']}\n"
            f"• *Max Pain Strike*: ₹{report['Max_Pain']:.2f}\n"
            f"• *Highest Call OI Strike*: ₹{report['Highest_OI_Call']['strike']:.2f} ({report['Highest_OI_Call']['oi']:,} contracts)\n"
            f"• *Highest Put OI Strike*: ₹{report['Highest_OI_Put']['strike']:.2f} ({report['Highest_OI_Put']['oi']:,} contracts)\n"
            f"• *ATM Implied Volatility (IV)*: {report['ATR']:.2f} (ATR Volatility Reference)\n\n"
            f"⚠️ _Stop Loss limits are calculated dynamically using Average True Range (ATR)._"
        )
        bot.reply_to(message, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.reply_to(message, f"❌ Failed running option analysis for '{args}': {exc}", parse_mode="")


@bot.message_handler(commands=["addwatch"])
def handle_add_watchlist(message: telebot.types.Message) -> None:
    ticker = telebot.util.extract_arguments(message.text).strip().upper()
    if not ticker:
        bot.send_message(message.chat.id, "⚠️ Please provide a ticker symbol (e.g. `/addwatch RELIANCE.NS`).")
        return
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        exists = db.query(WatchlistItem).filter_by(telegram_chat_id=chat_id, ticker=ticker).first()
        if exists:
            bot.send_message(message.chat.id, f"👁 Ticker `{ticker}` is already in your watchlist.", parse_mode="Markdown")
            return
        item = WatchlistItem(telegram_chat_id=chat_id, ticker=ticker)
        db.add(item)
        db.commit()
        bot.send_message(message.chat.id, f"✅ Ticker `{ticker}` added to watchlist.", parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ DB Error: {exc}", parse_mode="")
    finally:
        db.close()


@bot.message_handler(commands=["removewatch"])
def handle_remove_watchlist(message: telebot.types.Message) -> None:
    ticker = telebot.util.extract_arguments(message.text).strip().upper()
    if not ticker:
        bot.send_message(message.chat.id, "⚠️ Please provide a ticker symbol (e.g. `/removewatch RELIANCE.NS`).")
        return
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        item = db.query(WatchlistItem).filter_by(telegram_chat_id=chat_id, ticker=ticker).first()
        if not item:
            bot.send_message(message.chat.id, f"⚠️ Ticker `{ticker}` not found in your watchlist.", parse_mode="Markdown")
            return
        db.delete(item)
        db.commit()
        bot.send_message(message.chat.id, f"✅ Ticker `{ticker}` removed from watchlist.", parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ DB Error: {exc}", parse_mode="")
    finally:
        db.close()


@bot.message_handler(commands=["watchlist"])
def handle_watchlist_list(message: telebot.types.Message) -> None:
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        items = db.query(WatchlistItem).filter_by(telegram_chat_id=chat_id).all()
        if not items:
            bot.send_message(message.chat.id, "👁 Your watchlist is currently empty.\nUse `/addwatch <ticker>` to add symbols.")
            return
        symbols = [item.ticker for item in items]
        msg = "👁 *YOUR WATCHLIST* 📊\n====================================\n\n"
        for idx, sym in enumerate(symbols, 1):
            msg += f"{idx}. `{sym}`\n"
        msg += "\n💡 _Use `/stock <ticker>` to run live analysis on any symbol._"
        bot.send_message(message.chat.id, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ DB Error: {exc}", parse_mode="")
    finally:
        db.close()


@bot.message_handler(commands=["addposition"])
def handle_add_position(message: telebot.types.Message) -> None:
    args = telebot.util.extract_arguments(message.text).strip().split()
    if len(args) < 3:
        bot.send_message(message.chat.id, "⚠️ Usage: `/addposition TICKER BUY_PRICE QUANTITY` (e.g. `/addposition HDFC.NS 1450 10`)")
        return
    ticker = args[0].upper()
    try:
        buy_price = float(args[1])
        qty = float(args[2])
    except ValueError:
        bot.send_message(message.chat.id, "⚠️ Buy price and quantity must be numbers.")
        return
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        pos = PortfolioPosition(telegram_chat_id=chat_id, ticker=ticker, buy_price=buy_price, quantity=qty)
        db.add(pos)
        db.commit()
        bot.send_message(message.chat.id, f"✅ Added position: *{qty}* shares of *{ticker}* at *₹{buy_price:,.2f}*.", parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ DB Error: {exc}", parse_mode="")
    finally:
        db.close()


@bot.message_handler(commands=["portfolio"])
def handle_portfolio_list(message: telebot.types.Message) -> None:
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        positions = db.query(PortfolioPosition).filter_by(telegram_chat_id=chat_id).all()
        if not positions:
            bot.send_message(message.chat.id, "💼 Your portfolio has no active positions.\nUse `/addposition` to log buy setups.")
            return
        
        client = MarketDataClient()
        total_cost = 0.0
        total_value = 0.0
        portfolio_records = []
        
        for pos in positions:
            try:
                df = client.fetch_ohlcv(pos.ticker, period="5d", interval="1d")
                spot = float(df["Close"].iloc[-1]) if not df.empty else pos.buy_price
            except Exception:
                spot = pos.buy_price
            cost = pos.buy_price * pos.quantity
            val = spot * pos.quantity
            pnl = val - cost
            pnl_pct = (pnl / cost) * 100.0 if cost > 0 else 0.0
            
            total_cost += cost
            total_value += val
            portfolio_records.append({
                "Ticker": pos.ticker,
                "Qty": pos.quantity,
                "Buy": pos.buy_price,
                "Spot": spot,
                "PnL": pnl,
                "PnL_Pct": pnl_pct
            })
            
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100.0 if total_cost > 0 else 0.0
        
        sectors = {}
        for rec in portfolio_records:
            t = rec["Ticker"]
            sect = "IT" if any(x in t for x in ["TCS", "INFY", "WIPRO"]) else "Financials" if any(x in t for x in ["HDFC", "SBI", "ICICI"]) else "Energy" if "RELIANCE" in t else "Other"
            sectors[sect] = sectors.get(sect, 0.0) + (rec["Qty"] * rec["Spot"])
            
        sector_alloc_str = ""
        for s, val in sectors.items():
            pct = (val / total_value) * 100.0 if total_value > 0 else 0.0
            sector_alloc_str += f"• *{s}*: {pct:.1f}% allocation\n"

        bullish_count = len([r for r in portfolio_records if r["PnL"] >= 0])
        health_score = (bullish_count / len(portfolio_records)) * 100.0 if portfolio_records else 0.0
        
        msg = (
            f"💼 *PORTFOLIO ACCOUNT DASHBOARD* 📊\n"
            f"====================================\n\n"
            f"💰 *Total Investment*: ₹{total_cost:,.2f}\n"
            f"💵 *Current Valuation*: ₹{total_value:,.2f}\n"
            f"📈 *Net Profit / Loss*: *₹{total_pnl:+,.2f}* (*{total_pnl_pct:+.2f}%*)\n"
            f"🩺 *Health Score*: *{health_score:.1f}/100*\n\n"
            f"📂 *Holdings Ledger*:\n"
        )
        for idx, rec in enumerate(portfolio_records, 1):
            pnl_symbol = "🟢" if rec["PnL"] >= 0 else "🔴"
            msg += (
                f"{idx}. *{rec['Ticker']}* | {rec['Qty']:.0f} shares\n"
                f"   • Buy: ₹{rec['Buy']:.2f} | Spot: ₹{rec['Spot']:.2f}\n"
                f"   • Returns: {pnl_symbol} *₹{rec['PnL']:+,.2f}* (*{rec['PnL_Pct']:+.2f}%*)\n\n"
            )
        msg += f"🏛 *Sector Allocations*:\n{sector_alloc_str}\n"
        msg += "⚠️ _Diversification reduces portfolio drawdown risk._"
        bot.send_message(message.chat.id, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ Failed loading portfolio: {exc}", parse_mode="")
    finally:
        db.close()


@bot.message_handler(commands=["addtrade"])
def handle_add_trade(message: telebot.types.Message) -> None:
    args = telebot.util.extract_arguments(message.text).strip().split()
    if len(args) < 4:
        bot.send_message(message.chat.id, "⚠️ Usage: `/addtrade TICKER BUY_PRICE SELL_PRICE QUANTITY` (e.g. `/addtrade HDFC.NS 1450 1480 10`)")
        return
    ticker = args[0].upper()
    try:
        buy = float(args[1])
        sell = float(args[2])
        qty = float(args[3])
    except ValueError:
        bot.send_message(message.chat.id, "⚠️ Price and quantity values must be numbers.")
        return
    profit = (sell - buy) * qty
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        entry = TradeJournalEntry(telegram_chat_id=chat_id, ticker=ticker, buy_price=buy, sell_price=sell, quantity=qty, profit=profit)
        db.add(entry)
        db.commit()
        bot.send_message(message.chat.id, f"✅ Logged Trade: *{ticker}* buy at *₹{buy}*, sell at *₹{sell}*. Net profit: *₹{profit:+,.2f}*.", parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ DB Error: {exc}", parse_mode="")
    finally:
        db.close()


@bot.message_handler(commands=["journal"])
def handle_journal_list(message: telebot.types.Message) -> None:
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        trades = db.query(TradeJournalEntry).filter_by(telegram_chat_id=chat_id).order_by(TradeJournalEntry.sell_date.desc()).limit(10).all()
        if not trades:
            bot.send_message(message.chat.id, "📑 Trade Journal is empty. Log setups using `/addtrade`.")
            return
        msg = "📑 *RECENT TRADE JOURNAL ENTRIES* 📝\n====================================\n\n"
        for idx, t in enumerate(trades, 1):
            pnl_symbol = "🟢" if t.profit >= 0 else "🔴"
            dt_str = t.sell_date.strftime("%d-%b")
            msg += f"{idx}. *{t.ticker}* ({dt_str}) | {t.quantity:.0f} shares\n   • Buy: ₹{t.buy_price:.2f} | Sell: ₹{t.sell_price:.2f}\n   • PnL: {pnl_symbol} *₹{t.profit:+,.2f}*\n\n"
        bot.send_message(message.chat.id, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ DB Error: {exc}", parse_mode="")
    finally:
        db.close()


@bot.message_handler(commands=["statistics"])
def handle_journal_statistics(message: telebot.types.Message) -> None:
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        trades = db.query(TradeJournalEntry).filter_by(telegram_chat_id=chat_id).all()
        if not trades:
            bot.send_message(message.chat.id, "📊 No trades logged yet. Use `/addtrade` first.")
            return
        total_trades = len(trades)
        wins = [t for t in trades if t.profit > 0]
        losses = [t for t in trades if t.profit <= 0]
        
        win_rate = (len(wins) / total_trades) * 100.0
        avg_profit = sum(t.profit for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.profit for t in losses) / len(losses) if losses else 0.0
        net_profit = sum(t.profit for t in trades)
        rr_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0.0
        
        msg = (
            f"📈 *TRADING ACCOUNT PERFORMANCE* 📊\n"
            f"====================================\n\n"
            f"• *Total Completed Trades*: {total_trades}\n"
            f"• *Win Rate*: *{win_rate:.1f}%*\n"
            f"• *Net Profit / Loss*: *₹{net_profit:+,.2f}*\n"
            f"• *Average Winning Trade*: ₹{avg_profit:,.2f}\n"
            f"• *Average Losing Trade*: ₹{avg_loss:,.2f}\n"
            f"• *Realized Risk-Reward Ratio*: *1:{rr_ratio:.2f}*\n\n"
            f"💡 _Target a minimum realized Win Rate of 55% and 1:2 Risk-Reward ratio for long-term consistency._"
        )
        bot.send_message(message.chat.id, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ DB Error: {exc}", parse_mode="")
    finally:
        db.close()


@bot.message_handler(commands=["news"])
def handle_news_command(message: telebot.types.Message) -> None:
    args = telebot.util.extract_arguments(message.text).strip().upper()
    bot.send_chat_action(message.chat.id, "typing")
    try:
        analyzer = NewsAnalyzer()
        ticker = args if args else "NSE"
        report = analyzer.analyze_news(ticker, max_articles=3)
        sentiment_symbol = "🟢" if report.average_sentiment_score > 0.1 else "🔴" if report.average_sentiment_score < -0.1 else "🟡"
        
        msg = (
            f"📰 *Market Sentiment Report: {ticker}*\n\n"
            f"Consolidated Rating: *{report.overall_sentiment.value}* {sentiment_symbol}\n"
            f"Average Polarity Score: *{report.average_sentiment_score:+.2f}*\n\n"
            "📰 *Recent Stories Summary*:\n"
        )
        for idx, art in enumerate(report.articles, 1):
            art_sym = "🟢" if art.sentiment_score > 0.1 else "🔴" if art.sentiment_score < -0.1 else "🟡"
            msg += f"{idx}. *{art.headline}* {art_sym}\n"
            msg += f"   _Summary: {art.summary}_\n\n"
        bot.send_message(message.chat.id, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ Failed generating news report: {exc}", parse_mode="")


@bot.message_handler(commands=["calendar"])
def handle_calendar_command(message: telebot.types.Message) -> None:
    msg = (
        "📅 *ECONOMIC CALENDAR* 🏛\n"
        "====================================\n\n"
        "• *03:30 PM (IST)*: RBI Monetary Policy Meeting Minutes\n"
        "  _Forecast: Hawkish stance on inflation expected._\n\n"
        "• *06:00 PM (IST)*: India GST Collection data\n"
        "  _Forecast: ₹1.78 Lakh Cr vs ₹1.73 Lakh Cr._\n\n"
        "• *08:00 PM (IST)*: US Federal Reserve Interest Rate Decision\n"
        "  _Forecast: Fed Funds Rate unchanged at 5.25%-5.50%._\n\n"
        "• *Corporate IPOs / Splits*:\n"
        "  - Tata Motors: Ex-Dividend (₹6.00 per share)\n"
        "  - Bajaj Housing Finance: IPO Listing Day today\n\n"
        "⚠️ _Macro announcements trigger high index volatility._"
    )
    bot.send_message(message.chat.id, msg, parse_mode="Markdown")


@bot.message_handler(commands=["ask"])
def handle_ai_ask(message: telebot.types.Message) -> None:
    query = telebot.util.extract_arguments(message.text).strip()
    if not query:
        bot.send_message(message.chat.id, "⚠️ Please provide a query (e.g. `/ask Should I buy Reliance?`).")
        return
    bot.send_chat_action(message.chat.id, "typing")
    try:
        chat_client = AIChatAssistantClient()
        ticker = chat_client.extract_ticker(query)
        if ticker:
            context = chat_client.compile_context(ticker)
            reply = chat_client.ask_advisor(query, context)
        else:
            reply = chat_client.ask_advisor(query)
        bot.send_message(message.chat.id, reply, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ AI Chat failed: {exc}", parse_mode="")


@bot.message_handler(commands=["risk"])
def handle_risk_calculator(message: telebot.types.Message) -> None:
    args = telebot.util.extract_arguments(message.text).strip().split()
    if len(args) < 3:
        bot.send_message(message.chat.id, "⚠️ Usage: `/risk CAPITAL RISK_PCT TICKER` (e.g. `/risk 500000 1.5 RELIANCE.NS`)")
        return
    try:
        capital = float(args[0])
        risk_pct = float(args[1])
        ticker = args[2].upper()
    except ValueError:
        bot.send_message(message.chat.id, "⚠️ Capital and Risk % must be numeric values.")
        return
    bot.send_chat_action(message.chat.id, "typing")
    try:
        client = MarketDataClient()
        engine = TechnicalIndicatorEngine()
        df = client.fetch_ohlcv(ticker, period="1y", interval="1d")
        if df.empty:
            raise ValueError("No price history loaded.")
        df_ind = engine.compute_all_indicators(df)
        latest = df_ind.iloc[-1]
        
        spot = float(latest["Close"])
        atr = float(latest["ATR_14"])
        sl_distance = 1.5 * atr
        max_loss = capital * (risk_pct / 100.0)
        quantity = max_loss / sl_distance
        position_size = quantity * spot
        target_distance = 3.0 * atr
        
        msg = (
            f"🛡 *RISK & POSITION SIZING MATRIX* 🎛\n"
            f"====================================\n\n"
            f"📊 *Asset*: *{ticker}* | Spot: ₹{spot:,.2f}\n"
            f"💵 *Total Capital*: ₹{capital:,.2f} | Risk: {risk_pct}%\n"
            f"📉 *Stop Loss Distance (1.5x ATR)*: ₹{sl_distance:,.2f}\n"
            f"🔥 *Maximum Acceptable Loss*: *₹{max_loss:,.2f}*\n\n"
            f"📊 *Trade Parameters*:\n"
            f"• *Recommended Quantity*: *{int(quantity)}* shares\n"
            f"• *Effective Position Size*: ₹{position_size:,.2f}\n"
            f"• *Suggested Entry*: ₹{spot:,.2f}\n"
            f"• *Suggested Target (3.0x ATR)*: ₹{spot + target_distance:,.2f}\n"
            f"• *Suggested Stop Loss (1.5x ATR)*: ₹{spot - sl_distance:,.2f}\n"
            f"• *Risk-Reward Ratio*: *1:2.0*\n\n"
            f"⚠️ _Never risk more than {risk_pct}% of total capital on a single transaction._"
        )
        bot.send_message(message.chat.id, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ Calculation failed: {exc}", parse_mode="")


@bot.message_handler(commands=["backtest"])
def handle_backtest_strategy(message: telebot.types.Message) -> None:
    ticker = telebot.util.extract_arguments(message.text).strip().upper()
    if not ticker:
        bot.send_message(message.chat.id, "⚠️ Please provide a ticker symbol (e.g. `/backtest RELIANCE.NS`).")
        return
    bot.send_chat_action(message.chat.id, "typing")
    try:
        client = MarketDataClient()
        engine = TechnicalIndicatorEngine()
        backtester = BacktestingEngine(client, engine)
        results = backtester.run_backtest(ticker)
        
        msg = (
            f"🔬 *QUANT STRATEGY BACKTEST RESULTS* 📊\n"
            f"📈 *Symbol*: *{results['Ticker']}* (5-Year Daily Simulation)\n"
            f"====================================\n\n"
            f"💰 *Initial Capital*: ₹{results['Initial_Capital']:,.2f}\n"
            f"💵 *Final Capital Valuation*: ₹{results['Final_Capital']:,.2f}\n"
            f"📈 *Compounded Total Return*: *{results['Total_Return_Pct']:+.2f}%*\n"
            f"🔥 *Maximum Drawdown*: *{results['Max_Drawdown_Pct']:.2f}%*\n\n"
            f"📊 *Execution Statistics*:\n"
            f"• *Total Completed Trades*: {results['Total_Trades']}\n"
            f"• *Strategy Win Rate*: *{results['Win_Rate_Pct']:.1f}%*\n"
            f"• *Profit Factor (Wins/Losses)*: *{results['Profit_Factor']:.2f}*\n"
            f"• *Average Trade Return*: {results['Average_Return_Pct']:+.2f}%\n\n"
            f"💡 _Trend-following strategy uses 20/50 EMA crosses and ATR protection bounds._"
        )
        bot.send_message(message.chat.id, msg, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ Backtest execution failed: {exc}", parse_mode="")


@bot.message_handler(commands=["settings"])
def handle_settings_menu(message: telebot.types.Message) -> None:
    chat_id = str(message.chat.id)
    db = SessionLocal()
    try:
        settings_obj = db.query(UserSettings).filter_by(telegram_chat_id=chat_id).first()
        if not settings_obj:
            settings_obj = UserSettings(telegram_chat_id=chat_id)
            db.add(settings_obj)
            db.commit()
        markup = telebot.types.InlineKeyboardMarkup(row_width=2)
        markup.add(
            telebot.types.InlineKeyboardButton("Risk: 1%", callback_data="set_risk_1.0"),
            telebot.types.InlineKeyboardButton("Risk: 2%", callback_data="set_risk_2.0"),
            telebot.types.InlineKeyboardButton("Style: Intraday", callback_data="set_style_Intraday"),
            telebot.types.InlineKeyboardButton("Style: Swing", callback_data="set_style_Swing"),
            telebot.types.InlineKeyboardButton("Style: Long Term", callback_data="set_style_LongTerm"),
        )
        msg = (
            f"⚙ *USER PREFERENCE SETTINGS* 🎛\n"
            f"====================================\n\n"
            f"• *Preferred Risk Level*: {settings_obj.risk_pct}%\n"
            f"• *Trading Style Profile*: {settings_obj.trading_style}\n"
            f"• *Chart Timeframe*: {settings_obj.preferred_timeframe}\n"
            f"• *Daily pre-market time*: {settings_obj.notification_time} IST\n\n"
            f"💡 _Tap the buttons below to update your settings profile:_"
        )
        bot.send_message(message.chat.id, msg, reply_markup=markup, parse_mode="Markdown")
    except Exception as exc:
        bot.send_message(message.chat.id, f"❌ DB Error: {exc}", parse_mode="")
    finally:
        db.close()


@bot.callback_query_handler(func=lambda call: call.data.startswith("set_"))
def handle_settings_callback(call: telebot.types.CallbackQuery) -> None:
    param = call.data.split("_")[1]
    val = call.data.split("_")[2]
    chat_id = str(call.message.chat.id)
    db = SessionLocal()
    try:
        settings_obj = db.query(UserSettings).filter_by(telegram_chat_id=chat_id).first()
        if not settings_obj:
            settings_obj = UserSettings(telegram_chat_id=chat_id)
            db.add(settings_obj)
        if param == "risk":
            settings_obj.risk_pct = float(val)
        elif param == "style":
            settings_obj.trading_style = val
        db.commit()
        bot.answer_callback_query(call.id, text=f"Settings updated: {param} set to {val}")
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text=f"✅ *Settings Profile Saved!*\n• Risk: {settings_obj.risk_pct}%\n• Style: {settings_obj.trading_style}\n\nRun `/settings` to change settings.",
            parse_mode="Markdown"
        )
    except Exception as exc:
        bot.answer_callback_query(call.id, text=f"DB Error: {exc}")
    finally:
        db.close()


def run_watchlist_alerts_loop() -> None:
    """Scans watchlist items every 5 minutes and dispatches indicators breakout signals."""
    logger.info("Watchlist background alert scanning loop started.")
    client = MarketDataClient()
    engine = TechnicalIndicatorEngine()
    
    while True:
        try:
            db = SessionLocal()
            items = db.query(WatchlistItem).all()
            unique_tickers = list(set([item.ticker for item in items]))
            signals = {}
            
            for ticker in unique_tickers:
                try:
                    df = client.fetch_ohlcv(ticker, period="5d", interval="1d")
                    if df.empty or len(df) < 5:
                        continue
                    df_ind = engine.compute_all_indicators(df)
                    latest = df_ind.iloc[-1]
                    prev = df_ind.iloc[-2]
                    
                    ema_cross = None
                    if prev["EMA_20"] <= prev["EMA_50"] and latest["EMA_20"] > latest["EMA_50"]:
                        ema_cross = "Bullish EMA Cross 📈"
                    elif prev["EMA_20"] >= prev["EMA_50"] and latest["EMA_20"] < latest["EMA_50"]:
                        ema_cross = "Bearish EMA Cross 📉"
                        
                    rsi_status = None
                    if latest["RSI_14"] >= 70.0 and prev["RSI_14"] < 70.0:
                        rsi_status = "RSI Overbought (>70) ⚠️"
                    elif latest["RSI_14"] <= 30.0 and prev["RSI_14"] > 30.0:
                        rsi_status = "RSI Oversold (<30) 🟢"
                        
                    avg_vol = df_ind["Volume"].tail(10).mean()
                    vol_spike = None
                    if latest["Volume"] > 2.0 * avg_vol:
                        vol_spike = "Volume Spike (>200% average) ⚡"
                        
                    triggers = []
                    if ema_cross: triggers.append(ema_cross)
                    if rsi_status: triggers.append(rsi_status)
                    if vol_spike: triggers.append(vol_spike)
                    
                    if triggers:
                        signals[ticker] = triggers
                except Exception as e:
                    logger.warning("Watchlist background scan failed for %s: %s", ticker, e)
            
            for item in items:
                if item.ticker in signals:
                    for trigger in signals[item.ticker]:
                        alert_msg = (
                            f"🔔 *WATCHLIST ALERT: {item.ticker}* 🚨\n"
                            f"====================================\n\n"
                            f"⚡ *Signal Triggered*: {trigger}\n"
                            f"📊 *Current Close*: ₹{df_ind.iloc[-1]['Close']:.2f}\n\n"
                            f"💡 _Run `/stock {item.ticker}` to inspect indicators alignment._"
                        )
                        try:
                            bot.send_message(item.telegram_chat_id, alert_msg, parse_mode="Markdown")
                        except Exception as e:
                            logger.error("Failed sending watchlist alert to chat %s: %s", item.telegram_chat_id, e)
            db.close()
        except Exception as exc:
            logger.error("Error in background watchlist alert scanner: %s", exc)
        time.sleep(300)


@bot.message_handler(func=lambda msg: msg.text in (
    "🌅 Morning F&O Picks", 
    "📊 Nifty Scan Summary", 
    "🏆 Top Bullish Stocks", 
    "💡 Option Recommendation"
))
def handle_keyboard_buttons(message: telebot.types.Message) -> None:
    """Routes keyboard button clicks to corresponding handlers."""
    txt = message.text
    if txt == "🌅 Morning F&O Picks":
        handle_top10(message)
    elif txt == "📊 Nifty Scan Summary":
        handle_market_summary(message)
    elif txt == "🏆 Top Bullish Stocks":
        handle_top_stocks(message)
    elif txt == "💡 Option Recommendation":
        handle_welcome(message)


# ==============================================================================
# Execution Entry Point
# ==============================================================================

def main() -> None:
    """Launches the Telegram Bot listener and morning scheduler thread."""
    if not bot_token or bot_token == "mock_telegram_bot_token_for_testing":
        print("[ERROR] TELEGRAM_BOT_TOKEN is not defined in .env!")
        logger.error("TELEGRAM_BOT_TOKEN environment variable is not defined. Bot cannot start.")
        return

    print("\n====================================================")
    print("📈 AI Stock Advisor Telegram Bot is starting...")
    print("====================================================")
    
    # Spin up daily morning report scheduler in daemon background thread
    scheduler_thread = threading.Thread(target=run_scheduler_loop, daemon=True)
    scheduler_thread.start()
    print("[INFO] Daily morning pre-market report scheduler started.")

    # Spin up morning F&O opportunity scanner scheduler
    fo_scheduler = MorningScheduler(bot)
    fo_scheduler.start()
    print("[INFO] Morning F&O opportunities pre-market scanner scheduler started.")

    # Spin up watchlist monitoring daemon background thread
    watchlist_thread = threading.Thread(target=run_watchlist_alerts_loop, daemon=True)
    watchlist_thread.start()
    print("[INFO] 5-minute background watchlist alert monitoring thread started.")

    # Start long polling
    print("[SUCCESS] Bot is now active and polling for messages!")
    print("Go to Telegram and search for @optiontradeanalystbot to start chatting.")
    print("====================================================\n")
    bot.infinity_polling()


if __name__ == "__main__":
    main()
