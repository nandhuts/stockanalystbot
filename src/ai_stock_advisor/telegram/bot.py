"""
Telegram Bot Entrypoint.
Handles commands: /start, /help, /market, /topstocks, /stock, /news.
Sets up a background scheduler thread to send pre-market reports every morning.
"""
import logging
from pathlib import Path
import threading
import time
import telebot
import schedule
import pandas as pd

from config.settings import settings
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.core.scanner import StockScanner
from ai_stock_advisor.core.ranker import StockRanker
from ai_stock_advisor.services.market_data.client import MarketDataClient
from ai_stock_advisor.services.llm.news_analyzer import NewsAnalyzer

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

@bot.message_handler(commands=["start", "help"])
def handle_welcome(message: telebot.types.Message) -> None:
    """Welcomes the user and details available bot commands."""
    welcome_text = (
        "👋 *Welcome to the AI Stock Advisor Telegram Bot!* 📈🤖\n\n"
        "Evaluate trends, compute scores, and pull news analyses for Nifty 50 NSE securities.\n\n"
        "💬 *Available Command Interface*:\n"
        "• /market - General overview stats of the latest scan.\n"
        "• /topstocks - List the Top 10 stocks by breakout probability.\n"
        "• /stock `<ticker>` - Evaluate technical profiles for any ticker (e.g. `/stock INFY.NS`).\n"
        "• /news `<ticker>` - Summarize news sentiment for any ticker (e.g. `/news TCS.NS`)."
    )
    bot.reply_to(message, welcome_text)


@bot.message_handler(commands=["market"])
def handle_market_summary(message: telebot.types.Message) -> None:
    """Renders general index statistics from the latest scanner output."""
    scan_file = Path(settings.BASE_DIR) / "data" / "scan_results.csv"
    
    if not scan_file.exists():
        bot.reply_to(
            message, 
            "⚠️ Scanner output is currently empty. Run scanner on the dashboard to generate records."
        )
        return

    try:
        df = pd.read_csv(scan_file)
        total = len(df)
        bullish = len(df[df["Score"] >= 70])
        bearish = len(df[df["Score"] <= 30])
        neutral = total - bullish - bearish
        avg = df["Score"].mean()
        
        summary = (
            "📊 *Nifty 50 Market Scan Summary*:\n\n"
            f"• Total Securities: {total}\n"
            f"• Bullish (Score ≥ 70): {bullish} 🟢\n"
            f"• Neutral (30 < Score < 70): {neutral} 🟡\n"
            f"• Bearish (Score ≤ 30): {bearish} 🔴\n"
            f"• Avg Bullish Score: {avg:.1f} / 100"
        )
        bot.reply_to(message, summary)
    except Exception as exc:
        bot.reply_to(message, f"Failed reading market summary data: {exc}")


@bot.message_handler(commands=["topstocks"])
def handle_top_stocks(message: telebot.types.Message) -> None:
    """Renders the top 10 stocks based on the breakout probability scores."""
    rank_file = Path(settings.BASE_DIR) / "data" / "rankings_results.csv"
    
    # Generate rankings if missing
    if not rank_file.exists():
        try:
            client = MarketDataClient()
            engine = TechnicalIndicatorEngine()
            scanner = StockScanner(client, engine)
            ranker = StockRanker(scanner)
            df = ranker.rank_stocks(force_refresh=False)
        except Exception as exc:
            bot.reply_to(message, f"Failed generating ratings: {exc}")
            return
    else:
        df = pd.read_csv(rank_file)

    if df.empty:
        bot.reply_to(message, "Rankings are currently empty. Run scans to load values.")
        return

    # List Top 10
    top10 = df.head(10)
    msg = "🏆 *Top 10 Bullish Stock Rankings*:\n\n"
    for idx, (_, row) in enumerate(top10.iterrows(), 1):
        msg += f"{idx}. *{row['Ticker']}* - Prob: *{row['Probability_Score']}%* | Price: ₹{row['Close']:,.2f}\n"
        
    bot.reply_to(message, msg)


@bot.message_handler(commands=["stock"])
def handle_stock_technical_analysis(message: telebot.types.Message) -> None:
    """Runs technical indicators and scanner models for a specified ticker."""
    args = telebot.util.extract_arguments(message.text).strip().upper()
    if not args:
        bot.reply_to(message, "⚠️ Please provide a ticker symbol (e.g. `/stock INFY.NS` or `/stock AAPL`).")
        return

    bot.send_chat_action(message.chat.id, "typing")
    
    try:
        client = MarketDataClient()
        engine = TechnicalIndicatorEngine()
        scanner = StockScanner(client, engine)
        ranker = StockRanker(scanner)
        
        # Download and calculate
        df = client.fetch_ohlcv(args, period="1y", interval="1d", force_refresh=True)
        score_dict = scanner.score_stock(df)
        prob_score = ranker.calculate_probability_score(pd.Series(score_dict))
        
        trend_status = "Bullish Crossover" if score_dict["EMA_Crossover"] else "Rangebound"
        volume_status = "Volume Surge active" if score_dict["Volume_Spike"] else "Normal Volume"

        report = (
            f"📊 *Technical Profile: {args}*\n"
            f"💰 Price: ₹{score_dict['Close']:,.2f}\n\n"
            f"⚡ *Trend Score*: {int(score_dict['Score'])} / 100\n"
            f"🎯 *Breakout Probability*: {prob_score}%\n\n"
            f"🔬 *Indicator Details*:\n"
            f"• Above EMA 20: {'Yes' if score_dict['Above_EMA20'] else 'No'}\n"
            f"• Above EMA 50: {'Yes' if score_dict['Above_EMA50'] else 'No'}\n"
            f"• EMA Crossover: {trend_status}\n"
            f"• RSI (14): {score_dict['RSI']:.1f}\n"
            f"• MACD Signal: {'Bullish' if score_dict['MACD_Bullish'] else 'Bearish/Neutral'}\n"
            f"• ADX (14) Strength: {score_dict['ADX']:.1f}\n"
            f"• Volume Status: {volume_status}"
        )
        bot.reply_to(message, report)
    except Exception as exc:
        bot.reply_to(message, f"❌ Failed analyzing stock '{args}': {exc}")


@bot.message_handler(commands=["news"])
def handle_stock_news_sentiment(message: telebot.types.Message) -> None:
    """Fetches news and calculates sentiment polarity summaries using OpenAI."""
    args = telebot.util.extract_arguments(message.text).strip().upper()
    if not args:
        bot.reply_to(message, "⚠️ Please provide a stock symbol (e.g. `/news TCS.NS` or `/news AAPL`).")
        return

    bot.send_chat_action(message.chat.id, "typing")
    
    try:
        analyzer = NewsAnalyzer()
        report = analyzer.analyze_news(args, max_articles=3)
        
        sentiment_symbol = "🟢" if report.average_sentiment_score > 0.1 else "🔴" if report.average_sentiment_score < -0.1 else "🟡"
        
        msg = (
            f"📰 *News Sentiment Report: {args}*\n\n"
            f"Consolidated Rating: *{report.overall_sentiment.value}* {sentiment_symbol}\n"
            f"Average Polarity Score: *{report.average_sentiment_score:+.2f}*\n\n"
            "📰 *Recent Stories Summary*:\n"
        )
        
        if not report.articles:
            msg += "• No news found for this security."
        else:
            for idx, art in enumerate(report.articles, 1):
                art_sym = "🟢" if art.sentiment_score > 0.1 else "🔴" if art.sentiment_score < -0.1 else "🟡"
                msg += f"{idx}. *{art.headline}* {art_sym}\n"
                msg += f"   _Summary: {art.summary}_\n\n"
                
        bot.reply_to(message, msg)
    except Exception as exc:
        bot.reply_to(message, f"❌ Failed generating news analysis for '{args}': {exc}")


# ==============================================================================
# Execution Entry Point
# ==============================================================================

def main() -> None:
    """Launches the Telegram Bot listener and morning scheduler thread."""
    if not bot_token or bot_token == "mock_telegram_bot_token_for_testing":
        logger.error("TELEGRAM_BOT_TOKEN environment variable is not defined. Bot cannot start.")
        return

    logger.info("Initializing Telegram Bot listener...")
    
    # Spin up morning report scheduler in daemon background thread
    scheduler_thread = threading.Thread(target=run_scheduler_loop, daemon=True)
    scheduler_thread.start()

    # Start long polling
    logger.info("Telegram Bot active and polling for messages...")
    bot.infinity_polling()


if __name__ == "__main__":
    main()
