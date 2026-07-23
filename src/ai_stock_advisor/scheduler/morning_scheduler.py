"""
Morning Pre-Market F&O Opportunity Scheduler.
Orchestrates pre-market data scanning at 08:45 AM IST and Telegram alerts at 08:50 AM IST.
"""
from datetime import datetime, timezone, timedelta
import logging
import threading
import time
from typing import Any

from config.settings import settings
from ai_stock_advisor.scanner.morning_scanner import MorningOpportunityScanner

logger = logging.getLogger("ai_stock_advisor.scheduler.morning_scheduler")


class MorningScheduler:
    """
    Timezone-agnostic scheduler that coordinates scans and alerts using Indian Standard Time (IST).
    Runs as a daemon background thread.
    """

    def __init__(self, bot_client: Any) -> None:
        """Initializes scheduler with the active Telegram Bot instance."""
        self.bot = bot_client
        self.scanner = MorningOpportunityScanner()
        self.running = False
        self.last_scan_date = ""
        self.last_alert_date = ""

    @staticmethod
    def get_ist_now() -> datetime:
        """Calculates current time in Indian Standard Time (UTC + 5:30)."""
        # IST offset from UTC is +5 hours, 30 minutes
        return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

    def format_telegram_report(self, opportunities: list) -> str:
        """Structures the pre-market scanner results into a Telegram Markdown message."""
        now_ist = self.get_ist_now()
        date_str = now_ist.strftime("%d-%b-%Y")
        
        msg = (
            f"🌅 *MORNING F&O OPPORTUNITIES SCANNER* 📈\n"
            f"📅 *Trade Date*: {date_str} (Pre-Market Setup)\n"
            f"⚡ _Generated at 08:50 AM IST before open_\n\n"
            f"📢 *Market Outlook*:\n"
            f"NSE Derivatives scan complete. Ranked Top 10 high-probability setups based on historical indicators, volume spikes, news sentiment, and option OI grids.\n\n"
            f"====================================\n\n"
        )
        
        for idx, op in enumerate(opportunities, 1):
            msg += (
                f"{idx}. *{op['Ticker']}* ({op['Trend']}) 🎯\n"
                f"• *Prob / Conf*: {op['Probability']}% / {op['Confidence']}%\n"
                f"• *Entry*: ₹{op['Entry']:.2f}\n"
                f"• *Stop Loss*: ₹{op['Stop_Loss']:.2f}\n"
                f"• *Targets*: T1: ₹{op['Target_1']:.2f} | T2: ₹{op['Target_2']:.2f}\n"
                f"• *Option Trade*: Buy {op['Strike']} {op['Option_Type']} (Prem: {op['Premium_Range']})\n"
                f"• *Risk / Size*: {op['Risk_Level']} | {op['Position_Size']}\n"
                f"• *AI Analysis*: _{op['Explanation']}_\n\n"
            )
            
        msg += "💡 _Note: Sizing is based on portfolio risk weighting. Stop Loss is strict._"
        return msg

    def dispatch_telegram_alerts(self, report_text: str) -> None:
        """Sends the compiled markdown report to the configured Telegram Chat ID."""
        chat_id = settings.TELEGRAM_CHAT_ID
        if not chat_id or chat_id == "mock_telegram_chat_id_for_testing":
            logger.warning("No valid TELEGRAM_CHAT_ID configured. Telegram morning alert skipped.")
            return

        logger.info("Dispatching pre-market opportunities report to chat %s...", chat_id)
        try:
            # telebot expects Markdown by default as configured in bot.py
            self.bot.send_message(chat_id, report_text, parse_mode="Markdown")
            logger.info("Pre-market opportunities alert successfully sent to Telegram.")
        except Exception as exc:
            logger.error("Failed sending morning opportunities alert to Telegram: %s", str(exc))

    def run_scheduler_loop(self) -> None:
        """Main execution loop that triggers scans and dispatch tasks at daily IST marks."""
        logger.info("Morning F&O Scheduler active and monitoring IST clock...")
        self.running = True

        while self.running:
            try:
                now_ist = self.get_ist_now()
                current_date = now_ist.strftime("%Y-%m-%d")
                
                # Check for weekend (Saturday = 5, Sunday = 6)
                is_weekday = now_ist.weekday() < 5
                
                # Time markers
                current_time_str = now_ist.strftime("%H:%M")
                
                if is_weekday:
                    # 1. Trigger scan at 08:45 AM IST
                    if current_time_str == "08:45" and self.last_scan_date != current_date:
                        logger.info("IST clock marked 08:45 AM. Initializing pre-market scan...")
                        self.scanner.run_scan()
                        self.last_scan_date = current_date
                        
                    # 2. Trigger Telegram report dispatch at 08:50 AM IST
                    if current_time_str == "08:50" and self.last_alert_date != current_date:
                        logger.info("IST clock marked 08:50 AM. Compiling opportunities report...")
                        # Load results from cached JSON file
                        json_file = Path(settings.BASE_DIR) / "data" / "morning_opportunities.json"
                        if json_file.exists():
                            try:
                                import json
                                with open(json_file, "r", encoding="utf-8") as f:
                                    opportunities = json.load(f)
                                if opportunities:
                                    report_text = self.format_telegram_report(opportunities)
                                    self.dispatch_telegram_alerts(report_text)
                                else:
                                    logger.warning("Morning opportunities cache file is empty.")
                            except Exception as exc:
                                logger.error("Failed parsing morning opportunities file for dispatch: %s", str(exc))
                        else:
                            logger.warning("Morning opportunities cache file not found at 08:50 AM.")
                        
                        self.last_alert_date = current_date

            except Exception as exc:
                logger.error("Error encountered in morning scheduler loop: %s", str(exc))

            time.sleep(15)  # Check clock every 15 seconds

    def start(self) -> None:
        """Starts the scheduler thread."""
        thread = threading.Thread(target=self.run_scheduler_loop, daemon=True)
        thread.start()
        logger.info("Morning F&O Scheduler background thread successfully spawned.")
