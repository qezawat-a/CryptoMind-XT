import asyncio
import logging
import time
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes,
)
from config import Config
from bot.memory import LongTermMemory
from bot.ai_chat import AIChat
from bot.trader import XTTrader

logger = logging.getLogger("xt_telegram")

TELEGRAM_MAX_LEN = 4000


def _split_message(text: str) -> list:
    if len(text) <= TELEGRAM_MAX_LEN:
        return [text]
    return [text[i:i + TELEGRAM_MAX_LEN] for i in range(0, len(text), TELEGRAM_MAX_LEN)]


class TelegramBot:
    def __init__(self, trader: XTTrader, ai_chat: AIChat, memory: LongTermMemory):
        self.trader = trader
        self.ai = ai_chat
        self.memory = memory
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self._authorized_user_id = int(Config.TELEGRAM_USER_ID)
        self._loop = None
        self.trader.set_notify_callback(self._notify_from_thread)

    def _is_authorized(self, user_id: int) -> bool:
        return user_id == self._authorized_user_id

    def _notify_from_thread(self, message: str):
        """Called from the trader's worker thread, so the coroutine has to be
        handed to the bot's event loop rather than awaited here."""
        if self._loop is None or self._loop.is_closed():
            logger.warning(f"Notification dropped (event loop not ready): {message}")
            return
        asyncio.run_coroutine_threadsafe(self._send_notification(message), self._loop)

    async def _send_notification(self, message: str):
        try:
            for chunk in _split_message(message):
                await self.bot.send_message(chat_id=self._authorized_user_id, text=chunk)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text(
            "XT AI Trader Bot Ready!\n\n"
            "Commands:\n"
            "/pnl - Profit/Loss summary\n"
            "/status - Bot status\n"
            "/balance - Account balance\n"
            "/autotrade_on - Enable auto-trading\n"
            "/autotrade_off - Disable auto-trading\n"
            "/signal - Scan for signals\n"
            "/settings - View current settings\n"
            "/check_ai - Test AI connection\n"
            "/timeframes - View/change timeframes\n"
            "/margin_amount_pct <value> - Set margin %\n"
            "/margin_risk_pct <value> - Set risk %\n"
            "/close [trade_id] - Close a position\n"
            "/diag - Diagnose mid-management\n"
            "/sync - Sync with exchange positions\n"
            "/protect - Attach stops to unprotected positions\n"
            "/midmanage - Run breakeven + trailing now\n\n"
            "You can also chat with me normally to change settings!"
        )

    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        pnl = self.memory.get_total_pnl()
        stats = self.memory.get_trade_count()
        open_trades = self.memory.get_open_trades()
        response = (
            f"PNL Summary\n"
            f"Total PnL: {pnl:.4f} USDT\n"
            f"Total Trades: {stats['total']} | Open: {stats['open']} | Closed: {stats['closed']}\n"
            f"Wins: {stats['wins']} | Losses: {stats['losses']} | "
            f"Flat/Unknown PnL: {stats['flat_or_unknown']} | Winrate: {stats['winrate']}%\n"
        )
        if open_trades:
            response += "\nOpen Positions:\n"
            for t in open_trades:
                response += (f"  ID:{t['id']} {t['symbol']} {t['position_side']} "
                             f"Entry:{t['entry_price']} Amt:{t['amount']} "
                             f"Lev:{t['leverage']}x\n")
        await update.message.reply_text(response)

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, self.trader.get_status_report)
        for chunk in _split_message(report):
            await update.message.reply_text(chunk)

    async def cmd_autotrade_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        result = self.trader.start_auto_trade()
        await update.message.reply_text(result)

    async def cmd_autotrade_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        result = self.trader.stop_auto_trade()
        await update.message.reply_text(result)

    async def cmd_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text("Scanning signals...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.trader.scanner.scan_and_report)
        report = self.trader.scanner.format_signal_report(result)
        for chunk in _split_message(report):
            await update.message.reply_text(chunk)

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        settings = self.memory.get_all_settings()
        if not settings:
            await update.message.reply_text("No custom settings. Using defaults.")
            settings = Config.default_settings()
        response = "Current Settings:\n"
        for k, v in settings.items():
            response += f"  {k}: {v}\n"
        await update.message.reply_text(response)

    async def cmd_check_ai(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text("Checking AI connection...")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self.ai.chat, "Hello, respond with a brief confirmation that you are online."
        )
        await update.message.reply_text(
            f"AI model: {self.ai.get_model_info()}\n\nAI Response:\n{response}"
        )

    async def cmd_timeframes(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        args = context.args
        if args:
            tfs = args[0]
            valid = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
            tf_list = [t.strip().lower() for t in tfs.split(",")]
            invalid = [t for t in tf_list if t not in valid]
            if invalid:
                await update.message.reply_text(
                    f"Invalid timeframes: {invalid}. Valid: {', '.join(valid)}"
                )
                return
            self.memory.set_setting("timeframes", ",".join(tf_list))
            await update.message.reply_text(f"Timeframes set to: {', '.join(tf_list)}")
        else:
            tfs = self.memory.get_setting("timeframes", ",".join(Config.DEFAULT_TIMEFRAMES))
            await update.message.reply_text(
                f"Current timeframes: {tfs}\n"
                f"Change via: /timeframes 5m,15m,1h\n"
                f"Valid: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d"
            )

    async def cmd_margin_amount_pct(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        args = context.args
        if args:
            try:
                pct = float(args[0])
                pct = max(1.0, min(pct, 100.0))
                self.memory.set_setting("margin_amount_pct", pct)
                await update.message.reply_text(f"Margin amount % set to: {pct}%")
            except ValueError:
                await update.message.reply_text("Invalid value. Use a number like: /margin_amount_pct 10")
        else:
            pct = self.memory.get_setting("margin_amount_pct", Config.DEFAULT_MARGIN_AMOUNT_PCT)
            await update.message.reply_text(f"Current margin amount: {pct}%\nChange via: /margin_amount_pct 15")

    async def cmd_margin_risk_pct(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        args = context.args
        if args:
            try:
                pct = float(args[0])
                pct = max(0.1, min(pct, 10.0))
                self.memory.set_setting("margin_risk_pct", pct)
                await update.message.reply_text(f"Risk % set to: {pct}%")
            except ValueError:
                await update.message.reply_text("Invalid value. Use a number like: /margin_risk_pct 1")
        else:
            pct = self.memory.get_setting("margin_risk_pct", Config.DEFAULT_RISK_PCT)
            await update.message.reply_text(f"Current risk: {pct}%\nChange via: /margin_risk_pct 1.5")

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text("Fetching balance...")
        loop = asyncio.get_event_loop()
        try:
            item = await loop.run_in_executor(None, self.trader.risk._get_usdt_balance, True)
        except Exception as e:
            await update.message.reply_text(f"Failed to fetch balance: {e}")
            return
        if not item:
            await update.message.reply_text(
                "No USDT balance returned by XT. Check that the API key has "
                "futures permissions and that the futures account is opened.")
            return
        wallet = float(item.get("walletBalance") or 0)
        available = float(item.get("availableBalance") or 0)
        frozen = float(item.get("openOrderMarginFrozen") or 0)
        isolated = float(item.get("isolatedMargin") or 0)
        crossed = float(item.get("crossedMargin") or 0)
        await update.message.reply_text(
            f"BALANCE (USDT)\n"
            f"Wallet: {wallet:.4f}\n"
            f"Available: {available:.4f}\n"
            f"Order Margin Frozen: {frozen:.4f}\n"
            f"Isolated Margin: {isolated:.4f}\n"
            f"Crossed Margin: {crossed:.4f}\n"
            f"Recorded PnL: {self.memory.get_total_pnl():.4f}")

    async def cmd_diag(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, self.trader.diagnose)
        for chunk in _split_message(report):
            await update.message.reply_text(chunk)

    async def cmd_midmanage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, self.trader.run_mid_management)
        for chunk in _split_message(report):
            await update.message.reply_text(chunk)

    async def cmd_sync(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text("Syncing with exchange positions...")
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, self.trader.sync_positions)
        for chunk in _split_message(report):
            await update.message.reply_text(chunk)

    async def cmd_protect(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        await update.message.reply_text("Attaching stops to unprotected positions...")
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, self.trader.protect_open_positions)
        for chunk in _split_message(report):
            await update.message.reply_text(chunk)

    async def cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("Unauthorized.")
            return
        args = context.args
        loop = asyncio.get_event_loop()
        if args:
            try:
                trade_id = int(args[0])
            except ValueError:
                await update.message.reply_text("Invalid trade ID.")
                return
            result = await loop.run_in_executor(
                None, self.trader.close_specific_trade, trade_id)
        else:
            result = await loop.run_in_executor(None, self.trader.close_all_positions)
        for chunk in _split_message(result):
            await update.message.reply_text(chunk)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not self._is_authorized(user_id):
            await update.message.reply_text("Unauthorized.")
            return
        text = update.message.text.strip()
        if text.startswith("/"):
            await update.message.reply_text(
                "Unknown command. Use /start for command list.\n"
                "Or just chat with me naturally to manage your trading!"
            )
            return
        await update.message.reply_chat_action("typing")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.ai.chat, text)
        for chunk in _split_message(response):
            await update.message.reply_text(chunk)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Global error handler for the Telegram Application.

        Without it PTB just logs "No error handlers are registered" and the
        exception is dropped silently. Here we log the full traceback and tell
        the owner, so a failing command is never invisible.
        """
        err = context.error
        logger.error(f"Telegram handler error: {err}", exc_info=err)
        try:
            await self._send_notification(f"⚠️ Telegram error: {err}")
        except Exception:
            # Never let the error handler itself raise (it would crash polling).
            pass

    def run(self):
        app = Application.builder().token(Config.TELEGRAM_BOT_TOKEN).build()

        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("pnl", self.cmd_pnl))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("balance", self.cmd_balance))
        app.add_handler(CommandHandler("autotrade_on", self.cmd_autotrade_on))
        app.add_handler(CommandHandler("autotrade_off", self.cmd_autotrade_off))
        app.add_handler(CommandHandler("signal", self.cmd_signal))
        app.add_handler(CommandHandler("settings", self.cmd_settings))
        app.add_handler(CommandHandler("check_ai", self.cmd_check_ai))
        app.add_handler(CommandHandler("timeframes", self.cmd_timeframes))
        app.add_handler(CommandHandler("margin_amount_pct", self.cmd_margin_amount_pct))
        app.add_handler(CommandHandler("margin_risk_pct", self.cmd_margin_risk_pct))
        app.add_handler(CommandHandler("close", self.cmd_close))
        app.add_handler(CommandHandler("diag", self.cmd_diag))
        app.add_handler(CommandHandler("sync", self.cmd_sync))
        app.add_handler(CommandHandler("protect", self.cmd_protect))
        app.add_handler(CommandHandler("midmanage", self.cmd_midmanage))
        app.add_handler(CommandHandler("mid", self.cmd_midmanage))  # alias of /midmanage
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        app.add_error_handler(self.error_handler)

        app.post_init = self._register_bot_commands

        logger.info("Telegram bot polling started")
        app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)

    async def _register_bot_commands(self, app):
        from telegram import BotCommand
        # Capture the bot's running loop so worker threads can post notifications.
        self._loop = asyncio.get_running_loop()
        commands = [
            BotCommand("pnl", "Profit/Loss summary"),
            BotCommand("status", "Bot status & open positions"),
            BotCommand("balance", "Account balance"),
            BotCommand("autotrade_on", "Enable auto-trading"),
            BotCommand("autotrade_off", "Disable auto-trading"),
            BotCommand("signal", "Scan for trading signals"),
            BotCommand("settings", "View current settings"),
            BotCommand("check_ai", "Test AI connection"),
            BotCommand("timeframes", "View/change timeframes"),
            BotCommand("margin_amount_pct", "Set margin %"),
            BotCommand("margin_risk_pct", "Set risk %"),
            BotCommand("close", "Close a position"),
            BotCommand("diag", "Why breakeven/trailing has not fired"),
            BotCommand("sync", "Adopt XT positions into the bot"),
            BotCommand("protect", "Attach a stop to unprotected positions"),
            BotCommand("midmanage", "Run breakeven + trailing now"),
        ]
        await app.bot.set_my_commands(commands)
        logger.info("Bot commands menu registered")
