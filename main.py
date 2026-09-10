import logging
import os
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from config import Config
from bot.memory import LongTermMemory
from bot.trader import XTTrader

is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None

handlers = [logging.StreamHandler(sys.stdout)]
if not is_railway:
    os.makedirs("logs", exist_ok=True)
    handlers.append(logging.FileHandler("logs/trader.log"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("main")


def _db_kind(database_url: str) -> str:
    u = (database_url or "").lower()
    if "mysql" in u:
        return "MySQL"
    if "postgres" in u or "neon.tech" in u:
        return "Neon-Postgres"
    if "sqlite" in u or not u:
        return "SQLite"
    return "DB"


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


def _start_health_server():
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health check server listening on port {port}")
    server.serve_forever()


def _resolve_xt_credentials():
    """Load XT API key/secret from env or the AItradekit credential files.

    Mirrors AItradekit's load_credentials() so CryptoMind-XT runs in the same
    environment without re-entering the secret. Returns ("", "") if unset.
    """
    ak = os.getenv("XT_API_KEY", "")
    sk = os.getenv("XT_API_SECRET", "")
    if ak and sk:
        return ak, sk
    import json
    for path in (os.path.expanduser("~/.xt-tradekit/credentials.json"),
                 os.path.expanduser("~/.xt-exchange/credentials.json")):
        if os.path.exists(path):
            try:
                with open(path) as f:
                    creds = json.load(f)
                ak = creds.get("access_key", "")
                sk = creds.get("secret_key", "")
                if ak and sk:
                    return ak, sk
            except Exception:
                continue
    return "", ""


def run_agent_headless():
    """Run the AGENT in headless mode (no Telegram polling).
    Agent decides autonomously every AGENT_AUTONOMOUS_INTERVAL_SEC.
    """
    logger.info("Starting CryptoMind-XT in AGENT HEADLESS mode.")
    ak, sk = _resolve_xt_credentials()
    if not ak or not sk:
        logger.error("XT credentials not found. Set XT_API_KEY/XT_API_SECRET or "
                     "place them in ~/.xt-tradekit/credentials.json.")
        sys.exit(1)
    Config.XT_API_KEY = ak
    Config.XT_API_SECRET = sk

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        database_url = "sqlite:///data/memory.db"
        logger.warning("DATABASE_URL not set, using SQLite at data/memory.db (lost on deploy!)")
    else:
        logger.info(f"Database: {_db_kind(database_url)}")

    memory = LongTermMemory(database_url=database_url)
    for k, v in Config.default_settings().items():
        memory.set_setting_default(k, v)

    trader = XTTrader(memory=memory)
    trader.set_notify_callback(lambda m: logger.info(f"NOTIFY: {m}"))

    # Init Agent Brain
    from agent import Brain, Agent
    try:
        brain = Brain()
        agent = Agent(trader=trader, memory=memory, brain=brain)
        logger.info(f"Agent brain ready: {brain.get_model_info()}")
    except Exception as e:
        logger.error(f"Agent brain init failed: {e}", exc_info=True)
        sys.exit(1)

    try:
        balances = trader.xt.get_balances()
        usdt = next((b for b in balances if str(b.get("coin", "")).upper() == "USDT"), None)
        if usdt:
            logger.info(f"XT connection OK. USDT wallet: {usdt.get('walletBalance')}")
        else:
            logger.warning("XT connection OK but no USDT balance.")
    except Exception as e:
        logger.error(f"XT API check failed: {e}")

    try:
        adopted = trader.position_mgr.adopt_exchange_positions()
        for a in adopted:
            logger.warning(f"Adopted: {a['symbol']} {a['position_side']} {a['size']}c @ {a['entry_price']} {a['leverage']}x")
        if not adopted:
            logger.info("No untracked positions")
    except Exception as e:
        logger.error(f"Position adoption failed: {e}")

    # Start safety guards (breakeven/trailing/reconcile) without old auto-trade
    trader.start_mid_manager()
    logger.info(f"Agent autonomous loop every {Config.AGENT_AUTONOMOUS_INTERVAL_SEC}s. Ctrl+C to stop.")

    import threading
    stop = threading.Event()
    last_report_hd = 0.0

    def agent_loop():
        nonlocal last_report_hd
        while not stop.is_set():
            try:
                for ev in trader.position_mgr.reconcile_open_trades():
                    logger.info(f"Reconcile: {ev}")
                trader.check_positions_for_close()
                result = agent.autonomous_tick()
                logger.info(f"AGENT TICK: {result[:500]}")
                if "OPENED" in result or "CLOSED" in result:
                    logger.info(f"AGENT ACTION: {result}")
                # Periodic report for headless too
                try:
                    interval = int(memory.get_setting("report_interval_sec", Config.REPORT_INTERVAL_SEC))
                    now = time.time()
                    if interval > 0 and now - last_report_hd >= interval:
                        last_report_hd = now
                        report = trader.periodic_pnl_report()
                        logger.info(f"PERIODIC REPORT: {report[:500]}")
                except Exception as e:
                    logger.warning(f"Periodic report failed: {e}")
            except Exception as e:
                logger.error(f"Agent tick error: {e}", exc_info=True)
            stop.wait(Config.AGENT_AUTONOMOUS_INTERVAL_SEC)

    t = threading.Thread(target=agent_loop, daemon=True)
    t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down agent...")
    finally:
        stop.set()
        trader.stop_mid_manager()
        memory.close()
        logger.info("Agent stopped.")


def run_headless():
    """Run the auto-trader without Telegram / AI / MySQL.

    Uses XT credentials + a SQLite database and logs every notification (trades,
    reversals, PnL reports) to the logger, so a tail of the log is the status
    feed. This is the unified, single-process replacement for the standalone
    signal_bot + cron setup.
    """
    logger.info("Starting CryptoMind-XT in HEADLESS mode (no Telegram / AI).")
    ak, sk = _resolve_xt_credentials()
    if not ak or not sk:
        logger.error("XT credentials not found. Set XT_API_KEY/XT_API_SECRET or "
                     "place them in ~/.xt-tradekit/credentials.json.")
        sys.exit(1)
    Config.XT_API_KEY = ak
    Config.XT_API_SECRET = sk

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        database_url = "sqlite:///data/memory.db"
        logger.warning(
            "DATABASE_URL is not set — falling back to SQLite at data/memory.db. "
            "ALL DATA (trades, settings, history) WILL BE LOST on every deploy. "
            "Set DATABASE_URL to your Neon Postgres URL or MySQL URL in Railway Variables."
        )
    else:
        logger.info(f"Database: {_db_kind(database_url)}")

    memory = LongTermMemory(database_url=database_url)
    seeded = [k for k, v in Config.default_settings().items()
              if memory.set_setting_default(k, v)]
    if seeded:
        logger.info(f"Seeded default settings: {', '.join(sorted(seeded))}")
    else:
        logger.info("Existing settings preserved")

    trader = XTTrader(memory=memory)
    # Surface every notification (trades, reversals, PnL reports) via the log.
    trader.set_notify_callback(lambda m: logger.info(f"NOTIFY: {m}"))

    try:
        balances = trader.xt.get_balances()
        usdt = next((b for b in balances if str(b.get("coin", "")).upper() == "USDT"), None)
        if usdt:
            logger.info(f"XT connection OK. USDT wallet balance: "
                        f"{usdt.get('walletBalance')}")
        else:
            logger.warning("XT connection OK but no USDT balance found.")
    except Exception as e:
        logger.error(f"XT API check failed: {e}")

    try:
        adopted = trader.position_mgr.adopt_exchange_positions()
        for a in adopted:
            logger.warning(f"Adopted untracked position: {a['symbol']} "
                           f"{a['position_side']} {a['size']}c @ {a['entry_price']} "
                           f"{a['leverage']}x, stop={'yes' if a['has_stop'] else 'NO'}")
        if not adopted:
            logger.info("No untracked exchange positions found")
    except Exception as e:
        logger.error(f"Position adoption failed at startup: {e}")

    trader.start_auto_trade()
    # Breakeven/trailing/TP-SL protection runs regardless of auto-trade.
    trader.start_mid_manager()
    logger.info("Headless auto-trade started. Send SIGINT (Ctrl+C) to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down via KeyboardInterrupt...")
    finally:
        trader.stop_mid_manager()
        trader.stop_auto_trade()
        memory.close()
        logger.info("CryptoMind-XT stopped.")


def main():
    if "--headless" in sys.argv:
        run_headless()
        return
    if "--agent" in sys.argv or "--agent-headless" in sys.argv:
        run_agent_headless()
        return
    # Telegram + AI are only needed for interactive; keep lazy import
    # Try Agent first, fallback to old AIChat if brain fails
    use_agent = True
    # XT creds: env first, then ~/.xt-tradekit/credentials.json (same as headless modes).
    # Without this, XTClient sends an empty appkey and XT replies:
    # "Absence of [appKey] in client request is prohibited" -> Balance unavailable.
    _ak, _sk = _resolve_xt_credentials()
    if _ak and _sk:
        Config.XT_API_KEY = _ak
        Config.XT_API_SECRET = _sk
    elif not Config.XT_API_KEY or not Config.XT_API_SECRET:
        logger.error("XT credentials not found. Set XT_API_KEY/XT_API_SECRET or "
                     "place them in ~/.xt-tradekit/credentials.json. "
                     "Balance/position calls will fail with 'Absence of [appKey]'.")
    missing = Config.validate()
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Set them in Railway Variables or .env file.")
        sys.exit(1)

    logger.info("Initializing XT AI Trader (AGENT MODE)...")
    logger.info(f"Provider: {Config.get_effective_provider()} @ {Config.get_effective_base_url()}")
    logger.info(f"AI Model: {Config.get_effective_model() or 'auto'}")
    logger.info(f"Railway: {is_railway}")
    logger.info(f"PORT env: {os.getenv('PORT', '<unset>')}")

    # Railway always injects PORT; fall back to it as the deploy signal so the
    # health endpoint comes up even if RAILWAY_ENVIRONMENT is absent.
    should_serve_health = is_railway or os.getenv("PORT") is not None
    if should_serve_health:
        # Bring the health port up before any slow initialisation, otherwise
        # the platform's proxy kills the deploy before it is marked healthy.
        health_thread = threading.Thread(target=_start_health_server, daemon=True)
        health_thread.start()

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        database_url = "sqlite:///data/memory.db"
        logger.warning("DATABASE_URL is not set, falling back to SQLite at "
                       "data/memory.db. On Railway that file lives inside the "
                       "container and is DESTROYED on every deploy, so open trades "
                       "and settings will be lost. Point DATABASE_URL at Neon Postgres "
                       "or MySQL in the service's Variables.")
    else:
        masked = database_url.split("@")[-1] if "@" in database_url else database_url
        logger.info(f"DATABASE_URL resolved to host: {masked}")
    logger.info(f"Database: {_db_kind(database_url)}")

    memory = None
    try:
        memory = LongTermMemory(database_url=database_url)
        logger.info("Database schema ready")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        logger.error("Fix DATABASE_URL / DB availability before redeploying.")
        sys.exit(1)

    # Only seed missing keys, otherwise every Railway restart would wipe the
    # settings the user configured over Telegram.
    seeded = [k for k, v in Config.default_settings().items()
              if memory.set_setting_default(k, v)]
    if seeded:
        logger.info(f"Seeded default settings: {', '.join(sorted(seeded))}")
    else:
        logger.info("Existing settings preserved")
    # Auto-delete legacy settings (max_loss/max_profit now dynamic, not in settings)
    # DB-agnostic: use ORM instead of raw SQL with MySQL backticks (breaks Postgres).
    try:
        from bot.memory import Setting
        s = memory._session_factory()
        try:
            for legacy_key in getattr(Config, "LEGACY_SETTINGS", set()):
                s.query(Setting).filter(Setting.key == legacy_key).delete()
            s.commit()
        finally:
            s.close()
        logger.info("Legacy settings cleaned (max_loss/max_profit removed from DB)")
    except Exception as e:
        logger.warning(f"Legacy cleanup failed (ignore if first run): {e}")
    # Auto-clean buggy cooldowns >10min (4868s/4000s bug from old code)
    try:
        from sqlalchemy import text as sql_text2
        with memory._engine.begin() as conn:
            # Delete any cooldown that still has >600s remaining (cap is 600s now)
            rows = conn.execute(sql_text2("SELECT symbol, side, cooldown_until FROM cooldowns")).fetchall()
            now = time.time()
            for sym, side, until in rows:
                if until - now > 600:
                    conn.execute(sql_text2("DELETE FROM cooldowns WHERE symbol=:s AND side=:side"), {"s": sym, "side": side})
                    logger.info(f"Cleaned buggy cooldown {sym} {side} {until-now:.0f}s -> deleted")
        logger.info("Buggy cooldowns cleaned")
    except Exception as e:
        logger.warning(f"Cooldown cleanup failed: {e}")

    trader = XTTrader(memory=memory)

    # --- AGENT MODE (replaces old bot) ---
    agent = None
    ai_chat = None
    try:
        from agent import Brain, Agent
        brain = Brain()
        agent = Agent(trader=trader, memory=memory, brain=brain)
        logger.info(f"Agent brain ready: {brain.get_model_info()}")
        # Start autonomous agent loop (replaces trader.start_auto_trade deterministice bot)
        # Old bot loop deleted - agent decides itself
        agent_loop_stop = None
        def start_agent_autonomous():
            import threading
            stop = threading.Event()
            last_report = 0.0
            def loop():
                nonlocal last_report
                while not stop.is_set():
                    try:
                        for ev in trader.position_mgr.reconcile_open_trades():
                            logger.info(f"Reconcile: {ev}")
                        trader.check_positions_for_close()
                        res = agent.autonomous_tick()
                        logger.info(f"AGENT AUTONOMOUS: {res[:500]}")
                        # Periodic status report (report_interval_sec) - so user gets status every 60s even without trade
                        try:
                            interval = int(memory.get_setting("report_interval_sec", Config.REPORT_INTERVAL_SEC))
                            now = time.time()
                            if interval > 0 and now - last_report >= interval:
                                last_report = now
                                report = trader.periodic_pnl_report()
                                # Send to Telegram via notify and log
                                trader._notify(report)
                                logger.info(f"PERIODIC REPORT: {report[:400]}")
                        except Exception as e:
                            logger.warning(f"Periodic report failed: {e}")
                    except Exception as e:
                        logger.error(f"Agent tick error: {e}", exc_info=True)
                    stop.wait(Config.AGENT_AUTONOMOUS_INTERVAL_SEC)
            t = threading.Thread(target=loop, daemon=True)
            t.start()
            return stop
        agent_loop_stop = start_agent_autonomous()
        logger.info(f"Agent autonomous loop ON every {Config.AGENT_AUTONOMOUS_INTERVAL_SEC}s, report every {memory.get_setting('report_interval_sec', Config.REPORT_INTERVAL_SEC)}s (AGENT MODE - bot deleted)")
    except Exception as e:
        logger.error(f"Agent init failed, falling back to old AIChat: {e}", exc_info=True)
        from bot.ai_chat import AIChat
        ai_chat = AIChat(memory=memory)
        ai_chat.bind_trader(trader)
        logger.info(f"AI model fallback: {ai_chat.get_model_info()}")
        agent = None

    try:
        balances = trader.xt.get_balances()
        usdt = next((b for b in balances if str(b.get("coin", "")).upper() == "USDT"), None)
        if usdt:
            logger.info(f"XT connection OK. USDT wallet balance: {usdt.get('walletBalance')}")
        else:
            logger.warning("XT connection OK but no USDT balance found. "
                           "Is the futures account opened?")
    except Exception as e:
        logger.error(f"XT API check failed: {e}")
        logger.error("Verify XT_API_KEY / XT_API_SECRET, that the key has futures "
                     "permissions, and that this server's IP is whitelisted.")

    # A wiped database or a manually opened position would otherwise be
    # invisible to every guard, since they all iterate the local trade table.
    try:
        adopted = trader.position_mgr.adopt_exchange_positions()
        for a in adopted:
            logger.warning(f"Adopted untracked position: {a['symbol']} "
                           f"{a['position_side']} {a['size']}c @ {a['entry_price']} "
                           f"{a['leverage']}x, stop={'yes' if a['has_stop'] else 'NO'}")
        if not adopted:
            logger.info("No untracked exchange positions found")
    except Exception as e:
        logger.error(f"Position adoption failed at startup: {e}")

    # Telegram now talks to AGENT, not old AIChat - keep mid_manager for safety
    trader.start_mid_manager()
    if agent:
        from agent.telegram_agent import TelegramAgent
        telegram_bot = TelegramAgent(trader=trader, agent=agent, memory=memory)
        logger.info("Telegram -> Agent mode")
    else:
        from bot.telegram_bot import TelegramBot
        telegram_bot = TelegramBot(trader=trader, ai_chat=ai_chat, memory=memory)
        logger.info("Telegram -> Legacy AIChat mode")

    logger.info("XT AI Trader started. Telegram bot is listening...")
    logger.info("Send /start to your bot to begin.")

    try:
        telegram_bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down via KeyboardInterrupt...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Agent loop handles trading, old bot auto_trade not used anymore
        try:
            if 'agent_loop_stop' in locals() and agent_loop_stop:
                agent_loop_stop.set()
        except:
            pass
        trader.stop_mid_manager()
        try:
            trader.stop_auto_trade()
        except:
            pass
        memory.close()
        logger.info("XT AI Trader stopped.")


if __name__ == "__main__":
    main()
