import time
import logging
import threading
from config import Config
from bot.xt_client import XTClient, XTError
from bot.memory import LongTermMemory
from bot.risk_manager import RiskManager
from bot.strategies import StrategyEngine
from bot.signal_scanner import SignalScanner
from bot.position_manager import PositionManager

logger = logging.getLogger("xt_trader")


class XTTrader:
    def __init__(self, memory: LongTermMemory = None):
        self.memory = memory or LongTermMemory()
        self.xt = XTClient(
            host=Config.XT_FUTURES_HOST,
            access_key=Config.XT_API_KEY,
            secret_key=Config.XT_API_SECRET,
        )
        self.risk = RiskManager(self.xt, self.memory)
        self.scanner = SignalScanner(self.xt, self.memory)
        self.position_mgr = PositionManager(self.xt, self.memory, self.risk)
        self.engine = StrategyEngine()
        self._auto_trade_enabled = False
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        # Mid-management (breakeven, trailing stop, TP/SL protection) runs on
        # its own always-on guardian thread so it fires by itself without a
        # manual /midmanage and regardless of whether auto-trade is enabled.
        # The lock serializes the guardian against the manual /midmanage path
        # so two threads can never move the same stop concurrently.
        self._mid_lock = threading.Lock()
        self._mid_manager_thread = None
        self._stop_mid_manager = threading.Event()
        self._notify_callback = None

    def set_notify_callback(self, callback):
        self._notify_callback = callback

    def _notify(self, message: str):
        logger.info(f"NOTIFY: {message}")
        if self._notify_callback:
            try:
                self._notify_callback(message)
            except Exception as e:
                logger.error(f"Notification dispatch failed: {e}")

    # ---------- status ----------

    def get_status_report(self) -> str:
        symbol = self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL)
        try:
            balance = self.risk.get_total_balance()
            available = self.risk.get_available_balance()
            balance_line = f"Balance: {balance:.2f} USDT | Available: {available:.2f} USDT\n"
        except XTError as e:
            balance_line = f"Balance: unavailable ({e})\n"

        pnl = self.memory.get_total_pnl()
        stats = self.memory.get_trade_count()
        settings = self.memory.get_all_settings()

        report = "=== XT AI TRADER STATUS ===\n"
        report += f"Symbol: {symbol}\n"
        report += balance_line
        report += f"Total PnL: {pnl:.4f} USDT\n"
        report += (f"Trades: {stats['total']} total | {stats['open']} open | "
                   f"{stats['closed']} closed | {stats['winrate']}% WR\n")
        report += f"Auto-Trade: {'ON' if self._auto_trade_enabled else 'OFF'}\n"
        mid_on = bool(self._mid_manager_thread and self._mid_manager_thread.is_alive())
        if mid_on:
            mid_interval = int(self.memory.get_setting(
                "mid_manage_interval_sec", self.MID_MANAGE_DEFAULT_INTERVAL_SEC))
            report += (f"Mid-Management: ON (checking every "
                       f"{max(15, min(mid_interval, 3600))}s)\n")
        else:
            report += "Mid-Management: OFF\n"
        report += f"Cooldowns: {self._get_cooldown_status(symbol)}\n\n"

        report += "--- SETTINGS ---\n"
        report += f"Leverage: {settings.get('leverage', Config.DEFAULT_LEVERAGE)}x\n"
        report += f"Margin Mode: {settings.get('margin_mode', Config.DEFAULT_MARGIN_MODE)}\n"
        report += (f"Timeframes: {settings.get('timeframes', ','.join(Config.DEFAULT_TIMEFRAMES))}\n")
        report += f"Margin Amount: {settings.get('margin_amount_pct', Config.DEFAULT_MARGIN_AMOUNT_PCT)}%\n"
        report += f"Risk: {settings.get('margin_risk_pct', Config.DEFAULT_RISK_PCT)}%\n"
        report += f"Min Confidence: {settings.get('min_confidence', Config.MIN_CONFIDENCE)}%\n"
        report += f"Position Mode: {settings.get('position_mode', 'margin')}\n"

        try:
            report += self._format_contract_info(symbol)
        except XTError as e:
            report += f"Contract info unavailable: {e}\n"

        open_trades = self.memory.get_open_trades()
        if open_trades:
            report += "\n--- OPEN POSITIONS ---\n"
            # Fetch all exchange positions once and reuse them per trade,
            # instead of an N+1 pattern of one API call per open position.
            positions_batch = self.position_mgr.get_positions_batch_optimized()
            for t in open_trades:
                pos = self.position_mgr.get_position_pnl_optimized(
                    t["symbol"], t["position_side"], positions_batch=positions_batch)
                if not pos["exists"]:
                    report += (f"ID:{t['id']} {t['symbol']} {t['position_side']} "
                               f"NOT FOUND ON EXCHANGE (stale)\n")
                    continue
                report += (f"ID:{t['id']} {t['symbol']} {t['position_side']} "
                           f"Entry:{pos['entry_price']} Mark:{pos['mark_price']} "
                           f"Size:{int(pos['position_size'])}c "
                           f"PnL:{pos['unrealized_pnl']:.4f} ROI:{pos['roi']:.2f}% "
                           f"Lev:{pos['leverage']}x | {t['strategy']} "
                           f"Conf:{t['confidence']}%\n")
        return report

    def _format_contract_info(self, symbol: str) -> str:
        cs = self.risk.get_contract_size(symbol)
        return (f"\n--- CONTRACT ({symbol}) ---\n"
                f"Contract Size: {cs} | Min Qty: {self.risk.get_min_qty(symbol)} contracts\n"
                f"Min Notional: {self.risk.get_min_notional(symbol)} USDT | "
                f"Max Leverage: {self.risk.get_max_leverage(symbol)}x\n")

    def _get_cooldown_status(self, symbol: str) -> str:
        status = []
        for side in ["LONG", "SHORT"]:
            if self.memory.is_in_cooldown(symbol, side):
                rem = self.memory.get_cooldown_remaining(symbol, side)
                status.append(f"{side}: {rem:.0f}s remaining")
        return ", ".join(status) if status else "none"

    # ---------- signal gate ----------

    def _gate_checks(self, symbol: str, direction: str) -> str:
        """Returns a rejection reason, or None when the trade may proceed."""
        max_positions = int(self.memory.get_setting("max_positions", Config.MAX_POSITIONS))
        open_trades = self.memory.get_open_trades(symbol)
        if len(open_trades) >= max_positions:
            return f"max positions ({max_positions}) reached for {symbol}"
        if self.memory.is_in_cooldown(symbol, direction):
            rem = self.memory.get_cooldown_remaining(symbol, direction)
            return f"cooldown active - {rem:.0f}s remaining"
        for trade in open_trades:
            if trade["position_side"] == direction:
                return f"already have an open {direction} position for {symbol}"
        return None

    def scan_and_execute(self) -> dict:
        symbol = self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL)
        min_conf = int(self.memory.get_setting("min_confidence", Config.MIN_CONFIDENCE))
        result = self.scanner.scan_and_report(symbol)
        report = self.scanner.format_signal_report(result)
        direction = result["direction"]
        confidence = result["confidence"]

        if direction == "NEUTRAL" or confidence < min_conf:
            logger.info(f"No actionable signal for {symbol}: {direction} at {confidence}%")
            return {"action": "none", "report": report, "result": result}

        reason = self._gate_checks(symbol, direction)
        if reason:
            logger.info(f"Signal suppressed for {symbol}: {reason}")
            return {"action": "blocked", "reason": reason, "report": report, "result": result}

        logger.info(f"Signal to {direction} {symbol} at {confidence}% confidence")
        self._notify(f"Signal: {direction} {symbol} at {confidence}% "
                     f"[strength {result['signal_strength']:.2f}]")
        return {"action": "signal", "direction": direction, "report": report, "result": result}

    # ---------- scan + reversal ----------

    def _scan_cycle(self):
        """One scan, reused for both entry gating and reversal detection.

        Scanning once per cycle (instead of a separate scan for entry and a
        separate one for reversal) halves the kline calls on every interval.
        """
        symbol = self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL)
        min_conf = int(self.memory.get_setting("min_confidence", Config.MIN_CONFIDENCE))
        result = self.scanner.scan_and_report(symbol)
        report = self.scanner.format_signal_report(result)
        direction = result["direction"]
        confidence = result["confidence"]

        # 1) Reversal-close any open position facing an opposite signal.
        self._reversal_check(result)

        # 2) Entry when the signal is aligned and the gate passes.
        if direction == "NEUTRAL" or confidence < min_conf:
            logger.info(f"No actionable entry signal for {symbol}: "
                        f"{direction} at {confidence}%")
            return
        reason = self._gate_checks(symbol, direction)
        if reason:
            logger.info(f"Entry suppressed for {symbol}: {reason}")
            return
        logger.info(f"Signal to {direction} {symbol} at {confidence}% confidence")
        self._notify(f"Signal: {direction} {symbol} at {confidence}% "
                     f"[strength {result['signal_strength']:.2f}]")
        outcome = self.execute_trade(
            direction,
            self._decide_order_type(result),
            self._decide_time_in_force(result),
        )
        logger.info(f"Auto-trade result: {outcome}")
        # Notify the user if the trade was rejected after the signal was sent.
        if outcome and ("Cannot" in outcome or "rejected" in outcome.lower()
                        or "does not support" in outcome.lower()
                        or "could not" in outcome.lower()
                        or "not fill" in outcome.lower()):
            self._notify(f"Trade execution failed: {outcome}")

    def _reversal_check(self, result: dict):
        """Close an open position when the scanner produces the opposite
        direction at or above the reversal confidence threshold."""
        enabled = str(self.memory.get_setting("reversal_enabled",
                                              Config.REVERSAL_ENABLED)).lower()
        if enabled not in ("1", "true", "yes", "on"):
            return
        threshold = int(self.memory.get_setting("reversal_confidence",
                                                Config.REVERSAL_CONFIDENCE))
        if threshold <= 0:
            return

        direction = result.get("direction")
        confidence = result.get("confidence", 0)
        symbol = result.get("symbol")
        if not direction or direction == "NEUTRAL" or confidence < threshold:
            return

        for trade in self.memory.get_open_trades(symbol):
            side = trade["position_side"]
            opposite = "SHORT" if side == "LONG" else "LONG"
            if direction != opposite:
                continue
            logger.info(f"Reversal: open {side} {symbol} vs {direction} "
                        f"{confidence}% >= {threshold}% -> closing")
            ok, _, err = self.position_mgr.close_position(
                symbol, side, trade["id"])
            if ok:
                self._notify(
                    f"REVERSAL CLOSE: {side} {symbol} closed - opposite "
                    f"{direction} signal at {confidence}%")
            else:
                self._notify(f"REVERSAL CLOSE FAILED for {side} {symbol}: {err}")

    def periodic_pnl_report(self) -> str:
        """PnL + confidence snapshot pushed on a timer (see report_interval_sec)."""
        symbol = self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL)
        min_conf = int(self.memory.get_setting("min_confidence", Config.MIN_CONFIDENCE))
        try:
            result = self.scanner.scan_and_report(symbol)
            direction = result.get("direction")
            confidence = result.get("confidence", 0)
            aligned = (direction != "NEUTRAL" and confidence >= min_conf)
            head = (f"[report] {symbol} | signal={direction} "
                    f"conf={confidence}% ({'ALIGNED' if aligned else 'not aligned'})")
        except Exception as e:
            head = f"[report] {symbol} | scan error: {e}"

        open_trades = self.memory.get_open_trades()
        if not open_trades:
            return head + "\nNo open position.\n"
        lines = [head]
        for t in open_trades:
            pos = self.position_mgr.get_position_pnl(t["symbol"], t["position_side"])
            if not pos["exists"]:
                lines.append(
                    f"  ID:{t['id']} {t['symbol']} {t['position_side']}: "
                    f"not on exchange (stale)")
                continue
            lines.append(
                f"  PnL: {pos['unrealized_pnl']:+.4f} USDT | {t['position_side']} "
                f"{int(pos['position_size'])}c @ {pos['entry_price']:.6f} | "
                f"mark={pos['mark_price']:.6f} (ROI {pos['roi']:+.2f}%) "
                f"Lev {pos['leverage']}x")
        return "\n".join(lines) + "\n"

    # ---------- execution ----------

    def execute_trade(self, direction: str, order_type: str = "MARKET",
                      time_in_force: str = None) -> str:
        symbol = self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL)
        min_conf = int(self.memory.get_setting("min_confidence", Config.MIN_CONFIDENCE))
        requested_leverage = int(self.memory.get_setting("leverage", Config.DEFAULT_LEVERAGE))
        margin_mode = self.memory.get_setting("margin_mode", Config.DEFAULT_MARGIN_MODE)

        if direction not in ("LONG", "SHORT"):
            return f"Invalid direction: {direction}"

        reason = self._gate_checks(symbol, direction)
        if reason:
            return f"Cannot open trade: {reason}"

        try:
            if not self.risk.supports_order_type(symbol, order_type):
                return f"{symbol} does not support {order_type} orders"
        except XTError as e:
            return f"Could not read contract config for {symbol}: {e}"

        scan = self.scanner.scan_and_report(symbol)
        if scan["confidence"] < min_conf:
            return (f"Confidence {scan['confidence']}% is below threshold {min_conf}%.\n"
                    f"{self.scanner.format_signal_report(scan)}")
        if scan["direction"] != direction:
            return (f"Current signal direction is {scan['direction']}, not {direction}.\n"
                    f"Check /signal")

        price = scan["price"]
        if price <= 0:
            return "Could not get current price from XT. Aborting."

        confidence = scan["confidence"]
        strength = scan["signal_strength"]

        # Calculate provisional leverage (without notional)
        provisional_leverage = self.risk.validate_leverage(symbol, requested_leverage)

        # Calculate provisional TP/SL to get stop loss for risk-based position sizing
        provisional_tp_price, provisional_sl_price = self.position_mgr.calculate_dynamic_tpsl(
            symbol, direction, price, strength, confidence, provisional_leverage)

        # Calculate initial position size
        # For margin mode: we use provisional leverage (will be corrected later)
        # For risk mode: we need the stop loss price (doesn't depend on leverage)
        qty, size_mode, size_reason = self.risk.calculate_position_size(
            symbol, price, provisional_leverage, provisional_sl_price, order_type)
        if qty <= 0:
            return f"Cannot size position: {size_reason}"

        # Calculate notional from initial size
        notional = self.risk.contracts_to_notional(symbol, qty, price)

        # Validate leverage with actual notional to get the correct leverage
        leverage = self.risk.validate_leverage(symbol, requested_leverage, notional)

        # Recalculate position size if needed
        # For margin mode: position size depends on leverage, so recalculate with correct leverage
        # For risk mode: position size does NOT depend on leverage, so keep the initial size
        if self.memory.get_setting("position_mode", "margin") == "margin":
            qty, size_mode, size_reason = self.risk.calculate_position_size(
                symbol, price, leverage, provisional_sl_price, order_type)
            if qty <= 0:
                return f"Cannot size position: {size_reason}"

        # Calculate final notional
        notional = self.risk.contracts_to_notional(symbol, qty, price)

        # Calculate dynamic TP/SL using the correct leverage (for consistency)
        tp_price, sl_price = self.position_mgr.calculate_dynamic_tpsl(
            symbol, direction, price, strength, confidence, leverage)

        if margin_mode in ("CROSSED", "ISOLATED"):
            try:
                self.xt.set_position_type(symbol, direction, margin_mode)
            except XTError as e:
                logger.info(f"Margin mode unchanged for {symbol} {direction}: {e}")

        try:
            self.xt.set_leverage(symbol, direction, leverage)
        except XTError as e:
            return f"Could not set leverage to {leverage}x: {e}"

        if time_in_force is None:
            time_in_force = "IOC" if order_type == "MARKET" else "GTC"
        if not self.risk.supports_time_in_force(symbol, time_in_force):
            return f"{symbol} does not support timeInForce={time_in_force}"

        limit_price = self.risk.round_price(symbol, price) if order_type == "LIMIT" else None

        logger.info(f"Opening {direction} {symbol}: {qty} contracts "
                    f"(~{notional:.2f} USDT) at {price} lev={leverage}x "
                    f"tp={tp_price} sl={sl_price} conf={confidence}% mode={size_mode}")

        try:
            order_data = self.xt.create_order(
                symbol=symbol, position_side=direction,
                order_side="BUY" if direction == "LONG" else "SELL",
                order_type=order_type, orig_qty=qty, price=limit_price,
                time_in_force=time_in_force,
            )
        except XTError as e:
            return f"Order rejected by XT: {e}"

        self.risk.invalidate_balance_cache()
        order_id = self._extract_order_id(order_data)

        # XT's create-order response carries no fill price, so read it back from
        # the position rather than recording the pre-trade quote as the entry.
        entry_price = price
        filled_qty = 0
        for _ in range(3):
            pos = self.position_mgr.get_position_pnl(symbol, direction)
            if pos["exists"] and pos["position_size"] > 0:
                if pos["entry_price"] > 0:
                    entry_price = pos["entry_price"]
                filled_qty = int(round(pos["position_size"]))
                break
            # Non-blocking wait so a shutdown/stop signal interrupts the poll
            # instead of a hard sleep holding the worker thread up to 3s.
            self._stop_monitor.wait(1.0)

        if filled_qty <= 0:
            # A LIMIT order can rest unfilled. Leaving it open with no stop is
            # the exact situation that produced an unprotected position before.
            try:
                if order_id:
                    self.xt.cancel_order(order_id)
                    cancel_note = "unfilled order cancelled"
                else:
                    self.xt.cancel_all_orders(symbol)
                    cancel_note = "open orders cancelled"
            except XTError as e:
                cancel_note = f"could not cancel order: {e}"
            msg = (f"{order_type} {direction} {symbol} did not fill "
                   f"({qty} contracts requested). {cancel_note}. No position opened.")
            self._notify(msg)
            return msg

        # Recompute TP/SL against the real fill if it drifted from the quote.
        if abs(entry_price - price) / price > 0.001:
            tp_price, sl_price = self.position_mgr.calculate_dynamic_tpsl(
                symbol, direction, entry_price, strength, confidence, leverage)

        strategies_used = ",".join(scan.get("strategies_used", []))
        trade_id = self.memory.record_trade(
            symbol=symbol, position_side=direction, order_id=order_id,
            entry_price=entry_price, amount=filled_qty, leverage=leverage,
            confidence=confidence, strategy=strategies_used,
            signal_strength=strength,
            timeframe=self.memory.get_setting(
                "timeframes", ",".join(Config.DEFAULT_TIMEFRAMES)),
        )

        ok, protected_qty, _tpsl_pid, tpsl_error = self.position_mgr.attach_tpsl_to_position(
            symbol=symbol, position_side=direction,
            trigger_profit_price=tp_price, trigger_stop_price=sl_price,
        )
        tpsl_status = f"set on {protected_qty} contracts" if ok else f"FAILED: {tpsl_error}"

        liq_distance = self.position_mgr.liquidation_distance(entry_price, leverage)
        sl_distance = abs(entry_price - sl_price)

        summary = (f"Trade ID:{trade_id} OPENED {direction} {symbol}\n"
                   f"Entry: {entry_price} | Size: {filled_qty} contracts "
                   f"(~{self.risk.contracts_to_notional(symbol, filled_qty, entry_price):.2f} USDT)\n"
                   f"Leverage: {leverage}x | TP: {tp_price} | SL: {sl_price}\n"
                   f"SL is {sl_distance / entry_price * 100:.2f}% away | "
                   f"liquidation ~{liq_distance / entry_price * 100:.2f}% away\n"
                   f"Confidence: {confidence}% | Strength: {strength:.2f}\n"
                   f"Strategy: {strategies_used or 'n/a'}\n"
                   f"TP/SL: {tpsl_status} | Sizing: {size_mode}\n"
                   f"Margin Mode: {margin_mode}")
        self._notify(summary)

        if not ok:
            fail_action = self.memory.get_setting("on_tpsl_failure", "close")
            if fail_action == "close":
                self._notify(f"No stop loss could be placed on {direction} {symbol}. "
                             f"Closing the position immediately rather than leaving "
                             f"{leverage}x exposure unprotected.")
                closed, _, close_err = self.position_mgr.close_position(
                    symbol, direction, trade_id)
                if closed:
                    return summary + (f"\n\nPOSITION CLOSED: no stop loss could be "
                                      f"placed ({tpsl_error}).")
                self._notify(f"URGENT: {symbol} {direction} has no stop loss AND could "
                             f"not be closed ({close_err}). Close it manually on XT now.")
                return summary + f"\n\nURGENT: unprotected and could not close: {close_err}"
            self._notify(f"WARNING: {symbol} {direction} has NO exchange stop loss. "
                         f"The software stop (max_loss_pct) is the only protection, "
                         f"and it only checks every "
                         f"{self.memory.get_setting('guard_interval_sec', 15)}s.")
        return summary

    @staticmethod
    def _extract_order_id(order_data):
        if isinstance(order_data, dict):
            return order_data.get("orderId") or order_data.get("id")
        if isinstance(order_data, str) and order_data:
            return order_data
        if isinstance(order_data, list) and order_data and isinstance(order_data[0], dict):
            return order_data[0].get("orderId")
        return None

    # ---------- closing ----------

    def close_specific_trade(self, trade_id: int) -> str:
        target = self.memory.get_trade(trade_id)
        if not target:
            return f"Trade ID {trade_id} not found."
        if target["status"] == "CLOSED":
            return f"Trade ID {trade_id} is already closed."

        ok, _, error = self.position_mgr.close_position(
            target["symbol"], target["position_side"], trade_id)
        if not ok:
            return f"Failed to close trade {trade_id}: {error}"

        closed = self.memory.get_trade(trade_id)
        pnl = closed.get("pnl") or 0.0
        self._notify(f"CLOSED {target['position_side']} {target['symbol']}\n"
                     f"Entry: {target['entry_price']} | Exit: {closed.get('exit_price')}\n"
                     f"Realized PnL: {pnl:.4f} USDT | Total: {self.memory.get_total_pnl():.4f} USDT")
        return (f"Closed trade ID:{trade_id} {target['position_side']} {target['symbol']} "
                f"| PnL: {pnl:.4f} USDT")

    def close_all_positions(self) -> str:
        open_trades = self.memory.get_open_trades()
        if not open_trades:
            return "No open positions to close."
        return "\n".join(self.close_specific_trade(t["id"]) for t in open_trades)

    def run_mid_management(self) -> str:
        actions = self._run_mid_cycle()
        if not actions:
            return "No mid-position management actions needed.\n\n" + self.diagnose()
        report = "MID-POSITION MANAGEMENT:\n"
        for a in actions:
            report += f"Trade {a['trade_id']} {a['symbol']}: {a['details']}\n"
        return report

    def protect_open_positions(self) -> str:
        """Attaches an exchange TP/SL to any open position that lacks one."""
        open_trades = self.memory.get_open_trades()
        if not open_trades:
            return "No open trades recorded."
        lines = []
        for t in open_trades:
            profit_id, note = self.position_mgr.ensure_tpsl(
                t["symbol"], t["position_side"],
                signal_strength=t.get("signal_strength") or 0.6,
                confidence=t.get("confidence") or 70,
            )
            status = "OK" if profit_id else "FAILED"
            lines.append(f"{status} trade {t['id']} {t['symbol']} "
                         f"{t['position_side']}: {note}")
        return "\n".join(lines)

    def sync_positions(self) -> str:
        """Pulls open XT positions into the local DB and drops stale rows."""
        lines = []
        adopted = self.position_mgr.adopt_exchange_positions()
        for a in adopted:
            stop = "has stop" if a["has_stop"] else "NO STOP"
            lines.append(f"adopted trade {a['trade_id']}: {a['symbol']} "
                         f"{a['position_side']} {a['size']}c @ {a['entry_price']} "
                         f"{a['leverage']}x ({stop})")
        for c in self.position_mgr.reconcile_open_trades():
            lines.append(f"closed stale trade {c['trade_id']}: {c['symbol']} "
                         f"{c['position_side']} no longer on the exchange")
        if not lines:
            return "In sync: no untracked exchange positions, no stale local trades."
        return "\n".join(lines)

    def diagnose(self) -> str:
        """Explains, per open position, why breakeven/trailing has not acted."""
        out = ["=== MID-MANAGEMENT DIAGNOSTIC ==="]
        try:
            live = self.xt.get_positions()
            live_open = [p for p in live if float(p.get("positionSize") or 0) > 0]
            out.append(f"Exchange reports {len(live_open)} open position(s).")
            for p in live_open:
                out.append(f"  XT: {p.get('symbol')} {p.get('positionSide')} "
                           f"{int(float(p.get('positionSize') or 0))}c "
                           f"profitId={p.get('profitId')}")
        except XTError as e:
            out.append(f"Could not read exchange positions: {e}")

        open_trades = self.memory.get_open_trades()
        out.append(f"Local DB tracks {len(open_trades)} open trade(s).")
        if not open_trades:
            out.append("Nothing is managed, because every guard iterates the local DB. "
                       "Run /sync to adopt exchange positions. If the DB keeps "
                       "emptying on deploy, DATABASE_URL is not pointed at a persistent DB (Neon/MySQL).")
            return "\n".join(out)
        for t in open_trades:
            out.append(self.position_mgr.explain_mid_management(
                t["symbol"], t["position_side"]))
        return "\n".join(out)

    # ---------- safety net ----------

    def check_positions_for_close(self) -> list:
        """Software stop. The exchange TP/SL is primary; this catches TP/SL failures.
        Now dynamic per signal + liq distance, not fixed max_loss/max_profit setting."""
        closed = []
        # Dynamic: max_loss = liq distance * safety, max_profit = dynamic via signal strength
        # Fallback to old settings if present for compat, else use auto
        def get_dynamic_limits(symbol, side, entry, leverage):
            try:
                liq_dist = self.position_mgr.liquidation_distance(entry, leverage)
                liq_pct = (liq_dist / entry * 100) if entry else 40
                sl_pct = max(10.0, min(liq_pct * 0.8, 60.0))  # 80% to liq, capped 10-60%
                # TP: at least 1.5x SL, more for strong signal
                tp_pct = sl_pct * 1.5
                return sl_pct, tp_pct
            except:
                return 40.0, 80.0
        # For compat: if user still has old fixed values and wants them, they are ignored now
        _legacy_sl = float(self.memory.get_setting("max_loss_pct", 0) or 0)
        _legacy_tp = float(self.memory.get_setting("max_profit_pct", 0) or 0)
        use_legacy = _legacy_sl > 0 and _legacy_tp > 0 and False  # disabled - always dynamic now
        for trade in self.memory.get_open_trades():
            symbol = trade["symbol"]
            side = trade["position_side"]
            pos = self.position_mgr.get_position_pnl(symbol, side)
            if not pos["exists"]:
                continue
            roi = pos["roi"]
            sl_pct, _ = get_dynamic_limits(symbol, side, pos.get("entry_price") or trade.get("entry_price") or 0, pos.get("leverage") or trade.get("leverage") or Config.DEFAULT_LEVERAGE)
            # TP has no ceiling - let exchange TP handle profit, software only guards loss
            reason = None
            if roi <= -sl_pct:
                reason = "max_loss"
            if not reason:
                continue
            logger.info(f"{reason} triggered for trade {trade['id']}: ROI {roi:.2f}%")
            ok, _, err = self.position_mgr.close_position(symbol, side, trade["id"])
            if ok:
                closed.append({"trade_id": trade["id"], "reason": reason, "roi": roi})
                self._notify(f"{reason.upper().replace('_', ' ')} triggered - "
                             f"closed {side} {symbol} at ROI {roi:.2f}%")
            else:
                self._notify(f"Failed to close {side} {symbol} on {reason}: {err}")
        return closed

    # ---------- automatic mid-management ----------

    # How often the guardian re-checks every open position for breakeven /
    # trailing. This is a POLLING cadence, not a delay: as soon as ROI crosses
    # the threshold the stop is moved on the very next check, so a smaller
    # value reacts faster (trading on 1m/5m at high leverage wants seconds,
    # not minutes). Floor of 15s matches the guard loop cadence.
    MID_MANAGE_DEFAULT_INTERVAL_SEC = 30

    def start_mid_manager(self) -> bool:
        """Start the always-on mid-management guardian.

        Periodically runs breakeven, trailing stop and TP/SL protection over
        every open position so stops are managed automatically - no manual
        /midmanage (or /autotrade_on) required. Safe to call repeatedly.
        """
        if self._mid_manager_thread and self._mid_manager_thread.is_alive():
            return False
        self._stop_mid_manager.clear()
        self._mid_manager_thread = threading.Thread(
            target=self._mid_manager_loop, daemon=True,
            name="mid-manager-guardian")
        self._mid_manager_thread.start()
        logger.info("Automatic mid-management guardian started "
                    "(breakeven + trailing + TP/SL protection)")
        return True

    def stop_mid_manager(self):
        self._stop_mid_manager.set()
        if self._mid_manager_thread and self._mid_manager_thread.is_alive():
            self._mid_manager_thread.join(timeout=5)
        logger.info("Automatic mid-management guardian stopped")

    def _mid_manager_loop(self):
        while not self._stop_mid_manager.is_set():
            try:
                for action in self._run_mid_cycle():
                    self._notify_mid_action(action)
            except Exception as e:
                logger.error(f"Mid-management cycle error: {e}", exc_info=True)
            interval = int(self.memory.get_setting(
                "mid_manage_interval_sec", self.MID_MANAGE_DEFAULT_INTERVAL_SEC))
            interval = max(15, min(interval, 3600))
            self._stop_mid_manager.wait(interval)

    def _run_mid_cycle(self) -> list:
        """One mid-management pass over all open positions.

        Serialized under _mid_lock so the guardian and a manual /midmanage
        can never run the same exchange updates concurrently.
        """
        with self._mid_lock:
            return self.position_mgr.mid_manage_positions()

    def _notify_mid_action(self, action: dict):
        """Push one mid-management action to Telegram/log so automatic fires
        are visible instead of happening silently."""
        label = {
            "breakeven_activated": "BREAKEVEN",
            "trailing_updated": "TRAILING STOP",
            "tpsl_recovered": "TP/SL RE-ATTACHED",
            "tpsl_missing": "TP/SL PROBLEM",
            "position_adopted": "POSITION ADOPTED",
        }.get(action["action"])
        if not label:
            return
        if action["action"] == "position_adopted" and self._auto_trade_enabled:
            # The auto-trade loop already announces adoptions; avoid a duplicate.
            return
        self._notify(f"{label} {action['symbol']} trade {action['trade_id']}: "
                     f"{action['details']}")

    # ---------- auto trade loop ----------

    def start_auto_trade(self):
        if self._auto_trade_enabled:
            return "Auto-trade is already running."
        self._auto_trade_enabled = True
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(target=self._auto_trade_loop, daemon=True)
        self._monitor_thread.start()
        self._notify("Auto-Trade ENABLED")
        return "Auto-trade started. Bot will scan signals and execute trades automatically."

    def stop_auto_trade(self):
        if not self._auto_trade_enabled:
            return "Auto-trade is not running."
        self._auto_trade_enabled = False
        self._stop_monitor.set()
        self._notify("Auto-Trade DISABLED")
        return "Auto-trade stopped."

    def is_auto_trading(self) -> bool:
        return self._auto_trade_enabled

    def _auto_trade_loop(self):
        logger.info("Auto-trade monitoring loop started")
        last_scan = 0.0
        last_report = 0.0
        # Adoption fetches ALL exchange positions; run it at most every 5
        # minutes instead of on every ~15s loop iteration. Untracked positions
        # are still caught within one adoption window, and reconcile_open_trades
        # (which keys off the local DB) still runs every cycle for safety.
        adoption_interval = 300
        last_adoption = 0.0

        while not self._stop_monitor.is_set():
            scan_interval = int(self.memory.get_setting("scan_interval_sec", 60))
            guard_interval = int(self.memory.get_setting("guard_interval_sec", 15))
            now = time.time()
            try:
                # Adoption runs first: everything below keys off the local DB, so
                # an untracked position would otherwise be invisible to the guard.
                if now - last_adoption >= adoption_interval:
                    last_adoption = now
                    for event in self.position_mgr.adopt_exchange_positions():
                        warn = "" if event["has_stop"] else " It has NO exchange stop loss."
                        self._notify(f"Found untracked position on XT: {event['symbol']} "
                                     f"{event['position_side']} {event['size']}c @ "
                                     f"{event['entry_price']} {event['leverage']}x. "
                                     f"Now managed as trade {event['trade_id']}.{warn}")

                # Runs every guard_interval: cheap, and it is what protects capital.
                for event in self.position_mgr.reconcile_open_trades():
                    self._notify(f"Position {event['symbol']} {event['position_side']} "
                                 f"disappeared from the exchange (trade "
                                 f"{event['trade_id']}). Likely liquidation, external "
                                 f"close, or TP/SL fill. Marked closed locally.")
                self.check_positions_for_close()

                if now - last_scan >= scan_interval:
                    last_scan = now
                    self._scan_cycle()

                # Periodic PnL + confidence report, even with no events, so the
                # user always knows the bot is alive and the position's PnL.
                report_interval = int(self.memory.get_setting("report_interval_sec",
                                                             Config.REPORT_INTERVAL_SEC))
                if report_interval > 0 and now - last_report >= report_interval:
                    last_report = now
                    self._notify(self.periodic_pnl_report())

                # Mid-management (breakeven, trailing stop, TP/SL protection)
                # runs on its own always-on guardian thread (start_mid_manager),
                # independent of auto-trade and without a manual /midmanage.
            except Exception as e:
                logger.error(f"Auto-trade loop error: {e}", exc_info=True)
            self._stop_monitor.wait(guard_interval)

        logger.info("Auto-trade monitoring loop stopped")

    def _decide_order_type(self, signal_result: dict) -> str:
        configured = self.memory.get_setting("ai_order_type", "auto")
        if configured == "always_market":
            return "MARKET"
        if configured == "always_limit":
            return "LIMIT"
        # Auto mode stays on MARKET. A resting LIMIT order leaves the bot unable
        # to attach a stop loss until it fills, which is how a 50x position ended
        # up with no protection.
        return "MARKET"

    def _decide_time_in_force(self, signal_result: dict) -> str:
        configured = self.memory.get_setting("ai_time_in_force", "auto")
        if configured != "auto":
            return configured
        return None
