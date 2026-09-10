import time
import logging
import pandas as pd
from bot.xt_client import XTClient, XTError
from bot.memory import LongTermMemory
from bot.risk_manager import RiskManager
from bot.cache_manager import get_cache
from config import Config

logger = logging.getLogger("xt_position")

# TP/SL order states per XT docs. NOT_USED / USED do not exist.
TPSL_ACTIVE_STATES = ["NOT_TRIGGERED", "TRIGGERING"]


class PositionManager:
    def __init__(self, xt_client: XTClient, memory: LongTermMemory, risk_manager: RiskManager):
        self.xt = xt_client
        self.memory = memory
        self.risk = risk_manager

    # ---------- positions ----------

    def get_positions(self, symbol: str = None) -> list:
        try:
            return self.xt.get_positions(symbol)
        except XTError as e:
            logger.warning(f"Position fetch failed for {symbol}: {e}")
            return []

    def get_position(self, symbol: str, position_side: str) -> dict:
        """Returns {} when the exchange has no such open position."""
        for pos in self.get_positions(symbol):
            if pos.get("positionSide") != position_side:
                continue
            if float(pos.get("positionSize") or 0) <= 0:
                continue
            return pos
        return {}

    def adopt_exchange_positions(self) -> list:
        """Records any position that is open on XT but missing from the local DB.

        Everything else keys off memory.get_open_trades(), so a wiped database
        (SQLite on an ephemeral container) or a manually opened position leaves
        real exposure completely unmanaged.
        """
        adopted = []
        try:
            positions = self.xt.get_positions()
        except XTError as e:
            logger.warning(f"Could not enumerate exchange positions: {e}")
            return adopted

        known = {(t["symbol"], t["position_side"]) for t in self.memory.get_open_trades()}
        for pos in positions:
            size = float(pos.get("positionSize") or 0)
            if size <= 0:
                continue
            symbol = pos.get("symbol")
            side = pos.get("positionSide")
            if not symbol or not side or (symbol, side) in known:
                continue
            entry = float(pos.get("entryPrice") or 0)
            leverage = int(float(pos.get("leverage") or 1))
            trade_id = self.memory.record_trade(
                symbol=symbol, position_side=side, order_id=None,
                entry_price=entry, amount=int(size), leverage=leverage,
                confidence=0, strategy="ADOPTED", signal_strength=0.0,
                timeframe="",
            )
            logger.warning(f"Adopted untracked position {symbol} {side} "
                           f"{int(size)}c @ {entry} as trade {trade_id}")
            adopted.append({"trade_id": trade_id, "symbol": symbol,
                            "position_side": side, "size": int(size),
                            "entry_price": entry, "leverage": leverage,
                            "has_stop": bool(pos.get("profitId"))})
        return adopted

    def _public_mark_price(self, symbol: str) -> float:
        try:
            return float(self.xt.get_mark_price(symbol).get("p") or 0)
        except XTError as e:
            logger.warning(f"Mark price fallback failed for {symbol}: {e}")
            return 0.0

    def get_position_pnl(self, symbol: str, position_side: str) -> dict:
        pos = self.get_position(symbol, position_side)
        if not pos:
            return {"exists": False, "unrealized_pnl": 0.0, "roi": 0.0,
                    "entry_price": 0.0, "mark_price": 0.0, "leverage": 1,
                    "position_size": 0, "margin": 0.0, "profit_id": None,
                    "trigger_profit_price": 0.0, "trigger_stop_price": 0.0}
        entry = float(pos.get("entryPrice") or 0)
        mark = float(pos.get("calMarkPrice") or 0)
        if mark <= 0:
            # Without a mark price every ROI is 0, which silently disables
            # breakeven and trailing regardless of their thresholds.
            mark = self._public_mark_price(symbol)
        size = float(pos.get("positionSize") or 0)
        leverage = int(float(pos.get("leverage") or 1))
        # floatingPL is often 0/missing on XT for some symbols (e.g. uai_usdt) - calculate from price diff
        raw_pnl = float(pos.get("floatingPL") or pos.get("unrealizedProfit") or pos.get("profit") or 0)
        margin = float(pos.get("isolatedMargin") or 0)
        cs = self.risk.get_contract_size(symbol) or 1.0
        # ROI on margin: price move as a fraction of entry, amplified by leverage.
        roi = 0.0
        if entry > 0 and mark > 0:
            move = (mark - entry) / entry
            if position_side == "SHORT":
                move = -move
            roi = move * leverage * 100
        # Fallback PnL calc if exchange returns 0 but ROI is clearly non-zero (bug for uai_usdt)
        pnl = raw_pnl
        if abs(pnl) < 1e-9 and abs(roi) > 0.1 and size > 0 and cs > 0:
            # PnL = price_diff * size * contractSize
            price_diff = (mark - entry) if position_side == "LONG" else (entry - mark)
            pnl = price_diff * size * cs
        return {
            "exists": True,
            "unrealized_pnl": pnl,
            "roi": roi,
            "entry_price": entry,
            "mark_price": mark,
            "leverage": leverage,
            "position_size": size,
            "position_value": size * cs * mark,
            "margin": margin,
            "profit_id": pos.get("profitId") or None,
            "trigger_profit_price": float(pos.get("triggerProfitPrice") or 0),
            "trigger_stop_price": float(pos.get("triggerStopPrice") or 0),
            "position_type": pos.get("positionType") or "",
            "available_close_size": float(pos.get("availableCloseSize") or 0),
        }

    def get_positions_batch_optimized(self, symbol: str = None) -> dict:
        """Fetch ALL positions once, indexed by (symbol, position_side).

        Replaces the N+1 query pattern where get_position_pnl() calls
        get_position() -> get_positions() once per open trade. Call this once
        and pass the result to get_position_pnl_optimized() to drop 10+ API
        calls down to 1 for status / report loops.

        Returns {(symbol, position_side): position_dict, ...} for open positions.
        """
        try:
            all_positions = self.xt.get_positions(symbol)
        except XTError as e:
            logger.warning(f"Position batch fetch failed: {e}")
            return {}
        result = {}
        for pos in all_positions:
            sym = pos.get("symbol")
            side = pos.get("positionSide")
            size = float(pos.get("positionSize") or 0)
            if sym and side and size > 0:
                result[(sym, side)] = pos
        return result

    def get_position_pnl_optimized(self, symbol: str, position_side: str,
                                   positions_batch: dict = None) -> dict:
        """get_position_pnl reusing a pre-fetched batch of positions.

        When positions_batch is provided, no extra API call is made; otherwise
        it falls back to the single-position fetch (original behavior).
        """
        pos = None
        if positions_batch is not None:
            pos = positions_batch.get((symbol, position_side))
        else:
            pos = self.get_position(symbol, position_side)
        if not pos:
            return {"exists": False, "unrealized_pnl": 0.0, "roi": 0.0,
                    "entry_price": 0.0, "mark_price": 0.0, "leverage": 1,
                    "position_size": 0, "margin": 0.0, "profit_id": None,
                    "trigger_profit_price": 0.0, "trigger_stop_price": 0.0}
        entry = float(pos.get("entryPrice") or 0)
        mark = float(pos.get("calMarkPrice") or 0)
        if mark <= 0:
            mark = self._public_mark_price(symbol)
        size = float(pos.get("positionSize") or 0)
        leverage = int(float(pos.get("leverage") or 1))
        raw_pnl = float(pos.get("floatingPL") or pos.get("unrealizedProfit") or pos.get("profit") or 0)
        margin = float(pos.get("isolatedMargin") or 0)
        cs = self.risk.get_contract_size(symbol) or 1.0
        roi = 0.0
        if entry > 0 and mark > 0:
            move = (mark - entry) / entry
            if position_side == "SHORT":
                move = -move
            roi = move * leverage * 100
        pnl = raw_pnl
        if abs(pnl) < 1e-9 and abs(roi) > 0.1 and size > 0 and cs > 0:
            price_diff = (mark - entry) if position_side == "LONG" else (entry - mark)
            pnl = price_diff * size * cs
        return {
            "exists": True,
            "unrealized_pnl": pnl,
            "roi": roi,
            "entry_price": entry,
            "mark_price": mark,
            "leverage": leverage,
            "position_size": size,
            "position_value": size * cs * mark,
            "margin": margin,
            "profit_id": pos.get("profitId") or None,
            "trigger_profit_price": float(pos.get("triggerProfitPrice") or 0),
            "trigger_stop_price": float(pos.get("triggerStopPrice") or 0),
            "position_type": pos.get("positionType") or "",
            "available_close_size": float(pos.get("availableCloseSize") or 0),
        }

    # ---------- take profit / stop loss ----------

    def set_stop_loss_take_profit(self, symbol: str, position_side: str, contracts: int,
                                  trigger_profit_price: float, trigger_stop_price: float,
                                  expire_time_ms: int = None) -> tuple:
        if expire_time_ms is None:
            expire_time_ms = int(time.time() * 1000) + 86400 * 7 * 1000
        try:
            data = self.xt.create_tpsl(
                symbol=symbol, position_side=position_side, orig_qty=contracts,
                trigger_profit_price=trigger_profit_price,
                trigger_stop_price=trigger_stop_price,
                expire_time_ms=expire_time_ms,
            )
            return True, data, None
        except XTError as e:
            logger.error(f"TP/SL creation failed for {symbol} {position_side}: {e}")
            return False, None, str(e)

    def attach_tpsl_to_position(self, symbol: str, position_side: str,
                                trigger_profit_price: float, trigger_stop_price: float,
                                attempts: int = 3, delay: float = 1.0) -> tuple:
        """Sizes the TP/SL from the live position instead of the requested order
        quantity. A partially filled or unfilled order otherwise produces
        'more than available'."""
        last_error = "no position to protect"
        for attempt in range(1, attempts + 1):
            pos = self.get_position_pnl(symbol, position_side)
            available = int(pos.get("available_close_size") or 0)
            if available <= 0:
                size = int(pos.get("position_size") or 0)
                available = size
            if available > 0:
                ok, data, error = self.set_stop_loss_take_profit(
                    symbol, position_side, available,
                    trigger_profit_price, trigger_stop_price)
                if ok:
                    # XT's create-profit returns the new profitId directly; the
                    # caller needs it so it never has to re-discover the order
                    # (the position's profitId field can lag or read empty).
                    return True, available, data, None
                last_error = error
            else:
                last_error = "position not filled yet"
            if attempt < attempts:
                time.sleep(delay)
        return False, 0, None, last_error

    def _get_profit_id(self, symbol: str, position_side: str, pos: dict):
        """The id of the position's attached TP/SL entrust.

        Reads it from the position object first, then falls back to the active
        entrust list. The position's ``profitId`` field can be empty on XT even
        when a profit entrust exists; relying on it alone made the bot re-create
        TP/SL orders on every management cycle and blocked breakeven/trailing
        (both need a profitId to move the stop).
        """
        pid = pos.get("profit_id")
        if pid:
            return pid
        return self.find_tpsl(symbol, position_side).get("profitId")

    def ensure_tpsl(self, symbol: str, position_side: str,
                    trigger_stop_price: float = None,
                    trigger_profit_price: float = None,
                    signal_strength: float = 0.6, confidence: int = 70) -> tuple:
        """Guarantees the position has a live exchange TP/SL, creating one when
        the original order was rejected. Returns (profit_id, note)."""
        pos = self.get_position_pnl(symbol, position_side)
        if not pos["exists"]:
            return None, "no open position"
        existing = self._get_profit_id(symbol, position_side, pos)
        if existing:
            return existing, "already protected"

        entry = pos["entry_price"]
        leverage = pos["leverage"]
        if entry <= 0:
            return None, "position has no entry price"

        auto_tp, auto_sl = self.calculate_dynamic_tpsl(
            symbol, position_side, entry, signal_strength, confidence, leverage)
        tp = trigger_profit_price or auto_tp
        sl = trigger_stop_price or auto_sl

        # A stop already beyond the mark price would trigger instantly; pull it
        # to the safe side of the current price instead of letting XT reject it.
        mark = pos["mark_price"] or entry
        safety = float(self.memory.get_setting("sl_liquidation_safety", 0.5))
        max_dist = self.liquidation_distance(entry, leverage) * safety
        if position_side == "LONG":
            safe_sl = self.risk.round_price(symbol, entry - max_dist)
            safe_sl_price = self.risk.round_price(symbol, mark * 0.999)
            if mark <= safe_sl or safe_sl_price <= safe_sl:
                return None, (f"position is already past the safe stop level "
                              f"(mark {mark}, safe_sl {safe_sl}); a stop here would "
                              f"trigger instantly. Close it or widen sl_liquidation_safety.")
            sl = min(max(sl, safe_sl), safe_sl_price)
            tp = max(tp, self.risk.round_price(symbol, mark * 1.001))
        else:
            safe_sl = self.risk.round_price(symbol, entry + max_dist)
            safe_sl_price = self.risk.round_price(symbol, mark * 1.001)
            if mark >= safe_sl or safe_sl_price >= safe_sl:
                return None, (f"position is already past the safe stop level "
                              f"(mark {mark}, safe_sl {safe_sl}); a stop here would "
                              f"trigger instantly. Close it or widen sl_liquidation_safety.")
            sl = max(min(sl, safe_sl), safe_sl_price)
            tp = min(tp, self.risk.round_price(symbol, mark * 0.999))

        ok, contracts, created, error = self.attach_tpsl_to_position(
            symbol, position_side, tp, sl)
        if not ok:
            return None, f"could not create TP/SL: {error}"
        # The create response carries the profitId, but it may arrive wrapped in
        # the result object or lag and read empty. Extract a real profitId
        # string (never the raw response dict) and fall back to the active
        # entrust list for a moment so the returned id is always usable by
        # callers that pass it to cancel/update — keeping this return type
        # consistent with the "already protected" path.
        profit_id = self._extract_profit_id(created)
        for attempt in range(3):
            if profit_id:
                break
            profit_id = self.find_tpsl(symbol, position_side).get("profitId")
            if attempt < 2:
                time.sleep(1.0)
        logger.info(f"Attached TP/SL to existing {symbol} {position_side}: "
                    f"{contracts}c TP={tp} SL={sl} profit_id={profit_id}")
        return profit_id, f"created TP={tp} SL={sl} on {contracts} contracts"

    @staticmethod
    def _extract_profit_id(created) -> str:
        """Pull a profitId string out of a create_tpsl response.

        XT's ``/entrust/create-profit`` returns ``{"result": true}`` — a boolean
        success flag, NOT a profitId. Older code pathed this through
        ``str(True) == "True"`` and callers then passed the literal string
        "True" to cancel/update, which XT rejects with ``invalid_entrust``.

        Normalize to a real id string (or None so the caller can fall back to
        the active entrust list / the position's profitId field).
        """
        if not created:
            return None
        if isinstance(created, bool):
            # XT's success response — there is no id here.
            return None
        if isinstance(created, str):
            return created
        if isinstance(created, dict):
            return created.get("profitId") or created.get("profit_id") or None
        # Anything else (list, int, ...) is not a usable id.
        return None

    def get_active_tpsl(self, symbol: str) -> list:
        orders = []
        for state in TPSL_ACTIVE_STATES:
            try:
                orders.extend(self.xt.get_tpsl_orders(symbol, state=state))
            except XTError as e:
                logger.warning(f"TP/SL list failed for {symbol} state={state}: {e}")
        return orders

    def find_tpsl(self, symbol: str, position_side: str) -> dict:
        for order in self.get_active_tpsl(symbol):
            if order.get("positionSide") == position_side:
                return order
        return {}

    def _get_profit_entrust(self, symbol: str, position_side: str, pos: dict) -> dict:
        """Return the position's live TP/SL entrust WITH its real prices.

        XT's position object frequently reports profitId as empty and
        triggerProfitPrice/triggerStopPrice as 0 even when a profit entrust
        exists - the real values live in the active entrust list. Breakeven
        and trailing must read the entrust: updating a stop while passing
        trigger_profit_price=0 makes XT reject the update, so the stop never
        moves even when ROI is far past the threshold.
        """
        pid = pos.get("profit_id")
        if pid and (pos.get("trigger_profit_price") or 0) > 0:
            # The position object already carries usable prices - no extra call.
            return {
                "profitId": pid,
                "triggerProfitPrice": pos.get("trigger_profit_price"),
                "triggerStopPrice": pos.get("trigger_stop_price"),
            }
        return self.find_tpsl(symbol, position_side)

    def cancel_all_tpsl(self, symbol: str) -> bool:
        try:
            self.xt.cancel_all_tpsl(symbol)
            return True
        except XTError as e:
            logger.warning(f"Cancel TP/SL failed for {symbol}: {e}")
            return False

    def _move_stop(self, symbol: str, profit_id, new_sl: float,
                   keep_tp: float = None) -> bool:
        logger.info(f"MOVE_STOP {symbol}: profit_id={profit_id} new_sl={new_sl} keep_tp={keep_tp}")
        try:
            self.xt.update_tpsl(profit_id=profit_id, trigger_stop_price=new_sl,
                                trigger_profit_price=keep_tp if keep_tp is not None else None)
            logger.info(f"MOVE_STOP OK: {symbol} profit_id={profit_id}")
            return True
        except XTError as e:
            logger.warning(f"Move stop failed for {symbol} (profitId={profit_id}): {e}")
            return False

    def _breakeven_threshold(self) -> float:
        """ROI (on margin) at which the stop is dragged to entry.

        The stored value is trusted only inside a sane band: a threshold of
        30-50% (from the Aug 2026 config change) never fires before the
        ~1% stop, so every trade rode to a full loss. Clamp to the fixed
        default when the DB value is outside 1..25% ROI.
        """
        raw = float(self.memory.get_setting(
            "breakeven_threshold_pct", Config.BREAKEVEN_THRESHOLD_PCT))
        if raw <= 0 or raw > 25:
            return float(Config.BREAKEVEN_THRESHOLD_PCT)
        return raw

    def check_tpsl_breakeven(self, symbol: str, position_side: str) -> bool:
        pos = self.get_position_pnl(symbol, position_side)
        if not pos["exists"]:
            return False
        threshold = self._breakeven_threshold()
        if pos["roi"] < threshold:
            logger.debug(f"Breakeven skipped {symbol} {position_side}: "
                         f"ROI {pos['roi']:.2f}% < {threshold}%")
            return False
        entry = pos["entry_price"]
        if entry <= 0:
            return False
        entrust = self._get_profit_entrust(symbol, position_side, pos)
        profit_id = entrust.get("profitId")
        if not profit_id:
            logger.warning(f"Breakeven blocked for {symbol} {position_side}: "
                           f"no active TP/SL entrust found (position profitId "
                           f"empty and entrust list has none)")
            return False
        # Read the REAL stop/take prices from the entrust: the position object
        # reports 0 here, and passing trigger_profit_price=0 makes XT reject
        # the update (the bug that silently blocked every breakeven move).
        current_sl = float(entrust.get("triggerStopPrice") or 0)
        current_tp = float(entrust.get("triggerProfitPrice") or 0)
        # Only ever tighten. A manually placed stop that is already better than
        # breakeven must not be dragged back toward entry.
        #
        # A stop-loss is a *liquidation* order: for a LONG it sits BELOW entry
        # (sell when price drops) and for a SHORT it sits ABOVE entry (buy when
        # price rises). Moving the stop to entry means dragging it TOWARD the
        # current price, so:
        #   LONG:  new_sl = entry * 1.0005  (a small step ABOVE entry)
        #   SHORT: new_sl = entry * 0.9995  (a small step BELOW entry)
        # Tightening means the LONG stop rises (new_sl > current_sl) and the
        # SHORT stop falls (new_sl < current_sl).
        if position_side == "LONG":
            new_sl = self.risk.round_price(symbol, entry * 1.0005)
            if current_sl >= new_sl:
                logger.debug(f"Breakeven skip {symbol} LONG: current_sl={current_sl} >= new_sl={new_sl}")
                return False
        else:
            new_sl = self.risk.round_price(symbol, entry * 0.9995)
            if 0 < current_sl <= new_sl:
                logger.debug(f"Breakeven skip {symbol} SHORT: current_sl={current_sl} <= new_sl={new_sl}")
                return False
        logger.info(f"Breakeven {symbol} {position_side}: ROI={pos['roi']:.2f}% "
                    f"current_sl={current_sl} -> new_sl={new_sl} profit_id={profit_id}")
        keep_tp = current_tp if current_tp > 0 else None
        if self._move_stop(symbol, profit_id, new_sl, keep_tp):
            logger.info(f"Breakeven: {symbol} {position_side} ROI {pos['roi']:.2f}% "
                        f"SL {current_sl} -> {new_sl}")
            return True
        logger.warning(f"Breakeven move REJECTED for {symbol} {position_side}: "
                       f"new_sl={new_sl} profit_id={profit_id}")
        return False

    def trail_stop_loss(self, symbol: str, position_side: str) -> tuple:
        pos = self.get_position_pnl(symbol, position_side)
        if not pos["exists"]:
            return False, "no open position", None
        # Trigger is ROI on margin; distance is a raw price percentage. They were
        # previously the same setting, which made "trailing 2%" mean two things.
        trigger_roi = float(self.memory.get_setting("trailing_trigger_roi_pct", 0) or 0)
        distance_pct = float(self.memory.get_setting("trailing_distance_pct", 0) or 0)
        legacy = float(self.memory.get_setting("trailing_stop_pct", 2.0))
        # Old DBs stored a single legacy knob (10-50%). Treat an absurd value
        # as missing so a 50% "distance" cannot turn the trailing stop into a
        # no-op that never tightens (the SL sits 50% behind the mark).
        if not (0 < legacy < 20):
            legacy = float(Config.TRAILING_STOP_PCT)
        if trigger_roi <= 0 or trigger_roi > 100:
            trigger_roi = legacy
        if distance_pct <= 0 or distance_pct >= 20:
            distance_pct = float(Config.TRAILING_DISTANCE_PCT)
        if pos["roi"] < trigger_roi:
            return False, (f"ROI {pos['roi']:.2f}% below trailing trigger "
                           f"{trigger_roi}%"), None
        entrust = self._get_profit_entrust(symbol, position_side, pos)
        profit_id = entrust.get("profitId")
        if not profit_id:
            return False, "no active TP/SL entrust found (cannot move the stop)", None
        mark = pos["mark_price"]
        if mark <= 0:
            return False, "no mark price available", None
        # Real entrust prices, not the position object's zeroed fields.
        current_sl = float(entrust.get("triggerStopPrice") or 0)
        current_tp = float(entrust.get("triggerProfitPrice") or 0)
        if position_side == "LONG":
            new_sl = self.risk.round_price(symbol, mark * (1 - distance_pct / 100))
            improved = new_sl > current_sl
        else:
            new_sl = self.risk.round_price(symbol, mark * (1 + distance_pct / 100))
            improved = current_sl <= 0 or new_sl < current_sl
        if not improved:
            return False, f"no improvement (SL already {current_sl})", None
        keep_tp = current_tp if current_tp > 0 else None
        if self._move_stop(symbol, profit_id, new_sl, keep_tp):
            logger.info(f"Trailing: {symbol} {position_side} ROI {pos['roi']:.2f}% "
                        f"SL {current_sl} -> {new_sl}")
            return True, f"Trailing SL {current_sl} -> {new_sl}", new_sl
        return False, "stop update rejected by exchange", None

    def explain_mid_management(self, symbol: str, position_side: str) -> str:
        """Why breakeven/trailing did or did not act, for /diag."""
        pos = self.get_position_pnl(symbol, position_side)
        if not pos["exists"]:
            return f"{symbol} {position_side}: no open position on the exchange."
        be_threshold = float(self.memory.get_setting("breakeven_threshold_pct", 1.5))
        legacy = float(self.memory.get_setting("trailing_stop_pct", 2.0))
        trigger_roi = float(self.memory.get_setting("trailing_trigger_roi_pct", 0) or 0) or legacy
        distance_pct = float(self.memory.get_setting("trailing_distance_pct", 0) or 0) or legacy
        entrust = self._get_profit_entrust(symbol, position_side, pos)
        profit_id = entrust.get("profitId")
        entrust_sl = float(entrust.get("triggerStopPrice") or 0)
        entrust_tp = float(entrust.get("triggerProfitPrice") or 0)
        lines = [
            f"{symbol} {position_side}",
            f"  entry={pos['entry_price']} mark={pos['mark_price']} "
            f"lev={pos['leverage']}x size={int(pos['position_size'])}c",
            f"  ROI on margin = {pos['roi']:.2f}%",
            f"  exchange SL={entrust_sl} TP={entrust_tp} profitId={profit_id}",
            f"  breakeven fires at ROI >= {be_threshold}% -> "
            f"{'READY' if pos['roi'] >= be_threshold else 'not yet'}",
            f"  trailing fires at ROI >= {trigger_roi}% (distance {distance_pct}% of price) -> "
            f"{'READY' if pos['roi'] >= trigger_roi else 'not yet'}",
        ]
        if not profit_id:
            lines.append("  BLOCKED: no active TP/SL entrust found, so no stop can "
                         "be moved. The TP/SL order was never accepted for this "
                         "position.")
        if pos["mark_price"] <= 0:
            lines.append("  BLOCKED: exchange returned no mark price, so ROI reads 0.")
        return "\n".join(lines)

    def mid_manage_positions(self) -> list:
        actions = []
        for event in self.adopt_exchange_positions():
            actions.append({
                "trade_id": event["trade_id"], "symbol": event["symbol"],
                "action": "position_adopted",
                "details": (f"{event['position_side']} {event['size']}c @ "
                            f"{event['entry_price']} {event['leverage']}x was open on "
                            f"XT but untracked; now managed"
                            f"{'' if event['has_stop'] else ' (NO exchange stop)'}"),
            })
        for trade in self.memory.get_open_trades():
            symbol = trade["symbol"]
            side = trade["position_side"]
            # Breakeven and trailing can only move an existing stop, so a
            # position whose TP/SL order was rejected must get one first.
            profit_id, note = self.ensure_tpsl(
                symbol, side,
                signal_strength=trade.get("signal_strength") or 0.6,
                confidence=trade.get("confidence") or 70,
            )
            if profit_id and note.startswith("created"):
                actions.append({"trade_id": trade["id"], "symbol": symbol,
                                "action": "tpsl_recovered", "details": note})
            elif not profit_id and note not in ("no open position",):
                actions.append({"trade_id": trade["id"], "symbol": symbol,
                                "action": "tpsl_missing", "details": note})
            if self.check_tpsl_breakeven(symbol, side):
                actions.append({"trade_id": trade["id"], "symbol": symbol,
                                "action": "breakeven_activated",
                                "details": "Stop loss moved to entry"})
            trailed, msg, _ = self.trail_stop_loss(symbol, side)
            if trailed:
                actions.append({"trade_id": trade["id"], "symbol": symbol,
                                "action": "trailing_updated", "details": msg})
        return actions

    # ---------- dynamic TP/SL ----------

    def liquidation_distance(self, entry_price: float, leverage: int) -> float:
        """Approximate adverse price move that wipes the margin."""
        return entry_price / max(1, leverage)

    def calculate_dynamic_tpsl(self, symbol: str, position_side: str, entry_price: float,
                               signal_strength: float, confidence: int,
                               leverage: int = 1) -> tuple:
        atr = self._calculate_atr(symbol, "5m")
        if atr <= 0:
            atr = entry_price * 0.01
        strength_factor = 0.5 + max(0.0, min(1.0, signal_strength))
        tp_multiplier = 1.5 + (confidence / 100) * 2.0
        sl_multiplier = 1.0 + ((100 - confidence) / 100) * 1.5
        # No hard ceiling for TP - let strong signals run to 50% price move (up to 1250% ROI at 25x)
        # Previous 0.15 cap limited TP to 15% price move, user wants dynamic 100-1000 ROI no ceiling
        tp_distance = atr * tp_multiplier * strength_factor
        # Allow up to 50% price move for TP, no 15% cap
        tp_distance = min(tp_distance, entry_price * 0.5)
        sl_distance = max(atr * sl_multiplier / strength_factor, entry_price * 0.005)

        # A pure ATR stop can sit beyond the liquidation price at high leverage,
        # in which case the position is liquidated before the stop ever fires.
        safety = float(self.memory.get_setting("sl_liquidation_safety", 0.5))
        max_sl = self.liquidation_distance(entry_price, leverage) * safety
        if sl_distance > max_sl:
            logger.info(f"Clamping {symbol} SL distance {sl_distance:.8f} -> {max_sl:.8f} "
                        f"to stay inside liquidation at {leverage}x")
            sl_distance = max_sl

        if position_side == "LONG":
            tp = self.risk.round_price(symbol, entry_price + tp_distance)
            sl = self.risk.round_price(symbol, entry_price - sl_distance)
        else:
            tp = self.risk.round_price(symbol, entry_price - tp_distance)
            sl = self.risk.round_price(symbol, entry_price + sl_distance)
        return tp, sl

    def _calculate_atr(self, symbol: str, interval: str, period: int = 14) -> float:
        # ATR is recomputed every time a TP/SL is calculated (entry, fill
        # check, reversal, mid-management). Cache it per symbol/interval with a
        # 120s TTL so it is not re-fetched (kline + pandas) several times per
        # trade. ATR does not move meaningfully within two minutes, so the
        # cached value is still accurate for TP/SL placement.
        cache_key = f"atr_{symbol}_{interval}"
        cached = get_cache().get(cache_key)
        if cached is not None:
            return cached
        try:
            rows = self.xt.get_klines(symbol, interval, limit=period + 10)
        except XTError as e:
            logger.warning(f"ATR kline fetch failed for {symbol}: {e}")
            return 0.0
        if not rows:
            return 0.0
        df = pd.DataFrame(rows).rename(columns={"h": "high", "l": "low",
                                                "c": "close", "t": "timestamp"})
        for col in ["high", "low", "close", "timestamp"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if any(c not in df.columns for c in ["high", "low", "close"]):
            return 0.0
        df = df.dropna(subset=["high", "low", "close"])
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        df = df.reset_index(drop=True)
        if len(df) < period:
            return 0.0
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.tail(period).mean())
        get_cache().set(cache_key, atr, ttl_seconds=120.0)
        return atr

    # ---------- closing ----------

    def close_position(self, symbol: str, position_side: str, trade_id: int,
                       contracts: int = None) -> tuple:
        """Reads PnL BEFORE closing, since the position disappears afterwards."""
        pos = self.get_position_pnl(symbol, position_side)
        if not pos["exists"]:
            logger.info(f"{symbol} {position_side} already gone on exchange; "
                        f"marking trade {trade_id} closed")
            self.memory.close_trade(trade_id, pos["mark_price"], 0.0,
                                    notes="position not found on exchange")
            return False, None, "position not found on exchange"

        realized_pnl = pos["unrealized_pnl"]
        exit_price = pos["mark_price"]
        qty = int(contracts or pos["available_close_size"] or pos["position_size"])
        if qty <= 0:
            return False, None, "nothing available to close"

        # Cancel only THIS position's TP/SL, not all sides of the symbol.
        # cancel_all_tpsl destroys stops for hedged LONG+SHORT on the same symbol.
        pos_info = self.get_position_pnl(symbol, position_side)
        if pos_info.get("profit_id"):
            try:
                self.xt.cancel_tpsl(pos_info["profit_id"])
            except XTError as e:
                logger.warning(f"Could not cancel TP/SL {pos_info['profit_id']}: {e}")
        close_side = "SELL" if position_side == "LONG" else "BUY"
        try:
            data = self.xt.create_order(
                symbol=symbol, position_side=position_side, order_side=close_side,
                order_type="MARKET", orig_qty=qty, time_in_force="IOC",
            )
        except XTError as e:
            logger.error(f"Close order failed for {symbol} {position_side}: {e}")
            return False, None, str(e)

        self.risk.invalidate_balance_cache()
        self.memory.close_trade(trade_id, exit_price, realized_pnl)
        # Cooldown covers BOTH sides on this symbol, not just the side that
        # closed. Otherwise a signal flip opens the opposite side instantly.
        cooldown_min = int(self.memory.get_setting(
            "cooldown_minutes", Config.SIGNAL_COOLDOWN_MINUTES))
        self.memory.set_cooldown(symbol, "LONG", cooldown_min)
        self.memory.set_cooldown(symbol, "SHORT", cooldown_min)
        return True, data, None
                           
    def reconcile_open_trades(self) -> list:
        """Detects positions closed outside the bot (liquidation, ADL, manual, TP/SL hit)."""
        closed = []
        for trade in self.memory.get_open_trades():
            symbol = trade["symbol"]
            side = trade["position_side"]
            pos = self.get_position_pnl(symbol, side)
            if pos["exists"]:
                continue
            logger.warning(f"Trade {trade['id']} ({symbol} {side}) has no matching "
                           f"exchange position - closed externally")
            # For bot-opened trades the entry price is known, so we can estimate
            # PnL. For adopted trades (entry=0) we cannot.
            est_pnl = 0.0
            est_exit = 0.0
            entry = trade.get("entry_price") or 0
            amount = trade.get("amount") or 0
            if entry > 0 and amount > 0:
                try:
                    ticker = self.xt.get_agg_ticker(symbol)
                    est_exit = float(ticker.get("c") or 0)
                    if est_exit > 0:
                        cs = self.risk.get_contract_size(symbol)
                        price_diff = (est_exit - entry) if side == "LONG" else (entry - est_exit)
                        est_pnl = price_diff * amount * cs
                except XTError:
                    pass
            self.memory.close_trade(trade["id"], est_exit, est_pnl,
                                    notes="closed externally (liquidation/TPSL/manual)")
            cooldown_min = int(self.memory.get_setting(
                "cooldown_minutes", Config.SIGNAL_COOLDOWN_MINUTES))
            self.memory.set_cooldown(symbol, "LONG", cooldown_min)
            self.memory.set_cooldown(symbol, "SHORT", cooldown_min)
            closed.append({"trade_id": trade["id"], "symbol": symbol,
                           "position_side": side, "reason": "closed_externally"})
        return closed
