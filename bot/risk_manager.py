import math
import time
import logging
from config import Config
from bot.memory import LongTermMemory
from bot.xt_client import XTClient, XTError

logger = logging.getLogger("xt_risk")


class RiskManager:
    BALANCE_CACHE_TTL = 3.0

    def __init__(self, xt_client: XTClient, memory: LongTermMemory):
        self.xt = xt_client
        self.memory = memory
        self._symbol_configs = {}
        self._balance_cache = None
        self._balance_cache_at = 0.0

    # ---------- symbol metadata ----------

    def get_symbol_config(self, symbol: str) -> dict:
        if symbol not in self._symbol_configs:
            self._symbol_configs[symbol] = self.xt.get_symbol_detail(symbol) or {}
        return self._symbol_configs[symbol]

    def get_contract_size(self, symbol: str) -> float:
        return float(self.get_symbol_config(symbol).get("contractSize") or 0)

    def get_min_qty(self, symbol: str) -> int:
        return int(float(self.get_symbol_config(symbol).get("minQty") or 1))

    def get_min_notional(self, symbol: str) -> float:
        return float(self.get_symbol_config(symbol).get("minNotional") or 0)

    def get_max_notional(self, symbol: str) -> float:
        return float(self.get_symbol_config(symbol).get("maxNotional") or 0)

    def get_max_order_qty(self, symbol: str, order_type: str) -> int:
        cfg = self.get_symbol_config(symbol)
        key = "maxMarketOrderQty" if order_type == "MARKET" else "maxLimitOrderQty"
        val = cfg.get(key)
        return int(float(val)) if val else 0

    def get_price_precision(self, symbol: str) -> int:
        return int(self.get_symbol_config(symbol).get("pricePrecision") or 2)

    def get_price_step(self, symbol: str) -> float:
        return float(self.get_symbol_config(symbol).get("minStepPrice") or 0)

    def round_price(self, symbol: str, price: float) -> float:
        step = self.get_price_step(symbol)
        precision = self.get_price_precision(symbol)
        if step > 0:
            price = math.floor(price / step) * step
        return round(price, precision)

    def supports_order_type(self, symbol: str, order_type: str) -> bool:
        raw = self.get_symbol_config(symbol).get("supportOrderType") or ""
        return order_type in [x.strip() for x in raw.split(",") if x.strip()]

    def supports_time_in_force(self, symbol: str, tif: str) -> bool:
        raw = self.get_symbol_config(symbol).get("supportTimeInForce") or ""
        return tif in [x.strip() for x in raw.split(",") if x.strip()]

    # ---------- balance (rate limited: asset endpoints are 3 req/s) ----------

    def _get_usdt_balance(self, force: bool = False) -> dict:
        now = time.time()
        if not force and self._balance_cache is not None \
                and now - self._balance_cache_at < self.BALANCE_CACHE_TTL:
            return self._balance_cache
        item = {}
        for row in self.xt.get_balances():
            if str(row.get("coin", "")).upper() == "USDT":
                item = row
                break
        self._balance_cache = item
        self._balance_cache_at = now
        return item

    def invalidate_balance_cache(self):
        self._balance_cache = None

    def get_total_balance(self) -> float:
        return float(self._get_usdt_balance().get("walletBalance") or 0)

    def get_available_balance(self) -> float:
        return float(self._get_usdt_balance().get("availableBalance") or 0)

    def get_tradable_balance(self) -> float:
        """Balance term from the XT sizing formula: walletBalance - openOrderMarginFrozen."""
        item = self._get_usdt_balance()
        wallet = float(item.get("walletBalance") or 0)
        frozen = float(item.get("openOrderMarginFrozen") or 0)
        return max(0.0, wallet - frozen)

    # ---------- position sizing (in contracts) ----------

    def contracts_from_coin_qty(self, symbol: str, coin_qty: float) -> int:
        cs = self.get_contract_size(symbol)
        if cs <= 0:
            return 0
        return int(coin_qty / cs)

    def contracts_to_notional(self, symbol: str, contracts: int, price: float) -> float:
        return contracts * self.get_contract_size(symbol) * price

    def size_by_margin_pct(self, symbol: str, price: float, leverage: int,
                           margin_pct: float = None) -> int:
        if margin_pct is None:
            margin_pct = float(self.memory.get_setting(
                "margin_amount_pct", Config.DEFAULT_MARGIN_AMOUNT_PCT))
        balance = self.get_tradable_balance()
        cs = self.get_contract_size(symbol)
        if balance <= 0 or price <= 0 or cs <= 0:
            return 0
        return int((balance * (margin_pct / 100.0) * leverage) / (price * cs))

    def size_by_risk_pct(self, symbol: str, price: float, stop_loss: float,
                         risk_pct: float = None) -> int:
        if risk_pct is None:
            risk_pct = float(self.memory.get_setting(
                "margin_risk_pct", Config.DEFAULT_RISK_PCT))
        balance = self.get_tradable_balance()
        cs = self.get_contract_size(symbol)
        price_diff = abs(price - stop_loss)
        if balance <= 0 or price_diff <= 0 or cs <= 0:
            return 0
        return int((balance * (risk_pct / 100.0)) / (price_diff * cs))

    def calculate_position_size(self, symbol: str, price: float, leverage: int,
                                stop_loss_price: float = None,
                                order_type: str = "MARKET") -> tuple:
        """Returns (contracts, mode, reason). contracts == 0 means the trade must be skipped."""
        use_risk = self.memory.get_setting("position_mode", "margin") == "risk"
        if use_risk and stop_loss_price:
            qty = self.size_by_risk_pct(symbol, price, stop_loss_price)
            mode = "risk_based"
        else:
            qty = self.size_by_margin_pct(symbol, price, leverage)
            mode = "margin_based"
        return self._validate_size(symbol, qty, price, mode, order_type)

    def _validate_size(self, symbol: str, qty: int, price: float,
                       mode: str, order_type: str) -> tuple:
        if qty <= 0:
            return 0, mode, "computed size is 0 contracts (balance too small for one contract)"

        min_qty = self.get_min_qty(symbol)
        if qty < min_qty:
            return 0, mode, f"size {qty} below exchange minimum {min_qty} contracts"

        max_qty = self.get_max_order_qty(symbol, order_type)
        if max_qty and qty > max_qty:
            logger.info(f"Capping {symbol} size {qty} -> {max_qty} ({order_type} limit)")
            reason = f"size capped from {qty} to {max_qty} ({order_type} max)"
            qty = max_qty
        else:
            reason = "ok"

        notional = self.contracts_to_notional(symbol, qty, price)
        min_notional = self.get_min_notional(symbol)
        if min_notional and notional < min_notional:
            return 0, mode, (f"notional {notional:.2f} USDT below minimum "
                             f"{min_notional} USDT (size {qty} contracts)")

        max_notional = self.get_max_notional(symbol)
        if max_notional and notional > max_notional:
            cs = self.get_contract_size(symbol)
            qty = int(max_notional / (price * cs))
            logger.info(f"Capping {symbol} to {qty} contracts (max notional {max_notional})")
            if qty < min_qty:
                return 0, mode, "max notional cap pushes size below minimum"

        return qty, mode, reason

    # ---------- leverage ----------

    def get_max_leverage(self, symbol: str, notional: float = None) -> int:
        """maxLeverage lives inside result.leverageBrackets[], selected by notional value."""
        try:
            brackets = self.xt.get_leverage_brackets(symbol)
        except XTError as e:
            logger.warning(f"Could not read leverage brackets for {symbol}: {e}")
            return 1
        if not brackets:
            return 1
        brackets = sorted(brackets, key=lambda b: float(b.get("maxNominalValue") or 0))
        if notional is not None:
            for b in brackets:
                if notional <= float(b.get("maxNominalValue") or 0):
                    return int(float(b.get("maxLeverage") or 1))
        return max(int(float(b.get("maxLeverage") or 1)) for b in brackets)

    def validate_leverage(self, symbol: str, leverage: int, notional: float = None) -> int:
        max_lev = self.get_max_leverage(symbol, notional)
        clamped = max(1, min(leverage, max_lev))
        if clamped != leverage:
            logger.info(f"Leverage {leverage}x clamped to {clamped}x for {symbol}")
        return clamped
