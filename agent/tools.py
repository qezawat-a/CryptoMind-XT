"""
AgentTools - All XT + memory tools exposed to LLM as functions.
This replaces the old ai_chat FUNCTIONS list but with real trading reasoning.
Signal scanner is kept as a TOOL, not as a gate.
"""

import logging
from typing import Dict, List

logger = logging.getLogger("agent_tools")

# Tool definitions exposed to LLM (OpenAI function schema)
TOOLS = [
    {
        "name": "get_status",
        "description": "Get current bot status: open positions, PnL, settings, cooldowns. Always call first to understand state.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_balance",
        "description": "Read live USDT futures balance from XT (wallet, available, frozen).",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_market_data",
        "description": "Get current price, klines, and funding rate for a symbol. Use to analyze before trading.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. btc_usdt, eth_usdt (lowercase with underscore)"},
            "interval": {"type": "string", "description": "kline interval: 1m,5m,15m,1h,4h,1d", "enum": ["1m","3m","5m","15m","30m","1h","2h","4h","1d"]},
        }, "required": []}
    },
    {
        "name": "scan_market",
        "description": "Run technical indicator scan (RSI/EMA/MACD/BB etc) across timeframes. Returns direction, confidence, signal_strength. Agent should interpret this, not blindly follow.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "optional symbol, defaults to current"}
        }}
    },
    {
        "name": "get_contract_info",
        "description": "Get contract specs: contract size, min order, min notional, max leverage, price tick. IMPORTANT: min notional applies to the ORDER value (contracts x price x contractSize), NEVER compare it to wallet balance. With leverage, order notional = balance x margin% x leverage, so a 2 USDT balance at 75x opens ~38 USDT notional. Never refuse a trade because balance < min notional.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. btc_usdt"}
        }}
    },
    {
        "name": "estimate_trade",
        "description": "Calculate exact order size for a symbol WITHOUT placing any order. Returns contracts, notional in USDT, exchange minimum, and a YES/NO verdict. ALWAYS call this before deciding a trade is 'too small' - never guess from balance alone.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. saga_usdt (defaults to current symbol)"},
            "direction": {"type": "string", "enum": ["LONG", "SHORT"], "description": "Trade direction"},
            "leverage": {"type": "integer", "description": "Leverage 1-125 (defaults to setting)"},
        }, "required": []}
    },
    {
        "name": "open_trade",
        "description": "Open a futures trade. Only use when you have strong conviction after analyzing market. Size is calculated automatically from balance x margin% x leverage and checked against min notional inside. NEVER pre-reject a trade because wallet balance is below min notional - call estimate_trade for exact numbers, or just call open_trade and read the result.",
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["LONG", "SHORT"], "description": "Trade direction"},
            "symbol": {"type": "string", "description": "e.g. btc_usdt (defaults to current symbol)"},
            "leverage": {"type": "integer", "description": "Leverage 1-125 (defaults to setting)"},
        }, "required": ["direction"]}
    },
    {
        "name": "close_trade",
        "description": "Close a specific open trade by ID.",
        "parameters": {"type": "object", "properties": {
            "trade_id": {"type": "integer", "description": "Trade ID to close"}
        }, "required": ["trade_id"]}
    },
    {
        "name": "close_all_trades",
        "description": "Close all open positions immediately.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "set_leverage",
        "description": "Set leverage for next trades.",
        "parameters": {"type": "object", "properties": {
            "leverage": {"type": "integer", "description": "1-125"}
        }, "required": ["leverage"]}
    },
    {
        "name": "set_symbol",
        "description": "Change trading pair.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. btc_usdt, eth_usdt, sol_usdt"}
        }, "required": ["symbol"]}
    },
    {
        "name": "get_positions_detail",
        "description": "Get detailed PnL for all open positions from exchange (entry, mark, ROI, unrealized PnL).",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "manage_position",
        "description": "Run mid-position management (breakeven, trailing stop, PnL report).",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "do_not_trade",
        "description": "Explicitly decide to NOT trade now and explain why. Use this when market is unclear or risky.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string", "description": "Why not trading now"}
        }, "required": ["reason"]}
    },
    {
        "name": "remember",
        "description": "Store a note in long-term memory (market observation, lesson, preference).",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "Memory key"},
            "value": {"type": "string", "description": "Memory value"}
        }, "required": ["key", "value"]}
    },
    {
        "name": "set_setting",
        "description": "Change a trading setting (min_agreeing_strategies, report_interval_sec, mid_manage_interval_sec, timeframes, etc). Use to tune strategy.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "Setting key: min_agreeing_strategies, report_interval_sec, mid_manage_interval_sec, timeframes, min_confidence, tf_min_confidence, leverage, margin_amount_pct, etc"},
            "value": {"type": "string", "description": "New value (e.g. '2' for min_agreeing_strategies, '120' for report_interval_sec, '1m,5m,15m,4h' for timeframes)"}
        }, "required": ["key", "value"]}
    },
    {
        "name": "reset_cooldown",
        "description": "Clear a STUCK cooldown (remaining far beyond the normal window, e.g. hours from an old bug). Healthy cooldowns are refused automatically - never use this to force an entry after a stop.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. uai_usdt, defaults to current symbol"},
            "side": {"type": "string", "description": "LONG, SHORT, or ALL", "enum": ["LONG", "SHORT", "ALL"]}
        }}
    },
    {
        "name": "get_position_history",
        "description": "Get position history from exchange (closeProfit, fee, open/close price, leverage, working) via TradeListAll/PositionHistory endpoints.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. btc_usdt, defaults to current"},
            "limit": {"type": "integer", "description": "Number of history items (1-50)"}
        }}
    },
    {
        "name": "get_trade_history",
        "description": "Get detailed trade history with fee/takerMaker from exchange (TradeListAll).",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. btc_usdt, defaults to current"},
            "limit": {"type": "integer", "description": "Number of trades"}
        }}
    },
    {
        "name": "get_margin_call_info",
        "description": "Get margin call info (breakPrice=liq price, calMarkPrice) for all or specific symbol - shows liquidation price per doc Get Margin Call Information.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. btc_usdt, defaults to current, empty for all"}
        }}
    },
]


class AgentTools:
    # Settings tools are CHAT-ONLY. The autonomous loop must NEVER touch
    # config by itself (user complaint: agent scrambled leverage/margin alone).
    SETTINGS_TOOLS = {"set_setting", "set_leverage", "set_symbol", "reset_cooldown"}

    def __init__(self, trader, memory):
        self.trader = trader
        self.memory = memory
        from config import Config
        self.Config = Config

    def execute(self, name: str, args: Dict, allow_settings: bool = True) -> str:
        if not allow_settings and name in self.SETTINGS_TOOLS:
            return (f"Tool {name} is BLOCKED: the user did not ask for any "
                    f"settings change. NEVER change leverage/margin/symbol/"
                    f"timeframes/cooldowns on your own initiative. If you "
                    f"believe a change is needed, propose it in words and ask "
                    f"the user first. Continue with current settings.")
        handlers = {
            "get_status": self._get_status,
            "get_balance": self._get_balance,
            "get_market_data": self._get_market_data,
            "scan_market": self._scan_market,
            "get_contract_info": self._get_contract_info,
            "estimate_trade": self._estimate_trade,
            "open_trade": self._open_trade,
            "close_trade": self._close_trade,
            "close_all_trades": self._close_all_trades,
            "set_leverage": self._set_leverage,
            "set_symbol": self._set_symbol,
            "get_positions_detail": self._get_positions_detail,
            "manage_position": self._manage_position,
            "do_not_trade": self._do_not_trade,
            "remember": self._remember,
            "set_setting": self._set_setting,
            "reset_cooldown": self._reset_cooldown,
            "get_position_history": self._get_position_history,
            "get_trade_history": self._get_trade_history,
            "get_margin_call_info": self._get_margin_call_info,
        }
        h = handlers.get(name)
        if not h:
            return f"Unknown tool: {name}"
        try:
            return h(args or {})
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            return f"Tool {name} error: {e}"

    def _get_status(self, args: Dict) -> str:
        if not self.trader:
            return self.memory.get_trade_summary_for_ai()
        return self.trader.get_status_report()

    def _get_balance(self, args: Dict) -> str:
        try:
            item = self.trader.risk._get_usdt_balance(force=True)
        except Exception as e:
            return f"Failed to read balance: {e}"
        if not item:
            return "No USDT balance returned."
        # Show the comparable number right away so the LLM never compares
        # raw balance against min_notional (different things!):
        # max_notional = tradable x margin% x leverage.
        try:
            wallet = float(item.get("walletBalance") or 0)
            frozen = float(item.get("openOrderMarginFrozen") or 0)
            tradable = max(0.0, wallet - frozen)
            margin_pct = float(self.memory.get_setting(
                "margin_amount_pct", self.Config.DEFAULT_MARGIN_AMOUNT_PCT))
            leverage = int(self.memory.get_setting(
                "leverage", self.Config.DEFAULT_LEVERAGE))
            max_notional = tradable * (margin_pct / 100.0) * leverage
            buying_line = (f"Max order notional you can open: "
                           f"{tradable:.4f} x {margin_pct}% x {leverage}x = "
                           f"{max_notional:.2f} USDT "
                           f"(compare THIS to min_notional, never the raw balance)")
        except Exception:
            buying_line = ""
        out = (f"Wallet: {item.get('walletBalance')} USDT\n"
               f"Available: {item.get('availableBalance')} USDT\n"
               f"Frozen: {item.get('openOrderMarginFrozen')} USDT\n"
               f"Isolated: {item.get('isolatedMargin')} USDT\n"
               f"Crossed: {item.get('crossedMargin')} USDT")
        return out + (f"\n{buying_line}" if buying_line else "")

    def _get_market_data(self, args: Dict) -> str:
        symbol = args.get("symbol") or self.memory.get_setting("symbol", self.Config.DEFAULT_SYMBOL)
        interval = args.get("interval") or "1h"
        try:
            ticker = self.trader.xt.get_agg_ticker(symbol) or {}
            klines = self.trader.xt.get_klines(symbol, interval, limit=20) or []
            funding = self.trader.xt.get_funding_rate(symbol) or {}
            price = ticker.get("lastPrice") or ticker.get("price") or "?"
            out = f"{symbol} price: {price}\n"
            out += f"Ticker: {ticker}\n"
            out += f"Funding: {funding}\n"
            out += f"Last {len(klines)} klines ({interval}):\n"
            for k in klines[-5:]:
                out += f"  {k}\n"
            return out
        except Exception as e:
            return f"Market data error for {symbol}: {e}"

    def _scan_market(self, args: Dict) -> str:
        symbol = args.get("symbol") or self.memory.get_setting("symbol", self.Config.DEFAULT_SYMBOL)
        try:
            result = self.trader.scanner.scan_and_report(symbol)
            report = self.trader.scanner.format_signal_report(result)
            return report
        except Exception as e:
            return f"Scan failed for {symbol}: {e}"

    def _get_contract_info(self, args: Dict) -> str:
        symbol = args.get("symbol") or self.memory.get_setting("symbol", self.Config.DEFAULT_SYMBOL)
        try:
            cs = self.trader.risk.get_contract_size(symbol)
            price = self.trader.scanner.get_current_price(symbol)
            one = cs * price if price else 0
            return (f"{symbol}\n"
                    f"Contract size: {cs} ({one:.4f} USDT/contract @ {price})\n"
                    f"Min order: {self.trader.risk.get_min_qty(symbol)} contracts\n"
                    f"Min notional: {self.trader.risk.get_min_notional(symbol)} USDT\n"
                    f"Max leverage: {self.trader.risk.get_max_leverage(symbol)}x\n"
                    f"Price tick: {self.trader.risk.get_price_step(symbol)}")
        except Exception as e:
            return f"Contract info error for {symbol}: {e}"

    def _estimate_trade(self, args: Dict) -> str:
        """Dry-run sizing: exact numbers so the LLM never guesses wrong."""
        symbol = (args.get("symbol") or self.memory.get_setting(
            "symbol", self.Config.DEFAULT_SYMBOL)).lower()
        direction = (args.get("direction") or "LONG").upper()
        if direction not in ("LONG", "SHORT"):
            return "Invalid direction, use LONG or SHORT"
        try:
            leverage = int(args.get("leverage") or self.memory.get_setting(
                "leverage", self.Config.DEFAULT_LEVERAGE))
        except (TypeError, ValueError):
            leverage = self.Config.DEFAULT_LEVERAGE
        try:
            price = self.trader.scanner.get_current_price(symbol)
            if price <= 0:
                return f"No live price for {symbol}."
            _, prov_sl = self.trader.position_mgr.calculate_dynamic_tpsl(
                symbol, direction, price, 0.5, 80, leverage)
            qty, mode, reason = self.trader.risk.calculate_position_size(
                symbol, price, leverage, prov_sl, "MARKET")
            cs = self.trader.risk.get_contract_size(symbol)
            notional = self.trader.risk.contracts_to_notional(symbol, qty, price)
            min_n = self.trader.risk.get_min_notional(symbol)
            bal = self.trader.risk.get_tradable_balance()
            try:
                margin_pct = float(self.memory.get_setting(
                    "margin_amount_pct", self.Config.DEFAULT_MARGIN_AMOUNT_PCT))
            except (TypeError, ValueError):
                margin_pct = float(self.Config.DEFAULT_MARGIN_AMOUNT_PCT)
            verdict = ("YES - call open_trade" if qty > 0
                       else f"NO - {reason}")
            return (f"{symbol} {direction} {leverage}x @ {price}\n"
                    f"MATH (step by step):\n"
                    f"  1) margin money = {bal:.4f} x {margin_pct}% = "
                    f"{bal * (margin_pct / 100.0):.4f} USDT\n"
                    f"  2) order value = margin money x {leverage}x leverage\n"
                    f"  3) contracts = order value / (price x contractSize) = "
                    f"{qty} contracts\n"
                    f"  4) notional = {qty} x {price} x {cs} = "
                    f"{notional:.2f} USDT vs exchange min {min_n} USDT\n"
                    f"RULE: compare ONLY step-4 notional to min_notional. "
                    f"Raw balance ({bal:.4f}) is NEVER compared to min_notional.\n"
                    f"Verdict: {verdict}")
        except Exception as e:
            return f"Estimate failed for {symbol}: {e}"

    def _open_trade(self, args: Dict) -> str:
        if self.Config.AGENT_DRY_RUN == "true":
            return f"[DRY_RUN] Would open {args.get('direction')} {args.get('symbol','?')} leverage={args.get('leverage','?')} - dry run enabled, no order placed."
        direction = args.get("direction")
        if direction not in ("LONG", "SHORT"):
            return "Invalid direction, use LONG or SHORT"
        symbol = (args.get("symbol") or self.memory.get_setting("symbol", self.Config.DEFAULT_SYMBOL)).lower()
        leverage = args.get("leverage")
        # One-trade override ONLY: apply requested symbol/leverage for this
        # trade, then RESTORE the user's settings. Persisting them here was
        # the hole that scrambled config behind the settings lock.
        old_sym = self.memory.get_setting("symbol", self.Config.DEFAULT_SYMBOL)
        try:
            old_lev = int(self.memory.get_setting("leverage", self.Config.DEFAULT_LEVERAGE))
        except (TypeError, ValueError):
            old_lev = self.Config.DEFAULT_LEVERAGE
        overridden = []
        if args.get("symbol") and symbol != old_sym:
            self.memory.set_setting("symbol", symbol)
            overridden.append(f"symbol {old_sym}->{symbol} (this trade only)")
        if leverage and int(leverage) != old_lev:
            self.memory.set_setting("leverage", int(leverage))
            overridden.append(f"leverage {old_lev}->{leverage} (this trade only)")
        try:
            result = self.trader.execute_trade(direction, order_type="MARKET")
            # If blocked by any gate, give precise reason and auto-fix if possible
            low = result.lower()
            if "cooldown" in low:
                try:
                    rem = self.memory.get_cooldown_remaining(symbol, direction)
                    if rem > 600:
                        self._reset_cooldown({"symbol": symbol, "side": direction})
                        result += f"\n[Auto-reset cooldown {rem:.0f}s -> 0, retry next tick]"
                    elif "4868" in result or "3327" in result or "4000" in result:
                        self._reset_cooldown({"symbol": symbol, "side": direction})
                        result += f"\n[Auto-reset buggy cooldown, retry]"
                    # Also try once more immediately after reset if high confidence
                    if rem > 0 and "90%" in str(self.trader.scanner.scan_and_report(symbol).get("confidence",0)):
                        self._reset_cooldown({"symbol": symbol, "side": direction})
                        retry = self.trader.execute_trade(direction, order_type="MARKET")
                        result += f"\n[Retry after reset]: {retry}"
                except Exception:
                    pass
            low = result.lower()
            if "below minimum" in low or "computed size is 0" in low or "below exchange minimum" in low:
                result += (f"\n[HINT] The sized ORDER (contracts x price x "
                           f"contractSize) is below exchange minimum - this is "
                           f"about order math, NOT raw balance. Call "
                           f"estimate_trade to see exact numbers. Do NOT change "
                           f"settings or demand deposits on your own.")
            if "max positions" in low or "already have an open" in low:
                result += f"\n[HINT] Gate blocked: {result}. Check /status for open positions."
        finally:
            self.memory.set_setting("symbol", old_sym)
            self.memory.set_setting("leverage", old_lev)
        if overridden:
            result += f"\n[NOTE: {'; '.join(overridden)} - your saved settings were restored]"
        return result

    def _close_trade(self, args: Dict) -> str:
        tid = int(args["trade_id"])
        return self.trader.close_specific_trade(tid)

    def _close_all_trades(self, args: Dict) -> str:
        return self.trader.close_all_positions()

    def _set_leverage(self, args: Dict) -> str:
        lev = int(args["leverage"])
        lev = max(1, min(lev, 125))
        self.memory.set_setting("leverage", lev)
        return f"Leverage set to {lev}x"

    def _set_symbol(self, args: Dict) -> str:
        sym = args["symbol"].lower().strip()
        try:
            detail = self.trader.xt.get_symbol_detail(sym)
            if not detail or not detail.get("symbol"):
                return f"Symbol {sym} not found on XT."
        except Exception as e:
            return f"Could not validate {sym}: {e}"
        self.memory.set_setting("symbol", sym)
        return f"Symbol set to {sym}"

    def _get_positions_detail(self, args: Dict) -> str:
        open_trades = self.memory.get_open_trades()
        if not open_trades:
            return "No open positions."
        out = ""
        batch = self.trader.position_mgr.get_positions_batch_optimized()
        for t in open_trades:
            pos = self.trader.position_mgr.get_position_pnl_optimized(t["symbol"], t["position_side"], positions_batch=batch)
            if not pos["exists"]:
                out += f"ID:{t['id']} {t['symbol']} {t['position_side']} NOT FOUND ON EXCHANGE\n"
            else:
                out += (f"ID:{t['id']} {t['symbol']} {t['position_side']} Entry:{pos['entry_price']} Mark:{pos['mark_price']} "
                        f"Size:{int(pos['position_size'])}c PnL:{pos['unrealized_pnl']:.4f} ROI:{pos['roi']:.2f}% Lev:{pos['leverage']}x\n")
        return out

    def _manage_position(self, args: Dict) -> str:
        return self.trader.run_mid_management()

    def _do_not_trade(self, args: Dict) -> str:
        reason = args.get("reason", "No reason")
        self.memory.add_chat_message("assistant", f"Decision: DO NOT TRADE - {reason}")
        return f"Decision recorded: DO NOT TRADE - {reason}"

    def _reset_cooldown(self, args: Dict) -> str:
        symbol = (args.get("symbol") or self.memory.get_setting("symbol", self.Config.DEFAULT_SYMBOL)).lower()
        side = (args.get("side") or "ALL").upper()
        # GUARD: a healthy cooldown must NEVER be deleted just to force an
        # entry (this caused 3 rapid re-entries after stops). Only a STUCK
        # cooldown (remaining beyond the normal window, e.g. the old 4868s
        # bug) may be cleared.
        try:
            normal_min = int(self.memory.get_setting(
                "cooldown_minutes", self.Config.SIGNAL_COOLDOWN_MINUTES))
        except (TypeError, ValueError):
            normal_min = 3
        normal_min = max(1, min(normal_min, 10))
        sides = ["LONG", "SHORT"] if side == "ALL" else [side]
        rems = {s: self.memory.get_cooldown_remaining(symbol, s) for s in sides}
        peak = max(rems.values()) if rems else 0
        if 0 < peak <= normal_min * 60:
            return (f"Cooldown is HEALTHY ({peak:.0f}s left of {normal_min * 60:.0f}s) - "
                    f"refusing reset. Do NOT force entries; wait or use do_not_trade.")
        if peak <= 0:
            return f"No active cooldown for {symbol} {side} - nothing to reset."
        try:
            from sqlalchemy import text as sql_text
            with self.memory._engine.begin() as conn:
                if side == "ALL":
                    conn.execute(sql_text("DELETE FROM cooldowns WHERE symbol=:s"), {"s": symbol})
                    # also delete generic
                    conn.execute(sql_text("DELETE FROM cooldowns WHERE symbol=:s"), {"s": symbol})
                else:
                    conn.execute(sql_text("DELETE FROM cooldowns WHERE symbol=:s AND side=:side"), {"s": symbol, "side": side})
            return f"Cooldown reset for {symbol} {side} - you can trade now"
        except Exception as e:
            # fallback: set to 0
            try:
                self.memory.set_cooldown(symbol, "LONG", 0)
                self.memory.set_cooldown(symbol, "SHORT", 0)
                return f"Cooldown reset via fallback for {symbol}: {e}"
            except Exception as e2:
                return f"Reset failed: {e2}"

    def _get_position_history(self, args: Dict) -> str:
        symbol = args.get("symbol")
        limit = int(args.get("limit") or 5)
        try:
            data = self.trader.xt.get_position_history(symbol=symbol, page=1, size=limit)
            items = data.get("items") or data.get("list") or []
            if not items:
                return "No position history found."
            out = f"Position History ({len(items)}):\n"
            for it in items[:limit]:
                out += f"  {it.get('symbol')} {it.get('positionSide')} open:{it.get('closeOpenPrice')} close:{it.get('closePrice')} profit:{it.get('closeProfit')} fee:{it.get('totalFee')} lev:{it.get('startLeverage')}->{it.get('endLeverage')} time:{it.get('closeTime')}\n"
            return out
        except Exception as e:
            return f"Position history error: {e}"

    def _get_trade_history(self, args: Dict) -> str:
        symbol = args.get("symbol")
        limit = int(args.get("limit") or 5)
        try:
            data = self.trader.xt.get_all_trades(symbol=symbol)
            # get_all_trades returns list directly
            items = data if isinstance(data, list) else data.get("items") or []
            if not items:
                return "No trade history found."
            out = f"Trade History ({len(items)}):\n"
            for it in items[:limit]:
                out += f"  {it.get('symbol')} {it.get('positionSide')} {it.get('orderSide')} qty:{it.get('quantity')} price:{it.get('price')} fee:{it.get('fee')} ({it.get('takerMaker')}) time:{it.get('timestamp')}\n"
            return out
        except Exception as e:
            return f"Trade history error: {e}"

    def _get_margin_call_info(self, args: Dict) -> str:
        symbol = args.get("symbol")
        try:
            data = self.trader.xt.get_margin_call_info(symbol=symbol)
            if not data:
                return "No margin call info (no positions or no liq price)."
            out = f"Margin Call Info (breakPrice=liq):\n"
            for it in (data if isinstance(data, list) else [data]):
                out += (f"  {it.get('symbol')} {it.get('positionSide')} size:{it.get('positionSize')} entry:{it.get('entryPrice')} "
                        f"mark:{it.get('calMarkPrice')} breakPrice(liq):{it.get('breakPrice')} lev:{it.get('leverage')} type:{it.get('positionType')} isolatedMargin:{it.get('isolatedMargin')}\n")
            return out
        except Exception as e:
            return f"Margin call info error: {e}"

    def _remember(self, args: Dict) -> str:
        self.memory.set_ai_context(args["key"], args["value"])
        return f"Remembered {args['key']}: {args['value']}"

    def _set_setting(self, args: Dict) -> str:
        key = args.get("key", "").strip()
        value = args.get("value", "").strip()
        if not key:
            return "Missing key"
        # Validate and set known settings
        try:
            if key in ("min_agreeing_strategies", "signal_confirm_scans", "tf_min_confidence", "min_confidence", "leverage", "max_positions", "cooldown_minutes", "scan_interval_sec", "guard_interval_sec", "report_interval_sec", "mid_manage_interval_sec", "reversal_confidence"):
                self.memory.set_setting(key, int(float(value)))
                return f"Setting {key} set to {value} (int)"
            elif key in ("margin_amount_pct", "margin_risk_pct", "max_loss_pct", "max_profit_pct", "breakeven_threshold_pct", "trailing_stop_pct", "trailing_trigger_roi_pct", "trailing_distance_pct", "sl_liquidation_safety"):
                self.memory.set_setting(key, float(value))
                return f"Setting {key} set to {value} (float)"
            elif key in ("timeframes", "symbol", "margin_mode", "position_mode", "on_tpsl_failure"):
                self.memory.set_setting(key, value)
                return f"Setting {key} set to {value}"
            elif key in ("reversal_enabled",):
                self.memory.set_setting(key, value.lower() in ("1", "true", "yes", "on"))
                return f"Setting {key} set to {value}"
            else:
                self.memory.set_setting(key, value)
                return f"Setting {key} set to {value} (generic)"
        except Exception as e:
            return f"Failed to set {key}: {e}"


# Tool list for the AUTONOMOUS loop: everything except settings tools.
TOOLS_AUTO = [t for t in TOOLS if t["name"] not in AgentTools.SETTINGS_TOOLS]
