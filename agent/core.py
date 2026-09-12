"""
Agent - ReAct loop replacing the old deterministic bot.

Instead of: if signal > threshold -> trade (bot)
We do: LLM sees market + balance + positions + indicators -> reasons -> decides (agent)

Both autonomous loop and chat go through same ReAct engine.
"""

import json
import logging
import time
from typing import List, Dict

logger = logging.getLogger("agent_core")

import os
SOUL_PATH = os.path.join(os.path.dirname(__file__), "soul.md")
try:
    with open(SOUL_PATH, "r", encoding="utf-8") as f:
        SOUL = f.read()
except:
    SOUL = ""

SYSTEM_PROMPT = f"""{SOUL}

You are the AI Trader soul for exchange-traded futures. Act as an unemotional, evidence-driven trader and trading risk manager. Primary objective: maximize risk-adjusted returns while protecting capital. Secondary objective: preserve auditable decision trails. Use live market feeds, historical context, and order-execution status. Follow these rules:Data & AnalysisUse price, volume, order book, trade prints, VWAP, realized/IV (if available), liquidity, session context, and relevant macro/news flags.Combine multi-timeframe analysis (tick/1m/5m/1h/daily) and note market regime (trending, ranging, low/high volatility).Signal generationRequire at least two independent signals (e.g., price structure + momentum + volume profile or mean-reversion + liquidity exhaustion) before initiating a trade.Prefer risk-reward ≥ 1.5:1 for new trades unless part of an explicit strategy (scalps, spreads).Risk & Position SizingCalculate position size from pre-set risk budget: max_risk_per_trade_pct_of_equity (configurable).Enforce hard caps: max_position_size_per_instrument, max_total_exposure, max_leverage.Place stop-loss orders at submission time (automated OCO if possible) and define trailing logic.ExecutionPrefer limit orders unless market liquidity or immediacy requires market orders; account for expected slippage.Break large orders into child orders when appropriate (TWAP/VWAP) and respect exchange order rate limits.Risk controls & complianceDaily and intraday drawdown limits trigger immediate cessation of trading and alert human operator.Never attempt to access or manipulate other accounts or exchanges.Require human approval for any strategy changes, high-risk maneuvers, or trading around scheduled macro events.Reporting & DocumentationFor every trade, store: timestamp, instrument, contract, side, price, size, rationale, signals, risk metrics, and outcome.Provide concise human-readable trade summaries to the Telegram admin channel and a detailed audit log to storage.Behavior & SafetyBe conservative with confidence estimates. If confidence < threshold, propose but do not execute.Default to pass (no-trade) on ambiguous or illiquid conditions.If encountering an error, fail-safe by cancelling outstanding orders and alerting operator.trade_decision: {{action, instrument, contract, size, entry, stop, target, confidence, rationale, signals_used}}. If execution required, include confirmable checks and risk sizing calculation.

XT EXECUTION CONTEXT (keep compatible with tools):
- You trade XT.com USDT perpetual futures
- Orders are in CONTRACTS: 1 btc_usdt contract = 0.0001 BTC, 1 doge_usdt contract = 10 DOGE
- Symbols are lowercase with underscore: btc_usdt, eth_usdt, sol_usdt
- YOUR CAPABILITIES (tools): get_status, get_balance, get_positions_detail, get_market_data, scan_market, get_contract_info, estimate_trade, open_trade (LONG/SHORT), close_trade, close_all_trades, set_leverage, set_symbol, set_setting, manage_position, do_not_trade, remember, reset_cooldown
- SETTINGS LOCK: set_setting/set_leverage/set_symbol/reset_cooldown ONLY when the user explicitly asked for a settings change in THIS conversation (e.g. "symbol btc_usdt", "leverage ro bebar 50"). NEVER change settings on your own initiative - if you think a change is needed, propose it in words and ask first. Settings tools are hard-blocked otherwise.
- If user explicitly asks to change symbol/leverage (e.g. "symbol btc_usdt"), do set_symbol/set_leverage FIRST, do not scan instead. Never need 10 repeats.
- SIZING MATH (memorize, never violate):
  * order_notional = contracts x price x contractSize. ONLY this number is compared to min_notional. NEVER compare wallet balance to min_notional - they are different things.
  * max_notional_you_can_open = tradable_balance x margin_amount_pct% x leverage. Example: 2 USDT x 25% x 75 = ~38 USDT notional, which passes a 5 USDT minimum easily.
  * If estimate_trade says YES, the trade fits - open it. If it says NO, quote its reason. Never invent your own math, never demand deposits.
- Language: respond in same language as user (Finglish/Persian/English).
"""

# User message must explicitly mention settings before the agent may touch
# them in chat (user complaint: agent scrambled config on its own).
_SETTINGS_KEYWORDS = (
    "symbol", "leverage", "margin", "timeframe", "timeframes",
    "cooldown", "min_confidence", "tf_min", "min_agree", "max_positions",
    "reversal", "breakeven", "trailing", "scan_interval", "guard_interval",
    "report_interval", "risk_pct", "position_mode", "margin_mode",
    "set_setting", "set_leverage", "set_symbol", "reset_cooldown",
    "bezar", "بذار", "بزار", "avaz", "عوض", "taghir", "تغییر",
)


def _user_wants_settings_change(user_message: str) -> bool:
    msg = (user_message or "").lower()
    return any(k in msg for k in _SETTINGS_KEYWORDS)

class Agent:
    def __init__(self, trader, memory, brain):
        self.trader = trader
        self.memory = memory
        self.brain = brain
        from agent.tools import AgentTools, TOOLS, TOOLS_AUTO
        self.tools = AgentTools(trader, memory)
        self.TOOLS = TOOLS
        self.TOOLS_AUTO = TOOLS_AUTO
        from config import Config
        self.Config = Config

    def get_model_info(self) -> str:
        return self.brain.get_model_info()

    def _build_history_messages(self, user_message: str, limit: int = 20) -> List[Dict]:
        history = self.memory.get_chat_history(limit)
        # Remove last if duplicates user_message (avoid double user msg)
        # Also filter leaked system-reminders / operational mode messages that sometimes get stored
        msgs = []
        for m in history:
            content = m.get("content", "")
            # Filter internal system-reminder leaks that confuse the LLM
            if "system-reminder" in content.lower() or "operational mode has changed" in content.lower():
                continue
            msgs.append({"role": m["role"], "content": content})
        # If last is same as user_message, pop it to avoid dup
        if msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content") == user_message:
            msgs.pop()
        return msgs

    def chat(self, user_message: str) -> str:
        """Chat with agent - full ReAct loop with tool calling."""
        # Settings tools only fire when the USER explicitly asked for a
        # settings change. Otherwise the agent must never touch config.
        allow_settings = _user_wants_settings_change(user_message)
        self.memory.add_chat_message("user", user_message)
        context_summary = self.memory.get_trade_summary_for_ai()
        ai_ctx = self.memory.get_ai_context()

        system = SYSTEM_PROMPT
        system += f"\n\nCURRENT STATE:\n{context_summary}\n\nAI Memory:\n{json.dumps(ai_ctx, indent=2)}"

        messages: List[Dict] = [{"role": "system", "content": system}]
        # History without duplicating current message
        history = self._build_history_messages(user_message, 12)
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        max_steps = self.Config.AGENT_MAX_STEPS
        for step in range(max_steps):
            try:
                content, tool_calls, model_used = self.brain.chat(messages, tools=self.TOOLS, tool_choice="auto")
            except Exception as e:
                logger.error(f"Brain chat failed step {step}: {e}")
                err_msg = f"Agent brain error: {e}"
                self.memory.add_chat_message("assistant", err_msg)
                return err_msg

            if not tool_calls:
                # Final answer - filter leaked system-reminder from LLM output
                final = content or ""
                if not final.strip():
                    final = "Agent online - no tool needed. Balance ok, market neutral."
                if "system-reminder" in final.lower() or "operational mode has changed" in final.lower():
                    import re
                    final = re.sub(r"<system-reminder>.*?</system-reminder>", "", final, flags=re.DOTALL | re.IGNORECASE)
                    final = final.strip() or "Agent online."
                self.memory.add_chat_message("assistant", final)
                return final

            # Execute tools and append to messages
            # Need to add assistant message with tool_calls first
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}} for tc in tool_calls]
            })

            for tc in tool_calls:
                result = self.tools.execute(tc["name"], tc["arguments"],
                                            allow_settings=allow_settings)
                logger.info(f"Agent step {step} tool {tc['name']}({tc['arguments']}) -> {result[:200]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": result
                })

            # If last tool was do_not_trade or open/close, let loop continue to let LLM summarize

        # If max steps exceeded, get final summary
        try:
            content, _, _ = self.brain.chat(messages, tools=None)
            final = content or "Max steps reached without final answer."
            self.memory.add_chat_message("assistant", final)
            return final
        except Exception as e:
            return f"Max steps reached. Last error: {e}"

    def autonomous_tick(self) -> str:
        """
        One autonomous decision cycle.
        Called every AGENT_AUTONOMOUS_INTERVAL_SEC by the loop.
        The agent decides by itself whether to trade, manage, or wait.
        """
        prompt = (
            "This is your autonomous check. Analyze current market and positions, decide to trade or not.\n"
            "Steps: 1) get_status 2) get_balance 3) scan_market 4) get_market_data 5) decide open_trade or do_not_trade or manage_position.\n"
            "Be autonomous - you decide. Do not ask user for permission unless really risky.\n"
            "HARD RULE: you have NO settings tools here. NEVER change leverage, margin, symbol, timeframes or any setting - trade with current settings only."
        )
        # We don't add this to user chat history as it's system-initiated
        system = SYSTEM_PROMPT
        context_summary = self.memory.get_trade_summary_for_ai()
        system += f"\n\nCURRENT STATE:\n{context_summary}"
        messages: List[Dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

        max_steps = self.Config.AGENT_MAX_STEPS
        log = []
        for step in range(max_steps):
            try:
                content, tool_calls, _ = self.brain.chat(messages, tools=self.TOOLS_AUTO, tool_choice="auto")
            except Exception as e:
                err = f"Autonomous tick brain error: {e}"
                logger.error(err)
                return err

            if not tool_calls:
                # Agent produced final reasoning - filter leaked system-reminder
                final = content or "No action."
                if "system-reminder" in final.lower() or "operational mode has changed" in final.lower():
                    import re
                    final = re.sub(r"<system-reminder>.*?</system-reminder>", "", final, flags=re.DOTALL | re.IGNORECASE)
                    final = final.strip() or "No action - filtered"
                logger.info(f"Autonomous tick final: {final[:300]}")
                return final

            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}} for tc in tool_calls]
            })

            for tc in tool_calls:
                result = self.tools.execute(tc["name"], tc["arguments"], allow_settings=False)
                log.append(f"{tc['name']} -> {result[:120]}")
                logger.info(f"Autonomous step {step} {tc['name']} -> {result[:200]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": result
                })

                # If agent explicitly decided not to trade, we can finish early
                if tc["name"] == "do_not_trade":
                    return f"Agent decided NOT to trade: {tc['arguments'].get('reason','')} | {content}"

                if tc["name"] in ("open_trade", "close_trade", "close_all_trades"):
                    # Let one more loop to allow agent to summarize
                    pass

        # After max steps, ask for summary
        try:
            content, _, _ = self.brain.chat(messages, tools=None)
            return content or f"Autonomous tick steps: {'; '.join(log)}"
        except Exception as e:
            return f"Tick done, log: {'; '.join(log)} error: {e}"

    def remember(self, key: str, value: str):
        self.memory.set_ai_context(key, value)

    def recall(self, key: str = None):
        return self.memory.get_ai_context(key)
