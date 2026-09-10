import json
import logging
import re
import time
from openai import OpenAI
from config import Config
from bot.memory import LongTermMemory

logger = logging.getLogger("xt_ai")

# ---------------------------------------------------------------------------
# Output token budget. Some OpenAI-compatible endpoints default to a very
# small completion cap (as low as 50 tokens) when max_tokens is not sent.
# That truncates function-call JSON (so tool calls like set_symbol silently
# fail) and leaves no room for reasoning/thinking. Always send an explicit
# budget. Reasoning-family models (OpenAI o1/o3/o4/gpt-5) only accept
# max_completion_tokens, so pick the parameter per model family.
# ---------------------------------------------------------------------------
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_COMPLETION_TOKENS = 16384
REASONING_COMPLETION_MODELS = ("gcli/grok-4.6", "gcli/grok-4.5-high", "oc/big-pickle", "gpt-5")

SYSTEM_PROMPT = """You are an AI Trading Assistant for XT.com Futures.

Your capabilities:
1. Manage trading settings via function calls (symbol, leverage, margin mode, timeframes, risk, etc.)
2. Analyze market conditions and provide trade recommendations
3. Monitor open positions and suggest management actions
4. Interpret signal scan results and provide clear explanations
5. Remember user preferences and past trading context

AVAILABLE FUNCTIONS:
- get_status() - Get current bot status including open positions, PnL, and settings
- get_pnl() - Get profit/loss summary
- get_balance() - Read the live USDT futures balance from XT
- get_contract_info(symbol) - Contract size, min order in contracts, min notional, max leverage. NOTE: min notional is ORDER value (contracts x price x contractSize), NOT wallet balance. With leverage a small balance opens a bigger notional. Never block a trade because balance < min notional.
- set_symbol(symbol) - Change trading pair (e.g. btc_usdt, eth_usdt)
- set_leverage(leverage) - Set leverage
- set_margin_mode(mode) - Set margin mode: CROSSED or ISOLATED
- set_timeframes(timeframes) - Set timeframes for scanning (e.g. "5m,15m,1h")
- set_margin_amount_pct(pct) - Set margin percentage of balance to use per trade (1-100)
- set_margin_risk_pct(pct) - Set risk percentage for position sizing (0.1-10)
- set_min_confidence(confidence) - Set minimum confidence threshold (50-100)
- set_cooldown_minutes(minutes) - Set cooldown minutes after closing position (1-30)
- set_position_mode(mode) - margin (by margin %) or risk (by risk %)
- set_max_loss_pct(pct) / set_max_profit_pct(pct) - Software safety-net ROI limits
- scan_signals() - Run signal scan now
- open_trade(direction, order_type, time_in_force) - Open a trade based on current signals
- close_trade(trade_id) - Close a specific trade
- close_all_trades() - Close all open positions
- mid_manage() - Run mid-position management

IMPORTANT FACTS ABOUT XT FUTURES:
- Order quantity is measured in CONTRACTS, not coin amount. One btc_usdt contract
  is 0.0001 BTC; one doge_usdt contract is 10 DOGE. Use get_contract_info when the
  user asks how much a position is worth.
- Leverage is capped per symbol by a notional-value bracket, so a requested
  leverage may be clamped down. Report the clamped value when that happens.
- Symbols are lowercase with an underscore: btc_usdt, eth_usdt, sol_usdt.
- Every position gets an exchange-side TP/SL order. If TP/SL creation fails the
  bot says so explicitly - treat that as urgent and tell the user.
- ROI shown is return on margin (leverage-amplified), not raw price movement.

IMPORTANT RULES:
- When the user asks to change settings, use the function calls directly.
- NEVER imitate a function call in plain text or XML (for example "<tool call
  set_symbol ...>" or "tool call set_symbol with symbol is ..."). You have real
  functions; call them through the tools interface only. If you cannot make a
  real call, explain what you would do and ask for confirmation.
- Only call open_trade / close_trade / close_all_trades when the user clearly
  asks for that action. Never call them to illustrate what you could do.
- Always explain what you're doing before calling functions.
- If signal strength is high, suggest wider TP and tighter SL.
- If signal strength is low, suggest tighter TP and wider SL.
- Always remind about risk management.
- Format numbers clearly with proper precision.
- Be concise but informative.
"""

FUNCTIONS = [
    {
        "name": "get_status",
        "description": "Get the current status of the trading bot including open positions, PnL, and settings",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_pnl",
        "description": "Get profit/loss summary for all trades",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "set_symbol",
        "description": "Change the trading pair symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading pair symbol, e.g. btc_usdt, eth_usdt"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "set_leverage",
        "description": "Set the trading leverage multiplier",
        "parameters": {
            "type": "object",
            "properties": {
                "leverage": {"type": "integer", "description": "Leverage value, 1-125"}
            },
            "required": ["leverage"]
        }
    },
    {
        "name": "set_margin_mode",
        "description": "Set margin mode for positions",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["CROSSED", "ISOLATED"], "description": "Margin mode"}
            },
            "required": ["mode"]
        }
    },
    {
        "name": "set_timeframes",
        "description": "Set timeframes for signal scanning",
        "parameters": {
            "type": "object",
            "properties": {
                "timeframes": {"type": "string", "description": "Comma-separated timeframes, e.g. 5m,15m,1h,4h"}
            },
            "required": ["timeframes"]
        }
    },
    {
        "name": "set_margin_amount_pct",
        "description": "Set what percentage of your balance to use as margin per trade",
        "parameters": {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "description": "Percentage 1-100"}
            },
            "required": ["pct"]
        }
    },
    {
        "name": "set_margin_risk_pct",
        "description": "Set what percentage of your balance to risk per trade for position sizing",
        "parameters": {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "description": "Risk percentage 0.1-10"}
            },
            "required": ["pct"]
        }
    },
    {
        "name": "set_min_confidence",
        "description": "Set the minimum confidence threshold for signals to execute",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence": {"type": "integer", "description": "Confidence threshold 50-100"}
            },
            "required": ["confidence"]
        }
    },
    {
        "name": "set_cooldown_minutes",
        "description": "Set cooldown period after closing a position before new signals are accepted",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Cooldown minutes 1-30"}
            },
            "required": ["minutes"]
        }
    },
    {
        "name": "set_position_mode",
        "description": "Set position sizing mode: margin (by margin %) or risk (by risk %)",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["margin", "risk"], "description": "Position sizing mode"}
            },
            "required": ["mode"]
        }
    },
    {
        "name": "set_max_loss_pct",
        "description": "Set the software safety-net max loss as ROI on margin. The exchange stop loss is primary; this is the backup.",
        "parameters": {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "description": "Loss ROI percentage, 1-100"}
            },
            "required": ["pct"]
        }
    },
    {
        "name": "set_max_profit_pct",
        "description": "Set the software safety-net max profit as ROI on margin",
        "parameters": {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "description": "Profit ROI percentage, 1-1000"}
            },
            "required": ["pct"]
        }
    },
    {
        "name": "get_balance",
        "description": "Read the live USDT futures balance from XT",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_contract_info",
        "description": "Read contract specs for a symbol: contract size, minimum order in contracts, minimum notional, max leverage, price tick",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Optional, e.g. btc_usdt. Defaults to the active symbol."}
            }
        }
    },
    {
        "name": "scan_signals",
        "description": "Run a signal scan now and return results",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "open_trade",
        "description": "Open a new trade based on current signal scan results",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["LONG", "SHORT"], "description": "Trade direction"},
                "order_type": {"type": "string", "enum": ["MARKET", "LIMIT"], "description": "Order type"},
                "time_in_force": {"type": "string", "enum": ["GTC", "IOC", "FOK"], "description": "Time in force"}
            },
            "required": ["direction"]
        }
    },
    {
        "name": "close_trade",
        "description": "Close a specific trade by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "trade_id": {"type": "integer", "description": "Trade ID to close"}
            },
            "required": ["trade_id"]
        }
    },
    {
        "name": "close_all_trades",
        "description": "Close all currently open positions",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "mid_manage",
        "description": "Run mid-position management on all open positions (breakeven, trailing stop)",
        "parameters": {"type": "object", "properties": {}}
    },
]


class AIChat:
    # Fallback when auto-detection is impossible (no /models support, etc.).
    FALLBACK_MODEL = "gpt-4o"
    # Chat-model families we prefer when auto-selecting, best-first.
    PREFERRED_FAMILIES = (
        "gpt-4o", "gpt-4.1", "gpt-4", "o1", "o3", "claude",
        "llama-3.1", "llama-3", "llama-4", "mistral", "mixtral",
        "gemma", "deepseek-chat", "deepseek", "qwen", "gemini", "yi-",
    )
    # Substrings that mark clearly non-chat models to skip. NOTE: do not add
    # "instruct" or "vision" here — instruction-tuned and multimodal models are
    # exactly the chat models we want (e.g. "llama-3.1-8b-instruct:free").
    NON_CHAT_MARKERS = (
        "embed", "whisper", "tts", "audio", "dall", "image",
        "moderation", "rerank", "re-rank", "realtime", "omni",
        "ft:", "fine-tune",
    )
    # Substrings that mark smaller/cheaper models — preferred after free ones
    # because they are the cheapest paid option when an account has to pay.
    CHEAP_MARKERS = (
        "flash", "mini", "lite", "nano", "small", "distil",
        "-8b", "-7b", "-4b", "-3b", "-2b", "-1b", "-0.5b",
    )

    # Some (weaker) models cannot emit native tool_calls and instead write the
    # call as prose/XML, e.g. '<tool call set_symbol with symbol is btc_usdt>'
    # or 'tool call set_symbol(symbol="btc_usdt")'. _parse_text_tool_call is a
    # best-effort safety net so those models still get executed.
    _TEXT_TOOL_PATTERNS = (
        re.compile(
            r"<\s*(?:tool|function|invoke)\s+call\s*>\s*"
            r"(\w+)\s+with\s+(.*?)</\s*(?:tool|function|invoke)\s+call\s*>",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\btool\s+call\s+(\w+)\s+with\s+(.+?)(?=\n|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bcall\s+(?:the\s+)?(\w+)\s+function\s+with\s+(.+?)(?=\n|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(call|invoke)\s+(\w+)\s*\(\s*(\{.*?\}|[^()\n]*)\)\s*$",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(r"\b(\w+)\s*\(\s*(\{.*?\})\)", re.DOTALL),
    )

    @classmethod
    def _parse_text_tool_call(cls, content: str):
        """Return (func_name, args) for a tool call written as text/XML.

        Returns (None, None) when no recognizable call is present. Only names
        from FUNCTIONS are accepted, so normal prose never triggers execution.
        """
        if not content:
            return None, None
        known = {f["name"] for f in FUNCTIONS}
        for pat in cls._TEXT_TOOL_PATTERNS:
            for m in pat.finditer(content):
                groups = [g for g in m.groups() if g]
                if not groups:
                    continue
                name = groups[0].strip()
                rest = groups[1].strip() if len(groups) > 1 else ""
                if name.lower() not in ("call", "invoke") and name not in known:
                    continue
                if name.lower() in ("call", "invoke") and len(groups) > 1:
                    name = groups[1].strip() if len(groups) > 1 else ""
                    rest = groups[2].strip() if len(groups) > 2 else ""
                    if name not in known:
                        continue
                if rest.startswith("{"):
                    try:
                        parsed = json.loads(rest)
                        if isinstance(parsed, dict):
                            return name, parsed
                    except ValueError:
                        pass
                # Prose form: 'symbol is btc_usdt, leverage is 5'
                pairs = re.findall(r"(\w+)\s+(?:is|=)\s+\"?([^\"\s,}>]+)\"?", rest)
                if pairs:
                    return name, {k: v.strip('"\'') for k, v in pairs}
                if rest and "(" not in name:
                    return name, rest
        # Direct name(...) form: get_status(), set_leverage(25),
        # set_symbol({"symbol": "xrp_usdt"}), close_trade(trade_id=12)
        for known_name in known:
            m = re.search(r"\b" + re.escape(known_name) + r"\s*\(\s*([^)]*)\)", content)
            if not m:
                continue
            inside = m.group(1).strip()
            if not inside:
                return known_name, {}
            if inside.startswith("{"):
                try:
                    parsed = json.loads(inside)
                    if isinstance(parsed, dict):
                        return known_name, parsed
                except ValueError:
                    pass
            kv = re.findall(r"(\w+)\s*[=:]\s*\"?([^\"\s,})]+)\"?", inside)
            if kv:
                return known_name, {k: v.strip('"\'') for k, v in kv}
            # Bare single value, e.g. set_leverage(25): bind it to the
            # function's first required parameter when there is exactly one.
            reqs = []
            for f in FUNCTIONS:
                if f.get("name") == known_name:
                    reqs = list(f.get("parameters", {}).get("required") or [])
                    if not reqs:
                        reqs = list(
                            (f.get("parameters", {}).get("properties") or {}).keys()
                        )[:1]
                    break
            if len(reqs) == 1:
                return known_name, {reqs[0]: inside.strip().strip('"\'')}
        return None, None

    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        self.client = OpenAI(
            api_key=Config.AI_API_KEY,
            base_url=Config.AI_BASE_URL,
        )
        self._candidates: list = []
        self._candidate_index: int = 0
        self.model = self._resolve_model()
        self.model_source = "explicit" if Config.AI_MODEL else "auto"
        self.trader = None

    def bind_trader(self, trader_instance):
        self.trader = trader_instance

    def _resolve_model(self) -> str:
        """Return the configured model, or auto-detect one from the provider.

        In auto mode the full ranked candidate list is stored so that, if the
        chosen model turns out to be unaffordable (402 / quota), we can fall
        through to the next usable model instead of failing outright. The top
        candidate is also *verified* with a real call at startup (see
        ``_probe_models``) so we never start on a model that only looks good by
        name but can't actually be used on this endpoint.
        """
        if Config.AI_MODEL:
            self._candidates = [Config.AI_MODEL]
            self._candidate_index = 0
            return Config.AI_MODEL
        ids = self._list_provider_models()
        if not ids:
            logger.warning(
                "AI model auto-detection found no models at %s/models; "
                "falling back to '%s'. Set AI_MODEL explicitly to override.",
                Config.AI_BASE_URL, self.FALLBACK_MODEL,
            )
            self._candidates = [self.FALLBACK_MODEL]
            self._candidate_index = 0
            return self.FALLBACK_MODEL
        candidates = self._rank_models(ids)
        # Name ranking can be wrong (a provider's "flash" model may not be
        # free, or the top pick may be under-funded). Probe each candidate with
        # a tiny call and start on the FIRST that actually answers — so we land
        # on a model the account can genuinely use on ANY OpenAI-compatible
        # endpoint, not just OpenRouter. The runtime fallback still covers
        # in-session balance errors after this.
        probed = self._probe_models(candidates)
        if probed:
            chosen = probed
        else:
            # Every candidate failed its probe (usually a bad key / base URL, or
            # an account without credits). Make that obvious at startup instead
            # of only surfacing a cryptic error at first chat.
            logger.warning(
                "AI auto-probe could not verify ANY of the %d candidates at "
                "%s/models — the API key or base URL is likely invalid, or the "
                "account has no credit. Falling back to '%s'. Check AI_API_KEY "
                "is a valid key for AI_BASE_URL (e.g. an OpenRouter key for "
                "OpenRouter's URL, an OpenAI key for api.openai.com). You can "
                "also set AI_MODEL explicitly.",
                len(candidates), Config.AI_BASE_URL, candidates[0],
            )
            chosen = candidates[0]
        self._candidates = candidates
        self._candidate_index = candidates.index(chosen)
        logger.info(
            "AI model auto-selected: '%s' (verified from %d candidates at %s/models)",
            chosen, len(candidates), Config.AI_BASE_URL,
        )
        return chosen

    def _list_provider_models(self) -> list:
        """Fetch model ids from the OpenAI-compatible /models endpoint."""
        try:
            page = self.client.models.list()
        except Exception as e:  # many compatible providers don't expose /models
            logger.warning(
                "Could not list models from %s/models: %s", Config.AI_BASE_URL, e
            )
            return []
        data = getattr(page, "data", None)
        if not data:
            return []
        return [getattr(m, "id", None) for m in data if getattr(m, "id", None)]

    @classmethod
    def _rank_models(cls, ids: list) -> list:
        """Order chat-capable model ids best-first for try-and-fallback.

        Priority: free-tier (``:free``) → cheap-indicator models → the rest,
        each bucket ranked by family preference. At request time we try them in
        order and fall through on balance/quota errors, so we always end up on
        a model the account can actually afford — even when the cheapest/free
        pick is momentarily unavailable or under-funded.
        """
        chat_models = [
            i for i in ids
            if not any(m in i.lower() for m in cls.NON_CHAT_MARKERS)
        ]
        pool = chat_models or ids
        pool_lower = [i.lower() for i in pool]

        def by_family(models: list) -> list:
            def key(mid: str):
                low = mid.lower()
                for idx, fam in enumerate(cls.PREFERRED_FAMILIES):
                    if fam in low:
                        return idx
                return len(cls.PREFERRED_FAMILIES)
            return sorted(models, key=key)

        free = [i for i, low in zip(pool, pool_lower) if low.endswith(":free")]
        cheap = [
            i for i, low in zip(pool, pool_lower)
            if not low.endswith(":free")
            and any(m in low for m in cls.CHEAP_MARKERS)
        ]
        rest = [
            i for i, low in zip(pool, pool_lower)
            if not low.endswith(":free")
            and not any(m in low for m in cls.CHEAP_MARKERS)
        ]
        return by_family(free) + by_family(cheap) + by_family(rest)

    def _probe_models(self, candidates: list) -> str:
        """Return the first candidate that actually answers a tiny test call.

        Name-based ranking (``_rank_models``) is a heuristic: a provider's
        "flash" model may not be free, and the top pick may be under-funded, so
        the best-looking id is not always the one the account can use. Probing
        each candidate with the production call shape (tools + tool_choice)
        guarantees we *start* on a model that genuinely emits native
        tool_calls — models that only imitate calls in prose/XML are rejected.
        Falls through on balance/quota errors and any other failure; returns ''
        only if no candidate answers.
        """
        for mid in candidates:
            try:
                # A plain text probe proves nothing about function calling.
                # Mirror the production call shape (tools + tool_choice) and
                # only accept models that actually emit a native tool_call —
                # models that merely imitate calls in prose/XML are skipped.
                response = self.client.chat.completions.create(
                    model=mid,
                    messages=[{
                        "role": "user",
                        "content": "What is the bot's current status? Use the get_status function to check.",
                    }],
                    tools=[{"type": "function", "function": f} for f in FUNCTIONS],
                    tool_choice="auto",
                    max_tokens=512,
                    timeout=15,
                )
                if getattr(response.choices[0].message, "tool_calls", None):
                    return mid
                logger.warning(
                    "AI auto-probe of '%s' answered but issued no native "
                    "tool_call; trying next candidate", mid,
                )
            except Exception as e:
                logger.warning(
                    "AI auto-probe of '%s' failed (%s); trying next candidate",
                    mid, str(e).splitlines()[0],
                )
                continue
        return ""

    @staticmethod
    def _is_balance_error(err_str: str) -> bool:
        """True when the error means this model is unaffordable for the account.

        These are the cases where we should switch to another candidate model
        rather than give up: OpenAI ``insufficient_quota``, OpenRouter-style
        ``402 ... balance is positive, but it is not enough``, and other generic
        balance/payment messages. Plain rate-limit 429s are NOT balance errors
        and must NOT trigger a model switch.
        """
        low = err_str.lower()
        return (
            "402" in err_str
            or "insufficient_quota" in low
            or "insufficient_balance" in low
            or "balance is positive" in low
            or "not enough" in low
            or "payment required" in low
            or "exceeded your current quota" in low
        )

    def _next_candidate(self) -> str:
        """Return the next auto-ranked candidate model, or '' if none remain."""
        if self.model_source != "auto":
            return ""
        self._candidate_index += 1
        if 0 <= self._candidate_index < len(self._candidates):
            return self._candidates[self._candidate_index]
        return ""

    def get_model_info(self) -> str:
        return f"{self.model} ({self.model_source})"

    def execute_function(self, func_name: str, args: dict) -> str:
        if not self.trader and func_name not in ["get_status", "get_pnl"]:
            return "Trader not initialized. Please start the bot first."

        handler_map = {
            "get_status": self._get_status,
            "get_pnl": self._get_pnl,
            "set_symbol": self._set_symbol,
            "set_leverage": self._set_leverage,
            "set_margin_mode": self._set_margin_mode,
            "set_timeframes": self._set_timeframes,
            "set_margin_amount_pct": self._set_margin_amount_pct,
            "set_margin_risk_pct": self._set_margin_risk_pct,
            "set_min_confidence": self._set_min_confidence,
            "set_cooldown_minutes": self._set_cooldown_minutes,
            "set_position_mode": self._set_position_mode,
            "set_max_loss_pct": self._set_max_loss_pct,
            "set_max_profit_pct": self._set_max_profit_pct,
            "get_balance": self._get_balance,
            "get_contract_info": self._get_contract_info,
            "scan_signals": self._scan_signals,
            "open_trade": self._open_trade,
            "close_trade": self._close_trade,
            "close_all_trades": self._close_all_trades,
            "mid_manage": self._mid_manage,
        }

        handler = handler_map.get(func_name)
        if handler:
            return handler(args)
        return f"Unknown function: {func_name}"

    def _get_status(self, args: dict) -> str:
        if not self.trader:
            summary = self.memory.get_trade_summary_for_ai()
            settings = self.memory.get_all_settings()
            return f"{summary}\n\nSettings: {settings}"
        return self.trader.get_status_report()

    def _get_pnl(self, args: dict) -> str:
        pnl = self.memory.get_total_pnl()
        stats = self.memory.get_trade_count()
        return (f"Total PnL: {pnl:.4f} USDT\n"
                f"Total Trades: {stats['total']} | Open: {stats['open']} | Closed: {stats['closed']}\n"
                f"Wins: {stats['wins']} | Losses: {stats['losses']} | "
                f"Flat/Unknown PnL: {stats['flat_or_unknown']} | "
                f"Winrate: {stats['winrate']}%")

    def _set_symbol(self, args: dict) -> str:
        symbol = args["symbol"].lower().strip()
        # Validate against the exchange before saving.
        if self.trader:
            try:
                detail = self.trader.xt.get_symbol_detail(symbol)
                if not detail or not detail.get("symbol"):
                    return f"Symbol '{symbol}' not found on XT futures. Check the format (e.g. btc_usdt)."
            except Exception as e:
                return f"Could not validate symbol '{symbol}': {e}"
        self.memory.set_setting("symbol", symbol)
        return f"Trading pair set to: {symbol}"

    def _set_leverage(self, args: dict) -> str:
        lev = int(args["leverage"])
        # Clamp to per-symbol max from exchange brackets, not hardcoded 125.
        if self.trader:
            symbol = self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL)
            max_lev = self.trader.risk.get_max_leverage(symbol)
            if max_lev and lev > max_lev:
                return f"Leverage {lev}x exceeds max {max_lev}x for {symbol}. Set to {max_lev}x."
            if max_lev is None or max_lev <= 0:
                max_lev = 125
            lev = max(1, min(lev, max_lev))
        else:
            lev = max(1, min(lev, 125))
        self.memory.set_setting("leverage", lev)
        return f"Leverage set to: {lev}x"

    def _set_margin_mode(self, args: dict) -> str:
        mode = args["mode"].upper()
        if mode not in ("CROSSED", "ISOLATED"):
            return "Invalid margin mode. Use CROSSED or ISOLATED."
        self.memory.set_setting("margin_mode", mode)
        return f"Margin mode set to: {mode}"

    def _set_timeframes(self, args: dict) -> str:
        tfs = args["timeframes"].strip()
        valid = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
        tf_list = [t.strip().lower() for t in tfs.split(",")]
        invalid = [t for t in tf_list if t not in valid]
        if invalid:
            return f"Invalid timeframes: {invalid}. Valid: {valid}"
        self.memory.set_setting("timeframes", ",".join(tf_list))
        return f"Timeframes set to: {', '.join(tf_list)}"

    def _set_margin_amount_pct(self, args: dict) -> str:
        pct = float(args["pct"])
        pct = max(1.0, min(pct, 100.0))
        self.memory.set_setting("margin_amount_pct", pct)
        return f"Margin amount percentage set to: {pct}%"

    def _set_margin_risk_pct(self, args: dict) -> str:
        pct = float(args["pct"])
        pct = max(0.1, min(pct, 10.0))
        self.memory.set_setting("margin_risk_pct", pct)
        return f"Risk percentage set to: {pct}%"

    def _set_min_confidence(self, args: dict) -> str:
        conf = int(args["confidence"])
        conf = max(50, min(conf, 100))
        self.memory.set_setting("min_confidence", conf)
        return f"Minimum confidence threshold set to: {conf}%"

    def _set_cooldown_minutes(self, args: dict) -> str:
        mins = int(args["minutes"])
        mins = max(1, min(mins, 30))
        self.memory.set_setting("cooldown_minutes", mins)
        return f"Cooldown period set to: {mins} minutes"

    def _set_position_mode(self, args: dict) -> str:
        mode = args["mode"].lower()
        if mode not in ("margin", "risk"):
            return "Invalid mode. Use margin or risk."
        self.memory.set_setting("position_mode", mode)
        return f"Position sizing mode set to: {mode}"

    def _scan_signals(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot scan signals."
        result = self.trader.scanner.scan_and_report()
        report = self.trader.scanner.format_signal_report(result)
        return report

    def _open_trade(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot open trade."
        direction = args.get("direction", "")
        if not direction:
            return "Direction (LONG/SHORT) is required"
        order_type = args.get("order_type") or "MARKET"
        # None lets the trader pick the timeInForce the exchange accepts for
        # this order type (MARKET needs IOC, LIMIT defaults to GTC).
        time_in_force = args.get("time_in_force") or None
        return self.trader.execute_trade(direction, order_type, time_in_force)

    def _get_balance(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot read balance."
        try:
            item = self.trader.risk._get_usdt_balance(force=True)
        except Exception as e:
            return f"Failed to read balance from XT: {e}"
        if not item:
            return "No USDT balance returned by XT."
        return (f"Wallet: {item.get('walletBalance')} USDT\n"
                f"Available: {item.get('availableBalance')} USDT\n"
                f"Order margin frozen: {item.get('openOrderMarginFrozen')} USDT\n"
                f"Isolated margin: {item.get('isolatedMargin')} USDT\n"
                f"Crossed margin: {item.get('crossedMargin')} USDT")

    def _get_contract_info(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running."
        symbol = (args.get("symbol")
                  or self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL))
        try:
            risk = self.trader.risk
            price = self.trader.scanner.get_current_price(symbol)
            cs = risk.get_contract_size(symbol)
            one = cs * price
            return (f"{symbol}\n"
                    f"Contract size: {cs} ({one:.4f} USDT per contract at {price})\n"
                    f"Min order: {risk.get_min_qty(symbol)} contracts\n"
                    f"Min notional: {risk.get_min_notional(symbol)} USDT\n"
                    f"Max leverage: {risk.get_max_leverage(symbol)}x\n"
                    f"Price precision: {risk.get_price_precision(symbol)} "
                    f"(tick {risk.get_price_step(symbol)})")
        except Exception as e:
            return f"Failed to read contract config for {symbol}: {e}"

    def _set_max_loss_pct(self, args: dict) -> str:
        pct = max(1.0, min(float(args["pct"]), 100.0))
        self.memory.set_setting("max_loss_pct", pct)
        return f"Software max loss (ROI on margin) set to: -{pct}%"

    def _set_max_profit_pct(self, args: dict) -> str:
        pct = max(1.0, min(float(args["pct"]), 1000.0))
        self.memory.set_setting("max_profit_pct", pct)
        return f"Software max profit (ROI on margin) set to: +{pct}%"

    def _close_trade(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot close trade."
        trade_id = int(args["trade_id"])
        return self.trader.close_specific_trade(trade_id)

    def _close_all_trades(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot close trades."
        return self.trader.close_all_positions()

    def _mid_manage(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot manage positions."
        return self.trader.run_mid_management()

    def chat(self, user_message: str, user_id: str = None) -> str:
        self.memory.add_chat_message("user", user_message)
        history = self.memory.get_chat_history(30)
        context = self.memory.get_trade_summary_for_ai()
        ai_context = self.memory.get_ai_context()
        context_msg = f"Current trading context:\n{context}\n\nAI memory context:\n{json.dumps(ai_context, indent=2)}"
        context_msg += f"\n\nValid symbols use format like btc_usdt, eth_usdt (lowercase with underscore)."
        context_msg += f"\nValid timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d"
        context_msg += f"\nMargin modes: CROSSED or ISOLATED"
        context_msg += f"\nOrder types for open_trade: MARKET or LIMIT"
        context_msg += f"\nTime in force for open_trade: GTC, IOC, FOK"
        # Merge the context into the FIRST user message rather than prepending a
        # standalone user turn. Some providers (notably Anthropic) reject two
        # consecutive user messages with no assistant turn between them.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_msg},
        ]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        # Drop the very last user turn from history — it duplicates the
        # user_message we are about to append, and some providers reject the
        # resulting back-to-back user messages.
        if messages and messages[-1].get("role") == "user":
            messages.pop()
        messages.append({"role": "user", "content": user_message})
        result = self._call_with_functions(messages)
        self.memory.add_chat_message("assistant", result)
        return result

    def _completion_limit_kwargs(self, model: str) -> dict:
        """Return the output-budget kwarg accepted by the given model family."""
        low = (model or "").lower().lstrip("/")
        if low.startswith(REASONING_COMPLETION_MODELS):
            return {"max_completion_tokens": DEFAULT_MAX_COMPLETION_TOKENS}
        return {"max_tokens": DEFAULT_MAX_TOKENS}

    def _execute_safely(self, func_name: str, raw_args: str) -> str:
        """Execute a tool call without letting malformed/truncated arguments
        (typical when an endpoint caps output tokens mid-JSON) crash the chat
        loop. Any failure becomes a tool result the model can react to."""
        args = {}
        if raw_args:
            if isinstance(raw_args, dict):
                args = raw_args
            elif raw_args.strip():
                try:
                    parsed = json.loads(raw_args)
                    if not isinstance(parsed, dict):
                        return (f"Function {func_name} received non-object arguments "
                                f"({raw_args[:120]!r}). Please answer in plain text "
                                "instead of calling functions.")
                except ValueError:
                    return (f"Function {func_name} arguments were not valid JSON "
                            f"({raw_args[:120]!r}) - likely truncated output. Please "
                            "answer in plain text instead of calling functions.")
                args = parsed
        try:
            return self.execute_function(func_name, args)
        except KeyError as e:
            return (f"Function {func_name} is missing a required argument ({e}). "
                    "Tell the user what you were trying to do in plain text.")
        except Exception as e:
            return f"Function {func_name} failed: {e}"

    def _call_with_functions(self, messages: list, max_rounds: int = 5) -> str:
        # In auto mode leave room to fall through all ranked candidates in
        # addition to the function-call rounds, so a cheap/free pick that is
        # momentarily under-funded doesn't exhaust the loop before we reach a
        # model the account can actually use. Explicit models keep max_rounds.
        rounds = max(max_rounds, len(self._candidates)) \
            if self.model_source == "auto" else max_rounds
        for _ in range(rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[{"type": "function", "function": f} for f in FUNCTIONS],
                    tool_choice="auto",
                    timeout=30,
                    **self._completion_limit_kwargs(self.model),
                )
            except Exception as e:
                err_str = str(e)
                # Only retry on actual tool-related errors from the API,
                # not on generic "tool" substrings which catch unrelated strings.
                if "tool_use_failed" in err_str or "function_call" in err_str.lower():
                    messages.append({
                        "role": "user",
                        "content": f"Your last function call failed: {err_str}. Please respond in plain text instead of calling functions. If you need to perform an action, describe it clearly and I'll handle it."
                    })
                    continue
                # A balance/quota error means this model is unaffordable for the
                # account, not a transient failure. In auto mode, switch to the
                # next ranked candidate and try again instead of giving up.
                if self._is_balance_error(err_str):
                    next_model = self._next_candidate()
                    if next_model:
                        logger.warning(
                            "AI model '%s' unavailable for this account "
                            "(%s); switching to '%s'",
                            self.model, err_str.splitlines()[0], next_model,
                        )
                        self.model = next_model
                        continue
                return f"AI API error: {err_str}"

            choice = response.choices[0]
            message = choice.message

            if message.tool_calls:
                function_result_messages = []
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    result = self._execute_safely(func_name, tool_call.function.arguments or "")
                    function_result_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": result,
                    })
                messages.append(message.model_dump() if hasattr(message, "model_dump") else {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in message.tool_calls
                    ]
                })
                messages.extend(function_result_messages)
            elif getattr(message, "function_call", None):
                func_call = message.function_call
                func_name = func_call.name
                result = self._execute_safely(func_name, func_call.arguments or "")
                # Legacy function-call format. The assistant turn must carry
                # function_call so the provider can match it to the function
                # result that follows — omitting it makes the sequence
                # "assistant → function" invalid on some providers.
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "function_call": {
                        "name": func_name,
                        "arguments": func_call.arguments or "",
                    },
                })
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": result,
                })
            else:
                content = message.content or ""
                if content:
                    # Safety net for models that cannot emit native tool_calls
                    # and instead write them as text/XML (e.g. "<tool call
                    # set_symbol with symbol is btc_usdt>"). Execute the first
                    # recognizable call so settings changes actually happen.
                    func_name, t_args = self._parse_text_tool_call(content)
                    if func_name:
                        result = self._execute_safely(func_name, t_args)
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"(Executed the {func_name} call written in your "
                                f"reply. Result: {result} Now confirm to the user "
                                "in one short line what was done.)"
                            ),
                        })
                        continue
                return content or ""
        return "Max function call rounds exceeded. Please try a more specific request."

    def remember(self, key: str, value: str):
        self.memory.set_ai_context(key, value)

    def recall(self, key: str = None):
        return self.memory.get_ai_context(key)
