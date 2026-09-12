import time
import logging
import pandas as pd
from bot.xt_client import XTClient, XTError
from bot.strategies import StrategyEngine
from bot.memory import LongTermMemory
from config import Config

logger = logging.getLogger("xt_scanner")

# XT kline field names. Per XT docs: "a" is Volume, "v" is Turnover.
KLINE_FIELDS = {
    "t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close",
    "a": "volume", "v": "turnover", "s": "symbol",
}

VALID_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w"]

TF_WEIGHTS = {"1m": 0.5, "3m": 0.8, "5m": 1.0, "15m": 1.5, "30m": 2.0,
              "1h": 2.5, "2h": 2.8, "4h": 3.0, "1d": 4.0, "1w": 5.0}


class SignalScanner:
    def __init__(self, xt_client: XTClient, memory: LongTermMemory):
        self.xt = xt_client
        self.memory = memory
        self.engine = StrategyEngine()

    def fetch_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        try:
            rows = self.xt.get_klines(symbol, interval, limit=min(limit, 1500))
        except XTError as e:
            logger.warning(f"Kline fetch failed for {symbol} {interval}: {e}")
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).rename(columns=KLINE_FIELDS)
        for col in ["open", "high", "low", "close", "volume", "turnover", "timestamp"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        required = ["open", "high", "low", "close"]
        if any(c not in df.columns for c in required):
            logger.warning(f"Kline response missing OHLC columns: {list(df.columns)}")
            return pd.DataFrame()
        df = df.dropna(subset=required)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        # XT returns newest-first; strategies assume oldest-first.
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        df = df.reset_index(drop=True)
        # Drop the still-forming (live) last candle: indicators must only see
        # CLOSED candles. A forming candle repaints every scan - a cross on it
        # can appear LONG and vanish at close, opening trades against the real
        # trend (user sees SHORT on closed candles, bot acted on forming one).
        df = self._drop_forming_candle(df, interval)
        return df

    @staticmethod
    def _drop_forming_candle(df, interval: str):
        """Remove last row if its candle window hasn't closed yet."""
        try:
            import time as _time
            unit = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
            seconds = int(interval[:-1]) * unit.get(interval[-1:], 60)
            last_ts = float(df["timestamp"].iloc[-1])
            # XT sends ms; tolerate seconds just in case.
            if last_ts < 1e12:
                last_ts *= 1000
            if _time.time() * 1000 < last_ts + seconds * 1000:
                return df.iloc[:-1].reset_index(drop=True)
        except Exception:
            pass
        return df

    def get_current_price(self, symbol: str) -> float:
        try:
            data = self.xt.get_agg_ticker(symbol)
            price = float(data.get("c") or 0)
            if price > 0:
                return price
        except XTError as e:
            logger.warning(f"agg-ticker failed for {symbol}: {e}")
        try:
            data = self.xt.get_mark_price(symbol)
            return float(data.get("p") or 0)
        except XTError as e:
            logger.warning(f"mark-price failed for {symbol}: {e}")
        return 0.0

    def get_mark_price(self, symbol: str) -> float:
        try:
            return float(self.xt.get_mark_price(symbol).get("p") or 0)
        except XTError:
            return 0.0

    def scan_single_timeframe(self, symbol: str, interval: str,
                              min_confidence: int) -> dict:
        df = self.fetch_klines(symbol, interval)
        if df.empty or len(df) < 40:
            return {"direction": "NEUTRAL", "confidence": 0, "all_signals": [],
                    "strategies_used": [], "error": "insufficient_data"}
        min_agree = int(self.memory.get_setting("min_agreeing_strategies",
                                                Config.MIN_AGREEING_STRATEGIES))
        return self.engine.get_consensus(df, min_confidence, min_agree)

    def _resolve_intervals(self, intervals=None) -> list:
        if intervals is None:
            intervals = self.memory.get_setting("timeframes",
                                                ",".join(Config.DEFAULT_TIMEFRAMES))
        if isinstance(intervals, str):
            intervals = intervals.split(",")
        out = []
        for tf in intervals:
            tf = tf.strip().lower()
            if tf in VALID_INTERVALS:
                out.append(tf)
            elif tf:
                logger.warning(f"Dropping unsupported timeframe: {tf}")
        return out or list(Config.DEFAULT_TIMEFRAMES)

    def scan_multi_timeframe(self, symbol: str, intervals: list = None,
                             min_confidence: int = None) -> dict:
        if min_confidence is None:
            min_confidence = int(self.memory.get_setting("min_confidence",
                                                         Config.MIN_CONFIDENCE))
        intervals = self._resolve_intervals(intervals)
        tf_min_conf = int(self.memory.get_setting("tf_min_confidence",
                                                  Config.TF_MIN_CONFIDENCE))

        all_results = {}
        long_weight = 0.0
        short_weight = 0.0
        voted_weight = 0.0
        strategies_used = set()

        for tf in intervals:
            result = self.scan_single_timeframe(symbol, tf, tf_min_conf)
            all_results[tf] = result
            direction = result["direction"]
            if direction == "NEUTRAL":
                continue
            # Only signal-producing timeframes contribute weight — NEUTRAL ones
            # must not dilute the denominator, otherwise the agreement % wrongly
            # drops when the market is choppy and most TFs return NEUTRAL.
            weight = TF_WEIGHTS.get(tf, 1.0)
            voted_weight += weight
            contribution = weight * (result["confidence"] / 100.0)
            if direction == "LONG":
                long_weight += contribution
            else:
                short_weight += contribution
            strategies_used.update(result.get("strategies_used", []))

        overall = "NEUTRAL"
        strength = 0.0
        confidence = 0
        # Confidence reflects DIRECTIONAL ALIGNMENT across timeframes (how much
        # of the configured weight actually votes the winning way), scaled by
        # the winning side's average confidence. A unanimous 4/4 signal reaches
        # high confidence so the bot trades on a clearly-aligned trend; a split
        # or choppy market (some TFs NEUTRAL/opposed) is pulled down in step.
        # This replaces the old formula (confidence = strength * agreement),
        # which multiplied two sub-unity factors and capped even perfect 4/4
        # alignment near 77% -- so the bot almost never cleared the 80% bar.
        if voted_weight > 0 and long_weight != short_weight:
            if long_weight > short_weight:
                overall = "LONG"
                winner, loser = long_weight, short_weight
            else:
                overall = "SHORT"
                winner, loser = short_weight, long_weight
            strength = (winner - loser) / voted_weight
            total_w = sum(TF_WEIGHTS.get(t, 1.0) for t in intervals)
            aligned_w = sum(TF_WEIGHTS.get(t, 1.0) for t, r in all_results.items()
                            if r.get("direction") == overall)
            alignment = aligned_w / total_w if total_w else 0.0
            win_conf = winner / voted_weight if voted_weight else 0.0
            confidence = int(alignment * (60 + 40 * win_conf))

        # 1m CIRCUIT BREAKER: 15m votes on candles that closed up to 15 min
        # ago. In a vertical spike + dump the 1m already screams the opposite
        # while 15m still shows the old pump (e.g. lab LONG 92% at the top,
        # stopped seconds later). If 1m opposes overall at >= min_confidence,
        # block the entry - do not buy a dump, do not short a rip.
        veto_1m = None
        if overall != "NEUTRAL":
            r1 = all_results.get("1m") or {}
            d1, c1 = r1.get("direction"), r1.get("confidence", 0)
            if d1 and d1 != "NEUTRAL" and d1 != overall and c1 >= min_confidence:
                veto_1m = (f"1m {d1} {c1}% opposes {overall} - entry blocked "
                           f"(fast reversal in progress)")
                overall = "NEUTRAL"

        return {
            "direction": overall,
            "confidence": confidence,
            "signal_strength": strength,
            "strategies_used": sorted(strategies_used),
            "timeframe_results": all_results,
            "long_weight": long_weight,
            "short_weight": short_weight,
            "voted_weight": voted_weight,
            "veto_1m": veto_1m,
        }

    def scan_and_report(self, symbol: str = None) -> dict:
        if symbol is None:
            symbol = self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL)
        min_conf = int(self.memory.get_setting("min_confidence", Config.MIN_CONFIDENCE))
        intervals = self._resolve_intervals()
        result = self.scan_multi_timeframe(symbol, intervals, min_conf)
        result["price"] = self.get_current_price(symbol)
        result["symbol"] = symbol
        result["timestamp"] = time.time()
        if result["direction"] != "NEUTRAL" and result["confidence"] >= min_conf:
            # Skip recording if an identical signal was produced very recently
            # to avoid flooding the DB with duplicates from consecutive scans.
            recent = self.memory.get_recent_signals(symbol, limit=5)
            now = time.time()
            if not any(
                (s["direction"] == result["direction"]
                 and s["timeframe"] == ",".join(intervals)
                 and now - float(s["timestamp"]) < 120)
                for s in recent
            ):
                self.memory.record_signal(
                    symbol=symbol, direction=result["direction"],
                    strategy=",".join(result["strategies_used"]) or "MULTI",
                    timeframe=",".join(intervals), confidence=result["confidence"],
                    signal_strength=result["signal_strength"], price=result["price"],
                )
        return result

    @staticmethod
    def _lean_token(sig: dict, counted: set) -> str:
        """One strategy's standing opinion, e.g. EMA=SHORT(23%)* (*=counted)."""
        name = sig.get("strategy", "?")
        det = sig.get("details", {}) or {}
        mark = "*" if (sig.get("direction") in ("LONG", "SHORT")
                       and name in counted) else ""
        if name == "RSI" and det.get("rsi") is not None:
            if det.get("lean") and det["lean"] != "NEUTRAL":
                return f"RSI:{det['rsi']:.1f} {det['lean']}({det.get('lean_conf', 0)}%){mark}"
            return f"RSI:{det['rsi']:.1f}{mark}"
        if det.get("lean") and det["lean"] != "NEUTRAL":
            return f"{name}={det['lean']}({det.get('lean_conf', 0)}%){mark}"
        if sig.get("direction") in ("LONG", "SHORT"):
            return f"{name}={sig['direction']}({sig.get('confidence', 0)}%){mark}"
        if not det:
            return f"{name}=n/a"
        return f"{name}=flat"

    def _format_tf_line(self, r: dict) -> str:
        counted = set(r.get("strategies_used", []))
        sigs = r.get("all_signals", []) or []
        toks = [self._lean_token(s, counted) for s in sigs]
        if r["direction"] != "NEUTRAL":
            return f"{r['direction']} ({r['confidence']}%) [{' | '.join(toks)}]"
        leans = [(s["details"].get("lean_conf", 0), s["details"].get("lean"), s.get("strategy", "?"))
                 for s in sigs
                 if (s.get("details", {}) or {}).get("lean") not in (None, "NEUTRAL")]
        if leans:
            conf, side, name = max(leans)
            return f"NEUTRAL (lean {side} {conf}% via {name}) [{' | '.join(toks)}]"
        return f"NEUTRAL (flat) [{' | '.join(toks)}]"

    @staticmethod
    def _top_lean(tfs: dict):
        """Strongest opinion anywhere: (conf, source, side) or None."""
        best = None
        for tf, r in (tfs or {}).items():
            if r.get("error"):
                continue
            if r.get("direction") in ("LONG", "SHORT"):
                cand = (r.get("confidence", 0), f"{tf} TF vote", r["direction"])
                if best is None or cand[0] > best[0]:
                    best = cand
            for s in r.get("all_signals", []) or []:
                det = s.get("details", {}) or {}
                if det.get("lean") not in (None, "NEUTRAL"):
                    cand = (det.get("lean_conf", 0), f"{tf} {s.get('strategy', '?')}", det["lean"])
                    if best is None or cand[0] > best[0]:
                        best = cand
        return best

    def format_signal_report(self, result: dict) -> str:
        if "error" in result and "direction" not in result:
            return f"Signal Scan Error: {result['error']}"
        report = f"=== SIGNAL SCAN [{result.get('symbol', 'N/A')}] ===\n"
        if result.get("veto_1m"):
            report += f"VETO: {result['veto_1m']}\n"
        if result.get("veto_reason"):
            report += f"VETO: {result['veto_reason']}\n"
        tfs = result.get("timeframe_results", {})
        if result["direction"] != "NEUTRAL":
            report += f"Direction: {result['direction']}\n"
            report += f"Confidence: {result['confidence']}%\n"
            report += f"Signal Strength: {result.get('signal_strength', 0):.2f}\n"
        else:
            top = self._top_lean(tfs)
            if top:
                report += f"Direction: NEUTRAL (top lean {top[2]} {top[0]}% - {top[1]})\n"
            else:
                report += "Direction: NEUTRAL (flat - no lean data)\n"
        report += f"Price: {result.get('price', 0)}\n"
        if result.get("strategies_used"):
            report += f"Strategies: {', '.join(result['strategies_used'])}\n"
        else:
            gate = int(self.memory.get_setting("tf_min_confidence", Config.TF_MIN_CONFIDENCE))
            report += f"Strategies: none counted (all below {gate}% gate - leans per TF below)\n"
        for tf, r in tfs.items():
            if r.get("error"):
                report += f"  {tf}: no data ({r['error']})\n"
                continue
            report += f"  {tf}: {self._format_tf_line(r)}\n"
        if (result.get("voted_weight") or 0) > 0:
            longs = result.get('long_weight', 0)
            shorts = result.get('short_weight', 0)
            parts = []
            parts.append(f"Long: {longs:.2f}" if longs > 0 else "no LONG votes")
            parts.append(f"Short: {shorts:.2f}" if shorts > 0 else "no SHORT votes")
            parts.append(f"Voted: {result.get('voted_weight', 0):.2f}")
            report += "\n" + " | ".join(parts)
        else:
            report += "\nNo counted votes - every TF below gate (see leans above)."
        if "*" in report:
            report += "\n(* = counted vote)"
        return report
