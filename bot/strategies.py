import pandas as pd


class StrategyBase:
    name: str = "base"

    def calculate(self, df: pd.DataFrame) -> tuple:
        return "NEUTRAL", 0, {}


class EMAStrategy(StrategyBase):
    name = "EMA"

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        if fast_period >= slow_period:
            raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period})")
        if fast_period < 2 or slow_period < 2:
            raise ValueError("EMA periods must be >= 2")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate(self, df: pd.DataFrame) -> tuple:
        if len(df) < self.slow_period + 10:
            return "NEUTRAL", 0, {}
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast_period, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow_period, adjust=False).mean()
        df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
        prev_diff = df["ema_diff"].iloc[-2]
        curr_diff = df["ema_diff"].iloc[-1]
        if prev_diff < 0 and curr_diff > 0:
            strength = min(100, abs(curr_diff) / max(df["close"].iloc[-1], 0.01) * 10000)
            return "LONG", self._confidence(strength), {
                "fast": df["ema_fast"].iloc[-1], "slow": df["ema_slow"].iloc[-1]
            }
        elif prev_diff > 0 and curr_diff < 0:
            strength = min(100, abs(curr_diff) / max(df["close"].iloc[-1], 0.01) * 10000)
            return "SHORT", self._confidence(strength), {
                "fast": df["ema_fast"].iloc[-1], "slow": df["ema_slow"].iloc[-1]
            }
        return "NEUTRAL", 0, {}

    def _confidence(self, strength: float) -> int:
        return min(95, int(60 + strength * 2))


class MACDStrategy(StrategyBase):
    name = "MACD"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        if fast < 2 or slow < 2 or signal < 2:
            raise ValueError("MACD periods must be >= 2")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate(self, df: pd.DataFrame) -> tuple:
        if len(df) < self.slow + self.signal_period + 10:
            return "NEUTRAL", 0, {}
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["macd"] = df["ema_fast"] - df["ema_slow"]
        df["signal"] = df["macd"].ewm(span=self.signal_period, adjust=False).mean()
        df["histogram"] = df["macd"] - df["signal"]
        prev_hist = df["histogram"].iloc[-2]
        curr_hist = df["histogram"].iloc[-1]
        if curr_hist > 0 and prev_hist < 0:
            strength = min(100, abs(curr_hist) / abs(df["close"].iloc[-1]) * 50000)
            return "LONG", self._confidence(strength), {
                "macd": df["macd"].iloc[-1], "signal": df["signal"].iloc[-1],
                "histogram": curr_hist
            }
        elif curr_hist < 0 and prev_hist > 0:
            strength = min(100, abs(curr_hist) / abs(df["close"].iloc[-1]) * 50000)
            return "SHORT", self._confidence(strength), {
                "macd": df["macd"].iloc[-1], "signal": df["signal"].iloc[-1],
                "histogram": curr_hist
            }
        if curr_hist > 0 and prev_hist > 0 and df["macd"].iloc[-1] > df["macd"].iloc[-2]:
            trend_strength = abs(df["macd"].iloc[-1]) / abs(df["close"].iloc[-1]) * 10000
            conf = min(85, int(55 + trend_strength))
            return "LONG", conf, {}
        elif curr_hist < 0 and prev_hist < 0 and df["macd"].iloc[-1] < df["macd"].iloc[-2]:
            trend_strength = abs(df["macd"].iloc[-1]) / abs(df["close"].iloc[-1]) * 10000
            conf = min(85, int(55 + trend_strength))
            return "SHORT", conf, {}
        return "NEUTRAL", 0, {}

    def _confidence(self, strength: float) -> int:
        # Crossover confidence. strength is already capped at 100 by the caller
        # (histogram size scaled by *50000). Same shape as the other strategies,
        # but capped slightly under the trend-continuation cap so a histogram
        # zero-crossing never overrules a sustained MACD trend.
        return min(90, int(55 + strength * 0.5))


class RSIStrategy(StrategyBase):
    name = "RSI"

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        if period < 2:
            raise ValueError(f"RSI period must be >= 2, got {period}")
        if not (0 < oversold < overbought < 100):
            raise ValueError(f"RSI thresholds must satisfy 0 < oversold ({oversold}) < overbought ({overbought}) < 100")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def calculate(self, df: pd.DataFrame) -> tuple:
        if len(df) < self.period + 10:
            return "NEUTRAL", 0, {}
        df = df.copy()
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        prev_rsi = df["rsi"].iloc[-2]
        curr_rsi = df["rsi"].iloc[-1]
        if prev_rsi < self.oversold and curr_rsi > self.oversold:
            strength = min(100, (curr_rsi - self.oversold) * 2)
            return "LONG", self._confidence(strength), {"rsi": curr_rsi}
        elif prev_rsi > self.overbought and curr_rsi < self.overbought:
            strength = min(100, (self.overbought - curr_rsi) * 2)
            return "SHORT", self._confidence(strength), {"rsi": curr_rsi}
        elif curr_rsi < self.oversold:
            strength = min(100, (self.oversold - curr_rsi) * 2)
            return "LONG", self._confidence(strength) - 10, {"rsi": curr_rsi}
        elif curr_rsi > self.overbought:
            strength = min(100, (curr_rsi - self.overbought) * 2)
            return "SHORT", self._confidence(strength) - 10, {"rsi": curr_rsi}
        return "NEUTRAL", 0, {}

    def _confidence(self, strength: float) -> int:
        return min(90, int(60 + strength * 1.5))


class MomentumStrategy(StrategyBase):
    name = "MOMENTUM"

    def __init__(self, period: int = 10, threshold: float = 0.005):
        if period < 2:
            raise ValueError(f"Momentum period must be >= 2, got {period}")
        if threshold <= 0:
            raise ValueError(f"Momentum threshold must be > 0, got {threshold}")
        self.period = period
        self.threshold = threshold

    def calculate(self, df: pd.DataFrame) -> tuple:
        if len(df) < self.period + 10:
            return "NEUTRAL", 0, {}
        df = df.copy()
        df["momentum"] = df["close"] / df["close"].shift(self.period) - 1
        df["volume_sma"] = df["volume"].rolling(window=self.period).mean()
        current_vol = df["volume"].iloc[-1]
        avg_vol = df["volume_sma"].iloc[-1]
        vol_surge = current_vol > avg_vol * 1.2 if avg_vol > 0 else False
        curr_mom = df["momentum"].iloc[-1]
        prev_mom = df["momentum"].iloc[-2]
        if curr_mom > self.threshold and prev_mom < self.threshold and vol_surge:
            strength = min(100, abs(curr_mom) * 1000)
            return "LONG", self._confidence(strength), {
                "momentum": curr_mom, "vol_ratio": current_vol / avg_vol if avg_vol > 0 else 1
            }
        elif curr_mom < -self.threshold and prev_mom > -self.threshold and vol_surge:
            strength = min(100, abs(curr_mom) * 1000)
            return "SHORT", self._confidence(strength), {
                "momentum": curr_mom, "vol_ratio": current_vol / avg_vol if avg_vol > 0 else 1
            }
        if curr_mom > self.threshold:
            strength = min(100, abs(curr_mom) * 800)
            return "LONG", self._confidence(strength) - 15, {"momentum": curr_mom}
        elif curr_mom < -self.threshold:
            strength = min(100, abs(curr_mom) * 800)
            return "SHORT", self._confidence(strength) - 15, {"momentum": curr_mom}
        return "NEUTRAL", 0, {}

    def _confidence(self, strength: float) -> int:
        return min(90, int(55 + strength * 2))


class StrategyEngine:
    def __init__(self):
        self.strategies = {
            "EMA": EMAStrategy(),
            "MACD": MACDStrategy(),
            "RSI": RSIStrategy(),
            "MOMENTUM": MomentumStrategy(),
        }

    def calculate_all(self, df: pd.DataFrame) -> list:
        results = []
        for name, strategy in self.strategies.items():
            direction, confidence, details = strategy.calculate(df)
            results.append({
                "strategy": name,
                "direction": direction,
                "confidence": confidence,
                "details": details,
            })
        return results

    def get_consensus(self, df: pd.DataFrame, min_confidence: int = 80,
                      min_agree: int = 2) -> dict:
        results = self.calculate_all(df)
        long_signals = [r for r in results if r["direction"] == "LONG" and r["confidence"] >= min_confidence]
        short_signals = [r for r in results if r["direction"] == "SHORT" and r["confidence"] >= min_confidence]

        # RSI veto: if RSI is overbought (>70) don't LONG, if oversold (<30) don't SHORT
        # RSI is the first filter - even if 4 strategies agree LONG but RSI says overbought, veto
        rsi_entry = next((r for r in results if r["strategy"] == "RSI"), None)
        rsi_val = None
        if rsi_entry and rsi_entry.get("details", {}).get("rsi") is not None:
            rsi_val = float(rsi_entry["details"]["rsi"])
            if rsi_val >= 70:
                # Overbought - veto LONG
                long_signals = []
            if rsi_val <= 30:
                # Oversold - veto SHORT
                short_signals = []

        direction = "NEUTRAL"
        signal_strength = 0.0
        strategies_used = []
        long_score = sum(r["confidence"] for r in long_signals)
        short_score = sum(r["confidence"] for r in short_signals)
        total_score = long_score + short_score

        # RSI-FIRST: RSI has the first word. If it fires (>= min_confidence),
        # its side alone decides - the opposite side is discarded even if 2+
        # strategies agree against it. RSI needs no second vote.
        rsi_vote = None
        if (rsi_entry and rsi_entry.get("direction") in ("LONG", "SHORT")
                and rsi_entry.get("confidence", 0) >= min_confidence):
            rsi_vote = rsi_entry["direction"]
            if rsi_vote == "LONG":
                short_signals = []
            else:
                long_signals = []

        # Direction is decided by confidence mass, not signal count. With the
        # old count vote, SHORT(85) and LONG(66) tied 1:1 and the entire
        # timeframe was discarded as NEUTRAL despite a clear stronger side.
        #
        # BUT (when RSI is silent) a single strategy firing alone is no longer
        # enough: one noisy indicator on one timeframe (e.g. MOMENTUM
        # continuation on 1m noise) used to open trades by itself ->
        # instant losses. Require min_agree strategies on the winning side
        # before the timeframe votes at all.
        if rsi_vote == "LONG" and long_signals:
            direction = "LONG"
            strategies_used = [r["strategy"] for r in long_signals]
        elif rsi_vote == "SHORT" and short_signals:
            direction = "SHORT"
            strategies_used = [r["strategy"] for r in short_signals]
        elif rsi_vote is None:
            if long_score > short_score and len(long_signals) >= min_agree:
                direction = "LONG"
                strategies_used = [r["strategy"] for r in long_signals]
            elif short_score > long_score and len(short_signals) >= min_agree:
                direction = "SHORT"
                strategies_used = [r["strategy"] for r in short_signals]

        if direction != "NEUTRAL" and total_score > 0:
            signal_strength = abs(long_score - short_score) / total_score

        avg_confidence = 0
        if long_signals or short_signals:
            all_signals = long_signals + short_signals
            # Keep opposition visible in confidence: 85 SHORT against 66 LONG
            # becomes SHORT 75%, not an unopposed SHORT 85%.
            avg_confidence = int(sum(r["confidence"] for r in all_signals) / len(all_signals))

        # If vetoed, mark as NEUTRAL with veto reason
        veto_reason = None
        if rsi_val is not None:
            if rsi_val >= 70 and direction == "LONG":
                # Should not happen due to veto above, but keep as safety
                veto_reason = f"RSI {rsi_val:.1f} overbought - LONG vetoed"
                direction = "NEUTRAL"
            elif rsi_val <= 30 and direction == "SHORT":
                veto_reason = f"RSI {rsi_val:.1f} oversold - SHORT vetoed"
                direction = "NEUTRAL"

        return {
            "direction": direction,
            "confidence": avg_confidence if avg_confidence else 0,
            "signal_strength": signal_strength,
            "strategies_used": strategies_used,
            "all_signals": results,
            "long_count": len(long_signals),
            "short_count": len(short_signals),
            "rsi": rsi_val,
            "veto_reason": veto_reason,
        }
