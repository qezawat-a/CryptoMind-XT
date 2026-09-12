import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    XT_API_KEY: str = os.getenv("XT_API_KEY", "")
    XT_API_SECRET: str = os.getenv("XT_API_SECRET", "")

    # --- LLM Provider (OpenAI-compatible + Anthropic native, both custom) ---
    # AI_PROVIDER: auto | openai | anthropic
    #   auto = if ANTHROPIC_API_KEY is set and AI_API_KEY is empty -> anthropic, else openai-compatible
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "auto").lower().strip()
    AI_API_KEY: str = os.getenv("AI_API_KEY", "")
    # Comma-separated list for rotation when 50 keys exhausted (e.g. AI_API_KEYS=sk1,sk2,sk3)
    AI_API_KEYS: str = os.getenv("AI_API_KEYS", "")
    AI_BASE_URL: str = os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
    # When empty, auto-detects a chat model from the provider's /models endpoint.
    AI_MODEL: str = os.getenv("AI_MODEL", "")

    # Anthropic native API (custom base_url support as requested)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "")

    # Fallback chain: comma-separated models to try in order before auto-detection
    # e.g. AI_FALLBACK_MODELS=gpt-4o-mini,claude-3-5-haiku,gemini-flash
    AI_FALLBACK_MODELS: str = os.getenv("AI_FALLBACK_MODELS", "")

    # Agent behavior
    AGENT_MAX_STEPS: int = int(os.getenv("AGENT_MAX_STEPS", "8"))
    AGENT_AUTONOMOUS_INTERVAL_SEC: int = int(os.getenv("AGENT_AUTONOMOUS_INTERVAL_SEC", "15"))
    AGENT_DRY_RUN: str = os.getenv("AGENT_DRY_RUN", "false").lower()  # true = agent analyzes but does not open trades

    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_USER_ID: str = os.getenv("TELEGRAM_USER_ID", "")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/memory.db")

    XT_FUTURES_HOST: str = os.getenv("XT_FUTURES_HOST", "https://fapi.xt.com")

    DEFAULT_SYMBOL: str = os.getenv("DEFAULT_SYMBOL", "aero_usdt")
    DEFAULT_LEVERAGE: int = 75
    DEFAULT_MARGIN_MODE: str = "CROSSED"  # XT doc: positionType CROSSED/ISOLATED (Change Position Type)
    # Alias for doc - position_type is same as margin_mode, keep both for compat
    DEFAULT_POSITION_TYPE: str = "CROSSED"
    # Timeframes tuned with user: 1m,3m for micro entry, 15m for main trend
    # 4h/1h removed (dominated weight 59% and hid micro pullbacks), now 15m is dominant 53% weight
    DEFAULT_TIMEFRAMES: list = ["1m", "3m", "5m", "15m"]
    DEFAULT_MARGIN_AMOUNT_PCT: float = 25.0
    DEFAULT_RISK_PCT: float = 1.0
    SIGNAL_COOLDOWN_MINUTES: int = 3
    MAX_POSITIONS: int = 1
    MIN_CONFIDENCE: int = 80

    TF_MIN_CONFIDENCE: int = 70
    MIN_AGREEING_STRATEGIES: int = 2
    SIGNAL_CONFIRM_SCANS: int = 1

    SCAN_INTERVAL_SEC: int = 15
    GUARD_INTERVAL_SEC: int = 15
    # max_loss/max_profit are now dynamic per signal + liq distance, not user settings
    # Kept for backward compat but not shown in settings
    # ROI-on-margin thresholds for stop management. These are the values that
    # get seeded whenever a setting is missing from the DB (e.g. after a wipe),
    # so they are the owner's intended defaults - NOT the tiny 8%/15% that
    # yanked stops at noise level on 50x leverage. At 50x, 30% ROI is a 0.6%
    # price move and 45% is ~0.9%: real profit, not market noise.
    BREAKEVEN_THRESHOLD_PCT: float = 15.0
    TRAILING_STOP_PCT: float = 20.0

    TRAILING_TRIGGER_ROI_PCT: float = 20.0
    TRAILING_DISTANCE_PCT: float = 0.5

    SL_LIQUIDATION_SAFETY: float = 0.5
    ON_TPSL_FAILURE: str = "close"

    # Close an open position when the opposite-direction signal reaches this
    # confidence (reversal logic). Disabled when REVERSAL_ENABLED is False.
    REVERSAL_ENABLED: bool = True
    REVERSAL_CONFIDENCE: int = 79
    # How often (seconds) to push a PnL + confidence report even with no events.
    REPORT_INTERVAL_SEC: int = 30

    @classmethod
    def get_effective_provider(cls) -> str:
        """Return 'anthropic' or 'openai' based on AI_PROVIDER and keys."""
        if cls.AI_PROVIDER == "anthropic":
            return "anthropic"
        if cls.AI_PROVIDER == "openai":
            return "openai"
        # auto
        if cls.ANTHROPIC_API_KEY and not cls.AI_API_KEY:
            return "anthropic"
        return "openai"

    @classmethod
    def get_effective_api_key(cls) -> str:
        return cls.ANTHROPIC_API_KEY if cls.get_effective_provider() == "anthropic" else cls.AI_API_KEY

    @classmethod
    def get_effective_base_url(cls) -> str:
        return cls.ANTHROPIC_BASE_URL if cls.get_effective_provider() == "anthropic" else cls.AI_BASE_URL

    @classmethod
    def get_effective_model(cls) -> str:
        return cls.ANTHROPIC_MODEL if cls.get_effective_provider() == "anthropic" else cls.AI_MODEL

    @classmethod
    def validate(cls) -> list:
        missing = []
        # XT creds can come from env OR ~/.xt-tradekit/credentials.json, so not strictly required here
        # AI key: either AI_API_KEY or ANTHROPIC_API_KEY must be present
        # For local LLM (localhost/127.0.0.1/trycloudflare) key is not needed - allow dummy
        base = (cls.get_effective_base_url() or "").lower()
        is_local = any(x in base for x in ("localhost", "127.0.0.1", "trycloudflare.com", "railway.internal"))
        if not cls.AI_API_KEY and not cls.ANTHROPIC_API_KEY and not is_local:
            missing.append("AI_API_KEY or ANTHROPIC_API_KEY")
        if not cls.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not cls.TELEGRAM_USER_ID:
            missing.append("TELEGRAM_USER_ID")
        if cls.TELEGRAM_USER_ID and not cls.TELEGRAM_USER_ID.strip().isdigit():
            missing.append("TELEGRAM_USER_ID (must be a numeric Telegram user id)")
        if cls.DEFAULT_LEVERAGE < 1 or cls.DEFAULT_LEVERAGE > 125:
            missing.append("DEFAULT_LEVERAGE should be between 1 and 125")
        if cls.DEFAULT_MARGIN_AMOUNT_PCT < 1 or cls.DEFAULT_MARGIN_AMOUNT_PCT > 100:
            missing.append("DEFAULT_MARGIN_AMOUNT_PCT should be between 1 and 100")
        if cls.DEFAULT_RISK_PCT < 0.1 or cls.DEFAULT_RISK_PCT > 10:
            missing.append("DEFAULT_RISK_PCT should be between 0.1 and 10")
        if cls.MIN_CONFIDENCE < 50 or cls.MIN_CONFIDENCE > 100:
            missing.append("MIN_CONFIDENCE should be between 50 and 100")
        if cls.SL_LIQUIDATION_SAFETY <= 0 or cls.SL_LIQUIDATION_SAFETY > 1:
            missing.append("SL_LIQUIDATION_SAFETY should be between 0 and 1")
        if cls.BREAKEVEN_THRESHOLD_PCT <= 0 or cls.BREAKEVEN_THRESHOLD_PCT > 100:
            missing.append("BREAKEVEN_THRESHOLD_PCT should be between 0 and 100")
        if cls.TRAILING_DISTANCE_PCT <= 0 or cls.TRAILING_DISTANCE_PCT > 20:
            missing.append("TRAILING_DISTANCE_PCT should be between 0 and 20")
        return missing

    @classmethod
    def default_settings(cls) -> dict:
        return {
            "symbol": cls.DEFAULT_SYMBOL,
            "leverage": cls.DEFAULT_LEVERAGE,
            "margin_mode": cls.DEFAULT_MARGIN_MODE,
            "position_type": cls.DEFAULT_POSITION_TYPE,
            "timeframes": ",".join(cls.DEFAULT_TIMEFRAMES),
            "margin_amount_pct": cls.DEFAULT_MARGIN_AMOUNT_PCT,
            "margin_risk_pct": cls.DEFAULT_RISK_PCT,
            "min_confidence": cls.MIN_CONFIDENCE,
            "tf_min_confidence": cls.TF_MIN_CONFIDENCE,
            "min_agreeing_strategies": cls.MIN_AGREEING_STRATEGIES,
            "signal_confirm_scans": cls.SIGNAL_CONFIRM_SCANS,
            "cooldown_minutes": cls.SIGNAL_COOLDOWN_MINUTES,
            "max_positions": cls.MAX_POSITIONS,
            "position_mode": "margin",
            "scan_interval_sec": cls.SCAN_INTERVAL_SEC,
            "guard_interval_sec": cls.GUARD_INTERVAL_SEC,
            "breakeven_threshold_pct": cls.BREAKEVEN_THRESHOLD_PCT,
            "trailing_stop_pct": cls.TRAILING_STOP_PCT,
            "trailing_trigger_roi_pct": cls.TRAILING_TRIGGER_ROI_PCT,
            "trailing_distance_pct": cls.TRAILING_DISTANCE_PCT,
            "sl_liquidation_safety": cls.SL_LIQUIDATION_SAFETY,
            "on_tpsl_failure": cls.ON_TPSL_FAILURE,
            "reversal_enabled": cls.REVERSAL_ENABLED,
            "reversal_confidence": cls.REVERSAL_CONFIDENCE,
            "report_interval_sec": cls.REPORT_INTERVAL_SEC,
        }

    # Keys that are legacy (removed from default_settings) and should be auto-deleted from DB
    LEGACY_SETTINGS = {"max_loss_pct", "max_profit_pct"}
    # margin_mode is alias to position_type per doc - keep both but display as position_type
    SETTING_ALIASES = {"margin_mode": "position_type"}

    @classmethod
    def to_dict(cls) -> dict:
        redacted = {"XT_API_SECRET", "XT_API_KEY", "AI_API_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN"}
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith("_") and k.isupper() and k not in redacted}
