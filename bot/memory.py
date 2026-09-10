import json
import os
import time
from datetime import datetime
from threading import Lock
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(Float, nullable=False)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    position_side = Column(String(16), nullable=False)
    order_id = Column(String(64))
    entry_price = Column(Float)
    exit_price = Column(Float)
    amount = Column(Float)
    leverage = Column(Integer)
    pnl = Column(Float, default=0)
    confidence = Column(Integer)
    strategy = Column(String(128))
    signal_strength = Column(Float)
    timeframe = Column(String(128))
    opened_at = Column(Float)
    closed_at = Column(Float)
    status = Column(String(16), default="OPEN")
    notes = Column(Text)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False)
    direction = Column(String(16), nullable=False)
    strategy = Column(String(128), nullable=False)
    timeframe = Column(String(128), nullable=False)
    confidence = Column(Integer, nullable=False)
    signal_strength = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(Float, nullable=False)
    acted = Column(Boolean, default=False)


class Setting(Base):
    __tablename__ = "settings"
    key = Column(String(128), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(Float, nullable=False)


class Cooldown(Base):
    __tablename__ = "cooldowns"
    symbol = Column(String(32), primary_key=True)
    side = Column(String(16), primary_key=True)
    cooldown_until = Column(Float, nullable=False)


class AIContext(Base):
    __tablename__ = "ai_context"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(128), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    updated_at = Column(Float, nullable=False)


class LongTermMemory:
    _instance = None
    _lock = Lock()

    def __new__(cls, database_url: str = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, database_url: str = None):
        if self._initialized:
            return
        if database_url is None:
            database_url = os.getenv("DATABASE_URL", "sqlite:///data/memory.db")
        if database_url.startswith("mysql://"):
            database_url = database_url.replace("mysql://", "mysql+pymysql://", 1)
        elif database_url.startswith("postgres://"):
            # Heroku-style URL -> SQLAlchemy + psycopg2 (also used by some providers)
            database_url = database_url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif database_url.startswith("postgresql://") and "+psycopg2" not in database_url and "+asyncpg" not in database_url:
            # Plain postgresql:// (Neon default) -> use psycopg2 driver explicitly
            database_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        if "+psycopg2://" in database_url and "channel_binding=" in database_url:
            # psycopg2 (v2) does not understand channel_binding=require (only psycopg v3 does).
            # Neon adds it by default - strip it, sslmode=require is enough.
            import re
            database_url = re.sub(r"[?&]channel_binding=[^&]*", "", database_url)
            database_url = database_url.replace("?&", "?").rstrip("?&")
        self._database_url = database_url
        self._is_mysql = "mysql" in database_url
        self._is_postgres = "postgres" in database_url
        extra_args = {}
        if self._is_mysql:
            extra_args = {
                "pool_size": 5,
                "max_overflow": 10,
                "pool_recycle": 3600,
                "pool_pre_ping": True,
            }
        elif self._is_postgres:
            # Neon / Postgres: pool_pre_ping avoids stale pooled connections,
            # pool_recycle keeps Neon autosuspend happy.
            extra_args = {
                "pool_size": 5,
                "max_overflow": 10,
                "pool_recycle": 300,
                "pool_pre_ping": True,
            }
        else:
            db_path = database_url.replace("sqlite:///", "", 1)
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)
        self._engine = create_engine(self._database_url, echo=False, **extra_args)
        self._session_factory = scoped_session(sessionmaker(bind=self._engine))
        Base.metadata.create_all(self._engine)
        self._initialized = True

    def _session(self):
        return self._session_factory()

    def add_chat_message(self, role: str, content: str):
        s = self._session()
        try:
            s.add(ChatHistory(role=role, content=content, timestamp=time.time()))
            s.commit()
        finally:
            s.close()

    def get_chat_history(self, limit: int = 50) -> list:
        s = self._session()
        try:
            rows = s.query(ChatHistory).order_by(ChatHistory.id.desc()).limit(limit).all()
            return [{"role": r.role, "content": r.content} for r in reversed(rows)]
        finally:
            s.close()

    def record_trade(self, symbol: str, position_side: str, order_id,
                     entry_price: float, amount: float, leverage: int,
                     confidence: int, strategy: str, signal_strength: float,
                     timeframe: str):
        s = self._session()
        try:
            trade = Trade(
                symbol=symbol, position_side=position_side,
                order_id=str(order_id) if order_id is not None else None,
                entry_price=entry_price, amount=amount, leverage=leverage,
                confidence=confidence, strategy=strategy, signal_strength=signal_strength,
                timeframe=timeframe, opened_at=time.time(), status="OPEN",
            )
            s.add(trade)
            s.commit()
            return trade.id
        finally:
            s.close()

    def close_trade(self, trade_id: int, exit_price: float, pnl: float = 0.0,
                    notes: str = None):
        s = self._session()
        try:
            trade = s.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.closed_at = time.time()
                trade.status = "CLOSED"
                if notes:
                    trade.notes = notes
                s.commit()
        finally:
            s.close()

    def get_open_trades(self, symbol: str = None) -> list:
        s = self._session()
        try:
            q = s.query(Trade).filter(Trade.status == "OPEN")
            if symbol:
                q = q.filter(Trade.symbol == symbol)
            return [self._trade_to_dict(r) for r in q.all()]
        finally:
            s.close()

    def get_trade(self, trade_id: int) -> dict:
        s = self._session()
        try:
            row = s.query(Trade).filter(Trade.id == trade_id).first()
            return self._trade_to_dict(row) if row else {}
        finally:
            s.close()

    def get_trade_history(self, limit: int = 20) -> list:
        s = self._session()
        try:
            rows = s.query(Trade).order_by(Trade.id.desc()).limit(limit).all()
            return [self._trade_to_dict(r) for r in rows]
        finally:
            s.close()

    def get_total_pnl(self) -> float:
        s = self._session()
        try:
            result = s.query(Trade).filter(Trade.status == "CLOSED").all()
            return sum(r.pnl or 0 for r in result)
        finally:
            s.close()

    def get_trade_count(self) -> dict:
        s = self._session()
        try:
            total = s.query(Trade).count()
            closed = s.query(Trade).filter(Trade.status == "CLOSED").count()
            wins = s.query(Trade).filter(Trade.status == "CLOSED", Trade.pnl > 0).count()
            losses = s.query(Trade).filter(Trade.status == "CLOSED", Trade.pnl < 0).count()
            flat_or_unknown = closed - wins - losses
            open_count = s.query(Trade).filter(Trade.status == "OPEN").count()
            decided = wins + losses
            return {
                "total": total, "open": open_count, "closed": closed,
                "wins": wins, "losses": losses,
                "flat_or_unknown": flat_or_unknown,
                # Zero/unknown PnL rows are excluded from win rate, but are still
                # correctly counted as closed trades.
                "winrate": round(wins / decided * 100, 2) if decided > 0 else 0,
            }
        finally:
            s.close()

    def record_signal(self, symbol: str, direction: str, strategy: str,
                      timeframe: str, confidence: int, signal_strength: float,
                      price: float):
        s = self._session()
        try:
            s.add(Signal(
                symbol=symbol, direction=direction, strategy=strategy,
                timeframe=timeframe, confidence=confidence,
                signal_strength=signal_strength, price=price, timestamp=time.time(),
            ))
            s.commit()
        finally:
            s.close()

    def get_recent_signals(self, symbol: str = None, limit: int = 50) -> list:
        s = self._session()
        try:
            q = s.query(Signal)
            if symbol:
                q = q.filter(Signal.symbol == symbol)
            q = q.order_by(Signal.id.desc()).limit(limit)
            return [self._signal_to_dict(r) for r in q.all()]
        finally:
            s.close()

    def set_cooldown(self, symbol: str, side: str, duration_minutes: int):
        # Cap cooldown to max 10 min to prevent 4868s bug (4868/60=81min)
        duration_minutes = max(0, min(int(duration_minutes), 10))
        s = self._session()
        try:
            cooldown_until = time.time() + duration_minutes * 60
            existing = s.query(Cooldown).filter(
                Cooldown.symbol == symbol, Cooldown.side == side
            ).first()
            if existing:
                existing.cooldown_until = cooldown_until
            else:
                s.add(Cooldown(symbol=symbol, side=side, cooldown_until=cooldown_until))
            s.commit()
        finally:
            s.close()

    def is_in_cooldown(self, symbol: str, side: str) -> bool:
        s = self._session()
        try:
            row = s.query(Cooldown).filter(
                Cooldown.symbol == symbol, Cooldown.side == side
            ).first()
            return row is not None and row.cooldown_until > time.time()
        finally:
            s.close()

    def get_cooldown_remaining(self, symbol: str, side: str) -> float:
        s = self._session()
        try:
            row = s.query(Cooldown).filter(
                Cooldown.symbol == symbol, Cooldown.side == side
            ).first()
            if row and row.cooldown_until > time.time():
                return max(0, row.cooldown_until - time.time())
            return 0
        finally:
            s.close()

    def set_setting(self, key: str, value):
        s = self._session()
        try:
            existing = s.query(Setting).filter(Setting.key == key).first()
            if existing:
                existing.value = str(value)
                existing.updated_at = time.time()
            else:
                s.add(Setting(key=key, value=str(value), updated_at=time.time()))
            s.commit()
        finally:
            s.close()

    def set_setting_default(self, key: str, value) -> bool:
        """Writes only when the key is absent, so restarts don't clobber user settings."""
        s = self._session()
        try:
            if s.query(Setting).filter(Setting.key == key).first():
                return False
            s.add(Setting(key=key, value=str(value), updated_at=time.time()))
            s.commit()
            return True
        finally:
            s.close()

    def get_setting(self, key: str, default=None):
        s = self._session()
        try:
            row = s.query(Setting).filter(Setting.key == key).first()
            return row.value if row else default
        finally:
            s.close()

    def get_all_settings(self) -> dict:
        s = self._session()
        try:
            return {r.key: r.value for r in s.query(Setting).all()}
        finally:
            s.close()

    def set_ai_context(self, key: str, value: str):
        s = self._session()
        try:
            existing = s.query(AIContext).filter(AIContext.key == key).first()
            if existing:
                existing.value = value
                existing.updated_at = time.time()
            else:
                s.add(AIContext(key=key, value=value, updated_at=time.time()))
            s.commit()
        finally:
            s.close()

    def get_ai_context(self, key: str = None) -> dict:
        s = self._session()
        try:
            if key:
                row = s.query(AIContext).filter(AIContext.key == key).first()
                return row.value if row else None
            return {r.key: r.value for r in s.query(AIContext).order_by(AIContext.updated_at.desc()).all()}
        finally:
            s.close()

    def get_trade_summary_for_ai(self) -> str:
        trades = self.get_trade_history(30)
        stats = self.get_trade_count()
        pnl = self.get_total_pnl()
        open_trades = self.get_open_trades()
        settings = self.get_all_settings()

        summary = "=== TRADE SUMMARY ===\n"
        summary += f"Total PnL: {pnl:.4f} USDT\n"
        summary += f"Total Trades: {stats['total']} | Open: {stats['open']} | Closed: {stats['closed']}\n"
        summary += (f"Wins: {stats['wins']} | Losses: {stats['losses']} | "
                    f"Flat/Unknown PnL: {stats['flat_or_unknown']} | "
                    f"Winrate: {stats['winrate']}%\n\n")

        if open_trades:
            summary += "--- OPEN POSITIONS ---\n"
            for t in open_trades:
                summary += (f"ID:{t['id']} {t['symbol']} {t['position_side']} "
                            f"Entry:{t['entry_price']} Amt:{t['amount']} "
                            f"Lev:{t['leverage']}x | {t['strategy']} "
                            f"Conf:{t['confidence']}%\n")

        if trades:
            summary += "\n--- RECENT TRADES ---\n"
            for t in trades[:5]:
                summary += (f"{t['symbol']} {t['position_side']} "
                            f"Entry:{t['entry_price']} Exit:{t['exit_price']} "
                            f"PnL:{t['pnl']:.4f} | {t['strategy']}\n")

        if settings:
            summary += "\n--- ACTIVE SETTINGS ---\n"
            for k, v in settings.items():
                summary += f"{k}: {v}\n"

        return summary

    def close(self):
        self._session_factory.remove()
        self._engine.dispose()

    @staticmethod
    def _trade_to_dict(trade) -> dict:
        return {c.name: getattr(trade, c.name) for c in trade.__table__.columns}

    @staticmethod
    def _signal_to_dict(signal) -> dict:
        return {c.name: getattr(signal, c.name) for c in signal.__table__.columns}
