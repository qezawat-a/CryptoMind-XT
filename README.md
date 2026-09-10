# CryptoMind-XT — AI-Powered XT.com Futures Trading Bot

**🇬🇧 English** | [🇮🇷 فارسی](README.fa.md)

An automated **USDT-M futures trading bot** for the **XT.com** exchange, driven by multi-timeframe signal scanning, an OpenAI-compatible **AI assistant**, and full control from **Telegram**.

> ⚠️ **Warning:** This bot trades real money. Read `risk_manager.py` and `position_manager.py` and understand the risk settings before using it. You are solely responsible for any losses.

---

## Overview

- **Multi-timeframe signal scanning** — combines 4 strategies (EMA, MACD, RSI, Momentum) with a weighted vote across several timeframes
- **Exchange-side TP/SL** — every position immediately gets a real "profit/stop" order on XT (not just a software stop)
- **Mid-position management** — break-even stop, trailing stop, TP/SL recovery, exchange reconciliation
- **AI Brain** — function-calling assistant (OpenAI-compatible) that manages settings and analyzes state via Telegram chat
- **Risk management** — margin-% / risk-% position sizing, exchange-bracket leverage clamping, cached balance
- **Persistence** — trade history and settings in SQLite (local) or Neon Postgres / MySQL (Railway)

---

## Architecture

```
main.py                 → startup, health server, XT + Telegram wiring (AGENT MODE)
config.py               → environment (.env) config + defaults
bot/xt_client.py        → XT futures API client (HMAC-SHA256 signing, 429 handling)
bot/risk_manager.py     → position sizing, leverage clamping, balance
bot/position_manager.py → exchange TP/SL, break-even, trailing, closing, reconciliation
bot/signal_scanner.py   → multi-timeframe scan (RSI veto + strategy agreement)
bot/strategies.py       → EMA / MACD / RSI / Momentum strategies
bot/trader.py           → core trading logic (gates, execution, guardian loops)
bot/memory.py           → SQLAlchemy data models
agent/                  → ReAct agent: brain.py (LLM), core.py (loop), tools.py (19 chat / 15 autonomous tools), telegram_agent.py
```

---

## Local Install & Run

### Prerequisites
- Python 3.9+
- An XT.com account with **futures enabled** (USDT-M futures activated)
- An API key with futures permission (plus IP whitelist if you use it)
- A Telegram bot token (from `@BotFather`)

### Steps

```bash
# 1. Clone
git clone https://github.com/aqezawat-collab/CryptoMind-XT.git
cd CryptoMind-XT

# 2. Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your config from the sample
cp .env.example .env
```

### `.env` configuration

```env
XT_API_KEY=your_xt_api_key_here          # XT futures API key
XT_API_SECRET=your_xt_api_secret_here    # XT futures API secret

AI_API_KEY=your_openai_api_key_here      # AI key (OpenAI or any compatible service)
AI_BASE_URL=https://api.openai.com/v1    # OpenAI-compatible base URL
AI_MODEL=                                 # OPTIONAL — leave blank to auto-pick a chat model from the provider's /models endpoint

TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here   # Telegram bot token
TELEGRAM_USER_ID=your_telegram_user_id_here       # numeric user id (security)
DATABASE_URL=sqlite:///data/memory.db             # or a MySQL DSN
```

> `TELEGRAM_USER_ID` must be **numeric**. Ask `@userinfobot` on Telegram to find your numeric ID.

### Run

```bash
python3 main.py
```

Healthy startup logs:
```
XT connection OK. USDT wallet balance: ...
XT AI Trader started. Telegram bot is listening...
```

If you see a `signature` error or `XT API check failed`, see the **Signature troubleshooting** section below.

---

## Deploy on Railway (recommended)

> The bot is designed to run **better on Railway** — it stays online 24/7, whereas a local run stops whenever your machine is off. Be sure to point `DATABASE_URL` at a persistent DB (Neon Postgres recommended, MySQL also works — SQLite inside the container is wiped on every deploy).

### Steps

1. **Push the code to GitHub**, then in [railway.app](https://railway.app) create a new project → **Deploy from GitHub repo**.
2. Railway reads `railway.toml` / `nixpacks.toml` and handles build + start automatically.
3. Add a **Neon Postgres** database (or a **MySQL** service from the Railway catalog).
4. In the **Variables** tab set:

```env
XT_API_KEY=...
XT_API_SECRET=...
AI_API_KEY=...
AI_BASE_URL=https://api.openai.com/v1
AI_MODEL=gpt-4o
TELEGRAM_BOT_TOKEN=...
TELEGRAM_USER_ID=...
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler....neon.tech/neondb?sslmode=require   # important: persistent DB, e.g. Neon (or ${{ MySQL.MYSQL_URL }})
```

5. **Deploy.** Railway gives you a `*.up.railway.app` domain automatically.

> Without `DATABASE_URL` the bot falls back to local SQLite, which is **wiped on every deploy** (it logs a warning).

---

## AI Options

The bot works with **any OpenAI-compatible API**. Change the base URL and model in `.env` or Railway Variables:

| Variable | Example | Notes |
|---|---|---|
| `AI_API_KEY` | `sk-...` | API key |
| `AI_BASE_URL` | `https://api.openai.com/v1` | OpenAI's default; for others (e.g. Groq, DeepSeek, compatible proxies) put their URL |
| `AI_MODEL` | _(blank)_ | Optional. Leave blank to **auto-detect** a chat model from the provider's `/models` endpoint; set it to force a specific model |

Notes:
- The model must support **function calling / tools** (the bot uses them to change settings).
- **Auto model selection:** when `AI_MODEL` is empty, the bot calls the OpenAI-compatible `/models` endpoint of `AI_BASE_URL`, ranks the chat-capable models (free-tier first, then smaller/cheaper, then the rest), and at startup **probes each with a tiny test call**, starting on the first one that actually responds. So the model it picks is one your account can genuinely use on **any** OpenAI-compatible endpoint — not just OpenRouter — instead of a name that only looks good. On providers that expose free-tier models (OpenRouter `:free` suffix) it prefers a free model whenever one is listed, giving a zero-cost, usable model regardless of credit. If `/models` is unsupported or returns no chat model, it falls back to `gpt-4o`. You can always override by setting `AI_MODEL` explicitly.
- **Balance-aware fallback:** auto mode ranks candidates (free → cheap → capable) and, if the chosen model is rejected with a balance/quota error (e.g. a `402` "balance is positive but not enough", or OpenAI `insufficient_quota`), it automatically switches to the next affordable model and retries — so it keeps trying until it lands on a model your account can actually pay for. A plain rate-limit `429` is treated as transient and does **not** trigger a model switch.
- With a custom `AI_BASE_URL` + `AI_MODEL` you can switch from OpenAI to any compatible service.
- Auto model selection is **endpoint-agnostic**: it works for any OpenAI-compatible `/models` endpoint (Groq, DeepSeek, OpenRouter, OpenAI, local servers, ...). The `:free` preference is a generic suffix test, not an OpenRouter hardcode — providers without `:free` models simply fall through to the normal family-based pick.
- Use `/check_ai` to see which model was selected (`AI model: <name> (auto|explicit)`).

---

## Telegram Commands

| Command | Purpose |
|---|---|
| `/start` | Command list |
| `/status` | Full status, positions, PnL, settings |
| `/pnl` | PnL report |
| `/balance` | USDT futures balance |
| `/signal` | Run a signal scan now |
| `/history` | Position history |
| `/trades` | Trade history |
| `/liq` | Margin call / liquidation info |
| `/agent` | Force one agent decision tick now |
| `/close [trade_id]` | Close a specific / all positions |
| `/diag` | Why break-even/trailing has not fired |
| `/settings` | Show current settings |
| `/check_ai` | Test the AI connection |

**Plain chat works too** — e.g. "set leverage to 20", "what's the balance?", "show status". The agent runs its own autonomous loop (trades by itself); chat is for questions and explicit setting changes.

---

## Important Settings & Risk Warning

- **Default `leverage` is 75x** with `SL_LIQUIDATION_SAFETY=0.5` — the stop sits at most halfway to the liquidation price; at high leverage that distance is very small. **High leverage = high risk.**
- `on_tpsl_failure=close` — if XT rejects the stop, the bot **closes the position** instead of leaving it unprotected.
- `position_mode` — `margin` (size by margin %) or `risk` (size by risk % against the stop).
- Software stop is **dynamic** (a fraction of the liquidation distance, not a fixed `max_loss_pct` — those legacy keys are auto-deleted from the DB).
- `cooldown_minutes` — blocks re-entry after a close.
- **RSI veto + agreement** — RSI ≥ 70 vetoes LONG, RSI ≤ 30 vetoes SHORT; besides RSI, ≥2 strategies must agree (`min_agreeing_strategies`).
- **Settings lock** — the autonomous loop can never change leverage/margin/symbol/settings by itself (chat-only tools); `open_trade` overrides apply to one trade only and are restored afterwards.

---

## Signature Troubleshooting

If you get signature-validation errors despite correct keys:

1. The key needs **futures permission** and the **futures account must be open**.
2. If IP whitelisting is on, add the server IP.
3. The XT signature is `#{path}#{message}` and when the message is empty there is **no trailing `#`**. (This exact regression was fixed in PR #4 — if you run old code, `git pull`.)

---

## Official References

- [XT Futures API Docs](https://doc.xt.com/docs/futures/Access%20Description/BasicInformationOfTheInterface)
- [Official Python SDK (pyxt)](https://github.com/kelvinxue/pyxt)
- [Railway Docs](https://docs.railway.app)

---

**Disclaimer:** Using this bot means accepting full responsibility for financial risk. Futures trading can result in the loss of your entire capital.
