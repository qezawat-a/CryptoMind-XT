# CryptoMind-XT — ربات معامله‌گر هوشمند فیوچرز XT

[🇬🇧 English](README.md) | **🇮🇷 فارسی**

ربات معامله‌گر فیوچرز (USDT-M) صرافی **XT.com** که با **هوش مصنوعی**، اسکن سیگنال چندتایم‌فریم و کنترل کامل از طریق **تلگرام** معامله می‌کند.

> ⚠️ **توجه:** این ربات با پول واقعی معامله می‌کند. قبل از استفاده حتماً فایل‌های `risk_manager.py` و `position_manager.py` را بخوانید و تنظیمات ریسک را درک کنید. هیچ‌کس جز خود شما مسئول ضررهای احتمالی نیست.

---

## چکیده (چه کار می‌کند)

- **اسکن سیگنال چندتایم‌فریمی** — ترکیب ۴ استراتژی (EMA، MACD، RSI، Momentum) در چند تایم‌فریم با رأی‌گیری وزن‌دار
- **TP/SL خودکار روی خود صرافی** — هر موقعیت بلافاصله تیکت «profit/stop» صرافی می‌گیرد (نه فقط استاپ نرم‌افزاری)
- **مدیریت میانی موقعیت** — استاپ‌بریک‌ایون (break-even)، تریلینگ استاپ، بازیابی TP/SL، تازه‌سازی با صرافی
- **هوش مصنوعی (AI Brain)** — دستیار مبتنی بر function-calling (OpenAI-compatible) که از طریق چت تلگرام تنظیمات را مدیریت و وضعیت را تحلیل می‌کند
- **مدیریت ریسک** — سایز موقعیت بر پایه درصد مارجین یا درصد ریسک، کلَمپ لوریج بر اساس براکت صرافی، حافظهٔ موقت برای بالانس
- **ذخیرهٔ دائمی** — تاریخچهٔ معاملات و تنظیمات در SQLite (لوکال) یا Neon Postgres / MySQL (در Railway)

---

## معماری

```
main.py                 → راه‌اندازی، سرور سلامت (health check)، اتصال به XT و تلگرام
config.py               → تنظیمات از متغیرهای محیطی (.env) + مقادیر پیش‌فرض
bot/xt_client.py        → کلاینت API فیوچرز XT (امضای HMAC-SHA256، مدیریت نرخ ۴۲۹)
bot/risk_manager.py     → سایز موقعیت، کلَمپ لوریج، بالانس
bot/position_manager.py → TP/SL صرافی، بریک‌ایون، تریلینگ، بستن، تطبیق با صرافی
bot/signal_scanner.py   → اسکن چندتایم‌فریم
bot/strategies.py       → استراتژی‌های EMA / MACD / RSI / Momentum
bot/ai_chat.py          → دستیار هوش مصنوعی با function-calling
bot/trader.py           → منطق اصلی معامله (گیت‌ها، اجرا، حلقهٔ خودکار)
bot/telegram_bot.py     → ربات تلگرام (دستورات + چت با AI)
bot/memory.py           → مدل‌های پایگاه‌داده (SQLAlchemy)
```

---

## نصب و اجرای محلی (Local)

### پیش‌نیاز
- پایتون ۳.۹+
- اکانت XT.com با **فیوچرز باز** (فعال کردن USDT-M futures)
- API Key با دسترسی فیوچرز (و در صورت نیاز IP-whitelist سرور)
- ربات تلگرام (از `@BotFather`) + توکن

### مراحل

```bash
# 1. کلون
git clone https://github.com/aqezawat-collab/CryptoMind-XT.git
cd CryptoMind-XT

# 2. محیط مجازی (توصیه‌شده)
python3 -m venv venv
source venv/bin/activate        # لینوکس/mac
# venv\Scripts\activate         # ویندوز

# 3. نصب وابستگی‌ها
pip install -r requirements.txt

# 4. ساخت فایل تنظیمات از نمونه
cp .env.example .env
```

### پیکربندی `.env`

```env
XT_API_KEY=your_xt_api_key_here          # کلید API صرافی (دسترسی فیوچرز)
XT_API_SECRET=your_xt_api_secret_here    # سکرت API صرافی

AI_API_KEY=your_openai_api_key_here      # کلید هوش مصنوعی (OpenAI یا هر سازگار)
AI_BASE_URL=https://api.openai.com/v1    # آدرس سازگار با OpenAI (مثلاً برای مدل‌های دیگر)
AI_MODEL=gpt-4o                          # نام مدل

TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here   # توکن ربات تلگرام
TELEGRAM_USER_ID=your_telegram_user_id_here       # عددی (شناسه عددی کاربر؛ برای امنیت)
DATABASE_URL=sqlite:///data/memory.db             # یا DSN مای‌اس‌کیول
```

> `TELEGRAM_USER_ID` باید **عددی** باشد. برای پیدا کردن شناسهٔ عددی، به ربات `@userinfobot` در تلگرام پیام بدهید.

### اجرا

```bash
python3 main.py
```

خروجی‌های سالم:
```
XT connection OK. USDT wallet balance: ...
XT AI Trader started. Telegram bot is listening...
```

اگر در لاگ خطای `signature` یا `XT API check failed` دیدید، در انتهای همین فایل بخش «رفع مشکل امضا» را ببینید.

---

## استقرار روی Railway (پیشنهادی)

> ربات طوری طراحی شده که **بهتر روی Railway** اجرا شود — همیشه آنلاین می‌ماند، در حالی که در اجرای محلی اگر سیستم خاموش شود ربات قطع می‌شود. پس حتماً دیتابیس را به یک دیتابیس دائمی وصل کنید (Neon Postgres پیشنهادی، MySQL هم کار می‌کند؛ SQLite داخل کانتینر با هر دیپلوی از بین می‌رود).

### مراحل

1. **کد را در GitHub قرار دهید** و در [railway.app](https://railway.app) پروژهٔ جدید بسازید (Deploy from GitHub repo).
2. Railway فایل `railway.toml` / `nixpacks.toml` را می‌خواند و به‌صورت خودکار Build & Start می‌کند.
3. یک دیتابیس **Neon Postgres** اضافه کنید (یا **MySQL** از کاتالوگ Railway).
4. در تب **Variables** این متغیرها را بگذارید (رازها: XT_API_SECRET، AI_API_KEY، TELEGRAM_BOT_TOKEN):

```env
XT_API_KEY=...
XT_API_SECRET=...
AI_API_KEY=...
AI_BASE_URL=https://api.openai.com/v1
AI_MODEL=gpt-4o
TELEGRAM_BOT_TOKEN=...
TELEGRAM_USER_ID=...
DATABASE_URL=postgresql://user:pass@ep-xxx-pooler....neon.tech/neondb?sslmode=require   # مهم: دیتابیس دائمی مثل Neon (یا ${{ MySQL.MYSQL_URL }})
```

5. **Deploy** کنید. Railway به‌صورت خودکار یک دامنهٔ `*.up.railway.app` می‌دهد که ربات روی آن بالا می‌آید.

> اگر `DATABASE_URL` را ندهید، ربات به SQLite لوکال برمی‌گردد و در هر دیپلوی **داده‌ها پاک می‌شوند** (در لاگ هشدار می‌دهد).

---

## گزینه‌های هوش مصنوعی (AI Options)

این ربات با **هر سرویس سازگار با OpenAI API** کار می‌کند؛ کافی است آدرس و مدل را در `.env` یا Variables تغییر دهید:

| متغیر | مثال | توضیح |
|---|---|---|
| `AI_API_KEY` | `sk-...` | کلید API |
| `AI_BASE_URL` | `https://api.openai.com/v1` | برای OpenAI همیشه همین؛ برای دیگران (مثلاً Groq / DeepSeek / سرویس‌های سازگار) آدرسشان را بگذارید |
| `AI_MODEL` | `gpt-4o` | نام مدلِ سرویس مربوطه |

نکته‌ها:
- مدل باید **function calling / tools** را پشتیبانی کند (زیرا ربات با ابزارها تنظیمات را تغییر می‌دهد).
- با `AI_BASE_URL` و `AI_MODEL` دلخواه، ربات از OpenAI به سرویس دلخواه شما سوئیچ می‌شود.

---

## دستورات تلگرام

| دستور | کاربرد |
|---|---|
| `/start` | لیست دستورات |
| `/status` | وضعیت کامل، موقعیت‌ها، PnL، تنظیمات |
| `/pnl` | خلاصه سود/زیان |
| `/balance` | بالانس فیوچرز USDT |
| `/signal` | اجرای فوری اسکن سیگنال |
| `/history` | تاریخچه موقعیت‌ها |
| `/trades` | تاریخچه معاملات |
| `/liq` | اطلاعات مارجین‌کال/لیکوییدیشن |
| `/agent` | اجرای فوری یک تصمیم ایجنت |
| `/close [trade_id]` | بستن یک/همه موقعیت‌ها |
| `/diag` | چرا بریک‌ایون/تریلینگ عمل نکرده |
| `/settings` | نمایش تنظیمات فعلی |
| `/check_ai` | تست اتصال هوش مصنوعی |

**چت عادی** هم کار می‌کند — مثلاً بنویسید: «لوریج را ۲۰ بگذار»، «وضعیت را بگو»، «بالانس چقدر است». ایجنت خودش به‌صورت خودکار معامله می‌کند؛ چت برای سؤال و تغییر صریح تنظیمات است.

---

## تنظیمات مهم (و هشدار ریسک)

- **`leverage` پیش‌فرض ۷۵x** با `SL_LIQUIDATION_SAFETY=0.5` یعنی استاپ حداکثر در نصف فاصلهٔ تا لیکوئیدیشن گذاشته می‌شود؛ در لوریج بالا این فاصله بسیار کوچک می‌شود. **لوریج بالا = ریسک بالا.**
- `on_tpsl_failure=close` — اگر صرافی استاپ را قبول نکند، ربات موقعیت را می‌بندد (به‌جای رهاکردن بی‌استاپ).
- `position_mode` — `margin` (سایز بر پایه درصد مارجین) یا `risk` (سایز بر پایه درصد ریسک با استاپ).
- استاپ نرم‌افزاری **داینامیک** است (کسری از فاصله تا لیکوییدیشن، نه `max_loss_pct` ثابت — آن کلیدهای قدیمی خودکار از دیتابیس حذف می‌شوند).
- `cooldown_minutes` — بعد از بستن، از ورود مجدد جلوگیری می‌کند.
- **وتوی RSI + توافق** — RSI بالای ۷۰ LONG را وتو می‌کند، RSI زیر ۳۰ SHORT را؛ به‌جز RSI حداقل ۲ استراتژی باید هم‌نظر باشند (`min_agreeing_strategies`).
- **قفل تنظیمات** — حلقه خودکار ایجنت هرگز نمی‌تواند خودش لوریج/مارجین/سیمبل/تنظیمات را عوض کند (فقط از چت)؛ override در `open_trade` فقط برای همان یک معامله است و بعد restore می‌شود.

---

## رفع مشکل امضا (Signature)

اگر با وجود تنظیم صحیح کلیدها خطای معتبرسازی امضا می‌گیرید:

1. کلید باید دسترسی **فیوچرز** داشته باشد و **حساب فیوچرز باز** باشد.
2. اگر IP-whitelist فعال است، IP سرور را اضافه کنید.
3. امضای XT به شکل `#{path}#{message}` است و وقتی پیام خالی است **`#` انتهایی ندارد**. (تکرار باگ قدیمی که در PR #4 اصلاح شد — اگر کد قدیمی دارید، `git pull` بزنید.)

---

## منابع رسمی

- [مستندات API فیوچرز XT](https://doc.xt.com/docs/futures/Access%20Description/BasicInformationOfTheInterface)
- [SDK رسمی پایتون XT (pyxt)](https://github.com/kelvinxue/pyxt)
- [مستندات Railway](https://docs.railway.app)

---

**سلب مسئولیت:** استفاده از این ربات به معنای پذیرش مسئولیت کامل ریسک مالی توسط شماست. معاملات فیوچرز ممکن است به ازدست‌رفتن کل سرمایه منجر شود.
