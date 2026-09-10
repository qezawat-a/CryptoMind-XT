import hashlib
import hmac
import json
import logging
import os
import time
import requests
from requests.exceptions import RequestException

logger = logging.getLogger("xt_client")


class XTError(Exception):
    pass


class XTClient:
    """XT USDT-M futures REST client.

    Signing follows the live-verified CCXT implementation for /future v1:
    - Header keys are xt-validate-appkey / xt-validate-timestamp /
      xt-validate-signature. XT's own auth errors name these exact headers
      ("missing request header xt-validate-appkey", ...) and CCXT, the
      battle-tested client, sends exactly this prefix. (The doc.xt.com pages
      that show a "validate-*" prefix are misleading - the live server does
      not see those headers and replies "invalid signature".)
    - Signed string = "xt-validate-appkey=..&xt-validate-timestamp=.."
      + "#path" [+ "#payload"] where payload is EXACTLY what the server
      reconstructs: for POST the request BODY, for GET the query string.
    - POST bodies are JSON (like CCXT); signing the exact same string that is
      sent makes the server-side hash match. Set env XT_SIGN_BODY=form to send
      form-urlencoded instead (signature is then over the sorted key=value
      string). XT_SIGN_PREFIX can override the header prefix if XT ever
      migrates it.

    CROSS (CROSSED) margin notes:
    - XT position type value is "CROSSED" (NOT "CROSS"); "ISOLATED" is the
      other value. set_position_type() normalizes "CROSS" for you.
    - adjust_margin() and set_auto_margin() are ISOLATED-position features
      only. In CROSSED mode do NOT call them - they error or do nothing.
    - In CROSSED mode the only fix for an "insufficient margin/balance" error
      is to size the order down from availableBalance (get_balances()).
    """

    MARKET = "/future/market"
    USER = "/future/user"
    TRADE = "/future/trade"

    # The XT create-profit endpoint REQUIRES expireTime (ms). A far-future
    # value makes the TP/SL effectively permanent. Override per-call if needed.
    TP_SL_DEFAULT_EXPIRY_MS = 4102444800000  # 2100-01-01 00:00 UTC

    # Header prefix (server contract: "xt-validate"; override only if XT migrates).
    SIGN_PREFIX = os.getenv("XT_SIGN_PREFIX", "xt-validate").strip("-").lower()
    # POST body encoding: "json" (CCXT parity, default) or "form".
    SIGN_BODY = os.getenv("XT_SIGN_BODY", "json").lower()

    def __init__(self, host: str, access_key: str, secret_key: str, timeout: int = 10):
        self.host = host.rstrip("/")
        self._ak = access_key
        self._sk = secret_key
        self.timeout = timeout
        self._session = requests.Session()

    # ---------- signing (CCXT parity for /future v1) ----------

    def _dual_auth_headers(self, p: str, ts: str, sig: str) -> dict:
        """Single xt-validate-* prefix, byte-for-byte parity with the
        known-good tradekit (xt-tradekit/app/mcp/xt_tradekit/xt_futures.py)
        and CCXT. Extra/dual headers only confuse the gateway."""
        return {
            "Content-type": "application/x-www-form-urlencoded",
            f"{p}-appkey": self._ak,
            f"{p}-timestamp": ts,
            f"{p}-signature": sig,
            f"{p}-algorithms": "HmacSHA256",
            f"{p}-recvwindow": "60000",
        }

    def _signed_request_headers_and_payload(self, method: str, path: str, params: dict):
        """Return (headers, send_kwargs) for a signed request.

        The signed string is built over the exact payload that is sent, so the
        server-side recomputation matches: JSON/form body for POST, query
        string for GET.
        """
        p = self.SIGN_PREFIX
        ts = str(int(time.time() * 1000))
        msg = f"{p}-appkey={self._ak}&{p}-timestamp={ts}"

        if method == "GET":
            if params:
                param_str = "&".join(f"{k}={params[k]}" for k in sorted(params))
                msg += f"#{path}#{param_str}"
            else:
                msg += f"#{path}"
            sig = hmac.new(self._sk.encode(), msg.encode(), hashlib.sha256).hexdigest()
            headers = self._dual_auth_headers(p, ts, sig)
            return headers, {"params": params}

        # POST: sign over the exact body string that is transmitted.
        if self.SIGN_BODY == "form":
            param_str = "&".join(f"{k}={params[k]}" for k in sorted(params)) if params else ""
            body = param_str
            headers = {
                "Content-type": "application/x-www-form-urlencoded",
            }
        else:  # json (CCXT parity)
            body = json.dumps(params, separators=(",", ":")) if params else ""
            headers = {
                "Content-type": "application/json",
            }
        if body:
            msg += f"#{path}#{body}"
        else:
            msg += f"#{path}"
        sig = hmac.new(self._sk.encode(), msg.encode(), hashlib.sha256).hexdigest()
        headers = self._dual_auth_headers(p, ts, sig)
        if self.SIGN_BODY == "json":
            headers["Content-type"] = "application/json"
        return headers, {"data": body}

    # ---------- transport ----------

    def _request(self, method: str, path: str, params: dict = None, signed: bool = False):
        url = self.host + path
        params = {k: v for k, v in (params or {}).items() if v is not None}

        # Order-placing endpoints are NOT idempotent: blind-retrying after a
        # timeout/5xx can place a duplicate order. Never auto-retry these.
        no_retry = method == "POST" and (
            path.endswith("/order/create")
            or path.endswith("/order/create-batch")
            or path.endswith("/entrust/create-profit")
            or path.endswith("/entrust/create-plan")
            or path.endswith("/entrust/create-track")
        )

        max_retries = 1 if no_retry else 5
        for attempt in range(max_retries):
            try:
                if signed:
                    headers, send_kwargs = self._signed_request_headers_and_payload(
                        method, path, params)
                else:
                    headers = {"Content-type": "application/json"}
                    send_kwargs = {"params": params} if method == "GET" else {"data": params}

                if method == "GET":
                    resp = self._session.get(url, headers=headers, timeout=self.timeout,
                                             **send_kwargs)
                else:
                    resp = self._session.post(url, headers=headers, timeout=self.timeout,
                                              **send_kwargs)

                if resp.status_code == 429:
                    if no_retry:
                        raise XTError(f"{path} -> HTTP 429 rate limited (not retried, "
                                      f"order endpoint): {resp.text[:200]}")
                    retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"XT Rate Limit hit (429). Sleeping for {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                if 500 <= resp.status_code < 600:
                    if no_retry:
                        raise XTError(f"{path} -> HTTP {resp.status_code} (not retried, "
                                      f"order endpoint): {resp.text[:200]}")
                    wait_time = 2 ** attempt
                    logger.warning(f"XT Server Error ({resp.status_code}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                return self._unwrap(resp, path)

            except RequestException as e:
                if no_retry:
                    raise XTError(f"Network error on {path} (not retried, order endpoint): {e}")
                wait_time = 2 ** attempt
                logger.warning(f"Network error: {e}. Retrying in {wait_time}s...")
                if attempt == max_retries - 1:
                    raise XTError(f"Network error after {max_retries} attempts: {e}")
                time.sleep(wait_time)

        raise XTError(f"Request failed after {max_retries} attempts: {path}")

    def _public(self, method: str, path: str, params: dict = None):
        return self._request(method, path, params, signed=False)

    def _private(self, method: str, path: str, params: dict = None):
        if not self._ak or not self._sk:
            raise XTError(f"{path} -> XT_API_KEY/XT_API_SECRET is empty "
                          f"(server would reply 'Absence of [appKey]'). "
                          f"Set env or ~/.xt-tradekit/credentials.json")
        return self._request(method, path, params, signed=True)

    @staticmethod
    def _unwrap(resp, path: str):
        try:
            payload = resp.json()
        except ValueError:
            raise XTError(f"{path} -> HTTP {resp.status_code}, non-JSON body: {resp.text[:200]}")

        if not isinstance(payload, dict):
            return payload

        code = payload.get("returnCode", payload.get("rc", 0))

        if code != 0:
            err = payload.get("error") or {}
            msg = payload.get("msgInfo") or payload.get("mc") or "Unknown error"
            detail = err.get("msg") or err.get("code") or ""
            raise XTError(f"{path} -> returnCode={code} {msg} {detail}".strip())

        if "result" in payload:
            return payload["result"]
        if "data" in payload:
            return payload["data"]
        return payload

    # ---------- public market data ----------

    def get_symbol_detail(self, symbol: str) -> dict:
        return self._public("GET", f"{self.MARKET}/v1/public/symbol/detail", {"symbol": symbol})

    def get_klines(self, symbol: str, interval: str, limit: int = None,
                   start_time: int = None, end_time: int = None) -> list:
        return self._public("GET", f"{self.MARKET}/v1/public/q/kline", {
            "symbol": symbol, "interval": interval, "limit": limit,
            "startTime": start_time, "endTime": end_time,
        }) or []

    def get_agg_ticker(self, symbol: str) -> dict:
        return self._public("GET", f"{self.MARKET}/v1/public/q/agg-ticker", {"symbol": symbol}) or {}

    def get_mark_price(self, symbol: str) -> dict:
        return self._public("GET", f"{self.MARKET}/v1/public/q/symbol-mark-price",
                            {"symbol": symbol}) or {}

    def get_leverage_brackets(self, symbol: str) -> list:
        data = self._public("GET", f"{self.MARKET}/v1/public/leverage/bracket/detail",
                            {"symbol": symbol}) or {}
        return data.get("leverageBrackets") or []

    def get_funding_rate(self, symbol: str) -> dict:
        return self._public("GET", f"{self.MARKET}/v1/public/q/funding-rate",
                            {"symbol": symbol}) or {}

    # ---------- account ----------

    def get_balances(self) -> list:
        # In CROSSED mode size orders from "availableBalance" of the quote coin.
        # Primary: /balance/list. Age fail dad (mesle Railway ke baghie kar
        # mikone vali hamin yeki fail mide), fallback be single-coin va compat.
        try:
            data = self._private("GET", f"{self.USER}/v1/balance/list")
            if isinstance(data, list) and data:
                return data
            if isinstance(data, list):
                return data
        except XTError as e:
            first_err = e
        else:
            first_err = None
        # Fallback 1: single coin USDT (doc: Get User's Single Currency Fund Information)
        try:
            single = self._private("GET", f"{self.USER}/v1/compat/balance/usdt")
            norm = self._normalize_balance_payload(single)
            if norm:
                logger.info("get_balances: primary failed, used compat single-coin fallback")
                return norm
        except XTError:
            pass
        # Fallback 2: compat list (Contract Account Assets)
        try:
            compat = self._private("GET", f"{self.USER}/v1/compat/balance/list")
            norm = self._normalize_balance_payload(compat)
            if norm:
                logger.info("get_balances: primary failed, used compat list fallback")
                return norm
        except XTError:
            pass
        if first_err is not None:
            raise first_err
        return []

    @staticmethod
    def _normalize_balance_payload(data) -> list:
        """Compat/single-coin formats ro be shakle balance/list normal kon:
        [{coin, walletBalance, availableBalance, openOrderMarginFrozen, ...}]"""
        if isinstance(data, dict):
            # single-coin mamulan dict e: {coin, walletBalance/totalAmount, available...}
            data = [data]
        if not isinstance(data, list):
            return []
        out = []
        for row in data:
            if not isinstance(row, dict):
                continue
            coin = str(row.get("coin") or row.get("currency") or "USDT").upper()
            wallet = (row.get("walletBalance") or row.get("totalAmount")
                      or row.get("total") or row.get("balance") or 0)
            avail = (row.get("availableBalance") or row.get("available")
                     or row.get("availableAmount") or 0)
            frozen = (row.get("openOrderMarginFrozen") or row.get("frozen")
                      or row.get("frozenAmount") or 0)
            try:
                out.append({
                    "coin": coin,
                    "walletBalance": float(wallet or 0),
                    "availableBalance": float(avail or 0),
                    "openOrderMarginFrozen": float(frozen or 0),
                    "isolatedMargin": float(row.get("isolatedMargin") or 0),
                    "crossedMargin": float(row.get("crossedMargin") or 0),
                    "bonus": float(row.get("bonus") or 0),
                    "coupon": float(row.get("coupon") or 0),
                })
            except (TypeError, ValueError):
                continue
        # Age hame USDT nabud, harchi hast bargardun; risk_manager khodesh USDT ro peyda mikone
        return out

    def get_contract_account_assets(self, query_account_id: str = None) -> list:
        """GET /future/user/v1/compat/balance/list - Get Contract Account Assets (has userId, accountId, positionType, totalAmount, bonus, coupon, etc per doc)."""
        params = {}
        if query_account_id:
            params["queryAccountId"] = query_account_id
        data = self._private("GET", f"{self.USER}/v1/compat/balance/list", params)
        return data if isinstance(data, list) else []

    def get_account_info(self) -> dict:
        """GET /future/user/v1/account/info - Get Account Related Information (accountId, userId, allowTrade, allowOpenPosition, etc per doc)."""
        return self._private("GET", f"{self.USER}/v1/account/info") or {}

    def get_listen_key(self) -> str:
        # Valid for 8 hours (server-side) - re-request before expiry if you use
        # the user-data WebSocket stream.
        data = self._private("GET", f"{self.USER}/v1/user/listen-key")
        if isinstance(data, dict):
            return data.get("listenKey") or data.get("accessToken") or ""
        return data or ""

    # ---------- positions ----------

    def get_positions(self, symbol: str = None) -> list:
        # Use ACTIVE endpoint per doc: /future/user/v1/position (has floatingPL, profitId, calMarkPrice)
        # NOT /position/list (limited 12 fields, no PnL) - see doc Get active position information
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = self._private("GET", f"{self.USER}/v1/position", params)
        return data if isinstance(data, list) else []

    def get_positions_list(self, symbol: str = None) -> list:
        """Legacy list endpoint per doc Get Position Information (12 fields, no floatingPL)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = self._private("GET", f"{self.USER}/v1/position/list", params)
        return data if isinstance(data, list) else []

    def set_leverage(self, symbol: str, position_side: str, leverage: int):
        return self._private("POST", f"{self.USER}/v1/position/adjust-leverage", {
            "symbol": symbol, "positionSide": position_side, "leverage": leverage,
        })

    def set_position_type(self, symbol: str, position_side: str, position_type: str):
        # XT API values are "CROSSED" and "ISOLATED" (NOT "CROSS"). "CROSS" is
        # accepted and normalized so old callers keep working.
        pt = str(position_type).upper()
        if pt == "CROSS":
            pt = "CROSSED"
        if pt not in ("CROSSED", "ISOLATED"):
            raise XTError(f"set_position_type: invalid positionType {position_type!r} "
                          f"(expected CROSSED or ISOLATED)")
        return self._private("POST", f"{self.USER}/v1/position/change-type", {
            "symbol": symbol, "positionSide": position_side, "positionType": pt,
        })

    def adjust_margin(self, symbol: str, position_side: str, margin, direction: str):
        # ISOLATED-only feature. In CROSSED mode you cannot add/reduce margin on
        # a single position - do NOT call this when trading CROSSED.
        direction = str(direction).upper()
        if direction not in ("ADD", "SUB"):
            raise XTError(f"adjust_margin: direction must be ADD or SUB, got {direction!r}")
        return self._private("POST", f"{self.USER}/v1/position/margin", {
            "symbol": symbol, "positionSide": position_side,
            "margin": margin, "type": direction,
        })

    def set_auto_margin(self, symbol: str, position_side: str, enabled: bool):
        # ISOLATED-only feature. "Auto margin" does NOT exist in CROSSED mode
        # (the whole available balance already backs the position) - do NOT
        # call this when trading CROSSED; it errors or is a no-op.
        return self._private("POST", f"{self.USER}/v1/position/auto-margin", {
            "symbol": symbol, "positionSide": position_side, "autoMargin": bool(enabled),
        })

    def get_leverage_info(self, symbol: str) -> list:
        data = self._private("GET", f"{self.TRADE}/v1/position/leverage/list", {"symbol": symbol})
        if isinstance(data, dict):
            return data.get("items") or []
        return data if isinstance(data, list) else []

    # ---------- orders ----------

    def create_order(self, symbol: str, position_side: str, order_side: str,
                     order_type: str, orig_qty: int, price=None,
                     time_in_force: str = None, client_order_id: str = None,
                     reduce_only: bool = None):
        params = {
            "symbol": symbol, "positionSide": position_side, "orderSide": order_side,
            "orderType": order_type, "origQty": int(orig_qty),
        }
        if price is not None:
            params["price"] = price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if client_order_id is not None:
            params["clientOrderId"] = client_order_id
        # In CROSSED + low margin, close/reduce with reduce_only=True so the
        # order can never open a new position.
        if reduce_only is not None:
            params["reduceOnly"] = bool(reduce_only)
        return self._private("POST", f"{self.TRADE}/v1/order/create", params)

    def cancel_order(self, order_id):
        return self._private("POST", f"{self.TRADE}/v1/order/cancel", {"orderId": order_id})

    def cancel_all_orders(self, symbol: str):
        return self._private("POST", f"{self.TRADE}/v1/order/cancel-all", {"symbol": symbol})

    def get_order(self, order_id):
        return self._private("GET", f"{self.TRADE}/v1/order/detail", {"orderId": order_id})

    def get_orders(self, symbol: str = None, state: str = "NEW", page: int = 1, size: int = 50):
        params = {"state": state, "page": page, "size": size}
        if symbol:
            params["symbol"] = symbol
        data = self._private("GET", f"{self.TRADE}/v1/order-entrust/list", params) or {}
        return data.get("items") or []

    # ---------- take profit / stop loss ----------

    def create_tpsl(self, symbol: str, position_side: str, orig_qty: int,
                    trigger_profit_price, trigger_stop_price, expire_time_ms: int = None,
                    profit_order_type: str = "MARKET", stop_order_type: str = "MARKET",
                    profit_tif: str = "IOC", stop_tif: str = "IOC",
                    profit_price=None, stop_price=None):
        # expireTime is REQUIRED by the XT API for this endpoint. When omitted
        # it defaults to year 2100 -> effectively NO time limit.
        if expire_time_ms is None:
            expire_time_ms = self.TP_SL_DEFAULT_EXPIRY_MS
        params = {
            "symbol": symbol,
            "positionSide": position_side,
            "origQty": int(orig_qty),
            "triggerProfitPrice": trigger_profit_price,
            "triggerStopPrice": trigger_stop_price,
            "expireTime": int(expire_time_ms),
            "profitDelegateOrderType": profit_order_type,
            "profitDelegateTimeInForce": profit_tif,
            "stopDelegateOrderType": stop_order_type,
            "stopDelegateTimeInForce": stop_tif,
        }
        if profit_price is not None:
            params["profitDelegatePrice"] = profit_price
        if stop_price is not None:
            params["stopDelegatePrice"] = stop_price
        return self._private("POST", f"{self.TRADE}/v1/entrust/create-profit", params)

    def update_tpsl(self, profit_id, trigger_profit_price=None, trigger_stop_price=None):
        params = {"profitId": profit_id}
        if trigger_profit_price is not None:
            params["triggerProfitPrice"] = trigger_profit_price
        if trigger_stop_price is not None:
            params["triggerStopPrice"] = trigger_stop_price
        return self._private("POST", f"{self.TRADE}/v1/entrust/update-profit-stop", params)

    def cancel_tpsl(self, profit_id):
        return self._private("POST", f"{self.TRADE}/v1/entrust/cancel-profit-stop",
                             {"profitId": profit_id})

    def cancel_all_tpsl(self, symbol: str):
        return self._private("POST", f"{self.TRADE}/v1/entrust/cancel-all-profit-stop",
                             {"symbol": symbol})

    def get_tpsl_orders(self, symbol: str, state: str = "NOT_TRIGGERED",
                         page: int = 1, size: int = 50) -> list:
        params = {"state": state, "page": page, "size": size}
        if symbol:
            params["symbol"] = symbol
        data = self._private("GET", f"{self.TRADE}/v1/entrust/profit-list", params) or {}
        return data.get("items") or []

    # ---------- historical / new endpoints (doc 2025-12-05) ----------

    def get_tpsl_history(self, symbol: str = None, page: int = 1, size: int = 10,
                         start_time: int = None, end_time: int = None) -> dict:
        """GET /future/trade/v1/entrust/profit-list-history - Historical TP/SL orders."""
        params = {"page": page, "size": size}
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._private("GET", f"{self.TRADE}/v1/entrust/profit-list-history", params) or {}

    def get_active_positions_new(self, symbol: str = None) -> list:
        """GET /future/trade/v1/position/list/active - New active positions endpoint (2025-12-05)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = self._private("GET", f"{self.TRADE}/v1/position/list/active", params)
        if isinstance(data, dict):
            return data.get("items") or data.get("list") or []
        return data if isinstance(data, list) else []

    def get_position_history(self, symbol: str = None, page: int = 1, size: int = 10) -> dict:
        """GET /future/trade/v1/position/list-history - Historical positions."""
        params = {"page": page, "size": size}
        if symbol:
            params["symbol"] = symbol
        return self._private("GET", f"{self.TRADE}/v1/position/list-history", params) or {}

    def get_cross_margin(self, symbol: str) -> dict:
        """GET /future/trade/v1/position/cross-margin/{symbol} - Cross margin maintenance."""
        return self._private("GET", f"{self.TRADE}/v1/position/cross-margin/{symbol}") or {}

    def get_reverse_plan_orders(self, symbol: str = None, page: int = 1, size: int = 10) -> dict:
        """GET /future/trade/v1/entrust/reverse-plan-list - Get Reverse Plan Orders."""
        params = {"page": page, "size": size}
        if symbol:
            params["symbol"] = symbol
        return self._private("GET", f"{self.TRADE}/v1/entrust/reverse-plan-list", params) or {}

    def get_reverse_plan_history(self, symbol: str = None, page: int = 1, size: int = 10) -> dict:
        """GET /future/trade/v1/entrust/reverse-plan-list-history - Historical Reverse Plan Orders."""
        params = {"page": page, "size": size}
        if symbol:
            params["symbol"] = symbol
        return self._private("GET", f"{self.TRADE}/v1/entrust/reverse-plan-list-history", params) or {}

    def get_order_trade_history(self, symbol: str = None, page: int = 1, size: int = 10) -> dict:
        """GET /future/trade/v1/order/trade-history - Historical Orders with Trades."""
        params = {"page": page, "size": size}
        if symbol:
            params["symbol"] = symbol
        return self._private("GET", f"{self.TRADE}/v1/order/trade-history", params) or {}

    def get_all_trades(self, symbol: str = None) -> list:
        """GET /future/trade/v1/order/trade-list-all - All Trade Details (No Pagination)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = self._private("GET", f"{self.TRADE}/v1/order/trade-list-all", params)
        return data if isinstance(data, list) else []

    def get_single_coin_balance(self, coin: str) -> dict:
        """GET /future/user/v1/compat/balance/{coin} - Single Coin Balance."""
        return self._private("GET", f"{self.USER}/v1/compat/balance/{coin}") or {}

    def get_auto_deleverage_history(self, symbol: str = None, page: int = 1, size: int = 10,
                                    start_time: int = None, end_time: int = None) -> dict:
        """GET /future/user/v1/auto-deleverage/history - Auto Deleverage History."""
        params = {"page": page, "size": size}
        if symbol:
            params["symbol"] = symbol
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._private("GET", f"{self.USER}/v1/auto-deleverage/history", params) or {}

    def get_step_rate(self) -> dict:
        """GET /future/user/v1/step-rate - Get User's Step Rate (makerFee, takerFee, etc)."""
        # Doc shows both /v1/step-rate and /v1/user/step-rate, try primary
        try:
            return self._private("GET", f"{self.USER}/v1/step-rate") or {}
        except XTError:
            return self._private("GET", f"{self.USER}/v1/user/step-rate") or {}

    def get_margin_call_info(self, symbol: str = None) -> list:
        """GET /future/user/v1/position/break-list - Get Margin Call Information (breakPrice, calMarkPrice, liq price)."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = self._private("GET", f"{self.USER}/v1/position/break-list", params)
        return data if isinstance(data, list) else []
