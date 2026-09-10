"""
Brain - LLM provider with dual support: OpenAI-compatible + Anthropic native.

- Custom base_url for both (AI_BASE_URL, ANTHROPIC_BASE_URL)
- Auto model detection from /models when AI_MODEL is empty
- Auto fallback on 402 / insufficient_quota / balance errors -> next model
- Anthropic native via anthropic SDK when ANTHROPIC_API_KEY is set
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("agent_brain")

FALLBACK_MODEL_OPENAI = "gpt-4o-mini"
FALLBACK_MODEL_ANTHROPIC = "claude-3-5-haiku-20241022"

PREFERRED_FAMILIES = (
    "gpt-4o", "gpt-4.1", "gpt-4", "o1", "o3", "claude",
    "llama-3.1", "llama-3", "llama-4", "mistral", "mixtral",
    "gemma", "deepseek-chat", "deepseek", "qwen", "gemini", "yi-",
)

NON_CHAT_MARKERS = (
    "embed", "whisper", "tts", "audio", "dall", "image",
    "moderation", "rerank", "re-rank", "realtime", "omni",
    "ft:", "fine-tune",
)

CHEAP_MARKERS = (
    "flash", "mini", "lite", "nano", "small", "distil",
    "-8b", "-7b", "-4b", "-3b", "-2b", "-1b", "-0.5b",
)


class Brain:
    def __init__(self):
        from config import Config
        self.Config = Config
        self.provider = Config.get_effective_provider()  # openai or anthropic
        self.api_key = Config.get_effective_api_key()
        # Support 50 keys rotation: AI_API_KEYS=key1,key2,... plus AI_API_KEY
        raw_keys = (Config.AI_API_KEYS or "") + "," + (Config.AI_API_KEY or "")
        self._keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        # Deduplicate preserve order
        seen=set()
        uniq=[]
        for k in self._keys:
            if k not in seen:
                seen.add(k)
                uniq.append(k)
        self._keys = uniq or [self.api_key or "sk-local-dummy"]
        self._key_index = 0
        if self._keys:
            self.api_key = self._keys[0]
        self.base_url = Config.get_effective_base_url()
        self.explicit_model = Config.get_effective_model()
        self.fallback_models_str = Config.AI_FALLBACK_MODELS

        self.model: str = ""
        self.model_source: str = "auto"
        self._candidates: List[str] = []
        self._candidate_index: int = 0
        self._client = None
        self._anthropic_client = None

        self._init_client()
        self.model = self._resolve_model()

    def _init_client(self):
        if self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                # For local allow dummy key
                akey = self.api_key or "sk-local-dummy"
                self._anthropic_client = Anthropic(
                    api_key=akey,
                    base_url=self.base_url if self.base_url != "https://api.anthropic.com" else None,
                )
                logger.info(f"Brain init: anthropic base_url={self.base_url}")
            except ImportError:
                logger.warning("anthropic package not installed, falling back to openai-compatible client")
                self.provider = "openai"
                self._init_openai_client()
        else:
            self._init_openai_client()

    def _init_openai_client(self):
        from openai import OpenAI
        # For local LLM (localhost/trycloudflare) allow empty key - use dummy so SDK doesn't throw Missing credentials
        api_key = self.api_key or "sk-local-dummy"
        self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        logger.info(f"Brain init: openai-compatible base_url={self.base_url} with {len(self._keys)} keys rotation")

    # ---- model resolution ----

    def _resolve_model(self) -> str:
        if self.explicit_model:
            self._candidates = [self.explicit_model]
            self._candidate_index = 0
            self.model_source = "explicit"
            logger.info(f"Brain model explicit: {self.explicit_model}")
            return self.explicit_model

        # Check fallback env list
        if self.fallback_models_str:
            fallback_list = [m.strip() for m in self.fallback_models_str.split(",") if m.strip()]
            if fallback_list:
                self._candidates = fallback_list
                self._candidate_index = 0
                self.model_source = "fallback_env"
                logger.info(f"Brain model fallback_env: {fallback_list}")
                return fallback_list[0]

        # Auto detect
        ids = self._list_models()
        if not ids:
            fallback = FALLBACK_MODEL_ANTHROPIC if self.provider == "anthropic" else FALLBACK_MODEL_OPENAI
            logger.warning(f"No models from /models at {self.base_url}, fallback to {fallback}")
            self._candidates = [fallback]
            self.model_source = "hard_fallback"
            return fallback

        ranked = self._rank_models(ids)
        # Probe for working model
        probed = self._probe_models(ranked)
        if probed:
            self._candidates = ranked
            self._candidate_index = ranked.index(probed)
            self.model_source = "auto"
            logger.info(f"Brain auto-selected: {probed} from {len(ranked)} candidates")
            return probed
        else:
            logger.warning(f"Auto-probe failed for all {len(ranked)} candidates, using top {ranked[0]}")
            self._candidates = ranked
            self._candidate_index = 0
            self.model_source = "auto_unverified"
            return ranked[0]

    def _list_models(self) -> List[str]:
        if self.provider == "anthropic":
            # Anthropic doesn't have /models; try OpenAI-compatible /models if base_url is proxy
            # Many proxies (openrouter, litellm) expose /models even for anthropic
            try:
                from openai import OpenAI
                tmp = OpenAI(api_key=self.api_key, base_url=self.base_url)
                page = tmp.models.list()
                data = getattr(page, "data", None) or []
                ids = [getattr(m, "id", None) for m in data if getattr(m, "id", None)]
                if ids:
                    return ids
            except Exception as e:
                logger.warning(f"Anthropic model list via openai client failed: {e}")
            # Hardcoded anthropic models as fallback
            return [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307",
            ]
        else:
            try:
                page = self._client.models.list()
            except Exception as e:
                logger.warning(f"Could not list models from {self.base_url}/models: {e}")
                return []
            data = getattr(page, "data", None)
            if not data:
                return []
            return [getattr(m, "id", None) for m in data if getattr(m, "id", None)]

    @classmethod
    def _rank_models(cls, ids: List[str]) -> List[str]:
        chat_models = [i for i in ids if not any(m in i.lower() for m in NON_CHAT_MARKERS)]
        pool = chat_models or ids
        pool_lower = [i.lower() for i in pool]

        def by_family(models: List[str]) -> List[str]:
            def key(mid: str):
                low = mid.lower()
                for idx, fam in enumerate(PREFERRED_FAMILIES):
                    if fam in low:
                        return idx
                return len(PREFERRED_FAMILIES)
            return sorted(models, key=key)

        free = [i for i, low in zip(pool, pool_lower) if low.endswith(":free")]
        cheap = [i for i, low in zip(pool, pool_lower) if not low.endswith(":free") and any(m in low for m in CHEAP_MARKERS)]
        rest = [i for i, low in zip(pool, pool_lower) if not low.endswith(":free") and not any(m in low for m in CHEAP_MARKERS)]
        return by_family(free) + by_family(cheap) + by_family(rest)

    def _probe_models(self, candidates: List[str]) -> str:
        # Only probe for openai-compatible; anthropic probe is done via real call shape
        for mid in candidates:
            try:
                if self.provider == "anthropic":
                    # Tiny anthropic call
                    resp = self._anthropic_client.messages.create(
                        model=mid,
                        max_tokens=10,
                        messages=[{"role": "user", "content": "ping"}],
                    )
                    if resp and resp.content:
                        return mid
                else:
                    # OpenAI tool probe
                    resp = self._client.chat.completions.create(
                        model=mid,
                        messages=[{"role": "user", "content": "ping - respond with tool get_status"}],
                        tools=[{"type": "function", "function": {"name": "get_status", "description": "test", "parameters": {"type": "object", "properties": {}}}}],
                        tool_choice="auto",
                        max_tokens=100,
                        timeout=15,
                    )
                    # Any successful response counts; we don't strictly require tool_calls here for fallback probe
                    # But prefer tool_calls
                    if resp and resp.choices:
                        return mid
            except Exception as e:
                logger.warning(f"Probe {mid} failed: {str(e).splitlines()[0][:120]}")
                continue
        return ""

    @staticmethod
    def _is_balance_error(err_str: str) -> bool:
        low = err_str.lower()
        return (
            "402" in err_str
            or "insufficient_quota" in low
            or "insufficient_balance" in low
            or "balance is positive" in low
            or "not enough" in low
            or "payment required" in low
            or "exceeded your current quota" in low
            or "billing" in low
            or "freeusagelimit" in low
            or "free_usage" in low
            or "rate_limit_exceeded" in low
            or "429" in err_str and "rate_limit" in low
        )

    def _next_candidate(self) -> str:
        if self.model_source == "explicit":
            return ""
        self._candidate_index += 1
        if 0 <= self._candidate_index < len(self._candidates):
            return self._candidates[self._candidate_index]
        return ""

    def _next_key(self) -> str:
        # Rotate to next API key when current key hits 429/limit
        if len(self._keys) <= 1:
            return ""
        self._key_index += 1
        if 0 <= self._key_index < len(self._keys):
            self.api_key = self._keys[self._key_index]
            # Reinit client with new key
            try:
                self._init_client()
            except:
                pass
            logger.warning(f"Brain rotating to next API key #{self._key_index+1}/{len(self._keys)}")
            return self.api_key
        return ""

    def get_model_info(self) -> str:
        return f"{self.model} ({self.model_source} via {self.provider} @ {self.base_url})"

    # ---- completion ----

    def chat(self, messages: List[Dict], tools: List[Dict] = None, tool_choice: str = "auto") -> Tuple[str, List[Dict], str]:
        """
        Returns (content, tool_calls, raw_model_used)
        tool_calls: list of {"id": str, "name": str, "arguments": dict}
        Handles auto fallback on balance errors.
        """
        # openai-compatible and anthropic both support similar tool schemas but anthropic uses different SDK
        # For simplicity, we normalize tools to both
        max_rounds = max(5, len(self._candidates)) if self.model_source != "explicit" else 5
        last_err = ""
        for _ in range(max_rounds):
            try:
                if self.provider == "anthropic":
                    return self._chat_anthropic(messages, tools, tool_choice)
                else:
                    return self._chat_openai(messages, tools, tool_choice)
            except Exception as e:
                err_str = str(e)
                last_err = err_str
                if self._is_balance_error(err_str):
                    # Try next model first, then next key
                    nxt = self._next_candidate()
                    if nxt:
                        logger.warning(f"Model {self.model} balance error ({err_str[:100]}), switching to {nxt}")
                        self.model = nxt
                        continue
                    nxt_key = self._next_key()
                    if nxt_key:
                        # Reset candidates for new key
                        self._candidate_index = -1
                        nxt2 = self._next_candidate()
                        if nxt2:
                            self.model = nxt2
                            continue
                        continue
                # Also try next candidate on model not found
                if "model_not_found" in err_str.lower() or "does not exist" in err_str.lower():
                    nxt = self._next_candidate()
                    if nxt:
                        logger.warning(f"Model {self.model} not found, switching to {nxt}")
                        self.model = nxt
                        continue
                    nxt_key = self._next_key()
                    if nxt_key:
                        continue
                raise
        raise RuntimeError(f"All models failed, last error: {last_err}")

    def _chat_openai(self, messages: List[Dict], tools: List[Dict] = None, tool_choice: str = "auto"):
        kwargs = dict(model=self.model, messages=messages, timeout=45)
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["tool_choice"] = tool_choice
        # token limit: reasoning models need max_completion_tokens
        low = self.model.lower()
        if any(x in low for x in ("o1", "o3", "grok-4", "gpt-5")):
            kwargs["max_completion_tokens"] = 4096
        else:
            kwargs["max_tokens"] = 4096

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except:
                        args = {"_raw": args}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return content, tool_calls, self.model

    def _chat_anthropic(self, messages: List[Dict], tools: List[Dict] = None, tool_choice: str = "auto"):
        # Convert OpenAI messages/tools to Anthropic format
        # Anthropic: system as separate, tools as tools param
        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        system = "\n\n".join(system_msgs) if system_msgs else None
        anth_msgs = []
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "tool":
                # tool results become user messages with tool_result
                # Anthropic expects tool_use_id
                anth_msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", "tool"), "content": m.get("content", "")}]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                # assistant tool_calls -> anthropic tool_use blocks
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]})
                anth_msgs.append({"role": "assistant", "content": blocks})
            else:
                anth_msgs.append({"role": m["role"], "content": m["content"]})

        # Convert tools
        anth_tools = None
        if tools:
            anth_tools = []
            for t in tools:
                anth_tools.append({
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}})
                })

        kwargs = dict(model=self.model, messages=anth_msgs, max_tokens=4096)
        if system:
            kwargs["system"] = system
        if anth_tools:
            kwargs["tools"] = anth_tools

        resp = self._anthropic_client.messages.create(**kwargs)
        # Parse response
        content = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})
        return content, tool_calls, self.model

    def simple_chat(self, prompt: str) -> str:
        content, _, _ = self.chat([{"role": "user", "content": prompt}])
        return content
