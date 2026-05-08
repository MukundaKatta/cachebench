from __future__ import annotations

import enum
import functools
import hashlib
import inspect
import json
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class Provider(str, enum.Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    BEDROCK = "bedrock"


# Per-million-token USD prices. Override via CacheTracker(pricing=...).
# Keep current; vendors change pricing without notice.
DEFAULT_PRICING: dict[Provider, dict[str, float]] = {
    Provider.ANTHROPIC: {
        "input": 3.00,
        "cache_read": 0.30,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "output": 15.00,
    },
    Provider.OPENAI: {
        "input": 2.50,
        "cache_read": 1.25,
        "cache_write_5m": 2.50,
        "cache_write_1h": 2.50,
        "output": 10.00,
    },
    Provider.BEDROCK: {
        "input": 3.00,
        "cache_read": 0.30,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "output": 15.00,
    },
}


@dataclass(frozen=True)
class CallMetrics:
    provider: Provider
    prefix_id: str
    input_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    output_tokens: int
    elapsed_ms: float
    timestamp: float

    @property
    def hit_ratio(self) -> float:
        cacheable = self.cache_read_tokens + self.cache_creation_tokens
        if cacheable == 0:
            return float("nan")
        return self.cache_read_tokens / cacheable

    def cost_usd(self, pricing: dict[str, float]) -> float:
        return (
            self.input_tokens * pricing["input"]
            + self.cache_read_tokens * pricing["cache_read"]
            + self.cache_creation_tokens * pricing["cache_write_5m"]
            + self.output_tokens * pricing["output"]
        ) / 1_000_000

    def cost_saved_usd(self, pricing: dict[str, float]) -> float:
        cacheable_total = self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens
        full = cacheable_total * pricing["input"]
        actual = (
            self.input_tokens * pricing["input"]
            + self.cache_read_tokens * pricing["cache_read"]
            + self.cache_creation_tokens * pricing["cache_write_5m"]
        )
        return (full - actual) / 1_000_000


@dataclass
class CachePolicy:
    """Policy controlling miss-aware retry behavior."""

    retry_on_miss: bool = False
    delay_ms: int = 2000
    max_retries: int = 1

    @classmethod
    def miss_aware(
        cls, retry_on_miss: bool = True, delay_ms: int = 2000, max_retries: int = 1
    ) -> CachePolicy:
        return cls(retry_on_miss=retry_on_miss, delay_ms=delay_ms, max_retries=max_retries)


def fingerprint(
    messages: Optional[list] = None,
    system: Any = None,
    tools: Any = None,
    model: Optional[str] = None,
) -> str:
    """Stable hash of the cacheable prefix.

    Excludes the trailing user turn (which fragments the prefix space) and
    the model name (since Anthropic caches per-model). Pass `model=` to bind
    the fingerprint to a specific model.
    """
    payload = {
        "system": system,
        "tools": tools,
        "model": model,
        "prefix_messages": (messages or [])[:-1],
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract usage fields from a provider response in a vendor-agnostic way."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage", {})
    if usage is None:
        return {}

    def g(name: str) -> int:
        v = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    out = {
        "input_tokens": g("input_tokens") or g("prompt_tokens"),
        "cache_read_tokens": g("cache_read_input_tokens"),
        "cache_creation_tokens": g("cache_creation_input_tokens"),
        "output_tokens": g("output_tokens") or g("completion_tokens"),
    }

    # OpenAI nests cached tokens inside prompt_tokens_details
    if out["cache_read_tokens"] == 0:
        details = (
            usage.get("prompt_tokens_details")
            if isinstance(usage, dict)
            else getattr(usage, "prompt_tokens_details", None)
        )
        if details is not None:
            cached = (
                details.get("cached_tokens")
                if isinstance(details, dict)
                else getattr(details, "cached_tokens", 0)
            )
            if cached:
                out["cache_read_tokens"] = int(cached)
                # OpenAI's prompt_tokens already includes cached; subtract.
                out["input_tokens"] = max(0, out["input_tokens"] - int(cached))

    return out


class CacheTracker:
    """Wrap an LLM client call and record per-request prompt-cache metrics.

    Usage:
        tracker = CacheTracker(provider=Provider.ANTHROPIC)
        create = tracker.wrap(client.messages.create)
        response = create(model="claude-sonnet-4-20250514", ...)
        print(tracker.aggregate())
    """

    def __init__(
        self,
        provider: Provider = Provider.ANTHROPIC,
        pricing: Optional[dict[str, float]] = None,
        miss_alert_threshold: float = 0.6,
        on_miss_alert: Optional[Callable[[CallMetrics], None]] = None,
        policy: Optional[CachePolicy] = None,
        history_size: int = 10_000,
    ):
        self.provider = provider
        self.pricing = pricing or DEFAULT_PRICING[provider]
        self.miss_alert_threshold = miss_alert_threshold
        self.on_miss_alert = on_miss_alert or self._default_alert
        self.policy = policy or CachePolicy()
        self._calls: deque[CallMetrics] = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def _default_alert(self, m: CallMetrics) -> None:
        sys.stderr.write(
            f"[cachebench] {self.provider.value} hit_ratio={m.hit_ratio:.2f} "
            f"prefix={m.prefix_id} below {self.miss_alert_threshold:.2f}\n"
        )

    def wrap(self, fn: Callable) -> Callable:
        """Wrap a client method (sync or async) to record cache metrics."""
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def ainner(*args, **kwargs):
                return await self._do_call_async(fn, args, kwargs)

            return ainner

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            return self._do_call(fn, args, kwargs)

        return inner

    def _do_call(self, fn: Callable, args: tuple, kwargs: dict) -> Any:
        for attempt in range(self.policy.max_retries + 1):
            t0 = time.perf_counter()
            response = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            m = self._record(response, kwargs, elapsed_ms)
            if not self._should_retry(m, attempt):
                return response
            time.sleep(self.policy.delay_ms / 1000)
        return response  # noqa: F821 — last response if loop exits normally

    async def _do_call_async(self, fn: Callable, args: tuple, kwargs: dict) -> Any:
        import asyncio

        for attempt in range(self.policy.max_retries + 1):
            t0 = time.perf_counter()
            response = await fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            m = self._record(response, kwargs, elapsed_ms)
            if not self._should_retry(m, attempt):
                return response
            await asyncio.sleep(self.policy.delay_ms / 1000)
        return response

    def _should_retry(self, m: CallMetrics, attempt: int) -> bool:
        if not self.policy.retry_on_miss:
            return False
        if attempt >= self.policy.max_retries:
            return False
        # Retry only on a clear silent miss: cacheable prefix existed but ratio = 0.
        cacheable = m.cache_read_tokens + m.cache_creation_tokens
        return cacheable > 0 and m.hit_ratio == 0.0

    def _record(self, response: Any, kwargs: dict, elapsed_ms: float) -> CallMetrics:
        u = _extract_usage(response)
        m = CallMetrics(
            provider=self.provider,
            prefix_id=fingerprint(
                messages=kwargs.get("messages", []),
                system=kwargs.get("system"),
                tools=kwargs.get("tools"),
                model=kwargs.get("model"),
            ),
            input_tokens=u.get("input_tokens", 0),
            cache_read_tokens=u.get("cache_read_tokens", 0),
            cache_creation_tokens=u.get("cache_creation_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            elapsed_ms=elapsed_ms,
            timestamp=time.time(),
        )
        with self._lock:
            self._calls.append(m)

        cacheable = m.cache_read_tokens + m.cache_creation_tokens
        if cacheable > 0 and m.hit_ratio < self.miss_alert_threshold:
            try:
                self.on_miss_alert(m)
            except Exception:
                pass
        return m

    # --- Read APIs ----------------------------------------------------------

    def calls(self) -> list[CallMetrics]:
        with self._lock:
            return list(self._calls)

    def aggregate(self) -> dict:
        calls = self.calls()
        if not calls:
            return {"calls": 0}
        read = sum(c.cache_read_tokens for c in calls)
        write = sum(c.cache_creation_tokens for c in calls)
        cacheable = read + write
        return {
            "calls": len(calls),
            "hit_ratio": (read / cacheable) if cacheable else float("nan"),
            "tokens_read_from_cache": read,
            "tokens_written_to_cache": write,
            "cost_usd": sum(c.cost_usd(self.pricing) for c in calls),
            "cost_saved_usd": sum(c.cost_saved_usd(self.pricing) for c in calls),
        }

    def by_prefix(self) -> dict[str, dict]:
        groups: dict[str, list[CallMetrics]] = defaultdict(list)
        for c in self.calls():
            groups[c.prefix_id].append(c)
        out = {}
        for prefix_id, ms in groups.items():
            cacheable = sum(c.cache_read_tokens + c.cache_creation_tokens for c in ms)
            read = sum(c.cache_read_tokens for c in ms)
            out[prefix_id] = {
                "calls": len(ms),
                "hit_ratio": (read / cacheable) if cacheable else float("nan"),
                "cost_saved_usd": sum(c.cost_saved_usd(self.pricing) for c in ms),
            }
        return out

    def window(self, seconds: float) -> list[CallMetrics]:
        cutoff = time.time() - seconds
        return [c for c in self.calls() if c.timestamp >= cutoff]

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()
