# cachebench

Prompt-cache observability for LLM APIs. Per-call hit ratios, cost saved, regression alerts, miss-aware retry. Anthropic, OpenAI, and AWS Bedrock.

```bash
pip install cachebench
```

## Why

Prompt caching saves 50–90% of input tokens on Anthropic and OpenAI, but per-request hit rate is invisible from the SDK. Misses are silent. A single deploy that appends a timestamp to a system prompt can quietly halve your cache hit rate and double your bill — and you'll find out from the invoice. Anthropic's SDK [silently misses ~40% on back-to-back requests](https://github.com/anthropics/anthropic-sdk-python/issues/1451) at certain windows; OpenAI [cache mechanics differ across models](https://docs.openai.com/api-reference/chat).

`cachebench` wraps your client call and tells you, per request, what hit and what didn't.

## Quick start

```python
from anthropic import Anthropic
from cachebench import CacheTracker, Provider

client = Anthropic()
tracker = CacheTracker(provider=Provider.ANTHROPIC, miss_alert_threshold=0.6)
create = tracker.wrap(client.messages.create)

response = create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    system=[{"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": "Hello"}],
)

print(tracker.aggregate())
# {'calls': 1, 'hit_ratio': 0.94, 'cost_saved_usd': 0.012, ...}
```

## Features

- **Per-call attribution.** Every wrapped call records `cache_read_tokens`, `cache_creation_tokens`, `hit_ratio`, `cost_saved_usd`, `prefix_id` (stable hash of the cacheable prefix).
- **Regression alerts.** Configurable threshold; default fires on stderr when a request with a cacheable prefix hits below 60%. Pass `on_miss_alert=` to forward to your own logger / Slack / PagerDuty.
- **Miss-aware retry.** Opt in with `CachePolicy.miss_aware()` to retry once on a silent miss after a configurable delay (works around Anthropic's documented eventual-consistency window).
- **Per-prefix grouping.** `tracker.by_prefix()` shows hit rate per stable prefix — instantly tells you which system prompt regressed.
- **Multi-provider.** Anthropic, OpenAI (via `prompt_tokens_details.cached_tokens`), Bedrock (`AnthropicBedrock` client).
- **Async-aware.** `tracker.wrap` detects coroutines and wraps both sync and async paths.

## Recipes

**Alert to Slack on regression**

```python
import requests

def to_slack(m):
    requests.post(SLACK_URL, json={"text": f"Cache regression: {m.prefix_id} ratio={m.hit_ratio:.2f}"})

tracker = CacheTracker(provider=Provider.ANTHROPIC, on_miss_alert=to_slack)
```

**Retry around the Anthropic 40% miss bug**

```python
from cachebench import CachePolicy

tracker = CacheTracker(
    provider=Provider.ANTHROPIC,
    policy=CachePolicy.miss_aware(delay_ms=2000, max_retries=1),
)
```

**Find which prefix regressed**

```python
for prefix_id, stats in tracker.by_prefix().items():
    if stats["hit_ratio"] < 0.5:
        print(f"REGRESSED: {prefix_id} {stats}")
```

## What it doesn't do

- Not a proxy. Not a router. Not a cache itself — it observes the provider's cache, doesn't store responses.
- Not a billing dashboard. Exports metrics; aggregation/UI is your job.
- Doesn't modify prompts (no auto-injecting cache breakpoints — see other tools for that).

## Relationship to bedrock-kit

If you only use AWS Bedrock and you want a full client wrapper (adaptive throttle, structured-output JSON repair, cost ledger), use [bedrock-kit](https://github.com/MukundaKatta/bedrock-kit) — it tracks `cache_read_tokens` per call as part of its cost ledger.

Use `cachebench` when you want:

- Cross-provider observability (Anthropic + OpenAI direct API, plus Bedrock).
- Per-prefix attribution and hit-rate regression alerts.
- Miss-aware retry around Anthropic's documented eventual-consistency window.

The two compose cleanly: wrap a `bedrock_kit.BedrockClient.invoke` call with `CacheTracker.wrap` and you get bedrock-kit's client features plus cachebench's regression alerting.

## Pricing data

`DEFAULT_PRICING` ships with current Anthropic/OpenAI/Bedrock rates. Pricing changes; pass `pricing=` to override:

```python
tracker = CacheTracker(
    provider=Provider.ANTHROPIC,
    pricing={"input": 3.00, "cache_read": 0.30, "cache_write_5m": 3.75, "cache_write_1h": 6.00, "output": 15.00},
)
```

## License

MIT
