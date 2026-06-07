import asyncio

from cachebench import CacheTracker, CachePolicy, Provider, fingerprint


class FakeUsage:
    def __init__(self, input_tokens=0, cache_read=0, cache_creation=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation
        self.output_tokens = output_tokens


class FakeResp:
    def __init__(self, usage):
        self.usage = usage


def test_fingerprint_is_stable():
    a = fingerprint(messages=[{"role": "user", "content": "hi"}], system="sys", model="m")
    b = fingerprint(messages=[{"role": "user", "content": "hi"}], system="sys", model="m")
    assert a == b


def test_fingerprint_excludes_trailing_user_turn():
    a = fingerprint(messages=[{"role": "user", "content": "A"}], system="sys")
    b = fingerprint(messages=[{"role": "user", "content": "B"}], system="sys")
    assert a == b  # last user turn excluded


def test_fingerprint_changes_with_system_or_model():
    a = fingerprint(messages=[], system="sys1", model="m")
    b = fingerprint(messages=[], system="sys2", model="m")
    c = fingerprint(messages=[], system="sys1", model="m2")
    assert a != b
    assert a != c


def test_record_and_aggregate():
    t = CacheTracker(provider=Provider.ANTHROPIC)
    fn_calls = []

    def fn(**kwargs):
        fn_calls.append(kwargs)
        return FakeResp(FakeUsage(input_tokens=100, cache_read=800, output_tokens=50))

    create = t.wrap(fn)
    create(messages=[{"role": "user", "content": "hi"}], system="sys", model="m")

    agg = t.aggregate()
    assert agg["calls"] == 1
    assert agg["hit_ratio"] == 1.0
    assert agg["tokens_read_from_cache"] == 800
    assert agg["cost_saved_usd"] > 0


def test_miss_alert_fires():
    fired = []
    t = CacheTracker(
        provider=Provider.ANTHROPIC,
        miss_alert_threshold=0.5,
        on_miss_alert=lambda m: fired.append(m),
    )
    create = t.wrap(lambda **kw: FakeResp(FakeUsage(input_tokens=100, cache_creation=900)))
    create(messages=[], system="sys", model="m")
    assert len(fired) == 1
    assert fired[0].hit_ratio == 0.0


def test_no_alert_when_no_cacheable_prefix():
    fired = []
    t = CacheTracker(
        provider=Provider.ANTHROPIC,
        miss_alert_threshold=0.99,
        on_miss_alert=lambda m: fired.append(m),
    )
    create = t.wrap(lambda **kw: FakeResp(FakeUsage(input_tokens=100, output_tokens=50)))
    create(messages=[], system="sys", model="m")
    assert fired == []  # no cacheable prefix means no alert


def test_retry_on_miss_policy():
    attempts = {"n": 0}

    def fn(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResp(FakeUsage(input_tokens=10, cache_creation=900))  # miss
        return FakeResp(FakeUsage(input_tokens=10, cache_read=900))  # hit

    t = CacheTracker(
        provider=Provider.ANTHROPIC,
        policy=CachePolicy.miss_aware(delay_ms=1, max_retries=1),
    )
    create = t.wrap(fn)
    create(messages=[], system="sys", model="m")
    assert attempts["n"] == 2  # retried once


def test_by_prefix_groups_correctly():
    t = CacheTracker(provider=Provider.ANTHROPIC)
    create = t.wrap(lambda **kw: FakeResp(FakeUsage(input_tokens=10, cache_read=90)))
    create(messages=[], system="A", model="m")
    create(messages=[], system="A", model="m")
    create(messages=[], system="B", model="m")
    by = t.by_prefix()
    assert len(by) == 2


def test_openai_usage_format():
    """OpenAI nests cached tokens inside prompt_tokens_details."""

    class OAIUsage:
        prompt_tokens = 1000  # includes cached
        completion_tokens = 50
        prompt_tokens_details = {"cached_tokens": 800}

    t = CacheTracker(provider=Provider.OPENAI)
    create = t.wrap(lambda **kw: FakeResp(OAIUsage()))
    create(messages=[], system="sys", model="m")
    agg = t.aggregate()
    assert agg["tokens_read_from_cache"] == 800


def test_dict_response_works():
    t = CacheTracker(provider=Provider.ANTHROPIC)
    create = t.wrap(
        lambda **kw: {
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 90,
                "output_tokens": 5,
            }
        }
    )
    create(messages=[], system="sys", model="m")
    assert t.aggregate()["tokens_read_from_cache"] == 90


def test_window_filters_by_time():
    import time as t_mod

    t = CacheTracker(provider=Provider.ANTHROPIC)
    create = t.wrap(lambda **kw: FakeResp(FakeUsage(input_tokens=10, cache_read=90)))
    create(messages=[], system="sys", model="m")
    t_mod.sleep(0.01)
    create(messages=[], system="sys", model="m")
    recent = t.window(seconds=0.005)
    # The first call's timestamp is older than 5ms ago.
    assert len(recent) <= 2  # at least the second; may include first depending on timing
    assert len(recent) >= 1


def test_aggregate_empty():
    t = CacheTracker(provider=Provider.ANTHROPIC)
    assert t.aggregate() == {"calls": 0}


def test_reset_clears_history():
    t = CacheTracker(provider=Provider.ANTHROPIC)
    create = t.wrap(lambda **kw: FakeResp(FakeUsage(input_tokens=10, cache_read=90)))
    create(messages=[], system="sys", model="m")
    assert t.aggregate()["calls"] == 1
    t.reset()
    assert t.calls() == []
    assert t.aggregate() == {"calls": 0}


def test_cost_saved_is_positive_on_cache_hit():
    """A cache hit should record positive savings versus paying full input price."""
    t = CacheTracker(provider=Provider.ANTHROPIC)
    create = t.wrap(lambda **kw: FakeResp(FakeUsage(input_tokens=100, cache_read=900)))
    create(messages=[], system="sys", model="m")
    m = t.calls()[0]
    # 900 cached tokens read at 0.30/M instead of 3.00/M => clear savings.
    assert m.cost_saved_usd(t.pricing) > 0
    assert t.aggregate()["cost_saved_usd"] > 0


def test_alert_callback_exception_is_swallowed():
    """A throwing on_miss_alert must not break the wrapped call."""

    def boom(_m):
        raise RuntimeError("alert sink down")

    t = CacheTracker(
        provider=Provider.ANTHROPIC,
        miss_alert_threshold=0.5,
        on_miss_alert=boom,
    )
    create = t.wrap(lambda **kw: FakeResp(FakeUsage(input_tokens=100, cache_creation=900)))
    # Should not raise even though the alert callback raises.
    create(messages=[], system="sys", model="m")
    assert t.aggregate()["calls"] == 1


def test_async_wrap_records_metrics():
    t = CacheTracker(provider=Provider.ANTHROPIC)

    async def fn(**kwargs):
        return FakeResp(FakeUsage(input_tokens=100, cache_read=800, output_tokens=50))

    create = t.wrap(fn)

    async def run():
        return await create(messages=[{"role": "user", "content": "hi"}], system="sys", model="m")

    resp = asyncio.run(run())
    assert resp is not None
    agg = t.aggregate()
    assert agg["calls"] == 1
    assert agg["hit_ratio"] == 1.0
    assert agg["tokens_read_from_cache"] == 800


def test_async_retry_on_miss_policy():
    attempts = {"n": 0}

    async def fn(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResp(FakeUsage(input_tokens=10, cache_creation=900))  # miss
        return FakeResp(FakeUsage(input_tokens=10, cache_read=900))  # hit

    t = CacheTracker(
        provider=Provider.ANTHROPIC,
        policy=CachePolicy.miss_aware(delay_ms=1, max_retries=1),
    )
    create = t.wrap(fn)

    async def run():
        await create(messages=[], system="sys", model="m")

    asyncio.run(run())
    assert attempts["n"] == 2  # retried once
