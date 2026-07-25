"""Gemini backend: key rotation, response parsing, pacing, quota handling.

No network here. The transport is injected so behaviour is pinned
deterministically; live reachability is exercised by the eval runs.
"""

import json

import pytest

from rho.llm.gemini import GeminiClient, QuotaExhausted, RateLimited, load_api_keys


def test_load_api_keys_parses_json_array():
    keys = load_api_keys('GEMINI_API_KEY=["a","b","c"]')
    assert keys == ["a", "b", "c"]


def test_load_api_keys_accepts_bare_single_key():
    assert load_api_keys("GEMINI_API_KEY=solo") == ["solo"]


def test_load_api_keys_errors_when_absent():
    with pytest.raises(ValueError):
        load_api_keys("SOMETHING_ELSE=1")


def test_keys_rotate_round_robin():
    client = GeminiClient(api_keys=["k1", "k2", "k3"], transport=lambda *a, **k: ({}, {}))
    assert [client.next_key() for _ in range(4)] == ["k1", "k2", "k3", "k1"]


def _body(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_complete_json_returns_parsed_payload():
    def fake_transport(url, payload, timeout):
        return _body('{"skills":["Go"]}'), {}

    client = GeminiClient(api_keys=["k1"], transport=fake_transport)
    assert client.complete_json("prompt") == {"skills": ["Go"]}


def test_complete_json_ignores_leading_thought_parts():
    """`thoughtSignature` parts precede the answer; the JSON is always last."""

    def fake_transport(url, payload, timeout):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "reasoning...", "thought": True},
                            {"text": '{"ok":true}'},
                        ]
                    }
                }
            ]
        }, {}

    client = GeminiClient(api_keys=["k1"], transport=fake_transport)
    assert client.complete_json("prompt") == {"ok": True}


def test_complete_json_retries_on_failure_with_next_key():
    calls = []

    def flaky(url, payload, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("network blip")
        return _body('{"ok":true}'), {}

    client = GeminiClient(api_keys=["k1", "k2"], transport=flaky, requests_per_minute=1000)
    assert client.complete_json("prompt") == {"ok": True}
    assert len(calls) == 2
    assert "key=k1" in calls[0] and "key=k2" in calls[1]


def test_complete_json_raises_after_exhausting_keys():
    def always_fail(url, payload, timeout):
        raise RuntimeError("boom")

    client = GeminiClient(
        api_keys=["k1", "k2"], transport=always_fail, rounds=1, backoff=0
    )
    with pytest.raises(RuntimeError):
        client.complete_json("prompt")


def test_rate_limited_key_waits_server_reset_before_retrying():
    from rho.llm import gemini as mod

    slept, calls = [], []

    def limited_once(url, payload, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise RateLimited(9.0, "429")
        return _body('{"ok":true}'), {}

    client = GeminiClient(api_keys=["k1"], transport=limited_once, max_wait=70)
    original = mod.time.sleep
    mod.time.sleep = slept.append
    try:
        assert client.complete_json("p") == {"ok": True}
    finally:
        mod.time.sleep = original
    assert slept and 9.0 <= slept[0] <= 10.0


def test_rate_limited_key_falls_through_to_next_before_sleeping():
    """A single rate-limited key must not stall the whole call.

    Regression: sleeping on every rate-limited attempt (rather than once per
    full rotation) turned one stubborn pair into a ~20min stall in the
    199-pair calibrator run (7 keys x 3 rounds x up to 71s each).
    """
    slept, calls = [], []

    def first_key_limited(url, payload, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise RateLimited(70.0, "429")
        return _body('{"ok":true}'), {}

    client = GeminiClient(api_keys=["k1", "k2", "k3"], transport=first_key_limited)
    import rho.llm.gemini as mod

    original = mod.time.sleep
    mod.time.sleep = slept.append
    try:
        assert client.complete_json("p") == {"ok": True}
    finally:
        mod.time.sleep = original
    assert "key=k1" in calls[0] and "key=k2" in calls[1]
    assert slept == []  # fell through to k2 without sleeping


def test_rate_limit_wait_is_capped():
    from rho.llm import gemini as mod

    slept = []

    def always_limited(url, payload, timeout):
        raise RateLimited(3600.0, "429")

    client = GeminiClient(api_keys=["k1"], transport=always_limited, rounds=2, max_wait=70)
    original = mod.time.sleep
    mod.time.sleep = slept.append
    try:
        with pytest.raises(RuntimeError):
            client.complete_json("p")
    finally:
        mod.time.sleep = original
    assert all(s <= 71 for s in slept)


def test_is_daily_quota_detects_real_free_tier_body():
    """Regression: Google's daily-quota 429 DOES carry a RetryInfo detail (a
    bogus short delay — the quota resets once a day, not in 30s), so presence
    of RetryInfo cannot be used to rule out a daily quota. This is a real
    body observed from a Gemini free-tier daily quota."""
    from rho.llm.gemini import _is_daily_quota

    real_body = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota...",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            "quotaDimensions": {"model": "gemini-3.1-flash-lite"},
                            "quotaValue": "20",
                        }
                    ],
                },
                {
                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                    "retryDelay": "34s",
                },
            ],
        }
    }
    assert _is_daily_quota(real_body) is True


def test_is_daily_quota_false_for_per_minute_limit():
    from rho.llm.gemini import _is_daily_quota

    per_minute_body = {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests_per_minute",
                            "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                            "quotaValue": "10",
                        }
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "5s"},
            ],
        }
    }
    assert _is_daily_quota(per_minute_body) is False


def test_transport_raises_quota_exhausted_on_daily_quota_body():
    from rho.llm.gemini import _httpx_transport

    class FakeResponse:
        def __init__(self, body):
            self.status_code = 429
            self.content = json.dumps(body).encode()
            self.headers = {}

        def json(self):
            return json.loads(self.content)

    real_body = {
        "error": {
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {
                            "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                        }
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"},
            ],
        }
    }

    import types

    fake_httpx = types.SimpleNamespace(post=lambda *a, **k: FakeResponse(real_body))
    import sys

    saved = sys.modules.get("httpx")
    sys.modules["httpx"] = fake_httpx
    try:
        with pytest.raises(QuotaExhausted):
            _httpx_transport("u", {}, 10)
    finally:
        if saved is not None:
            sys.modules["httpx"] = saved


def test_transport_raises_quota_exhausted_on_403_denied_key():
    """A key can be individually revoked (403 PERMISSION_DENIED) without the
    account's quota being spent — observed live during a 300-résumé Phase-7
    run. Must be treated as permanently dead, same as QuotaExhausted, so the
    client rotates off it instead of burning retries on it every rotation."""
    from rho.llm.gemini import _httpx_transport

    class FakeResponse:
        def __init__(self, body, status_code):
            self.status_code = status_code
            self.content = json.dumps(body).encode()
            self.headers = {}

        def json(self):
            return json.loads(self.content)

    denied_body = {
        "error": {
            "code": 403,
            "message": "Your project has been denied access. Please contact support.",
            "status": "PERMISSION_DENIED",
        }
    }

    import sys
    import types

    fake_httpx = types.SimpleNamespace(post=lambda *a, **k: FakeResponse(denied_body, 403))
    saved = sys.modules.get("httpx")
    sys.modules["httpx"] = fake_httpx
    try:
        with pytest.raises(QuotaExhausted):
            _httpx_transport("u", {}, 10)
    finally:
        if saved is not None:
            sys.modules["httpx"] = saved


def test_daily_quota_exhaustion_moves_to_next_key_not_retried():
    """Unlike Groq's shared TPD, Gemini quota is per-key/per-project: rotate off
    the exhausted key rather than aborting the whole call."""
    calls = []

    def key1_exhausted(url, payload, timeout):
        calls.append(url)
        if "key=k1" in url:
            raise QuotaExhausted("RESOURCE_EXHAUSTED: quota exceeded per day")
        return _body('{"ok":true}'), {}

    client = GeminiClient(api_keys=["k1", "k2"], transport=key1_exhausted)
    assert client.complete_json("p") == {"ok": True}
    assert any("key=k2" in c for c in calls)


def test_all_keys_exhausted_raises_quota_exhausted():
    def always_exhausted(url, payload, timeout):
        raise QuotaExhausted("RESOURCE_EXHAUSTED")

    client = GeminiClient(api_keys=["k1", "k2"], transport=always_exhausted, rounds=1)
    with pytest.raises(QuotaExhausted):
        client.complete_json("p")


def test_response_schema_is_sent_when_provided():
    seen = {}

    def capture(url, payload, timeout):
        seen["config"] = payload["generationConfig"]
        return _body("{}"), {}

    client = GeminiClient(api_keys=["k1"], transport=capture)
    schema = {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}
    client.complete_json("p", response_schema=schema)
    assert seen["config"]["responseSchema"] == schema
    assert seen["config"]["responseMimeType"] == "application/json"


# --- request pacing -------------------------------------------------------


def test_pacer_allows_requests_within_limit():
    from rho.llm.gemini import RequestPacer

    clock = [0.0]
    pacer = RequestPacer(requests_per_minute=10, now=lambda: clock[0])
    assert pacer.acquire() == 0.0
    assert pacer.acquire() == 0.0


def test_pacer_waits_when_window_is_exhausted():
    from rho.llm.gemini import RequestPacer

    clock = [0.0]
    pacer = RequestPacer(requests_per_minute=2, now=lambda: clock[0])
    pacer.acquire()
    pacer.acquire()
    wait = pacer.acquire()
    assert 0 < wait <= 60


def test_pacer_frees_capacity_after_window_rolls():
    from rho.llm.gemini import RequestPacer

    clock = [0.0]
    pacer = RequestPacer(requests_per_minute=2, now=lambda: clock[0])
    pacer.acquire()
    pacer.acquire()
    clock[0] = 61.0
    assert pacer.acquire() == 0.0


def test_client_paces_before_sending():
    """The client must wait before sending, not after being refused."""
    clock, slept = [0.0], []

    def ok(url, payload, timeout):
        return _body('{"ok":true}'), {}

    client = GeminiClient(api_keys=["k"], transport=ok, requests_per_minute=1)
    client._pacers["k"]._now = lambda: clock[0]

    import rho.llm.gemini as mod

    original = mod.time.sleep
    mod.time.sleep = slept.append
    try:
        client.complete_json("first")
        assert client.complete_json("second") == {"ok": True}
    finally:
        mod.time.sleep = original
    assert slept and slept[0] > 0
