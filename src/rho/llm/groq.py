"""Groq backend for `qwen/qwen3.6-27b` (131k context).

Three things about this model and endpoint shape the client:

1. **Cloudflare blocks `urllib`.** Requests through `urllib.request` are rejected
   with `403 / error code: 1010` *before* Groq evaluates the key — a bogus key
   and a valid one fail identically. `httpx` and `curl` pass, so the transport
   is httpx rather than the stdlib used elsewhere in this repo.
2. **It is a reasoning model.** It emits a `<think>…</think>` block inline, which
   breaks both `json_schema` (unsupported by this model) and `json_object`
   (validation fails on the reasoning prefix). Passing `reasoning_format=hidden`
   suppresses the block server-side and yields clean JSON; `strip_reasoning`
   defends against anything that leaks through anyway.
3. **Several keys are available.** They round-robin so parallel workers spread
   load across quotas, and a failing key falls through to the next rather than
   killing the call.

**Quota reality (measured, free `on_demand` tier).** Two limits stack, and only
the per-minute one is visible in `x-ratelimit-*` headers:

- **TPM 8000** — reported in headers. `max_tokens` is reserved at *admission*,
  not on use, so a 4096-token ceiling claims half the window per call.
- **TPD 200000** — invisible in headers; it appears only in the 429 body
  (`... on tokens per day (TPD): Limit 200000, Used 198323`). This is the binding
  constraint for benchmark work: one 30-pair fabrication run costs ~136k tokens,
  so the daily budget affords roughly **1.5 runs per day**.

All five keys were observed draining in lockstep (~198.5k/200k each), so the
daily pool is effectively shared: adding keys does not buy more headroom. Plan
benchmark runs against TPD, not TPM, and treat a rate-limit failure late in a
session as "quota spent" rather than something a retry can fix.

Constrained decoding is therefore *prompt-plus-validation* rather than
grammar-enforced, unlike the Ollama path. Pydantic validation at the call site
is what rejects malformed output — no silent fills.
"""

import itertools
import json
import random
import re
import threading
import time

MODEL = "qwen/qwen3.6-27b"
_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_TIMEOUT_SECONDS = 180

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


_shared_budget: "TokenBudget | None" = None
_budget_lock = threading.Lock()


def shared_budget(tokens_per_minute: int = 8000) -> "TokenBudget":
    """One budget per process: the account-wide TPM cap is what is being paced."""
    global _shared_budget
    with _budget_lock:
        if _shared_budget is None:
            _shared_budget = TokenBudget(tokens_per_minute)
        return _shared_budget


def load_api_keys(env_text: str) -> list[str]:
    """Read GROQ_API_KEY from .env text. Accepts a JSON array or a bare key."""
    match = re.search(r"GROQ_API_KEY\s*=\s*(.+)", env_text)
    if not match:
        raise ValueError("GROQ_API_KEY not found in env text")
    raw = match.group(1).strip().strip("'\"")
    if raw.startswith("["):
        keys = json.loads(re.search(r"(\[.*?\])", raw, re.S).group(1))
    else:
        keys = [raw]
    keys = [k for k in (k.strip() for k in keys) if k]
    if not keys:
        raise ValueError("GROQ_API_KEY is empty")
    return keys


def load_api_keys_from_file(path: str = ".env") -> list[str]:
    with open(path) as fh:
        return load_api_keys(fh.read())


def strip_reasoning(content: str) -> str:
    """Return the JSON payload from a model response.

    `reasoning_format=hidden` should make this a no-op, but the model still
    occasionally wraps output in a fence or leaks a think block, and a parse
    failure costs a whole benchmark pair.
    """
    text = _THINK.sub("", content).strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start > 0:
        text = text[start:]
    return text.strip()


class TokenBudget:
    """Sliding-window token pacer shared by every worker on one account.

    Groq's free tier caps *tokens per minute* per account, so key rotation does
    not help: all keys draw on the same budget. Absorbing 429s and sleeping is
    strictly worse than not sending — under a saturated budget the retry loop
    spends its whole time asleep and throughput collapses to zero. This paces
    requests before they are sent instead.
    """

    def __init__(self, tokens_per_minute: int = 8000, now=time.monotonic):
        self.limit = tokens_per_minute
        self._now = now
        self._spent: list[tuple[float, int]] = []  # (timestamp, tokens)
        self._lock = threading.Lock()

    def acquire(self, tokens: int) -> float:
        """Record `tokens` against the budget; return seconds to wait first."""
        with self._lock:
            now = self._now()
            cutoff = now - 60.0
            self._spent = [(t, n) for t, n in self._spent if t > cutoff]
            used = sum(n for _, n in self._spent)
            wait = 0.0
            if used + tokens > self.limit and self._spent:
                # Wait for the oldest entry to age out of the window.
                wait = max(0.0, self._spent[0][0] + 60.0 - now)
            self._spent.append((now + wait, min(tokens, self.limit)))
            return wait


def estimate_tokens(text: str) -> int:
    """Rough token count for budgeting (~4 chars/token), plus generation headroom."""
    return len(text) // 4


class QuotaExhausted(Exception):
    """Daily token pool (TPD) is spent. Retrying cannot help; stop the run."""


class RateLimited(Exception):
    """429 from Groq, carrying the server's requested wait in seconds."""

    def __init__(self, retry_after: float, message: str):
        super().__init__(message)
        self.retry_after = retry_after


def _httpx_transport(url: str, headers: dict, payload: dict, timeout: int) -> str:
    import httpx

    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code == 429:
        body = response.text
        if "(TPD)" in body or "tokens per day" in body:
            # Not transient: the daily pool is gone and no retry recovers it.
            raise QuotaExhausted(f"Groq daily token quota exhausted: {body[:220]}")
        # The free tier limits *tokens* per minute (8k), not just requests, and
        # the reset header is far more accurate than guessing at a backoff.
        raw = response.headers.get("retry-after") or response.headers.get(
            "x-ratelimit-reset-tokens", "5"
        )
        raise RateLimited(_parse_duration(raw), f"429: {body[:120]}")
    response.raise_for_status()
    return response.text


def _parse_duration(raw: str) -> float:
    """Groq expresses resets as plain seconds or as e.g. `1m33.5s`."""
    raw = str(raw).strip()
    try:
        return float(raw)
    except ValueError:
        pass
    total, match = 0.0, re.findall(r"(\d+(?:\.\d+)?)([hms])", raw)
    for value, unit in match:
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total or 5.0


class GroqClient:
    """Thread-safe round-robin client over several API keys."""

    def __init__(
        self,
        api_keys: list[str] | None = None,
        model: str = MODEL,
        transport=_httpx_transport,
        temperature: float = 0.6,
        # Groq reserves `max_tokens` against the per-minute budget when the
        # request is admitted, not when it is spent: at 4096 a single call claims
        # half the 8k window and 429s immediately, while 1500 is served at once.
        # Sized to hold a full rewritten résumé and no more.
        max_tokens: int = 1500,
        rounds: int = 3,
        backoff: float = 2.0,
        max_wait: float = 70.0,
        budget: "TokenBudget | None" = None,
    ):
        self._keys = list(api_keys) if api_keys else load_api_keys_from_file()
        self._cycle = itertools.cycle(self._keys)
        self._lock = threading.Lock()
        self.model = model
        self.transport = transport
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rounds = rounds
        self.backoff = backoff
        self.max_wait = max_wait
        self.budget = budget

    def next_key(self) -> str:
        with self._lock:  # itertools.cycle is not thread-safe
            return next(self._cycle)

    def complete_json(self, prompt: str, temperature: float | None = None) -> dict:
        """Send `prompt`, return the parsed JSON object.

        Tries each key once before giving up, so a rate-limited or revoked key
        degrades to a retry instead of a failed benchmark pair.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens,
            # Suppress the <think> block server-side; see module docstring.
            "reasoning_format": "hidden",
        }
        if self.budget is not None:
            # Charge what Groq actually reserves at admission: the prompt plus the
            # FULL `max_tokens`, not an estimate of the likely completion. Charging
            # less lets the pacer admit roughly twice the account's allowance, and
            # the surplus comes back as 429s.
            wait = self.budget.acquire(estimate_tokens(prompt) + self.max_tokens)
            if wait > 0:
                time.sleep(wait)

        errors = []
        # One attempt per key per round; extra rounds exist so a burst of 429s
        # (shared per-account rate limits hit every key at once) waits rather
        # than burning the whole rotation in milliseconds and failing the pair.
        attempts = len(self._keys) * self.rounds
        for attempt in range(attempts):
            key = self.next_key()
            try:
                raw = self.transport(
                    _ENDPOINT,
                    {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    payload,
                    _TIMEOUT_SECONDS,
                )
                body = json.loads(raw)
                content = body["choices"][0]["message"]["content"]
                return json.loads(strip_reasoning(content))
            except QuotaExhausted:
                # Keys share the daily pool, so rotating and waiting are both
                # futile. Fail fast and loudly rather than after 15 attempts.
                raise
            except Exception as exc:  # try the next key before failing the call
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt + 1 >= attempts:
                    break
                if isinstance(exc, RateLimited):
                    # Keys may belong to different orgs, and the limit is per-org
                    # per-model: a key that 429s says nothing about the next one.
                    # Only sleep once the whole rotation has been tried, or a
                    # throttled first key would stall a call the others could serve.
                    if (attempt + 1) % len(self._keys) != 0:
                        continue
                    time.sleep(min(exc.retry_after, self.max_wait) + random.uniform(0, 1))
                elif (attempt + 1) % len(self._keys) == 0:
                    # Full jitter: concurrent workers must not retry in lockstep.
                    delay = min(self.backoff * 2 ** (attempt // len(self._keys)), 30.0)
                    time.sleep(random.uniform(0, delay))
        raise RuntimeError(
            f"all {len(self._keys)} Groq keys failed after {attempts} attempts: {errors[-3:]}"
        )
