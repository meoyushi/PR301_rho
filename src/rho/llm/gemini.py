"""Gemini backend (`gemini-3.1-flash-lite`) for JD analysis, extraction, and rewrite.

Same role as `rho.llm.groq`: a thread-safe round-robin client over several
free-tier API keys, used as a drop-in alternate backend behind the frozen
`analyze_jd`/`extract`/`rewrite` contracts.

Three things about this model and endpoint shape the client:

1. **It is a "thinking" model** (`thinking: true` in the model metadata) —
   Gemini's Flash tier now reasons before answering the way Groq's Qwen3.6
   does, but the reasoning happens server-side: the response never contains a
   `<think>` block to strip, and `usageMetadata.thoughtsTokenCount` reports it
   separately from `candidatesTokenCount`. Both count toward the visible
   output for budgeting purposes even though only `candidatesTokenCount`'s
   text is returned.
2. **`responseSchema` gives real constrained decoding**, unlike Groq's Qwen3.6
   (schema enforced by Pydantic after the fact, per `rho.llm.groq`'s
   docstring). This client passes a JSON Schema and `responseMimeType:
   application/json`, so malformed output should only occur if the schema
   itself under-constrains a field — Pydantic validation still runs behind it
   as the actual gate, exactly as the Ollama backend does.
3. **No rate-limit headers are exposed** on this endpoint (unlike Groq's
   `x-ratelimit-*`), so pacing cannot read the account's live budget the way
   `TokenBudget` does for Groq. `GeminiClient` paces at a conservative fixed
   rate per key instead and backs off hard on any 429, parsing Google's
   `retryInfo` detail when present.
"""

import itertools
import json
import random
import re
import threading
import time

MODEL = "gemini-3.1-flash-lite"
_ENDPOINT_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
_TIMEOUT_SECONDS = 180


def load_api_keys(env_text: str) -> list[str]:
    """Read GEMINI_API_KEY from .env text. Accepts a JSON array or a bare key."""
    match = re.search(r"GEMINI_API_KEY\s*=\s*(.+)", env_text)
    if not match:
        raise ValueError("GEMINI_API_KEY not found in env text")
    raw = match.group(1).strip().strip("'\"")
    if raw.startswith("["):
        keys = json.loads(re.search(r"(\[.*?\])", raw, re.S).group(1))
    else:
        keys = [raw]
    keys = [k for k in (k.strip() for k in keys) if k]
    if not keys:
        raise ValueError("GEMINI_API_KEY is empty")
    return keys


def load_api_keys_from_file(path: str = ".env") -> list[str]:
    with open(path) as fh:
        return load_api_keys(fh.read())


class QuotaExhausted(Exception):
    """Daily quota (RPD) is spent for this key. Retrying cannot help today."""


class RateLimited(Exception):
    """429 from Gemini, carrying the server's requested wait in seconds."""

    def __init__(self, retry_after: float, message: str):
        super().__init__(message)
        self.retry_after = retry_after


class RequestPacer:
    """Sliding-window requests-per-minute pacer, one per API key.

    Google does not expose live rate-limit headers on this endpoint, so this
    paces *before* sending rather than reacting to headers — conservative by
    construction, tightened only by actual 429s (see `GeminiClient.rounds`).
    """

    def __init__(self, requests_per_minute: int, now=time.monotonic):
        self.limit = requests_per_minute
        self._now = now
        self._sent: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> float:
        with self._lock:
            now = self._now()
            cutoff = now - 60.0
            self._sent = [t for t in self._sent if t > cutoff]
            wait = 0.0
            if len(self._sent) >= self.limit:
                wait = max(0.0, self._sent[0] + 60.0 - now)
            self._sent.append(now + wait)
            return wait


def _parse_retry_after(body: dict, headers) -> float:
    """Prefer the `retry-after` header; fall back to the error body's RetryInfo."""
    header_val = headers.get("retry-after")
    if header_val:
        try:
            return float(header_val)
        except ValueError:
            pass
    for detail in body.get("error", {}).get("details", []):
        if detail.get("@type", "").endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                return float(raw)
            except ValueError:
                pass
    return 5.0


def _is_daily_quota(body: dict) -> bool:
    """True when the 429's QuotaFailure names a per-day quota.

    Google's daily-quota 429 still carries a `RetryInfo` detail (a short delay
    that does NOT actually help — the quota resets once a day, not in 30s), so
    presence/absence of `RetryInfo` cannot distinguish it from a real
    per-minute limit. The `quotaId`/`quotaMetric` naming ("...PerDay...") is
    the only reliable signal; confirmed against a live 429 body
    (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`).
    """
    for detail in body.get("error", {}).get("details", []):
        if not detail.get("@type", "").endswith("QuotaFailure"):
            continue
        for violation in detail.get("violations", []):
            names = violation.get("quotaId", "") + violation.get("quotaMetric", "")
            if "PerDay" in names or "per_day" in names:
                return True
    return False


def _httpx_transport(url: str, payload: dict, timeout: int) -> tuple[dict, dict]:
    import httpx

    response = httpx.post(
        url, headers={"Content-Type": "application/json"}, json=payload, timeout=timeout
    )
    if response.status_code == 429:
        body = response.json() if response.content else {}
        message = json.dumps(body)[:300]
        if _is_daily_quota(body):
            raise QuotaExhausted(f"Gemini daily quota exhausted: {message}")
        raise RateLimited(_parse_retry_after(body, response.headers), f"429: {message}")
    if response.status_code == 403:
        # A key can be individually revoked ("project denied access") without
        # the whole account's quota being spent — observed live during a
        # 300-résumé Phase-7 run. Retrying a permanently-denied key every
        # rotation wastes the retry budget on something that can never
        # succeed; treat it the same as QuotaExhausted so the client rotates
        # it out and keeps serving the call from the other keys.
        body = response.json() if response.content else {}
        message = json.dumps(body)[:300]
        raise QuotaExhausted(f"Gemini key denied (403): {message}")
    response.raise_for_status()
    return response.json(), dict(response.headers)


class GeminiClient:
    """Thread-safe round-robin client over several Gemini API keys."""

    def __init__(
        self,
        api_keys: list[str] | None = None,
        model: str = MODEL,
        transport=_httpx_transport,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        requests_per_minute: int = 10,
        rounds: int = 3,
        backoff: float = 2.0,
        max_wait: float = 70.0,
    ):
        self._keys = list(api_keys) if api_keys else load_api_keys_from_file()
        self._cycle = itertools.cycle(self._keys)
        self._lock = threading.Lock()
        self.model = model
        self.transport = transport
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.rounds = rounds
        self.backoff = backoff
        self.max_wait = max_wait
        # One pacer per key: RPM is a per-key (per-project) limit, not shared.
        self._pacers = {k: RequestPacer(requests_per_minute) for k in self._keys}
        self._exhausted_today: set[str] = set()

    def next_key(self) -> str:
        with self._lock:  # itertools.cycle is not thread-safe
            return next(self._cycle)

    def complete_json(
        self,
        prompt: str,
        response_schema: dict | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Send `prompt`, return the parsed JSON object.

        Tries each key once per round before giving up, so a rate-limited or
        exhausted key degrades to a retry on the next key instead of failing
        the whole call.
        """
        generation_config = {
            "temperature": self.temperature if temperature is None else temperature,
            "maxOutputTokens": self.max_output_tokens,
            "responseMimeType": "application/json",
        }
        if response_schema is not None:
            generation_config["responseSchema"] = response_schema
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        errors = []
        live_keys = [k for k in self._keys if k not in self._exhausted_today]
        if not live_keys:
            raise QuotaExhausted("all Gemini keys exhausted their daily quota")
        attempts = len(live_keys) * self.rounds
        for attempt in range(attempts):
            key = self.next_key()
            if key in self._exhausted_today:
                continue
            wait = self._pacers[key].acquire()
            if wait > 0:
                time.sleep(wait)
            url = f"{_ENDPOINT_TMPL.format(model=self.model)}?key={key}"
            try:
                body, _headers = self.transport(url, payload, _TIMEOUT_SECONDS)
                text = body["candidates"][0]["content"]["parts"][-1]["text"]
                return json.loads(text)
            except QuotaExhausted as exc:
                self._exhausted_today.add(key)
                errors.append(f"{type(exc).__name__}: {exc}")
                live_keys = [k for k in self._keys if k not in self._exhausted_today]
                if not live_keys:
                    raise
                continue
            except Exception as exc:  # try the next key before failing the call
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt + 1 >= attempts:
                    break
                # Only sleep once a full rotation has been tried — a single
                # rate-limited key says nothing about the next one, and
                # sleeping on every attempt turns one bad pair into a ~20min
                # stall (7 keys x 3 rounds x up to 71s each).
                if (attempt + 1) % max(len(live_keys), 1) != 0:
                    continue
                if isinstance(exc, RateLimited):
                    time.sleep(min(exc.retry_after, self.max_wait) + random.uniform(0, 1))
                else:
                    delay = min(self.backoff * 2 ** (attempt // max(len(live_keys), 1)), 30.0)
                    time.sleep(random.uniform(0, delay))
        raise RuntimeError(
            f"all {len(live_keys)} live Gemini keys failed after {attempts} attempts: "
            f"{errors[-3:]}"
        )
