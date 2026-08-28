"""
pipeline/llm.py

One thin interface over whichever LLM is available. Ollama by default.

WHY AN INTERFACE INSTEAD OF CALLING OLLAMA DIRECTLY
---------------------------------------------------
Three consumers are coming: Verifier #1 (sufficiency gate), the answering
model, and an LLM judge for the 26 prose-answer questions the mechanical
comparator cannot score. Each needs to be swappable independently -- it is
entirely reasonable to run the verifier on a small local model and the
answering step on something stronger, and we should be able to measure that
rather than commit to it.

The MockClient matters more than it looks. It returns deterministic canned
responses, so the whole pipeline and its eval harness can be built and tested
without any model running at all. Every structural bug gets caught for free
before spending minutes per run on real inference.

NO NEW DEPENDENCIES
-------------------
Ollama is called over plain HTTP with urllib from the standard library. The
Anthropic/OpenAI paths import their SDKs lazily, so nothing is required unless
you actually select that provider.

USAGE
-----
    from pipeline.llm import get_client
    llm = get_client()                        # reads LLM_PROVIDER / LLM_MODEL
    llm.complete("Say hi")
    llm.complete_json("Return {\"ok\": true}")

    python pipeline/llm.py --check            # what is actually available
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Preference order when no model is specified. Small instruct models first:
# the verifier runs once per question and latency dominates the whole loop.
OLLAMA_PREFERRED = [
    "qwen2.5:7b-instruct", "qwen2.5:7b", "qwen2.5:latest",
    "llama3.1:8b-instruct-q4_K_M", "llama3.1:8b", "llama3.1:latest",
    "mistral:7b-instruct", "mistral:latest",
    "phi3:medium", "phi3:mini",
    "gemma2:9b", "gemma2:latest",
]


@dataclass
class LLMResponse:
    text: str
    model: str
    elapsed_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self, default=None):
        """Parse the response as JSON, tolerating the wrappers small models add.

        Local models routinely ignore 'respond with JSON only' and emit
        ```json fences, a preamble sentence, or trailing commentary. Failing
        the whole run on that would be brittle, so we extract the outermost
        {...} rather than trusting the model to behave.
        """
        t = self.text.strip()
        if t.startswith("```"):
            t = t.split("```")[1] if "```" in t[3:] else t[3:]
            if t.startswith("json"):
                t = t[4:]
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            pass
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                pass
        return default


class OllamaClient:
    def __init__(self, model: str | None = None, host: str = OLLAMA_HOST,
                 temperature: float = 0.0, timeout: int = 180):
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.model = model or self._auto_model()
        self.name = f"ollama:{self.model}"

    # -------------------------------------------------------------- server --

    def list_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as r:
                data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def _auto_model(self) -> str:
        available = self.list_models()
        if not available:
            raise RuntimeError(
                f"No Ollama models found at {self.host}.\n"
                "  Is Ollama running?   ollama serve\n"
                "  Pull a model:        ollama pull qwen2.5:7b-instruct\n"
                "  Then:                python pipeline/llm.py --check"
            )
        for want in OLLAMA_PREFERRED:
            for have in available:
                if have == want or have.startswith(want.split(":")[0] + ":"):
                    return have
        return available[0]

    # ------------------------------------------------------------ requests --

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 512, json_mode: bool = False) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        t0 = time.time()
        try:
            data = self._post("/api/generate", payload)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama request failed ({self.host}): {e}") from e
        return LLMResponse(text=data.get("response", ""), model=self.model,
                           elapsed_s=time.time() - t0, raw=data)

    def complete_json(self, prompt: str, system: str | None = None,
                      max_tokens: int = 512) -> LLMResponse:
        return self.complete(prompt, system=system, max_tokens=max_tokens,
                             json_mode=True)


class MockClient:
    """Deterministic canned responses. Lets the pipeline and its eval be built
    and debugged with no model running."""

    def __init__(self, responses: dict[str, str] | None = None,
                 default: str = '{"sufficient": true, "reason": "mock"}'):
        self.responses = responses or {}
        self.default = default
        self.model = "mock"
        self.name = "mock"
        self.calls: list[str] = []

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 512, json_mode: bool = False) -> LLMResponse:
        self.calls.append(prompt)
        for needle, out in self.responses.items():
            if needle in prompt:
                return LLMResponse(text=out, model="mock")
        return LLMResponse(text=self.default, model="mock")

    def complete_json(self, prompt: str, system: str | None = None,
                      max_tokens: int = 512) -> LLMResponse:
        return self.complete(prompt, system=system, max_tokens=max_tokens,
                             json_mode=True)


class GroqClient:
    """Groq's OpenAI-compatible API. Free tier, and fast enough to iterate on.

    WHY THIS EXISTS
    ---------------
    Local mistral:latest answered a trivial JSON prompt in 30.5s. Verifier #1
    runs once per question, so a single 127-question sweep would take ~64
    minutes -- and Days 6-7 need many sweeps (verifier, answerer, judge, plus
    every prompt revision). At that speed the measure-and-iterate loop this
    whole project is built on stops working.

    Groq serves the same class of open models on custom silicon at roughly
    100x that throughput. Same models, same open weights, just not CPU-bound.

    Uses urllib -- no SDK required. Set GROQ_API_KEY in your environment; the
    key is never read from or written to any file in this repo.

    RATE LIMITS: the free tier is roughly 30 requests/minute. 127 questions is
    therefore ~4-5 minutes, and 429s are retried with backoff rather than
    failing the run.
    """

    API_BASE = "https://api.groq.com/openai/v1"

    # Preference order, resolved against what the account can actually see.
    # Hosted providers retire model names constantly -- the llama-3.x names
    # this list originally contained were already gone.
    PREFERRED = [
        "openai/gpt-oss-120b",     # strongest general model on the free tier
        "qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",      # fastest capable option
        "groq/compound",
        "groq/compound-mini",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]

    # Models that are NOT general-purpose chat models and must never be
    # auto-selected. Falling through to one of these is a silent disaster:
    # the first version of this list only excluded whisper/guard/tts/embed, so
    # auto-selection landed on `allam-2-7b` -- an ARABIC-language model -- and
    # it passed the JSON smoke test perfectly while being entirely wrong for
    # verifying English SEC filings. A smoke test proves the plumbing works,
    # not that the model is appropriate.
    NOT_CHAT = (
        "whisper",      # speech recognition
        "orpheus",      # text to speech
        "guard",        # safety classifiers (prompt-guard, llama-guard)
        "safeguard",
        "embed",        # embedding models
        "allam",        # Arabic-language model
        "tts",
    )

    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 timeout: int = 120, max_retries: int = 5,
                 min_interval_s: float = 0.0):
        # THROTTLING. The free tier limits tokens-per-minute, not just
        # requests-per-minute, and verifier prompts run ~3k tokens each. A
        # 127-question sweep at full speed burns the TPM budget in about 70
        # questions, after which EVERY remaining call 429s. Observed exactly
        # that: 62 successful calls, then 63 consecutive failures, with the
        # per-call average climbing from 25s to 51s as retries piled up.
        #
        # Sleeping between calls is counter-intuitively FASTER than being
        # rate-limited, because a 429 costs a full retry ladder (2+4+8+16s)
        # and still fails.
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self.api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set.\n"
                "  Get a free key at https://console.groq.com/keys\n"
                "  PowerShell (this session):  $env:GROQ_API_KEY='...'\n"
                "  PowerShell (persistent)  :  "
                "[Environment]::SetEnvironmentVariable('GROQ_API_KEY','...','User')\n"
                "Never commit the key to the repo."
            )
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = model or self._auto_model()
        self.name = f"groq:{self.model}"

    def _headers(self) -> dict[str, str]:
        # The User-Agent is NOT optional. urllib defaults to "Python-urllib/3.x",
        # which Groq's Cloudflare front end rejects with HTTP 403 "error code:
        # 1010" -- a client-fingerprint block that happens BEFORE the API key is
        # ever examined. The symptom is misleading: it looks exactly like an
        # auth failure, and no amount of checking the key fixes it.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "analyst-copilot/1.0",
        }

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.API_BASE}/models",
                                         headers=self._headers())
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            return sorted(m["id"] for m in data.get("data", []))
        except Exception:
            return []

    def _auto_model(self) -> str:
        available = self.list_models()
        if not available:
            return self.PREFERRED[0]
        for want in self.PREFERRED:
            if want in available:
                return want
        chat = [m for m in available
                if not any(x in m.lower() for x in self.NOT_CHAT)]
        if not chat:
            raise RuntimeError(
                "No general-purpose chat model available on this Groq key.\n"
                f"  visible: {available}\n"
                "Set one explicitly:  $env:LLM_MODEL='openai/gpt-oss-20b'"
            )
        return chat[0]

    def _is_reasoning_model(self) -> bool:
        m = self.model.lower()
        return "gpt-oss" in m or "qwen3" in m or "compound" in m

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 512, json_mode: bool = False) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": max_tokens}

        # REASONING MODELS NEED HEADROOM AND THROTTLING.
        #
        # gpt-oss (and qwen3, compound) emit internal reasoning tokens BEFORE
        # the visible answer, and those count against max_tokens. Asking for a
        # 40-token JSON verdict with max_tokens=200 produced an empty response:
        # the budget was consumed reasoning, nothing was emitted, and Groq
        # rejected the request with HTTP 400 json_validate_failed and an EMPTY
        # failed_generation -- which reads like a prompt bug but is a budget bug.
        #
        # reasoning_effort="low" keeps the deliberation short (this is a
        # sufficiency judgement, not a proof), and the floor guarantees room
        # for reasoning plus the actual answer.
        if self._is_reasoning_model():
            payload["reasoning_effort"] = "low"
            payload["max_tokens"] = max(max_tokens, 1024)

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.time()
        delay = 2.0
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if self.min_interval_s:
                    wait = self.min_interval_s - (time.time() - self._last_call)
                    if wait > 0:
                        time.sleep(wait)
                self._last_call = time.time()
                req = urllib.request.Request(
                    f"{self.API_BASE}/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers=self._headers())
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                msg = data["choices"][0]["message"]
                text = msg.get("content") or ""
                # Some reasoning models return an empty `content` and put
                # everything in `reasoning`. Fall back to it so a usable answer
                # buried in the reasoning trace is not thrown away.
                if not text.strip() and msg.get("reasoning"):
                    text = msg["reasoning"]
                return LLMResponse(text=text, model=self.model,
                                   elapsed_s=time.time() - t0, raw=data)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")[:400]
                last_err = f"HTTP {e.code}: {body}"

                # json_validate_failed: the model produced nothing usable inside
                # the token budget. Escalate rather than give up -- first buy
                # more room, then drop strict JSON mode entirely and let
                # LLMResponse.json() extract the object from free text (it
                # already handles fences and preamble). Only then fail.
                if e.code == 400 and "json_validate_failed" in body:
                    if payload.get("max_tokens", 0) < 3000:
                        payload["max_tokens"] = 3000
                        payload["reasoning_effort"] = "low"
                        continue
                    if "response_format" in payload:
                        payload.pop("response_format")
                        continue
                    raise RuntimeError(
                        f"Groq could not produce JSON for this prompt: {body}"
                    ) from e

                if e.code == 401:
                    raise RuntimeError(
                        "Groq rejected the API key (HTTP 401). Check that "
                        "GROQ_API_KEY is set correctly in THIS shell:\n"
                        "  echo $env:GROQ_API_KEY"
                    ) from e
                if e.code == 403 and "1010" in body:
                    raise RuntimeError(
                        "Groq returned Cloudflare error 1010 (client blocked). "
                        "This is a User-Agent problem, not an API key problem."
                    ) from e
                # 429 = rate limited, 5xx = transient. Both worth retrying;
                # 400/401 are our fault and retrying just wastes time.
                if e.code in (429, 500, 502, 503, 529) and attempt < self.max_retries - 1:
                    # Honour the server's own retry hint when it gives one --
                    # guessing with exponential backoff either wastes time or
                    # retries too early and burns another slot.
                    hint = None
                    m = re.search(r'try again in ([\d.]+)s', body)
                    if m:
                        hint = float(m.group(1)) + 0.5
                    elif e.headers and e.headers.get("retry-after"):
                        try:
                            hint = float(e.headers["retry-after"]) + 0.5
                        except ValueError:
                            hint = None
                    sleep_for = hint if hint is not None else delay

                    # DISTINGUISH per-minute throttling from a DAILY cap.
                    # A hint of a few seconds means "slow down" and is worth
                    # waiting for. A hint of minutes means the daily quota is
                    # gone and no amount of waiting inside this run helps.
                    # Without this check one call sat in backoff for 31 minutes
                    # and still failed -- far worse than failing immediately.
                    if hint is not None and hint > 90:
                        raise RuntimeError(
                            f"Groq quota exhausted -- it asks us to wait "
                            f"{hint:.0f}s ({hint/60:.0f} min). This is the daily "
                            f"limit, not per-minute throttling.\n"
                            f"  Options: wait for the reset, use a different "
                            f"account's key, or switch to Gemini:\n"
                            f"    python eval/run_pipeline.py --provider gemini "
                            f"--model gemini-flash-latest"
                        ) from e

                    if e.code == 429:
                        self.min_interval_s = max(self.min_interval_s * 1.5, 2.0)
                    time.sleep(min(sleep_for, 30))
                    delay *= 2
                    continue
                raise RuntimeError(f"Groq request failed: {last_err}") from e
            except urllib.error.URLError as e:
                last_err = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"Groq request failed: {last_err}") from e
        raise RuntimeError(f"Groq request failed after retries: {last_err}")

    def complete_json(self, prompt: str, system: str | None = None,
                      max_tokens: int = 512) -> LLMResponse:
        # Groq's json_object mode requires the word "JSON" to appear in the
        # prompt; without it the API rejects the request outright.
        if "json" not in prompt.lower() and not (system and "json" in system.lower()):
            prompt += "\n\nRespond with JSON only."
        return self.complete(prompt, system=system, max_tokens=max_tokens,
                             json_mode=True)


class GeminiClient:
    """Google Gemini via the Generative Language API.

    Useful here as a SECOND OPINION as much as a faster backend. Verifier #1 is
    an LLM judging sufficiency, and a single model's blind spots become the
    system's blind spots. Being able to re-run the same gate on a different
    model family tells us whether a verdict reflects the evidence or just one
    model's temperament -- and disagreement between them is itself a useful
    abstain signal.

    Free tier at time of writing is generous on flash models but rate-limited
    per minute; 429s are retried with backoff.

    Key comes from GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment. Never
    hardcode it, never commit it -- a key pasted into a chat, a commit, or a
    screenshot should be treated as burned and rotated.
    """

    API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    # `gemini-flash-latest` is an alias Google repoints at the current flash
    # model, so it survives the deprecations that broke gemini-2.5-flash for
    # new keys. Preferring the alias over a pinned version means one less thing
    # to maintain -- at the cost of the model changing under us between runs,
    # which is why cached verdicts are keyed on the resolved model name.
    PREFERRED = [
        "gemini-flash-latest",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-flash-lite-latest",
        "gemini-2.5-pro",
    ]

    # Listed by the API but not general-purpose text models. `list_models`
    # returns everything that supports generateContent, which includes image
    # generation, TTS, computer-use agents and the deep-research family --
    # all of which would happily accept a sufficiency prompt and return
    # something useless.
    NOT_CHAT = (
        "tts", "image", "computer-use", "deep-research",
        "antigravity", "embedding", "aqa", "learnlm",
    )

    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 timeout: int = 120, max_retries: int = 5):
        self.api_key = (os.environ.get("GEMINI_API_KEY")
                        or os.environ.get("GOOGLE_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set.\n"
                "  Free key: https://aistudio.google.com/apikey\n"
                "  $env:GEMINI_API_KEY='...'                       (this shell)\n"
                "  [Environment]::SetEnvironmentVariable("
                "'GEMINI_API_KEY','...','User')   (persistent)"
            )
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = model or self._auto_model()
        self.name = f"gemini:{self.model}"

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(
                f"{self.API_BASE}/models?key={self.api_key}",
                headers={"User-Agent": "analyst-copilot/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read())
            out = []
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    out.append(m["name"].replace("models/", ""))
            return sorted(out)
        except Exception:
            return []

    def _auto_model(self) -> str:
        available = [m for m in self.list_models()
                     if not any(x in m.lower() for x in self.NOT_CHAT)]
        if not available:
            return self.PREFERRED[0]
        for want in self.PREFERRED:
            for have in available:
                if have == want or have.startswith(want):
                    return have
        # Newest-looking first: names sort roughly by version, and a model too
        # new to be in PREFERRED is a better bet than one too old.
        return sorted(available, reverse=True)[0]

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 512, json_mode: bool = False) -> LLMResponse:
        cfg: dict[str, Any] = {"temperature": self.temperature,
                               "maxOutputTokens": max_tokens}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": cfg,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = f"{self.API_BASE}/models/{self.model}:generateContent?key={self.api_key}"
        t0 = time.time()
        delay = 2.0
        last_err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json",
                             "User-Agent": "analyst-copilot/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                cands = data.get("candidates", [])
                text = ""
                if cands:
                    for p in cands[0].get("content", {}).get("parts", []):
                        text += p.get("text", "")
                    # MAX_TOKENS with empty text = the budget was spent on
                    # thinking. Same failure mode as gpt-oss; retry with room.
                    if not text.strip() and cands[0].get("finishReason") == "MAX_TOKENS":
                        if cfg["maxOutputTokens"] < 4000:
                            cfg["maxOutputTokens"] = 4000
                            continue
                return LLMResponse(text=text, model=self.model,
                                   elapsed_s=time.time() - t0, raw=data)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")[:300]
                last_err = f"HTTP {e.code}: {body}"
                if e.code in (400, 403) and "API_KEY" in body.upper():
                    raise RuntimeError(
                        f"Gemini rejected the API key. {body[:160]}") from e

                # Google lists models that new keys cannot actually call
                # ("no longer available to new users"), and helpfully names the
                # replacement in the error. Follow that pointer once rather
                # than failing -- deprecations happen faster than we can keep
                # a hardcoded list current.
                if e.code == 404 and attempt < self.max_retries - 1:
                    m = re.search(r"use models/([\w.\-]+)", body)
                    if m and m.group(1) != self.model:
                        print(f"  [gemini] {self.model} unavailable, "
                              f"switching to {m.group(1)}")
                        self.model = m.group(1)
                        self.name = f"gemini:{self.model}"
                        url = (f"{self.API_BASE}/models/{self.model}"
                               f":generateContent?key={self.api_key}")
                        continue
                if e.code in (429, 500, 503) and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"Gemini request failed: {last_err}") from e
            except urllib.error.URLError as e:
                last_err = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"Gemini request failed: {last_err}") from e
        raise RuntimeError(f"Gemini request failed after retries: {last_err}")

    def complete_json(self, prompt: str, system: str | None = None,
                      max_tokens: int = 512) -> LLMResponse:
        return self.complete(prompt, system=system,
                             max_tokens=max(max_tokens, 800), json_mode=True)


class AnthropicClient:  # pragma: no cover - not used with Ollama
    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 temperature: float = 0.0):
        import anthropic
        self._c = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.name = f"anthropic:{model}"

    def complete(self, prompt, system=None, max_tokens=512, json_mode=False):
        t0 = time.time()
        kw = {"model": self.model, "max_tokens": max_tokens,
              "temperature": self.temperature,
              "messages": [{"role": "user", "content": prompt}]}
        if system:
            kw["system"] = system
        m = self._c.messages.create(**kw)
        return LLMResponse(text=m.content[0].text, model=self.model,
                           elapsed_s=time.time() - t0)

    def complete_json(self, prompt, system=None, max_tokens=512):
        return self.complete(prompt + "\n\nRespond with JSON only.",
                             system=system, max_tokens=max_tokens)


def get_client(provider: str | None = None, model: str | None = None):
    """Factory. Reads LLM_PROVIDER and LLM_MODEL when not given explicitly."""
    # Default to groq when a key is present -- local mistral measured 30.5s
    # per trivial call, which makes the iterate-and-measure loop impractical.
    if os.environ.get("GROQ_API_KEY"):
        default = "groq"
    elif os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        default = "gemini"
    else:
        default = "ollama"
    provider = (provider or os.environ.get("LLM_PROVIDER", default)).lower()
    model = model or os.environ.get("LLM_MODEL") or None
    if provider == "groq":
        return GroqClient(model=model)
    if provider == "gemini":
        return GeminiClient(model=model)
    if provider == "ollama":
        return OllamaClient(model=model)
    if provider == "mock":
        return MockClient()
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-haiku-4-5-20251001")
    raise ValueError(f"unknown provider: {provider}")


SMOKE_PROMPT = ('Answer with JSON only: {"ok": true, "n": 42}. Nothing else.')


def _smoke(c, label: str) -> float | None:
    print(f"\nsmoke test [{label}] ...")
    try:
        r = c.complete_json(SMOKE_PROMPT, max_tokens=64)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None
    print(f"  raw    : {r.text[:120]!r}")
    print(f"  parsed : {r.json()}")
    print(f"  time   : {r.elapsed_s:.1f}s")
    if r.json() is None:
        print("  WARNING: JSON parsing failed -- a stricter model may be needed.")
    if r.elapsed_s:
        print(f"  -> a 127-question sweep would take ~{127*r.elapsed_s/60:.0f} min")
    return r.elapsed_s


def check(include_ollama: bool = False) -> None:
    print("=" * 70)
    print("GROQ  (primary)")
    print("=" * 70)
    if not os.environ.get("GROQ_API_KEY"):
        print("  GROQ_API_KEY not set. Free key: https://console.groq.com/keys")
        print("    $env:GROQ_API_KEY='gsk_...'          (this shell)")
        print("    [Environment]::SetEnvironmentVariable("
              "'GROQ_API_KEY','gsk_...','User')   (persistent)")
    else:
        try:
            c = GroqClient()
            avail = c.list_models()
            chat = [m for m in avail
                    if not any(x in m.lower() for x in GroqClient.NOT_CHAT)]
            print(f"  {len(avail)} models visible, {len(chat)} usable for chat")
            for m in chat:
                print(f"  - {m}{'   <- selected' if m == c.model else ''}")
            skipped = [m for m in avail if m not in chat]
            if skipped:
                print(f"  (skipped {len(skipped)} non-chat: "
                      f"{', '.join(skipped[:4])}{' ...' if len(skipped) > 4 else ''})")
            _smoke(c, c.name)
        except Exception as e:
            print(f"  {e}")

    print("\n" + "=" * 70)
    print("GEMINI  (second opinion)")
    print("=" * 70)
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print("  GEMINI_API_KEY not set. Free key: https://aistudio.google.com/apikey")
        print("    $env:GEMINI_API_KEY='...'")
    else:
        try:
            c = GeminiClient()
            avail = c.list_models()
            chat = [m for m in avail
                    if not any(x in m.lower() for x in GeminiClient.NOT_CHAT)]
            print(f"  {len(avail)} support generateContent, {len(chat)} are text models")
            for m in chat[:14]:
                print(f"  - {m}{'   <- selected' if m == c.model else ''}")
            if len(chat) > 14:
                print(f"  ... and {len(chat)-14} more")
            _smoke(c, c.name)
        except Exception as e:
            print(f"  {e}")

    if include_ollama:
        print("\n" + "=" * 70)
        print("OLLAMA  (offline fallback)")
        print("=" * 70)
        probe = OllamaClient.__new__(OllamaClient)
        probe.host = OLLAMA_HOST
        models = probe.list_models()
        if not models:
            print(f"  no models at {OLLAMA_HOST} (is `ollama serve` running?)")
        else:
            for m in models:
                print(f"  - {m}")
            try:
                c = OllamaClient()
                print(f"  auto-selected: {c.model}")
                _smoke(c, c.name)
            except Exception as e:
                print(f"  {e}")

    print("\n" + "=" * 70)
    print("get_client() would use: ", end="")
    try:
        print(get_client().name)
    except Exception as e:
        print(f"(error) {e}")
    if not include_ollama:
        print("\n(--ollama also probes the local fallback)")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        check(include_ollama="--ollama" in sys.argv)
    else:
        print(__doc__)
        print("Run with --check (add --ollama to also probe local).")
