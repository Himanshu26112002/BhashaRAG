"""Pluggable LLM backends.

- "anthropic":          Claude via the official Anthropic SDK (needs ANTHROPIC_API_KEY)
- "ollama":             a local open model, e.g. qwen2.5:3b (free, offline)
- "openai_compatible":  any OpenAI-compatible endpoint, e.g. Groq's free tier
- "none":               no generation; the app returns the best-matching passages
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from .config import Settings

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "user" | "assistant", "content": "..."}


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def generate(self, system: str, messages: list[Message], max_tokens: int = 1500) -> str: ...


class AnthropicLLM:
    def __init__(self, model: str, effort: str = "medium"):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.effort = effort
        self.name = f"anthropic:{model}"

    def generate(self, system: str, messages: list[Message], max_tokens: int = 16000) -> str:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                # If a safety classifier declines, the API re-runs the request on a
                # recommended fallback model instead of returning a refusal.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self._anthropic.RateLimitError as exc:
            raise LLMError("Claude rate limit reached; please retry shortly.") from exc
        except self._anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Claude API.") from exc

        if response.stop_reason == "refusal":
            return "I can't help with that request."
        return "".join(block.text for block in response.content if block.type == "text").strip()


class OllamaLLM:
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"

    def generate(self, system: str, messages: list[Message], max_tokens: int = 1500) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": max_tokens, "num_ctx": 8192},
        }
        try:
            r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama request failed ({exc}). Is `ollama serve` running and `{self.model}` pulled?"
            ) from exc
        return r.json()["message"]["content"].strip()


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str):
        if not api_key:
            raise LLMError("OPENAI_API_KEY is required for LLM_PROVIDER=openai_compatible")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = f"openai_compatible:{model}"

    def generate(self, system: str, messages: list[Message], max_tokens: int = 1500) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"LLM API error {exc.response.status_code}: {exc.response.text[:300]}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        return r.json()["choices"][0]["message"]["content"].strip()


def get_llm(settings: Settings) -> LLM | None:
    provider = settings.llm_provider
    if provider in {"", "none"}:
        return None
    if provider == "anthropic":
        return AnthropicLLM(settings.anthropic_model, settings.anthropic_effort)
    if provider == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    if provider == "openai_compatible":
        return OpenAICompatibleLLM(settings.openai_base_url, settings.openai_api_key, settings.openai_model)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'")
