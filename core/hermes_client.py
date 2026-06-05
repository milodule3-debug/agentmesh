"""
AgentMesh · core/hermes_client.py
Cloud-based multi-provider client — no local GPU needed.

Provider options:
  OPENROUTER  — serves actual Hermes 3 models. Recommended.
                Free $1 credit on signup. Cheapest Hermes: ~$0.001/1K tokens.
                https://openrouter.ai  → API Keys → Create Key

  GROQ        — 100% free tier. Llama 3.1 (same base as Hermes). Very fast.
                No Hermes available but great for sub-agents on a budget.
                https://console.groq.com → API Keys

  ANTHROPIC   — Claude models. You already have access via claude.ai.
                Best for orchestrator-level reasoning.
                https://console.anthropic.com → API Keys

Usage:
    from core.hermes_client import HermesClient, Provider

    # Hermes 3 via OpenRouter (recommended)
    client = HermesClient(provider=Provider.OPENROUTER, api_key="sk-or-...")

    # Free via Groq
    client = HermesClient(provider=Provider.GROQ, api_key="gsk_...")

    # Claude for orchestrator
    client = HermesClient(provider=Provider.ANTHROPIC, api_key="sk-ant-...")

    resp = client.chat([{"role": "user", "content": "Hello"}])
    print(resp.content)
"""

from __future__ import annotations
import json
import os
import time
import requests
from typing import Optional, Iterator
from dataclasses import dataclass
from enum import Enum


# ── Providers ─────────────────────────────────────────────────────────────────

class Provider(str, Enum):
    OPENROUTER = "openrouter"
    GROQ       = "groq"
    ANTHROPIC  = "anthropic"
    DEEPSEEK   = "deepseek"
    GEMINI     = "gemini"
    TOGETHER   = "together"


# ── Model catalogue ───────────────────────────────────────────────────────────

MODELS = {
    Provider.OPENROUTER: {
        "hermes-70b": "nousresearch/hermes-3-llama-3.1-70b",
        "hermes-8b":  "nousresearch/hermes-3-llama-3.1-8b",
        "default":    "nousresearch/hermes-3-llama-3.1-70b",
        "free":       "meta-llama/llama-3.1-8b-instruct:free",
    },
    Provider.GROQ: {
        "fast":    "llama-3.1-8b-instant",
        "smart":   "llama-3.3-70b-versatile",
        "default": "llama-3.1-8b-instant",
    },
    Provider.ANTHROPIC: {
        "fast":    "claude-haiku-4-5-20251001",
        "smart":   "claude-sonnet-4-6",
        "default": "claude-haiku-4-5-20251001",
    },
    Provider.DEEPSEEK: {
        "chat":     "deepseek-chat",
        "reasoner": "deepseek-reasoner",
        "default":  "deepseek-chat",
    },
    # Gemini — OpenAI-compatible endpoint, strong multimodal
    Provider.GEMINI: {
        "flash":   "gemini-2.0-flash",
        "pro":     "gemini-1.5-pro-latest",
        "fast":    "gemini-1.5-flash-latest",
        "default": "gemini-2.0-flash",
    },
    # Together AI — open source models, Llama/Mixtral/Qwen
    Provider.TOGETHER: {
        "llama405": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "llama70":  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "mixtral":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "qwen":     "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "default":  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    },
}

ENDPOINTS = {
    Provider.OPENROUTER: "https://openrouter.ai/api/v1/chat/completions",
    Provider.GROQ:       "https://api.groq.com/openai/v1/chat/completions",
    Provider.ANTHROPIC:  "https://api.anthropic.com/v1/messages",
    Provider.DEEPSEEK:   "https://api.deepseek.com/v1/chat/completions",
    Provider.GEMINI:     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    Provider.TOGETHER:   "https://api.together.xyz/v1/chat/completions",
}


# ── Response object ───────────────────────────────────────────────────────────

@dataclass
class HermesResponse:
    content: str
    tool_calls: list[dict]
    tokens_in: int
    tokens_out: int
    model: str
    provider: str
    elapsed_ms: int
    raw: dict

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def has_tool_call(self) -> bool:
        return len(self.tool_calls) > 0

    def first_tool(self) -> Optional[dict]:
        return self.tool_calls[0] if self.tool_calls else None

    def cost_estimate_usd(self) -> float:
        """Rough cost. OpenRouter charges vary; this is a conservative estimate."""
        # ~$0.001 per 1K tokens for Hermes 8B on OpenRouter
        return (self.total_tokens / 1000) * 0.001


# ── Main client ───────────────────────────────────────────────────────────────

class HermesClient:
    """
    Unified cloud client for AgentMesh sub-agents and orchestrator.
    Defaults to OpenRouter + Hermes 3 70B.
    Falls back gracefully to Groq free tier.
    """

    def __init__(
        self,
        provider: Provider = Provider.OPENROUTER,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.2,
        default_max_tokens: int = 2048,
    ):
        self.provider = Provider(provider)
        self.api_key = api_key or os.environ.get(self._env_key())
        if not self.api_key:
            raise HermesError(
                f"No API key for {self.provider}. "
                f"Set env var {self._env_key()} or pass api_key=..."
            )
        self.model = model or MODELS[self.provider]["default"]
        self.temperature = temperature
        self.default_max_tokens = default_max_tokens
        self._session = requests.Session()

    # ── Core chat ─────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] = None,
        max_tokens: int = None,
    ) -> HermesResponse:
        t0 = time.time()

        if self.provider == Provider.ANTHROPIC:
            data = self._call_anthropic(messages, system, tools, max_tokens)
        else:
            data = self._call_openai_compat(messages, system, tools, max_tokens)

        elapsed = int((time.time() - t0) * 1000)
        return self._parse(data, elapsed)

    def complete(self, prompt: str, system: str = "", max_tokens: int = None) -> str:
        """Convenience: single prompt → string reply."""
        resp = self.chat(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens,
        )
        return resp.content

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        max_tokens: int = None,
    ) -> HermesResponse:
        return self.chat(messages, system=system, tools=tools, max_tokens=max_tokens)

    def stream(self, messages: list[dict], system: str = "") -> Iterator[str]:
        """Stream tokens (OpenAI-compat providers only)."""
        if self.provider == Provider.ANTHROPIC:
            yield self.chat(messages, system).content
            return

        payload = self._build_openai_payload(messages, system)
        payload["stream"] = True

        headers = self._headers()
        with self._session.post(
            ENDPOINTS[self.provider],
            json=payload, headers=headers, stream=True, timeout=120
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line and line.startswith(b"data: "):
                    chunk_str = line[6:].decode()
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token
                    except Exception:
                        pass

    # ── OpenAI-compatible call (OpenRouter + Groq) ────────────────────────────

    def _call_openai_compat(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tokens: int,
    ) -> dict:
        payload = self._build_openai_payload(messages, system, tools, max_tokens)
        headers = self._headers()

        r = self._session.post(
            ENDPOINTS[self.provider],
            json=payload, headers=headers, timeout=120
        )
        if not r.ok:
            raise HermesError(
                f"{self.provider} API error {r.status_code}: {r.text[:300]}"
            )
        return r.json()

    def _build_openai_payload(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] = None,
        max_tokens: int = None,
    ) -> dict:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # OpenRouter extras (optional but good practice)
        if self.provider == Provider.OPENROUTER:
            payload["transforms"] = []   # disable prompt transform

        return payload

    # ── Anthropic call ────────────────────────────────────────────────────────

    def _call_anthropic(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict],
        max_tokens: int,
    ) -> dict:
        # Convert OpenAI tool format → Anthropic format
        ant_tools = []
        if tools:
            for t in tools:
                fn = t.get("function", t)
                ant_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })

        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max_tokens,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        if ant_tools:
            payload["tools"] = ant_tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "anthropic-dangerous-direct-browser-access": "true",
        }
        r = self._session.post(
            ENDPOINTS[Provider.ANTHROPIC],
            json=payload, headers=headers, timeout=120
        )
        if not r.ok:
            raise HermesError(f"Anthropic API error {r.status_code}: {r.text[:300]}")
        return r.json()

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse(self, data: dict, elapsed_ms: int) -> HermesResponse:
        if self.provider == Provider.ANTHROPIC:
            return self._parse_anthropic(data, elapsed_ms)
        return self._parse_openai(data, elapsed_ms)

    def _parse_openai(self, data: dict, elapsed_ms: int) -> HermesResponse:
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""

        # Tool calls
        tool_calls = []
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            tool_calls.append({"name": fn.get("name", ""), "arguments": args})

        # Hermes 3 sometimes wraps tool calls in <tool_call> tags in content
        if not tool_calls and content and "<tool_call>" in content:
            tool_calls = self._parse_hermes_tags(content)

        usage = data.get("usage", {})
        return HermesResponse(
            content=content,
            tool_calls=tool_calls,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            model=data.get("model", self.model),
            provider=self.provider,
            elapsed_ms=elapsed_ms,
            raw=data,
        )

    def _parse_anthropic(self, data: dict, elapsed_ms: int) -> HermesResponse:
        content_blocks = data.get("content", [])
        text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")

        tool_calls = []
        for b in content_blocks:
            if b.get("type") == "tool_use":
                tool_calls.append({
                    "name": b.get("name", ""),
                    "arguments": b.get("input", {}),
                })

        usage = data.get("usage", {})
        return HermesResponse(
            content=text,
            tool_calls=tool_calls,
            tokens_in=usage.get("input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            model=data.get("model", self.model),
            provider=self.provider,
            elapsed_ms=elapsed_ms,
            raw=data,
        )

    @staticmethod
    def _parse_hermes_tags(content: str) -> list[dict]:
        """Parse Hermes 3 native <tool_call>...</tool_call> XML."""
        import re
        results = []
        for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
            try:
                d = json.loads(match.group(1).strip())
                results.append({"name": d.get("name", ""), "arguments": d.get("arguments", d)})
            except Exception:
                pass
        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        if self.provider == Provider.OPENROUTER:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://leanprogressiq.com",
                "X-Title": "AgentMesh",
                "Content-Type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _env_key(self) -> str:
        return {
            Provider.OPENROUTER: "OPENROUTER_API_KEY",
            Provider.GROQ:       "GROQ_API_KEY",
            Provider.ANTHROPIC:  "ANTHROPIC_API_KEY",
            Provider.DEEPSEEK:   "DEEPSEEK_API_KEY",
            Provider.GEMINI:     "GEMINI_API_KEY",
            Provider.TOGETHER:   "TOGETHER_API_KEY",
        }[self.provider]

    def count_tokens(self, text: str) -> int:
        """Rough estimate: 4 chars ≈ 1 token."""
        return len(text) // 4

    def is_alive(self) -> bool:
        """Quick connectivity check."""
        check_urls = {
            Provider.OPENROUTER: "https://openrouter.ai/api/v1/models",
            Provider.GROQ:       "https://api.groq.com/openai/v1/models",
            Provider.DEEPSEEK:   "https://api.deepseek.com/v1/models",
            Provider.GEMINI:     "https://generativelanguage.googleapis.com/v1beta/openai/models",
            Provider.TOGETHER:   "https://api.together.xyz/v1/models",
            Provider.ANTHROPIC:  "https://api.anthropic.com/v1/models",
        }
        url = check_urls.get(self.provider, "https://api.groq.com/openai/v1/models")
        try:
            r = self._session.get(url, headers=self._headers(), timeout=5)
            return r.ok
        except Exception:
            return False


class HermesError(Exception):
    pass


# ── Multi-agent client pool ───────────────────────────────────────────────────

class ClientPool:
    """
    One client per agent — lets different agents use different providers/models.
    Orchestrator → Anthropic Claude (best reasoning)
    Sub-agents   → OpenRouter Hermes 8B (cheap) or Groq (free)
    """

    def __init__(
        self,
        default_provider: Provider = Provider.OPENROUTER,
        default_api_key: str = None,
    ):
        self.default_provider = default_provider
        self.default_api_key = default_api_key or os.environ.get(
            {Provider.OPENROUTER: "OPENROUTER_API_KEY",
             Provider.GROQ: "GROQ_API_KEY",
             Provider.ANTHROPIC: "ANTHROPIC_API_KEY"}[default_provider]
        )
        self._clients: dict[str, HermesClient] = {}

    def get(self, agent_id: str, **kwargs) -> HermesClient:
        if agent_id not in self._clients:
            self._clients[agent_id] = HermesClient(
                provider=kwargs.pop("provider", self.default_provider),
                api_key=kwargs.pop("api_key", self.default_api_key),
                **kwargs,
            )
        return self._clients[agent_id]

    def add(self, agent_id: str, client: HermesClient) -> None:
        """Register a pre-configured client for a specific agent."""
        self._clients[agent_id] = client
