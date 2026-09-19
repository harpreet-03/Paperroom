"""
Groq free-tier chat completions (OpenAI-compatible REST API, no SDK needed).
Docs: https://console.groq.com/docs/quickstart

Model default: openai/gpt-oss-120b. Groq deprecated llama-3.1-8b-instant and
llama-3.3-70b-versatile on 2026-08-16 (see console.groq.com/docs/deprecations)
-- they now 404 with "model_not_found". gpt-oss-120b/gpt-oss-20b and the
qwen3.6 family are their current recommended free-tier replacements; check
your Groq console for what's live if this 404s again in the future.

Env vars:
  GROQ_API_KEY   (required)
  GROQ_MODEL     (default: openai/gpt-oss-120b)
"""
from __future__ import annotations

import os

import requests

from src.llm.base import LLMProvider

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(LLMProvider):
    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Get a free key at console.groq.com.")
        self.model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        if resp.status_code == 429:
            raise RuntimeError("Groq rate limit hit (free tier) -- wait a bit and retry.")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]