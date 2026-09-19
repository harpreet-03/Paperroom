"""
Google AI Studio (Gemini) free tier via plain REST -- no google-generativeai
SDK dependency required.
Docs: https://ai.google.dev/gemini-api/docs

Env vars:
  GEMINI_API_KEY   (required, free at aistudio.google.com)
  GEMINI_MODEL     (default: gemini-1.5-flash)
"""
from __future__ import annotations

import os

import requests

from src.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self) -> None:
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Get a free key at aistudio.google.com.")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": 0.2},
        }
        if json_mode:
            payload["generationConfig"]["response_mime_type"] = "application/json"

        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 429:
            raise RuntimeError("Gemini free-tier rate limit hit -- wait a bit and retry.")
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
