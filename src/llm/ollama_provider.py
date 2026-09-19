"""
Fully local, fully free: talks to a locally running Ollama server.
Setup: install Ollama, then `ollama pull llama3.1` (or any model you like).

Env vars:
  OLLAMA_HOST    (default: http://localhost:11434)
  OLLAMA_MODEL   (default: llama3.1)
"""
from __future__ import annotations

import os

import requests

from src.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self.host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1")

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if json_mode:
            payload["format"] = "json"

        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
