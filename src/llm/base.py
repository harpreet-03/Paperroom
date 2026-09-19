"""
Minimal LLM provider abstraction so the graph nodes never talk to a
specific vendor SDK directly. Every provider exposes one method:

    chat(system: str, user: str, json_mode: bool = False) -> str

Providers are picked via the LLM_PROVIDER env var: "groq" | "gemini" | "ollama".
All three have a genuinely free way to run (Groq free tier, Gemini free
tier, or a fully local Ollama model) -- see README for setup + rate limits.
"""
from __future__ import annotations

import abc
import os


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        ...


class StubProvider(LLMProvider):
    """Deterministic offline provider used by the test suite and for
    dry-running the pipeline without any API key / network access."""

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        if json_mode:
            return (
                '{"summary": "[stub summary -- no LLM configured]", '
                '"problem_statement": "[stub]", "method": ["[stub]"], '
                '"key_results": ["[stub]"], "limitations": ["[stub]"], '
                '"suggested_questions": ["[stub]?"]}'
            )
        return "[stub answer -- no LLM configured; set LLM_PROVIDER + an API key]"


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or os.environ.get("LLM_PROVIDER", "stub")).lower()
    if name == "groq":
        from src.llm.groq_provider import GroqProvider
        return GroqProvider()
    if name == "gemini":
        from src.llm.gemini_provider import GeminiProvider
        return GeminiProvider()
    if name == "ollama":
        from src.llm.ollama_provider import OllamaProvider
        return OllamaProvider()
    return StubProvider()
