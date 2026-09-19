"""
QA loop -- not wired into the linear graph (it runs interactively, once per
turn, after the briefing exists), but written as a node-shaped function for
consistency: (state, question) -> (state, answer).

Two answering paths:

1. Meta/summary questions ("summarize this", "what's this paper about",
   "tl;dr", "give me an overview") -- these are broad by design and share
   almost no vocabulary with any single passage in the paper, so TF-IDF
   retrieval systematically scores them low and they used to fall through
   the grounding gate and get refused even though the paper obviously *is*
   about something and summarize() already produced a grounded briefing
   for exactly this case. Detected questions skip retrieval entirely and
   answer from state.briefing instead -- still grounded (the briefing
   itself was built from retrieved context), just via a different, more
   appropriate context source for this question shape.
2. Everything else -- normal retrieve-then-answer RAG, gated by
   GROUNDING_THRESHOLD: if the best-matching chunk's similarity is below
   the bar, refuse rather than let the LLM guess. This is the cheapest
   possible anti-hallucination guardrail and is fully deterministic (no
   LLM call wasted on a doomed answer).
"""
from __future__ import annotations

import re

from src.llm.base import get_provider
from src.state import AgentState
from src.vectorstore.tfidf_store import TfidfStore

NODE = "qa"
TOP_K = 4
GROUNDING_THRESHOLD = 0.05  # tune per corpus; TF-IDF cosine sims are typically small

SYSTEM_PROMPT = (
    "You are answering questions about ONE specific research paper using "
    "only the excerpts provided below. If the excerpts do not contain the "
    "answer, say clearly that the paper (as retrieved) does not seem to "
    "cover it -- do not guess or use outside knowledge. Cite which excerpt "
    "section(s) you used, e.g. '(from: method)'."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are answering a broad question about a research paper (e.g. "
    "\"summarize this\" or \"what's it about\") using the structured "
    "briefing below, which was itself generated from the paper's abstract "
    "and retrieved sections. Answer conversationally in a few sentences to "
    "a paragraph -- don't just restate the briefing's bullet points "
    "verbatim, synthesize them into a natural answer to the specific "
    "question asked."
)

_META_PATTERNS = [
    r"\bsummar(y|ize|ise)\b",
    r"\btl;?dr\b",
    r"\bwhat.{0,15}\bpaper\b.{0,15}\babout\b",
    r"\bwhat.{0,10}(is|does)\s+this\s+(paper|study|work)\b",
    r"\boverview\b",
    r"\bmain\s+(idea|point|takeaway|contribution)s?\b",
    r"\bwhat.{0,10}\b(did|do)\s+they\s+(do|find|show)\b",
    r"\bexplain\s+(this|the)\s+paper\b",
    r"\bgive\s+me\s+a\s+(quick\s+)?(rundown|recap|summary)\b",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)


def _is_meta_summary_question(question: str) -> bool:
    return bool(_META_RE.search(question))


def _briefing_context(state: AgentState) -> str:
    b = state.briefing
    parts = [
        f"Title: {b.title}",
        f"Why it matters: {b.summary}",
        f"Problem statement: {b.problem_statement}",
        "Method: " + "; ".join(b.method),
        "Key results: " + "; ".join(b.key_results),
        "Limitations: " + "; ".join(b.limitations),
    ]
    return "\n".join(parts)


def _answer_from_briefing(state: AgentState, question: str) -> str:
    context = _briefing_context(state)
    provider = get_provider()
    try:
        answer_text = provider.chat(
            SUMMARY_SYSTEM_PROMPT, f"{context}\n\nQuestion: {question}", json_mode=False
        )
    except Exception:
        # LLM unavailable -- fall back to the briefing's own summary rather
        # than a bare error, since we already have a grounded answer on hand.
        b = state.briefing
        answer_text = f"{b.summary}\n\nProblem: {b.problem_statement}"
    return answer_text


def answer(state: AgentState, question: str) -> str:
    if state.collection_name is None:
        return "No paper has been loaded yet -- run the pipeline first."

    if state.briefing is not None and _is_meta_summary_question(question):
        answer_text = _answer_from_briefing(state, question)
        state.qa_history.append((question, answer_text))
        return answer_text

    store = TfidfStore(state.collection_name)
    store._load()  # noqa: SLF001
    hits = store.query(question, top_k=TOP_K)

    if not hits or hits[0][1] < GROUNDING_THRESHOLD:
        answer_text = (
            "I couldn't find anything in the retrieved text of this paper "
            "that answers that -- it may not be covered, or may be phrased "
            "very differently than the question. Try rephrasing, or note "
            "this may be beyond what was parsed from the PDF."
        )
        state.qa_history.append((question, answer_text))
        return answer_text

    context = "\n\n".join(f"[{c.section}] {c.text}" for c, _ in hits)
    user_prompt = f"Excerpts:\n{context}\n\nQuestion: {question}"

    provider = get_provider()
    try:
        answer_text = provider.chat(SYSTEM_PROMPT, user_prompt, json_mode=False)
    except Exception as e:
        answer_text = f"(LLM call failed: {e})"

    state.qa_history.append((question, answer_text))
    return answer_text