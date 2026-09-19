"""
Node 6: produce the structured executive briefing.

Grounding strategy for this step: rather than stuffing the whole paper into
one giant prompt, we feed the LLM (a) the abstract + intro/conclusion
sections verbatim (usually where the "why it matters" and "limitations"
framing already live) and (b) the top TF-IDF-retrieved chunks for a couple
of fixed probe queries ("main contribution and method", "limitations and
future work", "key results"). This keeps the prompt bounded regardless of
paper length and re-uses the same retrieval path the QA loop uses, so
summarize() and qa() are grounded the same way.
"""
from __future__ import annotations

import json
import re

from src.llm.base import get_provider
from src.state import AgentState, Briefing
from src.vectorstore.tfidf_store import TfidfStore

NODE = "summarize"

PROBE_QUERIES = [
    "main contribution, problem statement and method",
    "key results, experiments and claims",
    "limitations, weaknesses and future work",
]

SYSTEM_PROMPT = (
    "You are a careful research-paper analyst. You only use the provided "
    "context; you never invent numbers, claims, or citations that are not "
    "present in it. If the context does not mention limitations explicitly, "
    "infer plausible ones cautiously and label them as inferred. Respond "
    "with a single JSON object and nothing else -- no markdown fences, no "
    "commentary."
)

JSON_SCHEMA_HINT = """
Return exactly this JSON shape:
{
  "summary": "<one paragraph, plain English, why this paper matters>",
  "problem_statement": "<1-3 sentences>",
  "method": ["<bullet>", "..."],
  "key_results": ["<bullet>", "..."],
  "limitations": ["<bullet>", "... at least 2 items, be explicit, do not skip this>"],
  "suggested_questions": ["<question a reader might ask>", "... 3-5 items"]
}
"""

QUESTION_QUALITY_HINT = """
Suggested questions must be specific, useful follow-ups grounded in the supplied
paper context. Ask about a named method, evaluation, result, limitation, or
claim from this paper. Do not use vague questions such as "What is this paper
about?", and do not invent dataset names, metrics, or results.
"""


def _gather_context(state: AgentState, store: TfidfStore) -> str:
    parts = []
    if state.parsed and state.parsed.sections.get("abstract"):
        parts.append("## Abstract\n" + state.parsed.sections["abstract"])
    elif state.selected_paper:
        parts.append("## Abstract\n" + state.selected_paper.abstract)

    for pq in PROBE_QUERIES:
        hits = store.query(pq, top_k=3)
        if hits:
            block = "\n".join(f"- ({c.section}) {c.text[:600]}" for c, _ in hits)
            parts.append(f"## Retrieved context for: {pq}\n{block}")

    return "\n\n".join(parts)[:16000]  # bound prompt size


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def run(state: AgentState) -> AgentState:
    if state.selected_paper is None or state.collection_name is None:
        state.fail(NODE, "summarize called before chunk_embed")
        return state

    store = TfidfStore(state.collection_name)
    store._load()  # noqa: SLF001 -- intentional reuse of persisted vectors

    context = _gather_context(state, store)
    paper = state.selected_paper

    user_prompt = (
        f"Paper title: {paper.title}\n"
        f"Authors: {', '.join(paper.authors)}\n\n"
        f"{context}\n\n{JSON_SCHEMA_HINT}\n{QUESTION_QUALITY_HINT}"
    )

    provider = get_provider()
    try:
        raw = provider.chat(SYSTEM_PROMPT, user_prompt, json_mode=True)
        data = _extract_json(raw)
    except Exception as e:
        state.fail(NODE, f"LLM summarization failed: {e}")
        return state

    try:
        briefing = Briefing(
            title=paper.title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            published=paper.published,
            link=paper.abs_url or f"https://arxiv.org/abs/{paper.arxiv_id}",
            summary=data.get("summary", ""),
            problem_statement=data.get("problem_statement", ""),
            method=list(data.get("method", [])),
            key_results=list(data.get("key_results", [])),
            limitations=list(data.get("limitations", [])),
            suggested_questions=list(data.get("suggested_questions", [])),
        )
    except Exception as e:
        state.fail(NODE, f"malformed briefing JSON from LLM: {e}")
        return state

    state.briefing = briefing
    state.status = "done"
    return state
