"""
Node 3: pick the single best candidate to build a briefing for.

Design decision: the brief is explicit that a CLI-only interaction is fine
and a UI is out of scope, so rather than pausing execution to ask the user
to pick from a numbered list (which would need an interactive prompt mid
graph-run), we auto-rank and take the top result -- but we print the full
ranked candidate list to stdout first so the user can see what was
considered and re-run with a more specific query / exact arXiv id if the
top pick isn't what they wanted. This is a documented tradeoff in the
README ("what we'd do differently with more time": let the CLI accept
`--pick <n>` against a cached candidate list).

Ranking signal (topic mode only): cheap lexical overlap between the query
and (title, abstract) plus a small recency bonus -- no LLM call needed for
this step, keeping it fast and free.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from src.state import AgentState, PaperMeta

NODE = "selection_ranking"


def _score(query: str, paper: PaperMeta) -> float:
    q_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not q_terms:
        return 0.0
    text = f"{paper.title} {paper.abstract}".lower()
    text_terms = set(re.findall(r"[a-z0-9]+", text))
    overlap = len(q_terms & text_terms) / len(q_terms)

    recency_bonus = 0.0
    try:
        published = datetime.fromisoformat(paper.published.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - published).days
        recency_bonus = max(0.0, 1.0 - age_days / (365 * 3)) * 0.15  # small nudge, capped
    except Exception:
        pass

    return overlap + recency_bonus


def run(state: AgentState) -> AgentState:
    if not state.candidates:
        state.fail(NODE, "selection_ranking called with no candidates")
        return state

    if state.mode == "paper_id" or len(state.candidates) == 1:
        state.selected_paper = state.candidates[0]
        state.status = "selected"
        return state

    for paper in state.candidates:
        paper.score = _score(state.query, paper)
    ranked = sorted(state.candidates, key=lambda p: p.score, reverse=True)
    state.candidates = ranked
    state.selected_paper = ranked[0]
    state.status = "selected"
    return state
