"""Node 1: decide whether the input is a specific paper id/URL or a topic."""
from __future__ import annotations

from src.arxiv_client import extract_arxiv_id
from src.state import AgentState

NODE = "query_understanding"


def run(state: AgentState) -> AgentState:
    arxiv_id = extract_arxiv_id(state.query)
    if arxiv_id:
        state.mode = "paper_id"
    else:
        state.mode = "topic"
    return state
