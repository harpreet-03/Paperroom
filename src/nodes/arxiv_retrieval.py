"""
Node 2: call the arXiv API.

Failure case handled here (§5): "what happens when arXiv returns zero or
many candidate papers for a vague topic?"
  - zero candidates for a topic -> status="no_candidates", graph routes to
    a friendly end state instead of crashing downstream nodes.
  - many candidates -> all are kept on state.candidates and handed to the
    selection_ranking node, which is where narrowing-to-one happens.
  - paper_id mode with an id that doesn't exist -> same "no_candidates" path.
"""
from __future__ import annotations

from src import arxiv_client
from src.state import AgentState

NODE = "arxiv_retrieval"
MAX_TOPIC_RESULTS = 8


def run(state: AgentState) -> AgentState:
    try:
        if state.mode == "paper_id":
            arxiv_id = arxiv_client.extract_arxiv_id(state.query)
            meta = arxiv_client.get_by_id(arxiv_id)
            state.candidates = [meta] if meta else []
        else:
            state.candidates = arxiv_client.search_topic(state.query, max_results=MAX_TOPIC_RESULTS)
    except Exception as e:
        state.fail(NODE, f"arXiv API request failed: {e}")
        return state

    if not state.candidates:
        state.status = "no_candidates"
    else:
        state.status = "have_candidates"
    return state
