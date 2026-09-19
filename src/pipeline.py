"""
Wires the nodes from src/nodes/* into the StateGraph and exposes
build_graph() + session save/load helpers.

Graph shape:

    query_understanding
            |
    arxiv_retrieval ----(no candidates)----> END (friendly message)
            |
      (have candidates)
            v
    selection_ranking
            |
      fetch_parse  ----(unrecoverable)-----> END (friendly message)
            |
      (parsed_ok | parsed_degraded, both continue -- degraded just
       means "summarize from the abstract only" rather than aborting)
            v
      chunk_embed  ----(no text at all)-----> END (friendly message)
            |
            v
       summarize  --------------------------> END (briefing ready)

QA is intentionally NOT a graph node: it's an interactive loop the CLI
drives after the graph finishes, reusing the persisted vector store
(state.collection_name) and the in-memory state object. See README
"state persistence" section.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Optional

from src.graph import END, StateGraph
from src.nodes import arxiv_retrieval, chunk_embed, fetch_parse, query_understanding, selection_ranking, summarize
from src.state import AgentState

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sessions")


def build_graph() -> StateGraph:
    g = StateGraph()
    g.add_node(query_understanding.NODE, query_understanding.run)
    g.add_node(arxiv_retrieval.NODE, arxiv_retrieval.run)
    g.add_node(selection_ranking.NODE, selection_ranking.run)
    g.add_node(fetch_parse.NODE, fetch_parse.run)
    g.add_node(chunk_embed.NODE, chunk_embed.run)
    g.add_node(summarize.NODE, summarize.run)

    g.set_entry(query_understanding.NODE)
    g.add_edge(query_understanding.NODE, arxiv_retrieval.NODE)

    g.add_conditional_edges(
        arxiv_retrieval.NODE,
        lambda s: selection_ranking.NODE if s.status == "have_candidates" else END,
    )
    g.add_edge(selection_ranking.NODE, fetch_parse.NODE)

    g.add_conditional_edges(
        fetch_parse.NODE,
        lambda s: chunk_embed.NODE if s.status in ("parsed_ok", "parsed_degraded") else END,
    )
    g.add_conditional_edges(
        chunk_embed.NODE,
        lambda s: summarize.NODE if s.status == "embedded" else END,
    )
    g.add_edge(summarize.NODE, END)

    return g


def run_pipeline(query: str, verbose: bool = False) -> AgentState:
    state = AgentState(query=query)
    graph = build_graph()
    return graph.run(state, verbose=verbose)


# ---------------------------------------------------------------------------
# Session persistence: how state survives between "summarize" and the QA
# loop / across separate CLI invocations. We keep the *live* Python object
# in-memory for the current process (fastest path, used by the interactive
# REPL) and additionally write a small JSON session file so a user can
# `--resume <arxiv_id>` later without re-running retrieval/parsing/embedding
# -- only the vector store (already persisted to disk by TfidfStore) plus
# this JSON are needed to rebuild enough state to keep asking questions.
# ---------------------------------------------------------------------------

def save_session(state: AgentState) -> Optional[str]:
    if state.briefing is None or state.collection_name is None:
        return None
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = os.path.join(SESSIONS_DIR, f"{state.collection_name}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "query": state.query,
                "collection_name": state.collection_name,
                "briefing": asdict(state.briefing),
                "qa_history": state.qa_history,
            },
            f,
            indent=2,
        )
    return path


def load_session(arxiv_id: str) -> Optional[AgentState]:
    path = os.path.join(SESSIONS_DIR, f"{arxiv_id.replace('/', '_')}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    from src.state import Briefing

    state = AgentState(query=data["query"])
    state.collection_name = data["collection_name"]
    state.briefing = Briefing(**data["briefing"])
    state.qa_history = [tuple(x) for x in data.get("qa_history", [])]
    state.status = "done"
    return state
