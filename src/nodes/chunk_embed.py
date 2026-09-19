"""Node 5: chunk parsed text and build/persist the local TF-IDF vector store."""
from __future__ import annotations

from src.chunking import chunk_paper
from src.state import AgentState
from src.vectorstore.tfidf_store import TfidfStore

NODE = "chunk_embed"


def run(state: AgentState) -> AgentState:
    if state.parsed is None or state.selected_paper is None:
        state.fail(NODE, "chunk_embed called before fetch_parse")
        return state

    sections = state.parsed.sections if state.parsed.parsed_ok else {}
    chunks = chunk_paper(sections, full_text_fallback=state.parsed.full_text)

    if not chunks:
        state.fail(NODE, "no text available to chunk (empty abstract + empty parse)")
        return state

    collection_name = state.selected_paper.arxiv_id.replace("/", "_")
    store = TfidfStore(collection_name)
    store.build(chunks)

    state.chunks = chunks
    state.collection_name = collection_name
    state.status = "embedded"
    return state
