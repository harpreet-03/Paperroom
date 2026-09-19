from unittest.mock import patch

from src.nodes import qa
from src.state import AgentState, Briefing, Chunk
from src.vectorstore.tfidf_store import TfidfStore


def _seeded_state(tmp_path, monkeypatch):
    monkeypatch.setattr("src.nodes.qa.TfidfStore", lambda name: TfidfStore(name, store_dir=str(tmp_path)))
    store = TfidfStore("qa-test", store_dir=str(tmp_path))
    store.build([
        Chunk("c0", "the model uses a transformer encoder with 12 layers", "method", 0),
        Chunk("c1", "we evaluate on the GLUE benchmark and report accuracy", "results", 1),
    ])
    state = AgentState(query="q")
    state.collection_name = "qa-test"
    return state


def test_qa_grounded_answer_uses_llm(tmp_path, monkeypatch):
    state = _seeded_state(tmp_path, monkeypatch)
    with patch("src.nodes.qa.get_provider") as mock_provider:
        mock_provider.return_value.chat.return_value = "It uses a 12-layer transformer encoder."
        result = qa.answer(state, "how many layers does the model have?")
    assert "12-layer" in result
    assert state.qa_history[-1][0] == "how many layers does the model have?"


def test_qa_refuses_when_similarity_too_low(tmp_path, monkeypatch):
    state = _seeded_state(tmp_path, monkeypatch)
    with patch("src.nodes.qa.get_provider") as mock_provider:
        result = qa.answer(state, "what is the capital of France?")
    mock_provider.return_value.chat.assert_not_called()
    assert "couldn't find" in result.lower()


def test_qa_without_loaded_paper():
    state = AgentState(query="q")
    result = qa.answer(state, "anything")
    assert "no paper" in result.lower()


def test_qa_uses_grounded_briefing_for_summary_request(tmp_path, monkeypatch):
    state = _seeded_state(tmp_path, monkeypatch)
    state.briefing = Briefing(
        title="A paper", authors=[], arxiv_id="x", published="", link="",
        summary="The paper studies a grounded retrieval system.",
        problem_statement="Reliable answers need source evidence.",
        method=["It retrieves relevant passages."],
        key_results=["It reduces unsupported answers."],
        limitations=[], suggested_questions=[],
    )
    with patch("src.nodes.qa.get_provider") as mock_provider:
        result = qa.answer(state, "summarise the paper")
    mock_provider.return_value.chat.assert_not_called()
    assert "grounded briefing" in result.lower()
    assert "retrieval system" in result
