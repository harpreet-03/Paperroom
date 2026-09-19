import os
from unittest.mock import patch

from src.pipeline import build_graph
from src.state import AgentState, PaperMeta

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _paper(arxiv_id="2401.12345", title="KV-Cache Compression for Efficient LLM Inference"):
    return PaperMeta(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Jane Doe"],
        abstract="We study techniques for compressing the key-value cache in transformers.",
        published="2024-01-15T00:00:00Z",
        updated="2024-01-15T00:00:00Z",
        categories=["cs.CL"],
        pdf_url=f"http://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"http://arxiv.org/abs/{arxiv_id}",
    )


def test_zero_candidates_routes_to_end_without_crashing():
    with patch("src.nodes.arxiv_retrieval.arxiv_client.search_topic", return_value=[]):
        graph = build_graph()
        state = graph.run(AgentState(query="a totally nonexistent research topic xyzzy"))
    assert state.status == "no_candidates"
    assert state.briefing is None


def test_many_candidates_picks_best_lexical_match():
    candidates = [_paper("2401.00001", "Unrelated Paper About Gardening Robots"),
                  _paper("2401.99999", "KV-Cache Compression Survey for LLMs")]
    with patch("src.nodes.arxiv_retrieval.arxiv_client.search_topic", return_value=candidates), \
         patch("src.nodes.fetch_parse.download_pdf", side_effect=RuntimeError("no network in test")), \
         patch("src.nodes.summarize.get_provider") as mock_provider:
        mock_provider.return_value.chat.return_value = (
            '{"summary": "s", "problem_statement": "p", "method": ["m"], '
            '"key_results": ["r"], "limitations": ["l1", "l2"], "suggested_questions": ["q?"]}'
        )
        graph = build_graph()
        state = graph.run(AgentState(query="KV-cache compression for LLMs"))

    # the survey paper should outrank the gardening-robots paper
    assert state.selected_paper.arxiv_id == "2401.99999"
    assert state.status == "done"
    assert state.briefing is not None
    assert len(state.briefing.limitations) >= 2


def test_pdf_download_failure_falls_back_to_abstract_and_still_produces_briefing():
    with patch("src.nodes.arxiv_retrieval.arxiv_client.search_topic", return_value=[_paper()]), \
         patch("src.nodes.fetch_parse.download_pdf", side_effect=RuntimeError("connection refused")), \
         patch("src.nodes.summarize.get_provider") as mock_provider:
        mock_provider.return_value.chat.return_value = (
            '{"summary": "s", "problem_statement": "p", "method": ["m"], '
            '"key_results": ["r"], "limitations": ["l1", "l2"], "suggested_questions": ["q?"]}'
        )
        graph = build_graph()
        state = graph.run(AgentState(query="KV-cache compression"))

    assert state.parsed.parsed_ok is False
    assert "download failed" in state.parsed.parse_warning.lower()
    assert state.briefing is not None
    assert state.status == "done"


def test_paper_id_mode_direct_lookup():
    with patch("src.nodes.arxiv_retrieval.arxiv_client.get_by_id", return_value=_paper()), \
         patch("src.nodes.fetch_parse.download_pdf", side_effect=RuntimeError("no network in test")), \
         patch("src.nodes.summarize.get_provider") as mock_provider:
        mock_provider.return_value.chat.return_value = (
            '{"summary": "s", "problem_statement": "p", "method": ["m"], '
            '"key_results": ["r"], "limitations": ["l1", "l2"], "suggested_questions": ["q?"]}'
        )
        graph = build_graph()
        state = graph.run(AgentState(query="2401.12345"))

    assert state.mode == "paper_id"
    assert state.selected_paper.arxiv_id == "2401.12345"
    assert state.status == "done"


def test_nonexistent_paper_id_routes_to_end():
    with patch("src.nodes.arxiv_retrieval.arxiv_client.get_by_id", return_value=None):
        graph = build_graph()
        state = graph.run(AgentState(query="9999.99999"))
    assert state.status == "no_candidates"
