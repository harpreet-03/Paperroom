"""Node 4: download the PDF and extract text/sections."""
from __future__ import annotations

import os

from src.pdf_parser import download_pdf, parse_pdf
from src.state import AgentState

NODE = "fetch_parse"
PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pdfs")


def run(state: AgentState) -> AgentState:
    paper = state.selected_paper
    if paper is None:
        state.fail(NODE, "no selected_paper to fetch")
        return state

    try:
        pdf_path = download_pdf(paper.pdf_url, PDF_DIR, paper.arxiv_id)
        state.pdf_path = pdf_path
    except Exception as e:
        # Can't even download -> fall back to abstract-only summarization
        # rather than aborting the whole run.
        state.parsed = _abstract_only(paper.abstract, f"PDF download failed: {e}")
        state.status = "parsed_degraded"
        return state

    parsed = parse_pdf(pdf_path, fallback_abstract=paper.abstract)
    state.parsed = parsed
    state.status = "parsed_ok" if parsed.parsed_ok else "parsed_degraded"
    return state


def _abstract_only(abstract: str, warning: str):
    from src.state import ParsedPaper
    return ParsedPaper(full_text=abstract, parsed_ok=False, parse_warning=warning)
