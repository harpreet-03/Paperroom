"""
Shared state object threaded through every node of the agent graph.

This is a plain dataclass (not a dict) so the "state shape" required by the
assessment brief is documented in one place and typo-checkable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    updated: str
    categories: List[str]
    pdf_url: str
    abs_url: str
    score: float = 0.0  # relevance score assigned during ranking


@dataclass
class ParsedPaper:
    full_text: str
    sections: Dict[str, str] = field(default_factory=dict)
    references_raw: str = ""
    parsed_ok: bool = True
    parse_warning: Optional[str] = None
    num_pages: int = 0


@dataclass
class Chunk:
    chunk_id: str
    text: str
    section: str
    order: int


@dataclass
class Briefing:
    title: str
    authors: List[str]
    arxiv_id: str
    published: str
    link: str
    summary: str
    problem_statement: str
    method: List[str]
    key_results: List[str]
    limitations: List[str]
    suggested_questions: List[str]


@dataclass
class AgentState:
    # ---- input ----
    query: str

    # ---- query understanding ----
    mode: Optional[str] = None  # "paper_id" | "topic"

    # ---- retrieval ----
    candidates: List[PaperMeta] = field(default_factory=list)
    selected_paper: Optional[PaperMeta] = None

    # ---- fetch & parse ----
    pdf_path: Optional[str] = None
    parsed: Optional[ParsedPaper] = None

    # ---- chunk & embed ----
    chunks: List[Chunk] = field(default_factory=list)
    collection_name: Optional[str] = None

    # ---- summarize ----
    briefing: Optional[Briefing] = None

    # ---- QA loop (grows across the interactive session) ----
    qa_history: List[Tuple[str, str]] = field(default_factory=list)

    # ---- control flow / diagnostics ----
    status: str = "init"          # init -> ... -> done | error
    errors: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)  # ordered list of node names visited

    def log(self, node_name: str) -> None:
        self.trace.append(node_name)

    def fail(self, node_name: str, message: str) -> None:
        self.errors.append(f"[{node_name}] {message}")
        self.status = "error"
