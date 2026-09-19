"""
PDF download + text/section extraction using PyMuPDF (fitz).

Handles the "PDF fails to parse cleanly" failure case (§5 of the brief):
  - scanned/image-only PDFs -> very little extractable text -> parsed_ok=False,
    caller falls back to the arXiv abstract for summarization.
  - huge papers -> we cap how much text we keep (MAX_CHARS) rather than
    failing outright, and note the truncation.
  - corrupted/unreadable file -> caught, parsed_ok=False, warning recorded.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import requests

from src.state import ParsedPaper

MAX_CHARS = 120_000          # guard against pathologically long PDFs
MIN_CHARS_FOR_OK = 500        # below this, treat as "didn't really parse"
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024

SECTION_HEADERS = [
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach",
    "experiments", "experimental setup", "results", "evaluation",
    "discussion", "limitations", "conclusion", "conclusions",
    "references", "acknowledgements", "acknowledgments", "appendix",
]
_HEADER_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(" + "|".join(SECTION_HEADERS) + r")\s*$",
    re.IGNORECASE,
)


def download_pdf(url: str, dest_dir: str, arxiv_id: str, timeout: int = 30) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{arxiv_id.replace('/', '_')}.pdf")
    temp_path = f"{dest_path}.part"
    headers = {"User-Agent": "arxiv-digest-agent/1.0"}
    with requests.get(url, timeout=timeout, headers=headers, stream=True) as resp:
        resp.raise_for_status()
        content_length = int(resp.headers.get("content-length", 0))
        if content_length > MAX_DOWNLOAD_BYTES:
            raise ValueError("PDF is larger than the 50 MB download limit")
        downloaded = 0
        with open(temp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_BYTES:
                    raise ValueError("PDF exceeded the 50 MB download limit")
                f.write(chunk)
    with open(temp_path, "rb") as f:
        if not f.read(1024).lstrip().startswith(b"%PDF"):
            raise ValueError("Downloaded response is not a valid PDF")
    os.replace(temp_path, dest_path)
    return dest_path


def parse_pdf(pdf_path: str, fallback_abstract: str = "") -> ParsedPaper:
    try:
        import pymupdf as fitz
    except ImportError as e:
        return ParsedPaper(
            full_text=fallback_abstract,
            parsed_ok=False,
            parse_warning=f"PyMuPDF not installed ({e}); using abstract only.",
        )

    try:
        doc = fitz.open(pdf_path)
        num_pages = doc.page_count
        raw_pages = [page.get_text("text") for page in doc]
        doc.close()
    except Exception as e:  # corrupted / unreadable file
        return ParsedPaper(
            full_text=fallback_abstract,
            parsed_ok=False,
            parse_warning=f"Failed to open/parse PDF ({e}); using abstract only.",
            num_pages=0,
        )

    full_text = "\n".join(raw_pages)
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    if len(full_text) < MIN_CHARS_FOR_OK:
        # Almost certainly a scanned / image-only paper -- no OCR in scope.
        return ParsedPaper(
            full_text=fallback_abstract or full_text,
            parsed_ok=False,
            parse_warning=(
                "Extracted text was implausibly short "
                f"({len(full_text)} chars) -- likely a scanned/image PDF. "
                "Falling back to the arXiv abstract."
            ),
            num_pages=num_pages,
        )

    truncated = False
    if len(full_text) > MAX_CHARS:
        full_text = full_text[:MAX_CHARS]
        truncated = True

    sections = _split_sections(full_text)
    references_raw = sections.pop("references", "")

    warning = "Truncated to first ~120k characters (large paper)." if truncated else None

    return ParsedPaper(
        full_text=full_text,
        sections=sections,
        references_raw=references_raw,
        parsed_ok=True,
        parse_warning=warning,
        num_pages=num_pages,
    )


def _split_sections(text: str) -> dict:
    """Best-effort heuristic split on lines that look like section headers.

    Academic PDFs have no reliable structural markup once flattened to text,
    so this is intentionally simple: a short line matching a known header
    name starts a new section. Good enough for chunking/summarization;
    not attempting a general-purpose layout parser.
    """
    lines = text.split("\n")
    sections: dict = {}
    current = "front_matter"
    buf: list = []
    for line in lines:
        m = _HEADER_RE.match(line.strip())
        if m and len(line.strip()) < 40:
            sections[current] = "\n".join(buf).strip()
            current = m.group(1).lower()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return {k: v for k, v in sections.items() if v}
