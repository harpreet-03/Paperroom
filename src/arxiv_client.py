"""
Thin wrapper around the official arXiv API (Atom feed) -- no scraping.
Docs: https://info.arxiv.org/help/api/user-manual.html

Only stdlib (xml.etree) + requests are used, no extra parsing dependency.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests

from src.state import PaperMeta

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE
)


def extract_arxiv_id(text: str) -> Optional[str]:
    """Pulls a bare arXiv id (e.g. '2401.12345') out of an id, a URL, or free text."""
    m = ARXIV_ID_RE.search(text.strip())
    return m.group(1) if m else None


def _parse_entry(entry: ET.Element) -> PaperMeta:
    def t(tag: str) -> str:
        el = entry.find(f"atom:{tag}", NS)
        return el.text.strip() if el is not None and el.text else ""

    raw_id = t("id")  # e.g. http://arxiv.org/abs/2401.12345v2
    arxiv_id = extract_arxiv_id(raw_id) or raw_id

    authors = [
        (a.find("atom:name", NS).text or "").strip()
        for a in entry.findall("atom:author", NS)
        if a.find("atom:name", NS) is not None
    ]
    categories = [
        c.attrib.get("term", "")
        for c in entry.findall("atom:category", NS)
        if c.attrib.get("term")
    ]
    pdf_url = ""
    for link in entry.findall("atom:link", NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = link.attrib.get("href", "")
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    return PaperMeta(
        arxiv_id=arxiv_id,
        title=re.sub(r"\s+", " ", t("title")).strip(),
        authors=authors,
        abstract=re.sub(r"\s+", " ", t("summary")).strip(),
        published=t("published"),
        updated=t("updated"),
        categories=categories,
        pdf_url=pdf_url,
        abs_url=raw_id,
    )


def search_topic(topic: str, max_results: int = 8, timeout: int = 15) -> List[PaperMeta]:
    """Free-text topic search, most-relevant first."""
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = requests.get(ARXIV_API, params=params, timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    return [_parse_entry(e) for e in root.findall("atom:entry", NS)]


def get_by_id(arxiv_id: str, timeout: int = 15) -> Optional[PaperMeta]:
    """Direct lookup by arXiv id."""
    params = {"id_list": arxiv_id}
    resp = requests.get(ARXIV_API, params=params, timeout=timeout)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", NS)
    if not entries:
        return None
    meta = _parse_entry(entries[0])
    # A nonexistent id still returns a single stub entry whose <title> is empty.
    if not meta.title:
        return None
    return meta
