"""
Simple, dependency-free chunker: split into paragraphs, then greedily pack
paragraphs into ~CHUNK_WORDS-word windows with WORD_OVERLAP words of overlap
carried into the next chunk (so a fact split across a paragraph boundary is
still retrievable from at least one chunk).

Word count is a good enough proxy for token count at this scale, and keeps
the module dependency-free (no tokenizer needed).
"""
from __future__ import annotations

import re
from typing import List

from src.state import Chunk

CHUNK_WORDS = 220
WORD_OVERLAP = 40


def chunk_section(section_name: str, text: str, start_order: int = 0) -> List[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: List[Chunk] = []
    i = 0
    order = start_order
    while i < len(words):
        window = words[i : i + CHUNK_WORDS]
        chunk_text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=f"{section_name}-{order}",
                text=chunk_text,
                section=section_name,
                order=order,
            )
        )
        order += 1
        if i + CHUNK_WORDS >= len(words):
            break
        i += CHUNK_WORDS - WORD_OVERLAP
    return chunks


def chunk_paper(sections: dict, full_text_fallback: str = "") -> List[Chunk]:
    all_chunks: List[Chunk] = []
    order = 0
    if sections:
        for name, text in sections.items():
            secs = chunk_section(name, text, start_order=order)
            order += len(secs)
            all_chunks.extend(secs)
    elif full_text_fallback:
        all_chunks.extend(chunk_section("full_text", full_text_fallback))
    return all_chunks
