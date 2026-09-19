from src.chunking import chunk_paper, chunk_section


def test_chunk_section_respects_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_section("body", text)
    assert len(chunks) >= 2
    # overlap: the WORD_OVERLAP words at the end of chunk 0's window should
    # also appear at the start of chunk 1's window.
    words0 = set(chunks[0].text.split())
    words1 = set(chunks[1].text.split())
    assert words0 & words1


def test_chunk_paper_empty_sections_falls_back_to_full_text():
    chunks = chunk_paper({}, full_text_fallback="hello world " * 50)
    assert chunks
    assert chunks[0].section == "full_text"


def test_chunk_paper_multiple_sections_preserves_section_names():
    sections = {"abstract": "short text here", "method": "word " * 300}
    chunks = chunk_paper(sections)
    section_names = {c.section for c in chunks}
    assert section_names == {"abstract", "method"}
