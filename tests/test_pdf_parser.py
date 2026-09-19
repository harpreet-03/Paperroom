import pymupdf as fitz

from src.pdf_parser import parse_pdf


def _make_pdf(path, text=None):
    doc = fitz.open()
    page = doc.new_page()
    if text:
        page.insert_textbox(fitz.Rect(36, 36, 559, 800), text, fontsize=9)
    doc.save(path)
    doc.close()


def test_parse_normal_pdf_extracts_sections(tmp_path):
    path = str(tmp_path / "paper.pdf")
    body = "Abstract\n" + ("This paper studies things in great detail. " * 40)
    body += "\nIntroduction\n" + ("More text about the topic and its background. " * 40)
    _make_pdf(path, body)
    parsed = parse_pdf(path, fallback_abstract="fallback")
    assert parsed.parsed_ok is True
    assert parsed.num_pages == 1


def test_parse_blank_scanned_pdf_falls_back(tmp_path):
    path = str(tmp_path / "scanned.pdf")
    _make_pdf(path, text=None)  # no extractable text -> simulates a scanned page
    parsed = parse_pdf(path, fallback_abstract="the abstract text")
    assert parsed.parsed_ok is False
    assert parsed.full_text == "the abstract text"
    assert "scanned" in parsed.parse_warning.lower()


def test_parse_corrupted_file_path(tmp_path):
    path = str(tmp_path / "not_a_pdf.pdf")
    with open(path, "w") as f:
        f.write("this is not a pdf")
    parsed = parse_pdf(path, fallback_abstract="fallback abstract")
    assert parsed.parsed_ok is False
    assert parsed.full_text == "fallback abstract"
