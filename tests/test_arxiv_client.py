import os

from src import arxiv_client

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _read(name):
    with open(os.path.join(FIXTURES, name)) as f:
        return f.read()


def test_extract_arxiv_id_from_bare_id():
    assert arxiv_client.extract_arxiv_id("2401.12345") == "2401.12345"


def test_extract_arxiv_id_from_url():
    assert arxiv_client.extract_arxiv_id("https://arxiv.org/abs/2401.12345v2") == "2401.12345"


def test_extract_arxiv_id_from_topic_returns_none():
    assert arxiv_client.extract_arxiv_id("recent work on KV-cache compression") is None


def test_parse_single_entry():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(_read("arxiv_single.xml"))
    entry = root.find("atom:entry", arxiv_client.NS)
    meta = arxiv_client._parse_entry(entry)
    assert meta.arxiv_id == "2401.12345"
    assert "KV-Cache" in meta.title
    assert meta.authors == ["Jane Doe", "John Smith"]
    assert meta.pdf_url.endswith("2401.12345v2")


def test_parse_empty_feed_yields_no_entries():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(_read("arxiv_empty.xml"))
    entries = root.findall("atom:entry", arxiv_client.NS)
    assert entries == []
