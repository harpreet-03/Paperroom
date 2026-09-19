from src.state import Chunk
from src.vectorstore.tfidf_store import TfidfStore


def test_tfidf_store_query_ranks_relevant_chunk_first(tmp_path):
    chunks = [
        Chunk("c0", "the cat sat on the mat", "body", 0),
        Chunk("c1", "transformers use attention mechanisms for sequence modeling", "body", 1),
        Chunk("c2", "gardening tips for growing tomatoes at home", "body", 2),
    ]
    store = TfidfStore("test-collection", store_dir=str(tmp_path))
    store.build(chunks)

    results = store.query("how does attention work in transformers", top_k=2)
    assert results[0][0].chunk_id == "c1"


def test_tfidf_store_persists_and_reloads(tmp_path):
    chunks = [Chunk("c0", "reinforcement learning reward shaping", "body", 0)]
    store = TfidfStore("persist-test", store_dir=str(tmp_path))
    store.build(chunks)

    reloaded = TfidfStore("persist-test", store_dir=str(tmp_path))
    results = reloaded.query("reward shaping", top_k=1)
    assert results and results[0][0].chunk_id == "c0"
