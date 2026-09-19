"""
Local, on-disk vector store backed by TF-IDF + cosine similarity.

Design decision (documented in README too): a from-scratch RAG assessment
does not need a production ANN index for a single paper's ~40-150 chunks --
brute-force cosine similarity over a TF-IDF matrix is exact, needs zero
downloaded model weights, is fully deterministic (good for tests), and
still gives sensible lexical grounding for QA over technical text where
exact terminology matters (e.g. "KV-cache", "LoRA rank").

Swap-in path to a denser embedding (sentence-transformers -> Chroma/FAISS)
is isolated to this one file; nothing else in the graph depends on how
similarity is computed. See README "Design Decisions" for the tradeoff.
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.state import Chunk

STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "vectorstore")


class TfidfStore:
    def __init__(self, collection_name: str, store_dir: str = STORE_DIR) -> None:
        self.collection_name = collection_name
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        self._path = os.path.join(self.store_dir, f"{collection_name}.pkl")
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.chunks: List[Chunk] = []

    # ---- build ----
    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=20000
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self._persist()

    # ---- query ----
    def query(self, text: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        if self.vectorizer is None:
            self._load()
        if self.vectorizer is None or self.matrix is None or not self.chunks:
            return []
        q_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(sims[i])) for i in ranked]

    # ---- persistence ----
    def _persist(self) -> None:
        with open(self._path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                    "chunks": [asdict(c) for c in self.chunks],
                },
                f,
            )

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        with open(self._path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.chunks = [Chunk(**c) for c in data["chunks"]]

    def exists(self) -> bool:
        return os.path.exists(self._path)
