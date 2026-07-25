from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer("all-mpnet-base-v2")


class Embedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return _model().encode(texts, normalize_embeddings=True)

    def cosine(self, a, b) -> float:
        return float(np.dot(a, b))  # already normalized
