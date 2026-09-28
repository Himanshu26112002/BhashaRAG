"""Multilingual sentence embeddings (Hindi and English land in the same vector space)."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around a SentenceTransformer.

    E5-family models were trained with "query: " / "passage: " prefixes and lose several
    points of recall without them, so they are added automatically.
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s", model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.max_seq_length = self.model.max_seq_length
        uses_e5_prefixes = "e5" in model_name.lower()
        self.query_prefix = "query: " if uses_e5_prefixes else ""
        self.passage_prefix = "passage: " if uses_e5_prefixes else ""

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # cosine similarity == inner product
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return vectors.astype(np.float32)
