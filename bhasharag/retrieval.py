"""Lexical retrieval (BM25), rank fusion and cross-encoder reranking."""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict

import numpy as np

from .text_utils import tokenize_for_bm25

log = logging.getLogger(__name__)


class BM25Index:
    """In-memory Okapi BM25 over all chunks, with an inverted index so only chunks that
    share a term with the query are scored. Rebuilt lazily after the knowledge base changes.

    BM25 complements dense retrieval: it nails exact matches such as scheme names,
    section numbers and IDs that embeddings tend to blur.

    IDF uses the Lucene variant log(1 + (N - n + 0.5) / (n + 0.5)), which is always
    positive. The classic form log((N - n + 0.5) / (n + 0.5)) is zero or negative for any
    term present in half the chunks, so a knowledge base with only one or two
    documents would get no lexical matches at all.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_len: list[int] = []
        self._avgdl = 0.0

    def build(self, chunks: list[tuple[int, str]]) -> None:
        self._ids = [cid for cid, _ in chunks]
        self._postings = defaultdict(list)
        self._doc_len = []
        for idx, (_, text) in enumerate(chunks):
            tokens = tokenize_for_bm25(text)
            self._doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((idx, tf))
        self._postings = dict(self._postings)
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

    def search(self, query: str, k: int, allowed_ids: set[int] | None = None) -> list[tuple[int, float]]:
        if not self._ids or self._avgdl == 0:
            return []
        n_docs = len(self._ids)
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize_for_bm25(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for idx, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[idx] / self._avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        hits = [(self._ids[idx], s) for idx, s in ranked if allowed_ids is None or self._ids[idx] in allowed_ids]
        return hits[:k]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists without needing comparable scores (Cormack et al., 2009).

    score(d) = sum over lists of 1 / (k + rank(d)).
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


class Reranker:
    """Multilingual cross-encoder: reads (query, passage) together, so it is far more
    precise than bi-encoder similarity, but too slow to run over the whole corpus.
    We only apply it to the fused top candidates."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import CrossEncoder

        log.info("Loading reranker %s", model_name)
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)
        return [float(s) for s in np.atleast_1d(scores)]
