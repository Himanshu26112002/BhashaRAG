"""FAISS vector index keyed by chunk id, so individual documents can be deleted."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int, path: Path | None = None):
        self.dim = dim
        self.path = path
        if path is not None and path.exists():
            # Read via Python instead of faiss.read_index: FAISS's C++ file API can fail
            # on Windows paths containing non-ASCII characters.
            buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            self.index = faiss.deserialize_index(buf)
            if self.index.d != dim:
                raise ValueError(
                    f"Index on disk has dimension {self.index.d}, embedder has {dim}. "
                    "The embedding model changed; the index must be rebuilt."
                )
        else:
            self.index = self._new_index()

    def _new_index(self) -> faiss.Index:
        # Exact inner-product search. For >1M chunks swap in IndexHNSWFlat or IVF.
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids:
            return
        self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), np.asarray(ids, dtype=np.int64))

    def remove(self, ids: list[int]) -> None:
        if ids:
            self.index.remove_ids(np.asarray(ids, dtype=np.int64))

    def reset(self) -> None:
        self.index = self._new_index()

    def search(self, query_vector: np.ndarray, k: int, allowed_ids: set[int] | None = None) -> list[tuple[int, float]]:
        if len(self) == 0 or k <= 0:
            return []
        # With a document filter, over-fetch and filter. Flat search is exhaustive anyway.
        fetch = len(self) if allowed_ids is not None else min(k, len(self))
        scores, ids = self.index.search(np.asarray(query_vector, dtype=np.float32).reshape(1, -1), fetch)
        hits = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
        if allowed_ids is not None:
            hits = [h for h in hits if h[0] in allowed_ids]
        return hits[:k]

    def save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(faiss.serialize_index(self.index).tobytes())
        tmp.replace(self.path)  # atomic on the same filesystem
