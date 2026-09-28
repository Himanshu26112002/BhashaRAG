"""The knowledge base: add / delete documents, hybrid search, and grounded answering."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunker import chunk_pages
from .config import Settings
from .db import MetadataDB
from .llm import LLM, LLMError, Message, get_llm
from .loaders import SUPPORTED_EXTENSIONS, UnsupportedFileType, load_document
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    CONTEXT_TEMPLATE,
    LANGUAGE_INSTRUCTIONS,
    REWRITE_SYSTEM_PROMPT,
    format_passages,
    no_llm_answer,
)
from .retrieval import BM25Index, Reranker, reciprocal_rank_fusion
from .text_utils import detect_language
from .vector_store import VectorStore

log = logging.getLogger(__name__)


class EmbedderLike(Protocol):
    model_name: str
    dim: int

    def count_tokens(self, text: str) -> int: ...
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


class RerankerLike(Protocol):
    model_name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class DuplicateDocumentError(ValueError):
    def __init__(self, existing: dict):
        super().__init__(f"'{existing['filename']}' is already in the knowledge base")
        self.existing = existing


class DocumentProcessingError(ValueError):
    pass


def _doc_to_dict(row) -> dict:
    d = dict(row)
    d["warnings"] = [w for w in d.get("warnings", "").split("\n") if w]
    d.pop("stored_path", None)
    return d


class KnowledgeBase:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbedderLike,
        reranker: RerankerLike | None = None,
        llm: LLM | None = None,
    ):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm
        self.db = MetadataDB(settings.db_path)
        self._lock = threading.RLock()
        self._bm25 = BM25Index()
        self._bm25_dirty = True
        self.vectors = self._open_vector_store()

    @classmethod
    def from_settings(cls, settings: Settings) -> "KnowledgeBase":
        from .embedder import Embedder

        embedder = Embedder(settings.embedding_model, settings.embedding_batch_size)
        reranker = Reranker(settings.reranker_model) if settings.use_reranker else None
        return cls(settings, embedder, reranker, get_llm(settings))

    # ------------------------------------------------------------------ index sync
    def _open_vector_store(self) -> VectorStore:
        """Load the FAISS index, rebuilding it if it is missing, stale, or was built
        with a different embedding model (e.g. after switching to a fine-tuned one)."""
        path = self.settings.index_path
        try:
            store = VectorStore(self.embedder.dim, path)
        except ValueError as exc:
            log.warning("%s Rebuilding.", exc)
            store = VectorStore(self.embedder.dim, None)
            store.path = path
        built_with = self.db.get_meta("embedding_model")
        if built_with != self.embedder.model_name or len(store) != self.db.count_chunks():
            log.info("Vector index out of date (built with %s, %d/%d vectors); re-embedding all chunks.",
                     built_with, len(store), self.db.count_chunks())
            self._rebuild_index(store)
        return store

    def _rebuild_index(self, store: VectorStore) -> None:
        store.reset()
        rows = self.db.all_chunks()
        batch = 256
        for start in range(0, len(rows), batch):
            part = rows[start:start + batch]
            store.add([r["id"] for r in part], self.embedder.embed_passages([r["text"] for r in part]))
        store.save()
        self.db.set_meta("embedding_model", self.embedder.model_name)

    def rebuild_index(self) -> None:
        with self._lock:
            self._rebuild_index(self.vectors)

    # ------------------------------------------------------------------ documents
    def list_documents(self) -> list[dict]:
        return [_doc_to_dict(r) for r in self.db.list_documents()]

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        if self.db.get_document(doc_id) is None:
            raise KeyError(doc_id)
        return [dict(r) for r in self.db.chunks_for_document(doc_id)]

    def add_document(self, filename: str, data: bytes) -> dict:
        filename = Path(filename).name or "document"
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileType(
                f"Unsupported file type '{ext or filename}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise DocumentProcessingError(f"File is larger than {self.settings.max_upload_mb} MB")

        sha256 = hashlib.sha256(data).hexdigest()
        existing = self.db.find_by_hash(sha256)
        if existing is not None:
            raise DuplicateDocumentError(_doc_to_dict(existing))

        # Heavy work (parsing, embedding) happens outside the lock so searches keep working.
        loaded = load_document(
            filename, data, enable_ocr=self.settings.enable_ocr, tesseract_cmd=self.settings.tesseract_cmd
        )
        chunks = chunk_pages(
            loaded.pages,
            self.embedder.count_tokens,
            max_tokens=self.settings.chunk_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise DocumentProcessingError(" ".join(loaded.warnings) or "No text found in document")
        vectors = self.embedder.embed_passages([c.text for c in chunks])

        doc_id = uuid.uuid4().hex
        stored_path = self.settings.uploads_dir / f"{doc_id}{ext}"
        stored_path.write_bytes(data)
        language = _dominant_language([(c.language, c.token_count) for c in chunks])

        try:
            with self._lock:
                if self.db.find_by_hash(sha256) is not None:  # lost a race with a parallel upload
                    raise DuplicateDocumentError(_doc_to_dict(self.db.find_by_hash(sha256)))
                with self.db.transaction() as conn:
                    self.db.insert_document(
                        conn,
                        id=doc_id,
                        filename=filename,
                        sha256=sha256,
                        stored_path=str(stored_path),
                        language=language,
                        num_pages=len(loaded.pages),
                        num_chunks=len(chunks),
                        size_bytes=len(data),
                        warnings="\n".join(loaded.warnings),
                    )
                    ids = [
                        self.db.insert_chunk(conn, doc_id, i, c.page, c.language, c.token_count, c.text)
                        for i, c in enumerate(chunks)
                    ]
                    # Inside the transaction: if FAISS fails, the SQLite rows roll back.
                    self.vectors.add(ids, vectors)
                self.vectors.save()
                self._bm25_dirty = True
        except Exception:
            stored_path.unlink(missing_ok=True)
            raise
        log.info("Added %s (%d chunks, %s)", filename, len(chunks), language)
        return _doc_to_dict(self.db.get_document(doc_id))

    def delete_document(self, doc_id: str) -> dict:
        with self._lock:
            row = self.db.get_document(doc_id)
            if row is None:
                raise KeyError(doc_id)
            chunk_ids = self.db.chunk_ids_for_document(doc_id)
            with self.db.transaction() as conn:
                self.db.delete_document(conn, doc_id)  # chunks cascade
                self.vectors.remove(chunk_ids)
            self.vectors.save()
            self._bm25_dirty = True
        Path(row["stored_path"]).unlink(missing_ok=True)
        log.info("Deleted %s (%d chunks)", row["filename"], len(chunk_ids))
        return _doc_to_dict(row)

    # ------------------------------------------------------------------ retrieval
    def _ensure_bm25(self) -> None:
        if self._bm25_dirty:
            self._bm25.build([(r["id"], r["text"]) for r in self.db.all_chunks()])
            self._bm25_dirty = False

    def search(self, query: str, top_k: int | None = None, doc_ids: list[str] | None = None) -> list[dict]:
        """Hybrid retrieval: dense (multilingual embeddings) + BM25, fused with RRF,
        then reranked with a cross-encoder."""
        top_k = top_k or self.settings.top_k
        n = max(self.settings.candidates_per_retriever, top_k)
        query_vector = self.embedder.embed_queries([query])[0]

        with self._lock:
            self._ensure_bm25()
            allowed = None
            if doc_ids:
                allowed = {cid for d in doc_ids for cid in self.db.chunk_ids_for_document(d)}
            dense = self.vectors.search(query_vector, n, allowed)
            lexical = self._bm25.search(query, n, allowed)
            fused = reciprocal_rank_fusion([[cid for cid, _ in dense], [cid for cid, _ in lexical]])[:n]
            rows = self.db.get_chunks([cid for cid, _ in fused])

        dense_scores = dict(dense)
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense, start=1)}
        bm25_rank = {cid: r for r, (cid, _) in enumerate(lexical, start=1)}
        hits = []
        for cid, rrf in fused:
            row = rows.get(cid)
            if row is None:
                continue
            hits.append({
                "chunk_id": cid,
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "page": row["page"],
                "language": row["language"],
                "text": row["text"],
                "dense_score": dense_scores.get(cid),
                "dense_rank": dense_rank.get(cid),
                "bm25_rank": bm25_rank.get(cid),
                "rrf_score": rrf,
                "rerank_score": None,
            })

        if self.reranker is not None and hits:
            for hit, score in zip(hits, self.reranker.score(query, [h["text"] for h in hits])):
                hit["rerank_score"] = score
            hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        return hits[:top_k]

    # ------------------------------------------------------------------ answering
    def answer(
        self,
        question: str,
        history: list[Message] | None = None,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict:
        question = question.strip()
        language = detect_language(question)
        history = _clean_history(history or [], self.settings.max_history_turns)
        timings: dict[str, float] = {}

        search_query = question
        if history and self.llm is not None and self.settings.rewrite_followups:
            t = time.perf_counter()
            search_query = self._rewrite_followup(question, history)
            timings["rewrite_ms"] = round((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        hits = self.search(search_query, top_k=top_k, doc_ids=doc_ids)
        timings["retrieval_ms"] = round((time.perf_counter() - t) * 1000)

        if self.db.count_chunks() == 0:
            answer = (
                "आपका नॉलेज बेस अभी खाली है। कृपया पहले कोई दस्तावेज़ अपलोड करें।"
                if language == "hi"
                else "Your knowledge base is empty. Upload a document first."
            )
        elif self.llm is None:
            answer = no_llm_answer(hits, language)
        else:
            system = ANSWER_SYSTEM_PROMPT.format(
                language_instruction=LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])
            )
            user_turn = CONTEXT_TEMPLATE.format(
                passages=format_passages(hits) if hits else "(no relevant passages found)",
                question=question,
            )
            t = time.perf_counter()
            answer = self.llm.generate(system, [*history, {"role": "user", "content": user_turn}])
            timings["generation_ms"] = round((time.perf_counter() - t) * 1000)

        return {
            "answer": answer,
            "language": language,
            "search_query": search_query,
            "sources": hits,
            "timings": timings,
            "llm": self.llm.name if self.llm else None,
        }

    def _rewrite_followup(self, question: str, history: list[Message]) -> str:
        transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-4:])
        prompt = f"Conversation so far:\n{transcript}\n\nLatest message: {question}"
        try:
            rewritten = self.llm.generate(REWRITE_SYSTEM_PROMPT, [{"role": "user", "content": prompt}], max_tokens=2000)
        except LLMError as exc:
            log.warning("Follow-up rewrite failed, using raw question: %s", exc)
            return question
        rewritten = rewritten.strip().strip('"').splitlines()[0] if rewritten.strip() else ""
        return rewritten or question

    def stats(self) -> dict:
        return {
            "documents": len(self.db.list_documents()),
            "chunks": self.db.count_chunks(),
            "embedding_model": self.embedder.model_name,
            "reranker": self.reranker.model_name if self.reranker else None,
            "llm": self.llm.name if self.llm else None,
        }


def _dominant_language(parts: list[tuple[str, int]]) -> str:
    weights: Counter[str] = Counter()
    for lang, tokens in parts:
        weights[lang] += tokens
    total = sum(weights.values()) or 1
    hi = (weights["hi"] + weights["mixed"]) / total
    if hi >= 0.8:
        return "hi"
    if hi >= 0.2:
        return "mixed"
    return "en"


def _clean_history(history: list[Message], max_turns: int) -> list[Message]:
    """Keep the last N turns, and make roles strictly alternate starting with 'user'
    (the Anthropic API requires this)."""
    cleaned: list[Message] = []
    for m in history:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1] = {"role": role, "content": cleaned[-1]["content"] + "\n\n" + content}
        else:
            cleaned.append({"role": role, "content": content})
    cleaned = cleaned[-max_turns * 2:]
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    # The next message we send is a user turn, so history must end with the assistant.
    while cleaned and cleaned[-1]["role"] != "assistant":
        cleaned.pop()
    return cleaned
