"""End-to-end knowledge-base tests with a tiny deterministic embedder (no model download)."""

import hashlib
import io

import numpy as np
import pytest

from bhasharag.config import Settings
from bhasharag.knowledge_base import DuplicateDocumentError, KnowledgeBase, _clean_history
from bhasharag.text_utils import tokenize_for_bm25


class HashEmbedder:
    """Bag-of-words hashed into a small vector. Just enough semantics for tests."""

    model_name = "test-hash-embedder"
    dim = 64

    def count_tokens(self, text):
        return len(text.split())

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize_for_bm25(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    embed_queries = embed_passages


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, system, messages, max_tokens=1500):
        self.calls.append((system, messages))
        return "answer [1]"


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path, use_reranker=False, chunk_tokens=40, chunk_overlap_tokens=5)
    s.ensure_dirs()
    return s


def make_docx(paragraphs):
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


EN_DOC = (
    "PM Kisan provides income support of 6000 rupees per year to farmer families. "
    "The amount is paid in three equal instalments.\n\n"
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen."
).encode()
HI_DOC = (
    "प्रधानमंत्री किसान योजना के तहत किसानों को हर साल 6000 रुपये मिलते हैं। "
    "यह राशि तीन किस्तों में दी जाती है।"
).encode()


def test_add_search_delete_roundtrip(settings):
    kb = KnowledgeBase(settings, HashEmbedder())
    en = kb.add_document("schemes.txt", EN_DOC)
    hi = kb.add_document("yojana.txt", HI_DOC)
    assert en["language"] == "en" and hi["language"] == "hi"
    assert len(kb.list_documents()) == 2

    hits = kb.search("photosynthesis glucose")
    assert hits[0]["filename"] == "schemes.txt"
    assert "Photosynthesis" in hits[0]["text"]

    hits = kb.search("किसान योजना")
    assert hits[0]["filename"] == "yojana.txt"

    kb.delete_document(hi["id"])
    assert [d["filename"] for d in kb.list_documents()] == ["schemes.txt"]
    assert all(h["filename"] != "yojana.txt" for h in kb.search("किसान योजना"))
    assert len(kb.vectors) == kb.db.count_chunks()
    assert not list(settings.uploads_dir.glob(f"{hi['id']}*"))


def test_duplicate_upload_rejected(settings):
    kb = KnowledgeBase(settings, HashEmbedder())
    kb.add_document("a.txt", EN_DOC)
    with pytest.raises(DuplicateDocumentError):
        kb.add_document("copy-of-a.txt", EN_DOC)


def test_docx_and_doc_filter(settings):
    kb = KnowledgeBase(settings, HashEmbedder())
    a = kb.add_document("notes.docx", make_docx(["Mitochondria is the powerhouse of the cell."]))
    b = kb.add_document("other.txt", b"The mitochondria paper was published in 1957.")
    hits = kb.search("mitochondria", doc_ids=[a["id"]])
    assert hits and all(h["doc_id"] == a["id"] for h in hits)
    assert b["id"] not in {h["doc_id"] for h in hits}


def test_index_persists_and_rebuilds(settings):
    kb = KnowledgeBase(settings, HashEmbedder())
    kb.add_document("a.txt", EN_DOC)
    n = len(kb.vectors)
    kb.db.conn.close()

    reopened = KnowledgeBase(settings, HashEmbedder())
    assert len(reopened.vectors) == n
    reopened.db.conn.close()

    settings.index_path.unlink()  # simulate a lost/corrupt index
    rebuilt = KnowledgeBase(settings, HashEmbedder())
    assert len(rebuilt.vectors) == n
    assert rebuilt.search("photosynthesis")[0]["filename"] == "a.txt"


def test_answer_uses_llm_with_language_instruction(settings):
    llm = FakeLLM()
    kb = KnowledgeBase(settings, HashEmbedder(), llm=llm)
    kb.add_document("yojana.txt", HI_DOC)
    res = kb.answer("किसान योजना में कितने रुपये मिलते हैं?")
    assert res["language"] == "hi"
    assert res["answer"] == "answer [1]"
    system, messages = llm.calls[-1]
    assert "Hindi (Devanagari" in system
    assert "[1] (source: yojana.txt, page 1)" in messages[-1]["content"]


def test_answer_without_llm_returns_passages(settings):
    kb = KnowledgeBase(settings, HashEmbedder())
    assert "empty" in kb.answer("anything")["answer"]
    kb.add_document("a.txt", EN_DOC)
    res = kb.answer("How much does PM Kisan pay?")
    assert "6000" in res["answer"]


def test_clean_history_alternates_and_ends_with_assistant():
    h = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "q1"},
        {"role": "user", "content": "q1b"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "dangling"},
    ]
    assert _clean_history(h, max_turns=5) == [
        {"role": "user", "content": "q1\n\nq1b"},
        {"role": "assistant", "content": "a1"},
    ]
