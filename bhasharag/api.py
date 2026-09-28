"""HTTP API + web UI.

Run with:  python -m uvicorn bhasharag.api:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT, get_settings
from .knowledge_base import DocumentProcessingError, DuplicateDocumentError, KnowledgeBase
from .llm import LLMError
from .loaders import UnsupportedFileType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Hugging Face Hub checks every model file online at startup; keep those requests out of the log.
for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
STATIC_DIR = PROJECT_ROOT / "static"

kb: KnowledgeBase | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global kb
    # Load models once at startup so the first request isn't slow.
    kb = KnowledgeBase.from_settings(get_settings())
    yield


app = FastAPI(title="BhashaRAG", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _kb() -> KnowledgeBase:
    if kb is None:
        raise HTTPException(503, "Knowledge base is still loading")
    return kb


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = []
    doc_ids: list[str] | None = None  # restrict search to these documents
    top_k: int | None = Field(default=None, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    doc_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", **_kb().stats()}


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return _kb().list_documents()


# Plain `def` endpoints run in FastAPI's threadpool, so CPU-heavy embedding work
# does not block the event loop.
@app.post("/api/documents")
def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    results = []
    for f in files:
        name = Path(f.filename or "document").name
        try:
            doc = _kb().add_document(name, f.file.read())
            results.append({"filename": name, "status": "added", "document": doc})
        except DuplicateDocumentError as exc:
            results.append({"filename": name, "status": "duplicate", "error": str(exc), "document": exc.existing})
        except (UnsupportedFileType, DocumentProcessingError) as exc:
            results.append({"filename": name, "status": "error", "error": str(exc)})
        except Exception as exc:  # a bad file must not abort the rest of the batch
            logging.exception("Failed to ingest %s", name)
            results.append({"filename": name, "status": "error", "error": f"Could not process file: {exc}"})
    return {"results": results}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict:
    try:
        return {"deleted": _kb().delete_document(doc_id)}
    except KeyError:
        raise HTTPException(404, "Document not found")


@app.get("/api/documents/{doc_id}/chunks")
def document_chunks(doc_id: str) -> list[dict]:
    try:
        return _kb().get_document_chunks(doc_id)
    except KeyError:
        raise HTTPException(404, "Document not found")


@app.post("/api/search")
def search(req: SearchRequest) -> list[dict]:
    return _kb().search(req.query, top_k=req.top_k, doc_ids=req.doc_ids)


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    try:
        return _kb().answer(
            req.question,
            history=[m.model_dump() for m in req.history],
            doc_ids=req.doc_ids,
            top_k=req.top_k,
        )
    except LLMError as exc:
        raise HTTPException(502, str(exc))
