"""Token-aware, sentence-preserving chunking.

Chunks are sized with the *embedding model's own tokenizer*. Character-based chunking
is misleading for Hindi: multilingual tokenizers often spend 2-4x more tokens per word
on Devanagari than on English, so a "1000 character" Hindi chunk can overflow the
model's 512-token window and get silently truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .loaders import Page
from .text_utils import detect_language, split_sentences

TokenCounter = Callable[[str], int]


@dataclass
class Chunk:
    page: int
    text: str
    token_count: int
    language: str


def chunk_pages(
    pages: list[Page],
    count_tokens: TokenCounter,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(_chunk_page(page, count_tokens, max_tokens, overlap_tokens))
    return chunks


def _chunk_page(page: Page, count_tokens: TokenCounter, max_tokens: int, overlap: int) -> list[Chunk]:
    pieces: list[tuple[str, int]] = []
    for sentence in split_sentences(page.text):
        n = count_tokens(sentence)
        if n <= max_tokens:
            pieces.append((sentence, n))
        else:
            pieces.extend(_split_long_sentence(sentence, count_tokens, max_tokens))

    chunks: list[Chunk] = []
    window: list[tuple[str, int]] = []
    window_tokens = 0
    for piece in pieces:
        if window and window_tokens + piece[1] > max_tokens:
            chunks.append(_make_chunk(page.number, window))
            # Carry trailing sentences forward so context spanning a boundary is not lost.
            carried: list[tuple[str, int]] = []
            carried_tokens = 0
            for prev in reversed(window):
                if carried_tokens + prev[1] > overlap:
                    break
                carried.insert(0, prev)
                carried_tokens += prev[1]
            window, window_tokens = carried, carried_tokens
        window.append(piece)
        window_tokens += piece[1]
    if window:
        chunks.append(_make_chunk(page.number, window))
    return chunks


def _split_long_sentence(sentence: str, count_tokens: TokenCounter, max_tokens: int) -> list[tuple[str, int]]:
    """Fallback for run-on text (tables, OCR output): pack words greedily."""
    out: list[tuple[str, int]] = []
    words = sentence.split()
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and count_tokens(candidate) > max_tokens:
            text = " ".join(current)
            out.append((text, count_tokens(text)))
            current = [word]
        else:
            current.append(word)
    if current:
        text = " ".join(current)
        out.append((text, count_tokens(text)))
    return out


def _make_chunk(page: int, window: list[tuple[str, int]]) -> Chunk:
    text = " ".join(s for s, _ in window)
    return Chunk(page=page, text=text, token_count=sum(n for _, n in window), language=detect_language(text))
