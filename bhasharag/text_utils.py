"""Language-aware text utilities for English, Hindi (Devanagari) and Hinglish (romanised Hindi).

Everything here is dependency-free so it can be unit-tested quickly.
"""

from __future__ import annotations

import re
import unicodedata

# Devanagari block is U+0900–U+097F. The danda "।" (U+0964) and double danda "॥" (U+0965)
# are sentence punctuation, so they are excluded from "word" characters.
_DEVANAGARI_LETTER = re.compile(r"[ऀ-ॣ०-ॿ]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[a-z0-9À-ɏ]+|[ऀ-ॣ०-ॿ]+")
_SENTENCE_END = re.compile(r"(?<=[.!?।॥])\s+|\n{2,}")
_ZERO_WIDTH = re.compile(r"[​⁠﻿]")  # keep ZWJ/ZWNJ: they change Devanagari rendering
_ZWJ_ZWNJ = re.compile(r"[‌‍]")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")

# High-frequency romanised-Hindi function words. If enough of these appear in Latin-script
# text, we treat it as Hinglish so the answer comes back in the same register.
_HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kaise", "kaisa", "kyun", "kyon", "nahi", "nahin", "mein", "mujhe",
    "hum", "aap", "tum", "ka", "ki", "ke", "ko", "se", "aur", "bhi", "kab", "kahan", "kitna",
    "kitne", "kaun", "wala", "wali", "liye", "yeh", "ye", "woh", "vo", "tha", "thi", "hota",
    "hoti", "karna", "karne", "kare", "batao", "bataiye", "chahiye", "sakta", "sakte", "milega",
    "milta", "yojana", "sarkar", "paisa", "kaam",
}

_EN_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "for",
    "and", "or", "with", "by", "at", "from", "as", "that", "this", "it", "its", "what", "which",
    "who", "how", "do", "does", "did", "can", "i", "me", "my", "you", "your", "we", "our",
}
_HI_STOPWORDS = {
    "है", "हैं", "था", "थी", "थे", "का", "की", "के", "को", "में", "से", "पर", "और", "या",
    "भी", "यह", "वह", "ये", "वे", "एक", "तो", "ही", "कि", "क्या", "कैसे", "लिए", "हो", "होता",
    "होती", "कर", "करें", "किया", "गया", "गई", "इस", "उस", "जो", "ने",
}
STOPWORDS = _EN_STOPWORDS | _HI_STOPWORDS | {"ka", "ki", "ke", "ko", "se", "hai", "hain", "aur", "mein"}


def normalize_text(text: str) -> str:
    """Canonicalise Unicode and whitespace.

    NFC matters for Hindi: the same syllable can be encoded as different code-point
    sequences (e.g. nukta forms), and un-normalised text silently breaks both BM25
    matching and tokenizer statistics.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Return one of: "hi" (Devanagari Hindi), "en", "hinglish" (romanised Hindi), "mixed".

    A script-ratio heuristic is enough here and has no model download; it only decides
    which language the *answer* should be written in.
    """
    deva = len(_DEVANAGARI_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    total = deva + latin
    if total == 0:
        return "en"
    deva_ratio = deva / total
    if deva_ratio >= 0.6:
        return "hi"
    if deva_ratio >= 0.15:
        return "mixed"
    words = re.findall(r"[a-z]+", text.lower())
    if words:
        marker_ratio = sum(w in _HINGLISH_MARKERS for w in words) / len(words)
        if marker_ratio >= 0.2 or (len(words) <= 4 and marker_ratio >= 0.25):
            return "hinglish"
    return "en"


def tokenize_for_bm25(text: str) -> list[str]:
    """Lexical tokens for BM25: lower-cased Latin words and whole Devanagari words (with matras)."""
    text = _ZWJ_ZWNJ.sub("", unicodedata.normalize("NFC", text.lower()))
    return [
        tok for tok in _WORD.findall(text)
        if tok.isdigit() or (len(tok) > 1 and tok not in STOPWORDS)
    ]


def split_sentences(text: str) -> list[str]:
    """Split on ., !, ?, the Hindi danda (।, ॥) and blank lines."""
    parts: list[str] = []
    for block in _SENTENCE_END.split(text):
        block = block.strip()
        if not block:
            continue
        # Single newlines inside a block are usually layout (PDF line wraps), but list items
        # and headings also live on their own line; keep short lines separate.
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        buf = ""
        for line in lines:
            if buf and (len(line) < 40 or line[:1] in "-•*" or line[:2].rstrip(".").isdigit()):
                parts.append(buf)
                buf = line
            else:
                buf = f"{buf} {line}".strip()
        if buf:
            parts.append(buf)
    return parts
