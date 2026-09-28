"""Build a retrieval benchmark from your own knowledge base using an LLM.

For each sampled chunk, the LLM writes one question in English, one in Hindi and one in
Hinglish that the chunk answers. That gives (query -> relevant chunk) pairs, including
*cross-lingual* ones such as a Hindi question about an English passage.

Queries are split into train and test **by chunk**, so no test passage is ever seen
during embedding fine-tuning (otherwise the evaluation would leak).

    python -m research.generate_synthetic_qa --n-chunks 300

Uses the LLM configured in .env (LLM_PROVIDER must not be "none").
Tip: spot-check ~50 generated questions by hand and fix or drop bad ones. A small,
manually verified test set is worth far more than a big noisy one.
"""

from __future__ import annotations

import argparse
import json
import random
import re

from bhasharag.config import get_settings
from bhasharag.llm import LLMError, get_llm

from .common import DATA_DIR, load_kb_chunks, write_jsonl

SYSTEM = """You create evaluation data for a search engine over Hindi and English documents.
Given a passage, write questions that a real user might ask, which this passage answers.

Rules:
- Each question must be answerable from the passage alone and be specific to it (mention the concrete topic; never say "the passage" or "the document").
- Write one question in each requested language: "en" = English; "hi" = Hindi in Devanagari; "hinglish" = Hindi in Roman script, as typed casually on a phone.
- Vary the phrasing; do not copy long spans of the passage verbatim.

Respond with JSON only, in this shape:
{"questions": [{"lang": "en", "question": "..."}, {"lang": "hi", "question": "..."}, {"lang": "hinglish", "question": "..."}]}"""


def parse_questions(raw: str) -> list[dict]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for q in data.get("questions", []):
        if isinstance(q, dict) and q.get("lang") in {"en", "hi", "hinglish"} and len(str(q.get("question", ""))) > 8:
            out.append({"lang": q["lang"], "question": str(q["question"]).strip()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-chunks", type=int, default=300)
    parser.add_argument("--min-tokens", type=int, default=40, help="Skip tiny chunks (headers, footers)")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    llm = get_llm(get_settings())
    if llm is None:
        raise SystemExit("Set LLM_PROVIDER in .env (anthropic / ollama / openai_compatible) to generate questions.")

    chunks = [c for c in load_kb_chunks() if c["token_count"] >= args.min_tokens]
    rng = random.Random(args.seed)
    rng.shuffle(chunks)
    chunks = chunks[: args.n_chunks]
    print(f"Generating questions for {len(chunks)} chunks with {llm.name}")

    records = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            raw = llm.generate(SYSTEM, [{"role": "user", "content": f"Passage:\n{chunk['text']}"}], max_tokens=4000)
        except LLMError as exc:
            print(f"  [{i}] LLM error, skipping: {exc}")
            continue
        questions = parse_questions(raw)
        for q in questions:
            records.append({
                "query": q["question"],
                "lang": q["lang"],
                "chunk_id": chunk["id"],
                "doc_id": chunk["doc_id"],
                "chunk_language": chunk["language"],
            })
        if i % 10 == 0 or i == len(chunks):
            print(f"  {i}/{len(chunks)} chunks, {len(records)} questions")

    chunk_ids = sorted({r["chunk_id"] for r in records})
    rng.shuffle(chunk_ids)
    test_ids = set(chunk_ids[: int(len(chunk_ids) * args.test_fraction)])
    train = [r for r in records if r["chunk_id"] not in test_ids]
    test = [r for r in records if r["chunk_id"] in test_ids]
    write_jsonl(DATA_DIR / "qa_train.jsonl", train)
    write_jsonl(DATA_DIR / "qa_test.jsonl", test)
    print(f"Wrote {len(train)} train and {len(test)} test queries to {DATA_DIR}")


if __name__ == "__main__":
    main()
