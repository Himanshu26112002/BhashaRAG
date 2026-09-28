"""Retrieval ablation study: which component actually helps, and for which language?

Systems compared on the held-out benchmark (from generate_synthetic_qa.py):
  1. BM25 only
  2. Dense only (multilingual embeddings)
  3. Hybrid (BM25 + dense, reciprocal rank fusion)
  4. Hybrid + cross-encoder reranker

Metrics: Recall@1/5/10, MRR@10, nDCG@10, overall and per query language.

    python -m research.evaluate_retrieval
    python -m research.evaluate_retrieval --embedding-model models/bhasha-e5-finetuned --tag finetuned
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from bhasharag.config import get_settings
from bhasharag.embedder import Embedder
from bhasharag.retrieval import BM25Index, Reranker, reciprocal_rank_fusion

from .common import DATA_DIR, RESULTS_DIR, load_kb_chunks, markdown_table, read_jsonl

KS = (1, 5, 10)


def metrics_for_rank(rank: int | None) -> dict[str, float]:
    """Each query has exactly one relevant chunk, so nDCG@10 = 1/log2(rank+1)."""
    m = {f"R@{k}": float(rank is not None and rank <= k) for k in KS}
    m["MRR@10"] = 1.0 / rank if rank is not None and rank <= 10 else 0.0
    m["nDCG@10"] = 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0
    return m


def rank_of(target: int, ranking: list[int]) -> int | None:
    try:
        return ranking.index(target) + 1
    except ValueError:
        return None


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qa", default=str(DATA_DIR / "qa_test.jsonl"))
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--reranker-model", default=settings.reranker_model)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--candidates", type=int, default=settings.candidates_per_retriever)
    parser.add_argument("--tag", default="baseline", help="Name for this run in the results file")
    args = parser.parse_args()

    queries = read_jsonl(Path(args.qa))
    chunks = load_kb_chunks()
    ids = [c["id"] for c in chunks]
    texts = {c["id"]: c["text"] for c in chunks}
    queries = [q for q in queries if q["chunk_id"] in texts]
    print(f"{len(queries)} queries over {len(chunks)} chunks")

    embedder = Embedder(args.embedding_model)
    passage_matrix = embedder.embed_passages([c["text"] for c in chunks])
    query_matrix = embedder.embed_queries([q["query"] for q in queries])
    bm25 = BM25Index()
    bm25.build([(c["id"], c["text"]) for c in chunks])
    reranker = None if args.no_rerank else Reranker(args.reranker_model)

    n = args.candidates
    results: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for qi, q in enumerate(queries):
        scores = passage_matrix @ query_matrix[qi]
        dense = [ids[i] for i in np.argsort(-scores)[:n]]
        lexical = [cid for cid, _ in bm25.search(q["query"], n)]
        hybrid = [cid for cid, _ in reciprocal_rank_fusion([dense, lexical])][:n]
        systems = {"BM25": lexical, "Dense": dense, "Hybrid (RRF)": hybrid}
        if reranker is not None:
            rr = reranker.score(q["query"], [texts[c] for c in hybrid])
            systems["Hybrid + rerank"] = [c for _, c in sorted(zip(rr, hybrid), key=lambda t: -t[0])]
        for name, ranking in systems.items():
            m = metrics_for_rank(rank_of(q["chunk_id"], ranking))
            results[name]["all"].append(m)
            results[name][q["lang"]].append(m)
        if (qi + 1) % 50 == 0:
            print(f"  {qi + 1}/{len(queries)}")

    metric_names = [f"R@{k}" for k in KS] + ["MRR@10", "nDCG@10"]
    mean = lambda ms, key: float(np.mean([m[key] for m in ms])) if ms else float("nan")

    overall = markdown_table(
        ["System", *metric_names],
        [[name, *[mean(by["all"], k) for k in metric_names]] for name, by in results.items()],
    )
    langs = sorted({q["lang"] for q in queries})
    per_lang = markdown_table(
        ["System", *[f"R@5 ({lang}, n={sum(q['lang'] == lang for q in queries)})" for lang in langs]],
        [[name, *[mean(by[lang], "R@5") for lang in langs]] for name, by in results.items()],
    )
    report = (
        f"## Run: {args.tag}\n\nEmbedding model: `{args.embedding_model}`  \n"
        f"Reranker: `{None if reranker is None else args.reranker_model}`  \n"
        f"Queries: {len(queries)} · Chunks: {len(chunks)}\n\n### Overall\n\n{overall}\n\n"
        f"### Recall@5 by query language\n\n{per_lang}\n"
    )
    print("\n" + report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "retrieval_eval.md").open("a", encoding="utf-8") as f:
        f.write(report + "\n")
    summary = {name: {k: mean(by["all"], k) for k in metric_names} for name, by in results.items()}
    (RESULTS_DIR / f"retrieval_eval_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Appended to {RESULTS_DIR / 'retrieval_eval.md'}")


if __name__ == "__main__":
    main()
