"""Domain-adapt the multilingual embedding model with contrastive learning.

Training data: (query, positive chunk, hard negative chunk) triplets built from
research/data/qa_train.jsonl.

Loss: MultipleNegativesRankingLoss (InfoNCE). For a batch of B queries, each query must
score its own passage above the B-1 other passages in the batch *and* above its mined
hard negative. Every batch therefore gives B x (B+1) contrastive comparisons.

Hard negatives: passages the *base* model ranks highly for the query but that are not
the answer. Mining them from ranks `--neg-rank-min..--neg-rank-max` (not rank 1-2) and
dropping candidates scoring within `--false-neg-margin` of the positive avoids training
on "negatives" that are really unlabeled positives.

    pip install -r requirements-research.txt
    python -m research.finetune_embeddings --epochs 2
    python -m research.evaluate_retrieval --embedding-model models/bhasha-e5-finetuned --tag finetuned

To use the fine-tuned model in the app, set EMBEDDING_MODEL=models/bhasha-e5-finetuned in
.env and restart. The index is re-embedded automatically on startup.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from bhasharag.config import PROJECT_ROOT, get_settings

from .common import DATA_DIR, load_kb_chunks, read_jsonl


def mine_hard_negatives(model, prefix_q: str, prefix_p: str, queries: list[dict], chunks: list[dict],
                        rank_min: int, rank_max: int, margin: float, seed: int) -> list[dict]:
    ids = [c["id"] for c in chunks]
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    p = model.encode([prefix_p + c["text"] for c in chunks], batch_size=32, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True)
    q = model.encode([prefix_q + x["query"] for x in queries], batch_size=64, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=True)
    rng = random.Random(seed)
    triplets = []
    for qi, item in enumerate(queries):
        pos_idx = id_to_idx.get(item["chunk_id"])
        if pos_idx is None:
            continue
        scores = p @ q[qi]
        pos_score = scores[pos_idx]
        order = [i for i in np.argsort(-scores) if i != pos_idx]
        candidates = [i for i in order[rank_min - 1: rank_max] if scores[i] < pos_score - margin]
        if not candidates:
            candidates = order[rank_max: rank_max + 10] or order[-10:]
        neg_idx = rng.choice(candidates)
        triplets.append({
            "anchor": prefix_q + item["query"],
            "positive": prefix_p + chunks[pos_idx]["text"],
            "negative": prefix_p + chunks[neg_idx]["text"],
        })
    return triplets


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default=settings.embedding_model)
    parser.add_argument("--train", default=str(DATA_DIR / "qa_train.jsonl"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "models" / "bhasha-e5-finetuned"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32, help="Bigger = more in-batch negatives")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--neg-rank-min", type=int, default=3)
    parser.add_argument("--neg-rank-max", type=int, default=30)
    parser.add_argument("--false-neg-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    try:  # sentence-transformers >= 6
        from sentence_transformers.sentence_transformer import losses
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError:  # 3.x - 5.x
        from sentence_transformers import losses
        from sentence_transformers.training_args import BatchSamplers

    queries = read_jsonl(Path(args.train))
    chunks = load_kb_chunks()
    model = SentenceTransformer(args.base_model)
    e5 = "e5" in args.base_model.lower()
    prefix_q, prefix_p = ("query: ", "passage: ") if e5 else ("", "")

    print(f"Mining hard negatives for {len(queries)} training queries…")
    triplets = mine_hard_negatives(model, prefix_q, prefix_p, queries, chunks,
                                   args.neg_rank_min, args.neg_rank_max, args.false_neg_margin, args.seed)
    dataset = Dataset.from_list(triplets).shuffle(seed=args.seed)
    print(f"{len(dataset)} training triplets")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        # Two queries about the same chunk in one batch would be false in-batch negatives.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=losses.MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    model.save_pretrained(args.output)
    print(f"\nSaved fine-tuned model to {args.output}")
    print(f"Next: python -m research.evaluate_retrieval --embedding-model {args.output} --tag finetuned")


if __name__ == "__main__":
    main()
