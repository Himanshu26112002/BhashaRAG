"""How efficiently do different tokenizers handle Hindi vs English?

Measures on your own corpus (the knowledge base, or any text file):
  - fertility        = subword tokens per whitespace word (lower is better)
  - chars/token      = characters of text covered per token (higher is better)
  - % words split    = share of words broken into 2+ tokens
  - UNK rate         = share of tokens that are the unknown token

It also trains a SentencePiece Unigram tokenizer on 80% of the corpus and evaluates every
tokenizer on the held-out 20%, so the comparison is fair.

Why it matters: an LLM's cost, latency and usable context all scale with token count.
If Hindi takes 3x the tokens of English, a Hindi user gets 1/3 of the context window.

    python -m research.tokenizer_study
    python -m research.tokenizer_study --corpus my_text.txt --vocab-size 8000
"""

from __future__ import annotations

import argparse
import csv
import random
import tempfile
from pathlib import Path

from bhasharag.text_utils import detect_language, normalize_text, split_sentences

from .common import RESULTS_DIR, load_kb_chunks, markdown_table

DEFAULT_TOKENIZERS = [
    "gpt2",                               # English-centric byte-level BPE
    "Qwen/Qwen2.5-0.5B",                  # modern multilingual LLM byte-level BPE
    "bert-base-multilingual-cased",       # mBERT WordPiece
    "xlm-roberta-base",                   # SentencePiece, 100 languages
    "intfloat/multilingual-e5-small",     # the embedding model this app uses
    "google/muril-base-cased",            # trained for Indian languages
]


class HFTokenizer:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.unk_id = self.tok.unk_token_id

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)


class SPTokenizer:
    def __init__(self, train_sentences: list[str], vocab_size: int, workdir: Path):
        import sentencepiece as spm

        corpus = workdir / "spm_train.txt"
        corpus.write_text("\n".join(train_sentences), encoding="utf-8")
        prefix = workdir / "bhasha_spm"
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(prefix),
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=0.9995,  # recommended for scripts with large character sets
            hard_vocab_limit=False,     # small corpora cannot always fill the full vocab
        )
        self.sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
        self.name = f"custom SentencePiece ({self.sp.get_piece_size()} vocab)"
        self.unk_id = self.sp.unk_id()
        self.model_path = Path(str(prefix) + ".model")

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text)


def measure(tokenizer, sentences: list[str], max_words: int = 5000) -> dict:
    tokens = words = chars = unk = 0
    for s in sentences:
        ids = tokenizer.encode(s)
        tokens += len(ids)
        words += len(s.split())
        chars += len(s)
        if tokenizer.unk_id is not None:
            unk += sum(i == tokenizer.unk_id for i in ids)

    vocab_words = [w for s in sentences for w in s.split()][:max_words]
    split = sum(len(tokenizer.encode(w)) > 1 for w in vocab_words)
    return {
        "fertility": tokens / max(words, 1),
        "chars_per_token": chars / max(tokens, 1),
        "pct_words_split": 100 * split / max(len(vocab_words), 1),
        "unk_rate": 100 * unk / max(tokens, 1),
        "tokens": tokens,
    }


def load_sentences(corpus: Path | None) -> list[str]:
    if corpus:
        text = corpus.read_text(encoding="utf-8")
        blocks = [text]
    else:
        blocks = [c["text"] for c in load_kb_chunks()]
    sentences = [s for b in blocks for s in split_sentences(normalize_text(b)) if len(s.split()) >= 3]
    if len(sentences) < 50:
        raise SystemExit(f"Only {len(sentences)} sentences found; add more documents for a meaningful study.")
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, help="Plain-text file (default: the knowledge base)")
    parser.add_argument("--tokenizers", nargs="*", default=DEFAULT_TOKENIZERS)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    sentences = load_sentences(args.corpus)
    random.Random(args.seed).shuffle(sentences)
    cut = int(0.8 * len(sentences))
    train, test = sentences[:cut], sentences[cut:]
    by_lang = {
        "hi": [s for s in test if detect_language(s) == "hi"],
        "en": [s for s in test if detect_language(s) == "en"],
    }
    print(f"{len(sentences)} sentences ({len(train)} train / {len(test)} test); "
          f"test split: {len(by_lang['hi'])} Hindi, {len(by_lang['en'])} English")

    tokenizers = []
    for name in args.tokenizers:
        try:
            tokenizers.append(HFTokenizer(name))
        except Exception as exc:
            print(f"  skipping {name}: {exc.__class__.__name__}: {str(exc)[:120]}")
    workdir = Path(tempfile.mkdtemp(prefix="spm_"))
    try:
        custom = SPTokenizer(train, args.vocab_size, workdir)
        tokenizers.append(custom)
    except Exception as exc:
        print(f"  could not train SentencePiece: {exc}")
        custom = None

    rows, csv_rows = [], []
    for tok in tokenizers:
        row = [tok.name]
        for lang in ("hi", "en"):
            if not by_lang[lang]:
                row += ["-", "-", "-"]
                continue
            m = measure(tok, by_lang[lang])
            row += [m["fertility"], m["chars_per_token"], m["pct_words_split"]]
            csv_rows.append({"tokenizer": tok.name, "language": lang, **m})
        hi_f, en_f = row[1], row[4]
        row.append(hi_f / en_f if isinstance(hi_f, float) and isinstance(en_f, float) else "-")
        rows.append(row)

    headers = ["Tokenizer", "HI fertility", "HI chars/tok", "HI % split",
               "EN fertility", "EN chars/tok", "EN % split", "HI/EN token tax"]
    table = markdown_table(headers, rows)
    print("\n" + table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tokenizer_study.md").write_text(
        "# Tokenizer study\n\n"
        f"Held-out test set: {len(by_lang['hi'])} Hindi and {len(by_lang['en'])} English sentences.\n\n"
        "*Fertility* = tokens per word. *HI/EN token tax* = how many times more tokens per word "
        "Hindi costs than English for that tokenizer.\n\n" + table + "\n",
        encoding="utf-8",
    )
    with (RESULTS_DIR / "tokenizer_study.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    if custom is not None:
        target = RESULTS_DIR / "bhasha_spm.model"
        target.write_bytes(custom.model_path.read_bytes())
        print(f"\nSaved custom tokenizer to {target}")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
