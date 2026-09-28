# Roadmap

A suggested order of work. Each phase ends with something you can demo or measure.

## Phase 1: Working app (done in this scaffold)
- [x] Upload / delete documents (PDF, DOCX, TXT, MD), duplicate detection
- [x] Hindi + English normalisation, token-aware chunking
- [x] Hybrid retrieval (multilingual-E5 + BM25 + RRF) and cross-encoder reranking
- [x] Grounded answers with citations in the user's language, follow-up rewriting
- [x] Web UI + REST API + tests

**Your tasks:** run it, upload 10–20 real documents (government scheme PDFs, NCERT Hindi and
English chapters, your college notes) and note where answers go wrong. Those failures become Phase 3.

## Phase 2: Measure (the part most student projects skip)
- [ ] `python -m research.tokenizer_study`: add the table and a bar chart to the README
- [ ] `python -m research.generate_synthetic_qa`, then hand-verify ~50 test questions
- [ ] `python -m research.evaluate_retrieval --tag baseline`: the ablation table
- [ ] Also evaluate `BAAI/bge-m3` as the embedder (`--embedding-model BAAI/bge-m3`) and compare

## Phase 3: Improve with deep learning
- [ ] Fine-tune the embedder (`research/finetune_embeddings.py`) and report the Recall@5 change,
      especially for cross-lingual queries (Hindi question → English passage)
- [ ] Add Hindi stemming or lemmatisation to BM25 and measure the change
- [ ] Transliteration: convert Hinglish queries to Devanagari (e.g. with IndicXlit) before BM25
- [ ] Try a larger reranker (`BAAI/bge-reranker-v2-m3`) and chart the latency vs. accuracy trade-off

## Phase 4: Trustworthiness
- [ ] Groundedness check: an NLI model (e.g. multilingual `mDeBERTa-xnli`) verifies that each
      answer sentence is entailed by its cited passage; flag unsupported sentences in the UI
- [ ] Answer-quality evaluation: faithfulness and answer relevance (e.g. with RAGAS or an LLM judge)
      on 100 questions, reported before and after the groundedness check
- [ ] "I don't know" calibration: refuse when the best rerank score is below a tuned threshold

## Phase 5: Ship it
- [ ] Streaming answers (Server-Sent Events)
- [ ] Dockerfile and deployment to Hugging Face Spaces or Render, with a live demo link in the README
- [ ] 2-minute demo video and a short write-up (blog or LinkedIn) of what you measured and learned

## Resume bullets (fill in your real numbers)
- Built **BhashaRAG**, a bilingual Hindi–English RAG system (FastAPI, FAISS, sentence-transformers) with
  hybrid BM25 + dense retrieval, cross-encoder reranking and citation-grounded answers in the user's language.
- Fine-tuned multilingual-E5 with contrastive learning (MNRL + mined hard negatives) on a 2K-query synthetic
  bilingual benchmark, improving cross-lingual Recall@5 from **X** to **Y**.
- Analysed tokenizer fertility across 6 tokenizers, showing Hindi costs up to **Z×** more tokens per word than
  English; trained a domain SentencePiece tokenizer that cut Hindi token count by **W%**.
