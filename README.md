# Cross-Lingual K-Culture RAG

A small, fully local Retrieval-Augmented Generation (RAG) project that answers **English and Korean** questions from **Korean Wikipedia** source documents about K-culture, and measures how well it does it.

The interesting question behind it: how does a language model connect two languages that look completely different in shape, like Korean and English? And does RAG actually add value when the model is tested on facts it cannot already know?

> **One-line takeaway:** the system finds the right document very well and never makes things up, but it answers the actual question only about a third of the time. The bottleneck is **chunking**, not search.

---

## Motivation

This started as a writing helper for German learners (A2–B1). The problem was that the models already know German grammar too well, so I could not tell whether the help came from my documents or from the model itself. The signal was too weak to measure.

So I pivoted to a harder cross-lingual case. A recent paper notes that Korean is a **low-resource language for RAG** and that good evaluation frameworks are still missing (Kim et al., 2025). The same work reports **EXAONE 3.5** as the strongest open Korean model, which motivated a **strong vs. weak generator** comparison (EXAONE 3.5 vs. Llama 3). For a light but fun corpus, I chose K-culture, pulled entirely from Wikipedia for copyright safety.

---

## Corpus

Seven Korean Wikipedia articles, saved as PDFs (130 pages total).

| File | Article | Source |
|---|---|---|
| `01_kpop_demon_hunters` | 케이팝 데몬 헌터스 (KPop Demon Hunters) | ko.wikipedia.org/wiki/케이팝_데몬_헌터스 |
| `02_bts` | 방탄소년단 (BTS) | ko.wikipedia.org/wiki/방탄소년단 |
| `03_blackpink` | 블랙핑크 (BLACKPINK) | ko.wikipedia.org/wiki/블랙핑크 |
| `04_squid_game` | 오징어 게임 (Squid Game) | ko.wikipedia.org/wiki/오징어_게임 |
| `05_parasite` | 기생충 (Parasite, film) | ko.wikipedia.org/wiki/기생충_(영화) |
| `06_Hallyu` | 한류 (Hallyu) | ko.wikipedia.org/wiki/한류 |
| `07_K-pop` | 케이팝 (K-pop) | ko.wikipedia.org/wiki/케이팝 |

> Corpus PDFs live in `Data/` and are included in the repo. They are sourced from Korean Wikipedia (CC BY-SA) — see Attribution below.

---

## Pipeline

Everything runs locally through Ollama. No cloud API.

```
PDFs → chunk → embed (Nomic) → FAISS → retrieve top-k → prompt → answer
```

- **Loader:** `PyPDFLoader` (7 Korean PDFs → 130 pages)
- **Chunking:** `RecursiveCharacterTextSplitter` — baseline 500 / overlap 50; large 1200 / overlap 200
- **Embedding:** `nomic-embed-text` via `OllamaEmbeddings` (switched from HuggingFace embeddings to stay Torch-free and avoid a Torch error)
- **Vector store:** FAISS
- **Search:** cosine similarity / MMR / HyDE; `k = 3` in the evaluation
- **Generator:** EXAONE 3.5 (strong) vs. Llama 3 (weak)
- **Prompt:** answer only from context, otherwise say the document does not state it, never invent facts

### The five settings (one dial changed at a time)

| Setting | Chunking | Search | Model |
|---|---|---|---|
| `baseline` | fixed (500) | cosine | EXAONE 3.5 |
| `exp1_largechunk` | large (1200) | cosine | EXAONE 3.5 |
| `exp2_mmr` | fixed (500) | MMR | EXAONE 3.5 |
| `exp3_hyde` | fixed (500) | HyDE | EXAONE 3.5 |
| `exp4_weakmodel` | fixed (500) | cosine | Llama 3 |

Language is handled **per question**: a bilingual question runs once in English and once in Korean, which gives the cross-lingual comparison.

---

## Repository structure

```
.
├── Data/                          # 7 Korean Wikipedia PDFs (not committed)
├── questions_kculture_v2.json     # 19 evaluation questions, 4 tiers
├── RAG_Ollama_interactive.py      # type a question, see chunks + answers (RAG vs no-RAG)
├── RAG_Ollama_evaluation.py       # runs the full 5×19 grid, saves the CSV
├── RAG_Kculture_result.csv        # 95 runs (raw output)
├── RAG_Kculture_scored.xlsx       # manual scores added
└── Engineering_Report_Kculture_RAG.html  # full write-up
```

---

## Setup

With [Ollama](https://ollama.com) running:

```bash
ollama pull exaone3.5        # strong Korean model
ollama pull llama3           # weak comparison
ollama pull nomic-embed-text # embeddings
```

Python packages (no Torch needed):

```bash
pip install langchain-community langchain-text-splitters langchain-ollama faiss-cpu pypdf
```

Place the seven PDFs in `Data/`.

---

## Usage

**Interactive** — type a question, watch the retrieved chunks and both answers (with and without RAG):

```bash
python RAG_Ollama_interactive.py
```

**Evaluation** — runs every setting in one go and writes `RAG_Kculture_result.csv`. The model is part of each setting, so the script swaps models automatically (run it once):

```bash
python RAG_Ollama_evaluation.py
```

After the run, fill the manual columns in the CSV: `answer_correct`, `abstained`, `hallucinated`, `faithful`.

---

## Evaluation design

The questions are built to **create score gaps between settings**, not to be easy.

| Tier | Focus | Idea |
|---|---|---|
| 1 — Time / RAG gap (T1–T4) | facts after the 2024 cutoff | only RAG can answer (e.g. 2026 awards, album dates) |
| 2 — Cross-lingual (X1–X3) | EN question, KO text, no shared words | Korean Wave ↔ 한류, Palme d'Or ↔ 황금종려상, barrier ↔ 혼문 |
| 3 — Grounding (G1–G4) | unanswerable / false premise | the right move is to refuse, not invent |
| 4 — Multi-hop (M1–M3) | two or more documents at once | small k or low variety fails |

---

## Results

| Setting | Model | Retrieval hit | Answer correct | Abstain | Hallucinated | Faithful |
|---|---|:--:|:--:|:--:|:--:|:--:|
| baseline | EXAONE 3.5 | 89.5% | 30.8% | 84.6% | 0.0% | 100% |
| exp1_largechunk | EXAONE 3.5 | 84.2% | 33.3% | 84.6% | 0.0% | 100% |
| exp2_mmr | EXAONE 3.5 | 89.5% | 35.7% | 80.0% | 0.0% | 100% |
| exp3_hyde | EXAONE 3.5 | 78.9% | 35.7% | 80.0% | 0.0% | 100% |
| exp4_weakmodel | Llama 3 | 89.5% | 35.7% | 80.0% | 0.0% | 100% |

By language, retrieval hit was **100% for English** questions and **71.1% for Korean** questions.

## Key findings

1. **No hallucination (0%), fully faithful (100%)** across all 95 runs, even with the weak model. The grounding prompt works.
2. **The RAG gap:** ~89.5% of documents found, but only ~31% answered, with 80%+ abstentions. Fixed-size chunking splits the key fact out of the retrieved piece, so the model honestly refuses.
3. **English ≠ Korean:** the embedding matches English questions to Korean text very well; Korean-to-Korean misses more often due to proper nouns and word endings.
4. **HyDE and large chunks backfired:** HyDE (English guess over a Korean DB) dropped retrieval to 78.9%; larger chunks added noise that diluted similarity.

## Next steps

- **Parent-document retrieval** so the full section reaches the model instead of a cut fragment.
- **Hybrid search (BM25 + dense)** so exact Korean proper nouns are matched directly.
- **Span-level scoring**, not just document-level, to expose the real failure point.

---

## Limitations

Single run, 19 questions per setting (differences of one or two questions are within noise). Retrieval is scored at the document level. A few gold answers still need a final check against the source text.

## Attribution

All corpus content is from Korean Wikipedia, used under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). Built with LangChain, FAISS, and Ollama.
