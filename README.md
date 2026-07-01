# Cross-Lingual K-Culture RAG

A small, fully local Retrieval-Augmented Generation (RAG) project that answers **English and Korean** questions from **Korean Wikipedia** documents about K-culture, and measures **where and why it breaks**.

The driving questions: how does a language model connect two very different languages (Korean and English)? Does RAG add real value on facts the model cannot already know? And what actually happens when you remove the "answer only from the context" safety rule?

> **One-line takeaway:** the system finds the right document very well (≈89% retrieval) and, with a grounding rule on, almost never makes things up. But it answers the actual question only about a third of the time, because fixed-size chunking cuts the key fact out of the retrieved passage. When the grounding rule is removed, accuracy jumps — and that is exactly where the strong and the weak model split apart.

---

## Motivation

This started as a writing helper for German learners (A2–B1), but the models already knew German grammar too well to measure any RAG effect. So I pivoted to a harder cross-lingual case. A recent paper notes that Korean is a **low-resource language for RAG** with few good evaluation frameworks (Kim et al., 2025), and reports **EXAONE 3.5** as the strongest open Korean model — which motivated a **strong vs. weak generator** comparison (EXAONE 3.5 vs. Llama 3). For a light corpus, I used K-culture, taken entirely from Wikipedia for copyright safety.

---

## Corpus

Seven Korean Wikipedia articles, saved as PDFs (130 pages total), in `Data/`.

| File | Article | Source |
|---|---|---|
| `01_kpop_demon_hunters` | 케이팝 데몬 헌터스 | ko.wikipedia.org/wiki/케이팝_데몬_헌터스 |
| `02_bts` | 방탄소년단 (BTS) | ko.wikipedia.org/wiki/방탄소년단 |
| `03_blackpink` | 블랙핑크 (BLACKPINK) | ko.wikipedia.org/wiki/블랙핑크 |
| `04_squid_game` | 오징어 게임 | ko.wikipedia.org/wiki/오징어_게임 |
| `05_parasite` | 기생충 (영화) | ko.wikipedia.org/wiki/기생충_(영화) |
| `06_Hallyu` | 한류 | ko.wikipedia.org/wiki/한류 |
| `07_K-pop` | 케이팝 | ko.wikipedia.org/wiki/케이팝 |

---

## Pipeline

Everything runs locally through Ollama. No cloud API.

```
PDFs → chunk → embed (nomic-embed-text) → FAISS → retrieve top-k → prompt → answer
```

- **Loader:** `PyPDFLoader` (7 PDFs → 130 pages)
- **Chunking:** `RecursiveCharacterTextSplitter` — fixed 500 / overlap 50; large 1200 / overlap 200
- **Embedding:** `nomic-embed-text` via `OllamaEmbeddings` (switched from HuggingFace to stay Torch-free)
- **Vector store:** FAISS · **Search:** cosine / MMR / HyDE · **k = 3**
- **Generator:** EXAONE 3.5 (strong) vs. Llama 3 (weak)
- **Prompt:** a **grounding** rule (answer only from context, else say it is not stated, never invent) — and a **loose** rule (may also use own knowledge) for the last two settings

### The 7 settings (one dial changed at a time)

| Setting | Chunking | Search | Model | Grounding |
|---|---|---|---|---|
| `baseline` | fixed (500) | cosine | EXAONE 3.5 | strict |
| `exp1_largechunk` | large (1200) | cosine | EXAONE 3.5 | strict |
| `exp2_mmr` | fixed (500) | MMR | EXAONE 3.5 | strict |
| `exp3_hyde` | fixed (500) | HyDE | EXAONE 3.5 | strict |
| `exp4_weakmodel` | fixed (500) | cosine | Llama 3 | strict |
| `exp5_nogrounding` | fixed (500) | cosine | EXAONE 3.5 | **loose** |
| `exp6_nogrounding_weak` | fixed (500) | cosine | Llama 3 | **loose** |

Language is handled **per question**: a bilingual question runs once in English and once in Korean.

---

## Repository structure

```
.
├── Data/                                  # 7 Korean Wikipedia PDFs
├── questions_kculture_v2.json             # 19 evaluation questions, 4 tiers
├── RAG_Ollama_evaluation.py               # runs the full 7×19 grid → CSV
├── RAG_Ollama_interactive.py              # live check (RAG vs no-RAG)
├── RAG_Ollama_interactive_v2.py           # live check with switchable search (cosine/MMR/HyDE)
├── RAG_Kculture_result_7config.csv        # 133 raw runs
├── RAG_Kculture_scored_7config_FINAL.xlsx # manual scores + Summary sheet
├── Engineering_Report_Kculture_RAG_EN.docx
├── Engineering_Report_Kculture_RAG_KO.docx
└── README.md
```

---

## Setup

With [Ollama](https://ollama.com) running:

```bash
ollama pull exaone3.5        # strong Korean model
ollama pull llama3           # weak comparison
ollama pull nomic-embed-text # embeddings
```

```bash
pip install langchain-community langchain-text-splitters langchain-ollama faiss-cpu pypdf
```

## Usage

```bash
python RAG_Ollama_evaluation.py     # runs every setting once, writes the result CSV
python RAG_Ollama_interactive_v2.py # type a question; set SEARCH = cosine/mmr/hyde
```

After the run, fill the manual columns (`answer_correct`, `abstained`, `hallucinated`, `faithful`) in the CSV.

---

## Evaluation design

Questions are built to **create score gaps between settings**, not to be easy.

| Tier | Focus | Idea |
|---|---|---|
| 1 — Time / RAG gap | facts after the 2024 cutoff | only RAG can answer (2026 awards, album dates) |
| 2 — Cross-lingual | EN question, KO text, no shared words | Korean Wave ↔ 한류, Palme d'Or ↔ 황금종려상, barrier ↔ 혼문 |
| 3 — Grounding | unanswerable / false premise | the right move is to refuse, not invent |
| 4 — Multi-hop | two or more documents at once | small k or low variety fails |

---

## Results (final, manually scored)

| Setting | Model | Grounding | Retrieval | Answer correct | Hallucinated | Faithful |
|---|---|:--:|:--:|:--:|:--:|:--:|
| baseline | EXAONE 3.5 | strict | 89.5% | 26% | 0 | 74% |
| exp1_largechunk | EXAONE 3.5 | strict | 84.2% | 26% | 0 | 79% |
| exp2_mmr | EXAONE 3.5 | strict | 89.5% | 32% | 1 | 79% |
| exp3_hyde | EXAONE 3.5 | strict | 89.5% | 37% | 1 | 84% |
| exp4_weakmodel | Llama 3 | strict | 89.5% | 21% | 0 | 79% |
| **exp5_nogrounding** | EXAONE 3.5 | **loose** | 89.5% | **58%** | 1 | 84% |
| **exp6_nogrounding_weak** | Llama 3 | **loose** | 89.5% | 32% | **4** | **58%** |

By language, retrieval was **100% for English** questions and **71.1% for Korean** — every retrieval miss in the whole study was a Korean query.

## Key findings

1. **The RAG gap.** ~89% of documents found, but only ~21–37% answered under grounding, with high abstention. Fixed-size chunking splits the key fact out of the retrieved passage, so the model honestly refuses ("the document does not state it").
2. **English ≠ Korean.** The embedding matches English questions to Korean text very well; Korean-to-Korean misses more (proper nouns, word endings).
3. **HyDE and large chunks backfired.** HyDE (an English guess over a Korean DB) and larger, noisier chunks both hurt retrieval.
4. **Grounding is a safety belt — and it hid the models.** With the rule on, EXAONE and Llama 3 looked identical (near-zero hallucination, similar accuracy). Removing it revealed the real gap: EXAONE loose reached **58% correct with 1 hallucination and 84% faithful**, while Llama 3 loose stalled at **32%, jumped to 4 hallucinations, and dropped to 58% faithful** (it even invented a fake 2005 drama). The strong model's real advantage is that it **stays honest without a strict prompt**.

## Next steps

- **Parent-document retrieval** + higher `k`, so the full section (not a cut fragment) reaches the model.
- **Hybrid search (BM25 + dense)** to match exact Korean proper nouns.
- **Span-level scoring**, not just document-level, to expose the real failure point.
- **Keep grounding on in any product**, especially with a weaker model.

## Limitations

Single run, 19 questions per setting (differences of one or two are within noise); retrieval scored at document level; four answer columns scored by hand; `k` held constant at 3. See the engineering report for details.

## Attribution

All corpus content is from Korean Wikipedia, used under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). Built with LangChain, FAISS, and Ollama.
