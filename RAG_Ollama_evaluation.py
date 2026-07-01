"""
RAG_Ollama_evaluation.py  (torch-free, Korean K-culture, v2 qualitative eval)
==========================================================================
Runs the whole experiment matrix in ONE go and saves RAG_Kculture_result.csv.
You do NOT run it twice — the model is part of 
each config, so the script swaps models automatically.

7 configurations (each changes ONE dial from baseline):
  baseline             fixed   | cosine | STRONG | grounded
  exp1_largechunk      large   |        |        |
  exp2_mmr                     | mmr    |        |
  exp3_hyde                    | (hyde) |        |
  exp4_weakmodel               |        | WEAK   |          <- generator effect
  exp5_nogrounding             |        | STRONG | LOOSE    <- grounding effect
  exp6_nogrounding_weak        |        | WEAK   | LOOSE    <- weak model unshackled (optional)

Language is handled PER QUESTION (not per config): a question tagged
["en","ko"] is run in BOTH languages -> the EN vs KO rows give the
cross-lingual penalty.

Output: RAG_Kculture_result_final.csv
Manual columns to fill after the run: answer_correct, abstained, hallucinated, faithful

Folder layout (same folder as this file):
  Data/                       <- 7 Korean corpus PDFs
  questions_kculture_v2.json  <- v2 question set

Setup (Ollama running):
  ollama pull exaone3.5         # strong Korean model (or qwen2.5)
  ollama pull llama3            # weak (comparison)
  ollama pull nomic-embed-text
Packages (no torch):
  pip install langchain-community langchain-text-splitters langchain-ollama faiss-cpu pypdf
"""

import json
import csv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
QUESTIONS = ROOT / "questions_kculture_v2.json"
OUT_CSV = ROOT / "RAG_Kculture_result_final.csv"

# ---- models ----
STRONG_MODEL = "exaone3.5"   
WEAK_MODEL = "llama3"        

embeddings = OllamaEmbeddings(model="nomic-embed-text")   # for multi-lang

# ---- 7 configurations (language is per-question, so no query_lang here) ----
# NEW: every config now has a "grounding" flag (True = strict, False = loose)
configs = [
    {"name": "baseline",              "chunking": "fixed", "search": "similarity", "hyde": False, "model": STRONG_MODEL, "grounding": True},
    {"name": "exp1_largechunk",       "chunking": "large", "search": "similarity", "hyde": False, "model": STRONG_MODEL, "grounding": True},
    {"name": "exp2_mmr",              "chunking": "fixed", "search": "mmr",        "hyde": False, "model": STRONG_MODEL, "grounding": True},
    {"name": "exp3_hyde",             "chunking": "fixed", "search": "similarity", "hyde": True,  "model": STRONG_MODEL, "grounding": True},
    {"name": "exp4_weakmodel",        "chunking": "fixed", "search": "similarity", "hyde": False, "model": WEAK_MODEL,   "grounding": True},
    # ---- NEW: grounding OFF (loose prompt) ----
    {"name": "exp5_nogrounding",      "chunking": "fixed", "search": "similarity", "hyde": False, "model": STRONG_MODEL, "grounding": False},
    {"name": "exp6_nogrounding_weak", "chunking": "fixed", "search": "similarity", "hyde": False, "model": WEAK_MODEL,   "grounding": False},  # optional: delete if not needed
]

# ---- load corpus once ----
documents = []
for path in sorted(DATA_DIR.glob("*")):
    if path.suffix.lower() == ".pdf":
        loaded = PyPDFLoader(str(path)).load()
    elif path.suffix.lower() in {".txt", ".md"}:
        loaded = TextLoader(str(path), encoding="utf-8").load()
    else:
        continue
    for d in loaded:
        d.metadata["doc"] = path.stem
    documents += loaded
print(f"Loaded {len(documents)} pages from Data/")

questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))

# ---- caches ----
_indexes, _models, _norag = {}, {}, {}

def get_index(chunking):
    if chunking not in _indexes:
        if chunking == "large":
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
        _indexes[chunking] = FAISS.from_documents(chunks, embeddings)
    return _indexes[chunking]

def get_model(name):
    if name not in _models:
        _models[name] = OllamaLLM(model=name)
    return _models[name]

def get_no_rag(model_name, query, llm):
    # WITHOUT retrieval; same for every config that shares (model, query) -> cache
    key = (model_name, query)
    if key not in _norag:
        _norag[key] = llm.invoke(query).replace("\n", " ").strip()
    return _norag[key]

# ---- two prompts: strict (grounded) vs loose (no grounding) ----
# STRICT: encourages abstention instead of hallucination
GROUNDED_PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say that the document does not
state it. Do not make up facts.
 
Context:
{context}
 
Question:
{question}
 
Answer:"""
 
# LOOSE: model may fall back on its own knowledge -> fewer abstentions, more hallucination
LOOSE_PROMPT = """Use the context below if it helps. You may also use your own
knowledge to give a complete answer.
 
Context:
{context}
 
Question:
{question}
 
Answer:"""
 
def get_prompt(grounding):
    return GROUNDED_PROMPT if grounding else LOOSE_PROMPT
 
# ---- run the matrix ----
rows = []
for cfg in configs:
    print(f"\n=== {cfg['name']}  (model={cfg['model']}, grounding={cfg['grounding']}) ===")
    index = get_index(cfg["chunking"])
    llm = get_model(cfg["model"])
    prompt_template = get_prompt(cfg["grounding"])   # NEW: pick prompt per config
 
    for q in questions:
        for lang in q.get("languages", ["en"]):
            qkey = "question_ko" if lang == "ko" else "question_en"
            if qkey not in q:
                continue
            query = q[qkey]
 
            # retrieve (similarity / mmr / hyde)
            if cfg["hyde"]:
                draft = llm.invoke(f"Write a short paragraph that could answer this question:\n{query}\nParagraph:")
                results = index.similarity_search(query + "\n" + draft, k=3)
            elif cfg["search"] == "mmr":
                results = index.max_marginal_relevance_search(query, k=3, fetch_k=20)
            else:
                results = index.similarity_search(query, k=3)
 
            context = "\n\n".join(d.page_content for d in results)
            rag_answer = llm.invoke(prompt_template.format(context=context, question=query))  # NEW
            no_rag_answer = get_no_rag(cfg["model"], query, llm)
 
            retrieved_docs = [d.metadata.get("doc", "") for d in results]
            wanted = [s.strip() for s in q["source"].replace("+", ",").split(",") if s.strip()]
            hit = any(any(w in rd for rd in retrieved_docs) for w in wanted)
 
            rows.append({
                "config": cfg["name"],
                "model": cfg["model"],
                "chunking": cfg["chunking"],
                "search": "hyde" if cfg["hyde"] else cfg["search"],
                "grounding": int(cfg["grounding"]),   # NEW column: 1=strict, 0=loose
                "id": q["id"],
                "tier": q.get("tier", ""),
                "discriminator": q.get("discriminator", ""),
                "lang": lang,
                "source": q["source"],
                "expected_behavior": q.get("expected_behavior", ""),
                "expected_answer": q.get("expected_answer", ""),
                "query": query,
                "rag_answer": rag_answer.replace("\n", " ").strip(),
                "no_rag_answer": no_rag_answer,
                "retrieved_docs": "|".join(retrieved_docs),
                "retrieval_hit": int(hit),
                # ---- fill these by hand after the run ----
                "answer_correct": "",     # 0/1  (RAG 답이 정답인가)
                "abstained": "",          # 0/1  (모른다/없다고 기권했나 — G1·G2용)
                "hallucinated": "",       # 0/1  (없는 사실을 지어냈나)
                "faithful": "",           # 0/1  (모든 주장이 retrieved context에 근거하나)
            })
            print(f"  {q['id']:>3} [{lang}]  hit={int(hit)}")
 
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows -> {OUT_CSV.name}")
print("Manual scoring columns: answer_correct / abstained / hallucinated / faithful")
print("TIP: compare exp5/exp6 (loose) against baseline/exp4 (strict) -> abstain down, hallucinated up")