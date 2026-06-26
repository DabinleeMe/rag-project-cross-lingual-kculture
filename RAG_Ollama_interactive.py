"""
RAG_Korean_interactive.py  (torch-free, Korean K-culture corpus)
================================================================

One-time setup (Ollama running):
    ollama pull llama3
    ollama pull nomic-embed-text
    # for better Korean answers:  ollama pull qwen2.5 or ollama pull exaone3.5  
Python packages (no torch):
    pip install langchain-community langchain-text-splitters langchain-ollama faiss-cpu pypdf
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# 💡 torch-free: embeddings: HuggingFaceEmbeddings  ->  OllamaEmbeddings 
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# ---- pick your model ----
MODEL = "exaone3.5"   # for better korean, change to "qwen2.5 or exaone3.5" and pull

# for multiple corpus 
pdf_files = [
    "Data/01_kpop_demon_hunters.pdf",
    "Data/02_bts.pdf",
    "Data/03_blackpink.pdf",
    "Data/04_squid_game.pdf",
    "Data/05_parasite.pdf",
    "Data/06_Hallyu.pdf",
    "Data/07_K-pop.pdf",
]

documents = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())


# OLLAMA CONFIG
local_model = OllamaLLM(model=MODEL)

# LOAD PDFs
print("Loading PDFs. . .")
print(f"Pages Loaded: {len(documents)}")

# load first page to check it's the right document
print("\nFIRST PAGE:\n")
print(documents[0].page_content[:1000])

# CHUNKING
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"\nChunks Created: {len(chunks)}")

# Show first 3 chunks
for i in range(min(3, len(chunks))):
    print("\n")
    print("=" * 60)
    print(f"CHUNK {i+1}")
    print("=" * 60)
    print(chunks[i].page_content)

# EMBEDDINGS  (💡 torch-free)
print("\nCreating Embeddings. . .")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector = embeddings.embed_query(chunks[0].page_content)
print("\nVector Length:")
print(len(vector))
print("\nFirst 60 Numbers:")
print(vector[:60])

# VECTORSTORE
print("\nCreating FAISS Database. . .")
clean_chunks = []
for doc in chunks:
    if doc and hasattr(doc, "page_content") and isinstance(doc.page_content, str):
        if doc.page_content.strip():
            clean_chunks.append(doc)

vector_db = FAISS.from_documents(clean_chunks, embeddings)
print(f"Vectors Stored: {vector_db.index.ntotal}")

# ========================================
# ASK QUESTIONS IN A LOOP  ('q' For quit)
# ========================================
while True:
    query = input("\nAsk a question about the K-culture (or 'q' to quit): ")

    # 종료 조건
    if query.strip().lower() in {"q", "quit", "exit"}:
        print("Bye! 👋")
        break
    if not query.strip():        # 빈 입력이면 다시 물어봄
        continue

    print("\nYou asked:", query)

    print("STEP 1")
    results = vector_db.similarity_search(query, k=10)

    print("STEP 2")
    print("\nRETRIEVED CHUNKS\n")
    for i, doc in enumerate(results):
        print("\n")
        print("=" * 60)
        print(f"CHUNK {i+1}")
        print("=" * 60)
        print(doc.page_content[:1000])

    # context + prompt
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"""

Answer only from the context

Context:
{context}

Question:
{query}

Answer:
"""

    # ===== WITH RAG =====
    print("\n⭐⭐Generating Answer with RAG via Ollama... \n")
    response = local_model.invoke(prompt)
    print("=" * 60)
    print("⭐OLLAMA ANSWER (WITH RAG):")
    print("=" * 60)
    print(response)

    # ===== WITHOUT RAG =====
    print("\n❌❌Generating Answer without RAG via Ollama... \n")
    normal_response = local_model.invoke(query)
    print("=" * 60)
    print("❌WITHOUT RAG:")
    print("=" * 60)
    print(normal_response)


# # ask question
# query = input("\nAsk a question about the K-culture: ")
# print("\nYou asked: ")
# print(query)

# print("STEP 1")
# results = vector_db.similarity_search(query, k=3)

# print("STEP 2")
# print("\nTOP 3 RETRIEVED CHUNKS\n")
# for i, doc in enumerate(results):
#     print("\n")
#     print("=" * 60)
#     print(f"CHUNK {i+1}")
#     print("=" * 60)
#     print(doc.page_content[:1000])

# # context + prompt
# context = "\n\n".join([doc.page_content for doc in results])
# prompt = f"""

# Answer only from the context

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

# # ========================================
# # WITH RAG
# # ========================================
# print("\n⭐⭐Generating Answer with RAG via Ollama... \n")
# response = local_model.invoke(prompt)
# print("=" * 60)
# print("⭐OLLAMA ANSWER (WITH RAG):")
# print("=" * 60)
# print(response)

# # ========================================
# # WITHOUT RAG
# # ========================================
# print("\n❌❌Generating Answer without RAG via Ollama... \n")
# normal_response = local_model.invoke(query)
# print("=" * 60)
# print("❌WITHOUT RAG:")
# print("=" * 60)
# print(normal_response)

