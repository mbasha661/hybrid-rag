"""Hybrid RAG over your own PDFs - LangChain + a local Chroma vector DB.

Pipeline:
  1. Ingest   read every PDF in ./data page by page (PyPDFLoader)
  2. Chunk    split pages into overlapping passages (RecursiveCharacterTextSplitter)
  3. Embed    free local model turns passages into meaning vectors (FastEmbed, no key)
  4. Store    vectors are saved in a local Chroma DB on disk (./chroma_db)
  5. Retrieve each question runs TWO searches: BM25 keyword search + Chroma
              vector search, fused with weighted reciprocal rank fusion
              (LangChain's EnsembleRetriever)
  6. Answer   top passages + your question go to a free LLM:
              Groq free tier (cloud) or Ollama (fully local), or "evidence"
              mode which just shows the passages with page citations.

Setup (free only, Python 3.10+):
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt

Run:
  python main.py ingest                        # index ./data PDFs into ./chroma_db (once)
  python main.py ask "your question"           # retrieve + show cited passages (offline)
  python main.py ask "your question" --llm groq    # free Groq key, set GROQ_API_KEY
  python main.py ask "your question" --llm ollama  # fully local, needs ollama serve
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# langchain-community prints a one-line sunset notice on import; it still works.
warnings.filterwarnings("ignore", message=".*langchain-community is being sunset.*")

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"      # put your PDFs here
DB_DIR = Path(__file__).parent / "chroma_db"   # local vector DB, auto-created
TOP_K = 4

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")   # free tier
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

PROMPT = """Answer the question using only the passages below.
Cite the file and PDF page for every claim, like [file.pdf p.3].
If the passages do not contain the answer, say so plainly.

Question: {question}

Passages:
{passages}

Answer (3-6 sentences):"""


def load_chunks() -> list[Document]:
    """Steps 1+2: read PDFs from ./data and split into overlapping chunks."""
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {DATA_DIR} - drop your PDFs there first.")
    docs = []
    for pdf in pdfs:
        docs.extend(PyPDFLoader(str(pdf)).load())  # one Document per page
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages from {len(pdfs)} PDF(s) -> {len(chunks)} chunks")
    return chunks


def get_db(chunks: list[Document] | None = None) -> Chroma:
    """Steps 3+4: embed chunks with the free local model and store in Chroma.
    On later runs the DB is loaded from disk - no re-embedding needed."""
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        print(f"Loading existing vector DB from {DB_DIR}")
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    if chunks is None:
        chunks = load_chunks()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=str(DB_DIR))
    print(f"Vector DB saved to {DB_DIR}")
    return db


def get_hybrid_retriever(db: Chroma, chunks: list[Document]) -> EnsembleRetriever:
    """Step 5: BM25 (exact words) + Chroma vectors (meaning), fused with RRF."""
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K
    vector = db.as_retriever(search_kwargs={"k": TOP_K})
    # Slight tilt to the vector lane: it generalizes better on paraphrased
    # questions, while BM25 still dominates on exact names and jargon.
    return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.45, 0.55])


def all_chunks_from_db(db: Chroma) -> list[Document]:
    """Rebuild the chunk list from a saved DB (needed for BM25 on later runs)."""
    raw = db.get()
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def cite(doc: Document) -> str:
    src = Path(doc.metadata.get("source", "document")).name
    page = doc.metadata.get("page", 0) + 1  # PyPDFLoader pages are 0-based
    return f"{src} p.{page}"


def answer(question: str, passages: list[Document], llm: str, graph_evidence: str = "") -> str:
    """Step 6: free LLM answer, or just show the passages (evidence mode)."""
    evidence = "\n\n".join(f"[{cite(d)}]\n{d.page_content}" for d in passages)
    if llm == "evidence":
        return evidence
    prompt = PROMPT.format(question=question, passages=evidence + ("\n\nGraph evidence (co-occurrence only, not verified facts):\n" + graph_evidence if graph_evidence else ""))
    if llm == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            raise SystemExit("Set GROQ_API_KEY (free key from https://console.groq.com)")
        from langchain_groq import ChatGroq

        return ChatGroq(model=GROQ_MODEL, temperature=0.2).invoke(prompt).content
    if llm == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=OLLAMA_MODEL, temperature=0.2).invoke(prompt).content
    raise SystemExit(f"Unknown --llm {llm!r}; use evidence, groq or ollama")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Index ./data PDFs into the local vector DB")
    p = sub.add_parser("ask", help="Ask a question against the vector DB")
    p.add_argument("question")
    p.add_argument("--retrieval", choices=["combined", "vector"], default="combined", help="Combine BM25+vectors with a local concept graph (default), or use the original vector lane")
    p.add_argument("--llm", choices=["evidence", "groq", "ollama"], default="evidence")
    args = parser.parse_args()

    if args.command == "ingest":
        if DB_DIR.exists() and any(DB_DIR.iterdir()):
            print(f"{DB_DIR} already exists - delete it to re-index.")
            return
        get_db()
        return

    db = get_db()
    chunks = all_chunks_from_db(db)
    retriever = get_hybrid_retriever(db, chunks)
    passages = retriever.invoke(args.question)
    graph_evidence = ""
    if args.retrieval == "combined":
        # The graph lane lives in the sibling "Graph RAG" folder.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Graph RAG"))
        from graph_rag import combine_results
        passages, links = combine_results(passages, chunks, args.question, TOP_K)
        graph_evidence = "\n".join(f"{a} <co-occurs with> {b} [{'; '.join(cite(chunks[i]) for i in ids[:2])}]" for a,b,ids in links)
        print("Graph links (co-occurrence, not verified relations):\n" + (graph_evidence or "No matching graph concepts."))
    print(f"\nTop {len(passages)} passages ({args.retrieval} retrieval):")
    for i, d in enumerate(passages, 1):
        print(f"  {i}. [{cite(d)}] {d.page_content[:110]}...")
    print("\nAnswer:\n" + answer(args.question, passages, args.llm, graph_evidence))


if __name__ == "__main__":
    main()
