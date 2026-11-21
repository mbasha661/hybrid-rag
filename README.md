# Hybrid Vector RAG (LangChain + local Chroma DB)

A small, complete RAG app by Mahaboob Basha. Drop PDFs in `data/`, index them
into a **local Chroma vector database on disk**, and ask questions. Every
question is answered by **hybrid retrieval**: BM25 keyword search plus vector
semantic search, fused into one ranked list. Answers cite the file and PDF page.

Free resources only: local embeddings, local vector DB, and a free LLM
(Groq free tier or local Ollama) for the answer step. No paid API, no cloud
database, no deploy needed.

## Live demo

A small browser version of this project (same BM25 + vector + RRF pipeline
running in your browser on the sample PDF) is deployed at:
https://hybrid-rag-mbasha661-1852s-projects.vercel.app
Its source is in `web/`.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py ingest              # indexes data/*.pdf into ./chroma_db (once)
python main.py ask "How are songs delivered to listeners over the internet?"
```

First run downloads the free embedding model once (~100 MB), then everything
works offline except the optional Groq answer mode.

## The pipeline

```text
1. Ingest    data/*.pdf -> PyPDFLoader, one document per page
2. Chunk     RecursiveCharacterTextSplitter (800 chars, 150 overlap)
3. Embed     FastEmbed BAAI/bge-small-en-v1.5, free local model, no API key
4. Store     Chroma vector DB persisted on disk in ./chroma_db
5. Retrieve  BM25Retriever (exact words) + Chroma vector search (meaning),
             fused by EnsembleRetriever with weighted reciprocal rank fusion
6. Answer    top passages + question -> free LLM, or evidence-only mode
```

Why hybrid: keyword search alone ranks the right passage only when your
question uses the PDF's exact words. Vector search understands meaning, so
"How are songs delivered to listeners?" finds the HLS streaming section even
though those words never appear in the question. Fusing both lists keeps exact
jargon matches strong and fixes paraphrased questions. The included test
proves this on the sample PDF.

## Answer modes

| Mode | Setup | Data handling |
|---|---|---|
| `evidence` (default) | none | fully local: shows the retrieved passages with page citations |
| `--llm ollama` | install Ollama, `ollama pull qwen2.5:3b` | question + passages go only to your own machine |
| `--llm groq` | free key from <https://console.groq.com>, `export GROQ_API_KEY=...` | question + retrieved passages are sent to Groq's free tier |

Only use a **free** Groq account; free quotas and available models change over
time. Override the model with `GROQ_MODEL` if needed. No key is bundled.

## Add your own documents

1. Delete `chroma_db/` to start fresh.
2. Drop your PDFs into `data/`.
3. Run `python main.py ingest` again.

## Files

```text
main.py            the complete app (single file)
requirements.txt   free dependencies only
data/              your PDFs (the Spotify app architecture PDF is the sample)
chroma_db/         the local vector DB, created by `ingest` (do not commit)
tests/test_hybrid.py   proof it works: python tests/test_hybrid.py
```

## Test

```bash
python tests/test_hybrid.py
```

Checks ingestion, vector DB save/reload, keyword search on an exact-term
question, and that hybrid retrieval beats keyword-only search on a
paraphrased question.

## Notes

- This is a learning project, not a Spotify product. The bundled PDF is an
  architecture/feature document used only as demo content.
- Nothing here deploys or publishes anything. Everything runs on your machine.


### Vector + Knowledge Graph RAG

`python main.py ask "your question"` now combines the original BM25/vector
retriever with a local concept graph (`graph_rag.py`). Concepts are linked
when they occur in the same sentence; query concepts seed two-hop traversal.
Each link retains source chunks/pages and is shown as evidence, never a
verified factual relationship. Both lanes feed a single LLM call.
Use `--retrieval vector` for the original BM25/vector behavior. No graph
service, paid API, or extra package is needed. The Vercel UI implements the
same lightweight graph approach and shows matching links and source pages.
