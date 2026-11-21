# Hybrid Vector RAG (LangChain + local Chroma DB)

A small, complete RAG app by Mahaboob Basha. Drop PDFs in `data/`, index them
into a **local Chroma vector database on disk**, and ask questions. Every
question is answered by **hybrid retrieval**: BM25 keyword search plus vector
semantic search, fused into one ranked list. Answers cite the file and PDF page.

Free resources only: local embeddings, local vector DB, and a free LLM
(Groq free tier or local Ollama) for the answer step. No paid API, no cloud
database, no deploy needed.

## Repository layout

```text
Vector RAG/   the hybrid BM25 + vector app: main.py, data/, requirements.txt, tests/
Graph RAG/    the local concept-graph lane: graph_rag.py and its tests
web/          combined browser demo of both lanes (the Vercel site below)
```

`Vector RAG/main.py` combines both lanes by default: it imports
`Graph RAG/graph_rag.py` and fuses graph results with the BM25/vector list.

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
pip install -r "Vector RAG/requirements.txt"
cd "Vector RAG"
python main.py ingest              # indexes data/*.pdf into ./chroma_db (once)
python main.py ask "How are songs delivered to listeners over the internet?"
```

First run downloads the free embedding model once (~100 MB), then everything
works offline except the optional Groq answer mode.

## Run with Docker

One command starts the web demo locally at <http://localhost:8080>
(the same UI as the Vercel site):

```bash
./web/fetch-model.sh     # once: downloads the free embedding model files
docker compose up
```

Run the CLI app (Vector RAG + Graph RAG) on demand:

```bash
docker compose run --rm rag ask "How are songs delivered to listeners over the internet?"
```

The first `ask` indexes the bundled sample PDF into a persistent Docker
volume, so later questions start fast. For a written answer with the free
Groq tier instead of local evidence mode:

```bash
GROQ_API_KEY=your_free_key docker compose run --rm rag ask "your question" --llm groq
```

To index your own PDFs, uncomment the `./data` volume line in
`docker-compose.yml`, drop your PDFs into a `data/` folder, remove the old
index with `docker volume rm hybrid-rag_rag-db`, and run
`docker compose run --rm rag ingest`.

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
Vector RAG/main.py            the complete app (single file)
Vector RAG/requirements.txt   free dependencies only
Vector RAG/data/              your PDFs (the Spotify app architecture PDF is the sample)
Vector RAG/chroma_db/         the local vector DB, created by `ingest` (do not commit)
Vector RAG/tests/test_hybrid.py   proof it works: python tests/test_hybrid.py
Graph RAG/graph_rag.py        the local concept-graph lane (no extra dependencies)
Graph RAG/tests/test_graph.py     graph unit tests
web/                          combined browser demo (source of the Vercel site)
```

## Test

```bash
cd "Vector RAG" && python tests/test_hybrid.py
cd "Graph RAG" && python tests/test_graph.py
```

The hybrid test checks ingestion, vector DB save/reload, keyword search on an
exact-term question, and that hybrid retrieval beats keyword-only search on a
paraphrased question. The graph test checks seed matching, two-hop traversal,
edge provenance, no-match behavior, and fusion deduplication.

## Notes

- This is a learning project, not a Spotify product. The bundled PDF is an
  architecture/feature document used only as demo content.
- Nothing here deploys or publishes anything. Everything runs on your machine.


### Vector + Knowledge Graph RAG

`python main.py ask "your question"` now combines the original BM25/vector
retriever with a local concept graph (`Graph RAG/graph_rag.py`). Concepts are linked
when they occur in the same sentence; query concepts seed two-hop traversal.
Each link retains source chunks/pages and is shown as evidence, never a
verified factual relationship. Both lanes feed a single LLM call.
Use `--retrieval vector` for the original BM25/vector behavior. No graph
service, paid API, or extra package is needed. The Vercel UI implements the
same lightweight graph approach and shows matching links and source pages.
