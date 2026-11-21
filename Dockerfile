# Hybrid Vector + Graph RAG - all-in-one image for the CLI app.
# Builds the "Vector RAG" and "Graph RAG" folders into one container and
# pre-downloads the free local embedding model, so runs work offline.
#
# Build:  docker build -t hybrid-rag .
# Use:    see docker-compose.yml and the README "Run with Docker" section.

FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so code changes do not trigger a full reinstall.
COPY ["Vector RAG/requirements.txt", "./requirements.txt"]
RUN pip install --no-cache-dir -r requirements.txt

# Keep the repo's folder layout: main.py imports the graph lane from the
# sibling "Graph RAG" folder by path.
COPY ["Vector RAG", "./Vector RAG"]
COPY ["Graph RAG", "./Graph RAG"]

# Download the free embedding model (~100 MB) once at build time so the
# first question does not wait on a download and offline runs work.
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5').embed_query('warmup')"

COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

WORKDIR "/app/Vector RAG"
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["--help"]
