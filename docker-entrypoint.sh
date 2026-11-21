#!/bin/sh
set -e

cd "/app/Vector RAG"

# Index the bundled sample PDF on first run, so `ask` works immediately.
# chroma_db lives on a named volume, so this happens only once.
if [ ! -d chroma_db ]; then
  echo "First run: indexing data/*.pdf into chroma_db ..."
  python main.py ingest
fi

exec python main.py "$@"
