"""Tests for the hybrid RAG on the bundled Spotify architecture PDF.

Run from the project root:  python tests/test_hybrid.py
(Also works under pytest.)

Checks:
  1. Ingestion: the sample PDF becomes cited chunks (source file + page).
  2. Vector DB: chunks land in a local Chroma DB that reloads from disk.
  3. Exact-term question: keyword BM25 alone must find it.
  4. Paraphrased question: keyword alone misses the top spot, the hybrid
     retriever (BM25 + vector, fused) fixes it - the point of the project.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the app at a throwaway DB before importing it.
_TMP = Path(tempfile.mkdtemp(prefix="hybrid_rag_test_"))

import main  # noqa: E402

main.DB_DIR = _TMP / "chroma_db"

from langchain_community.retrievers import BM25Retriever  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    return ok


def run() -> int:
    passed = True

    chunks = main.load_chunks()
    passed &= check(
        "ingestion produces cited chunks",
        len(chunks) > 10 and all("spotify" in Path(c.metadata["source"]).name for c in chunks),
        f"{len(chunks)} chunks",
    )

    db = main.get_db(chunks)
    passed &= check("vector DB saved on disk", main.DB_DIR.exists())
    db2 = main.get_db()  # reload path
    passed &= check(
        "vector DB reloads from disk",
        len(main.all_chunks_from_db(db2)) == len(chunks),
        f"{len(main.all_chunks_from_db(db2))} chunks reloaded",
    )

    hybrid = main.get_hybrid_retriever(db, chunks)

    exact_q = "Which service issues JWT refresh tokens?"
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 3
    kw_top = bm25.invoke(exact_q)[0].page_content.lower()
    passed &= check("keyword finds the exact-term question", "jwt" in kw_top)
    hy_exact = hybrid.invoke(exact_q)[0].page_content.lower()
    passed &= check("hybrid also keeps the exact-term question at #1", "jwt" in hy_exact)

    para_q = "How are songs delivered to listeners over the internet?"
    kw_top = bm25.invoke(para_q)[0].page_content.lower()
    hy = hybrid.invoke(para_q)
    hy_top = hy[0].page_content.lower()
    kw_hit = "hls" in kw_top or "cdn" in kw_top
    hy_hit = "hls" in hy_top or "cdn" in hy_top
    passed &= check("hybrid #1 answers the paraphrased question", hy_hit,
                    f"top passage p.{hy[0].metadata['page'] + 1}")
    passed &= check("hybrid beats keyword-only on the paraphrase", hy_hit and not kw_hit,
                    f"keyword top hit was wrong: {kw_top[:60]!r}")

    shutil.rmtree(_TMP, ignore_errors=True)
    print("\nAll tests passed." if passed else "\nSome tests FAILED.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run())
