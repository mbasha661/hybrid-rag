# web/ - browser demo (deployed on Vercel)

A static, fully client-side version of the hybrid pipeline: BM25 in JS +
the same bge-small-en-v1.5 embedding model running in the browser
(transformers.js), fused with weighted RRF. No server code at all.
Upload any PDF and it is parsed (pdf.js), chunked, embedded and searched
on your device. Installable as an offline PWA (manifest + service worker
cache the app and the model).

Live: https://hybrid-rag-mbasha661-1852s-projects.vercel.app

## Files

- `index.html`, `app.js` - the demo
- `chunks.json` - the 19 passages from `data/spotify_web_app_architecture.pdf`
  with precomputed bge-small-en-v1.5 vectors (regenerate with the export
  snippet from the project history if `data/` changes)
- `models/` - NOT committed. To run or redeploy locally, download the free
  quantized model files from huggingface.co/Xenova/bge-small-en-v1.5 into
  `models/Xenova/bge-small-en-v1.5/` (`config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`,
  `onnx/model_quantized.onnx`).

The optional Groq key field stores the key only in the browser's localStorage
and sends it only to api.groq.com (free tier).


## Vector + Knowledge Graph RAG

PDF upload feeds both lanes. The vector lane retains BM25 + local embeddings.
The graph lane extracts up to 180 frequent concepts, stores sentence-level
co-occurrence edges with source chunk IDs, and follows two hops from query
concepts. Combined retrieval uses weighted reciprocal-rank fusion
(BM25 0.30, vectors 0.40, graph 0.30). Both passages and matching graph links
feed one Groq API request when you provide your key. Without a key, the app
shows source passages and the question's graph.

This is lightweight local Graph RAG, not Neo4j or LLM-extracted semantic
triples. A link means terms occur together, not that a relationship is true.
English works best. Scanned PDFs need OCR; the demo indexes only the first
15 pages. Uploads stay in memory until refresh. The key stays in page memory
and goes only to Groq, not to this site's server. No key is saved.

Offline: app/model/sample are precached. Ask once online to cache the embedding
runtime; upload once online to cache PDF parsing libraries. LLM answers always
need internet. Uploaded PDFs are not saved for later sessions.
