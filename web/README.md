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
