// Offline cache: app shell + the local embedding model + sample data.
const CACHE = "hybrid-rag-v1";
const PRECACHE = [
  "./", "index.html", "app.js", "chunks.json", "manifest.webmanifest",
  "icon-192.png", "icon-512.png",
  "models/Xenova/bge-small-en-v1.5/config.json",
  "models/Xenova/bge-small-en-v1.5/tokenizer.json",
  "models/Xenova/bge-small-en-v1.5/tokenizer_config.json",
  "models/Xenova/bge-small-en-v1.5/special_tokens_map.json",
  "models/Xenova/bge-small-en-v1.5/vocab.txt",
  "models/Xenova/bge-small-en-v1.5/onnx/model_quantized.onnx",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Cache-first for app files and CDN libraries; network falls back to cache offline.
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then(
      (hit) =>
        hit ||
        fetch(e.request).then((resp) => {
          if (resp.ok || resp.type === "opaque") {
            const copy = resp.clone();
            caches.open(CACHE).then((c) => c.put(e.request, copy));
          }
          return resp;
        })
    )
  );
});
