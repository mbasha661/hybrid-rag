// Hybrid retrieval in the browser: BM25 (exact words) + vector cosine (meaning),
// fused with weighted reciprocal rank fusion. Mirrors the Python project.
const RRF_K = 60, W_BM25 = 0.45, W_VEC = 0.55, TOP_K = 4;

const $ = (id) => document.getElementById(id);
const status = (t) => { $("status").textContent = t; };

let chunks = null;      // [{id, doc, page, text, vec}]
let bm25 = null;        // {docTokens, df, avgdl, idf}
let embedder = null;    // transformers.js pipeline, lazy

const tokenize = (t) => t.toLowerCase().match(/[a-z0-9]+/g) || [];

function buildBM25(docs) {
  const docTokens = docs.map((d) => tokenize(d.text));
  const df = {}, dl = docTokens.map((t) => t.length);
  const avgdl = dl.reduce((a, b) => a + b, 0) / dl.length;
  for (const toks of docTokens) for (const t of new Set(toks)) df[t] = (df[t] || 0) + 1;
  const N = docs.length, idf = {};
  for (const [t, f] of Object.entries(df))
    idf[t] = Math.log(1 + (N - f + 0.5) / (f + 0.5)); // Robertson-Sparck Jones
  return { docTokens, idf, avgdl, dl, N };
}

function bm25Scores(query) {
  const k1 = 1.5, b = 0.75;
  const q = new Set(tokenize(query));
  return bm25.docTokens.map((toks, i) => {
    const tf = {};
    for (const t of toks) tf[t] = (tf[t] || 0) + 1;
    let s = 0;
    for (const t of q) {
      if (!bm25.idf[t] || !tf[t]) continue;
      const f = tf[t], norm = 1 - b + (b * bm25.dl[i]) / bm25.avgdl;
      s += (bm25.idf[t] * f * (k1 + 1)) / (f + k1 * norm);
    }
    return s;
  });
}

async function getEmbedder() {
  if (!embedder) {
    status("Loading the free embedding model into your browser (one time, ~30 MB)...");
    const { pipeline } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2");
    embedder = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", { dtype: "q8" });
  }
  return embedder;
}

async function vectorScores(query) {
  const emb = await getEmbedder();
  const out = await emb(query, { pooling: "mean", normalize: true });
  const q = out.data;
  return chunks.map((c) => c.vec.reduce((s, x, i) => s + x * q[i], 0)); // cosine (normalized)
}

const ranksOf = (scores) => {
  const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
  const r = new Array(scores.length);
  order.forEach(([, i], pos) => (r[i] = pos + 1));
  return r;
};

async function retrieve(query, mode) {
  const bm = bm25Scores(query);
  let vec = mode === "keyword" ? null : await vectorScores(query);
  const bmR = ranksOf(bm), vecR = vec ? ranksOf(vec) : null;
  let fused;
  if (mode === "keyword") fused = bm;
  else if (mode === "semantic") fused = vec;
  else fused = chunks.map((_, i) => W_BM25 / (RRF_K + bmR[i]) + W_VEC / (RRF_K + vecR[i]));
  return chunks
    .map((c, i) => ({ ...c, score: fused[i], bmRank: mode === "semantic" ? null : bmR[i], vecRank: vecR ? vecR[i] : null }))
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_K);
}

function renderHits(hits) {
  $("hits").innerHTML = hits
    .map((h, i) => {
      const r = [];
      if (h.bmRank) r.push(`BM25 #${h.bmRank}`);
      if (h.vecRank) r.push(`semantic #${h.vecRank}`);
      return `<div class="hit"><span class="cite">${i + 1}. ${h.doc} p.${h.page}</span>` +
        `<span class="ranks">${r.join(", ")}</span>` +
        `<div class="txt">${h.text.slice(0, 260)}${h.text.length > 260 ? "…" : ""}</div></div>`;
    })
    .join("");
}

async function groqAnswer(question, hits, key) {
  const evidence = hits.map((h) => `[${h.doc} p.${h.page}]\n${h.text}`).join("\n\n");
  const prompt = `Answer the question using only the passages below. Cite the file and PDF page for every claim, like [file.pdf p.3]. If the passages do not contain the answer, say so plainly.\n\nQuestion: ${question}\n\nPassages:\n${evidence}\n\nAnswer (3-6 sentences):`;
  const resp = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: "openai/gpt-oss-120b", messages: [{ role: "user", content: prompt }], temperature: 0.2 }),
  });
  if (!resp.ok) throw new Error(`Groq error ${resp.status}`);
  return (await resp.json()).choices[0].message.content.trim();
}

async function ask() {
  const q = $("q").value.trim();
  if (!q || !chunks) return;
  const mode = document.querySelector("input[name=mode]:checked").value;
  $("go").disabled = true;
  $("answer").textContent = "";
  try {
    const hits = await retrieve(q, mode);
    renderHits(hits);
    const key = $("groq").value.trim();
    if (key) {
      localStorage.setItem("groq_key", key);
      status("Asking Groq (free tier)...");
      $("answer").textContent = await groqAnswer(q, hits, key);
      status(`Top ${hits.length} passages (${mode}) + Groq answer.`);
    } else {
      status(`Top ${hits.length} passages (${mode}). Add a free Groq key above for a written answer.`);
    }
  } catch (e) {
    status("Error: " + e.message);
  } finally {
    $("go").disabled = false;
  }
}

(async () => {
  chunks = await (await fetch("chunks.json")).json();
  bm25 = buildBM25(chunks);
  status(`Ready: ${chunks.length} passages indexed from spotify_web_app_architecture.pdf. Ask away.`);
  const saved = localStorage.getItem("groq_key");
  if (saved) $("groq").value = saved;
})();
$("go").onclick = ask;
$("q").addEventListener("keydown", (e) => e.key === "Enter" && ask());
