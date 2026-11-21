// Hybrid retrieval in the browser: BM25 (exact words) + vector cosine (meaning),
// fused with weighted reciprocal rank fusion. Upload your own PDF - it is parsed,
// chunked and embedded entirely on your device.
const RRF_K = 60, W_BM25 = 0.45, W_VEC = 0.55, TOP_K = 4, MAX_UPLOAD_PAGES = 15;

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const status = (t, spin = false) => { statusEl.innerHTML = spin ? `<span class="spin"></span> ${t}` : t; };

let chunks = null;      // [{id, doc, page, text, vec}]
let bm25 = null;        // {docTokens, idf, avgdl, dl, N}
let embedder = null;    // transformers.js pipeline, lazy

const tokenize = (t) => t.toLowerCase().match(/[a-z0-9]+/g) || [];

// ---------- BM25 ----------
function buildBM25(docs) {
  const docTokens = docs.map((d) => tokenize(d.text));
  const df = {}, dl = docTokens.map((t) => t.length);
  const avgdl = dl.reduce((a, b) => a + b, 0) / dl.length;
  for (const toks of docTokens) for (const t of new Set(toks)) df[t] = (df[t] || 0) + 1;
  const N = docs.length, idf = {};
  for (const [t, f] of Object.entries(df))
    idf[t] = Math.log(1 + (N - f + 0.5) / (f + 0.5));
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

// ---------- Embeddings (free local model, hosted on this site) ----------
async function getEmbedder() {
  if (!embedder) {
    status("Loading the free embedding model into your browser (one time, ~30 MB)...", true);
    const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2");
    env.allowLocalModels = true;
    env.allowRemoteModels = false;
    env.localModelPath = "/models/";
    embedder = await pipeline("feature-extraction", "Xenova/bge-small-en-v1.5", { dtype: "q8" });
  }
  return embedder;
}

async function embedOne(text) {
  const emb = await getEmbedder();
  const out = await emb(text, { pooling: "mean", normalize: true });
  return Array.from(out.data);
}

async function vectorScores(query) {
  const q = await embedOne(query);
  return chunks.map((c) => c.vec.reduce((s, x, i) => s + x * q[i], 0));
}

// ---------- Hybrid fusion ----------
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

// ---------- PDF upload (parsed on your device with pdf.js) ----------
function chunkPages(pages, docName) {
  const CH = 800, OV = 150, out = [];
  for (const [page, text] of pages) {
    const words = text.split(/\s+/).filter(Boolean);
    let cur = [], len = 0;
    for (const w of words) {
      cur.push(w); len += w.length + 1;
      if (len >= CH) { out.push({ doc: docName, page, text: cur.join(" ") });
        // overlap: keep the tail worth ~OV chars
        const tail = []; let tl = 0;
        for (let i = cur.length - 1; i >= 0 && tl < OV; i--) { tail.unshift(cur[i]); tl += cur[i].length + 1; }
        cur = tail; len = tl; }
    }
    if (cur.length) out.push({ doc: docName, page, text: cur.join(" ") });
  }
  return out.filter((c) => c.text.trim().length > 40);
}

async function handlePdf(file) {
  status(`Reading ${file.name}...`, true);
  const pdfjs = await import("https://cdn.jsdelivr.net/npm/pdfjs-dist@4.2.67/+esm");
  pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.2.67/build/pdf.worker.min.mjs";
  const buf = await file.arrayBuffer();
  const pdf = await pdfjs.getDocument({ data: buf }).promise;
  const nPages = Math.min(pdf.numPages, MAX_UPLOAD_PAGES);
  const pages = [];
  for (let p = 1; p <= nPages; p++) {
    status(`Reading ${file.name}: page ${p}/${nPages}...`, true);
    const page = await pdf.getPage(p);
    const tc = await page.getTextContent();
    const text = tc.items.map((it) => it.str).join(" ").replace(/\s+/g, " ").trim();
    if (text) pages.push([p, text]);
  }
  if (pdf.numPages > MAX_UPLOAD_PAGES)
    status(`Note: only the first ${MAX_UPLOAD_PAGES} of ${pdf.numPages} pages are indexed in this browser demo.`);
  const pieces = chunkPages(pages, file.name);
  if (!pieces.length) { status(`No readable text found in ${file.name} (scanned PDFs need OCR).`); return; }
  const emb = await getEmbedder();
  const newChunks = [];
  for (let i = 0; i < pieces.length; i++) {
    status(`Embedding ${file.name}: ${i + 1}/${pieces.length} passages...`, true);
    const out = await emb(pieces[i].text, { pooling: "mean", normalize: true });
    newChunks.push({ id: i, ...pieces[i], vec: Array.from(out.data) });
  }
  chunks = newChunks;
  bm25 = buildBM25(chunks);
  $("docbadge").textContent = `Searching: ${file.name} (${chunks.length} passages)`;
  $("reset").style.display = "";
  status(`Indexed ${chunks.length} passages from ${file.name}. Ask away.`);
}

async function loadSample() {
  chunks = await (await fetch("chunks.json")).json();
  bm25 = buildBM25(chunks);
  $("docbadge").textContent = "Searching the sample: spotify_web_app_architecture.pdf (19 passages)";
  $("reset").style.display = "none";
  status(`Ready: ${chunks.length} passages indexed. Ask away.`);
}

// ---------- Render + answer ----------
function renderHits(hits) {
  $("hits").innerHTML = hits
    .map((h, i) => {
      const r = [];
      if (h.bmRank) r.push(`<span class="chip">BM25 #${h.bmRank}</span>`);
      if (h.vecRank) r.push(`<span class="chip">meaning #${h.vecRank}</span>`);
      return `<div class="hit"><span class="cite">${i + 1}. ${h.doc} · page ${h.page}</span>${r.join("")}` +
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
  $("answer").style.display = "none";
  try {
    const hits = await retrieve(q, mode);
    renderHits(hits);
    const key = $("groq").value.trim();
    if (key) {
      localStorage.setItem("groq_key", key);
      status("Asking Groq (free tier)...", true);
      $("answer").textContent = await groqAnswer(q, hits, key);
      $("answer").style.display = "";
      status(`Top ${hits.length} passages (${mode}) + Groq answer.`);
    } else {
      status(`Top ${hits.length} passages (${mode}).`);
    }
  } catch (e) {
    status("Error: " + e.message);
  } finally {
    $("go").disabled = false;
  }
}

// ---------- Wire up ----------
const drop = $("drop");
drop.onclick = () => $("file").click();
$("file").onchange = (e) => e.target.files[0] && handlePdf(e.target.files[0]).catch((e) => status("Error: " + e.message));
["dragover", "dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.toggle("over", ev === "dragover");
    if (ev === "drop" && e.dataTransfer.files[0])
      handlePdf(e.dataTransfer.files[0]).catch((err) => status("Error: " + err.message));
  })
);
$("reset").onclick = () => loadSample();
$("go").onclick = ask;
$("q").addEventListener("keydown", (e) => e.key === "Enter" && ask());
const saved = localStorage.getItem("groq_key");
if (saved) $("groq").value = saved;
loadSample();
