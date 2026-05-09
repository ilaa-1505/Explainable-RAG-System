function idfColor(n) {
    const stops = [
        { t: 0.0, r: 148, g: 163, b: 184 }, { t: 0.3, r: 99, g: 102, b: 241 },
        { t: 0.7, r: 139, g: 92, b: 246 }, { t: 1.0, r: 236, g: 72, b: 153 }
    ];
    const v = Math.max(0, Math.min(1, n));
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
        if (v >= stops[i].t && v <= stops[i + 1].t) { lo = stops[i]; hi = stops[i + 1]; break; }
    }
    const t2 = lo.t === hi.t ? 0 : (v - lo.t) / (hi.t - lo.t);
    return `rgb(${Math.round(lo.r + (hi.r - lo.r) * t2)},${Math.round(lo.g + (hi.g - lo.g) * t2)},${Math.round(lo.b + (hi.b - lo.b) * t2)})`;
}

function heatmapColor(v) {
    if (v >= 0) { const i = Math.round(v * 255); return `rgb(255,${255 - i},${255 - i})`; }
    const i = Math.round(-v * 255); return `rgb(${255 - i},${255 - i},255)`;
}

function escHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderQueryPanel(d) {
    const container = document.getElementById("query-panel");
    if (!d) { container.innerHTML = ""; return; }

    const maxIdfTok = d.tokens.reduce((best, t) => t.idf > best.idf ? t : best, d.tokens[0]);

    const pills = d.tokens.map(tok => {
        const bg = idfColor(tok.idf_normalized);
        const alpha = (0.15 + tok.idf_normalized * 0.85).toFixed(2);
        return `<div class="token-pill">
            <div class="pill-text" style="background:${bg};opacity:${alpha};min-width:32px;text-align:center">${escHtml(tok.token.replace('##', '·'))}</div>
            <div class="pill-id">#${tok.id}</div>
            <div class="pill-idf">idf ${tok.idf.toFixed(2)}</div>
        </div>`;
    }).join("");

    const cells = d.embedding.map((v, i) =>
        `<div class="heatmap-cell" style="background:${heatmapColor(v)}" title="dim ${i}: ${v.toFixed(3)}"></div>`
    ).join("");

    const cpx = d.stats.complexity, cpxPct = (cpx * 100).toFixed(0);
    const cpxLabel = cpx < 0.33 ? "Simple" : cpx < 0.66 ? "Moderate" : "Complex";

    const embNote = maxIdfTok
        ? `<div class="embed-note"> <b>"${escHtml(maxIdfTok.token)}"</b> has the highest IDF (${maxIdfTok.idf.toFixed(2)}) — it pulls the embedding vector the most and takes maximum retrieval weight.</div>`
        : "";

    container.innerHTML = `
        <div class="query-panel">
            <div class="panel-title">🔍 Query Analysis</div>
            <div class="token-row">${pills}</div>
            <div class="heatmap-label">Query Embedding — ${d.embedding.length}-dim PCA projection
                <span style="color:#94a3b8;margin-left:8px">■ <span style="color:#ef4444">positive</span> &nbsp;■ <span style="color:#6366f1">negative</span></span>
            </div>
            <div class="heatmap-strip">${cells}</div>
            ${embNote}
            <div class="stats-row">
                <div class="stat-chip"><div class="stat-label">Tokens</div><div class="stat-value">${d.stats.token_count}</div></div>
                <div class="stat-chip"><div class="stat-label">Unique</div><div class="stat-value">${d.stats.unique_tokens}</div></div>
                <div class="stat-chip"><div class="stat-label">Avg IDF</div><div class="stat-value">${d.stats.avg_idf}</div></div>
                <div class="complexity-wrap">
                    <div class="stat-label">Complexity — ${cpxLabel} (${cpxPct}%)</div>
                    <div class="complexity-bar-bg"><div class="complexity-bar-fill" style="width:${cpxPct}%"></div></div>
                </div>
            </div>
        </div>`;
}


const STAGES = [
    { key: "embed", label: "Embed", cls: "seg-embed" },
    { key: "vector", label: "Vector", cls: "seg-vector" },
    { key: "bm25", label: "BM25", cls: "seg-bm25" },
    { key: "hybrid", label: "Hybrid", cls: "seg-hybrid" },
    { key: "mmr", label: "MMR", cls: "seg-mmr" },
    { key: "rerank", label: "Rerank", cls: "seg-rerank" },
    { key: "llm", label: "LLM", cls: "seg-llm" },
];
const SEG_COLORS = {
    "seg-embed": "#6366f1", "seg-vector": "#0ea5e9", "seg-bm25": "#10b981",
    "seg-hybrid": "#f59e0b", "seg-mmr": "#8b5cf6", "seg-rerank": "#ef4444", "seg-llm": "#64748b"
};

function renderLatencyPanel(timings) {
    const container = document.getElementById("latency-panel");
    if (!timings) { container.innerHTML = ""; return; }
    const total = timings.total || STAGES.reduce((s, st) => s + (timings[st.key] || 0), 0);

    const segments = STAGES.map((s, idx) => {
        const ms = timings[s.key] || 0, pct = total > 0 ? (ms / total) * 100 : 0;
        const isLast = idx === STAGES.length - 1;
        const inner = pct > 5 ? ms + "ms" : "";
        return `<div class="latency-segment ${s.cls}" style="${isLast ? 'flex:1' : 'width:' + pct.toFixed(2) + '%'}"
            title="${s.label}: ${ms}ms (${pct.toFixed(1)}%)">${inner}</div>`;
    }).join("");

    const smallLabels = STAGES.map(s => {
        const ms = timings[s.key] || 0, pct = total > 0 ? (ms / total) * 100 : 0;
        if (ms === 0 || pct > 5) return "";
        return `<span class="latency-small-label" style="color:${SEG_COLORS[s.cls]}">${s.label} ${ms}ms</span>`;
    }).filter(Boolean).join("");
    const smallRow = smallLabels ? `<div class="latency-small-row">${smallLabels}</div>` : "";

    const chips = STAGES.map(s => {
        const ms = timings[s.key] || 0, pct = total > 0 ? ((ms / total) * 100).toFixed(1) : "0.0";
        return `<div class="latency-chip">
            <div class="chip-dot" style="background:${SEG_COLORS[s.cls]}"></div>
            ${s.label} <span class="chip-time">${ms}ms</span>
            <span style="color:#94a3b8">${pct}%</span></div>`;
    }).join("");

    const slowest = STAGES.reduce((b, s) => (timings[s.key] || 0) > (timings[b.key] || 0) ? s : b, STAGES[0]);
    const slowestMs = timings[slowest.key] || 0;
    const slowestPct = total > 0 ? Math.round((slowestMs / total) * 100) : 0;
    const slowestCallout = slowestMs > 0
        ? `<div class="slowest-callout"> <b>${slowest.label}</b> is taking the most time — <b>${slowestMs}ms</b> (${slowestPct}% of total)</div>`
        : "";

    const retrieval = STAGES.filter(s => s.key !== "llm");
    const bottleneck = retrieval.reduce((b, s) => (timings[s.key] || 0) > (timings[b.key] || 0) ? s : b, retrieval[0]);
    const bPct = total > 0 ? Math.round((timings[bottleneck.key] / total) * 100) : 0;
    const retrievalCallout = bPct > 15 && slowest.key !== "llm"
        ? `<div class="bottleneck-callout">⚠ Retrieval bottleneck: <b>${bottleneck.label}</b> taking <b>${bPct}%</b> of total (${timings[bottleneck.key]}ms)</div>`
        : "";
    const llmPct = total > 0 ? Math.round(((timings.llm || 0) / total) * 100) : 0;
    const llmCallout = llmPct > 50
        ? `<div class="llm-callout">💬 LLM generation is <b>${llmPct}%</b> of total time (${timings.llm}ms) — retrieval is fast ✅</div>`
        : "";

    container.innerHTML = `
        <div class="latency-panel">
            <div class="panel-title">⏱ Latency Timeline</div>
            <div class="latency-bar">${segments}</div>
            ${smallRow}
            <div class="latency-meta">${chips}<div class="latency-total">Total: ${total}ms</div></div>
            ${slowestCallout}${retrievalCallout}${llmCallout}
        </div>`;
}


function scoreColor(s) {
    if (s >= 0.7) return { bg: "#dcfce7", color: "#166534" };
    if (s >= 0.4) return { bg: "#fef9c3", color: "#854d0e" };
    return { bg: "#fee2e2", color: "#991b1b" };
}

function renderChunksPanel(chunks, scores, raw_scores, sources) {
    const container = document.getElementById("chunks-panel");
    let html = `<h4 style="margin-top:28px;margin-bottom:12px;color:#1e293b">📄 Context Chunks</h4>`;

    const order = scores
        .map((s, i) => ({ i, s }))
        .sort((a, b) => b.s - a.s)
        .map(x => x.i);

    order.forEach(i => {
        const score = scores[i];
        const sc = scoreColor(score);
        const src = sources[i] || {};
        const url = src.url || "";
        const title = src.title || url;

        let displayUrl = url;
        try {
            const u = new URL(url);
            displayUrl = u.hostname + u.pathname.replace(/\/$/, "");
        } catch (_) { }

        const sourceTag = url
            ? `<a class="chunk-source-link" href="${escHtml(url)}" target="_blank" title="${escHtml(title)}">
                   <span class="chunk-source-icon">🔗</span>${escHtml(displayUrl)}
               </a>`
            : "";

        html += `
            <div class="chunk-card">
                <div class="chunk-card-header">
                    <span class="chunk-card-title">Chunk ${i + 1}</span>
                    <span class="chunk-score-badge" style="background:${sc.bg};color:${sc.color}">
                        score ${score.toFixed(3)}
                    </span>
                    ${sourceTag}
                </div>
                <div class="chunk-card-body">
                    <div class="chunk-content">${marked.parse(chunks[i])}</div>
                </div>
            </div>`;
    });

    container.innerHTML = html;
    hljs.highlightAll();
}


async function ask() {
    const query = document.getElementById("query").value.trim();
    if (!query) return;

    document.getElementById("answer").innerHTML = "<p style='color:#94a3b8'>Thinking...</p>";
    document.getElementById("latency-panel").innerHTML = "";
    document.getElementById("query-panel").innerHTML = "<p style='font-size:12px;color:#94a3b8;padding:12px 0'>Analyzing query...</p>";
    document.getElementById("retrieval-panel").innerHTML = "";
    document.getElementById("mmr-panel").innerHTML = "";
    document.getElementById("chunks-panel").innerHTML = "";

    const [askRes, analyzeRes] = await Promise.all([
        fetch("/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) }),
        fetch("/analyze_query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) })
    ]);
    const [data, analysis] = await Promise.all([askRes.json(), analyzeRes.json()]);

    // 1. Answer
    document.getElementById("answer").innerHTML =
        `<h3>Answer:</h3><div class="answer-box">${marked.parse(data.answer)}</div>`;
    hljs.highlightAll();

    // 2. Latency
    renderLatencyPanel(data.timings);

    // 3. Query Analysis
    renderQueryPanel(analysis);

    // 4. Retrieval Comparison
    renderRetrievalTable(data.comparison_rows);

    // 5. MMR
    renderMMRPanel(data.mmr_data);

    // 6. Chunks
    renderChunksPanel(data.chunks, data.scores, data.raw_scores, data.sources);
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("query").addEventListener("keydown", e => {
        if (e.key === "Enter") ask();
    });
});