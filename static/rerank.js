// Store full texts in a module-level array — avoids HTML attribute escaping bugs
let _p2FullTexts = [];

function scoreBar(val, color) {
    const pct = (val * 100).toFixed(0);
    return `
        <div class="p2-score-wrap" title="${val.toFixed(4)}">
            <div class="p2-bar-bg">
                <div class="p2-bar-fill" style="width:${pct}%;background:${color}"></div>
            </div>
            <span class="p2-score-num">${val.toFixed(3)}</span>
        </div>`;
}

function rankArrow(delta) {
    if (delta > 0) return `<span class="p2-arrow up"   title="Moved up ${delta}">▲${delta}</span>`;
    if (delta < 0) return `<span class="p2-arrow down" title="Moved down ${Math.abs(delta)}">▼${Math.abs(delta)}</span>`;
    return `<span class="p2-arrow flat" title="No change">—</span>`;
}

function rowColor(rerank, passed) {
    if (!passed) return "rgba(239,68,68,0.06)";
    if (rerank >= 0.7) return "rgba(16,185,129,0.08)";
    if (rerank >= 0.4) return "rgba(245,158,11,0.07)";
    return "rgba(99,102,241,0.05)";
}

function borderColor(rerank, passed) {
    if (!passed) return "#fca5a5";
    if (rerank >= 0.7) return "#6ee7b7";
    if (rerank >= 0.4) return "#fcd34d";
    return "#c7d2fe";
}

function escHtml(s) {
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function renderRetrievalTable(rows) {
    const container = document.getElementById("retrieval-panel");
    if (!rows || !rows.length) { container.innerHTML = ""; return; }

    _p2FullTexts = rows.map(r => r.text_full || "");

    const topRow = [...rows].sort((a, b) => b.rerank_score - a.rerank_score)[0];

    const tableRows = rows.map((r, i) => {
        const isTop = r.idx === topRow.idx;
        const bg = rowColor(r.rerank_score, r.passed_threshold);
        const border = borderColor(r.rerank_score, r.passed_threshold);
        const dropped = !r.passed_threshold ? `<span class="p2-dropped">DROPPED</span>` : "";
        const winner = isTop ? `<span class="p2-winner" title="Highest combined rerank score">★ top</span>` : "";
        const preview = escHtml((r.text_preview || "").trim());

        return `
            <tr class="p2-row" style="background:${bg};border-left:3px solid ${border}">
                <td class="p2-td p2-rank">#${r.pre_rank + 1}</td>
                <td class="p2-td p2-rank">#${r.post_rank + 1} ${winner}${dropped}</td>
                <td class="p2-td p2-preview-cell" data-idx="${i}">
                    <div class="p2-preview">${preview}</div>
                    <div class="p2-tooltip"></div>
                </td>
                <td class="p2-td">${scoreBar(r.vector_score, "#0ea5e9")}</td>
                <td class="p2-td">${scoreBar(r.bm25_score, "#10b981")}</td>
                <td class="p2-td">${scoreBar(r.hybrid_score, "#f59e0b")}</td>
                <td class="p2-td">${scoreBar(r.rerank_score, r.passed_threshold ? "#6366f1" : "#ef4444")}</td>
                <td class="p2-td p2-delta">${rankArrow(r.rank_delta)}</td>
            </tr>`;
    }).join("");

    container.innerHTML = `
        <div class="p2-panel">
            <div class="panel-title">📊 Retrieval Comparison</div>
            <div class="p2-legend">
                <span><span class="p2-dot" style="background:#0ea5e9"></span>Vector</span>
                <span><span class="p2-dot" style="background:#10b981"></span>BM25</span>
                <span><span class="p2-dot" style="background:#f59e0b"></span>Hybrid</span>
                <span><span class="p2-dot" style="background:#6366f1"></span>Reranker</span>
                <span class="p2-legend-sep">|</span>
                <span><span class="p2-dot" style="background:#6ee7b7;border:1px solid #10b981"></span>passed</span>
                <span><span class="p2-dot" style="background:#fca5a5;border:1px solid #ef4444"></span>dropped</span>
            </div>
            <div class="p2-table-wrap">
                <table class="p2-table">
                    <thead>
                        <tr>
                            <th class="p2-th">Original Rank</th>
                            <th class="p2-th">Rerank</th>
                            <th class="p2-th">Chunk</th>
                            <th class="p2-th">Vector</th>
                            <th class="p2-th">BM25</th>
                            <th class="p2-th">Hybrid</th>
                            <th class="p2-th">Reranker</th>
                            <th class="p2-th">Δ</th>
                        </tr>
                    </thead>
                    <tbody>${tableRows}</tbody>
                </table>
            </div>
        </div>`;

    container.querySelectorAll(".p2-preview-cell").forEach(cell => {
        const tooltip = cell.querySelector(".p2-tooltip");
        const idx = parseInt(cell.dataset.idx);
        tooltip.textContent = _p2FullTexts[idx] || "";

        cell.addEventListener("mouseenter", () => tooltip.classList.add("visible"));
        cell.addEventListener("mouseleave", () => tooltip.classList.remove("visible"));
        cell.addEventListener("mousemove", e => {
            const spaceBelow = window.innerHeight - e.clientY;
            tooltip.style.left = "0px";
            if (spaceBelow < 220) {
                tooltip.style.bottom = "100%";
                tooltip.style.top = "auto";
            } else {
                tooltip.style.top = "100%";
                tooltip.style.bottom = "auto";
            }
        });
    });
}