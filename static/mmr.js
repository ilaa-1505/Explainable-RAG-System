let _p3 = {
    coords: null,   // [[x,y], ...]  UMAP coords, local index
    sims: null,   // [float, ...]  query-doc similarities
    previews: null,   // [str, ...]    chunk previews
    docIndices: null,   // [int, ...]    global doc indices
    mmrSelected: null,   // set of global doc indices — current MMR selection
    noMmrSelected: null,   // set — pure relevance top-10
    lambda: 0.7,
    sliderTimeout: null,
};

function renderMMRPanel(mmrData) {
    const container = document.getElementById("mmr-panel");
    if (!mmrData || !mmrData.umap_coords) {
        container.innerHTML = `<div class="p3-panel"><div class="panel-title">🔵 MMR Visualization</div><p style="color:#94a3b8;font-size:13px">UMAP unavailable — install umap-learn</p></div>`;
        return;
    }

    _p3.coords = mmrData.umap_coords;
    _p3.sims = mmrData.sims;
    _p3.previews = mmrData.doc_previews;
    _p3.docIndices = mmrData.doc_indices;
    _p3.mmrSelected = new Set(mmrData.mmr_selected);
    _p3.noMmrSelected = new Set(mmrData.no_mmr_selected);
    _p3.lambda = 0.7;

    container.innerHTML = `
        <div class="p3-panel">
            <div class="panel-title">🔵 MMR Visualization</div>

            <!-- Lambda slider -->
            <div class="p3-slider-row">
                <span class="p3-slider-label">λ = <b id="p3-lambda-val">0.70</b></span>
                <input id="p3-lambda" type="range" min="0" max="1" step="0.05" value="0.7" class="p3-slider">
                <span class="p3-slider-hint">← diversity &nbsp;&nbsp; relevance →</span>
            </div>

            <!-- UMAP + side-by-side row -->
            <div class="p3-top-row">

                <!-- UMAP scatter -->
                <div class="p3-scatter-wrap">
                    <div class="p3-section-label">UMAP Scatter — candidate chunks</div>
                    <canvas id="p3-umap" width="340" height="280" class="p3-canvas"></canvas>
                    <div class="p3-scatter-legend">
                        <span><span class="p3-dot-lg" style="background:#10b981"></span> MMR selected</span>
                        <span><span class="p3-dot-lg" style="background:#ef4444"></span> Rejected</span>
                    </div>
                    <div id="p3-tooltip" class="p3-scatter-tooltip"></div>
                </div>

                <!-- Side by side: No MMR vs MMR -->
                <div class="p3-side-wrap">
                    <div class="p3-side-col">
                        <div class="p3-section-label" style="color:#f59e0b">Without MMR (top relevance)</div>
                        <div id="p3-no-mmr-list" class="p3-chunk-list"></div>
                    </div>
                    <div class="p3-side-col">
                        <div class="p3-section-label" style="color:#10b981">With MMR (λ=<span id="p3-side-lambda">0.70</span>)</div>
                        <div id="p3-mmr-list" class="p3-chunk-list"></div>
                    </div>
                </div>

            </div>

            <!-- Similarity matrix -->
            <div class="p3-section-label" style="margin-top:18px">Similarity Matrix — top 10 MMR candidates</div>
            <div class="p3-matrix-wrap">
                <canvas id="p3-simmatrix" class="p3-canvas"></canvas>
            </div>

        </div>`;

    _drawAll();
    _drawSideBySide();
    _drawSimMatrix(mmrData.sim_matrix);


    document.getElementById("p3-lambda").addEventListener("input", function () {
        _p3.lambda = parseFloat(this.value);
        document.getElementById("p3-lambda-val").textContent = _p3.lambda.toFixed(2);
        document.getElementById("p3-side-lambda").textContent = _p3.lambda.toFixed(2);
        clearTimeout(_p3.sliderTimeout);
        _p3.sliderTimeout = setTimeout(_rerrunMMR, 300);
    });
}

function _drawAll() {
    const canvas = document.getElementById("p3-umap");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const PAD = 24;

    ctx.clearRect(0, 0, W, H);


    _p3.coords.forEach(([x, y], i) => {
        const cx = PAD + x * (W - PAD * 2);
        const cy = PAD + y * (H - PAD * 2);
        const isSelected = _p3.mmrSelected.has(_p3.docIndices[i]);
        const sim = _p3.sims[i] || 0;

        if (isSelected) {
            ctx.beginPath();
            ctx.arc(cx, cy, 11, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(16,185,129,0.18)";
            ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(cx, cy, isSelected ? 8 : 5, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? "#10b981" : "#ef4444";
        ctx.globalAlpha = isSelected ? 1 : 0.55 + sim * 0.45;
        ctx.fill();
        ctx.globalAlpha = 1;

        if (isSelected) {
            ctx.font = "bold 9px Arial";
            ctx.fillStyle = "#065f46";
            ctx.fillText(i, cx + 10, cy + 4);
        }
    });

    canvas.onmousemove = (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const PAD = 24;

        let hit = null;
        _p3.coords.forEach(([x, y], i) => {
            const cx = PAD + x * (W - PAD * 2);
            const cy = PAD + y * (H - PAD * 2);
            if (Math.hypot(mx - cx, my - cy) < 10) hit = i;
        });

        const tip = document.getElementById("p3-tooltip");
        if (hit !== null) {
            const isSelected = _p3.mmrSelected.has(_p3.docIndices[hit]);
            tip.innerHTML = `
                <b>${isSelected ? "✅ Selected" : "❌ Rejected"}</b> — chunk ${hit}<br>
                sim: ${(_p3.sims[hit] || 0).toFixed(3)}<br>
                <span style="color:#cbd5e1">${escP3(_p3.previews[hit] || "")}</span>`;
            tip.style.display = "block";
            tip.style.left = (mx + 12) + "px";
            tip.style.top = (my + 12) + "px";
        } else {
            tip.style.display = "none";
        }
    };
    canvas.onmouseleave = () => {
        const tip = document.getElementById("p3-tooltip");
        if (tip) tip.style.display = "none";
    };
}

function _drawSideBySide() {
    const noMmrEl = document.getElementById("p3-no-mmr-list");
    const mmrEl = document.getElementById("p3-mmr-list");
    if (!noMmrEl || !mmrEl) return;

    function makeList(selectedSet) {
        return _p3.docIndices
            .map((docIdx, i) => ({ docIdx, i, sim: _p3.sims[i] || 0, preview: _p3.previews[i] || "" }))
            .filter(({ docIdx }) => selectedSet.has(docIdx))
            .sort((a, b) => b.sim - a.sim)
            .map(({ i, sim, preview }) => `
                <div class="p3-side-item">
                    <div class="p3-side-sim">${sim.toFixed(3)}</div>
                    <div class="p3-side-text">${escP3(preview)}…</div>
                </div>`)
            .join("");
    }

    noMmrEl.innerHTML = makeList(_p3.noMmrSelected);
    mmrEl.innerHTML = makeList(_p3.mmrSelected);
}

function _drawSimMatrix(simMatrix) {
    if (!simMatrix) return;
    const localSelected = _p3.docIndices
        .map((d, i) => ({ d, i }))
        .filter(({ d }) => _p3.mmrSelected.has(d))
        .map(({ i }) => i)
        .slice(0, 10);

    const N = localSelected.length;
    const CELL = 32;
    const canvas = document.getElementById("p3-simmatrix");
    if (!canvas || N === 0) return;

    canvas.width = N * CELL;
    canvas.height = N * CELL;
    const ctx = canvas.getContext("2d");

    for (let r = 0; r < N; r++) {
        for (let c = 0; c < N; c++) {
            const ri = localSelected[r];
            const ci = localSelected[c];
            const val = (simMatrix[ri] && simMatrix[ri][ci] != null) ? simMatrix[ri][ci] : 0;
            ctx.fillStyle = simHeatColor(val);
            ctx.fillRect(c * CELL, r * CELL, CELL, CELL);

            ctx.font = "9px Arial";
            ctx.fillStyle = val > 0.6 ? "#fff" : "#334155";
            ctx.textAlign = "center";
            ctx.fillText(val.toFixed(2), c * CELL + CELL / 2, r * CELL + CELL / 2 + 3);
        }
    }

    ctx.font = "bold 9px Arial";
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "center";
    for (let i = 0; i < N; i++) {
        ctx.fillText(localSelected[i], i * CELL + CELL / 2, N * CELL + 12);
        ctx.fillText(localSelected[i], -6, i * CELL + CELL / 2 + 3);
    }
}

function simHeatColor(v) {
    const t = Math.max(0, Math.min(1, v));
    const r = Math.round(255 - t * (255 - 99));
    const g = Math.round(255 - t * (255 - 102));
    const b = Math.round(255 - t * (255 - 241));
    return `rgb(${r},${g},${b})`;
}

async function _rerrunMMR() {
    const res = await fetch("/mmr_rerun", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lambda: _p3.lambda })
    });
    const data = await res.json();
    if (data.error) { console.warn("MMR rerun:", data.error); return; }

    _p3.mmrSelected = new Set(data.selected_indices);
    _drawAll();
    _drawSideBySide();
}

function escP3(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}