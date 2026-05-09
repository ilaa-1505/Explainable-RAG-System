import os
import sys

DEBUG = False  # ← set False to hide ALL noise

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import time
import numpy as np
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
from sklearn.decomposition import PCA

from src.retrieval.query import (
    retrieve, embed_query, bm25_index, docs_all, metas_all,
    mmr_from_embs, _session
)
from src.generation.generate import generate_answer, build_prompt, build_context

class SuppressOutput:
    def __enter__(self):
        if not DEBUG:
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, *args):
        if not DEBUG:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._stdout
            sys.stderr = self._stderr

app = Flask(__name__)

with SuppressOutput():
    _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


# -----------------------------
# BM25 IDF
# -----------------------------
def _build_idf(bm25):
    return {term: max(0.0, float(val)) for term, val in bm25.idf.items()}

_idf_map = _build_idf(bm25_index)
_max_idf = max(_idf_map.values()) if _idf_map else 1.0


# -----------------------------
# PCA FIT
# -----------------------------
def _fit_pca(n_components=128):
    import random
    from sentence_transformers import SentenceTransformer

    sample = random.sample(docs_all, min(200, len(docs_all)))

    with SuppressOutput():
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        embs = model.encode(sample, normalize_embeddings=True)

    n_comp = min(n_components, embs.shape[0], embs.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(embs)
    return pca


_pca = _fit_pca(128)


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze_query", methods=["POST"])
def analyze_query():
    data = request.json
    query = data.get("query", "")

    enc = _tokenizer(query, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    tokens_raw = _tokenizer.convert_ids_to_tokens(input_ids)

    special = set(_tokenizer.all_special_tokens)
    tokens = [
        {"token": t, "id": int(i)}
        for t, i in zip(tokens_raw, input_ids)
        if t not in special
    ]

    for tok in tokens:
        word = tok["token"].lstrip("##").lower()
        raw_idf = _idf_map.get(word, 0.0)
        tok["idf"] = round(raw_idf, 4)
        tok["idf_normalized"] = round(raw_idf / _max_idf, 4) if _max_idf else 0.0

    idf_vals = [t["idf"] for t in tokens]
    avg_idf = round(sum(idf_vals) / len(idf_vals), 4) if idf_vals else 0.0
    unique_toks = len({t["token"] for t in tokens})
    complexity = round(min(1.0, (len(tokens) / 20) * 0.4 + (avg_idf / _max_idf) * 0.6), 3)

    q_emb = embed_query(query)
    projected = _pca.transform(q_emb.reshape(1, -1))[0]
    p_min, p_max = projected.min(), projected.max()

    normed = (
        ((projected - p_min) / (p_max - p_min) * 2 - 1).tolist()
        if p_max != p_min else [0.0] * len(projected)
    )

    return jsonify({
        "tokens": tokens,
        "embedding": [round(v, 4) for v in normed],
        "stats": {
            "token_count": len(tokens),
            "unique_tokens": unique_toks,
            "avg_idf": avg_idf,
            "complexity": complexity,
        }
    })


@app.route("/mmr_rerun", methods=["POST"])
def mmr_rerun():
    if _session["query_emb"] is None:
        return jsonify({"error": "No active session. Run a query first."}), 400

    data = request.json
    lambda_param = float(data.get("lambda", 0.7))
    lambda_param = max(0.0, min(1.0, lambda_param))

    selected = mmr_from_embs(
        _session["query_emb"],
        _session["doc_indices"],
        _session["embs"],
        k=10,
        lambda_param=lambda_param
    )

    return jsonify({
        "selected_indices": selected,
        "selected_local": [
            _session["doc_indices"].index(s) for s in selected
        ],
        "lambda": lambda_param,
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")

    if DEBUG:
        print(f"\n[API QUERY]: {query}\n")

    t0 = time.perf_counter()
    results, debug = retrieve(query)
    t_retrieve = time.perf_counter() - t0

    results = sorted(results, key=lambda x: (
        x["meta"].get("chunk_id", 0),
        x["meta"].get("global_chunk_id", 0)
    ))

    docs = [r["text"] for r in results]
    metas = [r["meta"] for r in results]
    raw_scores = [float(r["rerank_score"]) for r in results]

    context = build_context(docs, metas, raw_scores)

    t1 = time.perf_counter()
    prompt = build_prompt(query, context)
    answer = generate_answer(prompt)
    t_llm = time.perf_counter() - t1

    sources = [
        {"title": meta.get("title", "Source"), "url": meta.get("url", "")}
        for meta in metas
    ]

    stage_timings = debug.get("timings", {})
    stage_timings["llm"] = round(t_llm * 1000)
    stage_timings["total"] = round((t_retrieve + t_llm) * 1000)

    return jsonify({
        "answer": answer,
        "sources": sources,
        "chunks": docs,
        "scores": raw_scores,
        "raw_scores": raw_scores,
        "debug": debug,
        "timings": stage_timings
    })


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=DEBUG)