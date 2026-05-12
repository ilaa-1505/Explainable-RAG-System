import re
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from functools import lru_cache
from rank_bm25 import BM25Okapi

# ------------------ MODELS ------------------
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------ DB ------------------
client     = PersistentClient(path="embeddings/")
collection = client.get_collection(name="rag_docs")

data      = collection.get(include=["documents", "metadatas"])
docs_all  = data["documents"]
metas_all = data["metadatas"]
ids_all   = data["ids"]

# ------------------ BM25 INDEX ------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

bm25_corpus = [tokenize(doc) for doc in docs_all]
bm25_index  = BM25Okapi(bm25_corpus)

def bm25_search(query: str, top_n: int = 25) -> list[tuple[int, float]]:
    tokens     = tokenize(query)
    raw_scores = bm25_index.get_scores(tokens)
    max_s, min_s = raw_scores.max(), raw_scores.min()
    norm = (raw_scores - min_s) / (max_s - min_s) if max_s != min_s else np.zeros_like(raw_scores)
    top_indices = np.argsort(norm)[::-1][:top_n]
    return [(int(i), float(norm[i])) for i in top_indices]

# ------------------ EMBEDDING ------------------
@lru_cache(maxsize=128)
def embed_query(query: str):
    return embed_model.encode("query: " + query, normalize_embeddings=True)

# ------------------ HYBRID FUSION ------------------
def hybrid_fusion(vector_indices, vector_scores, bm25_results, alpha=0.5, top_n=25):
    vec_map  = dict(zip(vector_indices, vector_scores))
    bm25_map = dict(bm25_results)
    fused = [
        (idx, vec_map.get(idx, 0.0), bm25_map.get(idx, 0.0),
         alpha * vec_map.get(idx, 0.0) + (1 - alpha) * bm25_map.get(idx, 0.0))
        for idx in set(vec_map) | set(bm25_map)
    ]
    fused.sort(key=lambda x: x[3], reverse=True)
    return fused[:top_n]

# ------------------ MMR ------------------
def mmr(query_emb, doc_indices, k=10, lambda_param=0.7):
    embs = [
        np.array(collection.get(ids=[ids_all[i]], include=["embeddings"])["embeddings"][0])
        for i in doc_indices
    ]
    embs      = [e / np.linalg.norm(e) for e in embs]
    query_emb = query_emb / np.linalg.norm(query_emb)
    sims      = [np.dot(query_emb, e) for e in embs]

    best_idx = int(np.argmax(sims))
    selected = [doc_indices[best_idx]]
    sel_idx  = [best_idx]
    mmr_debug = []

    while len(selected) < min(k, len(doc_indices)):
        scores = [
            (lambda_param * sims[i] - (1 - lambda_param) * max(np.dot(embs[i], embs[j]) for j in sel_idx),
             i, sims[i], max(np.dot(embs[i], embs[j]) for j in sel_idx))
            for i in range(len(doc_indices)) if i not in sel_idx
        ]
        if not scores:
            break
        _, idx, rel, div = max(scores)
        selected.append(doc_indices[idx])
        sel_idx.append(idx)
        mmr_debug.append({
            "doc_index":         doc_indices[idx],
            "relevance":         float(rel),
            "diversity_penalty": float(div),
        })

    return selected, mmr_debug

# ------------------ MMR (rerun with cached embs) ------------------
def mmr_from_embs(query_emb, doc_indices, embs, k=10, lambda_param=0.7):
    """Same as mmr() but uses pre-fetched embeddings — for fast lambda slider reruns."""
    embs_n    = [e / np.linalg.norm(e) for e in embs]
    query_emb = query_emb / np.linalg.norm(query_emb)
    sims      = [np.dot(query_emb, e) for e in embs_n]

    best_idx = int(np.argmax(sims))
    selected = [doc_indices[best_idx]]
    sel_idx  = [best_idx]

    while len(selected) < min(k, len(doc_indices)):
        scores = [
            (lambda_param * sims[i] - (1 - lambda_param) * max(np.dot(embs_n[i], embs_n[j]) for j in sel_idx),
             i, sims[i], max(np.dot(embs_n[i], embs_n[j]) for j in sel_idx))
            for i in range(len(doc_indices)) if i not in sel_idx
        ]
        if not scores:
            break
        _, idx, rel, div = max(scores)
        selected.append(doc_indices[idx])
        sel_idx.append(idx)

    return selected

# ------------------ RERANK ------------------
def rerank(query, doc_indices, top_k=7):
    docs   = [docs_all[i] for i in doc_indices]
    pairs  = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    s_arr  = np.array(scores, dtype=float)
    if s_arr.max() != s_arr.min():
        s_arr = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min())
    else:
        s_arr = np.ones_like(s_arr)
    scored = sorted(zip(doc_indices, s_arr.tolist()), key=lambda x: x[1], reverse=True)
    return scored[:top_k], scored

_session = {
    "query_emb":    None,   # np.ndarray
    "doc_indices":  None,   # list[int]
    "embs":         None,   # list[np.ndarray]  — raw (not normalized)
    "sims":         None,   # list[float] query-doc cosine sims
    "umap_coords":  None,   # list[[x,y]] — computed once per query
    "sim_matrix":   None,   # NxN similarity matrix between candidates
}

def _compute_umap(embs_norm):
    try:
        import umap
        n = len(embs_norm)
        n_neighbors = min(5, n - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=0.1, random_state=42, verbose=False)
        coords = reducer.fit_transform(np.array(embs_norm))
        for dim in range(2):
            mn, mx = coords[:, dim].min(), coords[:, dim].max()
            if mx != mn:
                coords[:, dim] = (coords[:, dim] - mn) / (mx - mn)
        return coords.tolist()
    except Exception as e:
        print(f"[UMAP] Failed: {e}")
        return None

def _compute_sim_matrix(embs_norm):
    mat = np.array(embs_norm)
    sim = mat @ mat.T
    return np.clip(sim, -1, 1).tolist()

RERANK_THRESHOLD = 0.3
HYBRID_ALPHA     = 0.7

def retrieve(query, top_k=7):
    print(f"\nQuery: {query}")
    timings = {}

    # --- EMBED ---
    t = time.perf_counter()
    query_emb = embed_query(query)
    timings["embed"] = round((time.perf_counter() - t) * 1000)

    # --- VECTOR SEARCH ---
    t = time.perf_counter()
    results        = collection.query(query_embeddings=[query_emb.tolist()], n_results=25)
    vector_ids     = results["ids"][0]
    vector_dists   = results["distances"][0]
    vector_scores  = [1 - d for d in vector_dists]
    vector_indices = [ids_all.index(i) for i in vector_ids]
    timings["vector"] = round((time.perf_counter() - t) * 1000)
    print(f"[Vector Search] Retrieved: {len(vector_indices)} chunks")

    # --- BM25 ---
    t = time.perf_counter()
    bm25_results = bm25_search(query, top_n=25)
    timings["bm25"] = round((time.perf_counter() - t) * 1000)
    print(f"[BM25] Retrieved: {len(bm25_results)} chunks")

    # --- HYBRID FUSION ---
    t = time.perf_counter()
    fused          = hybrid_fusion(vector_indices, vector_scores, bm25_results, alpha=HYBRID_ALPHA)
    hybrid_indices = [idx for idx, _, _, _ in fused]
    score_lookup   = {idx: (vs, bs, hs) for idx, vs, bs, hs in fused}
    timings["hybrid"] = round((time.perf_counter() - t) * 1000)
    print(f"[Hybrid] Fused: {len(hybrid_indices)} chunks")

    # --- FETCH EMBEDDINGS for MMR + cache  ---
    t = time.perf_counter()
    raw_embs = [
        np.array(collection.get(ids=[ids_all[i]], include=["embeddings"])["embeddings"][0])
        for i in hybrid_indices
    ]
    embs_norm = [e / np.linalg.norm(e) for e in raw_embs]
    query_emb_n = query_emb / np.linalg.norm(query_emb)
    sims = [float(np.dot(query_emb_n, e)) for e in embs_norm]

    # --- MMR ---
    mmr_selected = mmr_from_embs(query_emb, hybrid_indices, raw_embs, k=10)
    mmr_debug = []
    timings["mmr"] = round((time.perf_counter() - t) * 1000)
    print(f"[MMR] Selected: {len(mmr_selected)} chunks")

    # --- CACHE session  ---
    umap_coords = _compute_umap(embs_norm)
    sim_matrix  = _compute_sim_matrix(embs_norm)

    _session["query_emb"]   = query_emb
    _session["doc_indices"] = hybrid_indices
    _session["embs"]        = raw_embs
    _session["sims"]        = sims
    _session["umap_coords"] = umap_coords
    _session["sim_matrix"]  = sim_matrix

    # --- RERANK ---
    t = time.perf_counter()
    top_final, full_rerank = rerank(query, mmr_selected, top_k)
    top_final = [(i, score) for i, score in top_final if score >= RERANK_THRESHOLD]
    timings["rerank"] = round((time.perf_counter() - t) * 1000)
    print(f"[Reranker] Selected: {len(top_final)} chunks (threshold: {RERANK_THRESHOLD})")

    # --- OUTPUT ---
    final = [
        {
            "text":         docs_all[i].replace("passage: ", ""),
            "meta":         metas_all[i],
            "rerank_score": float(score),
            "vector_score": score_lookup.get(i, (0, 0, 0))[0],
            "bm25_score":   score_lookup.get(i, (0, 0, 0))[1],
            "hybrid_score": score_lookup.get(i, (0, 0, 0))[2],
        }
        for i, score in top_final
    ]

    # --- MMR without diversity (pure relevance) ---
    no_mmr_selected = [
        hybrid_indices[i] for i in np.argsort(sims)[::-1][:10]
    ]

    debug_info = {
        "vector_count":    len(vector_indices),
        "bm25_count":      len(bm25_results),
        "hybrid_count":    len(hybrid_indices),
        "mmr_count":       len(mmr_selected),
        "rerank_count":    len(top_final),
        "mmr_details":     mmr_debug,
        "mmr_selected":    mmr_selected,
        "no_mmr_selected": no_mmr_selected,
        "rerank_full":     full_rerank,
        "score_lookup":    {str(k): v for k, v in score_lookup.items()},
        "timings":         timings,
        "umap_coords":     umap_coords,
        "sim_matrix":      sim_matrix,
        "doc_indices":     hybrid_indices,
        "sims":            sims,
        "doc_previews":    [docs_all[i][:80].replace("passage: ", "") for i in hybrid_indices],
    }

    return final, debug_info