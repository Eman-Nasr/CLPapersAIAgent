import json, time, csv, math
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import TOP_K, LATENCY_PERCENTILE



def _dcg(relevances: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def _ndcg_at_k(retrieved_ids: list[str],
               relevant_ids:  set[str],
               k:             int) -> float:
    if not relevant_ids:
        return 0.0
    rels    = [1 if pid in relevant_ids else 0 for pid in retrieved_ids[:k]]
    ideal   = [1] * min(len(relevant_ids), k)
    idcg    = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


def _recall_at_k(retrieved_ids: list[str],
                 relevant_ids:  set[str],
                 k:             int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for pid in retrieved_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids)


def _mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, pid in enumerate(retrieved_ids, 1):
        if pid in relevant_ids:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(results_all: list[dict], k: int = TOP_K) -> dict:
    recall_scores, ndcg_scores, mrr_scores = [], [], []
    latencies: list[float] = []

    for r in results_all:
        relevant = set(r.get("relevant_paper_ids", []))
        retrieved = [h["paper_id"] for h in r.get("top_k", [])]
        lat = r.get("latency_ms", 0.0)
        latencies.append(lat)

        if not relevant:
            continue  

        recall_scores.append(_recall_at_k(retrieved, relevant, k))
        ndcg_scores.append(_ndcg_at_k(retrieved, relevant, k))
        mrr_scores.append(_mrr(retrieved, relevant))

    def _mean(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    p95_lat = round(float(np.percentile(latencies, LATENCY_PERCENTILE)), 3) \
        if latencies else 0.0

    return {
        f"Recall@{k}":         _mean(recall_scores),
        f"NDCG@{k}":           _mean(ndcg_scores),
        "MRR":                 _mean(mrr_scores),
        "avg_latency_ms":      _mean(latencies),
        f"p{LATENCY_PERCENTILE}_latency_ms": p95_lat,
        "num_queries_evaluated": len(recall_scores),
        "num_queries_total":     len(results_all),
    }



def build_and_retrieve(
    all_chunks:    list[dict],
    queries:       list[dict],
    output_dir:    Path,
    metadata_list: list[dict] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a fast lookup: paper_id → metadata (for citation fields)
    meta_lookup: dict[str, dict] = {}
    if metadata_list:
        for m in metadata_list:
            meta_lookup[m["paper_id"]] = m
    else:
        # Try to load from disk
        meta_path = output_dir / "metadata.json"
        if meta_path.exists():
            for m in json.loads(meta_path.read_text()):
                meta_lookup[m["paper_id"]] = m

    # Build TF-IDF index 
    print(f"[tfidf] Building index over {len(all_chunks)} chunks …")
    t0    = time.time()
    texts = [c["text"] for c in all_chunks]

    vectorizer = TfidfVectorizer(
        max_features  = 50_000,
        ngram_range   = (1, 2),
        sublinear_tf  = True,
        min_df        = 2,
        strip_accents = "unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    build_time   = round(time.time() - t0, 3)
    print(f"[tfidf] Index built in {build_time}s | matrix: {tfidf_matrix.shape}")

    # Run queries 
    results_all: list[dict] = []
    for q in queries:
        t_q   = time.time()
        q_vec = vectorizer.transform([q["text"]])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:TOP_K]
        latency_ms = round((time.time() - t_q) * 1000, 3)

        hits = []
        for rank, idx in enumerate(top_idx, 1):
            chunk = all_chunks[idx]
            pid   = chunk["paper_id"]
            m     = meta_lookup.get(pid, {})

            hits.append({
                "rank":       rank,
                "chunk_id":   chunk["chunk_id"],
                "paper_id":   pid,
                "title":      m.get("title",    "unknown"),
                "year":       m.get("year",     "unknown"),
                "authors":    m.get("authors",  "unknown"),
                "filename":   m.get("filename", "unknown"),
                "filepath":   m.get("filepath", "unknown"),
                "page_start": chunk.get("page_start", "unknown"),
                "page_end":   chunk.get("page_end",   "unknown"),
                "score":      round(float(scores[idx]), 6),
                "word_count": chunk["word_count"],
                "snippet":    chunk["text"][:300] + (
                    "…" if len(chunk["text"]) > 300 else ""
                ),
            })

        results_all.append({
            "query_id":           q["query_id"],
            "query":              q["text"],
            "category":           q.get("category", ""),
            "relevant_paper_ids": q.get("relevant_paper_ids", []),
            "latency_ms":         latency_ms,
            "top_k":              hits,
        })

    # Evaluate 
    eval_metrics = evaluate_retrieval(results_all, k=TOP_K)
    print(f"[tfidf] Recall@{TOP_K}={eval_metrics[f'Recall@{TOP_K}']}  "
          f"NDCG@{TOP_K}={eval_metrics[f'NDCG@{TOP_K}']}  "
          f"MRR={eval_metrics['MRR']}  "
          f"avg_lat={eval_metrics['avg_latency_ms']}ms")

    retrieval_path = output_dir / "retrieval_results.json"
    retrieval_path.write_text(
        json.dumps(results_all, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    eval_out = output_dir / "evaluation_results.json"
    eval_out.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    eval_csv = output_dir / "evaluation_results.csv"
    with open(eval_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in eval_metrics.items():
            writer.writerow([k, v])

    index_meta = {
        "num_chunks":    len(all_chunks),
        "vocab_size":    len(vectorizer.vocabulary_),
        "ngram_range":   "(1, 2)",
        "max_features":  50_000,
        "build_time_s":  build_time,
        "top_k":         TOP_K,
        "num_queries":   len(queries),
        "note":          (
            "TF-IDF is a lexical baseline. "
            "Semantic retrieval (dense vectors, GraphRAG) is planned for "
            "future deliverables. AutoML reranking and River online learning "
            "will use these results as a cold-start signal."
        ),
    }
    (output_dir / "tfidf_index_meta.json").write_text(
        json.dumps(index_meta, indent=2), encoding="utf-8"
    )

    print(f"[tfidf] Done → {retrieval_path} | eval → {eval_csv}")
    return {
        "results":    results_all,
        "index_meta": index_meta,
        "eval":       eval_metrics,
    }


if __name__ == "__main__":
    import json
    from src.config import OUTPUTS_DIR
    tests  = sorted(OUTPUTS_DIR.glob("test*"))
    out    = tests[-1]
    chunks  = json.loads((out / "all_chunks.json").read_text())
    queries = json.loads((out / "query_set.json").read_text())
    build_and_retrieve(chunks, queries, out)