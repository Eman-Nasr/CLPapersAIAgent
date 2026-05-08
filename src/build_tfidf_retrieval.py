import json, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import TOP_K

def build_and_retrieve(all_chunks: list[dict],
                       queries:    list[dict],
                       output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build TF-IDF index 
    print(f"[tfidf] Building index over {len(all_chunks)} chunks …")
    t0       = time.time()
    texts    = [c["text"] for c in all_chunks]

    vectorizer = TfidfVectorizer(
        max_features  = 50_000,
        ngram_range   = (1, 2),
        sublinear_tf  = True,
        min_df        = 2,
        strip_accents = "unicode"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)   # (n_chunks, vocab)
    build_time   = round(time.time() - t0, 3)
    print(f"[tfidf] Index built in {build_time}s  "
          f"| matrix shape: {tfidf_matrix.shape}")

    # Run queries 
    results_all = []
    for q in queries:
        q_vec  = vectorizer.transform([q["text"]])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:TOP_K]

        hits = []
        for rank, idx in enumerate(top_idx, 1):
            chunk = all_chunks[idx]
            hits.append({
                "rank":       rank,
                "chunk_id":   chunk["chunk_id"],
                "paper_id":   chunk["paper_id"],
                "score":      round(float(scores[idx]), 6),
                "word_count": chunk["word_count"],
                "snippet":    chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else "")
            })

        results_all.append({
            "query_id": q["query_id"],
            "query":    q["text"],
            "category": q.get("category", ""),
            "top_k":    hits
        })

    # Save outputs 
    retrieval_path = output_dir / "retrieval_results.json"
    retrieval_path.write_text(
        json.dumps(results_all, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    # Index metadata
    index_meta = {
        "num_chunks":       len(all_chunks),
        "vocab_size":       len(vectorizer.vocabulary_),
        "ngram_range":      "(1, 2)",
        "max_features":     50000,
        "build_time_s":     build_time,
        "top_k":            TOP_K,
        "num_queries":      len(queries)
    }
    (output_dir / "tfidf_index_meta.json").write_text(
        json.dumps(index_meta, indent=2),
        encoding="utf-8"
    )

    print(f"[tfidf] Retrieval complete for {len(queries)} queries → {retrieval_path}")
    return {"results": results_all, "index_meta": index_meta}


if __name__ == "__main__":
    import json
    from src.config import OUTPUTS_DIR
    tests = sorted(OUTPUTS_DIR.glob("test*"))
    out   = tests[-1]
    chunks  = json.loads((out / "all_chunks.json").read_text())
    queries = json.loads((out / "query_set.json").read_text())
    build_and_retrieve(chunks, queries, out)