import sys
import os
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.decomposition import PCA

# Project paths
CURRENT_DIR = Path.cwd()
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "notebooks" else CURRENT_DIR
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT_ROOT))

# Plot style
matplotlib.rcParams["figure.facecolor"] ="#0f1117"
matplotlib.rcParams["axes.facecolor"] = "#161b22"
matplotlib.rcParams["text.color"] = "#e6edf3"
matplotlib.rcParams["axes.labelcolor"] = "#8b949e"
matplotlib.rcParams["xtick.color"] = "#8b949e"
matplotlib.rcParams["ytick.color"] = "#8b949e"

print("All imports OK")
print(f"Outputs will be saved in: {OUTPUT_DIR}")
# Connect to MongoDB
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["clpapers"]

# Load all chunks created by Member 1
chunks = list(db.chunks.find({}, {"_id": 0}))

if len(chunks) == 0:
    raise ValueError(
        "No chunks found in MongoDB. Run 01_ingestion_pipeline.ipynb first, "
        "and make sure Docker/MongoDB is running."
    )

print(f"Loaded {len(chunks)} chunks from MongoDB")
print(f"Unique papers: {len(set(c.get('paper_id', 'unknown') for c in chunks))}")
print("\nSample chunk keys:", list(chunks[0].keys()))
print("\nSample text:")
print(chunks[0]["text"][:500])
print("Loading sentence-transformer model...")

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

print(f"Model loaded: {MODEL_NAME}")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")

texts = [c["text"] for c in chunks]
chunk_ids = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]

print(f"Generating embeddings for {len(texts)} chunks...")

t0 = time.time()

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

embedding_time = round(time.time() - t0, 2)

print(f"\nEmbeddings generated in {embedding_time} seconds")
print(f"Embedding matrix shape: {embeddings.shape}")
if len(embeddings) >= 2:
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    paper_ids = [c.get("paper_id", "unknown") for c in chunks]
    unique_papers = sorted(list(set(paper_ids)))

    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_papers)))
    paper_color_map = {pid: colors[i] for i, pid in enumerate(unique_papers)}

    fig, ax = plt.subplots(figsize=(10, 6))

    for pid in unique_papers[:10]:
        mask = [i for i, p in enumerate(paper_ids) if p == pid]

        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=pid,
            alpha=0.6,
            s=15,
            color=paper_color_map[pid]
        )

    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Chunk Embeddings in 2D using PCA", color="#e6edf3")
    ax.legend(
        fontsize=7,
        loc="upper right",
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#c9d1d9"
    )
    ax.grid(True, color="#21262d", linewidth=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "embedding_pca.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Saved plot: embedding_pca.png")
else:
    print("Not enough chunks for PCA plot.")
    # Connect to Qdrant
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "clpapers_chunks"
VECTOR_DIM = model.get_sentence_embedding_dimension()

# Delete old collection if it exists
try:
    qdrant.delete_collection(COLLECTION_NAME)
    print(f"Deleted old collection: {COLLECTION_NAME}")
except Exception:
    print("No old collection found. Creating new collection.")

# Create Qdrant collection
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_DIM,
        distance=Distance.COSINE
    )
)

print(f"Created Qdrant collection: {COLLECTION_NAME}")
print(f"Vector dimension: {VECTOR_DIM}")
BATCH_SIZE = 100
total_uploaded = 0

for start in range(0, len(chunks), BATCH_SIZE):
    batch_chunks = chunks[start:start + BATCH_SIZE]
    batch_embeddings = embeddings[start:start + BATCH_SIZE]

    points = []

    for j, chunk in enumerate(batch_chunks):
        point_id = start + j

        payload = {
            "chunk_id": chunk.get("chunk_id", f"chunk_{point_id}"),
            "paper_id": chunk.get("paper_id", "unknown"),
            "title": chunk.get("title", "unknown"),
            "authors": chunk.get("authors", "unknown"),
            "year": chunk.get("year", "unknown"),
            "page_start": chunk.get("page_start", 0),
            "page_end": chunk.get("page_end", 0),
            "text": chunk.get("text", "")[:1000]
        }

        points.append(
            PointStruct(
                id=point_id,
                vector=batch_embeddings[j].tolist(),
                payload=payload
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    total_uploaded += len(points)

print(f"Uploaded {total_uploaded} vectors to Qdrant")

info = qdrant.get_collection(COLLECTION_NAME)
print(f"Collection size: {info.points_count} points")
tokenized_texts = [text.lower().split() for text in texts]

bm25 = BM25Okapi(tokenized_texts)

print(f"BM25 index built over {len(tokenized_texts)} chunks")
def qdrant_vector_search(query_vector, limit):
    """
    Compatible Qdrant vector search.
    Works with older qdrant.search() and newer qdrant.query_points().
    """
    if hasattr(qdrant, "search"):
        return qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )

    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    )

    return response.points


def dense_search(query: str, top_k: int = 5) -> list:
    """
    Dense semantic search using SentenceTransformer embeddings + Qdrant.
    """
    query_vector = model.encode([query])[0].tolist()

    results = qdrant_vector_search(query_vector, top_k)

    output = []

    for i, r in enumerate(results):
        payload = r.payload

        output.append({
            "rank": i + 1,
            "score": round(float(r.score), 4),
            "paper_id": payload.get("paper_id", "unknown"),
            "title": payload.get("title", "unknown"),
            "authors": payload.get("authors", "unknown"),
            "year": payload.get("year", "unknown"),
            "page_start": payload.get("page_start", 0),
            "snippet": payload.get("text", "")[:200] + "..."
        })

    return output


def bm25_search(query: str, top_k: int = 5) -> list:
    """
    Lexical search using BM25.
    """
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    output = []

    for i, idx in enumerate(top_indices):
        chunk = chunks[idx]

        output.append({
            "rank": i + 1,
            "score": round(float(scores[idx]), 4),
            "paper_id": chunk.get("paper_id", "unknown"),
            "title": chunk.get("title", "unknown"),
            "authors": chunk.get("authors", "unknown"),
            "year": chunk.get("year", "unknown"),
            "page_start": chunk.get("page_start", 0),
            "snippet": chunk.get("text", "")[:200] + "..."
        })

    return output


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5) -> list:
    """
    Hybrid search combines dense semantic search and BM25 lexical search.

    alpha = 0.5 means:
    50% dense semantic score
    50% BM25 lexical score
    """
    query_vector = model.encode([query])[0].tolist()

    dense_results = qdrant_vector_search(query_vector, len(chunks))

    dense_scores = {
        r.payload.get("chunk_id"): float(r.score)
        for r in dense_results
    }

    bm25_raw_scores = bm25.get_scores(query.lower().split())

    bm25_max = max(bm25_raw_scores) if max(bm25_raw_scores) > 0 else 1
    bm25_normalized = bm25_raw_scores / bm25_max

    combined_results = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")

        dense_score = dense_scores.get(chunk_id, 0.0)
        bm25_score = float(bm25_normalized[i])

        hybrid_score = alpha * dense_score + (1 - alpha) * bm25_score

        combined_results.append((hybrid_score, i))

    combined_results.sort(reverse=True)

    output = []

    for rank, (score, idx) in enumerate(combined_results[:top_k]):
        chunk = chunks[idx]

        output.append({
            "rank": rank + 1,
            "score": round(float(score), 4),
            "paper_id": chunk.get("paper_id", "unknown"),
            "title": chunk.get("title", "unknown"),
            "authors": chunk.get("authors", "unknown"),
            "year": chunk.get("year", "unknown"),
            "page_start": chunk.get("page_start", 0),
            "snippet": chunk.get("text", "")[:200] + "..."
        })

    return output


print("BM25, dense search, and hybrid search functions are ready.")
TEST_QUERIES = [
    "transformer architecture for sequence tasks",
    "large language model alignment",
    "retrieval augmented generation"
]

for query in TEST_QUERIES:
    print("\n" + "=" * 70)
    print(f'QUERY: "{query}"')
    print("=" * 70)

    print("\n--- HYBRID SEARCH: BM25 + DENSE ---")
    for r in hybrid_search(query, top_k=3):
        print(f"[{r['rank']}] {r['paper_id']} | score={r['score']} | {r['title'][:70]}")

    print("\n--- DENSE SEMANTIC SEARCH ONLY ---")
    for r in dense_search(query, top_k=3):
        print(f"[{r['rank']}] {r['paper_id']} | score={r['score']} | {r['title'][:70]}")

    print("\n--- BM25 SEARCH ONLY ---")
    for r in bm25_search(query, top_k=3):
        print(f"[{r['rank']}] {r['paper_id']} | score={r['score']} | {r['title'][:70]}")
    benchmark_query = "attention mechanism in transformers"
RUNS = 5

def benchmark_search_function(fn, label):
    times = []

    for _ in range(RUNS):
        start = time.time()
        fn(benchmark_query)
        end = time.time()

        times.append((end - start) * 1000)

    avg_time = round(sum(times) / len(times), 2)

    print(f"{label:<20} average latency: {avg_time:.2f} ms")

    return avg_time


print(f"Latency benchmark using query: {benchmark_query}")
print(f"Runs per method: {RUNS}\n")

bm25_latency = benchmark_search_function(bm25_search, "BM25")
dense_latency = benchmark_search_function(dense_search, "Dense Qdrant")
hybrid_latency = benchmark_search_function(hybrid_search, "Hybrid")
methods = ["BM25", "Dense\nQdrant", "Hybrid"]
latencies = [bm25_latency, dense_latency, hybrid_latency]
bar_colors = ["#f0a500", "#58a6ff", "#3fb950"]

fig, ax = plt.subplots(figsize=(7, 4))

bars = ax.bar(
    methods,
    latencies,
    color=bar_colors,
    alpha=0.85,
    width=0.5
)

for bar, value in zip(bars, latencies):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{value:.1f} ms",
        ha="center",
        va="bottom",
        color="#c9d1d9",
        fontsize=9
    )

ax.set_ylabel("Average Latency (ms)")
ax.set_title("Search Latency Comparison", color="#e6edf3")
ax.grid(True, axis="y", color="#21262d", linewidth=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "latency_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved plot: latency_comparison.png")
query = TEST_QUERIES[0]

hybrid_results = hybrid_search(query, top_k=5)
dense_results = dense_search(query, top_k=5)
bm25_results = bm25_search(query, top_k=5)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, results, title, color in zip(
    axes,
    [hybrid_results, dense_results, bm25_results],
    ["Hybrid", "Dense", "BM25"],
    ["#3fb950", "#58a6ff", "#f0a500"]
):
    labels = [r["paper_id"] for r in results]
    scores = [r["score"] for r in results]

    ax.barh(labels[::-1], scores[::-1], color=color, alpha=0.85)
    ax.set_title(title, color="#e6edf3")
    ax.set_xlabel("Score")
    ax.grid(True, axis="x", color="#21262d", linewidth=0.7)

fig.suptitle(f'Top-5 Results for Query: "{query}"', color="#e6edf3", fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "score_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved plot: score_comparison.png")
collection_info = qdrant.get_collection(COLLECTION_NAME)

summary = {
    "notebook": "02_semantic_retrieval.ipynb",
    "member": "Member 2",
    "task": "Semantic retrieval and hybrid search",
    "chunks_loaded": len(chunks),
    "embedding_model": MODEL_NAME,
    "embedding_dimension": VECTOR_DIM,
    "qdrant_collection": COLLECTION_NAME,
    "vectors_in_qdrant": collection_info.points_count,
    "search_methods": [
        "BM25 lexical search",
        "Dense semantic search using Qdrant",
        "Hybrid search using BM25 + dense scores"
    ],
    "hybrid_alpha": 0.5,
    "embedding_time_seconds": embedding_time,
    "latency_ms": {
        "bm25": bm25_latency,
        "dense_qdrant": dense_latency,
        "hybrid": hybrid_latency
    },
    "output_files": [
        "embedding_pca.png",
        "latency_comparison.png",
        "score_comparison.png"
    ]
}

summary_path = OUTPUT_DIR / "semantic_retrieval_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print(f"Saved summary JSON: {summary_path}")
print("=" * 60)
print("SEMANTIC RETRIEVAL COMPLETE")
print("=" * 60)

print(f"Chunks embedded: {len(chunks)}")
print(f"Embedding model: {MODEL_NAME}")
print(f"Embedding dimension: {VECTOR_DIM}")
print(f"Qdrant collection: {COLLECTION_NAME}")
print(f"Vectors in Qdrant: {collection_info.points_count}")
print("Search methods: BM25, Dense, Hybrid")
print("Hybrid alpha: 0.5")

print("\nLatency results:")
print(f"BM25 latency: {bm25_latency:.2f} ms")
print(f"Dense latency: {dense_latency:.2f} ms")
print(f"Hybrid latency: {hybrid_latency:.2f} ms")

print("\nFiles saved in outputs folder:")
print("- embedding_pca.png")
print("- latency_comparison.png")
print("- score_comparison.png")
print("- semantic_retrieval_summary.json")

print("\nYour Member 2 requirements are covered:")
print("1. Generate embeddings from chunks")
print("2. Store embeddings in Qdrant")
print("3. Build dense semantic search")
print("4. Build BM25 search")
print("5. Build hybrid search: BM25 + dense")
print("6. Show example search results")
print("7. Save plots and summary for the report")
