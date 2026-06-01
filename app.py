from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import os
import json
import time
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from qdrant_client.models import Distance, VectorParams, PointStruct


app = FastAPI(title="Paper Search API")

# ── Settings ─────────────────────────────────────────────
COLLECTION_NAME = "clpapers_chunks"
VECTOR_DIM = 384

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


# ── Choose output folder ─────────────────────────────────
def resolve_output_dir() -> Path:
    """
    Uses APP_OUTPUT_DIR if provided.
    Otherwise automatically uses the latest outputs/test* folder.
    """

    env_dir = os.getenv("APP_OUTPUT_DIR")

    if env_dir:
        p = Path(env_dir)
        if not p.is_absolute():
            p = BASE_DIR / p
        return p

    tests = [p for p in OUTPUTS_DIR.glob("test*") if p.is_dir()]

    if not tests:
        raise FileNotFoundError(
            "No outputs/test* folder found. Run your pipeline first."
        )

    # latest modified test folder
    return sorted(tests, key=lambda p: p.stat().st_mtime, reverse=True)[0]


OUTPUT_DIR = resolve_output_dir()
CHUNKS_PATH = OUTPUT_DIR / "all_chunks.json"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {CHUNKS_PATH}. "
        "Check that all_chunks.json exists inside your selected output folder."
    )


# ── Load chunks with UTF-8 ───────────────────────────────
with CHUNKS_PATH.open("r", encoding="utf-8") as f:
    chunks = json.load(f)


# ── Load metadata if available ───────────────────────────
metadata_lookup = {}

if METADATA_PATH.exists():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_lookup = {
        m.get("paper_id"): m
        for m in metadata
        if m.get("paper_id")
    }


# Add metadata into chunks if missing
for c in chunks:
    meta = metadata_lookup.get(c.get("paper_id"), {})
    c.setdefault("title", meta.get("title", "unknown"))
    c.setdefault("authors", meta.get("authors", "unknown"))
    c.setdefault("year", meta.get("year", "unknown"))
    c.setdefault("filename", meta.get("filename", "unknown"))


texts = [c.get("text", "") for c in chunks]


# ── BM25 setup ───────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


tokenized = [tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized)


# ── Dense setup ──────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant = QdrantClient(":memory:")
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
)

# ── Load cached embeddings or generate ───────────────────
# NEW — tied to the same output folder as your chunks
EMBEDDINGS_CACHE = OUTPUT_DIR / "embeddings.npy"

if EMBEDDINGS_CACHE.exists():
    print("✅ Loading cached embeddings...")
    embeddings = np.load(str(EMBEDDINGS_CACHE))
else:
    print("⚠️ No cache found — generating embeddings (~7 min)...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    EMBEDDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(EMBEDDINGS_CACHE), embeddings)
    print("✅ Saved for next time")

# ── Upload to Qdrant ─────────────────────────────────────
for i in range(0, len(chunks), 100):
    batch = chunks[i:i+100]
    embs  = embeddings[i:i+100]
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=i+j,
                vector=embs[j].tolist(),
                payload={
                    "chunk_id": batch[j].get("chunk_id"),
                    "paper_id": batch[j].get("paper_id"),
                    "title":    batch[j].get("title", "unknown"),
                    "text":     batch[j].get("text", "")[:500],
                }
            )
            for j in range(len(batch))
        ]
    )

print(f"✅ Qdrant ready — {len(chunks)} vectors loaded")


# Fast lookup
chunks_by_id = {
    c.get("chunk_id"): c
    for c in chunks
    if c.get("chunk_id")
}

chunk_index_by_id = {
    c.get("chunk_id"): i
    for i, c in enumerate(chunks)
    if c.get("chunk_id")
}


# ── Helper ───────────────────────────────────────────────
def merge_payload_with_chunk(payload: dict) -> dict:
    """
    Qdrant payload may only contain chunk_id/paper_id/text.
    This merges it with the full local chunk data.
    """

    if payload is None:
        payload = {}

    cid = payload.get("chunk_id")
    local_chunk = chunks_by_id.get(cid, {})

    merged = {**local_chunk, **payload}

    meta = metadata_lookup.get(merged.get("paper_id"), {})
    merged.setdefault("title", meta.get("title", "unknown"))
    merged.setdefault("authors", meta.get("authors", "unknown"))
    merged.setdefault("year", meta.get("year", "unknown"))
    merged.setdefault("filename", meta.get("filename", "unknown"))

    return merged


def make_result(rank: int, score: float, chunk: dict) -> dict:
    text = chunk.get("text", "")

    return {
        "rank": rank,
        "score": round(float(score), 4),
        "chunk_id": chunk.get("chunk_id"),
        "paper_id": chunk.get("paper_id"),
        "title": chunk.get("title", "unknown"),
        "authors": chunk.get("authors", "unknown"),
        "year": chunk.get("year", "unknown"),
        "filename": chunk.get("filename", "unknown"),
        "page_start": chunk.get("page_start", "unknown"),
        "page_end": chunk.get("page_end", "unknown"),
        "snippet": text[:250] + "..." if len(text) > 250 else text,
    }


# ── Search functions ────────────────────────────────────
def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    scores = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [
        make_result(rank=i + 1, score=scores[idx], chunk=chunks[idx])
        for i, idx in enumerate(top_idx)
    ]


def dense_search(query: str, top_k: int = 5) -> list[dict]:
    try:
        q_vec = model.encode([query])[0].tolist()

        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=top_k,
        )

        results = response.points

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Dense search failed. Make sure Qdrant is running and "
                f"the collection '{COLLECTION_NAME}' exists. Error: {e}"
            ),
        )

    output = []

    for i, r in enumerate(results):
        chunk = merge_payload_with_chunk(r.payload)
        output.append(make_result(i + 1, r.score, chunk))

    return output


def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.5,
    candidate_k: int = 50,
) -> list[dict]:
    """
    alpha = dense weight.
    0.5 means 50% dense + 50% BM25.
    """

    try:
        q_vec = model.encode([query])[0].tolist()

        dense_response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=candidate_k,
        )

        dense_results = dense_response.points

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Hybrid search failed because dense/Qdrant search failed. "
                "Make sure Qdrant is running and indexed. "
                f"Error: {e}"
            ),
        )

    dense_scores = {}
    dense_payloads = {}

    for r in dense_results:
        payload = r.payload or {}
        cid = payload.get("chunk_id")

        if cid:
            dense_scores[cid] = float(r.score)
            dense_payloads[cid] = merge_payload_with_chunk(payload)

    # BM25 scores
    bm25_raw = np.array(bm25.get_scores(tokenize(query)), dtype=float)

    bm25_max = float(np.max(bm25_raw)) if len(bm25_raw) else 0.0
    if bm25_max <= 0:
        bm25_max = 1.0

    bm25_norm = bm25_raw / bm25_max
    bm25_top_idx = np.argsort(bm25_raw)[::-1][:candidate_k]

    candidates = set(dense_scores.keys())

    for idx in bm25_top_idx:
        cid = chunks[idx].get("chunk_id")
        if cid:
            candidates.add(cid)

    combined = []

    for cid in candidates:
        idx = chunk_index_by_id.get(cid)

        if idx is None:
            chunk = dense_payloads.get(cid, {})
            bm25_score = 0.0
        else:
            chunk = chunks[idx]
            bm25_score = float(bm25_norm[idx])

        dense_score = dense_scores.get(cid, 0.0)

        final_score = alpha * dense_score + (1 - alpha) * bm25_score
        combined.append((final_score, chunk))

    combined.sort(key=lambda x: x[0], reverse=True)

    return [
        make_result(rank=i + 1, score=score, chunk=chunk)
        for i, (score, chunk) in enumerate(combined[:top_k])
    ]


# ── Request model ───────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    method: Literal["bm25", "dense", "hybrid"] = "hybrid"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Endpoints ───────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Paper Search API is running",
        "output_dir": str(OUTPUT_DIR),
        "chunks_loaded": len(chunks),
        "collection": COLLECTION_NAME,
    }


@app.post("/search")
def search(req: SearchRequest):
    t0 = time.time()

    if req.method == "bm25":
        results = bm25_search(req.query, req.top_k)

    elif req.method == "dense":
        results = dense_search(req.query, req.top_k)

    elif req.method == "hybrid":
        results = hybrid_search(
            req.query,
            top_k=req.top_k,
            alpha=req.alpha,
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid search method.")

    latency_ms = round((time.time() - t0) * 1000, 2)

    return {
        "query": req.query,
        "method": req.method,
        "top_k": req.top_k,
        "alpha": req.alpha if req.method == "hybrid" else None,
        "latency_ms": latency_ms,
        "results": results,
    }


@app.get("/search")
def search_get(
    query: str,
    top_k: int = 5,
    method: Literal["bm25", "dense", "hybrid"] = "hybrid",
    alpha: float = 0.5,
):
    return search(
        SearchRequest(
            query=query,
            top_k=top_k,
            method=method,
            alpha=alpha,
        )
    )