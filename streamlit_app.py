import streamlit as st
import requests
from collections import defaultdict
from pathlib import Path
import base64

def _img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ─────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Paper Search", page_icon="🔍", layout="wide")

st.title("🔍 Research Paper Search")
st.caption("BM25 | Dense | Hybrid Retrieval")

BASE = "http://localhost:8000"  # FastAPI must be running


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def set_query(title):
    st.session_state.query_input = title


def call_search_api(query, top_k, method):
    try:
        resp = requests.post(
            f"{BASE}/search",
            json={
                "query": query,
                "top_k": top_k,
                "method": method
            },
            timeout=20
        )

        if resp.status_code == 200:
            return resp.json(), None

        return None, f"{method.upper()} returned status code {resp.status_code}"

    except requests.exceptions.RequestException as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
if "query_input" not in st.session_state:
    st.session_state.query_input = ""


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results", 1, 10, 5)

    st.markdown("---")
    st.caption("Backend URL")
    st.code(BASE)


# ─────────────────────────────────────────────────────────────
# Search input
# ─────────────────────────────────────────────────────────────
query_input = st.text_input(
    "Enter your query",
    placeholder="e.g. retrieval augmented generation",
    key="query_input"
)

query = query_input.strip()


# ─────────────────────────────────────────────────────────────
# Example queries — ONLY show before searching
# ─────────────────────────────────────────────────────────────
if not query:
    st.divider()
    st.subheader("💡 Example Queries")

    examples = [
        "retrieval augmented generation",
        "transformer architecture for sequence tasks",
        "large language model alignment"
    ]

    for ex in examples:
        st.button(
            ex,
            key=f"example_{ex}",
            on_click=set_query,
            args=(ex,)
        )


# ─────────────────────────────────────────────────────────────
# Main search
# ─────────────────────────────────────────────────────────────
if query and len(query) > 2:
    results = {}
    errors = {}

    with st.spinner("Searching all 3 methods..."):
        for method in ["bm25", "dense", "hybrid"]:
            data, error = call_search_api(query, top_k, method)

            if data:
                results[method] = data

            if error:
                errors[method] = error

    if errors:
        with st.expander("⚠️ View backend errors"):
            for method, error in errors.items():
                st.write(f"**{method.upper()}**: {error}")

    valid_methods = {
        m: d for m, d in results.items()
        if d.get("results")
    }

    if not valid_methods:
        st.warning("No results found for this query. Try different keywords.")

    else:
        # ─────────────────────────────────────────────────────
        # Rank-based voting + global normalized score
        # ─────────────────────────────────────────────────────
        paper_votes = defaultdict(lambda: {
            "points": 0,
            "norm_score_sum": 0.0,
            "methods": [],
            "result": None
        })

        all_scores = [
            r.get("score", 0)
            for d in valid_methods.values()
            for r in d.get("results", [])
        ]

        global_min = min(all_scores)
        global_max = max(all_scores)
        global_diff = (global_max - global_min) or 1

        def normalize(score):
            return (score - global_min) / global_diff

        for method, data in valid_methods.items():
            for r in data.get("results", []):
                paper_key = r.get("paper_id") or r.get("title", "").strip().lower()
                rank = r.get("rank", top_k)
                points = max(0, top_k - rank + 1)
                norm_score = normalize(r.get("score", 0))

                paper_votes[paper_key]["points"] += points
                paper_votes[paper_key]["norm_score_sum"] += norm_score
                paper_votes[paper_key]["methods"].append(method.upper())
                paper_votes[paper_key]["result"] = r

        best_key = max(
            paper_votes,
            key=lambda k: (
                paper_votes[k]["points"] *
                (
                    paper_votes[k]["norm_score_sum"] /
                    len(paper_votes[k]["methods"])
                )
            )
        )

        best_info = paper_votes[best_key]
        best_result = best_info["result"]
        best_methods = list(dict.fromkeys(best_info["methods"]))

        # ─────────────────────────────────────────────────────
        # Best match banner
        # ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"""
### 🏆 Most Agreed Result for this Query

**Paper:** {best_result.get('title', 'Unknown title')}  
**Found by:** {", ".join(best_methods)}  
**Agreement Score:** {best_info["points"]}  
**Authors:** {best_result.get('authors', 'Unknown')}  

> {best_result.get('snippet', 'No snippet available.')}
""")
        st.markdown("---")

        # ─────────────────────────────────────────────────────
        # Method agreement
        # ─────────────────────────────────────────────────────
        st.markdown("### 🤝 Method Agreement")

        if len(valid_methods) == 3:
            bm25_papers = {
                r.get("paper_id") or r.get("title", "").strip().lower(): r
                for r in results["bm25"]["results"]
            }

            dense_papers = {
                r.get("paper_id") or r.get("title", "").strip().lower(): r
                for r in results["dense"]["results"]
            }

            hybrid_papers = {
                r.get("paper_id") or r.get("title", "").strip().lower(): r
                for r in results["hybrid"]["results"]
            }

            bm25_ids = set(bm25_papers.keys())
            dense_ids = set(dense_papers.keys())
            hybrid_ids = set(hybrid_papers.keys())

            all_three = bm25_ids & dense_ids & hybrid_ids

            two_of_three = (
                (bm25_ids & dense_ids) |
                (bm25_ids & hybrid_ids) |
                (dense_ids & hybrid_ids)
            ) - all_three

            if all_three:
                st.success(f"✅ {len(all_three)} paper(s) found by ALL 3 methods — highly relevant")

                for pid in all_three:
                    r = bm25_papers.get(pid) or dense_papers.get(pid) or hybrid_papers.get(pid)
                    st.markdown(f"- 📄 **{r.get('title', 'Unknown title')}**")

            if two_of_three:
                st.info(f"📌 {len(two_of_three)} paper(s) found by 2 methods")

                for pid in two_of_three:
                    r = bm25_papers.get(pid) or dense_papers.get(pid) or hybrid_papers.get(pid)

                    found_by = []

                    if pid in bm25_ids:
                        found_by.append("BM25")

                    if pid in dense_ids:
                        found_by.append("Dense")

                    if pid in hybrid_ids:
                        found_by.append("Hybrid")

                    st.markdown(
                        f"- 📄 **{r.get('title', 'Unknown title')}** — found by {', '.join(found_by)}"
                    )

            if not all_three and not two_of_three:
                st.warning("No overlap between methods. Each method returned different papers.")

        else:
            st.info("Agreement summary needs results from all 3 methods.")

        # ─────────────────────────────────────────────────────
        # Determine best column for highlighting
        # ─────────────────────────────────────────────────────
        method_points = defaultdict(int)

        for method, data in valid_methods.items():
            for r in data.get("results", []):
                paper_key = r.get("paper_id") or r.get("title", "").strip().lower()

                if paper_key == best_key:
                    method_points[method] += max(
                        0,
                        top_k - r.get("rank", top_k) + 1
                    )

        best_column_method = (
            max(method_points, key=lambda m: method_points[m])
            if method_points
            else None
        )

        # ─────────────────────────────────────────────────────
        # 3-column comparison
        # ─────────────────────────────────────────────────────
        st.markdown("### 🔍 All Methods Compared")

        col1, col2, col3 = st.columns(3)

        for col, method in zip([col1, col2, col3], ["bm25", "dense", "hybrid"]):
            data = results.get(method)
            is_best = method == best_column_method

            with col:
                if not data or not data.get("results"):
                    st.markdown(f"### {method.upper()}")
                    st.warning("No results found")
                    continue

                if is_best:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#1a4a1a;
                            border:2px solid #00cc44;
                            border-radius:8px;
                            padding:10px;
                            margin-bottom:10px;
                        ">
                            <h4 style="color:#00cc44">✅ {method.upper()} — TOP AGREEMENT</h4>
                            <small style="color:#aaa">{data.get('latency_ms', 'N/A')} ms</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.markdown(f"### {method.upper()}")
                    st.caption(f"{data.get('latency_ms', 'N/A')} ms")

                for r in data["results"]:
                    title = r.get("title", "Unknown title")
                    rank = r.get("rank", "?")

                    with st.expander(f"#{rank} — {title[:50]}..."):
                        st.metric("Score", r.get("score", 0))
                        st.write(f"**Authors:** {r.get('authors', 'Unknown')}")
                        st.write(f"**Year:** {r.get('year', 'Unknown')}")
                        st.write(r.get("snippet", "No snippet available."))

elif query and len(query) <= 2:
    st.info("Type at least 3 characters to search.")

# ─────────────────────────────────────────────────────────────
# Analytics & Evaluation Charts
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 System Analytics")

CACHE = Path("notebook_cache")


# Define all charts with improved descriptions
all_charts = {
    "📈 Evaluation": [
        (
            CACHE / "04_evaluation" / "eval_bar_chart.png",
            "Quality and Latency Comparison",
            "This chart compares BM25, Dense, and Hybrid retrieval using Recall@5, NDCG@5, "
            "MRR, average latency, and P95 latency. The quality chart shows how well each "
            "method retrieves and ranks relevant papers, while the latency chart shows how fast "
            "each method responds. Dense performs strongly in ranking quality, BM25 is the fastest, "
            "and Hybrid provides a combined retrieval approach using both lexical and semantic signals."
        ),
        (
            CACHE / "04_evaluation" / "eval_radar.png",
            "Method Profile Radar Chart",
            "This radar chart summarizes each retrieval method across multiple factors at the same time: "
            "Recall@5, NDCG@5, MRR, and speed. A larger shape means stronger overall performance. "
            "BM25 has an advantage in speed, Dense performs well on semantic quality metrics, and Hybrid "
            "shows how combining methods can balance retrieval effectiveness with search performance."
        ),
        (
            CACHE / "04_evaluation" / "eval_curves.png",
            "Recall@k and NDCG@k Curves",
            "These curves show how retrieval performance changes as k increases from 1 to 10. "
            "Recall@k measures how many relevant papers are found as more results are returned, while "
            "NDCG@k measures whether the most relevant papers appear near the top. A method with higher "
            "curves is better at finding useful papers earlier in the ranked list."
        ),
        (
            CACHE / "04_evaluation" / "eval_heatmap.png",
            "Per-Query NDCG@5 Heatmap",
            "This heatmap shows the NDCG@5 score for each query across BM25, Dense, and Hybrid retrieval. "
            "Each row represents one query, and darker green cells indicate stronger ranking performance. "
            "The chart helps identify which queries each method handles well and which query types are more "
            "challenging for a specific retrieval approach."
        ),
    ],

    "🔵 Embeddings": [
        (
            CACHE / "02_semantic_retrieval" / "pca_embeddings.png",
            "Chunk Embeddings Visualized with PCA",
            "This visualization shows high-dimensional chunk embeddings reduced into 2D using PCA. "
            "Each point represents a text chunk from a research paper, and colors represent different papers. "
            "Nearby points suggest that chunks have similar semantic meaning. This helps explain how Dense "
            "retrieval can find related content based on meaning rather than exact keyword matching."
        ),
        (
            CACHE / "02_semantic_retrieval" / "latency_comparison.png",
            "Search Latency Comparison",
            "This chart compares the average latency and P95 latency of BM25, Dense, and Hybrid search. "
            "Average latency shows the typical response time, while P95 latency shows the slower cases. "
            "BM25 is usually fastest because it uses keyword-based matching, while Dense and Hybrid search "
            "take longer because they rely on vector similarity and additional retrieval processing."
        ),
        (
            CACHE / "02_semantic_retrieval" / "top5_results.png",
            "Top-5 Retrieved Results Example",
            "This chart shows an example of the top five retrieved chunks returned by BM25, Dense, and Hybrid "
            "for the same query. It helps compare how each method ranks its results and whether the retrieved "
            "chunks appear relevant to the query. This is useful for qualitative evaluation, not just numerical "
            "metric comparison."
        ),
    ],

    "🕸️ Graph": [
        (
            CACHE / "03_graph_build" / "neo4j_dataflow_diagram.png",
            "Neo4j GraphRAG Architecture",
            "This diagram shows how the full retrieval system is connected. The ingestion stage extracts text, "
            "chunks, and metadata from PDFs. MongoDB stores the raw chunks and metadata, Qdrant stores vector "
            "embeddings for semantic search, and Neo4j stores the knowledge graph. The graph connects Paper, "
            "Author, and Topic nodes through relationships such as WROTE and HAS_TOPIC, allowing GraphRAG to "
            "add related context beyond the initially retrieved results."
        ),
    ],
}

tab1, tab2, tab3 = st.tabs(list(all_charts.keys()))

for tab, (tab_name, charts) in zip([tab1, tab2, tab3], all_charts.items()):
    with tab:
        existing = [(p, cap, desc) for p, cap, desc in charts if p.exists()]

        if not existing:
            st.info("No charts found in cache.")
            continue

        state_key = f"chart_idx_{tab_name}"
        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        idx = st.session_state[state_key]
        idx = max(0, min(idx, len(existing) - 1))
        path, caption, description = existing[idx]

        # ── Navigation ────────────────────────────────────
        col_prev, col_info, col_next = st.columns([1, 4, 1])

        with col_prev:
            if st.button("◀", key=f"prev_{tab_name}", disabled=(idx == 0)):
                st.session_state[state_key] -= 1
                st.rerun()

        with col_info:
            st.markdown(
                f"<div style='text-align:center; color:#aaa; padding-top:8px;'>"
                f"{idx + 1} / {len(existing)} — <b>{caption}</b></div>",
                unsafe_allow_html=True
            )

        with col_next:
            if st.button("▶", key=f"next_{tab_name}", disabled=(idx == len(existing) - 1)):
                st.session_state[state_key] += 1
                st.rerun()

        # ── Fixed size image ──────────────────────────────
        st.markdown(
            f"<div style='display:flex; justify-content:center;'>"
            f"<img src='data:image/png;base64,{_img_to_b64(path)}' "
            f"style='width:700px; height:450px; object-fit:contain; "
            f"background:#161b22; border-radius:8px; padding:10px;'/>"
            f"</div>",
            unsafe_allow_html=True
        )

        # ── Description ───────────────────────────────────
        st.markdown(
            f"<div style='background:#1e1e2e; border-left:3px solid #00cc44; "
            f"padding:12px; border-radius:4px; margin-top:12px; color:#ccc;'>"
            f"{description}"
            f"</div>",
            unsafe_allow_html=True
        )