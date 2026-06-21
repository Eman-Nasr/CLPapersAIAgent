import sys
import os
import json
import copy
import base64
import requests
from pathlib import Path
from collections import defaultdict

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ── Import notebook functions safely ────────────────────────────────────────
import os
from IPython.utils.capture import capture_output

def _load_nb(path, gdict):
    """
    Execute graphrag_executor.ipynb safely.
    Important: run it from the notebooks folder so Path("..") inside the
    notebook points to the project root.
    """
    path = Path(path).resolve()
    old_cwd = os.getcwd()

    gdict["__name__"] = "__streamlit_notebook__"
    gdict["__file__"] = str(path)

    # Notebook compatibility for Streamlit execution
    from IPython.display import display, HTML, Markdown

    gdict["display"] = display
    gdict["HTML"] = HTML
    gdict["Markdown"] = Markdown

    try:
        os.chdir(path.parent)

        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        for i, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") != "code":
                continue

            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)

            if not src.strip():
                continue

            try:
                with capture_output():
                    exec(compile(src, str(path), "exec"), gdict)
            except Exception as exc:
                # Do NOT silently ignore errors. Show which notebook cell failed.
                raise RuntimeError(
                    f"Error while loading {path.name}, code cell {i}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    finally:
        os.chdir(old_cwd)

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR    = PROJECT_ROOT / "notebook_cache"
INGESTION_DIR = CACHE_DIR / "01_ingestion"
NB_DIR = PROJECT_ROOT / "notebooks"

@st.cache_resource(show_spinner="Loading GraphRAG pipeline…")
def _load_pipeline():
    g = {}

    nb_path = NB_DIR / "graphrag_executor.ipynb"
    if not nb_path.exists():
        st.error(f"Notebook not found: {nb_path}")
        st.stop()

    _load_nb(nb_path, g)

    required = ["ask", "tfidf_vectorizer"]
    missing = [name for name in required if name not in g]

    if missing:
        st.error(
            "GraphRAG notebook loaded, but some required variables are missing: "
            f"{missing}. This usually means one setup cell in graphrag_executor.ipynb "
            "did not run correctly."
        )
        st.stop()

    meta_path = INGESTION_DIR / "metadata.json"
    if not meta_path.exists():
        st.error(f"Metadata file not found: {meta_path}")
        st.stop()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    g["TRUSTED_PAPER_IDS"] = {m["paper_id"] for m in meta if "paper_id" in m}

    return g

# ── Safety helpers (inline, no import needed) ───────────────────────────────
def _provenance_filter(response, trusted):
    resp = copy.deepcopy(response)
    ev  = [e for e in resp.get("final_evidence", []) if e.get("paper_id") in trusted]
    cit = [c for c in resp.get("citations",       []) if c.get("paper_id") in trusted]
    for i, x in enumerate(ev,  1): x["rank"] = i
    for i, x in enumerate(cit, 1): x["rank"] = i
    resp["final_evidence"] = ev
    resp["citations"]      = cit
    resp.setdefault("safety", {})["provenance_filter_applied"] = True
    return resp

def _pin_sources(response, allowed):
    resp = copy.deepcopy(response)
    ev  = [e for e in resp.get("final_evidence", []) if e.get("paper_id") in allowed]
    cit = [c for c in resp.get("citations",       []) if c.get("paper_id") in allowed]
    for i, x in enumerate(ev,  1): x["rank"] = i
    for i, x in enumerate(cit, 1): x["rank"] = i
    resp["final_evidence"] = ev
    resp["citations"]      = cit
    resp.setdefault("safety", {})["source_pinning_applied"] = True
    return resp

def _pin_sources(response, allowed):
    resp = copy.deepcopy(response)
    ev  = [e for e in resp.get("final_evidence", []) if e.get("paper_id") in allowed]
    cit = [c for c in resp.get("citations",       []) if c.get("paper_id") in allowed]
    for i, x in enumerate(ev,  1): x["rank"] = i
    for i, x in enumerate(cit, 1): x["rank"] = i
    resp["final_evidence"] = ev
    resp["citations"]      = cit
    resp.setdefault("safety", {})["source_pinning_applied"] = True
    return resp

# ── Old demo helpers: BM25 / Dense / Hybrid comparison ─────────────────────
BASE = "http://localhost:8000"  # FastAPI backend must be running


def _img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def set_query(title):
    st.session_state.search_query_input = title


def call_search_api(query, top_k, method):
    try:
        resp = requests.post(
            f"{BASE}/search",
            json={
                "query": query,
                "top_k": top_k,
                "method": method,
            },
            timeout=20,
        )

        if resp.status_code == 200:
            return resp.json(), None

        return None, f"{method.upper()} returned status code {resp.status_code}"

    except requests.exceptions.RequestException as e:
        return None, str(e)
    
# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CLPapersAIAgent", page_icon="📚", layout="wide")
st.title("📚 CLPapersAIAgent Demo")
st.caption("GraphRAG safety demo + BM25 / Dense / Hybrid retrieval comparison")

tab_graphrag, tab_search, tab_analytics = st.tabs(
    ["🧠 GraphRAG Ask", "🔍 BM25 / Dense / Hybrid Search", "📊 System Analytics"]
)

# =============================================================================
# TAB 1 — GraphRAG Ask
# =============================================================================
with tab_graphrag:
    with st.sidebar:
        st.header("⚙️ GraphRAG Settings")
        mode = st.selectbox(
            "Retrieval mode",
            ["hybrid_graphrag", "graph_guided", "vector_only"],
        )
        top_k = st.slider("Top-k seed chunks", 3, 10, 5)
        max_ev = st.slider("Max evidence chunks", 4, 16, 8)

        st.subheader("🔒 Safety controls")
        use_prov = st.checkbox(
            "Provenance filter",
            value=True,
            help="Remove any evidence not from the ingested corpus",
        )
        pin_ids_raw = st.text_area(
            "Source pinning (optional)",
            placeholder="Paste comma-separated paper_ids to restrict sources…",
            height=80,
        )

    question = st.text_input(
        "Ask a question about CL papers:",
        "How do recent papers improve retrieval augmented generation using reranking, graph-based expansion, and evaluation methods?",
        key="graphrag_question",
    )

    if st.button("Ask GraphRAG", type="primary"):
        g = _load_pipeline()
        ask_fn = g["ask"]
        trusted = g["TRUSTED_PAPER_IDS"]

        with st.spinner("Running GraphRAG pipeline…"):
            raw = ask_fn(
                question=question,
                mode=mode,
                top_k=top_k,
                max_total_evidence_chunks=max_ev,
                save_trace=False,
            )

        resp = copy.deepcopy(raw)

        if use_prov:
            resp = _provenance_filter(resp, trusted)

        if pin_ids_raw.strip():
            allowed_ids = {p.strip() for p in pin_ids_raw.split(",") if p.strip()}
            resp = _pin_sources(resp, allowed_ids)

        st.subheader("📝 Answer")
        st.write(resp.get("answer", "No answer generated."))

        with st.expander("📑 Citations", expanded=True):
            cit_df = pd.DataFrame(resp.get("citations", []))
            if not cit_df.empty:
                st.dataframe(cit_df, use_container_width=True)
            else:
                st.info("No citations available.")

        with st.expander("🔍 Evidence after safety"):
            ev_df = pd.DataFrame(resp.get("final_evidence", []))
            if not ev_df.empty:
                cols = [
                    "rank",
                    "source",
                    "paper_id",
                    "title",
                    "page_start",
                    "page_end",
                    "final_score",
                    "graph_reasons",
                ]
                cols = [c for c in cols if c in ev_df.columns]
                st.dataframe(ev_df[cols], use_container_width=True)
            else:
                st.info("No evidence available after filters.")

        with st.expander("⚖️ Before vs After safety"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Raw, before safety**")
                st.metric("Evidence chunks", len(raw.get("final_evidence", [])))
                st.metric("Citations", len(raw.get("citations", [])))

            with col2:
                st.markdown("**After safety filters**")
                st.metric("Evidence chunks", len(resp.get("final_evidence", [])))
                st.metric("Citations", len(resp.get("citations", [])))

            st.json(resp.get("safety", {}))

        with st.expander("🕵️ Quality checks"):
            st.json(resp.get("quality_checks", {}))

        with st.expander("⏱️ Timing"):
            st.json(resp.get("timing", {}))


# =============================================================================
# TAB 2 — Old BM25 / Dense / Hybrid Search UI
# =============================================================================
with tab_search:
    st.subheader("🔍 Research Paper Search")
    st.caption("BM25 | Dense | Hybrid Retrieval")

    col_a, col_b = st.columns([3, 1])

    with col_b:
        top_k_search = st.slider(
            "Number of results",
            1,
            10,
            5,
            key="search_top_k",
        )
        st.caption("Backend URL")
        st.code(BASE)

    if "search_query_input" not in st.session_state:
        st.session_state.search_query_input = ""

    search_query_input = st.text_input(
        "Enter your query",
        placeholder="e.g. retrieval augmented generation",
        key="search_query_input",
    )

    search_query = search_query_input.strip()

    if not search_query:
        st.divider()
        st.subheader("💡 Example Queries")

        examples = [
            "retrieval augmented generation",
            "transformer architecture for sequence tasks",
            "large language model alignment",
            "reranking in retrieval augmented generation",
        ]

        for ex in examples:
            st.button(
                ex,
                key=f"example_{ex}",
                on_click=set_query,
                args=(ex,),
            )

    if search_query and len(search_query) > 2:
        results = {}
        errors = {}

        with st.spinner("Searching BM25, Dense, and Hybrid methods..."):
            for method in ["bm25", "dense", "hybrid"]:
                data, error = call_search_api(search_query, top_k_search, method)

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
            st.warning(
                "No results found. Make sure FastAPI is running on port 8000."
            )

        else:
            # Rank-based voting + global normalized score
            paper_votes = defaultdict(
                lambda: {
                    "points": 0,
                    "norm_score_sum": 0.0,
                    "methods": [],
                    "result": None,
                }
            )

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
                    paper_key = (
                        r.get("paper_id")
                        or r.get("title", "").strip().lower()
                    )
                    rank = r.get("rank", top_k_search)
                    points = max(0, top_k_search - rank + 1)
                    norm_score = normalize(r.get("score", 0))

                    paper_votes[paper_key]["points"] += points
                    paper_votes[paper_key]["norm_score_sum"] += norm_score
                    paper_votes[paper_key]["methods"].append(method.upper())
                    paper_votes[paper_key]["result"] = r

            best_key = max(
                paper_votes,
                key=lambda k: (
                    paper_votes[k]["points"]
                    * (
                        paper_votes[k]["norm_score_sum"]
                        / len(paper_votes[k]["methods"])
                    )
                ),
            )

            best_info = paper_votes[best_key]
            best_result = best_info["result"]
            best_methods = list(dict.fromkeys(best_info["methods"]))

            st.markdown("---")
            st.markdown(
                f"""
### 🏆 Most Agreed Result for this Query

**Paper:** {best_result.get('title', 'Unknown title')}  
**Found by:** {", ".join(best_methods)}  
**Agreement Score:** {best_info["points"]}  
**Authors:** {best_result.get('authors', 'Unknown')}  

> {best_result.get('snippet', 'No snippet available.')}
"""
            )
            st.markdown("---")

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
                    (bm25_ids & dense_ids)
                    | (bm25_ids & hybrid_ids)
                    | (dense_ids & hybrid_ids)
                ) - all_three

                if all_three:
                    st.success(
                        f"✅ {len(all_three)} paper(s) found by ALL 3 methods — highly relevant"
                    )

                    for pid in all_three:
                        r = (
                            bm25_papers.get(pid)
                            or dense_papers.get(pid)
                            or hybrid_papers.get(pid)
                        )
                        st.markdown(f"- 📄 **{r.get('title', 'Unknown title')}**")

                if two_of_three:
                    st.info(f"📌 {len(two_of_three)} paper(s) found by 2 methods")

                    for pid in two_of_three:
                        r = (
                            bm25_papers.get(pid)
                            or dense_papers.get(pid)
                            or hybrid_papers.get(pid)
                        )

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
                    st.warning(
                        "No overlap between methods. Each method returned different papers."
                    )

            else:
                st.info("Agreement summary needs results from all 3 methods.")

            method_points = defaultdict(int)

            for method, data in valid_methods.items():
                for r in data.get("results", []):
                    paper_key = (
                        r.get("paper_id")
                        or r.get("title", "").strip().lower()
                    )

                    if paper_key == best_key:
                        method_points[method] += max(
                            0,
                            top_k_search - r.get("rank", top_k_search) + 1,
                        )

            best_column_method = (
                max(method_points, key=lambda m: method_points[m])
                if method_points
                else None
            )

            st.markdown("### 🔍 All Methods Compared")

            col1, col2, col3 = st.columns(3)

            for col, method in zip(
                [col1, col2, col3],
                ["bm25", "dense", "hybrid"],
            ):
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
                            unsafe_allow_html=True,
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

    elif search_query and len(search_query) <= 2:
        st.info("Type at least 3 characters to search.")


# =============================================================================
# TAB 3 — Analytics charts from old file
# =============================================================================
with tab_analytics:
    st.subheader("📊 System Analytics")

    CACHE = PROJECT_ROOT / "notebook_cache"

    all_charts = {
        "📈 Evaluation": [
            (
                CACHE / "04_evaluation" / "eval_bar_chart.png",
                "Quality and Latency Comparison",
                "This chart compares BM25, Dense, and Hybrid retrieval using Recall@5, NDCG@5, MRR, average latency, and P95 latency.",
            ),
            (
                CACHE / "04_evaluation" / "eval_radar.png",
                "Method Profile Radar Chart",
                "This radar chart summarizes each retrieval method across Recall@5, NDCG@5, MRR, and speed.",
            ),
            (
                CACHE / "04_evaluation" / "eval_curves.png",
                "Recall@k and NDCG@k Curves",
                "These curves show how retrieval performance changes as k increases from 1 to 10.",
            ),
            (
                CACHE / "04_evaluation" / "eval_heatmap.png",
                "Per-Query NDCG@5 Heatmap",
                "This heatmap shows the NDCG@5 score for each query across BM25, Dense, and Hybrid retrieval.",
            ),
        ],
        "🔵 Embeddings": [
            (
                CACHE / "02_semantic_retrieval" / "pca_embeddings.png",
                "Chunk Embeddings Visualized with PCA",
                "This visualization shows high-dimensional chunk embeddings reduced into 2D using PCA.",
            ),
            (
                CACHE / "02_semantic_retrieval" / "latency_comparison.png",
                "Search Latency Comparison",
                "This chart compares the average latency and P95 latency of BM25, Dense, and Hybrid search.",
            ),
            (
                CACHE / "02_semantic_retrieval" / "top5_results.png",
                "Top-5 Retrieved Results Example",
                "This chart shows an example of the top five retrieved chunks returned by BM25, Dense, and Hybrid.",
            ),
        ],
        "🕸️ Graph": [
            (
                CACHE / "03_graph_build" / "neo4j_dataflow_diagram.png",
                "Neo4j GraphRAG Architecture",
                "This diagram shows how MongoDB, Qdrant, and Neo4j connect inside the full retrieval system.",
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

            col_prev, col_info, col_next = st.columns([1, 4, 1])

            with col_prev:
                if st.button("◀", key=f"prev_{tab_name}", disabled=(idx == 0)):
                    st.session_state[state_key] -= 1
                    st.rerun()

            with col_info:
                st.markdown(
                    f"<div style='text-align:center; color:#aaa; padding-top:8px;'>"
                    f"{idx + 1} / {len(existing)} — <b>{caption}</b></div>",
                    unsafe_allow_html=True,
                )

            with col_next:
                if st.button(
                    "▶",
                    key=f"next_{tab_name}",
                    disabled=(idx == len(existing) - 1),
                ):
                    st.session_state[state_key] += 1
                    st.rerun()

            st.markdown(
                f"<div style='display:flex; justify-content:center;'>"
                f"<img src='data:image/png;base64,{_img_to_b64(path)}' "
                f"style='width:700px; height:450px; object-fit:contain; "
                f"background:#161b22; border-radius:8px; padding:10px;'/>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='background:#1e1e2e; border-left:3px solid #00cc44; "
                f"padding:12px; border-radius:4px; margin-top:12px; color:#ccc;'>"
                f"{description}"
                f"</div>",
                unsafe_allow_html=True,
            )
