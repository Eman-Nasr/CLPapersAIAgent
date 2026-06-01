import streamlit as st
import requests

st.set_page_config(page_title="Paper Search", page_icon="🔍", layout="wide")
st.title("🔍 Research Paper Search")
st.caption("BM25 | Dense | Hybrid Retrieval")

BASE = "http://localhost:8000"  # FastAPI must be running

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of results", 1, 10, 5)

# ── Search bar ────────────────────────────────────────────────
query = st.text_input("Enter your query", placeholder="e.g. retrieval augmented generation")

# NEW
if query:
    with st.spinner("Searching all 3 methods..."):
        results = {}
        for method in ["bm25", "dense", "hybrid"]:
            resp = requests.post(f"{BASE}/search", json={
                "query": query, "top_k": top_k, "method": method
            })
            if resp.status_code == 200:
                results[method] = resp.json()

    if results:
    # ── Find best method (highest top-1 score) ────────────
        valid_methods = {
            m: d for m, d in results.items() 
            if d.get("results")  # skip methods with empty results
        }

        if not valid_methods:
            st.warning("No results found for this query. Try different keywords.")
        
        else:
            best_method = max(valid_methods, key=lambda m: valid_methods[m]["results"][0]["score"])
            best_result = valid_methods[best_method]["results"][0]

            st.markdown("---")
            st.markdown(f"""
            ### 🏆 Best Match for this Query
            **Method:** `{best_method.upper()}`  
            **Paper:** {best_result['title']}  
            **Score:** {best_result['score']}  
            **Authors:** {best_result['authors']}  
            > {best_result['snippet']}
            """)
            st.markdown("---")

            st.markdown("### 🔍 All Methods Compared")
            col1, col2, col3 = st.columns(3)

            for col, method in zip([col1, col2, col3], ["bm25", "dense", "hybrid"]):
                data = results.get(method)
                is_best = (method == best_method)

                with col:
                    if not data or not data.get("results"):
                        # ── No results for this method ────────
                        st.markdown(f"### {method.upper()}")
                        st.warning("No results found")
                        continue

                    if is_best:
                        st.markdown(
                            f"<div style='background-color:#1a4a1a; border:2px solid #00cc44;"
                            f"border-radius:8px; padding:10px; margin-bottom:10px;'>"
                            f"<h4 style='color:#00cc44'>✅ {method.upper()} — BEST</h4>"
                            f"<small style='color:#aaa'>{data['latency_ms']} ms</small></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(f"### {method.upper()}")
                        st.caption(f"{data['latency_ms']} ms")

                    for r in data["results"]:
                        with st.expander(f"#{r['rank']} — {r['title'][:50]}..."):
                            st.metric("Score", r["score"])
                            st.write(f"**Authors:** {r['authors']}")
                            st.write(f"**Year:** {r['year']}")
                            st.write(r["snippet"])

# ── Example queries ───────────────────────────────────────────
st.divider()
st.subheader("💡 Example Queries")
examples = [
    "transformer architecture for sequence tasks",
    "retrieval augmented generation",
    "large language model alignment"
]
for ex in examples:
    if st.button(ex):
        st.rerun()   # re-runs with the query pre-filled if you wire it up