import sys, json, time
from pathlib import Path
from datetime import datetime

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config             import get_next_output_dir
from organize_pdfs      import organize_pdfs
from create_metadata    import create_metadata
from extract_text       import extract_all
from chunk_text         import chunk_all
from create_query_set   import create_query_set
from build_tfidf_retrieval import build_and_retrieve

def run_pipeline():
    OUT = get_next_output_dir()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PDF-Papers AI Agent  |  Output → {OUT}")
    print(f"{'='*60}\n")

    run_log = {
        "output_dir": str(OUT),
        "started_at": datetime.now().isoformat(),
        "steps":      {}
    }
    t_total = time.time()

    def timed_step(name, fn, *args):
        print(f"── Step: {name}")
        t       = time.time()
        result  = fn(*args)
        elapsed = round(time.time() - t, 2)
        run_log["steps"][name] = {"elapsed_s": elapsed, "status": "ok"}
        print(f"   ✓ done in {elapsed}s\n")
        return result

    inventory     = timed_step("01_organize_pdfs",    organize_pdfs,       OUT)
    metadata_list = timed_step("02_create_metadata",  create_metadata,     inventory, OUT)
    all_text      = timed_step("03_extract_text",     extract_all,         metadata_list, OUT)
    all_chunks    = timed_step("04_chunk_text",       chunk_all,           all_text, OUT)
    queries       = timed_step("05_create_query_set", create_query_set,    OUT)
    retrieval_out = timed_step("06_build_tfidf",      build_and_retrieve,  all_chunks, queries, OUT)

    run_log["finished_at"]     = datetime.now().isoformat()
    run_log["total_elapsed_s"] = round(time.time() - t_total, 2)
    run_log["summary"] = {
        "papers":          len(metadata_list),
        "chunks":          len(all_chunks),
        "queries":         len(queries),
        "index_vocab":     retrieval_out["index_meta"]["vocab_size"],
        "retrieval_top_k": retrieval_out["index_meta"]["top_k"]
    }

    (OUT / "run_summary.json").write_text(json.dumps(run_log, indent=2))

    print(f"{'='*60}")
    print(f"  Pipeline complete in {run_log['total_elapsed_s']}s")
    print(f"  Papers : {run_log['summary']['papers']}")
    print(f"  Chunks : {run_log['summary']['chunks']}")
    print(f"  Vocab  : {run_log['summary']['index_vocab']}")
    print(f"  Output : {OUT}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_pipeline()