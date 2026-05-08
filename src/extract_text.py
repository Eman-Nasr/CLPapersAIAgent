import json, time
from pathlib import Path
import fitz  # PyMuPDF

def extract_text_from_pdf(filepath: str) -> list[dict]:
    """Returns a list of {page_num, text} dicts."""
    pages = []
    try:
        doc = fitz.open(filepath)
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            pages.append({"page_num": i + 1, "text": text})
        doc.close()
    except Exception as e:
        pages.append({"page_num": -1, "text": f"[EXTRACTION ERROR: {e}]"})
    return pages

def extract_all(metadata_list: list[dict], output_dir: Path) -> dict:
    extracted_dir = output_dir / "extracted_text"
    extracted_dir.mkdir(parents=True, exist_ok=True)

    extraction_log = []
    all_text = {}

    for meta in metadata_list:
        paper_id = meta["paper_id"]
        filepath = meta["filepath"]
        t0       = time.time()

        pages    = extract_text_from_pdf(filepath)
        elapsed  = round(time.time() - t0, 3)

        total_chars = sum(len(p["text"]) for p in pages)
        empty_pages = sum(1 for p in pages if len(p["text"]) < 20)

        # Save per-paper JSON
        out_file = extracted_dir / f"{paper_id}.json"
        out_file.write_text(
            json.dumps({
                "paper_id": paper_id,
                "filename": meta["filename"],
                "pages": pages
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        all_text[paper_id] = pages
        extraction_log.append({
            "paper_id":    paper_id,
            "pages":       len(pages),
            "total_chars": total_chars,
            "empty_pages": empty_pages,
            "elapsed_s":   elapsed,
            "status":      "ok" if total_chars > 100 else "low_content"
        })

    # Save combined + log
    (output_dir / "all_extracted_text.json").write_text(
    json.dumps(all_text, indent=2, ensure_ascii=False),
    encoding="utf-8"
    )

    (output_dir / "extraction_log.json").write_text(
        json.dumps(extraction_log, indent=2),
        encoding="utf-8"
    )

    ok_count  = sum(1 for e in extraction_log if e["status"] == "ok")
    print(f"[extract] {ok_count}/{len(metadata_list)} papers extracted successfully")
    return all_text


if __name__ == "__main__":
    import json
    from src.config import OUTPUTS_DIR
    tests = sorted(OUTPUTS_DIR.glob("test*"))
    out   = tests[-1]
    meta  = json.loads((out / "metadata.json").read_text())
    extract_all(meta, out)