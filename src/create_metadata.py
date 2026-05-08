import json, re
from pathlib import Path
import fitz  # PyMuPDF
from config import METADATA_FIELDS

# Matches: paper1.pdf, paper23.pdf, paper208.pdf
PAPER_ID_RE = re.compile(r"^paper(\d+)$", re.IGNORECASE)

def parse_filename(filename: str) -> dict:
    stem  = Path(filename).stem          
    match = PAPER_ID_RE.match(stem)

    if match:
        number   = int(match.group(1))
        paper_id = f"paper{number}"
        title    = f"Paper {number}"    
    else:
        # Fallback for any unexpected naming
        paper_id = stem
        title    = stem.replace("_", " ").replace("-", " ")

    return {"paper_id": paper_id, "title": title, "filename": filename}

def get_page_count(filepath: str) -> int:
    try:
        doc = fitz.open(filepath)
        n   = doc.page_count
        doc.close()
        return n
    except Exception:
        return -1

def create_metadata(inventory: list[dict], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_list = []

    for item in sorted(inventory, key=lambda x: int(
            PAPER_ID_RE.match(Path(x["filename"]).stem).group(1)
            if PAPER_ID_RE.match(Path(x["filename"]).stem) else 0)):

        meta = parse_filename(item["filename"])
        meta.update({
            "authors":    "unknown",
            "year":       "2024",          # default; enrich later if needed
            "page_count": get_page_count(item["filepath"]),
            "size_bytes": item["size_bytes"],
            "md5":        item["md5"],
            "filepath":   item["filepath"]
        })
        for field in METADATA_FIELDS:
            meta.setdefault(field, "unknown")
        metadata_list.append(meta)

    out_path = output_dir / "metadata.json"
    out_path.write_text(json.dumps(metadata_list, indent=2))
    print(f"[metadata] Created metadata for {len(metadata_list)} papers → {out_path}")
    return metadata_list


if __name__ == "__main__":
    import json
    from config import OUTPUTS_DIR
    tests = sorted(OUTPUTS_DIR.glob("test*"))
    out   = tests[-1]
    inv   = json.loads((out / "pdf_inventory.json").read_text())
    create_metadata(inv, out)