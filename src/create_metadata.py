import json, re, csv
from pathlib import Path
import fitz  # PyMuPDF
from src.config import METADATA_FIELDS

PAPER_ID_RE = re.compile(r"^paper(\d+)$", re.IGNORECASE)

YEAR_RE = re.compile(r"\b(20[2-9]\d)\b")

MONTH_YEAR_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},?\s*(20[2-9]\d)"
    r"|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+(20[2-9]\d)",
    re.IGNORECASE,
)


def _extract_year(text: str) -> str:
    month_matches = MONTH_YEAR_RE.findall(text)
    for pair in month_matches:
        for y in pair:
            if y:
                return y
    years = YEAR_RE.findall(text)
    return years[0] if years else "unknown"


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _heuristic_title(first_page_text: str) -> str:
    lines = [_clean_line(l) for l in first_page_text.splitlines()
             if _clean_line(l)][:40]
    candidates = []
    for line in lines:
        # Skip obvious non-title lines
        if re.match(r"^\d+$", line):                 
            continue
        if re.search(r"@|arxiv|preprint|http|doi\.", line, re.I):
            continue
        if len(line) < 10 or len(line) > 250:
            continue
        candidates.append(line)

    if not candidates:
        return "unknown"

    # Prefer lines that are ALL-CAPS or Title Case and reasonably long
    for line in candidates:
        if len(line) > 20 and (line.istitle() or line.isupper()):
            return line

    # Fallback: first reasonable-length candidate
    for line in candidates:
        if len(line) > 20:
            return line

    return candidates[0] if candidates else "unknown"


def _heuristic_authors(first_page_text: str) -> str:
    lines = [_clean_line(l) for l in first_page_text.splitlines()
             if _clean_line(l)][:60]


    author_pattern = re.compile(
        r"^(?:[A-Z][a-zA-Zéàèùâêîôûäëïöü\-\']+ ){1,3}"
        r"(?:and |, )?[A-Z][a-zA-Zéàèùâêîôûäëïöü\-\']+"
    )
    skip = re.compile(r"@|university|institute|department|school|abstract|"
                      r"introduction|arxiv|http|©|\d{4}", re.I)

    authors_found = []
    for line in lines:
        if skip.search(line):
            continue
        if author_pattern.match(line) and len(line) < 200:
            authors_found.append(line)
            if len(authors_found) >= 3:
                break

    return "; ".join(authors_found) if authors_found else "unknown"


def _extract_meta_from_pdf(filepath: str) -> dict:
    result = {"title": "unknown", "authors": "unknown", "year": "unknown"}
    try:
        doc = fitz.open(filepath)
        pdf_meta = doc.metadata 

        raw_title = (pdf_meta.get("title") or "").strip()
        if raw_title and len(raw_title) > 5:
            result["title"] = raw_title

        raw_author = (pdf_meta.get("author") or "").strip()
        if raw_author and len(raw_author) > 2:
            result["authors"] = raw_author

        raw_date = (pdf_meta.get("creationDate") or
                    pdf_meta.get("modDate") or "").strip()
        year_match = YEAR_RE.search(raw_date)
        if year_match:
            result["year"] = year_match.group(1)

        first_page_text = ""
        if doc.page_count > 0:
            first_page_text = doc[0].get_text("text")

        if result["title"] == "unknown" and first_page_text:
            result["title"] = _heuristic_title(first_page_text)

        if result["authors"] == "unknown" and first_page_text:
            result["authors"] = _heuristic_authors(first_page_text)

        if result["year"] == "unknown" and first_page_text:
            two_pages = first_page_text
            if doc.page_count > 1:
                two_pages += doc[1].get_text("text")
            result["year"] = _extract_year(two_pages)

        doc.close()
    except Exception as e:
        result["_error"] = str(e)

    return result


def _parse_filename(filename: str) -> dict:
    stem  = Path(filename).stem
    match = PAPER_ID_RE.match(stem)
    if match:
        number   = int(match.group(1))
        paper_id = f"paper{number}"
    else:
        paper_id = stem
    return {"paper_id": paper_id, "filename": filename}


def create_metadata(inventory: list[dict], output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_list = []

    # Sort by numeric paper id
    def sort_key(item):
        m = PAPER_ID_RE.match(Path(item["filename"]).stem)
        return int(m.group(1)) if m else 0

    for item in sorted(inventory, key=sort_key):
        meta = _parse_filename(item["filename"])

        enriched = _extract_meta_from_pdf(item["filepath"])
        meta.update({
            "title":      enriched.get("title",   "unknown"),
            "authors":    enriched.get("authors", "unknown"),
            "year":       enriched.get("year",    "unknown"),
            "page_count": item.get("page_count") or _get_page_count(item["filepath"]),
            "size_bytes": item["size_bytes"],
            "md5":        item["md5"],
            "filepath":   item["filepath"],
        })
        for field in METADATA_FIELDS:
            meta.setdefault(field, "unknown")
        metadata_list.append(meta)

    out_json = output_dir / "metadata.json"
    out_json.write_text(json.dumps(metadata_list, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    out_csv = output_dir / "metadata.csv"
    csv_fields = ["paper_id", "title", "authors", "year", "filename",
                  "page_count", "size_bytes", "md5"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata_list)

    print(f"[metadata] {len(metadata_list)} papers → {out_json}  |  {out_csv}")
    return metadata_list


def _get_page_count(filepath: str) -> int:
    try:
        doc = fitz.open(filepath)
        n   = doc.page_count
        doc.close()
        return n
    except Exception:
        return -1


if __name__ == "__main__":
    import json
    from src.config import OUTPUTS_DIR
    tests = sorted(OUTPUTS_DIR.glob("test*"))
    out   = tests[-1]
    inv   = json.loads((out / "pdf_inventory.json").read_text())
    create_metadata(inv, out)
