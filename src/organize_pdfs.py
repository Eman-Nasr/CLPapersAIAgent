import json, hashlib
from pathlib import Path
from src.config import PAPERS_DIR

def compute_md5(path: Path, chunk_size: int = 65_536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()

def organize_pdfs(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PAPERS_DIR}")

    inventory, duplicates, invalid = [], [], []
    seen_hashes: dict[str, str] = {}

    for pdf in pdf_files:
        md5 = compute_md5(pdf)
        if md5 in seen_hashes:
            duplicates.append({"file": pdf.name, "duplicate_of": seen_hashes[md5]})
            continue
        seen_hashes[md5] = pdf.name

        # Basic PDF check (starts with %PDF)
        with open(pdf, "rb") as f:
            header = f.read(4)
        if header != b"%PDF":
            invalid.append(pdf.name)
            continue

        inventory.append({
            "filename": pdf.name,
            "paper_id": pdf.stem,
            "filepath": str(pdf),
            "md5": md5,
            "size_bytes": pdf.stat().st_size
        })

    report = {
        "total_found":    len(pdf_files),
        "valid":          len(inventory),
        "duplicates":     duplicates,
        "invalid":        invalid
    }

    (output_dir / "pdf_inventory.json").write_text(json.dumps(inventory, indent=2))
    (output_dir / "organization_report.json").write_text(json.dumps(report, indent=2))

    print(f"[organize] {len(inventory)} valid PDFs | "
          f"{len(duplicates)} duplicates | {len(invalid)} invalid")
    return inventory


if __name__ == "__main__":
    from src.config import get_next_output_dir
    organize_pdfs(get_next_output_dir())