import json
from pathlib import Path
from src.config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS

def words_to_text(words: list[str]) -> str:
    return " ".join(words)

def chunk_paper(paper_id: str, pages: list[dict],
                chunk_size: int = CHUNK_SIZE_WORDS,
                overlap:    int = CHUNK_OVERLAP_WORDS) -> list[dict]:
    # Concatenate all page text
    full_text = " ".join(p["text"] for p in pages if p["text"])
    words     = full_text.split()

    chunks, idx, chunk_idx = [], 0, 0
    step = chunk_size - overlap

    while idx < len(words):
        window = words[idx: idx + chunk_size]
        chunks.append({
            "chunk_id":   f"{paper_id}_chunk_{chunk_idx:04d}",
            "paper_id":   paper_id,
            "chunk_index": chunk_idx,
            "start_word": idx,
            "end_word":   idx + len(window),
            "word_count": len(window),
            "text":       words_to_text(window)
        })
        chunk_idx += 1
        idx       += step

    return chunks

def chunk_all(all_text: dict, output_dir: Path) -> list[dict]:
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_chunks   = []
    chunk_stats  = []

    for paper_id, pages in all_text.items():
        paper_chunks = chunk_paper(paper_id, pages)
        all_chunks.extend(paper_chunks)

        # Save per-paper chunks
        (chunks_dir / f"{paper_id}_chunks.json").write_text(
            json.dumps(paper_chunks, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        chunk_stats.append({
            "paper_id":   paper_id,
            "num_chunks": len(paper_chunks),
            "total_words": sum(c["word_count"] for c in paper_chunks)
        })

    # Save all chunks combined
    (output_dir / "all_chunks.json").write_text(
        json.dumps(all_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (output_dir / "chunk_stats.json").write_text(
        json.dumps(chunk_stats, indent=2),
        encoding="utf-8"
    )

    avg = sum(s["num_chunks"] for s in chunk_stats) / max(len(chunk_stats), 1)
    print(f"[chunk] {len(all_chunks)} total chunks across "
          f"{len(chunk_stats)} papers (avg {avg:.1f} chunks/paper)")
    return all_chunks


if __name__ == "__main__":
    import json
    from src.config import OUTPUTS_DIR
    tests = sorted(OUTPUTS_DIR.glob("test*"))
    out   = tests[-1]
    all_text = json.loads((out / "all_extracted_text.json").read_text())
    chunk_all(all_text, out)