from pathlib import Path

ROOT_DIR    = Path(__file__).parent.parent  
PAPERS_DIR  = ROOT_DIR / "data" / "papers"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Auto-increment test folder 
def get_next_output_dir() -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [d for d in OUTPUTS_DIR.iterdir()
                if d.is_dir() and d.name.startswith("test")]
    indices = []
    for d in existing:
        try:
            indices.append(int(d.name.replace("test", "")))
        except ValueError:
            pass
    next_idx = max(indices, default=0) + 1
    return OUTPUTS_DIR / f"test{next_idx}"

# Chunking 
CHUNK_SIZE_WORDS    = 400
CHUNK_OVERLAP_WORDS = 50

# Retrieval 
TOP_K = 5

# Metadata fields 
METADATA_FIELDS = ["paper_id", "title", "authors", "year", "filename", "page_count"]