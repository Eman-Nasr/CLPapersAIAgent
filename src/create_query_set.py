import json, re
from pathlib import Path


# ── Seed queries ──────────────────────────────────────────────────────────────
SEED_QUERIES = [
    # Core NLP tasks
    {"query_id": "q001", "text": "transformer architecture for sequence-to-sequence tasks",       "category": "architecture"},
    {"query_id": "q002", "text": "large language model pre-training on multilingual corpora",     "category": "pretraining"},
    {"query_id": "q003", "text": "instruction tuning and RLHF alignment for LLMs",               "category": "alignment"},
    {"query_id": "q004", "text": "retrieval augmented generation for open-domain question answering", "category": "RAG"},
    {"query_id": "q005", "text": "named entity recognition in low-resource languages",            "category": "NER"},
    # Reasoning & evaluation
    {"query_id": "q006", "text": "chain-of-thought prompting for mathematical reasoning",         "category": "reasoning"},
    {"query_id": "q007", "text": "benchmark evaluation of LLMs on commonsense reasoning",        "category": "evaluation"},
    {"query_id": "q008", "text": "hallucination detection and factual consistency in generation", "category": "hallucination"},
    # Efficiency
    {"query_id": "q009", "text": "parameter efficient fine-tuning with LoRA and adapter layers", "category": "efficiency"},
    {"query_id": "q010", "text": "knowledge distillation for compressing large language models",  "category": "compression"},
    # Multimodal & cross-lingual
    {"query_id": "q011", "text": "vision language models for image captioning and VQA",          "category": "multimodal"},
    {"query_id": "q012", "text": "cross-lingual transfer learning for machine translation",      "category": "translation"},
    # Agents & tools
    {"query_id": "q013", "text": "LLM-based agents with tool use and function calling",          "category": "agents"},
    {"query_id": "q014", "text": "code generation and program synthesis with language models",   "category": "code"},
    # Safety & bias
    {"query_id": "q015", "text": "bias and fairness in natural language processing models",      "category": "fairness"},
    {"query_id": "q016", "text": "jailbreak attacks and safety evaluation of LLMs",              "category": "safety"},
    # Specific architectures
    {"query_id": "q017", "text": "mixture of experts scaling in transformer language models",    "category": "architecture"},
    {"query_id": "q018", "text": "attention mechanism improvements and sparse attention",        "category": "architecture"},
    # Applications
    {"query_id": "q019", "text": "clinical information extraction and medical NLP",              "category": "application"},
    {"query_id": "q020", "text": "sentiment analysis and opinion mining from social media",      "category": "application"},
]


QUERY_KEYWORDS: dict[str, list[str]] = {
    "q001": ["transformer", "sequence.to.sequence", "seq2seq", "encoder.decoder"],
    "q002": ["pre.train", "multilingual", "language model", "corpus"],
    "q003": ["instruction", "rlhf", "alignment", "reinforcement learning from human"],
    "q004": ["retrieval.augmented", "rag", "open.domain", "question answering"],
    "q005": ["named entity", "ner", "low.resource", "information extraction"],
    "q006": ["chain.of.thought", "cot", "mathematical reasoning", "reasoning"],
    "q007": ["benchmark", "commonsense", "evaluation", "reasoning"],
    "q008": ["hallucination", "factual", "consistency", "faithfulness"],
    "q009": ["lora", "adapter", "parameter.efficient", "peft", "fine.tun"],
    "q010": ["knowledge distillation", "distill", "compression", "pruning"],
    "q011": ["vision.language", "vqa", "image caption", "multimodal"],
    "q012": ["cross.lingual", "machine translation", "transfer learning", "multilingual"],
    "q013": ["agent", "tool use", "function call", "autonomous"],
    "q014": ["code generation", "program synthesis", "codex", "coding"],
    "q015": ["bias", "fairness", "gender", "stereotyp"],
    "q016": ["jailbreak", "safety", "adversarial", "red.team"],
    "q017": ["mixture of experts", "moe", "sparse model", "scaling"],
    "q018": ["attention", "sparse attention", "linear attention", "efficient transformer"],
    "q019": ["clinical", "medical", "biomedical", "health", "ehr"],
    "q020": ["sentiment", "opinion", "social media", "twitter", "aspect"],
}


def _silver_label(query_id: str, metadata_list: list[dict]) -> list[str]:
    """Return paper_ids whose titles match any keyword for this query."""
    patterns = QUERY_KEYWORDS.get(query_id, [])
    if not patterns:
        return []
    combined = re.compile("|".join(patterns), re.IGNORECASE)
    relevant = []
    for m in metadata_list:
        text_to_search = (m.get("title", "") + " " + m.get("authors", "")).lower()
        if combined.search(text_to_search):
            relevant.append(m["paper_id"])
    return relevant


def create_query_set(output_dir: Path,
                     metadata_list: list[dict] | None = None) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata_list is None:
        meta_path = output_dir / "metadata.json"
        if meta_path.exists():
            metadata_list = json.loads(meta_path.read_text())
        else:
            metadata_list = []

    queries = []
    for q in SEED_QUERIES:
        relevant = _silver_label(q["query_id"], metadata_list)
        queries.append({
            **q,
            "relevant_paper_ids": relevant,           
            "num_relevant":       len(relevant),
        })

    # query_set.json (pipeline-internal) 
    (output_dir / "query_set.json").write_text(
        json.dumps(queries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # labeled_query_set.json (grading deliverable) 
    (output_dir / "labeled_query_set.json").write_text(
        json.dumps(queries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    labeled_count = sum(1 for q in queries if q["num_relevant"] > 0)
    print(f"[queries] {len(queries)} queries saved "
          f"({labeled_count} with silver relevant_paper_ids) "
          f"→ query_set.json + labeled_query_set.json")
    return queries


if __name__ == "__main__":
    from src.config import OUTPUTS_DIR
    tests = sorted(OUTPUTS_DIR.glob("test*"))
    out   = tests[-1]
    create_query_set(out)