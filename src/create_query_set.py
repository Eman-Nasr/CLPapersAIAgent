import json
from pathlib import Path

# Curated queries representative of cs.CL 2024–2026 topics
SEED_QUERIES = [
    # Core NLP tasks
    {"query_id": "q001", "text": "transformer architecture for sequence-to-sequence tasks", "category": "architecture"},
    {"query_id": "q002", "text": "large language model pre-training on multilingual corpora", "category": "pretraining"},
    {"query_id": "q003", "text": "instruction tuning and RLHF alignment for LLMs", "category": "alignment"},
    {"query_id": "q004", "text": "retrieval augmented generation for open-domain question answering", "category": "RAG"},
    {"query_id": "q005", "text": "named entity recognition in low-resource languages", "category": "NER"},
    # Reasoning & evaluation
    {"query_id": "q006", "text": "chain-of-thought prompting for mathematical reasoning", "category": "reasoning"},
    {"query_id": "q007", "text": "benchmark evaluation of LLMs on commonsense reasoning", "category": "evaluation"},
    {"query_id": "q008", "text": "hallucination detection and factual consistency in generation", "category": "hallucination"},
    # Efficiency
    {"query_id": "q009", "text": "parameter efficient fine-tuning with LoRA and adapter layers", "category": "efficiency"},
    {"query_id": "q010", "text": "knowledge distillation for compressing large language models", "category": "compression"},
    # Multimodal & cross-lingual
    {"query_id": "q011", "text": "vision language models for image captioning and VQA", "category": "multimodal"},
    {"query_id": "q012", "text": "cross-lingual transfer learning for machine translation", "category": "translation"},
    # Agents & tools
    {"query_id": "q013", "text": "LLM-based agents with tool use and function calling", "category": "agents"},
    {"query_id": "q014", "text": "code generation and program synthesis with language models", "category": "code"},
    # Safety & bias
    {"query_id": "q015", "text": "bias and fairness in natural language processing models", "category": "fairness"},
    {"query_id": "q016", "text": "jailbreak attacks and safety evaluation of LLMs", "category": "safety"},
    # Specific architectures
    {"query_id": "q017", "text": "mixture of experts scaling in transformer language models", "category": "architecture"},
    {"query_id": "q018", "text": "attention mechanism improvements and sparse attention", "category": "architecture"},
    # Applications
    {"query_id": "q019", "text": "clinical information extraction and medical NLP", "category": "application"},
    {"query_id": "q020", "text": "sentiment analysis and opinion mining from social media", "category": "application"},
]

def create_query_set(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "query_set.json"
    out_path.write_text(json.dumps(SEED_QUERIES, indent=2))
    print(f"[queries] {len(SEED_QUERIES)} queries saved → {out_path}")
    return SEED_QUERIES


if __name__ == "__main__":
    from src.config import OUTPUTS_DIR
    tests = sorted(OUTPUTS_DIR.glob("test*"))
    out   = tests[-1]
    create_query_set(out)