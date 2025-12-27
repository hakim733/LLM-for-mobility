import json
import re
import statistics
from typing import List, Dict, Any
from rapidfuzz import fuzz, process
import numpy as np
from collections import defaultdict
from src.rag_pipeline import RAGPipeline  # Add temperature param to answer()
from src.vector_store import VectorStore

def load_dataset(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        return json.load(f)

def compute_correctness(answers: List[str], ground_truths: List[str], threshold: float = 80.0) -> float:
    """Binary correctness via fuzzy string matching to ground truth claims."""
    correct = 0
    for answer in answers:
        best_match = max(ground_truths, key=lambda gt: fuzz.ratio(answer.lower(), gt.lower()))
        if fuzz.ratio(answer.lower(), best_match.lower()) >= threshold:
            correct += 1
    return correct / len(answers) if answers else 0.0

def compute_citation_accuracy(answer: str, sources: List[Dict[str, str]]) -> float:
    """Extract citations and verify they match retrieved sources."""
    citations = re.findall(r'Source (\w+\.pdf), Page (\d+)', answer)
    valid_sources = {s['source'].split('/')[-1]: s['page'] for s in sources}
    
    if not citations:
        return 0.0  # No citations = 0 accuracy
    
    valid_cites = 0
    for source_file, page_num in citations:
        if source_file in valid_sources and valid_sources[source_file] == page_num:
            valid_cites += 1
    return valid_cites / len(citations)

def compute_consistency(answers: List[str]) -> float:
    """Jaccard similarity across all answer pairs."""
    if len(answers) < 2:
        return 1.0
    similarities = []
    for i in range(len(answers)):
        for j in range(i+1, len(answers)):
            sim = fuzz.token_set_ratio(answers[i], answers[j]) / 100.0
            similarities.append(sim)
    return statistics.mean(similarities)

# Main evaluation
vs = VectorStore()
rag = RAGPipeline(vs, llm_model="llama3.1")

dataset = load_dataset("eval_dataset.json")
ground_truths = {
    "What is the time period covered?": ["2022-2026", "Trafikförsörjningsprogrammet 2022-2026"],
    "How many people use public transport?": ["65%", "65 procent reser med kollektivtrafik"]
}  # Extract from your PDFs[file:5]

metrics = defaultdict(list)
for result in dataset:
    query, temp = result["query"], result["temp"]
    answers = [r["answer"] for r in result["answers"]]
    all_sources = [src for r in result["answers"] for src in r["sources"]]
    
    corr = compute_correctness(answers, ground_truths.get(query, []))
    cit_acc = np.mean([compute_citation_accuracy(r["answer"], all_sources) for r in result["answers"]])
    cons = compute_consistency(answers)
    
    metrics[f"temp_{temp}"].append({"correctness": corr, "citation": cit_acc, "consistency": cons})

# Summary table
print("| Temperature | Correctness | Citation Acc | Consistency |")
print("|-------------|-------------|--------------|-------------|")
for temp_str, scores in metrics.items():
    print(f"| {temp_str} | {np.mean([s['correctness'] for s in scores]):.1%} | "
          f"{np.mean([s['citation'] for s in scores]):.1%} | "
          f"{np.mean([s['consistency'] for s in scores]):.3f} |")

# Temperature variance
consistencies = [np.mean([s['consistency'] for s in scores]) for scores in metrics.values()]
variance = np.std(consistencies)
print(f"\nTemperature Variance (Consistency Std Dev): {variance:.3f}")
