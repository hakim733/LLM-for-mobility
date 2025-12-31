import sys
from pathlib import Path

# Add project root to path to allow imports when run as script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import re
import statistics
from typing import List, Dict, Any
from rapidfuzz import fuzz
import numpy as np
from collections import defaultdict
from src.config import LLM_MODEL_NAME


def load_dataset(json_path: str) -> List[Dict]:
    """Load evaluation dataset from JSON file."""
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
    """Extract citations and verify they match retrieved sources. Handles multiple formats."""
    
    # Flexible regex to catch common citation formats
    citation_patterns = [
        r'\[?Source[=:]\s*(.+?\.pdf)\s*[,:]\s*Page[=:]\s*(\d+)',
        r'Source\s+(.+?\.pdf),\s*Page\s+(\d+)',
        r'Source[:=]\s*(.+?\.pdf).*?Page[:=]\s*(\d+)',
        r'file[:=]\s*(.+?\.pdf).*?page[:=]\s*(\d+)'
    ]
    
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, answer, re.IGNORECASE))
        if citations:  # Found some, no need to try others
            break
    
    # Build valid sources dict with normalized keys (supports multiple pages per file)
    valid_sources = {}
    for s in sources:
        if isinstance(s, dict) and 'source' in s and 'page' in s:
            filename = s['source'].split('/')[-1].strip().lower()
            page = str(s['page']).strip()
            if filename not in valid_sources:
                valid_sources[filename] = set()
            valid_sources[filename].add(page)
        elif isinstance(s, str):
            # Handle string format: "source file.pdf, page 45"
            parts = s.split(',')
            if len(parts) >= 2:
                filename = parts[0].strip().split('/')[-1].strip().lower()
                page = parts[1].strip().split()[-1].strip()
                if filename not in valid_sources:
                    valid_sources[filename] = set()
                valid_sources[filename].add(page)
    
    if not citations:
        return 0.0  # No citations = 0 accuracy
    
    valid_cites = 0
    for source_file, page_num in citations:
        source_file_norm = source_file.strip().lower()
        page_num_norm = str(page_num).strip()
        
        for valid_file, valid_pages in valid_sources.items():
            if source_file_norm == valid_file and page_num_norm in valid_pages:
                valid_cites += 1
                break
    
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


def run_evaluation(json_path: str = "data/eval_dataset_" + LLM_MODEL_NAME + ".json") -> Dict[str, Any]:
    """Run full RAG evaluation and return formatted results for Streamlit."""
    dataset = load_dataset(json_path)
    
    ground_truths = {
        "Vad är kundnöjdhetsmålet för Skånetrafiken år 2025?": [
            "Jag hittar inget direkt svar på kundnöjdhetsmålet för Skånetrafiken år 2025 i de tillgängliga texterna. Men jag kan konstatera att det nämns att \"40% marknadsandel för kollektivtrafiken\" är ett politiskt antaget mål [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 45]."
        ],
        "Vad är tidsperioden för programmet?": [
            "Programmet omfattar perioden 2020-2030. [Source: trafikförsorjningsprogram-for-skane-2020-2030.pdf, Page: 8]"
        ],
        "Vad är målet för kollektivtrafikens marknadsandel i Skåne år 2030?": [
            "Enligt trafikförsörjningsprogrammet för Skåne 2020-2030 är målet att kollektivtrafikens marknadsandel ska vara ett genomsnitt för hela Skåne, där större städer bidrar till målet i betydligt högre grad än mindre orter och landsbygden. [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 23]"
        ],
        "Vilken andel skåningar ska ha minst 10 resmöjligheter till tillväxtmotorer inom 60 minuter?": [
            "Minst 92% av skåningarna ska erbjudas minst 10 dagliga (vardagar) resmöjligheter till någon av regionens tillväxtmotorer med en restid på maximalt 60 minuter. [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 24]"
        ],
        "Vilken andel skåningar ska ha minst 10 resmöjligheter till regionala kärnor inom 45 minuter?": [
            "Minst 92% av sk\u00e5ningarna ska erbjudas minst 10 dagliga (vardagar) resm\u00f6jligheter till n\u00e5gon av regionens regionala k\u00e4rnor med en restid p\u00e5 maximalt 45 minuter. [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 24]"
        ],
        "Hur många reser dagligen med kollektivtrafiken i Skåne?": [
            "Jag hittar inget direkt svar på frågan om hur många reser dagligen med kollektivtrafiken i Skåne. Men jag kan konstatera att resandeutvecklingen i Skåne ökar stadigt, även om det finns en något vikande trend [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 38]."
        ],
        # Add more queries as needed
    }
   
    metrics = defaultdict(list)
    for result in dataset:
        query, temp = result["query"], result["temp"]
        answers = [r["answer"] for r in result["answers"]]
        all_sources = [src for r in result["answers"] for src in r["sources"]]
        
        corr = compute_correctness(answers, ground_truths.get(query, []))
        cit_acc = np.mean([compute_citation_accuracy(r["answer"], all_sources) for r in result["answers"]])
        cons = compute_consistency(answers)
        
        metrics[f"temp_{temp}"].append({"correctness": corr, "citation": cit_acc, "consistency": cons})
    
    # Format results for Streamlit
    summary_table = []
    for temp_str, scores in metrics.items():
        corr_mean = np.mean([s['correctness'] for s in scores])
        cit_mean = np.mean([s['citation'] for s in scores])
        cons_mean = np.mean([s['consistency'] for s in scores])
        summary_table.append({
            "Temperature": temp_str, 
            "Correctness": f"{corr_mean:.1%}", 
            "Citation Acc": f"{cit_mean:.1%}", 
            "Consistency": f"{cons_mean:.3f}"
        })
    
    consistencies = [np.mean([s['consistency'] for s in scores]) for scores in metrics.values()]
    variance = np.std(consistencies)
    
    return {
        "summary_table": summary_table,
        "variance": variance,
        "message": f"Temperature Variance (Consistency Std Dev): {variance:.3f}"
    }

if __name__ == "__main__":
    results = run_evaluation()
    print(LLM_MODEL_NAME + " Evaluation Results")
    print(results["summary_table"])
    print(results["message"])
    

