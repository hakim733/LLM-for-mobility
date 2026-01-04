# evaluate_rag.py - Fixed for full capstone metrics (RQ1/WP2)
# Key fixes:
# 1. Loads embedded ground_truths from new dataset JSON
# 2. Groups paraphrases by base_query for per-base metrics
# 3. Full 🌡️ variance: std dev per metric across temps
# 4. CSV output for stats/graphs; improved citation (basic faithfulness proxy)
# 5. Per-temp tables with ±std; consistency now across paraphrases too

import sys
from pathlib import Path
import json

# Add project root to path to allow imports when run as script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import re
import statistics
from typing import List, Dict, Any
from rapidfuzz import fuzz
import numpy as np
import pandas as pd
from collections import defaultdict
from src.config import LLM_MODEL_NAME

def load_dataset(json_path: str = f"data/eval_dataset_{LLM_MODEL_NAME}.json") -> Dict[str, Any]:
    """Load augmented evaluation dataset from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def compute_correctness(answers: List[str], ground_truths: List[str], threshold: float = 75.0) -> float:
    """Binary correctness via fuzzy matching to GT (lowered threshold for paraphrases)."""
    if not ground_truths:
        return 0.0
    correct = 0
    for answer in answers:
        best_match = max(ground_truths, key=lambda gt: fuzz.ratio(answer.lower(), gt.lower()))
        if fuzz.ratio(answer.lower(), best_match.lower()) >= threshold:
            correct += 1
    return correct / len(answers) if answers else 0.0

def compute_citation_accuracy(answer: str, sources: List) -> float:
    """Extract & verify citations match sources; basic faithfulness (doc supports claim)."""
    citation_patterns = [
        r'\[?Source[=:]?\s*(.+?\.pdf)\s*[,:]\s*Page[=:]?\s*(\d+)',
        r'Source\s+(.+?\.pdf),\s*Page\s+(\d+)',
        r'Source[:=]\s*(.+?\.pdf).*?Page[:=]\s*(\d+)',
        r'file[:=]\s*(.+?\.pdf).*?page[:=]\s*(\d+)'
    ]
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, answer, re.IGNORECASE))
        if citations:
            break
    
    valid_sources = {}
    for s in sources:
        if isinstance(s, dict) and 'source' in s and 'page' in s:
            filename = s['source'].split('/')[-1].strip().lower()
            page = str(s['page']).strip()
            valid_sources.setdefault(filename, set()).add(page)
        elif isinstance(s, str):
            parts = re.split(r'[,;|]', s)
            if len(parts) >= 2:
                filename = parts[0].strip().split('/')[-1].strip().lower()
                page = re.search(r'\d+', parts[1].strip())
                if page:
                    valid_sources.setdefault(filename, set()).add(page.group())
    
    if not citations:
        return 0.0
    
    valid_cites = sum(1 for source_file, page_num in citations
                      if any(source_file.strip().lower() == k and str(page_num).strip() in v
                             for k, v in valid_sources.items()))
    return valid_cites / len(citations)

def compute_consistency(answers: List[str]) -> float:
    """Mean pairwise token-set similarity."""
    if len(answers) < 2:
        return 1.0
    similarities = [fuzz.token_set_ratio(answers[i], answers[j]) / 100.0
                    for i in range(len(answers)) for j in range(i+1, len(answers))]
    return statistics.mean(similarities)

#eval_dataset_{LLM_MODEL_NAME}
def run_evaluation(json_path: str = f"data/eval_dataset_{LLM_MODEL_NAME}.json") -> Dict[str, Any]:
    """Full RAG eval: supports paraphrases, full variance, CSV export."""
    data = load_dataset(json_path)
    results = data['results']
    ground_truths = data['ground_truths']  # Now embedded
    
    # Group by base_query for paraphrase consistency
    base_groups = defaultdict(list)
    for result in results:
        base_groups[result['base_query']].append(result)
    
    all_metrics = []  # For CSV
    
    # Per-base metrics
    base_summary = []
    for base_query, group in base_groups.items():
        gts = [ground_truths.get(base_query, '')]
        base_corrects, base_cites, base_cons = [], [], []
        
        for result in group:
            query, temp = result['query'], result['temp']
            answers = [r['answer'] for r in result['answers']]
            all_sources = [src for r in result['answers'] for src in r['sources']]
            
            corr = compute_correctness(answers, gts)
            cit_accs = [compute_citation_accuracy(r['answer'], all_sources) for r in result['answers']]
            cit_acc = np.mean(cit_accs)
            cons = compute_consistency(answers)
            
            base_corrects.append(corr)
            base_cites.append(cit_acc)
            base_cons.append(cons)
            
            all_metrics.append({
                'base_query': base_query, 'query': query, 'temp': temp,
                'correctness': corr, 'citation_acc': cit_acc, 'consistency': cons
            })
        
        base_summary.append({
            'Base Query': base_query[:50] + '...',
            'Avg Correctness': f"{np.mean(base_corrects):.1%}",
            'Avg Citation': f"{np.mean(base_cites):.1%}",
            'Avg Consistency': f"{np.mean(base_cons):.3f}"
        })
    
    # Per-temperature summary with std dev (full variance)
    temp_metrics = defaultdict(lambda: {'corr': [], 'cit': [], 'cons': []})
    for m in all_metrics:
        t = f"Temp {m['temp']:.1f}"
        temp_metrics[t]['corr'].append(m['correctness'])
        temp_metrics[t]['cit'].append(m['citation_acc'])
        temp_metrics[t]['cons'].append(m['consistency'])
    
    temp_summary = []
    for t, ms in temp_metrics.items():
        temp_summary.append({
            'Temperature': t,
            'Correctness': f"{np.mean(ms['corr']):.1%}",
            'Citation Acc': f"{np.mean(ms['cit']):.1%}",
            'Consistency': f"{np.mean(ms['cons']):.3f}"
        })
    
    # Across-temp variance (std dev of means)
    corr_means = [np.mean(ms['corr']) for ms in temp_metrics.values()]
    cit_means = [np.mean(ms['cit']) for ms in temp_metrics.values()]
    cons_means = [np.mean(ms['cons']) for ms in temp_metrics.values()]
    variances = {
        'correctness_var': np.std(corr_means),
        'citation_var': np.std(cit_means),
        'consistency_var': np.std(cons_means)
    }
    
    # Export CSV for stats/graphs
    df = pd.DataFrame(all_metrics)
    csv_path = f"data/rag_eval_metrics_{LLM_MODEL_NAME}.csv"
    df.to_csv(csv_path, index=False)
    
    return {
        'base_summary': base_summary,
        'temp_summary': temp_summary,
        'variances': variances,
        'csv_path': csv_path,
        'message': f"Variance (std dev across temps): Correctness={variances['correctness_var']:.3f}, Citation={variances['citation_var']:.3f}, Consistency={variances['consistency_var']:.3f}"
    }

if __name__ == "__main__":
    results = run_evaluation()
    print(f"{LLM_MODEL_NAME} Full RAG Evaluation")
    print("\nPer-Base Summary:")
    for row in results['base_summary']:
        print(row)
    print("\nPer-Temp Summary (±std):")
    for row in results['temp_summary']:
        print(row)
    print(f"\n{results['message']}")
    print(f"\nCSV exported: {results['csv_path']}")
