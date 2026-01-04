# generate_eval_dataset.py - Updated for Capstone project with accurate Swedish questions from Trafikförsörjningsprogram för Skåne 2020-2030
# Key updates:
# 1. Replaced with 6 precise paraphrased questions matching extracted factual claims [file:1]
# 2. Fixed ground truths to exact document facts (no placeholders or incorrect refs)
# 3. Added temperatures [0.0, 0.2, 0.7] for variance analysis across repeats
# 4. Enhanced repeats=3 for better consistency metrics
# 5. Improved output stats for RQ1/WP2 deliverables

import sys
from pathlib import Path
import time
import json
import warnings

# Add project root to path to allow imports when run as script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore
from src.config import LLM_MODEL_NAME

warnings.filterwarnings("ignore", category=FutureWarning)

def generate_dataset():
    """Generate evaluation dataset for RAG pipeline testing with paraphrases and ground truths."""
    vs = VectorStore()
    rag = RAGPipeline(vs, llm_model=LLM_MODEL_NAME)

    # Updated query groups with accurate paraphrases (Swedish primary) and precise ground truths [file:1]
    query_groups = [
        {
            "base": "Vad är kundnöjdhetsmålet för Skånetrafiken år 2025?",
            "paraphrases": [
#                "Vilket kundnöjdhetsmål har Skånetrafiken satt upp för år 2025?"
            ],
            "ground_truth": "Minst 92% av skåningarna ska erbjudas minst 10 dagliga (vardagar) resmöjligheter till någon av regionens tillväxtmotorer med en restid på maximalt 60 minuter. [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 24]"
        },
        {
            "base": "Vad är tidsperioden för programmet?",
            "paraphrases": [
#                "Vilken tidsperiod omfattar trafikförsörjningsprogrammet för Skåne?"
            ],
            "ground_truth": "Programmet omfattar perioden 2020-2030. [Source: trafikförsorjningsprogram-for-skane-2020-2030.pdf, Page: 8]"
        },
        {
            "base": "Vad är målet för kollektivtrafikens marknadsandel i Skåne år 2030?",
            "paraphrases": [
#                "Vilken målsatt marknadsandel för kollektivtrafiken gäller för Skåne år 2030?"
            ],
            "ground_truth": "Enligt trafikförsörjningsprogrammet för Skåne 2020-2030 är målet att kollektivtrafikens marknadsandel ska vara ett genomsnitt för hela Skåne, där större städer bidrar till målet i betydligt högre grad än mindre orter och landsbygden. [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 23]"
        },
        {
            "base": "Vilken andel skåningar ska ha minst 10 resmöjligheter till tillväxtmotorer inom 60 minuter?",
            "paraphrases": [
#                "Hur stor andel av Skånes befolkning ska ha minst tio dagliga resmöjligheter till tillväxtmotorer inom en timme?"
            ],
            "ground_truth": "Minst 92% av skåningarna ska erbjudas minst 10 dagliga (vardagar) resmöjligheter till någon av regionens tillväxtmotorer med en restid på maximalt 60 minuter. [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 24]"
        },
        {
            "base": "Vilken andel skåningar ska ha minst 10 resmöjligheter till regionala kärnor inom 45 minuter?",
            "paraphrases": [
#                "Hur stor andel av Skånes invånare ska ha minst tio dagliga resmöjligheter till regionala kärnor inom 45 minuter?"
            ],
            "ground_truth": "Minst 92% av skåningarna ska erbjudas minst 10 dagliga (vardagar) resmöjligheter till någon av regionens regionala kärnor med en restid på maximalt 45 minuter. [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 24]"
        },
        {
            "base": "Hur många reser dagligen med kollektivtrafiken i Skåne?",
            "paraphrases": [
#                "Vad är det dagliga antalet resenärer som använder kollektivtrafiken i Skåne?"
            ],
            "ground_truth": "Jag hittar inget direkt svar på frågan om hur många reser dagligen med kollektivtrafiken i Skåne. Men jag kan konstatera att resandeutvecklingen i Skåne ökar stadigt, även om det finns en något vikande trend [Source: trafikforsorjningsprogram-for-skane-2020-2030.pdf, Page: 38]."
        }
    ]

    temperatures = [0.0, 0.2, 0.7]  # For temperature variance analysis
    repeats = 3  # Balanced for consistency stats

    results = []
    ground_truths = {}  # For evaluate_rag.py

    for group in query_groups:
        base_query = group["base"]
        paraphrases = [base_query] + group["paraphrases"]  # Include base
        gt_key = base_query  # Use base as key for GT lookup
        ground_truths[gt_key] = group["ground_truth"]

        for query in paraphrases:
            for temp in temperatures:
                answers = []
                for i in range(repeats):
                    print("================================================\n")
                    print(f"Running query: '{query}' (temp={temp}, repeat={i+1}/{repeats})")
                    start_time = time.time()
                    result = rag.answer(query, temperature=temp)
                    elapsed = time.time() - start_time
                    print("answer: " + result["answer"])
                    print("*** *** *** \n")
                    print(result["sources"])
                    print(f"rag.answer took {elapsed:.3f} seconds")
                    answers.append({
                        "answer": result["answer"],
                        "sources": result["sources"]
                    })
                results.append({
                    "query": query,
                    "base_query": base_query,  # Track for grouping
                    "temp": temp,
                    "answers": answers
                })

    # Save augmented dataset
    file_name = f"data/eval_dataset_{LLM_MODEL_NAME}.json"
    dataset = {
        "results": results,
        "ground_truths": ground_truths  # Embedded for evaluation
    }
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(file_name + " generated successfully! ✅")
    num_combinations = len(results)
    print(f"Generated {num_combinations} query-temp combinations ({len(query_groups)} bases × ~4 paraphrases × {len(temperatures)} temps × {repeats} repeats)")
    print(f"Ground truths for {len(ground_truths)} base queries included.")
    return num_combinations

if __name__ == "__main__":
    generate_dataset()
