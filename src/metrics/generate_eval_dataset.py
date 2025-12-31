# generate_eval_dataset.py - Fixed for Streamlit integration
import sys
from pathlib import Path

# Add project root to path to allow imports when run as script
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import warnings
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore
from src.config import LLM_MODEL_NAME

warnings.filterwarnings("ignore", category=FutureWarning)

def generate_dataset():
    """Generate evaluation dataset for RAG pipeline testing."""
    vs = VectorStore()
    rag = RAGPipeline(vs, llm_model=LLM_MODEL_NAME)
    
    queries = [
        "Vad är kundnöjdhetsmålet för Skånetrafiken år 2025?",
        "Vad är tidsperioden för programmet?",
        "Vad är målet för kollektivtrafikens marknadsandel i Skåne år 2030?",
        "Vilken andel skåningar ska ha minst 10 resmöjligheter till tillväxtmotorer inom 60 minuter?",
        "Vilken andel skåningar ska ha minst 10 resmöjligheter till regionala kärnor inom 45 minuter?",
        "Hur många reser dagligen med kollektivtrafiken i Skåne?"
    #    "What is the time period of the program?",
    #    "What is the target market share for public transport in Skåne by 2030?",
    #    "What percentage of Skåne residents should have at least 10 travel options to growth engines within 60 minutes?",
    #    "What percentage of Skåne residents should have at least 10 travel options to regional cores within 45 minutes?",
    #    "How many people travel daily with public transport in Skåne?",
    #    "What is the customer satisfaction target for Skånetrafiken by 2025?"
    ]

    temperatures = [0.0, 0.1, 0.5] #, 0.1, 0.5
    repeats = 10
    results = []
    
    for query in queries:
        for temp in temperatures:
            answers = []
            for i in range(repeats):
                print(f"Running query: '{query}' (temp={temp}, repeat={i+1}/{repeats})")
                result = rag.answer(query, temperature=temp)
                answers.append({"answer": result["answer"], "sources": result["sources"]})
            results.append({"query": query, "temp": temp, "answers": answers})
    
    # Save raw dataset
    file_name = "eval_dataset_" + LLM_MODEL_NAME + ".json"
    with open(file_name, "w") as f:
        json.dump(results, f, indent=2)
    
    print(file_name + " generated successfully! ✅")
    print(f"Generated {len(results)} query-temp combinations with {repeats} repeats each.")
    return len(results)

if __name__ == "__main__":
    generate_dataset()

