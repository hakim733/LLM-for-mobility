from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore
import json, statistics
from typing import List, Dict

# Load your indexed vector store and pipeline
vs = VectorStore()
rag = RAGPipeline(vs, llm_model="llama3.1")

queries = ["What is the time period covered?", "How many people use public transport?"]  # Paraphrased from docs
temperatures = [0.0, 0.1, 0.5]
repeats = 10
results = []

for query in queries:
    for temp in temperatures:
        answers = []
        for _ in range(repeats):
            # Modify rag.answer to accept temp param (add to ollama.generate options)
            result = rag.answer(query, temperature=temp)
            answers.append({"answer": result["answer"], "sources": result["sources"]})
        results.append({"query": query, "temp": temp, "answers": answers})

# Save raw dataset
with open("eval_dataset.json", "w") as f:
    json.dump(results, f, indent=2)
