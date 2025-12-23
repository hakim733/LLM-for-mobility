"""
RAG pipeline: retrieve from ChromaDB and generate an answer using Ollama.

Design goals:
- Closed-corpus: answer ONLY from retrieved context
- Citation-first: every claim must cite [Source: ..., Page: ...]
- Safe fallback: if context is insufficient -> say so
- Deterministic handling of explicit factual metadata (e.g. year ranges in titles)
- Deterministic handling of system-level questions (e.g. indexed documents)
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import re

import ollama


class RAGPipeline:
    def __init__(self, vector_store, llm_model: str = "llama3", top_k: int = 4):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.top_k = top_k

    # ==================================================
    # Context formatting
    # ==================================================
    def _format_context(
        self, results: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        sources: List[Dict[str, Any]] = []
        blocks: List[str] = []

        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            source = meta.get("source", "unknown")
            page = meta.get("page", "n/a")

            sources.append({"source": source, "page": page})

            blocks.append(
                f"[Context {i}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Text:\n{doc}\n"
            )

        return "\n\n".join(blocks), sources

    # ==================================================
    # Prompt
    # ==================================================
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""
You are an academic assistant specialized in mobility and transport in Skåne.
You MUST answer using ONLY the provided context from official documents.

You may restate explicit factual elements such as document titles,
headings, or date ranges if they directly answer the question.
Do NOT infer intentions, motivations, or add external knowledge.

If no explicit factual information is present, say exactly:
"I cannot answer based on the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

RESPONSE RULES:
- Answer in Swedish unless the user asks otherwise.
- Be concise and factual.
- Every factual sentence must end with a citation formatted exactly like:
  [Source: <filename>, Page: <page>]
- Do NOT cite sources that are not in the context.
- Do NOT guess or add external knowledge.
""".strip()

    # ==================================================
    # Deterministic helpers
    # ==================================================
    @staticmethod
    def _extract_year_range(text: str) -> str | None:
        match = re.search(r"(19|20)\d{2}\s*[–-]\s*(19|20)\d{2}", text)
        return match.group(0) if match else None

    @staticmethod
    def _is_document_list_question(question: str) -> bool:
        q = question.lower()
        return any(
            phrase in q
            for phrase in [
                "vilka dokument",
                "vilka filer",
                "indexerats",
                "använder systemet",
                "documents are indexed",
            ]
        )

    def _get_indexed_documents(self) -> List[str]:
        collection = self.vector_store.collection
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        return sorted(
            {meta["source"] for meta in metadatas if "source" in meta}
        )

    # ==================================================
    # Main entry point
    # ==================================================
    def answer(self, question: str) -> Dict[str, Any]:
        # ----------------------------------------------
        # System-level questions (NO LLM)
        # ----------------------------------------------
        if self._is_document_list_question(question):
            docs = self._get_indexed_documents()
            if not docs:
                return {
                    "answer": "Inga dokument har indexerats i systemet.",
                    "sources": [],
                }

            answer = "Följande dokument har indexerats i systemet:\n"
            for d in docs:
                answer += f"- {d}\n"

            return {
                "answer": answer.strip(),
                "sources": [],
            }

        # ----------------------------------------------
        # Retrieval
        # ----------------------------------------------
        results = self.vector_store.query(question, n_results=self.top_k)
        context, sources = self._format_context(results)

        # ----------------------------------------------
        # Deterministic factual normalization
        # ----------------------------------------------
        q_lower = question.lower()

        if "tidsperiod" in q_lower or "period" in q_lower:
            for doc, meta in zip(
                results["documents"][0], results["metadatas"][0]
            ):
                year_range = self._extract_year_range(doc)
                if year_range:
                    return {
                        "answer": (
                            f"Trafikförsörjningsprogrammet omfattar perioden "
                            f"{year_range}. "
                            f"[Source: {meta['source']}, Page: {meta['page']}]"
                        ),
                        "sources": [
                            {
                                "source": meta["source"],
                                "page": meta["page"],
                            }
                        ],
                    }

        # ----------------------------------------------
        # LLM-based answer (streaming, safe)
        # ----------------------------------------------
        prompt = self._build_prompt(question, context)
        answer_text = ""

        try:
            for chunk in ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_ctx": 4096,
                },
                stream=True,
            ):
                if "response" in chunk:
                    answer_text += chunk["response"]

        except Exception:
            return {
                "answer": "I cannot answer based on the provided documents.",
                "sources": sources,
            }

        final_answer = answer_text.strip()

        if not final_answer:
            final_answer = "I cannot answer based on the provided documents."

        return {
            "answer": final_answer,
            "sources": sources,
        }
