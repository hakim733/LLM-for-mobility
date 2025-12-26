"""
Global configuration for the Skåne Mobility RAG project.

This file centralizes all paths, model names, and
chunking parameters used across the pipeline.
"""

from pathlib import Path

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_documents"

MODELS_DIR = PROJECT_ROOT / "models"
CHROMA_DB_DIR = MODELS_DIR / "chroma_db"

# -----------------------------
# Document processing
# -----------------------------
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# -----------------------------
# Embedding model
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# LLM (Ollama)
# -----------------------------
LLM_MODEL_NAME = "llama3.1"

# -----------------------------
# Retrieval
# -----------------------------
TOP_K = 4




