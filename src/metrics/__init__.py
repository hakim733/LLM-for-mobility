"""Metrics package for RAG evaluation and dataset generation."""

from .evaluate_rag import run_evaluation
from .generate_eval_dataset import generate_dataset

__all__ = ["run_evaluation", "generate_dataset"]

